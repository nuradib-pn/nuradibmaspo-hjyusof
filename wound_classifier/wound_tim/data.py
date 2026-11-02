"""Dataset indexing, splitting and loading for wound type + T / I / M labels.

The wound types follow the TIME scoring in the clinical manual: type = T + I + M where
T in {0: necrotic/slough <25%, 4: >25%}, I in {0: no infection, 2: infection},
M in {1: dry/minimal exudate, 2: moderate/wet}. So TYPE 1..8 decode to three binary labels.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IGNORE = -100  # label value meaning "not annotated"; skipped by the loss
HEAD_NAMES = ("tissue", "infection", "moisture")


def type_number(folder_name: str) -> int | None:
    """'TYPE 1-Slough <25-...' -> 1, 'TYPE 8' -> 8, anything else -> None."""
    m = re.match(r"\s*type[\s_-]*([1-8])\b", folder_name, flags=re.I)
    return int(m.group(1)) if m else None


def tim_from_type(n: int) -> tuple[int, int, int]:
    """Binary (tissue>25%, infection, wet exudate) for TIME wound type 1..8."""
    k = n - 1
    return k // 4, (k // 2) % 2, k % 2


def type_from_tim(t: int, i: int, m: int) -> int:
    return 4 * t + 2 * i + m + 1


def capture_date(filename: str) -> str | None:
    """Capture date from phone / messaging-app filenames (20201118_105218.jpg, IMG-20201105-WA0012.jpg)."""
    m = re.search(r"(20\d{2})(\d{2})(\d{2})[_-]", filename)
    if m and 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31:
        return "".join(m.groups())
    return None


def file_md5(path: str | Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def scan_images(data_root: str | Path) -> pd.DataFrame:
    """One row per image found under data_root/<wound_type>/..."""
    root = Path(data_root)
    rows = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(class_dir.rglob("*")):
            if f.suffix.lower() in IMAGE_EXTS:
                rows.append({"path": str(f), "relpath": str(f.relative_to(root)), "wound_type": class_dir.name})
    if not rows:
        raise FileNotFoundError(f"No images found under {root}/<class>/")
    return pd.DataFrame(rows)


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove byte-identical files: keep one copy within a class, drop every copy that sits in two classes."""
    df = df.copy()
    df["md5"] = [file_md5(p) for p in df["path"]]
    n_classes = df.groupby("md5")["wound_type"].transform("nunique")
    cross = df[n_classes > 1]
    df = df[n_classes == 1].drop_duplicates("md5")
    return df.reset_index(drop=True), {"cross_class_removed": cross["relpath"].tolist(), "kept": len(df)}


def build_index(cfg: dict) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return (index dataframe with integer label columns, vocabularies)."""
    df = scan_images(cfg["data_root"])
    report = {}
    if cfg.get("drop_duplicates", True):
        df, report = drop_duplicates(df)

    if cfg.get("labels_csv"):
        labels = pd.read_csv(cfg["labels_csv"])
        key = "relpath" if "relpath" in labels.columns else "filename"
        if key == "filename":
            df["filename"] = df["relpath"].map(lambda p: Path(p).name)
        labels = labels.drop(columns=[c for c in ("wound_type",) if c in labels.columns])
        df = df.merge(labels, on=key, how="left")

    vocabs: dict[str, list[str]] = {"wound_type": cfg.get("wound_types") or sorted(df["wound_type"].unique())}
    for head in HEAD_NAMES:
        vocabs[head] = list(cfg["heads"][head]["classes"])

    for name, classes in vocabs.items():
        lookup = {str(c).strip().lower(): i for i, c in enumerate(classes)}
        col = df[name] if name in df.columns else pd.Series([np.nan] * len(df))
        unknown = sorted({str(v) for v in col.dropna() if str(v).strip().lower() not in lookup})
        if unknown:
            raise ValueError(f"Column '{name}' has values not in its vocabulary {classes}: {unknown}")
        df[f"{name}_y"] = [IGNORE if pd.isna(v) else lookup[str(v).strip().lower()] for v in col]

    if cfg.get("derive_tim_from_type"):
        # Fill T / I / M wherever a CSV did not already provide them.
        nums = df["wound_type"].map(type_number)
        if nums.isna().any():
            raise ValueError(f"derive_tim_from_type needs folders named 'TYPE 1'..'TYPE 8'; got {sorted(df.wound_type[nums.isna()].unique())}")
        tim = np.array([tim_from_type(int(n)) for n in nums])
        for j, head in enumerate(HEAD_NAMES):
            missing = df[f"{head}_y"] == IGNORE
            df.loc[missing, f"{head}_y"] = tim[missing.to_numpy(), j]

    if "patient_id" not in df.columns:
        df["patient_id"] = np.nan
    if df["patient_id"].isna().any() and cfg.get("group_by") == "capture_date":
        # No patient IDs: photos taken on the same day are treated as one group so near-identical
        # shots never straddle train and test. Undated files are their own group.
        dates = df["relpath"].map(lambda p: capture_date(Path(p).name))
        fallback = "file:" + df["relpath"]
        df["patient_id"] = df["patient_id"].fillna(("date:" + dates).where(dates.notna(), fallback))
    df.attrs["dedup_report"] = report
    return df, vocabs


def split_index(df: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int) -> pd.DataFrame:
    """Add a 'split' column. Stratified by wound type; grouped by patient_id when every row has one,
    so no group's images appear in two splits (avoids optimistic leakage)."""
    df = df.copy()
    y = df["wound_type_y"].to_numpy()
    idx = np.arange(len(df))
    if df["patient_id"].notna().all():
        groups = df["patient_id"].astype(str).to_numpy()

        def carve(sub, frac):
            n_splits = max(2, round(1 / frac))
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            rest, held = next(sgkf.split(sub, y[sub], groups[sub]))
            return sub[rest], sub[held]

        rest, test = carve(idx, test_fraction)
        train, val = carve(rest, val_fraction / (1 - test_fraction))
    else:
        rest, test = train_test_split(idx, test_size=test_fraction, stratify=y, random_state=seed)
        train, val = train_test_split(rest, test_size=val_fraction / (1 - test_fraction), stratify=y[rest], random_state=seed)
    df["split"] = ""
    df.loc[df.index[train], "split"] = "train"
    df.loc[df.index[val], "split"] = "val"
    df.loc[df.index[test], "split"] = "test"
    return df


def make_transforms(image_size: int, train: bool):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                # Mild colour jitter only: wound colour carries tissue / exudate signal.
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
                transforms.ToTensor(),
                norm,
            ]
        )
    return transforms.Compose([transforms.Resize(int(image_size * 1.14)), transforms.CenterCrop(image_size), transforms.ToTensor(), norm])


def load_rgb(path: str, max_side: int | None = None) -> Image.Image:
    img = Image.open(path)
    if max_side:
        img.draft("RGB", (max_side, max_side))  # fast JPEG downscale while decoding 12 MP phone photos
    img = ImageOps.exif_transpose(img).convert("RGB")  # phone photos carry EXIF rotation
    if max_side:
        img.thumbnail((max_side, max_side))
    return img


class WoundDataset(Dataset):
    LABEL_COLS = ("wound_type",) + HEAD_NAMES

    def __init__(self, df: pd.DataFrame, transform, max_side: int | None = 512):
        self.paths = df["path"].tolist()
        self.labels = {name: torch.tensor(df[f"{name}_y"].to_numpy(), dtype=torch.long) for name in self.LABEL_COLS}
        self.transform = transform
        self.max_side = max_side

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_rgb(self.paths[i], self.max_side)
        return self.transform(img), {k: v[i] for k, v in self.labels.items()}
