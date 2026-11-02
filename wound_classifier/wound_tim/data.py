"""Dataset indexing, splitting and loading for wound type + T / I / M labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IGNORE = -100  # label value meaning "not annotated"; skipped by the loss
HEAD_NAMES = ("tissue", "infection", "moisture")


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


def build_index(cfg: dict) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return (index dataframe with integer label columns, vocabularies)."""
    df = scan_images(cfg["data_root"])

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

    if "patient_id" not in df.columns:
        df["patient_id"] = np.nan
    return df, vocabs


def split_index(df: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int) -> pd.DataFrame:
    """Add a 'split' column. Stratified by wound type; patient-grouped when patient_id is given,
    so no patient's images appear in two splits (avoids optimistic leakage)."""
    df = df.copy()
    y = df["wound_type_y"].to_numpy()
    if df["patient_id"].notna().all():
        groups = df["patient_id"].astype(str).to_numpy()

        def carve(idx, frac):
            n_splits = max(2, round(1 / frac))
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            rest, held = next(sgkf.split(idx, y[idx], groups[idx]))
            return idx[rest], idx[held]

        idx = np.arange(len(df))
        rest, test = carve(idx, test_fraction)
        train, val = carve(rest, val_fraction / (1 - test_fraction))
    else:
        idx = np.arange(len(df))
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


class WoundDataset(Dataset):
    LABEL_COLS = ("wound_type",) + HEAD_NAMES

    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["path"].tolist()
        self.labels = {name: torch.tensor(df[f"{name}_y"].to_numpy(), dtype=torch.long) for name in self.LABEL_COLS}
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), {k: v[i] for k, v in self.labels.items()}
