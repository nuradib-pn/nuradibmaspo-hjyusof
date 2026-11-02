"""Dataset survey: class counts, image sizes, exact duplicates and T / I / M label coverage.

    python -m wound_tim.survey --data-root data/wounds [--labels-csv labels.csv] [--out survey.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

from .data import HEAD_NAMES, scan_images


def survey(data_root: str, labels_csv: str | None = None) -> dict:
    df = scan_images(data_root)
    sizes, modes, hashes, unreadable = [], Counter(), defaultdict(list), []
    for p, rel in zip(df["path"], df["relpath"]):
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
                modes[im.mode] += 1
        except Exception:  # noqa: BLE001 - report, don't crash
            unreadable.append(rel)
            continue
        hashes[hashlib.md5(Path(p).read_bytes()).hexdigest()].append(rel)

    dups = [v for v in hashes.values() if len(v) > 1]
    cross_class = [v for v in dups if len({Path(r).parts[0] for r in v}) > 1]
    w = pd.Series([s[0] for s in sizes])
    h = pd.Series([s[1] for s in sizes])
    counts = df["wound_type"].value_counts().sort_index()

    out = {
        "n_images": len(df),
        "class_counts": counts.to_dict(),
        "imbalance_ratio_max_over_min": float(counts.max() / counts.min()),
        "extensions": df["path"].map(lambda p: Path(p).suffix.lower()).value_counts().to_dict(),
        "color_modes": dict(modes),
        "width": w.describe().round(1).to_dict() if len(w) else {},
        "height": h.describe().round(1).to_dict() if len(h) else {},
        "unreadable": unreadable,
        "exact_duplicate_groups": len(dups),
        "exact_duplicate_groups_across_classes": cross_class,
    }
    if labels_csv:
        labels = pd.read_csv(labels_csv)
        out["labels_columns"] = list(labels.columns)
        out["label_coverage"] = {h: {"labelled": int(labels[h].notna().sum()), "values": labels[h].value_counts().to_dict()} for h in HEAD_NAMES if h in labels}
        out["has_patient_id"] = "patient_id" in labels.columns
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--labels-csv")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    res = survey(args.data_root, args.labels_csv)
    text = json.dumps(res, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
