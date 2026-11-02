"""End-to-end smoke tests on tiny synthetic images (random init, no downloads)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from wound_tim import data, survey, train

CFG = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"


def make_type_folders(root: Path, per_class=10, seed=0):
    """TYPE 1..8 folders with phone-style dated filenames: two images per capture day."""
    rng = np.random.default_rng(seed)
    for c in range(1, 9):
        d = root / (f"TYPE {c}-Slough <25-No Infection-Dry" if c == 1 else f"TYPE {c}")
        d.mkdir(parents=True)
        for i in range(per_class):
            arr = (rng.random((40, 48, 3)) * 255).astype("uint8")
            Image.fromarray(arr).save(d / f"2020{c:02d}{10 + i // 2:02d}_1015{i:02d}.jpg")
    return root


def write_cfg(tmp_path, data_root, **overrides):
    cfg = yaml.safe_load(CFG.read_text())
    cfg.update(data_root=str(data_root), image_size=32, **overrides)
    cfg["model"]["backbone"] = "resnet18"
    cfg["train"].update(epochs=2, batch_size=8, num_workers=0, freeze_backbone_epochs=1)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.mark.parametrize("n", range(1, 9))
def test_type_tim_round_trip(n):
    t, i, m = data.tim_from_type(n)
    assert data.type_from_tim(t, i, m) == n
    # Manual: T score 0/4, I score 0/2, M score 1/2; type = T + I + M.
    assert 4 * t + 2 * i + (1 + m) == n


def test_parsers():
    assert data.type_number("TYPE 1-Slough <25-No Infaction-Dry(Minimal)") == 1
    assert data.type_number("TYPE 8") == 8
    assert data.type_number("misc") is None
    assert data.capture_date("20201118_105218.jpg") == "20201118"
    assert data.capture_date("IMG-20201105-WA0012.jpg") == "20201105"
    assert data.capture_date("photo_6253396233662089265_y.jpg") is None


def test_derived_labels_dedup_and_date_groups(tmp_path):
    root = make_type_folders(tmp_path / "data")
    # A byte-identical copy filed under another type must be dropped (both copies).
    shutil.copy(root / "TYPE 3" / "20200310_101500.jpg", root / "TYPE 7" / "Copy of 20200310_101500.jpg")
    # A same-class duplicate keeps one copy.
    shutil.copy(root / "TYPE 2" / "20200210_101500.jpg", root / "TYPE 2" / "20200210_101500 (1).jpg")
    cfg = yaml.safe_load(write_cfg(tmp_path, root).read_text())
    df, vocabs = data.build_index(cfg)
    assert len(df) == 80 - 1
    assert len(df.attrs["dedup_report"]["cross_class_removed"]) == 2
    row = df[df.relpath.str.startswith("TYPE 6")].iloc[0]  # type 6 = >25%, no infection, wet
    assert (row.tissue_y, row.infection_y, row.moisture_y) == (1, 0, 1)
    split = data.split_index(df, 0.15, 0.15, 42)
    assert (split.groupby("patient_id")["split"].nunique() == 1).all(), "a capture day leaked across splits"


def test_train_end_to_end_with_type_from_tim(tmp_path):
    root = make_type_folders(tmp_path / "data")
    cfg = write_cfg(tmp_path, root)
    res = train.main(["--config", str(cfg), "--no-pretrained", "--out-dir", str(tmp_path / "run")])
    assert {"wound_type", "tissue", "infection", "moisture", "wound_type_from_tim"} <= set(res["test"])
    assert res["skipped_heads"] == []
    preds = pd.read_csv(tmp_path / "run" / "test_predictions.csv")
    assert {"wound_type_pred", "wound_type_from_tim_pred"} <= set(preds.columns)
    json.dumps(res)


def test_csv_labels_without_type_derivation(tmp_path):
    rng = np.random.default_rng(1)
    rows = []
    for c in range(8):
        d = tmp_path / "data" / f"class_{c}"
        d.mkdir(parents=True)
        for i in range(10):
            Image.fromarray((rng.random((40, 48, 3)) * 255).astype("uint8")).save(d / f"img_{i}.jpg")
            rows.append({"relpath": f"class_{c}/img_{i}.jpg", "tissue": ["lt25", "gt25"][i % 2], "patient_id": f"p{c}_{i // 2}"})
    csv = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    cfg = write_cfg(tmp_path, tmp_path / "data", labels_csv=str(csv), derive_tim_from_type=False, group_by=None)
    res = train.main(["--config", str(cfg), "--no-pretrained", "--out-dir", str(tmp_path / "run")])
    assert set(res["skipped_heads"]) == {"infection", "moisture"}
    assert "wound_type_from_tim" not in res["test"]

    s = survey.survey(str(tmp_path / "data"), str(csv))
    assert s["n_images"] == 80 and s["label_coverage"]["tissue"]["labelled"] == 80
