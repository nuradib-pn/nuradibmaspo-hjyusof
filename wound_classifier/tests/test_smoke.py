"""End-to-end smoke test on tiny synthetic images (random init, no downloads)."""

import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from wound_tim import survey, train

CFG = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"


def make_dataset(root: Path, n_classes=8, per_class=10, with_patients=False):
    rng = np.random.default_rng(0)
    rows = []
    for c in range(n_classes):
        d = root / f"type_{c}"
        d.mkdir(parents=True)
        for i in range(per_class):
            arr = (rng.random((40, 48, 3)) * 255).astype("uint8")
            Image.fromarray(arr).save(d / f"img_{i}.jpg")
            # Partial labels: T on every image, I on half, M on none.
            rows.append(
                {
                    "relpath": f"type_{c}/img_{i}.jpg",
                    "tissue": ["granulation", "slough", "necrotic", "epithelial"][i % 4],
                    "infection": ["absent", "present"][i % 2] if i % 2 == 0 or i < 6 else None,
                    "patient_id": f"p{c}_{i // 2}" if with_patients else None,
                }
            )
    return rows


def write_cfg(tmp_path, data_root, labels_csv=None):
    cfg = yaml.safe_load(CFG.read_text())
    cfg.update(data_root=str(data_root), labels_csv=labels_csv, image_size=32)
    cfg["model"]["backbone"] = "resnet18"
    cfg["train"].update(epochs=2, batch_size=8, num_workers=0, freeze_backbone_epochs=1)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_train_without_tim_labels(tmp_path):
    make_dataset(tmp_path / "data")
    cfg = write_cfg(tmp_path, tmp_path / "data")
    res = train.main(["--config", str(cfg), "--no-pretrained", "--out-dir", str(tmp_path / "run")])
    assert set(res["test"]) == {"wound_type"}
    assert res["skipped_heads"] == ["tissue", "infection", "moisture"]
    assert (tmp_path / "run" / "best.pt").exists()


def test_train_with_partial_tim_labels_and_patients(tmp_path):
    import pandas as pd

    rows = make_dataset(tmp_path / "data", with_patients=True)
    csv = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    cfg = write_cfg(tmp_path, tmp_path / "data", str(csv))
    res = train.main(["--config", str(cfg), "--no-pretrained", "--out-dir", str(tmp_path / "run")])
    assert {"wound_type", "tissue", "infection"} <= set(res["test"])
    assert res["skipped_heads"] == ["moisture"]
    splits = pd.read_csv(tmp_path / "run" / "splits.csv")
    by_patient = splits.groupby("patient_id")["split"].nunique()
    assert (by_patient == 1).all(), "a patient leaked across splits"

    s = survey.survey(str(tmp_path / "data"), str(csv))
    assert s["n_images"] == 80 and len(s["class_counts"]) == 8
    assert s["label_coverage"]["tissue"]["labelled"] == 80
    json.dumps(s, default=str)
