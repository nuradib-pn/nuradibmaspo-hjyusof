# Wound classifier baseline (wound type + T / I / M)

A multi-task baseline that predicts, from one wound photo:

| Head         | What it predicts                                  | Default classes (edit in `configs/baseline.yaml`) |
| ------------ | ------------------------------------------------- | ------------------------------------------------- |
| `wound_type` | the 8 wound types (one folder per type)           | discovered from folder names                      |
| `tissue`     | **T**: dominant tissue in the wound bed           | granulation, slough, necrotic, epithelial         |
| `infection`  | **I**: clinical signs of infection / inflammation | absent, present                                   |
| `moisture`   | **M**: exudate level                              | dry, low, moderate, high                          |

One ImageNet-pretrained backbone (ResNet-50 by default) feeds four linear heads. Each head uses class-weighted cross-entropy that
ignores unlabelled images, so T / I / M can be partly labelled or missing entirely: a head with no training labels is dropped and
reported as skipped.

## Data layout

```
data/wounds/
  <wound_type_1>/img001.jpg
  <wound_type_2>/...
  labels.csv            # optional
```

`labels.csv` columns (all optional except the key): `relpath` (e.g. `<wound_type_1>/img001.jpg`) or `filename`, then `tissue`,
`infection`, `moisture`, `patient_id`. Leave a cell empty when it is not annotated. When `patient_id` is present for every image, the
split is grouped by patient so the same patient never appears in both train and test.

## Run

```bash
cd wound_classifier
python -m pip install -r requirements.txt
python -m wound_tim.survey --data-root data/wounds --labels-csv data/wounds/labels.csv --out survey.json
python -m wound_tim.train  --config configs/baseline.yaml --data-root data/wounds --labels-csv data/wounds/labels.csv
python -m pytest            # smoke test on synthetic images
```

Outputs go to `runs/baseline/`: `best.pt` (checkpoint with vocabularies), `splits.csv` (70/15/15 stratified split), and
`metrics.json` (accuracy, balanced accuracy, macro-F1, confusion matrix and per-class report for each head on the test split).
The best epoch is chosen by mean validation macro-F1 across the active heads. A GPU is strongly recommended (Colab works).
