# Wound classifier baseline (wound type + T / I / M)

A multi-task baseline that predicts, from one wound photo:

| Head         | What it predicts                   | Classes                                           |
| ------------ | ---------------------------------- | ------------------------------------------------- |
| `wound_type` | the 8 TIME wound types             | folder names (`TYPE 1` ... `TYPE 8`)              |
| `tissue`     | **T**: necrotic tissue / slough    | `lt25` (<25%, score 0), `gt25` (>25%, score 4)    |
| `infection`  | **I**: clinical signs of infection | `absent` (score 0), `present` (score 2)           |
| `moisture`   | **M**: exudate                     | `dry_minimal` (score 1), `moderate_wet` (score 2) |

The clinical manual defines the wound type as the score T + I + M, so each type is one T / I / M combination:

| Type | T    | I       | M            |
| ---- | ---- | ------- | ------------ |
| 1    | <25% | absent  | dry/minimal  |
| 2    | <25% | absent  | moderate/wet |
| 3    | <25% | present | dry/minimal  |
| 4    | <25% | present | moderate/wet |
| 5    | >25% | absent  | dry/minimal  |
| 6    | >25% | absent  | moderate/wet |
| 7    | >25% | present | dry/minimal  |
| 8    | >25% | present | moderate/wet |

With `derive_tim_from_type: true` (the default) the folder name alone gives all four labels. The test report also rebuilds
the type from the three T / I / M predictions (`wound_type_from_tim`) so it can be compared with the direct 8-way head.

One ImageNet-pretrained backbone (ResNet-50 by default) feeds four linear heads with class-weighted cross-entropy.

## Data handling

- **Duplicates.** Byte-identical files are removed; a file filed under two different types is dropped entirely.
- **Leakage.** There are no patient IDs, and the same wound is often photographed several times in one sitting. With
  `group_by: capture_date` the split is grouped by the date in the filename (`20201118_105218.jpg`,
  `IMG-20201105-WA0012.jpg`), so same-day photos never straddle train and test. Undated files are their own group. A
  `patient_id` column in `labels_csv` overrides this.
- **Split.** 70 / 15 / 15, stratified by wound type. The best epoch is picked by mean validation macro-F1 over all heads.

## Run in Colab (recommended)

Open [`colab_baseline.ipynb`](colab_baseline.ipynb) in Colab with a GPU runtime and run all. It mounts Drive, copies the
`Wound categorisation` folder to local disk, clones this branch, then runs the survey and training. Results are saved
back to Drive.

## Run locally

```bash
cd wound_classifier
python -m pip install -r requirements.txt
python -m wound_tim.survey --data-root "data/Wound categorisation" --out survey.json
python -m wound_tim.train --config configs/baseline.yaml --data-root "data/Wound categorisation"
python -m pytest            # smoke tests on synthetic images
```

If `download.pytorch.org` is blocked, pass a local ImageNet checkpoint with `--weights-path` (timm ResNet checkpoints from
the pytorch-image-models GitHub releases use torchvision's parameter names).

Outputs go to `runs/baseline/`: `best.pt` (checkpoint with vocabularies), `splits.csv`, `test_predictions.csv`, and
`metrics.json` (accuracy, balanced accuracy, macro-F1, confusion matrix and per-class report for each head).

## Optional `labels_csv`

Columns: `relpath` (e.g. `TYPE 3/img001.jpg`) or `filename`, then any of `tissue`, `infection`, `moisture`, `patient_id`.
Values given here take priority over the ones derived from the type; leave a cell empty when unknown.
