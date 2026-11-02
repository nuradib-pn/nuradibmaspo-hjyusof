"""Train and evaluate the multi-task baseline.

    python -m wound_tim.train --config configs/baseline.yaml [--data-root ...] [--labels-csv ...]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .data import HEAD_NAMES, IGNORE, WoundDataset, build_index, make_transforms, split_index
from .model import WoundTIMNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(y: np.ndarray, n: int) -> torch.Tensor:
    y = y[y != IGNORE]
    counts = np.bincount(y, minlength=n).astype(float)
    w = np.where(counts > 0, counts.sum() / (n * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def multitask_loss(outputs, targets, criteria, weights):
    total, parts = 0.0, {}
    for name, logits in outputs.items():
        y = targets[name]
        if (y != IGNORE).any():  # a batch with no labels for a head contributes nothing
            loss = criteria[name](logits, y)
            parts[name] = loss.item()
            total = total + weights[name] * loss
    return total, parts


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues = {}, {}
    for x, t in loader:
        out = model(x.to(device))
        for name, logits in out.items():
            preds.setdefault(name, []).append(logits.argmax(1).cpu().numpy())
            trues.setdefault(name, []).append(t[name].numpy())
    return {k: np.concatenate(v) for k, v in preds.items()}, {k: np.concatenate(v) for k, v in trues.items()}


def score(preds, trues, vocabs) -> dict:
    report = {}
    for name in preds:
        m = trues[name] != IGNORE
        if not m.any():
            continue
        y, p = trues[name][m], preds[name][m]
        labels = list(range(len(vocabs[name])))
        report[name] = {
            "n": int(m.sum()),
            "accuracy": accuracy_score(y, p),
            "balanced_accuracy": balanced_accuracy_score(y, p),
            "macro_f1": f1_score(y, p, labels=labels, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(y, p, labels=labels).tolist(),
            "per_class": classification_report(y, p, labels=labels, target_names=vocabs[name], output_dict=True, zero_division=0),
        }
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--data-root")
    ap.add_argument("--labels-csv")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--out-dir")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (offline smoke tests)")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    for key, val in (("data_root", args.data_root), ("labels_csv", args.labels_csv)):
        if val:
            cfg[key] = val
    tcfg = cfg["train"]
    if args.epochs:
        tcfg["epochs"] = args.epochs
    if args.out_dir:
        tcfg["out_dir"] = args.out_dir
    if args.no_pretrained:
        cfg["model"]["pretrained"] = False

    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(tcfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    df, vocabs = build_index(cfg)
    df = split_index(df, cfg["val_fraction"], cfg["test_fraction"], cfg["seed"])
    df.drop(columns=["path"]).to_csv(out / "splits.csv", index=False)
    train_df = df[df.split == "train"]

    # Only keep T / I / M heads that have at least one labelled training image.
    active = ["wound_type"] + [h for h in HEAD_NAMES if (train_df[f"{h}_y"] != IGNORE).any()]
    skipped = [h for h in HEAD_NAMES if h not in active]
    print(f"device={device}  images={len(df)}  split={df.split.value_counts().to_dict()}")
    print(f"active heads={active}" + (f"  (no labels, skipped: {skipped})" if skipped else ""))

    head_weights = {"wound_type": 1.0, **{h: cfg["heads"][h]["weight"] for h in HEAD_NAMES}}
    criteria = {}
    for h in active:
        w = class_weights(train_df[f"{h}_y"].to_numpy(), len(vocabs[h])).to(device) if tcfg["class_weighting"] else None
        criteria[h] = nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE)

    mk = lambda split, train: DataLoader(  # noqa: E731
        WoundDataset(df[df.split == split], make_transforms(cfg["image_size"], train)),
        batch_size=tcfg["batch_size"],
        shuffle=train,
        num_workers=tcfg["num_workers"],
        pin_memory=device.type == "cuda",
    )
    loaders = {"train": mk("train", True), "val": mk("val", False), "test": mk("test", False)}

    mcfg = cfg["model"]
    model = WoundTIMNet({h: len(vocabs[h]) for h in active}, mcfg["backbone"], mcfg["pretrained"], mcfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["epochs"])

    best, best_epoch, history = -1.0, -1, []
    for epoch in range(tcfg["epochs"]):
        model.set_backbone_trainable(epoch >= tcfg["freeze_backbone_epochs"])
        model.train()
        t0, running, nb = time.time(), 0.0, 0
        for x, t in loaders["train"]:
            x = x.to(device)
            t = {k: v.to(device) for k, v in t.items() if k in active}
            loss, _ = multitask_loss(model(x), t, criteria, head_weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running, nb = running + loss.item(), nb + 1
        sched.step()

        val = score(*predict(model, loaders["val"], device), vocabs)
        # Model selection: mean macro-F1 across every head that has validation labels.
        sel = float(np.mean([r["macro_f1"] for r in val.values()]))
        history.append({"epoch": epoch, "train_loss": running / max(nb, 1), "val_mean_macro_f1": sel, **{f"val_{h}_macro_f1": r["macro_f1"] for h, r in val.items()}})
        print(f"epoch {epoch:3d}  loss {running / max(nb, 1):.4f}  val mean macro-F1 {sel:.4f}  ({time.time() - t0:.0f}s)")
        if sel > best:
            best, best_epoch = sel, epoch
            torch.save({"model": model.state_dict(), "vocabs": vocabs, "active_heads": active, "config": cfg}, out / "best.pt")
        elif epoch - best_epoch >= tcfg["early_stop_patience"]:
            print(f"early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    test = score(*predict(model, loaders["test"], device), vocabs)
    result = {"best_epoch": best_epoch, "best_val_mean_macro_f1": best, "skipped_heads": skipped, "vocabs": vocabs, "test": test, "history": history}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print("\nTEST")
    for h, r in test.items():
        print(f"  {h:11s} n={r['n']:5d}  acc={r['accuracy']:.3f}  bal_acc={r['balanced_accuracy']:.3f}  macro_f1={r['macro_f1']:.3f}")
    return result


if __name__ == "__main__":
    main()
