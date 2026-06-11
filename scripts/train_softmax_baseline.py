#!/usr/bin/env python3
"""Train the closed-set softmax baseline (Baseline 0 per ADR-0001).

DGCNN encoder + K-way per-point classification head, trained with
Hungarian-matching cross-entropy. Heavy lift — requires a working CUDA / PyG
environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the closed-set softmax baseline.")
    p.add_argument(
        "--num-classes",
        type=int,
        default=8,
        help="K, the closed-set class count. roofNTNU's observed max is 8.",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-root", default="data/roofNTNU/train_test_split")
    p.add_argument("--output-dir", default="artifacts/softmax_baseline")
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    import torch
    from torch.utils.data import DataLoader

    from roofseg.baselines.softmax_classifier import (
        HungarianClassificationLoss,
        SoftmaxClassifier,
    )
    from roofseg.device import describe_device, select_device
    from roofseg.seed import set_seed
    import trainModel as tm

    set_seed(args.seed)
    device = select_device(args.device)
    print(f"Device: {describe_device(device)}")

    config = tm.Config()
    if args.batch_size:
        config.batch_size = args.batch_size

    train_ds = tm.LiDARPointCloudDataset(
        base_dir=args.data_root,
        split="train",
        max_points=config.max_points,
        sampling_method=config.sampling_method,
    )
    val_ds = tm.LiDARPointCloudDataset(
        base_dir=args.data_root,
        split="val",
        max_points=config.max_points,
        sampling_method=config.sampling_method,
    )
    print(f"train scenes: {len(train_ds)} | val scenes: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    model = SoftmaxClassifier(config=config, num_classes=args.num_classes).to(device)
    loss_fn = HungarianClassificationLoss(num_classes=args.num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pth"
    log_path = out_dir / "train_log.json"

    history = {
        "num_classes": args.num_classes,
        "epochs": [],
        "best_val_loss": float("inf"),
        "best_epoch": 0,
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n_train = 0
        for points, labels, _ in train_loader:
            points = points.to(device).float()
            labels = labels.to(device).long()
            optim.zero_grad()
            logits = model(points)
            loss = loss_fn(logits, labels)
            loss.backward()
            optim.step()
            train_loss += loss.item()
            n_train += 1
        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for points, labels, _ in val_loader:
                points = points.to(device).float()
                labels = labels.to(device).long()
                logits = model(points)
                val_loss += loss_fn(logits, labels).item()
                n_val += 1
        val_loss /= max(n_val, 1)

        history["epochs"].append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )
        if val_loss < history["best_val_loss"]:
            history["best_val_loss"] = val_loss
            history["best_epoch"] = epoch
            torch.save(model.state_dict(), ckpt_path)
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)
        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"({time.time() - t0:.0f}s)"
        )

    print(f"\nBest val_loss={history['best_val_loss']:.4f} at epoch {history['best_epoch']}")
    print(f"Checkpoint: {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
