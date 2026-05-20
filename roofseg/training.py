"""Reusable PointUNet training loop.

Factored out of :func:`trainModel.train_pipeline` so we can train on
arbitrary :class:`torch.utils.data.Dataset` subsets (e.g. one fold of the
training data during cross-prediction harvesting) without duplicating the
loop.

Lazy imports of PyG-touching modules; safe to ``import roofseg.training``
on machines without a working PyG/torch_scatter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class TrainHistory:
    epochs: list[dict[str, float]] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = -1


def train_pointunet(
    train_dataset,
    val_dataset,
    *,
    config: Any,
    epochs: int,
    save_path: str | Path,
    device: torch.device,
    log_path: str | Path | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    verbose: bool = True,
) -> tuple[torch.nn.Module, TrainHistory]:
    """Train PointUNet end-to-end on the supplied datasets.

    ``train_dataset`` / ``val_dataset`` can be the raw
    :class:`LiDARPointCloudDataset` instances OR a
    :class:`torch.utils.data.Subset` over one. Validation may be ``None`` (no
    early stopping; best model = last epoch).

    The function saves the best checkpoint (lowest val loss, or last if val
    is None) to ``save_path`` and returns the in-memory model loaded with
    those weights plus a :class:`TrainHistory` for logging.
    """
    # Lazy imports — these touch PyG and only work on the CUDA server.
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from roofseg.models import build_pointunet

    bs = batch_size if batch_size is not None else config.batch_size
    lr = learning_rate if learning_rate is not None else config.lr

    model = build_pointunet(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # DiscriminativeLoss lives in trainModel.
    import trainModel as tm

    criterion = tm.DiscriminativeLoss(
        delta_v=config.delta_v,
        delta_d=config.delta_d,
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
    ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, pin_memory=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=bs, shuffle=False, pin_memory=True)
        if val_dataset is not None
        else None
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    history = TrainHistory()
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]", disable=not verbose)
        for points, labels, _ in loop:
            points = points.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            embeddings = model(points)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item())
            loop.set_postfix(loss=float(loss.item()))

        avg_train = train_loss_sum / max(len(train_loader), 1)
        avg_val: float | None = None

        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for points, labels, _ in val_loader:
                    points = points.to(device)
                    labels = labels.to(device)
                    embeddings = model(points)
                    val_loss_sum += float(criterion(embeddings, labels).item())
            avg_val = val_loss_sum / max(len(val_loader), 1)
            if avg_val < history.best_val_loss:
                history.best_val_loss = avg_val
                history.best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.epochs.append(
            {"epoch": epoch + 1, "train_loss": avg_train, "val_loss": float(avg_val) if avg_val is not None else float("nan")}
        )
        if verbose:
            if avg_val is not None:
                print(f"  epoch {epoch + 1}: train={avg_train:.4f} val={avg_val:.4f}")
            else:
                print(f"  epoch {epoch + 1}: train={avg_train:.4f}")

        if log_path is not None:
            with open(log_path, "w") as f:
                json.dump(
                    {
                        "epochs": history.epochs,
                        "best_val_loss": history.best_val_loss,
                        "best_epoch": history.best_epoch,
                    },
                    f,
                    indent=2,
                )

    # No val loader → keep last weights.
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    torch.save(best_state, str(save_path))
    model.load_state_dict(best_state)
    model.eval()
    return model, history
