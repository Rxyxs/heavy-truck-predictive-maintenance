"""Redes neuronales en PyTorch con el mismo protocolo de entrenamiento en todo el repositorio.

* Minimo ``min_epochs`` epocas (100 por defecto) antes de que la parada temprana pueda actuar: una red
  cortada a las 20 epocas suele quedar lejos de converger y la comparacion con los arboles seria injusta.
* Parada temprana por paciencia sobre la perdida de VALIDACION (un conjunto separado del de entrenamiento,
  nunca el de prueba), y se restaura el mejor punto de control (checkpoint), no el ultimo.
* El mejor estado se guarda en memoria y, si se pide, en disco (``checkpoint_path``).

Si hay GPU (CUDA) se entrena ahi; si no, en CPU. Con la semilla fija las corridas en CPU son deterministicas;
en GPU pueden variar en los ultimos decimales.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class TrainConfig:
    min_epochs: int = 100
    max_epochs: int = 300
    patience: int = 20
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    checkpoint_path: Path | None = None


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    epochs_run: int = 0


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """Perceptron multicapa con normalizacion por lotes y dropout."""

    def __init__(self, n_in: int, n_out: int, hidden: tuple[int, ...] = (256, 128), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _batches(n: int, batch_size: int, rng: np.random.Generator | None, dev: torch.device | str = "cpu"):
    idx = torch.as_tensor(rng.permutation(n) if rng is not None else np.arange(n), device=dev)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


def _weighted_loss(loss_fn, out, y, w):
    per_row = loss_fn(out, y)
    if per_row.ndim > 1:
        per_row = per_row.mean(dim=tuple(range(1, per_row.ndim)))
    return (per_row * w).sum() / w.sum()


def fit_network(
    model: nn.Module,
    loss_fn,
    X_tr: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray | None,
    X_va: np.ndarray, y_va: np.ndarray, w_va: np.ndarray | None,
    config: TrainConfig = TrainConfig(),
    y_dtype=torch.float32,
) -> TrainHistory:
    """Entrena ``model`` con Adam y devuelve la historia; deja cargado el mejor estado de validacion.

    ``loss_fn`` debe devolver la perdida por fila (``reduction="none"``) para poder ponderarla.
    """
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    dev = device()
    model.to(dev)

    def tensor(a, dtype):
        return torch.as_tensor(a, dtype=dtype, device=dev)

    Xt, Xv = tensor(X_tr, torch.float32), tensor(X_va, torch.float32)
    yt, yv = tensor(y_tr, y_dtype), tensor(y_va, y_dtype)
    wt = tensor(np.ones(len(X_tr)) if w_tr is None else w_tr, torch.float32)
    wv = tensor(np.ones(len(X_va)) if w_va is None else w_va, torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(config.patience // 2, 1))

    history = TrainHistory()
    best_loss, best_state, since_best = np.inf, copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total, weight = 0.0, 0.0
        for b in _batches(len(Xt), config.batch_size, rng, dev):
            if len(b) < 2:  # BatchNorm no admite lotes de una fila
                continue
            opt.zero_grad()
            loss = _weighted_loss(loss_fn, model(Xt[b]), yt[b], wt[b])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * float(wt[b].sum())
            weight += float(wt[b].sum())
        val = evaluate_loss(model, loss_fn, Xv, yv, wv, config.batch_size)
        sched.step(val)
        history.train_loss.append(total / max(weight, 1e-12))
        history.val_loss.append(val)
        if val < best_loss - 1e-6:
            best_loss, best_state, since_best = val, copy.deepcopy(model.state_dict()), 0
            history.best_epoch = epoch
            if config.checkpoint_path is not None:
                config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({k: v.cpu() for k, v in best_state.items()}, config.checkpoint_path)
        else:
            since_best += 1
        history.epochs_run = epoch
        if epoch >= config.min_epochs and since_best >= config.patience:
            break
    model.load_state_dict(best_state)
    model.to("cpu")  # el modelo entrenado vive en CPU: se puede serializar y predecir sin GPU
    model.eval()
    return history


@torch.no_grad()
def evaluate_loss(model, loss_fn, X, y, w, batch_size: int = 4096) -> float:
    model.eval()
    total, weight = 0.0, 0.0
    for b in _batches(len(X), batch_size, None, X.device):
        loss = _weighted_loss(loss_fn, model(X[b]), y[b], w[b])
        total += float(loss) * float(w[b].sum())
        weight += float(w[b].sum())
    return total / max(weight, 1e-12)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    Xt = torch.as_tensor(X, dtype=torch.float32)
    return torch.cat([model(Xt[b]) for b in _batches(len(Xt), batch_size, None)]).numpy()
