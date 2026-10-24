"""GRU sobre el historial de lecturas de cada camion de SCANIA Component X.

Los modelos tabulares ven una lectura (con sus tasas recientes ya resumidas). Este modelo ve las ultimas ``LAGS``
lecturas del camion, cada una con las mismas features invariantes a la escala, y aprende la trayectoria.

* ``add_lags`` agrega, para cada lectura, las ``LAGS`` lecturas previas del mismo camion (``<col>__lag<k>``, con
  ``k = 0`` la actual) y un indicador ``__pad<k>`` cuando el camion todavia no tenia esa lectura. Solo mira hacia atras.
* ``GRUSequenceClassifier`` tiene la misma interfaz que ``zoo.TabularModel`` (``fit`` con conjunto de parada temprana,
  ``predict_proba`` sobre un DataFrame), asi que entra al mismo protocolo de seleccion. Se entrena con
  ``src.models.nets``: minimo 100 epocas, parada temprana y checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch import nn

from src.models import nets
from src.models.zoo import signed_log1p

LAGS = 8
N_FEATURES = 48


def lag_columns(cols: list[str], lags: int = LAGS) -> list[str]:
    """Orden temporal de la secuencia: de la lectura mas antigua (``lag<lags-1>``) a la actual (``lag0``)."""
    out = []
    for k in reversed(range(lags)):
        out += [f"{c}__lag{k}" for c in cols] + [f"__pad{k}"]
    return out


def add_lags(feats: pl.DataFrame, cols: list[str], lags: int = LAGS) -> pl.DataFrame:
    """``vehicle_id``, ``time_step`` y las columnas de ``lag_columns``, una fila por lectura."""
    vid = "vehicle_id"
    df = feats.select([vid, *dict.fromkeys(["time_step", *cols])]).sort([vid, "time_step"])  # time_step puede estar en cols
    exprs = []
    for k in range(lags):
        exprs += [pl.col(c).shift(k).over(vid).cast(pl.Float32).alias(f"{c}__lag{k}") for c in cols]
        exprs.append((pl.int_range(pl.len()).over(vid) < k).cast(pl.Float32).alias(f"__pad{k}"))
    return df.select([vid, "time_step", *exprs])


def top_features(lgb_model, columns: list[str], categorical: list[str], n: int = N_FEATURES) -> list[str]:
    """Las ``n`` features numericas con mayor ganancia en un LightGBM ya ajustado (solo con datos de ajuste)."""
    gain = pd.Series(lgb_model.booster_.feature_importance("gain"), index=lgb_model.booster_.feature_name())
    gain = gain[[c for c in gain.index if c not in categorical]]
    return [c for c in gain.sort_values(ascending=False).index[:n] if c in columns]


class _GRUNet(nn.Module):
    def __init__(self, steps: int, step_features: int, n_static: int, n_classes: int, hidden: int = 64):
        super().__init__()
        self.steps, self.step_features = steps, step_features
        self.gru = nn.GRU(step_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_classes))

    def forward(self, x):
        seq = x[:, : self.steps * self.step_features].reshape(-1, self.steps, self.step_features)
        static = x[:, self.steps * self.step_features:]
        out, _ = self.gru(seq)
        return self.head(torch.cat([out[:, -1], static], dim=1))


class GRUSequenceClassifier:
    name = "gru"


    def __getstate__(self):
        # La carpeta de checkpoints es una ruta local (WindowsPath en Windows) que no se puede reconstruir en otro
        # sistema operativo; el modelo entrenado no la necesita.
        state = dict(self.__dict__)
        state["checkpoint_dir"] = None
        return state
    def __init__(self, n_classes: int, base_cols: list[str], categorical: list[str], lags: int = LAGS, seed: int = 42,
                 checkpoint_dir: Path | None = None, train_config: nets.TrainConfig | None = None):
        self.n_classes, self.base_cols, self.categorical = n_classes, list(base_cols), list(categorical)
        self.lags, self.seed, self.checkpoint_dir = lags, seed, checkpoint_dir
        self.train_config = train_config
        self.seq_cols = lag_columns(self.base_cols, lags)
        self.info: dict = {}

    # ------------------------------------------------------------------ preparacion
    def _fit_prep(self, X: pd.DataFrame) -> None:
        values = signed_log1p(X[[f"{c}__lag0" for c in self.base_cols]].to_numpy(dtype=float))
        self.mean_ = np.nanmean(values, axis=0)
        self.std_ = np.nanstd(values, axis=0)
        self.std_[~np.isfinite(self.std_) | (self.std_ < 1e-9)] = 1.0
        self.mean_[~np.isfinite(self.mean_)] = 0.0
        self.levels_ = {c: [v for v, n in X[c].astype(str).value_counts().items() if n >= 50] for c in self.categorical}

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        blocks = []
        for k in reversed(range(self.lags)):
            v = signed_log1p(X[[f"{c}__lag{k}" for c in self.base_cols]].to_numpy(dtype=float))
            v = np.nan_to_num((v - self.mean_) / self.std_, nan=0.0, posinf=0.0, neginf=0.0).clip(-10, 10)
            blocks += [v, X[[f"__pad{k}"]].to_numpy(dtype=float)]
        static = [(X[c].astype(str).to_numpy()[:, None] == np.array(levels)[None, :]).astype(float)
                  for c, levels in self.levels_.items()]
        return np.hstack(blocks + static).astype(np.float32)

    # ------------------------------------------------------------------ API comun
    def fit(self, X_tr, y_tr, w_tr=None, X_va=None, y_va=None, w_va=None):
        self._fit_prep(X_tr)
        A_tr, A_va = self._transform(X_tr), self._transform(X_va)
        n_static = sum(len(v) for v in self.levels_.values())
        cfg = self.train_config or nets.TrainConfig(batch_size=512, seed=self.seed)
        if self.checkpoint_dir is not None:
            cfg.checkpoint_path = Path(self.checkpoint_dir) / f"gru_seed{self.seed}.pt"
        nets.set_seed(self.seed)
        self.net_ = _GRUNet(self.lags, len(self.base_cols) + 1, n_static, self.n_classes)
        loss = nn.CrossEntropyLoss(reduction="none")
        w_tr = np.ones(len(A_tr)) if w_tr is None else np.asarray(w_tr, dtype=float)
        w_va = np.ones(len(A_va)) if w_va is None else np.asarray(w_va, dtype=float)
        hist = nets.fit_network(self.net_, loss, A_tr, np.asarray(y_tr).astype(int), w_tr, A_va, np.asarray(y_va).astype(int),
                                w_va, cfg, y_dtype=torch.long)
        self.info = {"best_epoch": hist.best_epoch, "epochs_run": hist.epochs_run, "lags": self.lags,
                     "step_features": len(self.base_cols)}
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        from scipy.special import softmax

        return softmax(nets.predict(self.net_, self._transform(X)), axis=1)
