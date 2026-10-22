"""Estimacion de la vida util restante (RUL) sobre el benchmark C-MAPSS de la NASA.

Entrada: un DataFrame con una fila por (unidad, ciclo): ``unit``, ``cycle``, ``op1..op3`` y ``s1..s21``.
Salida: RUL estimada, siempre >= 0.

Decisiones (todas conocidas del benchmark, ninguna ajustada mirando el conjunto de prueba):

* Etiqueta de entrenamiento: ``RUL = ciclo_final - ciclo`` recortada a ``cap`` (125 por defecto). Al inicio de la vida de un
  motor los sensores no distinguen 300 de 200 ciclos restantes; recortar evita que el modelo intente explicar ese ruido.
  El recorte solo afecta el entrenamiento: la evaluacion usa la RUL verdadera del archivo ``RUL_FD00x.txt`` sin recortar.
* Sensores: se descartan los constantes en el entrenamiento. Se normalizan por condicion de operacion (KMeans sobre
  ``op1..op3``; con una sola condicion equivale a estandarizar), porque en FD002 y FD004 el regimen de vuelo mueve los
  sensores mas que el desgaste.
* Variables: valor actual, media y desviacion en ventana de 10 ciclos, y cambio respecto a 10 ciclos antes, calculadas solo
  con el pasado de cada motor (una prediccion en el ciclo t no ve el ciclo t+1).
* Metricas oficiales: RMSE y el score asimetrico de la NASA, que castiga mas sobreestimar la RUL (avisar tarde) que
  subestimarla (avisar temprano).

Modelos: ``ridge`` (lineal), ``random_forest``, ``lightgbm`` y ``xgboost`` sobre las variables de ventana
(``RulEstimator``), y ``cnn`` (convolucion 1D) y ``gru`` (red recurrente) sobre la secuencia de los ultimos 30 ciclos
normalizados (``SequenceRulEstimator``), entrenadas con ``src.models.nets``: minimo 100 epocas, parada temprana
con motores separados del entrenamiento y restauracion del mejor checkpoint.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OP_COLUMNS = ["op1", "op2", "op3"]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *OP_COLUMNS, *SENSOR_COLUMNS]
WINDOW = 10
SEQ_WINDOW = 30
SEED = 0
TABULAR_KINDS = ["ridge", "random_forest", "lightgbm", "xgboost"]
SEQUENCE_KINDS = ["cnn", "gru"]


# ---------------------------------------------------------------------------- metricas
def rmse(y_true, y_pred) -> float:
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean(d**2)))


def nasa_score(y_true, y_pred) -> float:
    """Score asimetrico de la NASA (PHM08): suma de exp(-d/13) - 1 si d < 0 (subestima) y exp(d/10) - 1 si d >= 0
    (sobreestima), con d = prediccion - RUL verdadera. Menor es mejor; sobreestimar pesa mas (10 < 13)."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


# ---------------------------------------------------------------------------- etiquetas
def train_rul(df: pd.DataFrame, cap: float | None = 125) -> pd.Series:
    """RUL de entrenamiento: ciclos hasta el ultimo observado de cada motor, recortada a ``cap``."""
    rul = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    return rul.clip(upper=cap) if cap is not None else rul.astype(float)


def last_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Ultima observacion de cada unidad, ordenada por unidad (lo que evalua el benchmark en el conjunto de prueba)."""
    return df.sort_values(["unit", "cycle"]).groupby("unit", as_index=False).tail(1).reset_index(drop=True)


# ---------------------------------------------------------------------------- modelo
def _make_model(kind: str):
    if kind == "lightgbm":
        return LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.7, random_state=SEED, verbosity=-1)
    if kind == "random_forest":
        return RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=SEED)
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if kind == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.7,
                            min_child_weight=5, tree_method="hist", random_state=SEED, n_jobs=-1)
    raise ValueError(f"modelo desconocido: {kind}")


class RulEstimator:
    def __init__(self, kind: str = "lightgbm", cap: float | None = 125, n_conditions: int | None = None):
        self.kind, self.cap, self.n_conditions = kind, cap, n_conditions

    # -- preparacion de variables
    def _fit_scaler(self, df: pd.DataFrame) -> None:
        self.sensors_ = [c for c in SENSOR_COLUMNS if c in df.columns and df[c].std() > 1e-9]
        if self.n_conditions is None:  # una condicion si los ajustes de operacion casi no varian (FD001, FD003)
            self.n_conditions = 6 if df[OP_COLUMNS].round(1).drop_duplicates().shape[0] > 10 else 1
        self.kmeans_ = (KMeans(self.n_conditions, n_init=10, random_state=SEED).fit(df[OP_COLUMNS])
                        if self.n_conditions > 1 else None)
        cond = self._condition(df)
        grouped = df[self.sensors_].groupby(cond)
        self.mean_, self.std_ = grouped.mean(), grouped.std().replace(0, 1.0).fillna(1.0)

    def _condition(self, df: pd.DataFrame) -> np.ndarray:
        return self.kmeans_.predict(df[OP_COLUMNS]) if self.kmeans_ is not None else np.zeros(len(df), dtype=int)

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
        cond = pd.Series(self._condition(df))
        z = (df[self.sensors_] - self.mean_.loc[cond].to_numpy()) / self.std_.loc[cond].to_numpy()
        z["unit"] = df["unit"]
        g = z.groupby("unit")[self.sensors_]
        out = [z[self.sensors_].add_suffix("_z"),
               g.transform(lambda s: s.rolling(WINDOW, min_periods=1).mean()).add_suffix("_m10"),
               g.transform(lambda s: s.rolling(WINDOW, min_periods=2).std()).fillna(0.0).add_suffix("_sd10"),
               (z[self.sensors_] - g.shift(WINDOW)).fillna(0.0).add_suffix("_d10")]
        X = pd.concat(out, axis=1)
        X["cycle"] = df["cycle"].to_numpy()
        X.index = pd.MultiIndex.from_arrays([df["unit"], df["cycle"]], names=["unit", "cycle"])
        return X

    # -- API
    def fit(self, train: pd.DataFrame) -> "RulEstimator":
        self._fit_scaler(train)
        X = self.features(train)
        y = train_rul(train.sort_values(["unit", "cycle"]).reset_index(drop=True), self.cap).to_numpy()
        self.columns_ = list(X.columns)
        self.model_ = _make_model(self.kind).fit(X.to_numpy(), y)
        return self

    def predict_rows(self, df: pd.DataFrame) -> pd.Series:
        """RUL estimada para cada fila (unit, cycle), nunca negativa."""
        X = self.features(df)
        return pd.Series(np.clip(self.model_.predict(X[self.columns_].to_numpy()), 0.0, None), index=X.index)

    def predict_last(self, df: pd.DataFrame) -> pd.Series:
        """RUL estimada en la ultima observacion de cada unidad (indice: unit)."""
        p = self.predict_rows(df)
        last = df.groupby("unit")["cycle"].max()
        return pd.Series([p.loc[(u, c)] for u, c in last.items()], index=last.index, name="rul_pred")


# ---------------------------------------------------------------------------- redes sobre secuencias
def _cnn(n_features: int, window: int):
    from torch import nn

    return nn.Sequential(
        nn.Conv1d(n_features, 32, 5, padding=2), nn.ReLU(),
        nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
        nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
        nn.Flatten(), nn.Dropout(0.2), nn.Linear(64 * window, 64), nn.ReLU(), nn.Linear(64, 1),
    )


def _gru(n_features: int):
    import torch
    from torch import nn

    class GRURegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(n_features, 64, num_layers=2, batch_first=True, dropout=0.2)
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):  # x: (lote, variables, tiempo), el mismo formato que la CNN
            out, _ = self.gru(torch.transpose(x, 1, 2))
            return self.head(out[:, -1])

    return GRURegressor()


class SequenceRulEstimator(RulEstimator):
    """CNN 1D o GRU sobre los ultimos ``window`` ciclos de cada motor (sensores normalizados por condicion).

    La ventana del ciclo t usa los ciclos t-window+1..t del mismo motor; si el motor tiene menos historia se rellena
    hacia atras repitiendo su primer ciclo. La etiqueta es la RUL recortada dividida por el techo (queda en [0, 1]).
    ``valid_fraction`` de los motores de entrenamiento se separa para la parada temprana.
    """

    def __init__(self, kind: str = "cnn", cap: float | None = 125, n_conditions: int | None = None,
                 window: int = SEQ_WINDOW, valid_fraction: float = 0.15, seed: int = SEED, train_config=None):
        if kind not in SEQUENCE_KINDS:
            raise ValueError(f"modelo desconocido: {kind}")
        super().__init__(kind, cap, n_conditions)
        self.window, self.valid_fraction, self.seed = window, valid_fraction, seed
        self.train_config = train_config

    def windows(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.MultiIndex]:
        df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
        cond = pd.Series(self._condition(df))
        z = ((df[self.sensors_] - self.mean_.loc[cond].to_numpy()) / self.std_.loc[cond].to_numpy()).to_numpy(np.float32)
        blocks = []
        for _, idx in df.groupby("unit", sort=True).indices.items():
            unit = z[idx]
            padded = np.vstack([np.repeat(unit[:1], self.window - 1, axis=0), unit])
            blocks.append(np.lib.stride_tricks.sliding_window_view(padded, self.window, axis=0))  # (T, F, window)
        index = pd.MultiIndex.from_arrays([df["unit"], df["cycle"]], names=["unit", "cycle"])
        return np.concatenate(blocks).astype(np.float32), index

    def fit(self, train: pd.DataFrame) -> "SequenceRulEstimator":
        import torch

        from src.models import nets

        train = train.sort_values(["unit", "cycle"]).reset_index(drop=True)
        self._fit_scaler(train)
        units = np.array(sorted(train["unit"].unique()))
        rng = np.random.default_rng(self.seed)
        valid_units = set(rng.choice(units, max(1, int(len(units) * self.valid_fraction)), replace=False))
        y = (train_rul(train, self.cap) / self.cap).to_numpy(np.float32).reshape(-1, 1)
        X, _ = self.windows(train)
        is_valid = train["unit"].isin(valid_units).to_numpy()
        config = self.train_config or nets.TrainConfig(batch_size=256, seed=self.seed)
        nets.set_seed(self.seed)
        self.model_ = _cnn(X.shape[1], self.window) if self.kind == "cnn" else _gru(X.shape[1])
        loss = torch.nn.MSELoss(reduction="none")
        self.history_ = nets.fit_network(self.model_, loss, X[~is_valid], y[~is_valid], None,
                                         X[is_valid], y[is_valid], None, config)
        self.info = {"best_epoch": self.history_.best_epoch, "epochs_run": self.history_.epochs_run}
        return self

    def predict_rows(self, df: pd.DataFrame) -> pd.Series:
        from src.models import nets

        X, index = self.windows(df)
        pred = nets.predict(self.model_, X).reshape(-1) * self.cap
        return pd.Series(np.clip(pred, 0.0, None), index=index)


def make_estimator(kind: str, cap: float | None = 125, **kwargs) -> RulEstimator:
    return SequenceRulEstimator(kind, cap=cap, **kwargs) if kind in SEQUENCE_KINDS else RulEstimator(kind, cap=cap)
