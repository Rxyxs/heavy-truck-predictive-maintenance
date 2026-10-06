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
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

OP_COLUMNS = ["op1", "op2", "op3"]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *OP_COLUMNS, *SENSOR_COLUMNS]
WINDOW = 10
SEED = 0


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
    raise ValueError(f"modelo desconocido: {kind}")


class RulEstimator:
    def __init__(self, kind: str = "lightgbm", cap: float | None = 125, n_conditions: int | None = None):
        self.kind, self.cap, self.n_conditions = kind, cap, n_conditions

    # -- preparacion de variables
    def _fit_scaler(self, df: pd.DataFrame) -> None:
        self.sensors_ = [c for c in SENSOR_COLUMNS if df[c].std() > 1e-9]
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
