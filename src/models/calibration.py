"""Calibracion de la probabilidad de riesgo y sensibilidad del ahorro al costo de una visita.

El clasificador entrega probabilidades por clase, pero entrenado con las lecturas sanas submuestreadas
(y re-ponderadas) y con muchas lecturas por camion, sus probabilidades salen infladas sobre camiones
nuevos: en validacion predice 19,5% de riesgo medio y la tasa observada es 2,7%. La decision de costo
esperado minimo depende de que la probabilidad sea creible, asi que se calibra.

La calibracion es una regresion isotonica sobre ``P(algun riesgo) = 1 - p0`` ajustada con los camiones
retenidos del entrenamiento (el modelo nunca los vio; validacion y test no participan). Despues se
reescalan las clases 1-4 para que sumen el riesgo calibrado, conservando como se reparte el riesgo entre
ellas.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.data.scania_loader import COST_MATRIX
from src.models import cost_decision as C

EPS = 1e-12
SENSITIVITY_MULTIPLIERS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
CALIBRATION_EDGES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0000001)
BREAK_EVEN_THRESHOLD_PCT = 1.0  # un ahorro de 1% o menos se considera agotado
BREAK_EVEN_GRID = np.geomspace(0.05, 2000.0, 301)  # de x0,05 a x2000, pasos de ~3,6%


def fit_risk_calibrator(proba: np.ndarray, y_class: np.ndarray, weight: np.ndarray | None = None) -> IsotonicRegression:
    """Isotonica de ``1 - p0`` contra "este camion termino fallando" (clase > 0)."""
    risk = 1.0 - np.asarray(proba)[:, 0]
    failing = (np.asarray(y_class) > 0).astype(float)
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(risk, failing, sample_weight=weight)


def apply_risk_calibration(proba: np.ndarray, x_thresholds: np.ndarray, y_thresholds: np.ndarray) -> np.ndarray:
    """Probabilidades por clase con el riesgo calibrado; cada fila sigue sumando 1.

    Recibe los umbrales de la isotonica (no el objeto) para que la pagina web, que no tiene sklearn,
    reproduzca exactamente la misma funcion: interpolacion lineal entre umbrales y recorte en los extremos.
    """
    proba = np.asarray(proba, dtype=float)
    risk = 1.0 - proba[:, 0]
    calibrated = np.interp(risk, x_thresholds, y_thresholds)
    scale = np.where(risk > EPS, calibrated / np.maximum(risk, EPS), 0.0)
    out = proba.copy()
    out[:, 1:] = proba[:, 1:] * scale[:, None]
    out[:, 0] = 1.0 - calibrated
    return out


def export_calibrator(calibrator: IsotonicRegression) -> dict:
    # 10 decimales: la isotonica tiene escalones muy empinados y redondear a 6 los corre lo bastante para
    # que la pagina (que solo recibe estos umbrales) se separe de sklearn en ~2e-4.
    return {"x": np.round(calibrator.X_thresholds_, 10).tolist(), "y": np.round(calibrator.y_thresholds_, 10).tolist()}


def scaled_cost_matrix(k: float, cost=None) -> np.ndarray:
    """Matriz de costos con el costo de revisar un camion sano (fila 0, columnas 1-4) multiplicado por ``k``."""
    m = np.array(COST_MATRIX if cost is None else cost, dtype=float)
    m[0, 1:] *= k
    return m


def saving_at(y_true, proba: np.ndarray, k: float) -> dict:
    """Ahorro de la politica de costo esperado minimo frente a no recomendar nada, con visita a ``k`` veces el costo del reto."""
    y_true = np.asarray(y_true)
    m = scaled_cost_matrix(k)
    pred = C.bayes_decision(proba, m)
    policy = float(m[y_true, pred].sum())
    nothing = float(m[y_true, 0].sum())
    return {
        "k": float(k), "costo_politica": policy, "costo_nada": nothing,
        "ahorro_pct": 100.0 * (1.0 - policy / nothing) if nothing else float("nan"),
        "alarmados_pct": 100.0 * float((pred > 0).mean()),
    }


def sensitivity(y_true, proba: np.ndarray, multipliers=SENSITIVITY_MULTIPLIERS) -> list[dict]:
    return [saving_at(y_true, proba, k) for k in multipliers]


def break_even(y_true, proba: np.ndarray) -> float | None:
    """Mayor costo de visita (multiplo del reto) de la grilla en que el ahorro todavia supera el umbral.

    Se recorre una grilla fina y no se hace una biseccion: con probabilidades recalibradas el ahorro queda
    pegado a ~0 (en vez de caer) y una biseccion, que supone un cruce limpio, devolvia cualquier cosa.
    ``None`` si el ahorro sigue por sobre el umbral en toda la grilla; el primer punto de la grilla (x0,05) si
    ya no ahorra en ninguno, de modo que un modelo que pierde al costo del reto devuelve menos de 1.
    """
    last = float(BREAK_EVEN_GRID[0])
    for k in BREAK_EVEN_GRID:
        if saving_at(y_true, proba, float(k))["ahorro_pct"] > BREAK_EVEN_THRESHOLD_PCT:
            last = float(k)
    return None if last >= float(BREAK_EVEN_GRID[-1]) else last


def calibration_table(y_true, proba: np.ndarray, edges=CALIBRATION_EDGES) -> list[dict]:
    """Por tramo de riesgo predicho: cuantos camiones, riesgo medio predicho y tasa de falla observada."""
    risk = 1.0 - np.asarray(proba)[:, 0]
    failing = (np.asarray(y_true) > 0).astype(float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (risk >= lo) & (risk < hi)
        if mask.any():
            rows.append({"desde": float(lo), "hasta": float(min(hi, 1.0)), "n": int(mask.sum()),
                         "predicho": float(risk[mask].mean()), "observado": float(failing[mask].mean())})
    return rows


def calibration_error(y_true, proba: np.ndarray) -> dict:
    """Riesgo medio predicho contra tasa observada, sobre todos los camiones."""
    risk = 1.0 - np.asarray(proba)[:, 0]
    return {"riesgo_medio_predicho": float(risk.mean()), "tasa_observada": float((np.asarray(y_true) > 0).mean())}


# ------------------------------------------------------------------ Platt (dos parametros)
def _logit_risk(proba: np.ndarray) -> np.ndarray:
    risk = np.clip(1.0 - np.asarray(proba, dtype=float)[:, 0], 1e-6, 1.0 - 1e-6)
    return np.log(risk / (1.0 - risk))


def fit_platt(proba: np.ndarray, y_class: np.ndarray) -> dict:
    """Escalado de Platt de ``logit(1 - p0)``: ``riesgo = sigmoide(a * logit + b)``.

    Tiene dos parametros, asi que es estable con las ~140 fallas de un conjunto. Una pendiente ``a`` muy
    por debajo de 1 quiere decir que las probabilidades originales exageran el riesgo.
    """
    from sklearn.linear_model import LogisticRegression

    z = _logit_risk(proba).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, (np.asarray(y_class) > 0).astype(int))
    return {"a": round(float(lr.coef_[0][0]), 6), "b": round(float(lr.intercept_[0]), 6)}


def apply_platt(proba: np.ndarray, a: float, b: float) -> np.ndarray:
    """Probabilidades por clase con el riesgo recalibrado; cada fila sigue sumando 1."""
    proba = np.asarray(proba, dtype=float)
    risk = 1.0 - proba[:, 0]
    calibrated = 1.0 / (1.0 + np.exp(-(a * _logit_risk(proba) + b)))
    scale = np.where(risk > EPS, calibrated / np.maximum(risk, EPS), 0.0)
    out = proba.copy()
    out[:, 1:] = proba[:, 1:] * scale[:, None]
    out[:, 0] = 1.0 - calibrated
    return out
