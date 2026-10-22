"""Utilidades de evaluacion compartidas: umbral de costo minimo e intervalos por bootstrap."""
from __future__ import annotations

from typing import Callable

import numpy as np


def cost_curve(y_true, score, cost_fp: float, cost_fn: float) -> tuple[np.ndarray, np.ndarray]:
    """Costo total de alarmar a todo camion con ``score >= t`` para cada umbral candidato ``t``.

    Los candidatos son los valores unicos de ``score`` mas ``+inf`` (no alarmar a nadie). Devuelve
    ``(umbrales, costos)`` con los umbrales en orden decreciente.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    # Posicion del ultimo elemento de cada bloque de puntajes empatados: el umbral alarma al bloque completo.
    last_of_block = np.r_[np.nonzero(np.diff(s_sorted))[0], len(s_sorted) - 1]
    tp = np.cumsum(y_sorted)[last_of_block]
    fp = np.cumsum(1 - y_sorted)[last_of_block]
    fn = y.sum() - tp
    thresholds = np.r_[np.inf, s_sorted[last_of_block]]
    costs = np.r_[cost_fn * y.sum(), cost_fp * fp + cost_fn * fn]
    return thresholds, costs


def best_threshold(y_true, score, cost_fp: float, cost_fn: float) -> tuple[float, float]:
    """Umbral que minimiza el costo total (en caso de empate, el mas alto: menos alarmas)."""
    thresholds, costs = cost_curve(y_true, score, cost_fp, cost_fn)
    i = int(np.argmin(costs))
    return float(thresholds[i]), float(costs[i])


def bootstrap(stat: Callable[..., float], *arrays, n: int = 2000, seed: int = 0, groups=None) -> dict:
    """Intervalo percentil al 95% de ``stat(*arrays)`` remuestreando filas (o grupos enteros si se pasa ``groups``)."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    point = float(stat(*arrays))
    if groups is None:
        m = len(arrays[0])
        draws = [rng.integers(0, m, m) for _ in range(n)]
    else:
        groups = np.asarray(groups)
        uniq, inverse = np.unique(groups, return_inverse=True)
        members = [np.nonzero(inverse == g)[0] for g in range(len(uniq))]
        draws = [np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))]) for _ in range(n)]
    values = np.array([stat(*[a[idx] for a in arrays]) for idx in draws], dtype=float)
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {"point": point, "ci95": [float(lo), float(hi)], "n_boot": n}
