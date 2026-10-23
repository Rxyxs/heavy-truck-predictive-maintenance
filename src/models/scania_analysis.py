"""Analisis adicionales de SCANIA Component X, guardados en ``results.json`` para que las figuras no necesiten datos.

* ``kaplan_meier``: supervivencia del componente en toda la flota de entrenamiento y por la especificacion que
  mas la separa (prueba log-rank sobre las 8 especificaciones).
* ``lead_time``: cuanto riesgo asigna el modelo a medida que se acerca la reparacion, sobre los camiones retenidos
  del entrenamiento (todas sus lecturas). Dice con cuanta anticipacion se ve una falla.
* ``curves``: curvas ROC y precision-recall de "algun riesgo" para cada modelo, en validacion y test.
* ``cost_breakdown``: de donde sale el costo de cada modelo (falsas alarmas, fallas no vistas, ventana equivocada).
* ``paired_bootstrap``: diferencia de ahorro de cada modelo contra el elegido, remuestreando los mismos camiones.
* ``cut_timing``: en que punto de la vida del camion cae la lectura evaluada en cada conjunto.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.data import scania_loader as S
from src.models import cost_decision as C

SPEC_COLUMNS = [f"Spec_{i}" for i in range(8)]
LEAD_BINS = [0, 6, 12, 24, 48, 96, 192, 1e9]
N_POINTS = 60
SEED = 42


def _downsample(x: np.ndarray, y: np.ndarray, n: int = N_POINTS) -> dict:
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).round().astype(int))
    return {"x": np.round(x[idx], 4).tolist(), "y": np.round(y[idx], 4).tolist()}


def kaplan_meier(tte: pl.DataFrame, specs: pl.DataFrame, min_count: int = 300, top: int = 4) -> dict:
    df = tte.join(specs, on="vehicle_id", how="inner").to_pandas()
    km = KaplanMeierFitter().fit(df["duration"], df["event"])
    grid = np.linspace(0, float(df["duration"].max()), 80)
    out = {"overall": {"t": np.round(grid, 1).tolist(), "s": np.round(km.survival_function_at_times(grid).to_numpy(), 4).tolist()},
           "trucks": int(len(df)), "events": int(df["event"].sum()), "logrank": {}}
    for c in SPEC_COLUMNS:
        counts = df[c].value_counts()
        group = df[c].where(df[c].isin(counts[counts >= min_count].index), "rare")
        test = multivariate_logrank_test(df["duration"], group, df["event"])
        out["logrank"][c] = {"statistic": round(float(test.test_statistic), 1), "p_value": float(test.p_value),
                             "groups": int(group.nunique())}
    best = max(out["logrank"], key=lambda c: out["logrank"][c]["statistic"])
    out["by_spec"] = {"spec": best, "curves": []}
    for cat in df[best].value_counts().index[:top]:
        part = df[df[best] == cat]
        k = KaplanMeierFitter().fit(part["duration"], part["event"])
        out["by_spec"]["curves"].append({"category": str(cat), "trucks": int(len(part)), "events": int(part["event"].sum()),
                                         "s": np.round(k.survival_function_at_times(grid).to_numpy(), 4).tolist()})
    return out


def lead_time(frame: pd.DataFrame, probas: dict[str, np.ndarray]) -> dict:
    """Riesgo medio predicho y fraccion de lecturas con alarma, por tiempo restante hasta la reparacion."""
    ttf = frame["time_to_failure"].to_numpy(dtype=float)
    failed = ~np.isnan(ttf)
    bins = pd.cut(pd.Series(ttf[failed]), LEAD_BINS, right=False)
    labels = [f"{int(a)}-{int(b)}" if b < 1e8 else f"{int(a)}+" for a, b in zip(LEAD_BINS[:-1], LEAD_BINS[1:])]
    out = {"bins": labels, "readouts": bins.value_counts(sort=False).tolist(),
           "trucks": [int(frame.loc[failed, "vehicle_id"][(bins == b).to_numpy()].nunique()) for b in bins.cat.categories],
           "models": {}}
    healthy = (frame["event"].to_numpy() == 0)
    for name, p in probas.items():
        risk = 1.0 - p[:, 0]
        alarm = C.bayes_decision(p) > 0
        g_risk = pd.Series(risk[failed]).groupby(bins.to_numpy(), observed=False).mean()
        g_alarm = pd.Series(alarm[failed].astype(float)).groupby(bins.to_numpy(), observed=False).mean()
        out["models"][name] = {"mean_risk": np.round(g_risk.to_numpy(), 4).tolist(),
                               "alarm_share": np.round(g_alarm.to_numpy(), 4).tolist(),
                               "healthy_mean_risk": round(float(risk[healthy].mean()), 4),
                               "healthy_alarm_share": round(float(alarm[healthy].mean()), 4)}
    return out


def curves(y: np.ndarray, probas: dict[str, np.ndarray]) -> dict:
    out = {}
    target = (y > 0).astype(int)
    for name, p in probas.items():
        score = 1.0 - p[:, 0]
        fpr, tpr, _ = roc_curve(target, score)
        prec, rec, _ = precision_recall_curve(target, score)
        out[name] = {"roc": _downsample(fpr, tpr), "pr": _downsample(rec[::-1], prec[::-1]),
                     "auc": round(float(roc_auc_score(target, score)), 4),
                     "ap": round(float(average_precision_score(target, score)), 4)}
    out["base_rate"] = round(float(target.mean()), 4)
    return out


def cost_breakdown(y: np.ndarray, p: np.ndarray) -> dict:
    cost = np.asarray(S.COST_MATRIX)
    pred = C.bayes_decision(p)
    c = cost[y, pred]
    healthy, failing = y == 0, y > 0
    return {"false_alarms": int(c[healthy].sum()), "missed": int(c[failing & (pred == 0)].sum()),
            "wrong_window": int(c[failing & (pred > 0)].sum()), "total": int(c.sum())}


def paired_bootstrap(y: np.ndarray, probas: dict[str, np.ndarray], reference: str, n: int = 2000, seed: int = SEED) -> dict:
    """Ahorro de cada modelo menos el del elegido, con los mismos camiones remuestreados en cada replica."""
    cost = np.asarray(S.COST_MATRIX)
    per_truck = {k: cost[y, C.bayes_decision(p)] for k, p in probas.items()}
    nothing = cost[y, 0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n, len(y)))
    denom = nothing[idx].sum(axis=1)
    savings = {k: 100.0 * (1.0 - v[idx].sum(axis=1) / denom) for k, v in per_truck.items()}
    out = {}
    for k in probas:
        if k == reference:
            continue
        d = savings[k] - savings[reference]
        out[k] = {"mean_diff": round(float(d.mean()), 2), "ci95": [round(float(v), 2) for v in np.percentile(d, [2.5, 97.5])],
                  "p_better": round(float((d > 0).mean()), 3)}
    return out


def cut_timing(train_rand: pl.DataFrame, others: dict[str, pl.DataFrame]) -> dict:
    edges = np.arange(0, 520, 20)
    out = {"edges": edges.tolist()}
    for name, frame in {"train_random": train_rand, **others}.items():
        t = frame["time_step"].to_numpy()
        hist, _ = np.histogram(np.clip(t, 0, edges[-1] - 1e-6), edges)
        out[name] = {"share": np.round(hist / hist.sum(), 4).tolist(), "median": round(float(np.median(t)), 1)}
    return out
