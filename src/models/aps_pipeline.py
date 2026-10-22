"""APS Failure at Scania Trucks: seis familias de modelos y un ensamble bajo la metrica de costo del reto.

Protocolo (fijado antes de mirar el test):

1. Validacion cruzada estratificada de 5 particiones sobre los 60.000 camiones de train. En cada particion,
   los modelos con parada temprana (boosting y la red) separan un 15% estratificado de su parte de
   entrenamiento para decidir cuando parar; las predicciones fuera de particion (OOF) nunca vieron ese camion.
2. El umbral de alarma de cada modelo es el que minimiza el costo 10 / 500 sobre sus predicciones OOF.
3. La prediccion de test es el promedio de los 5 modelos de particion, con el umbral del paso 2.
   El modelo "elegido" es el de menor costo OOF; el test solo se mira para informar.
4. Intervalo al 95% del costo de test por bootstrap de camiones.

Se informa tambien, solo como referencia de optimismo y marcado como no valido, el costo que tendria cada
modelo si el umbral se ajustara mirando el test: es lo que hacen muchos trabajos publicados con este dataset.

    python -m src.models.aps_pipeline
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data import aps_loader as A
from src.models import evaluation as V
from src.models.zoo import MODEL_NAMES, TabularModel, class_balanced_weights

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "aps"
SCORES_PATH = ROOT / "data" / "processed" / "aps_scores.parquet"
CHECKPOINTS = ROOT / "data" / "processed" / "models" / "aps"
SEED = 42
N_FOLDS = 5
ES_FRACTION = 0.15
ENSEMBLE = ["lightgbm", "xgboost", "catboost"]  # promedio simple de probabilidades, fijado de antemano
USES_EARLY_STOPPING = {"lightgbm", "xgboost", "catboost", "mlp"}


def cross_fit(name: str, X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, n_folds: int = N_FOLDS):
    """Predicciones OOF sobre train y promedio de los modelos de particion sobre test."""
    w = class_balanced_weights(y)
    oof = np.zeros(len(y))
    test = np.zeros(len(X_test))
    infos = []
    for fold, (fit_idx, oof_idx) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=SEED).split(X, y)):
        model = TabularModel(name, 2, seed=SEED + fold, checkpoint_dir=CHECKPOINTS / f"fold{fold}")
        if name in USES_EARLY_STOPPING:
            fit_in, es = train_test_split(fit_idx, test_size=ES_FRACTION, stratify=y[fit_idx], random_state=SEED + fold)
            model.fit(X.iloc[fit_in], y[fit_in], w[fit_in], X.iloc[es], y[es], w[es])
        else:
            model.fit(X.iloc[fit_idx], y[fit_idx], w[fit_idx])
        oof[oof_idx] = model.predict_proba(X.iloc[oof_idx])[:, 1]
        test += model.predict_proba(X_test)[:, 1] / n_folds
        infos.append(model.info)
    return oof, test, infos


def summarize(name: str, y_tr, oof, y_te, test, seconds: float | None = None, infos=None) -> dict:
    thr, oof_cost = V.best_threshold(y_tr, oof, A.COST_FALSE_ALARM, A.COST_MISSED)
    pred = (test >= thr).astype(int)
    fa = int(((pred == 1) & (y_te == 0)).sum())
    missed = int(((pred == 0) & (y_te == 1)).sum())
    _, oracle = V.best_threshold(y_te, test, A.COST_FALSE_ALARM, A.COST_MISSED)
    ci = V.bootstrap(lambda yt, s: A.total_cost(yt, s >= thr), y_te, test, n=2000, seed=SEED)
    out = {
        "threshold_from_oof": thr,
        "oof": {"cost": int(oof_cost), "roc_auc": float(roc_auc_score(y_tr, oof)),
                "pr_auc": float(average_precision_score(y_tr, oof))},
        "test": {"cost": A.total_cost(y_te, pred), "cost_ci95": [int(round(v)) for v in ci["ci95"]],
                 "false_alarms": fa, "missed": missed, "recall": float(1 - missed / y_te.sum()),
                 "roc_auc": float(roc_auc_score(y_te, test)), "pr_auc": float(average_precision_score(y_te, test))},
        "not_valid_threshold_tuned_on_test": {"cost": int(oracle)},
    }
    if seconds is not None:
        out["seconds"] = round(seconds, 1)
    if infos:
        out["fold_info"] = infos
    return out


def run(models: list[str] | None = None) -> dict:
    models = models or MODEL_NAMES
    A.download()
    X_tr_raw, y_tr = A.read_split("train")
    X_te_raw, y_te = A.read_split("test")
    for split, (X, y) in {"train": (X_tr_raw, y_tr), "test": (X_te_raw, y_te)}.items():
        assert (len(X), int(y.sum())) == A.EXPECTED[split], f"{split}: {len(X)} filas, {y.sum()} positivos"
    X_tr, X_te = A.engineer(X_tr_raw), A.engineer(X_te_raw)

    results = {
        "data": {"train_trucks": len(y_tr), "train_failures": int(y_tr.sum()), "test_trucks": len(y_te),
                 "test_failures": int(y_te.sum()), "features": X_tr.shape[1],
                 "missing_share": round(float(X_tr_raw.isna().to_numpy().mean()), 4)},
        "protocol": {"cv_folds": N_FOLDS, "early_stopping_fraction": ES_FRACTION, "cost_false_alarm": A.COST_FALSE_ALARM,
                     "cost_missed": A.COST_MISSED, "threshold": "minimum cost on out-of-fold train predictions",
                     "selection": "lowest out-of-fold cost", "ensemble": ENSEMBLE},
        "baselines": {"no_alarms": A.total_cost(y_te, np.zeros_like(y_te)), "alarm_everyone": A.total_cost(y_te, np.ones_like(y_te))},
        "ida2016_podium": A.IDA2016_PODIUM,
        "models": {},
    }
    scores = {"y_train": y_tr}
    test_scores = {"y_test": y_te}
    for name in models:
        t0 = time.time()
        oof, test, infos = cross_fit(name, X_tr, y_tr, X_te)
        results["models"][name] = summarize(name, y_tr, oof, y_te, test, time.time() - t0, infos)
        scores[name], test_scores[name] = oof, test
        r = results["models"][name]
        print(f"[APS] {name:14s} OOF {r['oof']['cost']:>7,}  test {r['test']['cost']:>7,} {r['test']['cost_ci95']}  "
              f"FA {r['test']['false_alarms']:4d} missed {r['test']['missed']:3d}  ({r['seconds']:.0f}s)", flush=True)

    if all(m in scores for m in ENSEMBLE):
        oof = np.mean([scores[m] for m in ENSEMBLE], axis=0)
        test = np.mean([test_scores[m] for m in ENSEMBLE], axis=0)
        results["models"]["ensemble"] = summarize("ensemble", y_tr, oof, y_te, test)
        scores["ensemble"], test_scores["ensemble"] = oof, test
        r = results["models"]["ensemble"]
        print(f"[APS] ensemble       OOF {r['oof']['cost']:>7,}  test {r['test']['cost']:>7,} {r['test']['cost_ci95']}")

    selected = min(results["models"], key=lambda m: results["models"][m]["oof"]["cost"])
    results["selected_by_oof"] = selected
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_parquet(SCORES_PATH.with_name("aps_oof_scores.parquet"))
    pd.DataFrame(test_scores).to_parquet(SCORES_PATH)
    print(f"[APS] elegido por OOF: {selected} -> test {results['models'][selected]['test']['cost']:,}")
    return results


if __name__ == "__main__":
    import sys

    run(sys.argv[1:] or None)
