"""Vida util restante en los cuatro subconjuntos de NASA C-MAPSS: seis modelos y un ensamble.

    python -m src.models.rul_benchmark [--subsets FD001 FD002] [--models ridge lightgbm cnn]

Protocolo (fijado antes de mirar el test):

* Validacion cruzada de 5 particiones agrupada por motor sobre el entrenamiento: RMSE sobre todas las filas de los
  motores no vistos, contra la RUL recortada a 125 (la misma etiqueta del entrenamiento).
* Las redes separan ademas un 15% de los motores de su parte de entrenamiento para la parada temprana.
* Ensamble: promedio de los tres modelos con menor RMSE de validacion cruzada (su RMSE de CV sale del promedio
  de las predicciones fuera de particion, asi que tambien se elige sin mirar el test).
* El elegido de cada subconjunto es el de menor RMSE de CV. El test (ultimo ciclo de cada motor contra
  ``RUL_FD00x.txt``, sin recorte) se mira una vez, con todos los modelos, para informar.

Escribe ``reports/cmapss/results.json`` y ``reports/cmapss/test_predictions.csv``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data import cmapss_loader
from src.models.rul_estimator import (SEQUENCE_KINDS, TABULAR_KINDS, RulEstimator, last_rows, make_estimator,
                                      nasa_score, rmse, train_rul)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "cmapss"
SUBSETS = cmapss_loader.SUBSETS
MODELS = TABULAR_KINDS + SEQUENCE_KINDS
CAP = 125
N_FOLDS = 5
ENSEMBLE_SIZE = 3


def read_split(data_dir: Path | None, subset: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """(entrenamiento, prueba, RUL verdadera al final de cada motor de prueba), via el cargador del proyecto."""
    return cmapss_loader.read_subset(subset, data_dir or cmapss_loader.RAW_DIR)


def _test_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    d = y_pred - y_true
    return {"rmse": rmse(y_true, y_pred), "nasa_score": nasa_score(y_true, y_pred),
            "share_overestimated": float(np.mean(d > 0)), "mean_error": float(np.mean(d))}


def cross_validate(train: pd.DataFrame, kind: str) -> tuple[dict, pd.Series]:
    """RMSE fuera de particion sobre todas las filas de motores no vistos; devuelve tambien las predicciones."""
    train = train.sort_values(["unit", "cycle"]).reset_index(drop=True)
    truth = pd.Series(train_rul(train, CAP).to_numpy(), index=pd.MultiIndex.from_arrays([train["unit"], train["cycle"]]))
    oof = []
    for fold, (fit_idx, val_idx) in enumerate(GroupKFold(N_FOLDS).split(train, groups=train["unit"])):
        extra = {"seed": fold} if kind in SEQUENCE_KINDS else {}
        model = make_estimator(kind, cap=CAP, **extra).fit(train.iloc[fit_idx])
        oof.append(model.predict_rows(train.iloc[val_idx]))
    pred = pd.concat(oof).reindex(truth.index)
    return {"cv_rmse_rows": rmse(truth.to_numpy(), pred.to_numpy()), "n_folds": N_FOLDS}, pred


def sensor_trends(train: pd.DataFrame, top: int = 6, step: int = 10, max_rul: int = 250) -> dict:
    """Media de cada sensor (normalizado por condicion de operacion) segun los ciclos que faltan para la falla.

    Muestra por que se recorta la RUL: al inicio de la vida los sensores no cambian, y la degradacion aparece
    en los ultimos ~100-150 ciclos.
    """
    est = RulEstimator("ridge")
    est._fit_scaler(train)
    df = train.sort_values(["unit", "cycle"]).reset_index(drop=True)
    cond = pd.Series(est._condition(df))
    z = (df[est.sensors_] - est.mean_.loc[cond].to_numpy()) / est.std_.loc[cond].to_numpy()
    rul = train_rul(df, None).to_numpy()
    corr = z.apply(lambda col: np.corrcoef(col, rul)[0, 1]).abs().sort_values(ascending=False)
    chosen = list(corr.index[:top])
    bins = (np.minimum(rul, max_rul) // step) * step
    means = z[chosen].groupby(bins).mean()
    return {"rul": means.index.astype(int).tolist(), "sensors": {c: np.round(means[c].to_numpy(), 3).tolist() for c in chosen},
            "abs_corr_with_rul": {c: round(float(corr[c]), 3) for c in chosen}}


def run_subset(data_dir: Path | None, subset: str, models: list[str]) -> tuple[dict, pd.DataFrame]:
    train, test, rul_true = read_split(data_dir, subset)
    test_last = last_rows(test)
    assert len(test_last) == len(rul_true)
    train_sorted = train.sort_values(["unit", "cycle"]).reset_index(drop=True)
    truth_cv = train_rul(train_sorted, CAP).to_numpy()
    baseline = float(train_rul(train, CAP).mean())
    out = {"n_train_units": int(train["unit"].nunique()), "n_test_units": int(len(rul_true)),
           **cmapss_loader.SUBSET_INFO[subset], "sensor_trends": sensor_trends(train),
           "baseline_constant": {"value": baseline, **_test_metrics(rul_true, np.full(len(rul_true), baseline))},
           "models": {}}
    oof, test_pred = {}, {}
    for kind in models:
        t0 = time.time()
        cv, oof[kind] = cross_validate(train, kind)
        est = make_estimator(kind, cap=CAP).fit(train)
        test_pred[kind] = est.predict_last(test).reindex(test_last["unit"]).to_numpy()
        out["models"][kind] = {**cv, "test": _test_metrics(rul_true, test_pred[kind]), "seconds": round(time.time() - t0, 1)}
        if getattr(est, "info", None):
            out["models"][kind]["final_fit"] = est.info
        if getattr(est, "history_", None) is not None:
            out["models"][kind]["history"] = {"train_loss": [round(v, 5) for v in est.history_.train_loss],
                                              "val_loss": [round(v, 5) for v in est.history_.val_loss]}
        print(f"{subset} {kind}: cv {cv['cv_rmse_rows']:.2f}, test rmse {out['models'][kind]['test']['rmse']:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    ranked = sorted(models, key=lambda k: out["models"][k]["cv_rmse_rows"])
    if len(ranked) >= ENSEMBLE_SIZE:
        members = ranked[:ENSEMBLE_SIZE]
        ens_oof = np.mean([oof[k].to_numpy() for k in members], axis=0)
        test_pred["ensemble"] = np.mean([test_pred[k] for k in members], axis=0)
        out["models"]["ensemble"] = {"members": members, "cv_rmse_rows": rmse(truth_cv, ens_oof), "n_folds": N_FOLDS,
                                     "test": _test_metrics(rul_true, test_pred["ensemble"])}
    out["selected_by_cv"] = min(out["models"], key=lambda k: out["models"][k]["cv_rmse_rows"])
    out["selected_test"] = out["models"][out["selected_by_cv"]]["test"]
    preds = pd.DataFrame({"subset": subset, "unit": test_last["unit"].to_numpy(), "rul_true": rul_true,
                          **{k: np.round(v, 2) for k, v in test_pred.items()}})
    return out, preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--subsets", nargs="*", default=SUBSETS)
    parser.add_argument("--models", nargs="*", default=MODELS)
    args = parser.parse_args()
    result = {"config": {"rul_cap_training": CAP, "window_features": 10, "window_sequence": 30, "cv_folds": N_FOLDS,
                         "models": args.models, "ensemble": f"mean of the {ENSEMBLE_SIZE} lowest-CV models",
                         "test_protocol": "RUL at the last observed cycle of each test engine vs. RUL_FD00x.txt"},
              "subsets": {}}
    frames = []
    for subset in args.subsets:
        result["subsets"][subset], preds = run_subset(args.data_dir, subset, args.models)
        frames.append(preds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    pd.concat(frames, ignore_index=True).to_csv(OUT_DIR / "test_predictions.csv", index=False)
    print("escrito", OUT_DIR / "results.json")


if __name__ == "__main__":
    main()
