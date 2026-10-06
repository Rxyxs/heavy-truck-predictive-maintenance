"""Evalua LightGBM y Random Forest en los cuatro subconjuntos de C-MAPSS y exporta ``tmp_agent_b/rul_metrics.json``.

    python -m src.models.rul_benchmark [--data-dir DIR]

La eleccion entre modelos se hace por validacion cruzada agrupada por motor sobre el entrenamiento; el conjunto de prueba
solo se mira una vez, con ambos modelos, para informar (no para elegir).

FD001 se carga con ``src.data.cmapss_loader.build_fd001`` (descarga verificada con hash). Ese cargador no cubre
FD002-FD004, que se leen con ``_read_raw`` desde ``data/raw/cmapss/raw`` (descarga manual del mismo zip de la NASA).
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
from src.models.rul_estimator import COLUMNS, RulEstimator, last_rows, nasa_score, rmse, train_rul

ROOT = Path(__file__).resolve().parents[2]
RAW_OTHER = ROOT / "data" / "raw" / "cmapss" / "raw"  # FD002-FD004
OUT_DIR = ROOT / "tmp_agent_b"
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
MODELS = ["lightgbm", "random_forest"]
CAP = 125
N_FOLDS = 5


def _load_fd001() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """FD001 con el cargador del proyecto; la RUL verdadera del ultimo ciclo es ``rul_uncapped`` de ``test_last``."""
    ds = cmapss_loader.build_fd001()
    rename = {f"setting_{i}": f"op{i}" for i in (1, 2, 3)}
    cols = ["unit", "cycle", *rename, *ds.sensor_columns]
    train, test = ds.train.select(cols).rename(rename).to_pandas(), ds.test.select(cols).rename(rename).to_pandas()
    rul = ds.test_last.sort("unit")["rul_uncapped"].to_numpy().astype(float)
    return train, test, rul


def read_split(data_dir: Path | None, subset: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """(entrenamiento, prueba, RUL verdadera al final de cada motor de prueba)."""
    if subset == "FD001":
        return _load_fd001()
    data_dir = data_dir or RAW_OTHER
    train = pd.read_csv(data_dir / f"train_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    test = pd.read_csv(data_dir / f"test_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    rul = pd.read_csv(data_dir / f"RUL_{subset}.txt", header=None)[0].to_numpy(dtype=float)
    return train, test, rul


def _test_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    d = y_pred - y_true
    return {"rmse": rmse(y_true, y_pred), "nasa_score": nasa_score(y_true, y_pred),
            "share_overestimated": float(np.mean(d > 0)), "mean_error": float(np.mean(d))}


def cross_validate(train: pd.DataFrame, kind: str) -> dict:
    """RMSE sobre todas las filas de motores no vistos (RUL verdadera recortada a CAP, como en el entrenamiento)."""
    units = train["unit"].to_numpy()
    errors = []
    for fit_idx, val_idx in GroupKFold(N_FOLDS).split(train, groups=units):
        model = RulEstimator(kind, cap=CAP).fit(train.iloc[fit_idx])
        val = train.iloc[val_idx]
        truth = train_rul(train, CAP).iloc[val_idx]
        truth.index = pd.MultiIndex.from_arrays([val["unit"], val["cycle"]])
        pred = model.predict_rows(val)
        errors.append((pred - truth.reindex(pred.index)).to_numpy())
    e = np.concatenate(errors)
    return {"cv_rmse_rows": float(np.sqrt(np.mean(e**2))), "n_folds": N_FOLDS}


def run_subset(data_dir: Path, subset: str) -> dict:
    train, test, rul_true = read_split(data_dir, subset)
    test_last = last_rows(test)
    assert len(test_last) == len(rul_true)
    baseline = float(train_rul(train, CAP).mean())
    out = {"n_train_units": int(train["unit"].nunique()), "n_test_units": int(len(rul_true)),
           "baseline_constant": {"value": baseline, **_test_metrics(rul_true, np.full(len(rul_true), baseline))},
           "models": {}}
    for kind in MODELS:
        t0 = time.time()
        cv = cross_validate(train, kind)
        est = RulEstimator(kind, cap=CAP).fit(train)
        pred = est.predict_last(test).reindex(test_last["unit"]).to_numpy()
        out["models"][kind] = {**cv, "test": _test_metrics(rul_true, pred), "seconds": time.time() - t0}
        print(f"{subset} {kind}: cv {cv['cv_rmse_rows']:.2f}, test rmse {out['models'][kind]['test']['rmse']:.2f}", flush=True)
    out["selected_by_cv"] = min(MODELS, key=lambda k: out["models"][k]["cv_rmse_rows"])
    out["selected_test"] = out["models"][out["selected_by_cv"]]["test"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None, help="solo FD002-FD004")
    parser.add_argument("--subsets", nargs="*", default=SUBSETS)
    args = parser.parse_args()
    result = {"config": {"rul_cap_training": CAP, "window": 10, "cv_folds": N_FOLDS, "models": MODELS,
                         "test_protocol": "RUL at the last observed cycle of each test engine vs. RUL_FD00x.txt"},
              "subsets": {s: run_subset(args.data_dir, s) for s in args.subsets}}
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / "rul_metrics.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("escrito", path)


if __name__ == "__main__":
    main()
