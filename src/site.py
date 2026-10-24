"""Genera la pagina de GitHub Pages (``docs/index.html``) a partir de los resultados versionados.

La pagina recalcula en el navegador la decision de costo esperado minimo de SCANIA Component X con el costo de
una visita a taller que elija el lector, para el mismo LightGBM con dos conjuntos de features: invariantes a la
escala y en escala cruda (la version anterior). Incrusta sus probabilidades por camion
(``reports/scania/predictions.csv``); todo lo demas sale de los ``results.json``, asi que no puede
desincronizarse del README.

    python -m src.site
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.scania_loader import COST_MATRIX
from src.models import calibration as K

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("site_template.html")
REPORTS = ROOT / "reports"
RESULTS_JSON = REPORTS / "scania" / "results.json"
PREDICTIONS_CSV = REPORTS / "scania" / "predictions.csv"
FIGURES_DIR = REPORTS / "figures"
DOCS_DIR = ROOT / "docs"
LANGS = ("en", "es")
FIGURES_USED = ["overview.png", "scania_scale_shift.png", "scania_models.png", "scania_confusion.png", "scania_shap.png",
                "scania_lead_time.png", "scania_seed_spread.png", "aps_models.png", "aps_threshold.png", "aps_missingness.png",
                "cmapss_models.png", "cmapss_error_by_rul.png", "metropt_failures.png", "metropt_operating.png"]
PROB_SCALE = 1_000_000  # las probabilidades viajan como enteros: son mas cortas que los decimales
SPLITS = ("validation", "test")
MODEL_ORDER = ["logistic", "random_forest", "lightgbm", "xgboost", "catboost", "mlp", "gru", "ensemble", "lightgbm_escala_cruda"]


def _calibration_rows(rows: list[dict]) -> list[dict]:
    return [{"desde": r["desde"], "hasta": r["hasta"], "n": r["n"], "predicho": round(r["predicho"], 4),
             "observado": round(r["observado"], 4)} for r in rows]


def _round(v, d=2):
    return None if v is None else round(float(v), d)


def _other_datasets() -> dict:
    out = {}
    path = REPORTS / "aps" / "results.json"
    if path.exists():
        r = json.loads(path.read_text(encoding="utf-8"))
        sel = r["selected_by_oof"]
        out["aps"] = {"model": sel, "cost": r["models"][sel]["test"]["cost"], "ci": r["models"][sel]["test"]["cost_ci95"],
                      "winner": r["ida2016_podium"][0]["cost"], "nothing": r["baselines"]["no_alarms"],
                      "trucks": r["data"]["train_trucks"] + r["data"]["test_trucks"],
                      "tuned": r["models"][sel]["not_valid_threshold_tuned_on_test"]["cost"],
                      "missing": r.get("analysis", {}).get("missingness"),
                      "models": [{"key": k, "oof": m["oof"]["cost"], "cost": m["test"]["cost"], "ci": m["test"]["cost_ci95"],
                                  "fa": m["test"]["false_alarms"], "missed": m["test"]["missed"]} for k, m in r["models"].items()]}
    path = REPORTS / "cmapss" / "results.json"
    if path.exists():
        r = json.loads(path.read_text(encoding="utf-8"))
        out["cmapss"] = {s: {"model": d["selected_by_cv"], "rmse": round(d["selected_test"]["rmse"], 1),
                             "baseline": round(d["baseline_constant"]["rmse"], 1),
                             "conditions": d["conditions"], "modes": d["fault_modes"],
                             "all": {k: round(m["test"]["rmse"], 1) for k, m in d["models"].items()}}
                         for s, d in r["subsets"].items()}
    path = REPORTS / "metropt" / "results.json"
    if path.exists():
        r = json.loads(path.read_text(encoding="utf-8"))
        ae = r["detectors"].get("autoencoder__absolute")
        if ae:
            out["metropt"] = {"detected": ae["detected_2h"], "chance": ae["chance_detected_2h"],
                              "fa_week": ae["false_alarms_per_week"],
                              "detectors": [{"key": k, "detector": d["detector"], "view": d["view"], "detected": d["detected_2h"],
                                             "chance": d["chance_detected_2h"], "fa": d["false_alarms_per_week"]}
                                            for k, d in r["detectors"].items()],
                              "seeds": r.get("autoencoder_seeds", [])}
    return out


def build_data(results: dict, predictions: pd.DataFrame) -> dict:
    clf = results["clasificador"]
    selected = results["seleccionado"]
    models = results["modelos"]
    raw_key, lgb_key = "lightgbm_escala_cruda", "lightgbm"
    ttf = results["tiempo_restante"]

    pred, calibration = {}, {}
    for split in SPLITS:
        part = predictions[predictions["split"] == split].reset_index(drop=True)
        if len(part) != clf[split]["n_vehiculos"]:
            raise ValueError(f"{split}: {len(part)} filas en predictions.csv, pero results.json dice {clf[split]['n_vehiculos']}")
        y = part["class_label"].astype(int).to_numpy()
        pred[split] = {"y": y.tolist()}
        for field in ("l", "r"):
            probs = part[[f"{field}{k}" for k in range(5)]].to_numpy()
            pred[split][field] = [int(round(v * PROB_SCALE)) for v in probs.ravel()]
        calibration[split] = {mode: _calibration_rows(K.calibration_table(y, part[[f"{field}{k}" for k in range(5)]].to_numpy()))
                              for mode, field in (("lgb", "l"), ("raw", "r"))}

    table = []
    for key in [k for k in MODEL_ORDER if k in models]:
        m = models[key]
        table.append({"key": key, "select": m["seleccion"]["costo_por_1000_camiones"],
                      **{f"{s[:3]}_{f}": m[s][f] for s in SPLITS for f in ("ahorro_pct", "ahorro_ic95", "riesgo_medio_predicho")},
                      "val_be": _round(m["validation"]["costo_visita_equilibrio"]), "tes_be": _round(m["test"]["costo_visita_equilibrio"])})

    adv = results["escala"]["validacion_adversarial"]
    factors = [f for f in results["escala"]["factor_por_variable"] if f["validation"] and f["test"]]
    surv = results["supervivencia"]["modelos_covariables_ambas"]
    return {
        "cost_matrix": COST_MATRIX,
        "selected": selected,
        "pred": pred,
        "calibration": calibration,
        "models": table,
        "scale": {
            "adv_raw": {k: v["auc"] for k, v in adv["todas"].items()},
            "adv_inv": {k: v["auc"] for k, v in adv["invariantes"].items()},
            "min": min(min(f["validation"], f["test"]) for f in factors),
            "max": max(max(f["validation"], f["test"]) for f in factors),
            "n_vars": len(factors),
        },
        "key": {
            "trucks_train": results["datos"]["vehiculos_train"], "fails_train": results["datos"]["reparaciones_train"],
            "n_val": clf["validation"]["n_vehiculos"], "n_test": clf["test"]["n_vehiculos"],
            "fails_val": clf["validation"]["costo_esperado_minimo"]["fallas_reales"],
            "fails_test": clf["test"]["costo_esperado_minimo"]["fallas_reales"],
            "auc_test": round(clf["test"]["auc_algun_riesgo"], 2),
            **{f"{s[:3]}_{k}": v for s in SPLITS for k, v in {
                "saving": models[selected][s]["ahorro_pct"], "ci": models[selected][s]["ahorro_ic95"],
                "be": _round(models[selected][s]["costo_visita_equilibrio"]), "be_raw": _round(models[raw_key][s]["costo_visita_equilibrio"]),
                "be_lgb": _round(models[lgb_key][s]["costo_visita_equilibrio"]),
                "saving_raw": models[raw_key][s]["ahorro_pct"], "saving_lgb": models[lgb_key][s]["ahorro_pct"],
                "risk": models[selected][s]["riesgo_medio_predicho"], "risk_raw": models[raw_key][s]["riesgo_medio_predicho"],
                "risk_lgb": models[lgb_key][s]["riesgo_medio_predicho"],
                "obs": models[selected][s]["tasa_observada"],
                "recall": models[selected][s]["recall_alarma"], "recall_raw": models[raw_key][s]["recall_alarma"],
                "fa": models[selected][s]["tasa_falsa_alarma"], "fa_raw": models[raw_key][s]["tasa_falsa_alarma"],
            }.items()},
        },
        "ttf": {"mae": round(ttf["mae"], 2), "mae_base": round(ttf["mae_baseline_mediana"], 2),
                "mae48": round(ttf["mae_ultimas_48"], 2), "mae48_base": round(ttf["mae_ultimas_48_baseline"], 2)},
        "survival": {k: round(v, 3) for k, v in surv.items()},
        "others": _other_datasets(),
    }


def render(data: dict) -> str:
    payload = json.dumps(data, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")  # un "</script>" dentro de un dato cerraria la etiqueta
    return TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload)


def build_site(docs_dir: Path = DOCS_DIR, results_json: Path = RESULTS_JSON, predictions_csv: Path = PREDICTIONS_CSV,
               figures_dir: Path = FIGURES_DIR) -> Path:
    results = json.loads(results_json.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_csv)
    docs_dir.mkdir(parents=True, exist_ok=True)
    out = docs_dir / "index.html"
    out.write_text(render(build_data(results, predictions)), encoding="utf-8")
    (docs_dir / ".nojekyll").write_text("", encoding="utf-8")
    target = docs_dir / "figures"
    if target.exists():
        shutil.rmtree(target)
    for lang in LANGS:
        (target / lang).mkdir(parents=True, exist_ok=True)
        for name in FIGURES_USED:
            shutil.copy2(figures_dir / lang / name, target / lang / name)
    return out


def decode(data: dict, split: str, field: str = "p") -> tuple[np.ndarray, np.ndarray]:
    """Etiquetas y probabilidades de un conjunto tal como las ve la pagina (para pruebas)."""
    y = np.array(data["pred"][split]["y"])
    p = np.array(data["pred"][split][field], dtype=float).reshape(-1, 5) / PROB_SCALE
    return y, p


if __name__ == "__main__":
    path = build_site()
    print(f"{path} ({path.stat().st_size / 1024:.0f} KB)")
