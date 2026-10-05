"""Genera la pagina de GitHub Pages (``docs/index.html``) a partir de ``results.json`` y de las
probabilidades por camion (``reports/predictions.csv``).

La pagina recalcula en el navegador la decision de costo esperado minimo con el costo de una visita
a taller que elija el lector. Para eso incrusta las probabilidades de los camiones de validacion y
test; todo lo demas sale de ``results.json``, asi que no puede desincronizarse del README.

    python -m src.site
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from src.data.scania_loader import COST_MATRIX

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("site_template.html")
RESULTS_JSON = ROOT / "reports" / "results.json"
PREDICTIONS_CSV = ROOT / "reports" / "predictions.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
DOCS_DIR = ROOT / "docs"
FIGURES_USED = ["02_matriz_confusion.png", "03_shap_tiempo_restante.png"]
PROB_SCALE = 1_000_000  # las probabilidades viajan como enteros: son mas cortas que los decimales
SPLITS = ("validation", "test")


def _saving(rules: dict) -> float:
    """Ahorro del costo esperado minimo frente a no recomendar nada, en %."""
    return 100.0 * (1.0 - rules["costo_esperado_minimo"]["costo_total"] / rules["siempre_sano"]["costo_total"])


def _saving_at_one(sensitivity: list[dict]) -> float:
    """Ahorro (%) con la visita al costo del reto (k = 1) dentro de una tabla de sensibilidad."""
    return round(next(row["ahorro_pct"] for row in sensitivity if row["k"] == 1.0), 2)


def _calibration_rows(rows: list[dict]) -> list[dict]:
    return [{"desde": r["desde"], "hasta": r["hasta"], "n": r["n"], "predicho": round(r["predicho"], 4), "observado": round(r["observado"], 4)} for r in rows]


def build_data(results: dict, predictions: pd.DataFrame) -> dict:
    clf = results["clasificador"]
    val, test = clf["validation"], clf["test"]
    ttf = results["tiempo_restante"]

    pred = {}
    for split in SPLITS:
        part = predictions[predictions["split"] == split].reset_index(drop=True)
        if len(part) != clf[split]["n_vehiculos"]:
            raise ValueError(f"{split}: {len(part)} filas en predictions.csv, pero results.json dice {clf[split]['n_vehiculos']}")
        probs = part[[f"p{k}" for k in range(5)]].to_numpy()
        pred[split] = {
            "y": part["class_label"].astype(int).tolist(),
            "p": [int(round(v * PROB_SCALE)) for v in probs.ravel()],
        }

    return {
        "cost_matrix": COST_MATRIX,
        "pred": pred,
        "platt": clf["platt"],
        "calibration": {s: _calibration_rows(clf[s]["calibracion_tabla"]) for s in SPLITS},
        "cal_summary": {
            s: {
                "pred": round(clf[s]["calibracion_global"]["riesgo_medio_predicho"], 4),
                "obs": round(clf[s]["calibracion_global"]["tasa_observada"], 4),
                "top_obs": round(clf[s]["calibracion_tabla"][-1]["observado"], 4),
            }
            for s in SPLITS
        },
        "key": {
            "saving_val": round(_saving(val), 2), "saving_test": round(_saving(test), 2),
            "auc_test": round(test["auc_algun_riesgo"], 2),
            "trucks_train": results["datos"]["vehiculos_train"], "fails_train": results["datos"]["reparaciones_train"],
            "n_val": val["n_vehiculos"], "n_test": test["n_vehiculos"],
            "fails_val": val["costo_esperado_minimo"]["fallas_reales"], "fails_test": test["costo_esperado_minimo"]["fallas_reales"],
            "recall_val": round(val["costo_esperado_minimo"]["recall_alarma"], 4),
            "recall_test": round(test["costo_esperado_minimo"]["recall_alarma"], 4),
            "cost_val": val["costo_esperado_minimo"]["costo_total"], "cost_test": test["costo_esperado_minimo"]["costo_total"],
            "nothing_val": val["siempre_sano"]["costo_total"], "nothing_test": test["siempre_sano"]["costo_total"],
            "be_val": round(val["costo_visita_equilibrio"], 2), "be_test": round(test["costo_visita_equilibrio"], 2),
            "recal_saving_val": _saving_at_one(val["recalibrado_con_el_otro_conjunto"]["sensibilidad_costo_visita"]),
            "recal_saving_test": _saving_at_one(test["recalibrado_con_el_otro_conjunto"]["sensibilidad_costo_visita"]),
            "recal_be_val": round(val["recalibrado_con_el_otro_conjunto"]["costo_visita_equilibrio"], 2),
            "recal_be_test": round(test["recalibrado_con_el_otro_conjunto"]["costo_visita_equilibrio"], 2),
        },
        "ttf": {
            "mae": round(ttf["mae"], 2), "mae_base": round(ttf["mae_baseline_mediana"], 2),
            "mae48": round(ttf["mae_ultimas_48"], 2), "mae48_base": round(ttf["mae_ultimas_48_baseline"], 2),
        },
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
    target.mkdir(exist_ok=True)
    for name in FIGURES_USED:
        shutil.copy2(figures_dir / name, target / name)
    return out


if __name__ == "__main__":
    path = build_site()
    print(f"{path} ({path.stat().st_size / 1024:.0f} KB)")

