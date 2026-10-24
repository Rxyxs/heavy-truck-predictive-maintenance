"""Demo: camiones de ejemplo, modelo publicado y una puntuacion de punta a punta sin entrenar nada.

    python -m src.demo fetch-model     # descarga el modelo entrenado desde el release de GitHub (~40 MB)
    python -m src.demo examples        # escribe examples/truck_*.json desde la validacion oficial (requiere los datos)
    python -m src.demo score           # puntua los ejemplos con el modelo y muestra la recomendacion

Los ejemplos son camiones reales del conjunto de validacion de SCANIA Component X (CC BY 4.0, (c) Scania CV AB),
con su historial de lecturas hasta la lectura evaluada. Sirven de cuerpo para ``POST /score``.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
MODELS_DIR = ROOT / "data" / "processed" / "models"
RELEASE_URL = "https://github.com/Rxyxs/heavy-truck-predictive-maintenance/releases/download/{tag}/scania-model.zip"
MODEL_TAG = "v1.0.0"
MODEL_FILES = ["risk_classifier.joblib", "feature_columns.json", "readout_columns.json", "spec_categories.json",
               "calibrator.json", "sequence_spec.json"]


def package_model(out: Path) -> Path:
    """Empaqueta los artefactos que necesita la API (para adjuntarlos a un release)."""
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in MODEL_FILES:
            if (MODELS_DIR / name).exists():
                zf.write(MODELS_DIR / name, name)
    return out


def fetch_model(tag: str = MODEL_TAG, models_dir: Path = MODELS_DIR) -> list[Path]:
    resp = requests.get(RELEASE_URL.format(tag=tag), timeout=300)
    resp.raise_for_status()
    models_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(models_dir)
        return [models_dir / n for n in zf.namelist()]


# Elegidos a proposito para mostrar los dos lados: un camion sano, dos fallas que el modelo detecta y una que no.
EXAMPLE_IDS = [10, 189, 1285, 3309]


def make_examples(ids: list[int] = EXAMPLE_IDS, out_dir: Path = EXAMPLES_DIR) -> list[Path]:
    """Camiones del conjunto de validacion, con todo su historial hasta la lectura evaluada."""
    import polars as pl

    from src.data import scania_loader as S

    readouts = S.read_readouts("validation")
    labels = S.read_labels("validation")
    specs = S.read_specs("validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("truck_*.json"):
        old.unlink()
    paths = []
    for vid in ids:
        cls = int(labels.filter(pl.col("vehicle_id") == vid)["class_label"][0])
        hist = readouts.filter(pl.col("vehicle_id") == vid).sort("time_step").drop("vehicle_id")
        body = {
            "vehicle_id": int(vid), "class_label": cls,
            "readouts": [{k: (None if v is None else float(v)) for k, v in row.items()} for row in hist.iter_rows(named=True)],
            "specs": dict(specs.filter(pl.col("vehicle_id") == vid).drop("vehicle_id").row(0, named=True)),
        }
        path = out_dir / f"truck_{vid}_class{cls}.json"
        path.write_text(json.dumps(body, separators=(",", ":")), encoding="utf-8")
        paths.append(path)
    return paths


def score_examples(examples_dir: Path = EXAMPLES_DIR) -> list[dict]:
    from src.models.scorer import RiskScorer

    scorer = RiskScorer.load()
    rows = []
    for path in sorted(examples_dir.glob("truck_*.json")):
        body = json.loads(path.read_text(encoding="utf-8"))
        out = scorer.score(body["readouts"], body["specs"])
        rows.append({"file": path.name, "true_class": body.get("class_label"), "readouts": out["n_lecturas"],
                     "any_risk": out["probabilidad_algun_riesgo"], "recommended": out["clase_recomendada"],
                     "action": out["accion_recomendada"]})
    return rows


def main(argv: list[str]) -> None:
    cmd = argv[0] if argv else ""
    if cmd == "fetch-model":
        for p in fetch_model():
            print(p)
    elif cmd == "examples":
        for p in make_examples():
            print(p)
    elif cmd == "score":
        for row in score_examples():
            print(f"{row['file']:28s} real={row['true_class']}  lecturas={row['readouts']:3d}  riesgo={row['any_risk']:.3f}  "
                  f"recomienda={row['recommended']}  {row['action']}")
    elif cmd == "package":
        print(package_model(Path(argv[1]) if len(argv) > 1 else ROOT / "scania-model.zip"))
    else:
        print(__doc__)
        raise SystemExit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
