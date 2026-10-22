"""Carga del dataset real APS Failure at Scania Trucks (IDA 2016 Industrial Challenge).

Camiones Scania en uso diario. El sistema estudiado es el de presion de aire (APS), que alimenta frenos y
cambios. Clase positiva: falla de un componente especifico del APS; negativa: camiones con fallas en
componentes NO relacionados con el APS. 170 variables anonimizadas (contadores y 7 histogramas de 10 bins).

* Train: 60.000 camiones (1.000 positivos, 1,7%).  Test: 16.000 (375 positivos, 2,3%).
* Metrica oficial: costo total = 10 x (revisiones innecesarias) + 500 x (fallas no detectadas).
* El archivo de descripcion trae el podio del reto: 9.920 / 10.900 / 11.480.

Fuente: UCI Machine Learning Repository, dataset 421 (https://archive.ics.uci.edu/dataset/421).
Licencia de los datos: GNU GPL v3, (c) 2016 Scania CV AB.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "aps"
URL = "https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip"
FILES_SHA256 = {
    "aps_failure_training_set.csv": "bb484302e3a3a1c8ef5e1f0129c4dc7cbd58f350867f95b575461ca21ab6b9da",
    "aps_failure_test_set.csv": "2cdf6f7661c7b4c63333c93cdec36a3a82350176b604a2312cf82799fb2712f3",
}
DESCRIPTION = "aps_failure_description.txt"

COST_FALSE_ALARM = 10   # revision innecesaria en taller
COST_MISSED = 500       # falla no detectada (posible averia en ruta)
# Podio del reto, copiado del archivo de descripcion que acompana a los datos (puntaje = costo total en test).
IDA2016_PODIUM = [
    {"team": "Costa & Nascimento", "cost": 9920, "false_alarms": 542, "missed": 9},
    {"team": "Gondek, Hafner & Sampson", "cost": 10900, "false_alarms": 490, "missed": 12},
    {"team": "Garnaik, Das, Sreepada & Patra", "cost": 11480, "false_alarms": 398, "missed": 15},
]
EXPECTED = {"train": (60000, 1000), "test": (16000, 375)}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_downloaded(raw_dir: Path = RAW_DIR) -> bool:
    return all((raw_dir / n).is_file() and _sha256(raw_dir / n) == h for n, h in FILES_SHA256.items())


def download(raw_dir: Path = RAW_DIR, force: bool = False) -> list[Path]:
    """Descarga el zip de UCI y extrae los dos CSV, verificando su SHA-256. Idempotente."""
    if not force and is_downloaded(raw_dir):
        return [raw_dir / n for n in FILES_SHA256]
    resp = requests.get(URL, timeout=300)
    resp.raise_for_status()
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in [*FILES_SHA256, DESCRIPTION]:
            (raw_dir / name).write_bytes(zf.read(name))
    bad = [n for n, h in FILES_SHA256.items() if _sha256(raw_dir / n) != h]
    if bad:
        raise RuntimeError(f"Checksum distinto al publicado en {bad}")
    return [raw_dir / n for n in FILES_SHA256]


def _header_line(path: Path) -> int:
    """Los CSV traen 20 lineas de licencia antes del encabezado; se busca la linea que empieza con ``class,``."""
    with path.open("r", encoding="latin-1") as fh:
        for i, line in enumerate(fh):
            if line.startswith("class,"):
                return i
    raise ValueError(f"{path.name}: no se encontro el encabezado")


def read_split(split: str, raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, np.ndarray]:
    """``X`` (170 columnas numericas, faltantes como NaN) e ``y`` (1 = falla del APS)."""
    if split not in ("train", "test"):
        raise ValueError(f"split debe ser 'train' o 'test', no {split!r}")
    path = raw_dir / f"aps_failure_{'training' if split == 'train' else 'test'}_set.csv"
    df = pd.read_csv(path, skiprows=_header_line(path), na_values=["na"])
    y = (df.pop("class") == "pos").astype(int).to_numpy()
    return df.astype("float64"), y


def histogram_groups(columns: list[str]) -> dict[str, list[str]]:
    """Variables con varios bins (``ag_000..ag_009``): son los 7 histogramas del dataset."""
    groups: dict[str, list[str]] = {}
    for c in columns:
        groups.setdefault(c.split("_")[0], []).append(c)
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}


def engineer(X: pd.DataFrame) -> pd.DataFrame:
    """Variables crudas + cantidad de faltantes por camion + total y fraccion de cada bin de histograma.

    Todo se calcula fila a fila (no usa estadisticas de otros camiones), asi que no hay fuga entre train y test.
    Se descarta ``cd_000``, que es constante.
    """
    out = X.drop(columns=[c for c in ("cd_000",) if c in X.columns]).copy()
    extra = {"n_missing": X.isna().sum(axis=1).astype(float)}
    for var, cols in histogram_groups(list(X.columns)).items():
        total = X[cols].sum(axis=1, min_count=1)
        extra[f"{var}_hist_total"] = total
        safe = total.where(total > 0)
        for c in cols:
            extra[f"{c}_frac"] = X[c] / safe
    return pd.concat([out, pd.DataFrame(extra, index=X.index)], axis=1)


def total_cost(y_true, y_pred) -> int:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    false_alarms = int(((y_pred == 1) & (y_true == 0)).sum())
    missed = int(((y_pred == 0) & (y_true == 1)).sum())
    return COST_FALSE_ALARM * false_alarms + COST_MISSED * missed
