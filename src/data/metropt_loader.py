"""Carga del dataset real MetroPT-3: unidad de produccion de aire (APU) del compresor de un tren de metro.

Senales de presion, temperatura del aceite, corriente del motor y valvulas, de febrero a agosto de 2020.
No trae etiquetas por fila: la empresa entrego un registro de fallas (fugas de aire) con su inicio y fin,
que es lo que se usa para evaluar la deteccion de anomalias. Es el mismo tipo de sistema que el APS de un
camion (aire comprimido para frenos y servicios), en otro vehiculo.

Fuente: UCI Machine Learning Repository, dataset 791 (https://archive.ics.uci.edu/dataset/791), CC BY 4.0.
Descripcion: Veloso et al., "The MetroPT dataset for predictive maintenance", Scientific Data 9, 764 (2022).

Dos detalles del archivo que no coinciden con su descripcion:

* La descripcion dice 15.169.480 puntos a 1 Hz; el CSV trae 1.516.948 filas, una cada ~10 segundos (la
  primera columna, sin nombre, es un indice que avanza de 10 en 10).
* La tabla de fallas numera "#1, #1, #3, #4" y asigna a la falla de fines de mayo una mantencion el
  "30Apr at 12:00", anterior a la propia falla. Se lee como 30 de mayo, en linea con las otras dos
  (mantencion al dia siguiente del fin de la falla).
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import polars as pl
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "metropt"
URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
CSV_NAME = "MetroPT3(AirCompressor).csv"
CSV_SHA256 = "db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24"
EXPECTED_ROWS = 1_516_948

ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
DIGITAL = ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]

# Registro de fallas de la empresa (todas fugas de aire, severidad "high stress"). ``maintenance`` es None
# cuando el reporte no la indica.
FAILURES = [
    {"id": 1, "start": "2020-04-18 00:00", "end": "2020-04-18 23:59", "maintenance": None},
    {"id": 2, "start": "2020-05-29 23:30", "end": "2020-05-30 06:00", "maintenance": "2020-05-30 12:00"},
    {"id": 3, "start": "2020-06-05 10:00", "end": "2020-06-07 14:30", "maintenance": "2020-06-08 16:00"},
    {"id": 4, "start": "2020-07-15 14:30", "end": "2020-07-15 19:00", "maintenance": "2020-07-16 00:00"},
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(raw_dir: Path = RAW_DIR, force: bool = False) -> Path:
    """Descarga el zip de UCI (~218 MB) y extrae el CSV, verificando su SHA-256. Idempotente."""
    target = raw_dir / CSV_NAME
    if not force and target.is_file() and _sha256(target) == CSV_SHA256:
        return target
    resp = requests.get(URL, timeout=600)
    resp.raise_for_status()
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        target.write_bytes(zf.read(CSV_NAME))
    if _sha256(target) != CSV_SHA256:
        raise RuntimeError("El CSV de MetroPT-3 no coincide con el checksum esperado")
    return target


def read_raw(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    """Todas las lecturas, ordenadas por tiempo, sin la columna de indice."""
    df = pl.read_csv(raw_dir / CSV_NAME, try_parse_dates=True)
    df = df.drop([c for c in df.columns if c in ("", "_duplicated_0")])
    return df.sort("timestamp")


def failures() -> pd.DataFrame:
    out = pd.DataFrame(FAILURES)
    for c in ("start", "end", "maintenance"):
        out[c] = pd.to_datetime(out[c])
    return out
