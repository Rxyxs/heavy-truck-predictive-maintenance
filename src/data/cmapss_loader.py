"""Carga del dataset NASA C-MAPSS (turbofan simulado). FD001 con sus variables moviles; FD001-FD004 en crudo.

Los cuatro subconjuntos combinan una o seis condiciones operativas y uno o dos modos de falla:
FD001 (1 condicion, 1 modo), FD002 (6, 1), FD003 (1, 2), FD004 (6, 2). ``download_all`` los descarga
todos, verificando cada archivo por SHA-256, y ``read_subset`` los lee en crudo.

FD001 trae 100 motores de entrenamiento, corridos hasta la falla, y 100 motores de test, cortados en un
ciclo previo a la falla; el archivo ``RUL_FD001.txt`` da la vida util restante real (RUL) de cada motor de
test en su ultimo ciclo. Cada fila es un ciclo con 3 ajustes operativos y 21 sensores.

Fuente: NASA Prognostics Center of Excellence, "Turbofan Engine Degradation Simulation Data Set"
(https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
Referencia: Saxena et al., "Damage propagation modeling for aircraft engine run-to-failure simulation", PHM 2008.

Los ids de motor de train y de test se solapan (ambos van de 1 a 100) pero son motores distintos, asi que
toda tabla de salida lleva la columna ``split`` y cualquier agrupacion debe hacerse por ``(split, unit)``.
Todas las transformaciones (RUL, ventanas moviles) se calculan dentro de cada motor y dentro de cada split,
con ventanas que solo miran hacia atras: ningun valor de un ciclo depende de ciclos posteriores ni de otros motores.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "cmapss"
SUMMARY_PATH = ROOT / "reports" / "cmapss" / "dataset_summary.json"

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
OUTER_SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
INNER_ZIP = "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip"
FILES_SHA256 = {
    "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
    "test_FD001.txt": "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851",
    "RUL_FD001.txt": "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca",
}

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
ALL_FILES_SHA256 = {
    **FILES_SHA256,
    "train_FD002.txt": "dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f",
    "test_FD002.txt": "de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02",
    "RUL_FD002.txt": "c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b",
    "train_FD003.txt": "2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2",
    "test_FD003.txt": "299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790",
    "RUL_FD003.txt": "df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb",
    "train_FD004.txt": "27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218",
    "test_FD004.txt": "1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5",
    "RUL_FD004.txt": "196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1",
}
SUBSET_INFO = {
    "FD001": {"conditions": 1, "fault_modes": 1}, "FD002": {"conditions": 6, "fault_modes": 1},
    "FD003": {"conditions": 1, "fault_modes": 2}, "FD004": {"conditions": 6, "fault_modes": 2},
}

SETTINGS = ["setting_1", "setting_2", "setting_3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTINGS, *SENSORS]
# Los 7 sensores que se descartan: s1, s5, s10, s16, s18, s19 son constantes en train y s6 solo toma
# dos valores. Quedan los 14 sensores informativos de uso habitual en FD001.
DROPPED_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
INFORMATIVE_SENSORS = [s for s in SENSORS if s not in DROPPED_SENSORS]

MAX_RUL = 125  # techo de la RUL lineal por tramos (125 o 130 en la literatura)
WINDOWS = (5, 10)
EXPECTED_ROWS = {"train": 20631, "test": 13096}
EXPECTED_UNITS = 100


# ------------------------------------------------------------------ descarga
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_downloaded(raw_dir: Path = RAW_DIR) -> bool:
    """Los tres archivos existen y su checksum coincide con el publicado."""
    return all((raw_dir / n).is_file() and _sha256((raw_dir / n).read_bytes()) == h for n, h in FILES_SHA256.items())


def _fetch(url: str, retries: int) -> bytes:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, timeout=120) as resp:
                resp.raise_for_status()
                return resp.content
        except requests.RequestException as exc:
            last = exc
            time.sleep(2 * attempt)
    raise RuntimeError(f"No se pudo descargar {url}: {last}")


def extract_fd001(outer_zip: bytes, raw_dir: Path, expected: dict[str, str] = FILES_SHA256) -> list[Path]:
    """Saca los archivos pedidos (por defecto FD001) del zip anidado de NASA y verifica el checksum de cada uno."""
    with zipfile.ZipFile(io.BytesIO(outer_zip)) as outer, zipfile.ZipFile(io.BytesIO(outer.read(INNER_ZIP))) as inner:
        payload = {name: inner.read(name) for name in expected}
    bad = [name for name, data in payload.items() if _sha256(data) != expected[name]]
    if bad:
        raise RuntimeError(f"Checksum distinto al esperado en {bad}: el archivo de origen cambio o esta corrupto")
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, data in payload.items():
        target = raw_dir / name
        target.write_bytes(data)
        paths.append(target)
    return paths


def download(raw_dir: Path = RAW_DIR, url: str = URL, retries: int = 3, force: bool = False) -> list[Path]:
    """Descarga FD001 si falta o esta alterado. Idempotente: si ya esta completo no toca la red."""
    if not force and is_downloaded(raw_dir):
        return [raw_dir / n for n in FILES_SHA256]
    blob = _fetch(url, retries)
    if _sha256(blob) != OUTER_SHA256:
        raise RuntimeError("El zip descargado no coincide con el checksum publicado")
    return extract_fd001(blob, raw_dir)


def download_all(raw_dir: Path = RAW_DIR, url: str = URL, retries: int = 3, force: bool = False) -> list[Path]:
    """Descarga los 12 archivos de FD001-FD004 si falta alguno o no coincide su checksum."""
    if not force and all((raw_dir / n).is_file() and _sha256((raw_dir / n).read_bytes()) == h
                         for n, h in ALL_FILES_SHA256.items()):
        return [raw_dir / n for n in ALL_FILES_SHA256]
    blob = _fetch(url, retries)
    if _sha256(blob) != OUTER_SHA256:
        raise RuntimeError("El zip descargado no coincide con el checksum publicado")
    return extract_fd001(blob, raw_dir, expected=ALL_FILES_SHA256)


def read_subset(subset: str, raw_dir: Path = RAW_DIR, auto_download: bool = True):
    """(train, test, RUL real del ultimo ciclo de cada motor de test) de un subconjunto, en crudo.

    Columnas: ``unit``, ``cycle``, ``op1..op3``, ``s1..s21`` (pandas), el formato que usa ``rul_estimator``.
    """
    import pandas as pd

    if subset not in SUBSETS:
        raise ValueError(f"subconjunto desconocido: {subset!r}")
    if auto_download:
        download_all(raw_dir)
    names = ["unit", "cycle", "op1", "op2", "op3", *SENSORS]
    train = pd.read_csv(raw_dir / f"train_{subset}.txt", sep=r"\s+", header=None, names=names)
    test = pd.read_csv(raw_dir / f"test_{subset}.txt", sep=r"\s+", header=None, names=names)
    rul = np.loadtxt(raw_dir / f"RUL_{subset}.txt").astype(float).reshape(-1)
    return train, test, rul


# ------------------------------------------------------------------ lectura
def _read_table(path: Path) -> pl.DataFrame:
    values = np.loadtxt(path)  # separado por espacios, con espacios al final de cada linea
    df = pl.DataFrame(values, schema=COLUMNS, orient="row")
    return df.with_columns(pl.col("unit").cast(pl.Int32), pl.col("cycle").cast(pl.Int32)).sort(["unit", "cycle"])


def read_split(split: str, raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    """Ciclos crudos de ``train`` o ``test``, ordenados por motor y ciclo."""
    if split not in ("train", "test"):
        raise ValueError(f"split debe ser 'train' o 'test', no {split!r}")
    return _read_table(raw_dir / f"{split}_FD001.txt")


def read_test_rul(raw_dir: Path = RAW_DIR) -> np.ndarray:
    """RUL real de cada motor de test en su ultimo ciclo; la fila i corresponde al motor i+1."""
    return np.loadtxt(raw_dir / "RUL_FD001.txt").astype(int).reshape(-1)


# ------------------------------------------------------------------ RUL
def add_train_rul(train: pl.DataFrame, max_rul: int = MAX_RUL) -> pl.DataFrame:
    """RUL de train: ciclos que faltan hasta la falla (el ultimo ciclo de cada motor es la falla), con techo.

    ``rul_uncapped`` = ultimo ciclo del motor - ciclo actual; ``rul`` = min(rul_uncapped, max_rul).
    """
    uncapped = (pl.col("cycle").max().over("unit") - pl.col("cycle")).cast(pl.Int32).alias("rul_uncapped")
    return train.with_columns(uncapped).with_columns(pl.col("rul_uncapped").clip(upper_bound=max_rul).alias("rul"))


def add_test_rul(test: pl.DataFrame, rul_last: np.ndarray, max_rul: int = MAX_RUL) -> pl.DataFrame:
    """RUL de test a partir de ``RUL_FD001.txt``: RUL del ultimo ciclo + ciclos que faltan para ese ultimo ciclo."""
    units = sorted(test["unit"].unique().to_list())
    if len(units) != len(rul_last):
        raise ValueError(f"{len(units)} motores de test pero {len(rul_last)} valores de RUL")
    lookup = pl.DataFrame({"unit": units, "rul_at_last_cycle": [int(v) for v in rul_last]}).with_columns(
        pl.col("unit").cast(pl.Int32), pl.col("rul_at_last_cycle").cast(pl.Int32)
    )
    out = test.join(lookup, on="unit", how="left")
    uncapped = (pl.col("rul_at_last_cycle") + pl.col("cycle").max().over("unit") - pl.col("cycle")).cast(pl.Int32).alias("rul_uncapped")
    return out.with_columns(uncapped).with_columns(pl.col("rul_uncapped").clip(upper_bound=max_rul).alias("rul")).drop("rul_at_last_cycle")


# ------------------------------------------------------------------ ingenieria de caracteristicas
def feature_names(sensors: list[str] = INFORMATIVE_SENSORS, windows: tuple[int, ...] = WINDOWS) -> list[str]:
    return [f"{s}_{stat}_{w}" for s in sensors for w in windows for stat in ("mean", "std")]


def add_rolling_features(
    df: pl.DataFrame, sensors: list[str] = INFORMATIVE_SENSORS, windows: tuple[int, ...] = WINDOWS
) -> pl.DataFrame:
    """Media y desviacion estandar moviles de cada sensor, por motor, con ventana que solo mira hacia atras.

    El valor en el ciclo t usa los ciclos max(1, t-w+1)..t del mismo motor. En los primeros ciclos la
    ventana esta incompleta: la media usa los ciclos disponibles y la desviacion (muestral, ddof=1) es 0.0
    con un solo ciclo. Requiere el DataFrame de UN solo split, ordenado por motor y ciclo.
    """
    if df.select(pl.struct("unit", "cycle").is_duplicated().any()).item():
        raise ValueError("hay pares (unit, cycle) repetidos: el DataFrame mezcla splits o esta duplicado")
    df = df.sort(["unit", "cycle"])
    exprs = []
    for s in sensors:
        for w in windows:
            exprs.append(pl.col(s).rolling_mean(window_size=w, min_samples=1).over("unit").alias(f"{s}_mean_{w}"))
            exprs.append(pl.col(s).rolling_std(window_size=w, min_samples=2).over("unit").fill_null(0.0).alias(f"{s}_std_{w}"))
    return df.with_columns(exprs)


# ------------------------------------------------------------------ dataset completo
@dataclass
class CMAPSSDataset:
    train: pl.DataFrame  # todos los ciclos de train, con rul y caracteristicas
    test: pl.DataFrame  # todos los ciclos de test, con rul y caracteristicas
    test_last: pl.DataFrame  # solo el ultimo ciclo de cada motor de test (la evaluacion estandar)
    feature_columns: list[str]
    sensor_columns: list[str]
    max_rul: int
    windows: tuple[int, ...]


def build_fd001(
    raw_dir: Path = RAW_DIR, max_rul: int = MAX_RUL, windows: tuple[int, ...] = WINDOWS, auto_download: bool = True
) -> CMAPSSDataset:
    """Descarga (si falta), lee, calcula la RUL y agrega las caracteristicas temporales de train y test por separado."""
    if auto_download:
        download(raw_dir)
    parts = {}
    for split in ("train", "test"):
        raw = read_split(split, raw_dir)
        keep = ["unit", "cycle", *SETTINGS, *INFORMATIVE_SENSORS]
        with_rul = add_train_rul(raw.select(keep), max_rul) if split == "train" else add_test_rul(raw.select(keep), read_test_rul(raw_dir), max_rul)
        parts[split] = add_rolling_features(with_rul, windows=windows).with_columns(pl.lit(split).alias("split"))
    test_last = parts["test"].filter(pl.col("cycle") == pl.col("cycle").max().over("unit"))
    return CMAPSSDataset(parts["train"], parts["test"], test_last, feature_names(windows=windows),
                         list(INFORMATIVE_SENSORS), max_rul, windows)


# ------------------------------------------------------------------ resumen
def check_causality(ds: CMAPSSDataset, units: int = 5, cut: int = 60) -> bool:
    """Recalcula las caracteristicas con cada serie cortada en ``cut`` ciclos y compara con las del dataset completo.

    Si alguna ventana mirara hacia adelante, los valores de los primeros ``cut`` ciclos cambiarian al
    quitar el futuro; si coinciden exactamente, el calculo es causal.
    """
    for frame in (ds.train, ds.test):
        for unit in sorted(frame["unit"].unique().to_list())[:units]:
            one = frame.filter(pl.col("unit") == unit)
            head = one.filter(pl.col("cycle") <= cut)
            base = head.drop(ds.feature_columns)
            redo = add_rolling_features(base, ds.sensor_columns, ds.windows)
            if not np.allclose(redo.select(ds.feature_columns).to_numpy(), head.select(ds.feature_columns).to_numpy(), rtol=0, atol=1e-9):
                return False
    return True


def _describe(values: pl.Series) -> dict:
    return {"min": float(values.min()), "max": float(values.max()), "mean": round(float(values.mean()), 3),
            "median": float(values.median()), "std": round(float(values.std()), 3)}


def summarize(ds: CMAPSSDataset) -> dict:
    cycles = {name: frame.group_by("unit").agg(pl.col("cycle").max().alias("n")) for name, frame in (("train", ds.train), ("test", ds.test))}
    model_cols = ds.sensor_columns + ds.feature_columns
    return {
        "dataset": "NASA C-MAPSS FD001",
        "source": URL,
        "files_sha256": FILES_SHA256,
        "max_rul": ds.max_rul,
        "windows": list(ds.windows),
        "rows": {"train": ds.train.height, "test": ds.test.height, "test_last_cycle": ds.test_last.height},
        "units": {"train": ds.train["unit"].n_unique(), "test": ds.test["unit"].n_unique()},
        "cycles_per_unit": {k: _describe(v["n"]) for k, v in cycles.items()},
        "sensors": {"informative": ds.sensor_columns, "dropped": DROPPED_SENSORS,
                    "drop_reason": "constantes en train (s1, s5, s10, s16, s18, s19) o con solo dos valores distintos (s6)"},
        "features": {"n_raw_sensors": len(ds.sensor_columns), "n_engineered": len(ds.feature_columns),
                     "n_model_inputs": len(model_cols), "examples": ds.feature_columns[:4]},
        "nulls_in_model_inputs": int(sum(ds.train[c].null_count() + ds.test[c].null_count() for c in model_cols)),
        "rul": {
            "train_capped": _describe(ds.train["rul"]),
            "train_uncapped": _describe(ds.train["rul_uncapped"]),
            "share_of_train_rows_at_cap": round(float((ds.train["rul"] == ds.max_rul).mean()), 4),
            "test_last_cycle": _describe(ds.test_last["rul"]),
            "test_last_cycle_share_at_cap": round(float((ds.test_last["rul"] == ds.max_rul).mean()), 4),
        },
        "train_sensor_stats": {s: _describe(ds.train[s]) for s in ds.sensor_columns},
        "leakage_checks": {
            "features_are_causal": check_causality(ds),
            "unit_ids_overlap_between_train_and_test": bool(set(ds.train["unit"].to_list()) & set(ds.test["unit"].to_list())),
            "rows_keyed_by_split_and_unit": sorted(ds.train["split"].unique().to_list() + ds.test["split"].unique().to_list()),
            "rul_columns_excluded_from_features": not any(c.startswith("rul") for c in ds.feature_columns),
        },
    }


def main() -> None:
    ds = build_fd001()
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summarize(ds), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"train {ds.train.shape}, test {ds.test.shape}, test_last {ds.test_last.shape}; resumen en {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
