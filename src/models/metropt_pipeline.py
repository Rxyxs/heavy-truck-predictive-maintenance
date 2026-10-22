"""Deteccion de fugas de aire en MetroPT-3 sin etiquetas: cinco detectores evaluados contra el registro de fallas.

Protocolo (fijado antes de mirar el periodo de evaluacion):

* Ventanas de 10 minutos con al menos 30 lecturas. Por ventana: medias de las senales analogas, desviacion de
  TP2, corriente y aceite, minimos de Reservoirs y TP3, fraccion de tiempo activa de cada senal digital y
  cantidad de arranques en carga (flancos de subida de ``DV_eletric``).
* Dos vistas de las mismas variables: ``absoluta`` y ``deriva_ajustada`` (cada variable menos su mediana de los
  7 dias anteriores, calculada solo con el pasado). La segunda existe porque el ciclo de trabajo y la
  temperatura del aceite suben con el verano y un detector entrenado en invierno lo confunde con una falla.
* Referencia "normal": 1 de febrero al 15 de marzo para ajustar, 16 al 31 de marzo para fijar el umbral
  (percentil 99,5 del puntaje suavizado). No hay fallas reportadas antes del 18 de abril.
* Puntaje suavizado con la mediana de 3 ventanas (30 minutos). Alarma = puntaje suavizado sobre el umbral.
* Evaluacion del 1 de abril al 31 de agosto: una falla esta detectada si hay alarma entre 2 horas antes de su
  inicio y su fin (y, como sensibilidad, 24 horas antes). Una falsa alarma es un episodio de alarma que empieza
  fuera de las zonas de falla (desde 24 h antes del inicio hasta la mantencion, o 12 h despues del fin).
* Nivel de azar: la misma serie de alarmas desplazada en el tiempo (corrimiento circular, 500 veces) conserva
  cuanto y como alarma el detector pero no cuando. Las fallas que "detecta" esa serie desplazada son las que se
  detectarian por suerte; un detector util tiene que quedar claramente por encima.

Detectores: ``lps_rule`` (el presostato de baja presion que ya trae el equipo, sin aprendizaje), ``duty_zscore``
(desvio del ciclo de trabajo), ``pca`` (error de reconstruccion), ``isolation_forest`` y ``autoencoder``
(PyTorch, minimo 100 epocas con parada temprana y checkpoint).

    python -m src.models.metropt_pipeline
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.data import metropt_loader as M
from src.models import nets

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "metropt"
PROCESSED = ROOT / "data" / "processed"
SEED = 42
WINDOW = "10m"
MIN_SAMPLES = 30
SMOOTH = 3
QUANTILE = 0.995
FIT_PERIOD = ("2020-02-01", "2020-03-16")
CALIB_PERIOD = ("2020-03-16", "2020-04-01")
EVAL_PERIOD = ("2020-04-01", "2020-09-01")
HORIZONS_H = (2, 24)
N_SHIFTS = 500
PRE_ZONE_H = 24
POST_ZONE_H = 12
DETECTORS = ["lps_rule", "duty_zscore", "pca", "isolation_forest", "autoencoder"]


# ------------------------------------------------------------------ variables por ventana
def window_features(raw: pl.DataFrame, every: str = WINDOW, min_samples: int = MIN_SAMPLES) -> pd.DataFrame:
    df = raw.with_columns((pl.col("DV_eletric").diff().fill_null(0) > 0).cast(pl.Int32).alias("load_start"))
    aggs = [pl.len().alias("n")]
    aggs += [pl.col(c).mean().alias(f"{c}_mean") for c in M.ANALOG]
    aggs += [pl.col(c).std().alias(f"{c}_std") for c in ("TP2", "Motor_current", "Oil_temperature")]
    aggs += [pl.col(c).min().alias(f"{c}_min") for c in ("Reservoirs", "TP3")]
    aggs += [pl.col(c).mean().alias(f"{c}_frac") for c in M.DIGITAL]
    aggs.append(pl.col("load_start").sum().alias("load_starts"))
    out = df.group_by_dynamic("timestamp", every=every).agg(aggs).filter(pl.col("n") >= min_samples)
    return out.to_pandas().set_index("timestamp").drop(columns="n").fillna(0.0)


def drift_adjusted(features: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """Cada variable menos su mediana de los ``days`` dias anteriores (ventana cerrada a la izquierda: solo pasado)."""
    baseline = features.rolling(f"{days}D", closed="left", min_periods=50).median()
    return (features - baseline).dropna()


def period(frame: pd.DataFrame, bounds: tuple[str, str]) -> pd.DataFrame:
    return frame.loc[(frame.index >= bounds[0]) & (frame.index < bounds[1])]


# ------------------------------------------------------------------ detectores
class AutoEncoder(torch.nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8), torch.nn.ReLU(),
            torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, n),
        )

    def forward(self, x):
        return self.net(x)


def score_detector(name: str, fit: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.Series], dict]:
    """Ajusta el detector en ``fit`` y devuelve el puntaje (mayor = mas anomalo) de cada marco de ``frames``."""
    info: dict = {}
    if name == "lps_rule":
        return {k: f["LPS_frac"] for k, f in frames.items()}, {"threshold_rule": "LPS active in the window"}
    if name == "duty_zscore":
        mu, sd = fit["DV_eletric_frac"].mean(), fit["DV_eletric_frac"].std() or 1.0
        return {k: (f["DV_eletric_frac"] - mu) / sd for k, f in frames.items()}, info
    scaler = StandardScaler().fit(fit)

    def z(f):
        return np.nan_to_num(scaler.transform(f), nan=0.0, posinf=0.0, neginf=0.0).clip(-20, 20)

    if name == "pca":
        pca = PCA(n_components=0.95, random_state=SEED).fit(z(fit))
        info["components"] = int(pca.n_components_)

        def score(f):
            a = z(f)
            return ((a - pca.inverse_transform(pca.transform(a))) ** 2).mean(axis=1)
    elif name == "isolation_forest":
        forest = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1).fit(z(fit))

        def score(f):
            return -forest.score_samples(z(f))
    elif name == "autoencoder":
        a = z(fit).astype(np.float32)
        rng = np.random.default_rng(SEED)
        valid = rng.random(len(a)) < 0.15
        model = AutoEncoder(a.shape[1])
        hist = nets.fit_network(model, torch.nn.MSELoss(reduction="none"), a[~valid], a[~valid], None,
                                a[valid], a[valid], None,
                                nets.TrainConfig(batch_size=256, seed=SEED,
                                                 checkpoint_path=PROCESSED / "models" / "metropt_autoencoder.pt"))
        info.update(best_epoch=hist.best_epoch, epochs_run=hist.epochs_run)

        def score(f):
            b = z(f).astype(np.float32)
            return ((nets.predict(model, b) - b) ** 2).mean(axis=1)
    else:
        raise ValueError(name)
    return {k: pd.Series(score(f), index=f.index) for k, f in frames.items()}, info


def smooth(score: pd.Series, k: int = SMOOTH) -> pd.Series:
    return score.rolling(k, min_periods=1).median()


# ------------------------------------------------------------------ evaluacion por eventos
def episodes(alarm: pd.Series, max_gap: pd.Timedelta = pd.Timedelta("20min")) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Agrupa ventanas en alarma consecutivas (separadas por <= ``max_gap``) en episodios (inicio, fin)."""
    times = alarm.index[alarm.to_numpy()]
    if len(times) == 0:
        return []
    out, start, prev = [], times[0], times[0]
    for t in times[1:]:
        if t - prev > max_gap:
            out.append((start, prev))
            start = t
        prev = t
    out.append((start, prev))
    return out


def failure_zones(failures: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    zones = []
    for _, f in failures.iterrows():
        end = f["maintenance"] if pd.notna(f["maintenance"]) else f["end"] + pd.Timedelta(hours=POST_ZONE_H)
        zones.append((f["start"] - pd.Timedelta(hours=PRE_ZONE_H), max(end, f["end"] + pd.Timedelta(hours=POST_ZONE_H))))
    return zones


def detected_count(alarm_values: np.ndarray, index: pd.DatetimeIndex, failures: pd.DataFrame, hours: int) -> int:
    count = 0
    for _, f in failures.iterrows():
        inside = (index >= f["start"] - pd.Timedelta(hours=hours)) & (index <= f["end"])
        count += bool(alarm_values[inside].any())
    return count


def chance_level(alarm: pd.Series, failures: pd.DataFrame, hours: int, n: int = N_SHIFTS, seed: int = SEED) -> float:
    """Fallas detectadas en promedio por la misma serie de alarmas desplazada al azar (corrimiento circular)."""
    rng = np.random.default_rng(seed)
    values = alarm.to_numpy()
    m = len(values)
    shifts = rng.integers(int(0.1 * m), int(0.9 * m), n)
    return float(np.mean([detected_count(np.roll(values, k), alarm.index, failures, hours) for k in shifts]))


def evaluate(alarm: pd.Series, failures: pd.DataFrame) -> dict:
    """Fallas detectadas, adelanto y falsas alarmas por semana, sobre el periodo de evaluacion."""
    zones = failure_zones(failures)
    out = {"per_failure": []}
    for _, f in failures.iterrows():
        row = {"id": int(f["id"])}
        for h in HORIZONS_H:
            window = alarm[(alarm.index >= f["start"] - pd.Timedelta(hours=h)) & (alarm.index <= f["end"])]
            hit = window[window]
            row[f"detected_{h}h"] = bool(len(hit))
            row[f"lead_hours_{h}h"] = round((f["start"] - hit.index[0]).total_seconds() / 3600, 2) if len(hit) else None
        out["per_failure"].append(row)
    eps = episodes(alarm)
    false = [e for e in eps if not any(z0 <= e[0] <= z1 for z0, z1 in zones)]
    in_zone = np.zeros(len(alarm), dtype=bool)
    for z0, z1 in zones:
        in_zone |= (alarm.index >= z0) & (alarm.index <= z1)
    normal_weeks = float((~in_zone).sum()) * 10 / (60 * 24 * 7)
    for h in HORIZONS_H:
        out[f"detected_{h}h"] = int(sum(r[f"detected_{h}h"] for r in out["per_failure"]))
        leads = [r[f"lead_hours_{h}h"] for r in out["per_failure"] if r[f"lead_hours_{h}h"] is not None]
        out[f"median_lead_hours_{h}h"] = float(np.median(leads)) if leads else None
        out[f"chance_detected_{h}h"] = round(chance_level(alarm, failures, h), 2)
    out["false_alarm_episodes"] = len(false)
    out["false_alarms_per_week"] = round(len(false) / normal_weeks, 2) if normal_weeks else None
    out["share_of_normal_time_in_alarm"] = round(float(alarm[~in_zone].mean()), 4)
    out["normal_weeks_observed"] = round(normal_weeks, 1)
    return out


# ------------------------------------------------------------------ main
def run() -> dict:
    M.download()
    raw = M.read_raw()
    assert raw.height == M.EXPECTED_ROWS, raw.height
    absolute = window_features(raw)
    views = {"absolute": absolute, "drift_adjusted": drift_adjusted(absolute)}
    fails = M.failures()
    dt = raw["timestamp"].diff().dt.total_seconds()
    results = {
        "data": {"rows": raw.height, "start": str(raw["timestamp"].min()), "end": str(raw["timestamp"].max()),
                 "median_sampling_seconds": float(dt.median()), "gaps_over_10min": int((dt > 600).sum()),
                 "hours_in_gaps": round(float(dt.filter(dt > 600).sum()) / 3600, 1),
                 "windows": int(len(absolute)), "features": int(absolute.shape[1])},
        "protocol": {"window": WINDOW, "min_samples": MIN_SAMPLES, "smoothing_windows": SMOOTH, "threshold_quantile": QUANTILE,
                     "fit_period": FIT_PERIOD, "calibration_period": CALIB_PERIOD, "evaluation_period": EVAL_PERIOD,
                     "detection_horizons_hours": HORIZONS_H, "failure_zone_hours": [PRE_ZONE_H, POST_ZONE_H]},
        "failures": [{k: (str(v) if v is not None else None) for k, v in f.items()} for f in M.FAILURES],
        "detectors": {},
    }
    timeline = {}
    for view_name, feats in views.items():
        frames = {"fit": period(feats, FIT_PERIOD), "calib": period(feats, CALIB_PERIOD), "eval": period(feats, EVAL_PERIOD)}
        for det in DETECTORS:
            if det == "lps_rule" and view_name != "absolute":
                continue  # la regla del presostato no depende de la vista
            scores, info = score_detector(det, frames["fit"], frames)
            calib = smooth(scores["calib"])
            threshold = 0.0 if det == "lps_rule" else float(np.quantile(calib, QUANTILE))
            evaluation = smooth(scores["eval"])
            alarm = evaluation > threshold
            key = det if det == "lps_rule" else f"{det}__{view_name}"
            results["detectors"][key] = {"detector": det, "view": view_name, "threshold": threshold,
                                         "calibration_alarm_share": round(float((calib > threshold).mean()), 4),
                                         **info, **evaluate(alarm, fails)}
            timeline[key] = evaluation / threshold if threshold > 0 else evaluation
            r = results["detectors"][key]
            print(f"[MetroPT] {key:32s} detectadas 2h {r['detected_2h']}/4 (azar {r['chance_detected_2h']})  "
                  f"24h {r['detected_24h']}/4 (azar {r['chance_detected_24h']})  "
                  f"falsas/semana {r['false_alarms_per_week']}  tiempo en alarma {r['share_of_normal_time_in_alarm']:.1%}",
                  flush=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(timeline).to_parquet(PROCESSED / "metropt_scores.parquet")
    return results


if __name__ == "__main__":
    run()
