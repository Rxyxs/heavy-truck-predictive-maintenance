"""MetroPT-3: variables por ventana, ajuste por deriva y evaluacion por eventos, con series sinteticas."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.data import metropt_loader as M
from src.models import metropt_pipeline as P

ROOT = Path(__file__).resolve().parents[1]


def synthetic_raw(minutes=60, start="2020-03-01 00:00:00"):
    """Una lectura cada 10 s; el compresor arranca en carga al minuto 5 y al 25 de cada media hora."""
    t = pd.date_range(start, periods=minutes * 6, freq="10s")
    minute = (np.arange(len(t)) // 6) % 30
    load = ((minute >= 5) & (minute < 10)) | ((minute >= 25) & (minute < 28))
    data = {"timestamp": t}
    for c in M.ANALOG:
        data[c] = np.where(load, 9.0, 8.0)
    for c in M.DIGITAL:
        data[c] = np.zeros(len(t))
    data["DV_eletric"] = load.astype(float)
    return pl.DataFrame(data)


def test_window_features_count_load_starts_and_duty_by_hand():
    f = P.window_features(synthetic_raw(30))
    assert len(f) == 3 and f.index[0] == pd.Timestamp("2020-03-01 00:00")
    assert f["load_starts"].tolist() == [1, 0, 1]       # arranques a los minutos 5 y 25
    assert f["DV_eletric_frac"].tolist() == pytest.approx([0.5, 0.0, 0.3])


def test_windows_with_too_few_readings_are_dropped():
    raw = synthetic_raw(30).filter(pl.col("timestamp") < pl.datetime(2020, 3, 1, 0, 12))  # 2 minutos en la 2a ventana
    assert len(P.window_features(raw)) == 1


def test_drift_adjustment_only_uses_the_past():
    idx = pd.date_range("2020-02-01", periods=24 * 6 * 20, freq="10min")
    feats = pd.DataFrame({"a": np.random.default_rng(0).normal(size=len(idx))}, index=idx)
    full = P.drift_adjusted(feats)
    cut = P.drift_adjusted(feats.iloc[: len(feats) // 2])
    common = cut.index
    pd.testing.assert_frame_equal(full.loc[common], cut)


def test_episodes_merge_consecutive_windows():
    idx = pd.date_range("2020-04-01", periods=10, freq="10min")
    alarm = pd.Series([0, 1, 1, 0, 0, 1, 0, 0, 1, 1], index=idx).astype(bool)
    eps = P.episodes(alarm)
    assert [(s.strftime("%H:%M"), e.strftime("%H:%M")) for s, e in eps] == [("00:10", "00:20"), ("00:50", "00:50"), ("01:20", "01:30")]


def test_evaluate_counts_detection_lead_time_and_false_alarms():
    idx = pd.date_range("2020-04-01", "2020-04-30", freq="10min")
    alarm = pd.Series(False, index=idx)
    fails = pd.DataFrame({"id": [1], "start": [pd.Timestamp("2020-04-20 12:00")], "end": [pd.Timestamp("2020-04-20 18:00")],
                          "maintenance": [pd.NaT]})
    alarm.loc["2020-04-20 11:00":"2020-04-20 13:00"] = True    # 1 h antes del inicio: detectada con 1 h de adelanto
    alarm.loc["2020-04-05 03:00":"2020-04-05 03:20"] = True    # falsa alarma
    alarm.loc["2020-04-19 20:00"] = True                       # dentro de las 24 h previas: ni acierto ni falsa
    out = P.evaluate(alarm, fails)
    assert out["detected_2h"] == 1 and out["per_failure"][0]["lead_hours_2h"] == pytest.approx(1.0)
    assert out["per_failure"][0]["lead_hours_24h"] == pytest.approx(16.0)
    assert out["false_alarm_episodes"] == 1


def test_chance_level_is_low_for_a_rare_alarm_and_high_for_a_constant_one():
    idx = pd.date_range("2020-04-01", "2020-08-31", freq="10min")
    fails = M.failures()
    rare = pd.Series(False, index=idx)
    rare.iloc[::5000] = True
    assert P.chance_level(rare, fails, 2, n=50) < 0.5
    assert P.chance_level(pd.Series(True, index=idx), fails, 2, n=5) == 4


def test_failure_log_is_ordered_and_maintenance_follows_each_failure():
    f = M.failures()
    assert (f["start"] < f["end"]).all() and f["start"].is_monotonic_increasing
    with_maintenance = f.dropna(subset=["maintenance"])
    assert (with_maintenance["maintenance"] > with_maintenance["end"]).all()


def test_reported_results_are_internally_consistent():
    path = ROOT / "reports" / "metropt" / "results.json"
    if not path.exists():
        pytest.skip("se genera con `python -m src.models.metropt_pipeline`")
    r = json.loads(path.read_text(encoding="utf-8"))
    for d in r["detectors"].values():
        assert 0 <= d["detected_2h"] <= d["detected_24h"] <= 4
        assert d["detected_2h"] == sum(f["detected_2h"] for f in d["per_failure"])
        assert 0 <= d["chance_detected_2h"] <= 4
