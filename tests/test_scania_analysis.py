"""Analisis adicionales de SCANIA Component X sobre datos sinteticos con resultado conocido."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.data.scania_loader import COST_MATRIX
from src.models import scania_analysis as SA


def proba_for(actions: np.ndarray) -> np.ndarray:
    """Probabilidades que hacen que la regla de costo esperado minimo elija exactamente ``actions``."""
    p = np.full((len(actions), 5), 1e-6)
    p[np.arange(len(actions)), actions] = 1.0
    return p / p.sum(axis=1, keepdims=True)


def test_cost_breakdown_splits_the_total_by_hand():
    y = np.array([0, 0, 3, 4, 2])
    pred = np.array([4, 0, 0, 4, 1])  # una falsa alarma, una falla no vista, un acierto, una ventana equivocada
    out = SA.cost_breakdown(y, proba_for(pred))
    assert out["false_alarms"] == COST_MATRIX[0][4]
    assert out["missed"] == COST_MATRIX[3][0]
    assert out["wrong_window"] == COST_MATRIX[2][1] + COST_MATRIX[4][4]
    assert out["total"] == out["false_alarms"] + out["missed"] + out["wrong_window"]


def test_paired_bootstrap_of_a_model_against_a_copy_of_itself_is_zero():
    rng = np.random.default_rng(0)
    y = rng.choice([0, 0, 0, 0, 4], 500)
    p = proba_for(np.where(rng.random(500) < 0.3, 4, 0))
    out = SA.paired_bootstrap(y, {"a": p, "b": p.copy()}, reference="a", n=200)
    assert out["b"]["mean_diff"] == 0 and out["b"]["ci95"] == [0, 0] and out["b"]["p_better"] == 0


def test_paired_bootstrap_prefers_the_model_that_catches_more_failures():
    y = np.array([0] * 90 + [4] * 10)
    good = proba_for(np.r_[np.zeros(90, int), np.full(10, 4)])
    bad = proba_for(np.zeros(100, int))
    out = SA.paired_bootstrap(y, {"bad": bad, "good": good}, reference="bad", n=300)
    assert out["good"]["mean_diff"] > 0 and out["good"]["p_better"] == 1.0


def test_lead_time_bins_failing_readouts_and_reports_healthy_trucks_apart():
    frame = pd.DataFrame({"vehicle_id": [1, 1, 1, 2, 3], "time_to_failure": [3.0, 30.0, 300.0, np.nan, np.nan],
                          "event": [1, 1, 1, 0, 0]})
    risk = np.array([0.9, 0.5, 0.1, 0.02, 0.04])
    p = np.c_[1 - risk, np.zeros((5, 3)), risk]
    out = SA.lead_time(frame, {"m": p})
    assert out["bins"][0] == "0-6" and out["bins"][-1] == "192+"
    assert out["readouts"] == [1, 0, 0, 1, 0, 0, 1]
    assert out["models"]["m"]["mean_risk"][0] == pytest.approx(0.9)
    assert out["models"]["m"]["healthy_mean_risk"] == pytest.approx(0.03)


def test_curves_have_the_expected_endpoints():
    rng = np.random.default_rng(1)
    y = rng.choice([0, 1], 400, p=[0.9, 0.1])
    score = y * 0.5 + rng.random(400) * 0.6
    p = np.c_[1 - score, score, np.zeros((400, 3))]
    out = SA.curves(y, {"m": p})
    roc = out["m"]["roc"]
    assert roc["x"][0] == 0 and roc["x"][-1] == 1 and roc["y"][-1] == 1
    assert 0.5 < out["m"]["auc"] <= 1 and out["base_rate"] == pytest.approx(y.mean(), abs=1e-4)


def test_cut_timing_shares_sum_to_one():
    frames = {k: pl.DataFrame({"time_step": np.random.default_rng(i).uniform(0, 400, 200)}) for i, k in enumerate(["a", "b"])}
    out = SA.cut_timing(frames["a"], {"b": frames["b"]})
    for k in ("train_random", "b"):
        assert sum(out[k]["share"]) == pytest.approx(1.0, abs=1e-3)


def test_kaplan_meier_picks_the_specification_that_separates_survival():
    rng = np.random.default_rng(2)
    n = 1200
    spec = rng.choice(["fast", "slow"], n)
    duration = np.where(spec == "fast", rng.exponential(80, n), rng.exponential(400, n)).clip(1, 500)
    event = (duration < 500).astype(int)
    tte = pl.DataFrame({"vehicle_id": np.arange(n), "duration": duration, "event": event})
    specs = pl.DataFrame({"vehicle_id": np.arange(n), **{f"Spec_{i}": rng.choice(["a", "b"], n) for i in range(8)}})
    specs = specs.with_columns(pl.Series("Spec_3", spec))
    out = SA.kaplan_meier(tte, specs)
    assert out["by_spec"]["spec"] == "Spec_3"
    assert out["overall"]["s"][0] == pytest.approx(1.0) and out["overall"]["s"][-1] < out["overall"]["s"][0]
