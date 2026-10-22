"""Umbral de costo minimo e intervalos por bootstrap, contra valores calculados a mano."""
import numpy as np
import pytest

from src.models import evaluation as V


def brute_force_cost(y, s, t, c_fp, c_fn):
    alarm = s >= t
    return c_fp * int((alarm & (y == 0)).sum()) + c_fn * int((~alarm & (y == 1)).sum())


def test_cost_curve_by_hand():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.4, 0.1])
    thresholds, costs = V.cost_curve(y, s, cost_fp=10, cost_fn=500)
    # inf: nadie alarmado (2 fallas perdidas); 0.9: pierde una; 0.8: pierde una y una falsa; 0.4: una falsa; 0.1: dos falsas
    assert thresholds.tolist() == [np.inf, 0.9, 0.8, 0.4, 0.1]
    assert costs.tolist() == [1000, 500, 510, 10, 20]


def test_best_threshold_matches_brute_force_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.random(300) + 0.3 * y, 1)  # muchos empates
    t, cost = V.best_threshold(y, s, 10, 500)
    candidates = np.r_[np.unique(s), np.inf]
    best = min(brute_force_cost(y, s, c, 10, 500) for c in candidates)
    assert cost == best == brute_force_cost(y, s, t, 10, 500)


def test_best_threshold_never_alarms_when_alarms_are_too_expensive():
    y = np.array([0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.1])
    t, cost = V.best_threshold(y, s, cost_fp=1000, cost_fn=1)
    assert t == np.inf and cost == 1


def test_bootstrap_interval_contains_the_point_estimate():
    x = np.random.default_rng(1).normal(5, 1, 500)
    out = V.bootstrap(np.mean, x, n=300, seed=0)
    assert out["ci95"][0] < out["point"] < out["ci95"][1]
    assert out["ci95"][1] - out["ci95"][0] == pytest.approx(2 * 1.96 / np.sqrt(500), rel=0.25)


def test_group_bootstrap_resamples_whole_groups():
    groups = np.repeat(np.arange(10), 5)
    values = np.repeat(np.arange(10, dtype=float), 5)  # constante dentro de cada grupo
    seen = []
    V.bootstrap(lambda v: seen.append(len(v)) or 0.0, values, n=20, seed=0, groups=groups)
    assert all(n == 50 for n in seen[1:])  # siempre 10 grupos completos de 5
