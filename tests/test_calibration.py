"""Calibracion y sensibilidad del ahorro al costo de una visita. Sin datos reales: todo sintetico o
derivado de reports/predictions.csv, que si esta versionado."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from src.data.scania_loader import COST_MATRIX
from src.models import calibration as K

ROOT = Path(__file__).resolve().parents[1]


def _proba_from_risk(risk):
    """Probabilidades de 5 clases con el riesgo repartido a partes iguales entre las clases 1-4."""
    risk = np.asarray(risk, dtype=float)
    return np.column_stack([1 - risk] + [risk / 4] * 4)


# ------------------------------------------------------------------------- matriz y ahorro
def test_scaled_matrix_only_changes_the_cost_of_an_unnecessary_visit():
    m = K.scaled_cost_matrix(3.0)
    assert m[0, 1:].tolist() == [21.0, 24.0, 27.0, 30.0]
    assert m[0, 0] == 0
    base = np.array(COST_MATRIX, dtype=float)
    assert (m[1:] == base[1:]).all()  # las fallas no detectadas y los errores de ventana no cambian


def test_scaled_matrix_does_not_modify_the_challenge_matrix():
    K.scaled_cost_matrix(50.0)
    assert COST_MATRIX[0] == [0, 7, 8, 9, 10]


def test_saving_by_hand():
    # Camion A: sano, riesgo bajo -> no se alarma. Camion B: falla (clase 4), riesgo alto -> se alarma.
    proba = np.array([[0.99, 0.0025, 0.0025, 0.0025, 0.0025], [0.20, 0.0, 0.0, 0.0, 0.80]])
    y = np.array([0, 4])
    out = K.saving_at(y, proba, 1.0)
    assert out["costo_nada"] == 500.0 and out["costo_politica"] == 0.0
    assert out["ahorro_pct"] == pytest.approx(100.0) and out["alarmados_pct"] == pytest.approx(50.0)


def test_policy_alarms_less_as_the_visit_gets_more_expensive():
    rng = np.random.default_rng(0)
    proba = _proba_from_risk(rng.uniform(0, 0.6, 2000))
    y = (rng.random(2000) < 0.03).astype(int) * 4
    alarms = [K.saving_at(y, proba, k)["alarmados_pct"] for k in (0.5, 1, 2, 5, 10, 50)]
    assert all(a >= b for a, b in zip(alarms, alarms[1:]))


def test_with_calibrated_probabilities_the_policy_never_loses_in_expectation():
    """La propiedad que las probabilidades crudas del modelo violan: con probabilidades CREIBLES, la politica de
    costo esperado minimo no puede salir peor que no recomendar nada, por caro que sea revisar un camion."""
    rng = np.random.default_rng(1)
    risk = rng.beta(0.6, 10, 60000)
    y = (rng.random(60000) < risk).astype(int) * 4  # las etiquetas salen de ESAS probabilidades
    proba = _proba_from_risk(risk)
    for k in (0.5, 1, 3, 10, 50):
        assert K.saving_at(y, proba, k)["ahorro_pct"] > -1.0, k


def test_overconfident_probabilities_can_make_the_policy_lose_money():
    rng = np.random.default_rng(2)
    true_risk = rng.beta(0.6, 10, 60000)
    y = (rng.random(60000) < true_risk).astype(int) * 4
    inflated = np.clip(true_risk * 6, 0, 0.99)  # el modelo exagera el riesgo 6 veces
    assert K.saving_at(y, _proba_from_risk(inflated), 10.0)["ahorro_pct"] < -20.0


def test_break_even_is_none_when_the_saving_never_runs_out():
    y = np.array([4] * 50 + [0] * 50)
    perfect = _proba_from_risk(np.array([1.0] * 50 + [0.0] * 50))
    assert K.break_even(y, perfect) is None


def test_break_even_below_one_when_the_policy_already_loses_at_the_challenge_cost():
    rng = np.random.default_rng(3)
    y = (rng.random(20000) < 0.005).astype(int) * 4  # con 2% de fallas alarmar a todos todavia ahorra algo
    useless = _proba_from_risk(np.full(20000, 0.5))  # alarma a todos, sin informacion
    be = K.break_even(y, useless)
    assert be is not None and be < 1.0


def test_break_even_is_where_the_saving_falls_to_the_threshold():
    rng = np.random.default_rng(4)
    risk = rng.beta(0.6, 8, 20000)
    y = (rng.random(20000) < risk * 0.3).astype(int) * 4  # el modelo exagera ~3x
    proba = _proba_from_risk(risk)
    be = K.break_even(y, proba)
    assert be is not None
    assert K.saving_at(y, proba, be * 0.8)["ahorro_pct"] > K.BREAK_EVEN_THRESHOLD_PCT
    assert K.saving_at(y, proba, be * 1.25)["ahorro_pct"] <= K.BREAK_EVEN_THRESHOLD_PCT + 1e-6


# -------------------------------------------------------------------------- tabla y error
def test_calibration_table_counts_every_truck_once():
    rng = np.random.default_rng(5)
    proba = _proba_from_risk(rng.uniform(0, 1, 1000))
    table = K.calibration_table((rng.random(1000) < 0.1).astype(int), proba)
    assert sum(r["n"] for r in table) == 1000
    assert all(r["desde"] < r["hasta"] for r in table)


def test_calibration_table_by_hand():
    proba = _proba_from_risk([0.02, 0.03, 0.40, 0.45])
    table = K.calibration_table(np.array([0, 1, 1, 0]), proba)
    low, high = table[0], table[-1]
    assert low["n"] == 2 and low["predicho"] == pytest.approx(0.025) and low["observado"] == pytest.approx(0.5)
    assert high["n"] == 2 and high["predicho"] == pytest.approx(0.425) and high["observado"] == pytest.approx(0.5)


def test_global_calibration_error():
    out = K.calibration_error(np.array([0, 0, 0, 4]), _proba_from_risk([0.1, 0.2, 0.3, 0.4]))
    assert out["riesgo_medio_predicho"] == pytest.approx(0.25) and out["tasa_observada"] == pytest.approx(0.25)


# ------------------------------------------------------------------------------- isotonica
def test_exported_isotonic_thresholds_reproduce_sklearn():
    """La pagina web no tiene sklearn: debe reproducir la funcion con solo los umbrales exportados."""
    rng = np.random.default_rng(6)
    risk = rng.beta(0.8, 6, 5000)
    y = (rng.random(5000) < risk * 0.4).astype(int)
    proba = _proba_from_risk(risk)
    cal = K.fit_risk_calibrator(proba, y * 4)
    exp = K.export_calibrator(cal)
    mine = K.apply_risk_calibration(proba, np.array(exp["x"]), np.array(exp["y"]))
    assert (1 - mine[:, 0]) == pytest.approx(cal.predict(risk), abs=1e-7)


def test_isotonic_calibration_is_monotone_and_keeps_rows_summing_to_one():
    rng = np.random.default_rng(7)
    risk = rng.uniform(0, 1, 3000)
    cal = K.fit_risk_calibrator(_proba_from_risk(risk), (rng.random(3000) < risk * 0.2).astype(int) * 4)
    exp = K.export_calibrator(cal)
    out = K.apply_risk_calibration(_proba_from_risk(risk), np.array(exp["x"]), np.array(exp["y"]))
    assert out.sum(axis=1) == pytest.approx(np.ones(3000), abs=1e-9)
    order = np.argsort(risk)
    assert (np.diff(1 - out[order, 0]) >= -1e-9).all()


def test_isotonic_calibration_works_with_sample_weights():
    rng = np.random.default_rng(8)
    risk = rng.uniform(0, 1, 2000)
    y = (rng.random(2000) < 0.1).astype(int) * 4
    weights = np.where(y > 0, 1.0, 20.0)
    assert isinstance(K.fit_risk_calibrator(_proba_from_risk(risk), y, weights), IsotonicRegression)


# ---------------------------------------------------------------------------------- Platt
def test_platt_recovers_a_known_overconfidence():
    rng = np.random.default_rng(9)
    n = 200000
    z = rng.normal(-1.0, 2.0, n)
    risk = 1 / (1 + np.exp(-z))
    truth = 1 / (1 + np.exp(-(0.2 * z - 3.0)))
    y = (rng.random(n) < truth).astype(int) * 4
    fit = K.fit_platt(_proba_from_risk(risk), y)
    assert fit["a"] == pytest.approx(0.2, abs=0.03) and fit["b"] == pytest.approx(-3.0, abs=0.15)


def test_platt_with_slope_one_and_no_shift_is_the_identity():
    risk = np.array([0.02, 0.1, 0.5, 0.9])
    proba = _proba_from_risk(risk)
    assert K.apply_platt(proba, 1.0, 0.0) == pytest.approx(proba, abs=1e-6)


def test_platt_keeps_rows_summing_to_one_and_the_split_among_risky_classes():
    proba = np.array([[0.7, 0.1, 0.1, 0.05, 0.05], [0.95, 0.04, 0.0, 0.01, 0.0]])
    out = K.apply_platt(proba, 0.2, -3.0)
    assert out.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert out[:, 1:] / out[:, 1:].sum(axis=1, keepdims=True) == pytest.approx(proba[:, 1:] / proba[:, 1:].sum(axis=1, keepdims=True))


def test_platt_does_not_change_the_ranking_of_trucks():
    rng = np.random.default_rng(10)
    proba = _proba_from_risk(rng.uniform(0.001, 0.99, 500))
    out = K.apply_platt(proba, 0.17, -3.3)
    assert (np.argsort(1 - proba[:, 0]) == np.argsort(1 - out[:, 0])).all()


# --------------------------------------------------------- integridad con results.json
@pytest.fixture(scope="module")
def committed():
    results = json.loads((ROOT / "reports" / "results.json").read_text(encoding="utf-8"))["clasificador"]
    preds = pd.read_csv(ROOT / "reports" / "predictions.csv")
    out = {}
    for split in ("validation", "test"):
        d = preds[preds["split"] == split]
        out[split] = (d["class_label"].to_numpy(), d[[f"p{i}" for i in range(5)]].to_numpy())
    return results, out


@pytest.mark.parametrize("split", ["validation", "test"])
def test_predictions_file_reproduces_the_reported_costs(committed, split):
    results, data = committed
    y, proba = data[split]
    assert len(y) == results[split]["n_vehiculos"]
    got = K.saving_at(y, proba, 1.0)
    assert got["costo_politica"] == results[split]["costo_esperado_minimo"]["costo_total"]
    assert got["costo_nada"] == results[split]["siempre_sano"]["costo_total"]


@pytest.mark.parametrize("split", ["validation", "test"])
def test_reported_break_even_matches_a_recomputation(committed, split):
    results, data = committed
    y, proba = data[split]
    assert K.break_even(y, proba) == pytest.approx(results[split]["costo_visita_equilibrio"], rel=1e-4)


@pytest.mark.parametrize("target,source", [("test", "validation"), ("validation", "test")])
def test_reported_platt_results_match_a_recomputation(committed, target, source):
    results, data = committed
    y_s, p_s = data[source]
    fit = K.fit_platt(p_s, y_s)
    # results.json se calcula con probabilidades en precision completa y predictions.csv guarda 6 decimales
    assert fit["a"] == pytest.approx(results["platt"][source]["a"], abs=0.01)
    assert fit["b"] == pytest.approx(results["platt"][source]["b"], abs=0.02)
    y_t, p_t = data[target]
    out = K.sensitivity(y_t, K.apply_platt(p_t, **fit))
    reported = results[target]["recalibrado_con_el_otro_conjunto"]["sensibilidad_costo_visita"]
    for mine, theirs in zip(out, reported):
        assert mine["ahorro_pct"] == pytest.approx(theirs["ahorro_pct"], abs=0.3)  # predictions.csv guarda 6 decimales


def test_raw_probabilities_overstate_the_risk_and_recalibration_fixes_the_average(committed):
    results, _ = committed
    for split in ("validation", "test"):
        raw = results[split]["calibracion_global"]
        assert raw["riesgo_medio_predicho"] > 2.5 * raw["tasa_observada"]
        recal = results[split]["recalibrado_con_el_otro_conjunto"]["calibracion_global"]
        assert recal["riesgo_medio_predicho"] < 1.6 * recal["tasa_observada"]
