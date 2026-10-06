"""Estimador de RUL y metricas de C-MAPSS: valores calculados a mano, propiedades exactas y un umbral sobre datos reales."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import rul_estimator as R

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "cmapss" / "raw"


def synthetic(n_units=30, seed=0):
    """Motores sinteticos: 7 sensores derivan con el desgaste, el resto es constante; una sola condicion de operacion."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        life = int(rng.integers(150, 250))
        for c in range(1, life + 1):
            row = {"unit": u, "cycle": c, "op1": 0.0, "op2": 0.0, "op3": 100.0}
            for i in range(1, 22):
                row[f"s{i}"] = 500.0 + (i * 3.0 * (c / life) + rng.normal(0, 0.3) if i <= 7 else 0.0)
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- metricas
def test_rmse_by_hand():
    assert R.rmse([10, 20, 30], [13, 16, 30]) == pytest.approx(math.sqrt((9 + 16 + 0) / 3))
    assert R.rmse([5.0], [5.0]) == 0.0


def test_nasa_score_by_hand():
    # d = prediccion - verdad. d < 0: exp(-d/13) - 1 ; d >= 0: exp(d/10) - 1
    assert R.nasa_score([100], [87]) == pytest.approx(math.exp(1) - 1)       # subestima 13
    assert R.nasa_score([100], [110]) == pytest.approx(math.exp(1) - 1)      # sobreestima 10: mismo castigo con 3 ciclos menos
    assert R.nasa_score([100, 50], [87, 60]) == pytest.approx(2 * (math.exp(1) - 1))
    assert R.nasa_score([40, 70], [40, 70]) == 0.0


def test_the_score_punishes_overestimating_more_than_underestimating():
    for d in (5, 10, 20, 40):
        assert R.nasa_score([100], [100 + d]) > R.nasa_score([100], [100 - d])


# ---------------------------------------------------------------------------- etiquetas
def test_training_rul_counts_down_to_zero_and_is_capped():
    df = pd.DataFrame({"unit": [1] * 5 + [2] * 3, "cycle": [1, 2, 3, 4, 5, 1, 2, 3]})
    assert R.train_rul(df, cap=None).tolist() == [4, 3, 2, 1, 0, 2, 1, 0]
    assert R.train_rul(df, cap=3).tolist() == [3, 3, 2, 1, 0, 2, 1, 0]


def test_last_rows_keeps_the_final_cycle_of_each_unit_sorted_by_unit():
    df = pd.DataFrame({"unit": [2, 1, 1, 2, 1], "cycle": [1, 2, 1, 2, 3], "x": range(5)})
    out = R.last_rows(df)
    assert out["unit"].tolist() == [1, 2] and out["cycle"].tolist() == [3, 2]


# ------------------------------------------------------------------------------ modelo
@pytest.fixture(scope="module")
def fitted():
    train = synthetic(30, seed=0)
    return train, R.RulEstimator("lightgbm", cap=125).fit(train)


@pytest.mark.parametrize("kind", ["lightgbm", "random_forest"])
def test_predictions_are_never_negative(kind):
    train = synthetic(12, seed=1)
    model = R.RulEstimator(kind, cap=125).fit(train)
    pred = model.predict_rows(train)
    assert (pred >= 0).all() and len(pred) == len(train)


def test_predictions_stay_non_negative_when_extrapolating_far_beyond_the_training_range(fitted):
    train, model = fitted
    extreme = train[train["unit"] == 1].copy()
    for i in range(1, 8):
        extreme[f"s{i}"] = extreme[f"s{i}"] + 1e4
    assert (model.predict_rows(extreme) >= 0).all()


def test_constant_sensors_are_dropped(fitted):
    _, model = fitted
    assert model.sensors_ == [f"s{i}" for i in range(1, 8)]


def test_a_prediction_never_uses_later_cycles(fitted):
    """Las variables en el ciclo t son iguales si se recortan los ciclos posteriores: sin fuga hacia adelante."""
    train, model = fitted
    unit = train[train["unit"] == 3]
    full = model.features(unit)
    cut = model.features(unit[unit["cycle"] <= 80])
    pd.testing.assert_frame_equal(full.loc[cut.index], cut)


def test_the_model_beats_a_constant_on_synthetic_engines(fitted):
    _, model = fitted
    test = synthetic(10, seed=99)
    last = R.last_rows(test)
    truth = np.zeros(len(last))  # los motores sinteticos corren hasta la falla: RUL 0 en el ultimo ciclo
    pred = model.predict_last(test).reindex(last["unit"]).to_numpy()
    baseline = np.full(len(truth), float(R.train_rul(synthetic(30, seed=0), 125).mean()))
    assert R.rmse(truth, pred) < R.rmse(truth, baseline) / 2


def test_predict_last_returns_one_value_per_unit(fitted):
    train, model = fitted
    out = model.predict_last(train)
    assert out.index.tolist() == sorted(train["unit"].unique()) and len(out) == train["unit"].nunique()


def test_unknown_model_kind_fails_loudly():
    with pytest.raises(ValueError):
        R.RulEstimator("xgboost").fit(synthetic(3))


# ----------------------------------------------------------------------------- datos reales
@pytest.mark.skipif(not (DATA / "train_FD001.txt").exists(), reason="C-MAPSS no descargado (data/raw/cmapss/raw)")
def test_fd001_error_beats_a_constant_and_meets_the_expected_threshold():
    from src.models.rul_benchmark import read_split

    train, test, rul_true = read_split(DATA, "FD001")
    model = R.RulEstimator("lightgbm", cap=125).fit(train)
    pred = model.predict_last(test).reindex(R.last_rows(test)["unit"]).to_numpy()
    assert (pred >= 0).all() and len(pred) == len(rul_true) == 100
    baseline = np.full(100, float(R.train_rul(train, 125).mean()))
    assert R.rmse(rul_true, pred) < 25.0                     # medido: 18.8
    assert R.rmse(rul_true, pred) < R.rmse(rul_true, baseline) * 0.6
    assert R.nasa_score(rul_true, pred) < R.nasa_score(rul_true, baseline) / 10


def test_the_exported_metrics_are_consistent_with_themselves():
    path = ROOT / "tmp_agent_b" / "rul_metrics.json"
    if not path.exists():
        pytest.skip("se genera con `python -m src.models.rul_benchmark`")
    data = json.loads(path.read_text(encoding="utf-8"))
    for name, s in data["subsets"].items():
        pick = min(s["models"], key=lambda k: s["models"][k]["cv_rmse_rows"])
        assert s["selected_by_cv"] == pick and s["selected_test"] == s["models"][pick]["test"], name
        for m in s["models"].values():
            assert m["test"]["rmse"] < s["baseline_constant"]["rmse"] and m["test"]["nasa_score"] >= 0
