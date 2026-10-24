"""GRU sobre el historial de lecturas de SCANIA: rezagos causales, relleno y aprendizaje en datos sinteticos."""
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

from src.models import nets
from src.models import sequence_scania as SQ


def history(n_vehicles=3, n_reads=6):
    rows = [{"vehicle_id": v, "time_step": float(t), "a": float(10 * v + t), "b": float(t)}
            for v in range(n_vehicles) for t in range(1, n_reads + 1)]
    return pl.DataFrame(rows)


def test_lags_look_back_within_the_vehicle_and_flag_padding():
    out = SQ.add_lags(history(), ["a"], lags=3).sort(["vehicle_id", "time_step"]).to_pandas()
    first = out.iloc[0]   # primera lectura del camion 0: sin historia
    assert first["a__lag0"] == 1 and np.isnan(first["a__lag1"]) and first["__pad1"] == 1 and first["__pad0"] == 0
    third = out.iloc[2]
    assert (third["a__lag0"], third["a__lag1"], third["a__lag2"]) == (3, 2, 1) and third["__pad2"] == 0
    v1 = out[out.vehicle_id == 1].iloc[0]  # el camion 1 no hereda lecturas del camion 0
    assert v1["__pad1"] == 1 and np.isnan(v1["a__lag1"])


def test_lags_do_not_change_when_the_future_is_removed():
    full = SQ.add_lags(history(), ["a", "b"], lags=4).sort(["vehicle_id", "time_step"]).to_pandas()
    cut = SQ.add_lags(history().filter(pl.col("time_step") <= 4), ["a", "b"], lags=4).sort(["vehicle_id", "time_step"]).to_pandas()
    merged = cut.merge(full, on=["vehicle_id", "time_step"], suffixes=("", "_full"))
    for c in SQ.lag_columns(["a", "b"], 4):
        assert np.allclose(merged[c], merged[c + "_full"], equal_nan=True)


def test_sequence_order_runs_from_oldest_to_current():
    cols = SQ.lag_columns(["a"], 3)
    assert cols == ["a__lag2", "__pad2", "a__lag1", "__pad1", "a__lag0", "__pad0"]


def test_gru_learns_a_trend_that_a_single_readout_cannot_show():
    rng = np.random.default_rng(0)
    n, lags = 1500, 4
    y = rng.integers(0, 2, n)
    slope = np.where(y == 1, 1.0, -1.0)
    start = rng.normal(0, 3, n)  # el nivel actual por si solo no separa las clases; la pendiente si
    data = {f"x__lag{k}": start - slope * k + rng.normal(0, 0.3, n) for k in range(lags)}
    data.update({f"__pad{k}": np.zeros(n) for k in range(lags)})
    data["spec"] = rng.choice(["a", "b"], n)
    X = pd.DataFrame(data)
    model = SQ.GRUSequenceClassifier(2, ["x"], ["spec"], lags=lags, seed=0,
                                     train_config=nets.TrainConfig(min_epochs=5, max_epochs=15, patience=3, batch_size=128))
    model.fit(X[:1000], y[:1000], None, X[1000:1250], y[1000:1250], None)
    p = model.predict_proba(X[1250:])
    assert p.shape == (250, 2) and np.allclose(p.sum(axis=1), 1, atol=1e-5)
    assert roc_auc_score(y[1250:], p[:, 1]) > 0.9
    assert model.info["epochs_run"] >= 5


def test_time_step_can_be_one_of_the_lagged_features():
    out = SQ.add_lags(history(), ["time_step", "a"], lags=2).sort(["vehicle_id", "time_step"]).to_pandas()
    assert list(out.columns[:2]) == ["vehicle_id", "time_step"]
    assert out["time_step__lag1"].iloc[1] == 1.0
