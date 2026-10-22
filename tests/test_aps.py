"""APS Failure at Scania Trucks: lectura, variables y metrica de costo. Sin red salvo la prueba con datos reales."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import aps_loader as A

ROOT = Path(__file__).resolve().parents[1]
LICENSE_HEADER = "\n".join(f"license line {i}" for i in range(20)) + "\n"


def write_csv(path: Path, rows: list[str]) -> None:
    path.write_text(LICENSE_HEADER + "class,aa_000,ag_000,ag_001,ag_002,cd_000\n" + "\n".join(rows) + "\n", encoding="latin-1")


@pytest.fixture
def toy_dir(tmp_path):
    write_csv(tmp_path / "aps_failure_training_set.csv", ["neg,10,1,2,7,1209600", "pos,20,na,na,na,1209600", "neg,na,0,0,0,1209600"])
    write_csv(tmp_path / "aps_failure_test_set.csv", ["pos,5,1,1,2,1209600"])
    return tmp_path


def test_reads_past_the_license_header_and_parses_na(toy_dir):
    X, y = A.read_split("train", toy_dir)
    assert y.tolist() == [0, 1, 0]
    assert X.shape == (3, 5) and np.isnan(X.loc[1, "ag_000"]) and np.isnan(X.loc[2, "aa_000"])


def test_unknown_split_fails():
    with pytest.raises(ValueError):
        A.read_split("validation")


def test_engineered_histogram_fractions_sum_to_one_and_constant_column_is_dropped(toy_dir):
    X, _ = A.read_split("train", toy_dir)
    F = A.engineer(X)
    assert "cd_000" not in F.columns
    fracs = F[["ag_000_frac", "ag_001_frac", "ag_002_frac"]]
    assert fracs.iloc[0].sum() == pytest.approx(1.0) and fracs.iloc[0]["ag_002_frac"] == pytest.approx(0.7)
    assert fracs.iloc[1].isna().all()          # histograma ausente
    assert fracs.iloc[2].isna().all()          # histograma en cero: fraccion indefinida, no 0/0
    assert F["n_missing"].tolist() == [0, 3, 1]


def test_engineering_is_row_wise_so_test_rows_cannot_leak_into_train(toy_dir):
    X, _ = A.read_split("train", toy_dir)
    alone = A.engineer(X.iloc[[0]])
    together = A.engineer(pd.concat([X.iloc[[0]], X.iloc[[1]] * 1000]))
    pd.testing.assert_frame_equal(alone, together.iloc[[0]])


def test_total_cost_by_hand():
    y = np.array([0, 0, 1, 1, 0])
    pred = np.array([1, 0, 0, 1, 1])
    assert A.total_cost(y, pred) == 2 * 10 + 1 * 500


def test_the_published_podium_is_consistent_with_the_cost_metric():
    for row in A.IDA2016_PODIUM:
        assert row["cost"] == 10 * row["false_alarms"] + 500 * row["missed"]


@pytest.mark.skipif(not A.is_downloaded(), reason="APS no descargado (data/raw/aps)")
def test_real_files_have_the_documented_size_and_class_balance():
    for split, (rows, positives) in A.EXPECTED.items():
        X, y = A.read_split(split)
        assert (len(X), int(y.sum())) == (rows, positives) and X.shape[1] == 170


def test_reported_results_are_internally_consistent():
    path = ROOT / "reports" / "aps" / "results.json"
    if not path.exists():
        pytest.skip("se genera con `python -m src.models.aps_pipeline`")
    r = json.loads(path.read_text(encoding="utf-8"))
    assert r["selected_by_oof"] == min(r["models"], key=lambda m: r["models"][m]["oof"]["cost"])
    for m in r["models"].values():
        t = m["test"]
        assert t["cost"] == 10 * t["false_alarms"] + 500 * t["missed"]
        assert t["cost_ci95"][0] <= t["cost"] <= t["cost_ci95"][1]
        assert m["not_valid_threshold_tuned_on_test"]["cost"] <= t["cost"]  # ajustar en test siempre luce mejor
        assert t["cost"] < r["baselines"]["alarm_everyone"] < r["baselines"]["no_alarms"]
