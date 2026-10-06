"""Pruebas del cargador NASA C-MAPSS FD001 (``src/data/cmapss_loader.py``).

Las pruebas sinteticas (tablas de pocas filas con resultado calculado a mano) no necesitan red. Las que usan los
datos reales dependen del fixture ``fd001``: usa ``data/raw/cmapss`` y, si falta, lo descarga; si no hay red,
esas pruebas se omiten en vez de fallar.
"""
import io
import zipfile

import numpy as np
import polars as pl
import pytest
import requests

from src.data import cmapss_loader as C


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def fd001():
    try:
        return C.build_fd001()
    except Exception as exc:  # sin red y sin datos locales
        pytest.skip(f"C-MAPSS FD001 no disponible: {exc}")


def _toy(unit_lengths: dict[int, int], value=lambda u, c: float(c), sensors=("s2",)) -> pl.DataFrame:
    rows = [{"unit": u, "cycle": c, **{s: value(u, c) for s in sensors}} for u, n in unit_lengths.items() for c in range(1, n + 1)]
    return pl.DataFrame(rows).with_columns(pl.col("unit").cast(pl.Int32), pl.col("cycle").cast(pl.Int32))


# ----------------------------------------------------------------------------- sensores
def test_there_are_14_informative_sensors_and_7_dropped_ones():
    assert len(C.INFORMATIVE_SENSORS) == 14
    assert len(C.DROPPED_SENSORS) == 7
    assert set(C.INFORMATIVE_SENSORS) | set(C.DROPPED_SENSORS) == set(C.SENSORS)
    assert not set(C.INFORMATIVE_SENSORS) & set(C.DROPPED_SENSORS)


def test_the_informative_sensors_are_exactly_the_ones_that_vary_in_train(fd001):
    raw = C.read_split("train")
    varying = [s for s in C.SENSORS if raw[s].n_unique() > 2]
    assert varying == C.INFORMATIVE_SENSORS  # seleccion hecha solo con train
    for s in C.DROPPED_SENSORS:
        assert raw[s].n_unique() <= 2


# ----------------------------------------------------------------------------- RUL (sintetico)
def test_train_rul_counts_down_to_zero_at_the_last_cycle_and_is_capped():
    out = C.add_train_rul(_toy({1: 5, 2: 3}), max_rul=3).sort(["unit", "cycle"])
    assert out.filter(pl.col("unit") == 1)["rul_uncapped"].to_list() == [4, 3, 2, 1, 0]
    assert out.filter(pl.col("unit") == 1)["rul"].to_list() == [3, 3, 2, 1, 0]
    assert out.filter(pl.col("unit") == 2)["rul"].to_list() == [2, 1, 0]  # el techo no toca valores menores


def test_train_rul_is_computed_per_unit_not_over_the_whole_table():
    out = C.add_train_rul(_toy({1: 10, 2: 3}), max_rul=125)
    assert out.filter((pl.col("unit") == 2) & (pl.col("cycle") == 1))["rul"].item() == 2  # no 9


def test_test_rul_adds_the_remaining_cycles_to_the_value_in_the_rul_file():
    test = _toy({1: 3, 2: 2})
    out = C.add_test_rul(test, np.array([10, 4]), max_rul=125).sort(["unit", "cycle"])
    assert out.filter(pl.col("unit") == 1)["rul"].to_list() == [12, 11, 10]
    assert out.filter(pl.col("unit") == 2)["rul"].to_list() == [5, 4]


def test_test_rul_cap_applies_to_the_uncapped_value():
    out = C.add_test_rul(_toy({1: 3}), np.array([10]), max_rul=11).sort("cycle")
    assert out["rul_uncapped"].to_list() == [12, 11, 10]
    assert out["rul"].to_list() == [11, 11, 10]


def test_test_rul_requires_one_value_per_unit():
    with pytest.raises(ValueError):
        C.add_test_rul(_toy({1: 3, 2: 2}), np.array([10]))


# ----------------------------------------------------------------------------- caracteristicas (sintetico)
def test_rolling_mean_and_std_match_hand_computed_values():
    df = _toy({1: 4}, value=lambda u, c: float(c))  # sensor = 1, 2, 3, 4
    out = C.add_rolling_features(df, sensors=["s2"], windows=(3,)).sort("cycle")
    assert out["s2_mean_3"].to_list() == pytest.approx([1.0, 1.5, 2.0, 3.0])
    assert out["s2_std_3"].to_list() == pytest.approx([0.0, np.sqrt(0.5), 1.0, 1.0])  # ddof=1; 0.0 con un solo ciclo


def test_feature_names_and_count_follow_sensors_times_windows_times_two():
    names = C.feature_names(["s2", "s3"], (5, 10))
    assert names == ["s2_mean_5", "s2_std_5", "s2_mean_10", "s2_std_10", "s3_mean_5", "s3_std_5", "s3_mean_10", "s3_std_10"]
    assert len(C.feature_names()) == 14 * len(C.WINDOWS) * 2


def test_windows_never_mix_data_from_different_units():
    both = _toy({1: 6, 2: 6}, value=lambda u, c: float(c + 100 * u))
    alone = both.filter(pl.col("unit") == 2)
    a = C.add_rolling_features(both, ["s2"], (3,)).filter(pl.col("unit") == 2).sort("cycle")
    b = C.add_rolling_features(alone, ["s2"], (3,)).sort("cycle")
    assert a["s2_mean_3"].to_list() == b["s2_mean_3"].to_list()
    assert a["s2_std_3"].to_list() == b["s2_std_3"].to_list()
    first = a.filter(pl.col("cycle") == 1)
    assert first["s2_mean_3"].item() == 201.0  # el primer ciclo del motor 2 no ve al motor 1


def test_features_at_cycle_t_do_not_change_when_the_future_changes():
    base = _toy({1: 12}, value=lambda u, c: float(c * c))
    future_edited = base.with_columns(pl.when(pl.col("cycle") > 7).then(pl.col("s2") * 1000 + 5).otherwise(pl.col("s2")).alias("s2"))
    a = C.add_rolling_features(base, ["s2"], (3, 5)).filter(pl.col("cycle") <= 7).sort("cycle")
    b = C.add_rolling_features(future_edited, ["s2"], (3, 5)).filter(pl.col("cycle") <= 7).sort("cycle")
    for col in C.feature_names(["s2"], (3, 5)):
        assert a[col].to_list() == b[col].to_list()


def test_a_future_change_does_move_later_features_so_the_check_is_not_vacuous():
    base = _toy({1: 12})
    edited = base.with_columns(pl.when(pl.col("cycle") == 8).then(999.0).otherwise(pl.col("s2")).alias("s2"))
    a = C.add_rolling_features(base, ["s2"], (3,)).filter(pl.col("cycle") == 9)["s2_mean_3"].item()
    b = C.add_rolling_features(edited, ["s2"], (3,)).filter(pl.col("cycle") == 9)["s2_mean_3"].item()
    assert a != b


def test_rolling_features_reject_a_table_that_mixes_splits():
    with pytest.raises(ValueError):
        C.add_rolling_features(pl.concat([_toy({1: 3}), _toy({1: 3})]), ["s2"], (3,))


# ----------------------------------------------------------------------------- descarga (sin red)
def _fake_zip(files: dict[str, bytes]) -> bytes:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr(C.INNER_ZIP, inner.getvalue())
    return outer.getvalue()


def test_extract_writes_the_files_when_checksums_match(tmp_path):
    files = {"train_FD001.txt": b"1 1 0\n", "test_FD001.txt": b"2 2 0\n", "RUL_FD001.txt": b"7\n"}
    expected = {n: C._sha256(d) for n, d in files.items()}
    paths = C.extract_fd001(_fake_zip(files), tmp_path, expected)
    assert {p.name for p in paths} == set(files)
    assert (tmp_path / "RUL_FD001.txt").read_bytes() == b"7\n"


def test_extract_rejects_a_corrupted_file_and_writes_nothing(tmp_path):
    files = {"train_FD001.txt": b"1 1 0\n", "test_FD001.txt": b"2 2 0\n", "RUL_FD001.txt": b"7\n"}
    expected = {n: C._sha256(d) for n, d in files.items()}
    files["test_FD001.txt"] = b"tampered\n"
    with pytest.raises(RuntimeError, match="test_FD001.txt"):
        C.extract_fd001(_fake_zip(files), tmp_path, expected)
    assert not list(tmp_path.iterdir())


def test_download_is_a_no_op_when_the_files_are_already_valid(fd001, monkeypatch):
    monkeypatch.setattr(C, "_fetch", lambda *a, **k: pytest.fail("no debia tocar la red"))
    assert C.is_downloaded()
    assert [p.name for p in C.download()] == list(C.FILES_SHA256)


def test_download_redownloads_when_a_file_is_missing_and_checks_the_outer_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "_fetch", lambda *a, **k: b"not the nasa zip")
    with pytest.raises(RuntimeError, match="checksum"):
        C.download(tmp_path)


def test_fetch_gives_up_after_the_configured_retries(monkeypatch):
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise requests.ConnectionError("sin red")

    monkeypatch.setattr(C.requests, "get", boom)
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="No se pudo descargar"):
        C._fetch("http://example.invalid", retries=3)
    assert len(calls) == 3


def test_read_split_rejects_unknown_splits():
    with pytest.raises(ValueError):
        C.read_split("validation")


# ----------------------------------------------------------------------------- dataset real: dimensiones
def test_dimensions_match_the_published_dataset(fd001):
    assert fd001.train.height == C.EXPECTED_ROWS["train"]
    assert fd001.test.height == C.EXPECTED_ROWS["test"]
    assert fd001.train["unit"].n_unique() == fd001.test["unit"].n_unique() == C.EXPECTED_UNITS
    assert fd001.test_last.height == C.EXPECTED_UNITS


def test_every_table_has_the_same_columns_and_the_expected_count(fd001):
    expected = 2 + len(C.SETTINGS) + 14 + 2 + 14 * len(fd001.windows) * 2 + 1  # claves, ajustes, sensores, 2 RUL, caracteristicas, split
    assert fd001.train.columns == fd001.test.columns == fd001.test_last.columns
    assert len(fd001.train.columns) == expected
    assert len(fd001.feature_columns) == 56 and len(set(fd001.feature_columns)) == 56


def test_train_and_test_share_the_same_schema_so_they_can_be_stacked(fd001):
    assert fd001.train.schema == fd001.test.schema == fd001.test_last.schema
    assert fd001.train["rul"].dtype == fd001.test["rul"].dtype


def test_there_are_no_nulls_or_non_finite_values_in_the_model_inputs(fd001):
    cols = fd001.sensor_columns + fd001.feature_columns
    for frame in (fd001.train, fd001.test):
        arr = frame.select(cols).to_numpy()
        assert np.isfinite(arr).all()
        assert frame.select(pl.all().null_count()).sum_horizontal().item() == 0


def test_cycles_are_contiguous_from_one_and_sorted_within_each_unit(fd001):
    for frame in (fd001.train, fd001.test):
        per_unit = frame.group_by("unit").agg(pl.col("cycle").min().alias("lo"), pl.col("cycle").max().alias("hi"), pl.len().alias("n"))
        assert (per_unit["lo"] == 1).all() and (per_unit["hi"] == per_unit["n"]).all()
        assert frame.equals(frame.sort(["unit", "cycle"]))


def test_test_engines_are_cut_before_failure_so_they_are_shorter_than_train_on_average(fd001):
    assert fd001.test.group_by("unit").len()["len"].mean() < fd001.train.group_by("unit").len()["len"].mean()


# ----------------------------------------------------------------------------- dataset real: RUL
def test_train_rul_is_zero_at_the_last_cycle_of_every_unit(fd001):
    last = fd001.train.filter(pl.col("cycle") == pl.col("cycle").max().over("unit"))
    assert (last["rul"] == 0).all() and last.height == C.EXPECTED_UNITS


def test_train_rul_equals_cycles_to_failure_below_the_cap_and_never_exceeds_it(fd001):
    df = fd001.train.with_columns((pl.col("cycle").max().over("unit") - pl.col("cycle")).alias("expected"))
    assert (df["rul_uncapped"] == df["expected"]).all()
    assert df["rul"].max() == fd001.max_rul == C.MAX_RUL
    assert (df["rul"] == df["expected"].clip(upper_bound=fd001.max_rul)).all()
    below = df.filter(pl.col("expected") < fd001.max_rul)
    assert (below["rul"] == below["expected"]).all()


def test_train_rul_decreases_by_exactly_one_per_cycle_until_it_leaves_the_cap(fd001):
    for unit, g in fd001.train.group_by("unit"):
        diffs = g.sort("cycle")["rul_uncapped"].diff().drop_nulls()
        assert (diffs == -1).all()


def test_the_cap_binds_on_the_early_life_of_every_train_unit(fd001):
    cap_share = fd001.train.group_by("unit").agg((pl.col("rul") == fd001.max_rul).sum().alias("n"))
    assert (cap_share["n"] > 0).all()  # el motor mas corto tiene 128 ciclos > 125


def test_a_different_cap_changes_only_the_capped_target(fd001):
    other = C.build_fd001(max_rul=130, auto_download=False)
    assert other.train["rul"].max() == 130
    assert (other.train["rul_uncapped"] == fd001.train["rul_uncapped"]).all()
    assert other.train.select(fd001.feature_columns).equals(fd001.train.select(fd001.feature_columns))


def test_test_rul_at_the_last_cycle_matches_the_rul_file_up_to_the_cap(fd001):
    expected = np.minimum(C.read_test_rul(), fd001.max_rul)
    got = fd001.test_last.sort("unit")["rul"].to_numpy()
    assert (got == expected).all()
    uncapped = fd001.test_last.sort("unit")["rul_uncapped"].to_numpy()
    assert (uncapped == C.read_test_rul()).all()


def test_test_rul_decreases_by_one_per_cycle_within_each_unit(fd001):
    for unit, g in fd001.test.group_by("unit"):
        assert (g.sort("cycle")["rul_uncapped"].diff().drop_nulls() == -1).all()


def test_test_last_holds_the_final_cycle_of_each_test_unit(fd001):
    maxima = fd001.test.group_by("unit").agg(pl.col("cycle").max().alias("c")).sort("unit")
    last = fd001.test_last.sort("unit")
    assert (last["cycle"] == maxima["c"]).all()


# ----------------------------------------------------------------------------- dataset real: fuga de datos
def test_train_and_test_rows_carry_their_split_because_unit_ids_overlap(fd001):
    assert set(fd001.train["split"].unique()) == {"train"} and set(fd001.test["split"].unique()) == {"test"}
    assert set(fd001.train["unit"].to_list()) == set(fd001.test["unit"].to_list())  # mismos ids 1-100, motores distintos
    stacked = pl.concat([fd001.train, fd001.test])
    assert stacked.select(pl.struct("split", "unit", "cycle").is_duplicated().any()).item() is False


def test_train_and_test_are_different_engines_not_copies_of_each_other(fd001):
    a = fd001.train.filter(pl.col("unit") == 1).sort("cycle").head(30)["s2"].to_numpy()
    b = fd001.test.filter(pl.col("unit") == 1).sort("cycle").head(30)["s2"].to_numpy()
    assert not np.array_equal(a, b)


def test_test_features_are_computed_from_test_data_alone(fd001):
    raw_test = C.read_split("test").select(["unit", "cycle", *C.SETTINGS, *C.INFORMATIVE_SENSORS])
    redo = C.add_rolling_features(raw_test, windows=fd001.windows).sort(["unit", "cycle"])
    assert np.allclose(redo.select(fd001.feature_columns).to_numpy(), fd001.test.sort(["unit", "cycle"]).select(fd001.feature_columns).to_numpy(), rtol=0, atol=1e-12)


def test_train_features_do_not_depend_on_the_test_set(fd001):
    raw_train = C.read_split("train").select(["unit", "cycle", *C.SETTINGS, *C.INFORMATIVE_SENSORS])
    redo = C.add_rolling_features(raw_train, windows=fd001.windows).sort(["unit", "cycle"])
    assert np.allclose(redo.select(fd001.feature_columns).to_numpy(), fd001.train.select(fd001.feature_columns).to_numpy(), rtol=0, atol=1e-12)


def test_real_features_are_causal(fd001):
    assert C.check_causality(fd001)


def test_the_target_never_enters_the_features(fd001):
    assert not any(c.startswith("rul") for c in fd001.feature_columns + fd001.sensor_columns)
    for col in fd001.feature_columns[:12]:
        assert not np.array_equal(fd001.train[col].to_numpy(), fd001.train["rul"].to_numpy())


def test_first_cycle_features_use_only_that_cycle(fd001):
    first = fd001.train.filter(pl.col("cycle") == 1)
    for s in fd001.sensor_columns:
        assert np.allclose(first[f"{s}_mean_{fd001.windows[0]}"].to_numpy(), first[s].to_numpy())
        assert (first[f"{s}_std_{fd001.windows[0]}"] == 0.0).all()


def test_the_summary_reports_what_the_tables_contain(fd001):
    s = C.summarize(fd001)
    assert s["rows"] == {"train": 20631, "test": 13096, "test_last_cycle": 100}
    assert s["features"]["n_model_inputs"] == 70 and s["nulls_in_model_inputs"] == 0
    assert s["leakage_checks"]["features_are_causal"] is True
    assert s["leakage_checks"]["rul_columns_excluded_from_features"] is True
    assert s["rul"]["train_capped"]["max"] == C.MAX_RUL
