"""Las cifras principales de ambos README salen de los results.json versionados (y no al reves)."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
READMES = {"en": ROOT / "README.md", "es": ROOT / "README.es.md"}


def load(name):
    path = ROOT / "reports" / name / "results.json"
    if not path.exists():
        pytest.skip(f"falta {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(v, decimals, lang):
    s = f"{v:,.{decimals}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".") if lang == "es" else s


@pytest.fixture(scope="module", params=["en", "es"])
def readme(request):
    return request.param, READMES[request.param].read_text(encoding="utf-8")


def test_no_placeholder_is_left(readme):
    lang, text = readme
    assert "CMAPSS_" not in text and "MLP_" not in text


def test_scania_selected_model_row(readme):
    lang, text = readme
    r = load("scania")
    m = r["modelos"][r["seleccionado"]]
    for split in ("validation", "test"):
        s = m[split]
        expected = f"{fmt(s['ahorro_pct'], 1, lang)}% ({fmt(s['ahorro_ic95'][0], 1, lang)}–{fmt(s['ahorro_ic95'][1], 1, lang)})"
        assert expected in text, expected
        assert f"×{fmt(s['costo_visita_equilibrio'], 2, lang)}" in text


def test_scania_every_model_row_matches_results(readme):
    lang, text = readme
    r = load("scania")
    for key, m in r["modelos"].items():
        assert fmt(m["seleccion"]["costo_por_1000_camiones"], 0, lang) in text, key
        for split in ("validation", "test"):
            s = m[split]
            row = f"{fmt(s['ahorro_pct'], 1, lang)}% ({fmt(s['ahorro_ic95'][0], 1, lang)}–{fmt(s['ahorro_ic95'][1], 1, lang)})"
            assert row in text, (key, split, row)
        be = " / ".join(f"×{fmt(m[sp]['costo_visita_equilibrio'], 2, lang)}" for sp in ("validation", "test"))
        assert be in text, (key, be)


def test_aps_table_matches_results(readme):
    lang, text = readme
    r = load("aps")
    for key, m in r["models"].items():
        t = m["test"]
        row = f"{fmt(t['cost'], 0, lang)} ({fmt(t['cost_ci95'][0], 0, lang)}–{fmt(t['cost_ci95'][1], 0, lang)})"
        assert row in text, (key, row)
        assert fmt(m["oof"]["cost"], 0, lang) in text, key


def test_metropt_table_matches_results(readme):
    lang, text = readme
    r = load("metropt")
    for key, d in r["detectors"].items():
        assert f"{fmt(d['chance_detected_2h'], 2, lang)} | {fmt(d['false_alarms_per_week'], 1, lang)} |" in text.replace("**", ""), key


def test_cmapss_selected_rmse_per_subset(readme):
    lang, text = readme
    r = load("cmapss")
    for subset, d in r["subsets"].items():
        assert fmt(d["selected_test"]["rmse"], 1, lang) in text, subset
