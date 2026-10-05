"""La pagina de GitHub Pages se genera desde results.json y predictions.csv. Sin red ni navegador."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from src import site
from src.models import calibration as K

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = re.compile(r'<script type="application/json" id="data">(.*?)</script>', re.S)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    docs = tmp_path_factory.mktemp("docs")
    out = site.build_site(docs_dir=docs)
    html = out.read_text(encoding="utf-8")
    return docs, html, json.loads(PAYLOAD.search(html).group(1).replace("<\\/", "</"))


@pytest.fixture(scope="module")
def results():
    return json.loads((ROOT / "reports" / "results.json").read_text(encoding="utf-8"))["clasificador"]


def decode(data, split):
    y = np.array(data["pred"][split]["y"])
    p = np.array(data["pred"][split]["p"], dtype=float).reshape(-1, 5) / site.PROB_SCALE
    return y, p


# --------------------------------------------------------------------------- archivos
def test_the_page_is_written_with_its_assets(built):
    docs, html, _ = built
    assert (docs / "index.html").exists() and (docs / ".nojekyll").exists()
    assert {p.name for p in (docs / "figures").glob("*.png")} == set(site.FIGURES_USED)
    assert "__DATA__" not in html


def test_embedded_data_is_strict_json(built):
    _, html, _ = built
    raw = PAYLOAD.search(html).group(1)
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw.replace("<\\/", "</"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def test_a_closing_script_tag_inside_the_data_cannot_break_the_page():
    html = site.render({"x": "</script><script>alert(1)</script>"})
    payload = PAYLOAD.search(html).group(1)
    assert "</" not in payload and "<\\/script>" in payload


def test_the_page_has_no_external_scripts_or_styles():
    html = site.TEMPLATE.read_text(encoding="utf-8")
    assert not re.search(r'<script[^>]+src="https?://', html)
    assert not re.search(r'<link[^>]+href="https?://', html)


# ---------------------------------------------------------------- datos incrustados
def test_every_truck_has_five_probabilities_that_sum_to_one(built):
    _, _, data = built
    for split in site.SPLITS:
        y, p = decode(data, split)
        assert len(y) == len(p) and p.shape[1] == 5
        assert np.abs(p.sum(axis=1) - 1.0).max() < 5e-4
        assert set(y.tolist()) <= {0, 1, 2, 3, 4}


def test_the_row_count_matches_results_json(built, results):
    _, _, data = built
    for split in site.SPLITS:
        assert len(data["pred"][split]["y"]) == results[split]["n_vehiculos"]


def test_the_row_count_mismatch_is_rejected(results):
    import pandas as pd

    preds = pd.read_csv(site.PREDICTIONS_CSV)
    with pytest.raises(ValueError, match="filas"):
        site.build_data({"clasificador": results, "datos": {}, "tiempo_restante": {}}, preds.iloc[:-5])


# ----------------------------------------------- la logica de la pagina, en Python
def page_saving(data, split, k, recal):
    """Reimplementacion en Python de lo que la pagina calcula en el navegador, sobre los datos incrustados."""
    y, p = decode(data, split)
    if recal:
        other = "validation" if split == "test" else "test"
        p = K.apply_platt(p, data["platt"][other]["a"], data["platt"][other]["b"])
    return K.saving_at(y, p, k)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_the_embedded_data_reproduces_the_official_costs(built, results, split):
    _, _, data = built
    out = page_saving(data, split, 1.0, recal=False)
    assert out["costo_politica"] == results[split]["costo_esperado_minimo"]["costo_total"]
    assert out["costo_nada"] == results[split]["siempre_sano"]["costo_total"]


@pytest.mark.parametrize("split", ["validation", "test"])
def test_the_page_recalibrated_saving_matches_the_pipeline(built, results, split):
    _, _, data = built
    reported = results[split]["recalibrado_con_el_otro_conjunto"]["sensibilidad_costo_visita"]
    for row in reported:
        mine = page_saving(data, split, row["k"], recal=True)["ahorro_pct"]
        assert mine == pytest.approx(row["ahorro_pct"], abs=0.3), (split, row["k"])


@pytest.mark.parametrize("split", ["validation", "test"])
def test_the_page_raw_saving_curve_matches_the_pipeline(built, results, split):
    _, _, data = built
    for row in results[split]["sensibilidad_costo_visita"]:
        assert page_saving(data, split, row["k"], recal=False)["ahorro_pct"] == pytest.approx(row["ahorro_pct"], abs=0.05)


def test_key_numbers_come_from_the_results_file(built, results):
    _, _, data = built
    k = data["key"]
    assert k["cost_val"] == results["validation"]["costo_esperado_minimo"]["costo_total"]
    assert k["cost_test"] == results["test"]["costo_esperado_minimo"]["costo_total"]
    assert k["saving_val"] == pytest.approx(100 * (1 - 41141 / 57400), abs=0.01)
    assert k["be_val"] == round(results["validation"]["costo_visita_equilibrio"], 2)
    assert k["recal_saving_test"] == pytest.approx(
        next(r["ahorro_pct"] for r in results["test"]["recalibrado_con_el_otro_conjunto"]["sensibilidad_costo_visita"] if r["k"] == 1.0), abs=0.01
    )


def test_calibration_table_is_embedded_for_both_sets(built):
    _, _, data = built
    for split in site.SPLITS:
        rows = data["calibration"][split]
        assert sum(r["n"] for r in rows) == len(data["pred"][split]["y"])
        assert data["cal_summary"][split]["pred"] > 2 * data["cal_summary"][split]["obs"]


# ------------------------------------------------------------- coherencia con el README
def test_page_figures_appear_in_the_readme(built):
    _, _, data = built
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    k = data["key"]
    for text in (f"{k['cost_val']:,}", f"{k['cost_test']:,}", f"{k['be_val']:.2f}", f"{k['be_test']:.2f}",
                 f"{k['recal_be_val']:.2f}", f"{k['recal_be_test']:.2f}"):
        assert text in readme, text


# --------------------------------------------------------------------------- textos
def _i18n_keys(template, lang):
    start = template.index(f"  {lang}: {{")
    end = template.index("\n  }", start)  # cierre del bloque del idioma (dos espacios de sangria)
    # varias claves comparten linea ("link_repo: ..., link_readme: ..."): se aceptan al inicio o tras una coma
    return set(re.findall(r'(?:^\s{4}|,\s)([a-z0-9_]+):\s*["\[]', template[start:end], re.M))


def test_both_languages_define_the_same_keys():
    template = site.TEMPLATE.read_text(encoding="utf-8")
    es, en = _i18n_keys(template, "es"), _i18n_keys(template, "en")
    assert es == en, (es ^ en)


def test_every_translated_element_has_a_key_in_both_languages():
    template = site.TEMPLATE.read_text(encoding="utf-8")
    used = set(re.findall(r'data-i18n="([a-z0-9_]+)"', template))
    assert used <= _i18n_keys(template, "es") and used <= _i18n_keys(template, "en")


def test_the_page_is_accessible_by_keyboard():
    template = site.TEMPLATE.read_text(encoding="utf-8")
    assert "tabindex" in template and "ArrowLeft" in template and "aria-pressed" in template and "aria-describedby" in template
