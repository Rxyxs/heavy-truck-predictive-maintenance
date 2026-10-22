"""La pagina de GitHub Pages se genera desde los results.json y predictions.csv versionados. Sin red ni navegador."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
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
    return json.loads(site.RESULTS_JSON.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- archivos
def test_the_page_is_written_with_its_figures_in_both_languages(built):
    docs, html, _ = built
    assert (docs / "index.html").exists() and (docs / ".nojekyll").exists()
    for lang in site.LANGS:
        assert {p.name for p in (docs / "figures" / lang).glob("*.png")} == set(site.FIGURES_USED)
    assert "__DATA__" not in html


def test_every_figure_the_page_asks_for_is_published():
    used = set(re.findall(r'data-fig="([^"]+)"', site.TEMPLATE.read_text(encoding="utf-8")))
    assert used == set(site.FIGURES_USED)


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
@pytest.mark.parametrize("field", ["l", "r"])
def test_every_truck_has_five_probabilities_that_sum_to_one(built, field):
    _, _, data = built
    for split in site.SPLITS:
        y, p = site.decode(data, split, field)
        assert len(y) == len(p) and p.shape[1] == 5
        assert np.abs(p.sum(axis=1) - 1.0).max() < 5e-4
        assert set(y.tolist()) <= {0, 1, 2, 3, 4}


def test_the_row_count_mismatch_is_rejected(results):
    preds = pd.read_csv(site.PREDICTIONS_CSV)
    with pytest.raises(ValueError, match="filas"):
        site.build_data(results, preds.iloc[:-5])


@pytest.mark.parametrize("split", ["validation", "test"])
@pytest.mark.parametrize("field,key", [("l", "lightgbm"), ("r", "lightgbm_escala_cruda")])
def test_the_embedded_data_reproduces_the_official_costs(built, results, split, field, key):
    _, _, data = built
    y, p = site.decode(data, split, field)
    out = K.saving_at(y, p, 1.0)
    model = results["modelos"][key][split]
    assert out["costo_politica"] == model["costo"]
    assert out["costo_nada"] == model["costo_no_hacer_nada"]


def test_the_selected_model_file_reproduces_the_reported_costs(results):
    preds = pd.read_csv(site.PREDICTIONS_CSV)
    for split in site.SPLITS:
        d = preds[preds["split"] == split]
        out = K.saving_at(d["class_label"].to_numpy(), d[[f"p{k}" for k in range(5)]].to_numpy(), 1.0)
        assert out["costo_politica"] == results["clasificador"][split]["costo_esperado_minimo"]["costo_total"]


@pytest.mark.parametrize("split", ["validation", "test"])
@pytest.mark.parametrize("field,key", [("l", "lightgbm"), ("r", "lightgbm_escala_cruda")])
def test_the_page_saving_curves_match_the_pipeline(built, results, split, field, key):
    _, _, data = built
    y, p = site.decode(data, split, field)
    model = results["modelos"][key][split]
    for row in model["sensibilidad"]:
        assert K.saving_at(y, p, row["k"])["ahorro_pct"] == pytest.approx(row["ahorro_pct"], abs=0.05), (split, field, row["k"])


def test_key_numbers_come_from_the_results_file(built, results):
    _, _, data = built
    k, sel = data["key"], results["modelos"][results["seleccionado"]]
    assert k["val_saving"] == sel["validation"]["ahorro_pct"] and k["tes_saving"] == sel["test"]["ahorro_pct"]
    assert k["val_be"] == round(sel["validation"]["costo_visita_equilibrio"], 2)
    assert k["tes_be_raw"] == round(results["modelos"]["lightgbm_escala_cruda"]["test"]["costo_visita_equilibrio"], 2)
    assert [m["key"] for m in data["models"]][-1] == "lightgbm_escala_cruda"


def test_calibration_tables_cover_every_truck_for_both_models(built):
    _, _, data = built
    for split in site.SPLITS:
        for model in ("lgb", "raw"):
            assert sum(r["n"] for r in data["calibration"][split][model]) == len(data["pred"][split]["y"])


# ------------------------------------------------------------- coherencia con el README
def test_page_numbers_appear_in_the_readme(built):
    _, _, data = built
    k = data["key"]
    for readme in ("README.md", "README.es.md"):
        text = (ROOT / readme).read_text(encoding="utf-8")
        sep = "," if readme == "README.es.md" else "."
        for v in (k["val_be"], k["tes_be"], k["val_be_raw"], k["tes_be_raw"], k["val_be_lgb"], k["tes_be_lgb"]):
            assert f"×{v:.2f}".replace(".", sep) in text, (readme, v)


# --------------------------------------------------------------------------- textos
def _i18n_keys(template, lang):
    start = template.index(f"  {lang}: {{")
    end = template.index("\n  }", start)  # cierre del bloque del idioma (dos espacios de sangria)
    # varias claves comparten linea ("link_repo: ..., link_readme: ..."): se aceptan al inicio o tras una coma
    return set(re.findall(r'(?:^\s{4}|,\s)([a-z0-9_]+):\s*["\[{]', template[start:end], re.M))


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
