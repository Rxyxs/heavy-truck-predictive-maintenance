"""Figuras de resultados de los cuatro datasets, en ingles y en espanol.

Lee los ``results.json`` de ``reports/<dataset>/`` (y las predicciones versionadas) y escribe
``reports/figures/en/*.png`` y ``reports/figures/es/*.png``. Una figura cuyo insumo falta se omite.

    python -m src.models.make_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PROCESSED = ROOT / "data" / "processed"

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GRAY, BLUE, ORANGE, AQUA = "#9a9994", "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#e8f0fb", "#9cc0ef", "#2a78d6", "#123f78"])

T = {
    "en": {
        "model": {"logistic": "Logistic regression", "random_forest": "Random forest", "lightgbm": "LightGBM",
                  "xgboost": "XGBoost", "catboost": "CatBoost", "mlp": "MLP (PyTorch)", "ensemble": "Ensemble",
                  "lightgbm_escala_cruda": "LightGBM, raw-scale features", "ridge": "Ridge", "cnn": "1D-CNN (PyTorch)",
                  "gru": "GRU (PyTorch)"},
        "val": "Validation", "test": "Test", "selected": "selected",
        "scale_title": "Each variable is rescaled by a different factor in each split",
        "scale_x": "Median rate relative to the training split (log scale; 1 = same scale)",
        "scale_var": "Variable", "same": "same scale as train",
        "models_title": "SCANIA Component X: saving against doing nothing, 95% bootstrap interval",
        "saving_x": "Saving in the official cost (%)",
        "calib_title": "Mean predicted risk divided by the observed failure rate (1 = calibrated)",
        "calib_x": "Predicted / observed (log scale)",
        "visit_title": "Saving as the cost of an unnecessary workshop visit grows",
        "visit_x": "Cost of a visit (multiple of the benchmark's 7–10)", "visit_y": "Saving (%)",
        "scale_inv": "Scale-invariant features", "scale_raw": "Raw-scale features",
        "conf_title": "Test set, selected model: confusion matrix (row %, count in brackets)",
        "conf_x": "Recommended class (action)", "conf_y": "True class",
        "shap_title": "What drives the time-to-failure estimate",
        "shap_x": "Mean |SHAP| on log(1 + time to failure)",
        "aps_title": "APS failures: official test cost, 95% bootstrap interval (lower is better)",
        "aps_x": "Total cost = 10 × false alarms + 500 × missed failures",
        "podium": "IDA 2016 winner",
        "cmapss_title": "C-MAPSS: test RMSE on the last cycle of each engine (lower is better)",
        "rmse": "RMSE (cycles)", "constant": "constant",
        "pred_title": "C-MAPSS test engines: predicted vs. true remaining life (model selected by cross-validation)",
        "true_rul": "True RUL (cycles)", "pred_rul": "Predicted RUL",
        "metro_title": "MetroPT-3: failures detected (alarm from 2 h before onset to the end) vs. chance",
        "metro_x": "Failures detected out of 4", "chance": "expected by chance", "detected": "detected",
        "fa_week": "false alarms/week",
        "det": {"lps_rule": "Low-pressure switch (rule)", "duty_zscore": "Duty-cycle z-score", "pca": "PCA residual",
                "isolation_forest": "Isolation forest", "autoencoder": "Autoencoder (PyTorch)"},
        "view": {"absolute": "absolute", "drift_adjusted": "drift-adjusted"},
        "timeline_title": "MetroPT-3, autoencoder: anomaly score over the evaluation period",
        "timeline_y": "Score / threshold (log scale)", "failure": "reported air leak", "threshold": "alarm threshold",
        "overview": ["SCANIA Component X: scale fix", "APS: test cost vs. challenge podium",
                     "C-MAPSS: RMSE vs. constant baseline", "MetroPT-3: failures detected vs. chance"],
        "before_after": ["raw-scale", "scale-invariant"], "predicted_risk": "mean predicted risk", "observed": "observed",
    },
    "es": {
        "model": {"logistic": "Regresión logística", "random_forest": "Bosque aleatorio", "lightgbm": "LightGBM",
                  "xgboost": "XGBoost", "catboost": "CatBoost", "mlp": "MLP (PyTorch)", "ensemble": "Ensamble",
                  "lightgbm_escala_cruda": "LightGBM, features en escala cruda", "ridge": "Ridge",
                  "cnn": "CNN 1D (PyTorch)", "gru": "GRU (PyTorch)"},
        "val": "Validación", "test": "Test", "selected": "elegido",
        "scale_title": "Cada variable viene reescalada por un factor distinto en cada conjunto",
        "scale_x": "Tasa mediana relativa a train (escala log; 1 = misma escala)",
        "scale_var": "Variable", "same": "misma escala que train",
        "models_title": "SCANIA Component X: ahorro frente a no hacer nada, intervalo bootstrap al 95%",
        "saving_x": "Ahorro en el costo oficial (%)",
        "calib_title": "Riesgo medio predicho dividido por la tasa de falla observada (1 = calibrado)",
        "calib_x": "Predicho / observado (escala log)",
        "visit_title": "Ahorro a medida que sube el costo de una visita innecesaria al taller",
        "visit_x": "Costo de una visita (múltiplo del 7–10 del reto)", "visit_y": "Ahorro (%)",
        "scale_inv": "Features invariantes a la escala", "scale_raw": "Features en escala cruda",
        "conf_title": "Test, modelo elegido: matriz de confusión (% por fila, conteo entre paréntesis)",
        "conf_x": "Clase recomendada (acción)", "conf_y": "Clase real",
        "shap_title": "Qué mueve la estimación del tiempo hasta la falla",
        "shap_x": "|SHAP| medio sobre log(1 + tiempo restante)",
        "aps_title": "Fallas del APS: costo oficial en test, intervalo bootstrap al 95% (menor es mejor)",
        "aps_x": "Costo total = 10 × falsas alarmas + 500 × fallas no detectadas",
        "podium": "ganador IDA 2016",
        "cmapss_title": "C-MAPSS: RMSE en test sobre el último ciclo de cada motor (menor es mejor)",
        "rmse": "RMSE (ciclos)", "constant": "constante",
        "pred_title": "Motores de test de C-MAPSS: vida restante predicha vs. real (modelo elegido por validación cruzada)",
        "true_rul": "RUL real (ciclos)", "pred_rul": "RUL predicha",
        "metro_title": "MetroPT-3: fallas detectadas (alarma desde 2 h antes del inicio hasta el fin) vs. azar",
        "metro_x": "Fallas detectadas de 4", "chance": "esperado por azar", "detected": "detectadas",
        "fa_week": "falsas alarmas/semana",
        "det": {"lps_rule": "Presostato de baja (regla)", "duty_zscore": "z-score del ciclo de trabajo",
                "pca": "Residuo PCA", "isolation_forest": "Isolation forest", "autoencoder": "Autoencoder (PyTorch)"},
        "view": {"absolute": "absoluta", "drift_adjusted": "ajustada por deriva"},
        "timeline_title": "MetroPT-3, autoencoder: puntaje de anomalía en el periodo de evaluación",
        "timeline_y": "Puntaje / umbral (escala log)", "failure": "fuga de aire reportada", "threshold": "umbral de alarma",
        "overview": ["SCANIA Component X: corrección de escala", "APS: costo en test vs. podio del reto",
                     "C-MAPSS: RMSE vs. constante", "MetroPT-3: fallas detectadas vs. azar"],
        "before_after": ["escala cruda", "invariante"], "predicted_risk": "riesgo medio predicho", "observed": "observado",
    },
}


# ------------------------------------------------------------------ utilidades
def _load(name: str) -> dict | None:
    path = REPORTS_DIR / name / "results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _style(ax, grid_axis: str = "x"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def _title(fig_or_ax, text, size=12):
    if hasattr(fig_or_ax, "suptitle"):
        fig_or_ax.suptitle(text, x=0.01, ha="left", color=INK, fontsize=size)
    else:
        fig_or_ax.set_title(text, loc="left", color=INK, fontsize=size)


def _save(fig, lang: str, name: str) -> Path:
    out = FIGURES_DIR / lang
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def _fmt(v: float, lang: str, decimals: int = 0) -> str:
    s = f"{v:,.{decimals}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".") if lang == "es" else s


# ------------------------------------------------------------------ SCANIA Component X
def scania_scale(r: dict, lang: str) -> Path:
    t = T[lang]
    # Una variable con tasa mediana 0 en train (la 370) no tiene factor definido: se omite.
    rows = sorted((d for d in r["escala"]["factor_por_variable"] if d["validation"] and d["test"]), key=lambda d: d["validation"])
    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=SURFACE)
    _style(ax)
    y = np.arange(len(rows))
    for i, d in enumerate(rows):
        lo, hi = sorted([d["validation"], d["test"]])
        ax.plot([lo, hi], [i, i], color=GRID, linewidth=2, zorder=1)
    ax.scatter([d["validation"] for d in rows], y, s=46, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=t["val"])
    ax.scatter([d["test"] for d in rows], y, s=46, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=t["test"])
    ax.axvline(1.0, color=INK_2, linewidth=1)
    ax.set_ylim(-0.8, len(rows) - 0.5)
    ax.text(1.03, -0.65, t["same"], color=INK_2, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5], ["×0.1", "×0.2", "×0.5", "×1", "×2", "×5"])
    ax.set_yticks(y, [f"{d['variable']} ({d['bins']})" for d in rows])
    ax.set_xlabel(t["scale_x"], color=INK_2, fontsize=9)
    ax.set_ylabel(t["scale_var"] + " (bins)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _title(ax, t["scale_title"])
    return _save(fig, lang, "scania_scale_shift.png")


SCANIA_ORDER = ["logistic", "random_forest", "lightgbm", "xgboost", "catboost", "mlp", "ensemble", "lightgbm_escala_cruda"]


def _scania_color(key: str, selected: str) -> str:
    return BLUE if key == selected else ORANGE if key.endswith("_escala_cruda") else GRAY


def scania_models(r: dict, lang: str) -> Path:
    t = T[lang]
    keys = [k for k in SCANIA_ORDER if k in r["modelos"]][::-1]
    sel = r["seleccionado"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=SURFACE, sharey=True, sharex=True)
    for ax, split in zip(axes, ("validation", "test")):
        _style(ax)
        for i, k in enumerate(keys):
            m = r["modelos"][k][split]
            c = _scania_color(k, sel)
            ax.plot(m["ahorro_ic95"], [i, i], color=c, linewidth=2, solid_capstyle="round")
            ax.scatter([m["ahorro_pct"]], [i], s=50, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.text(m["ahorro_ic95"][1] + 1.5, i, _fmt(m["ahorro_pct"], lang, 1) + "%", va="center", fontsize=8.5,
                    color=INK if k == sel else INK_2)
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.set_title(t["val"] if split == "validation" else t["test"], loc="left", color=INK_2, fontsize=10)
        ax.set_xlabel(t["saving_x"], color=INK_2, fontsize=9)
    axes[0].set_yticks(range(len(keys)), [t["model"][k] + (f" ({t['selected']})" if k == sel else "") for k in keys])
    _title(fig, t["models_title"])
    fig.tight_layout()
    return _save(fig, lang, "scania_models.png")


def scania_calibration(r: dict, lang: str) -> Path:
    t = T[lang]
    keys = [k for k in SCANIA_ORDER if k in r["modelos"]][::-1]
    sel = r["seleccionado"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4), facecolor=SURFACE)
    _style(ax)
    for i, k in enumerate(keys):
        for split, color, dy in (("validation", BLUE, 0.13), ("test", ORANGE, -0.13)):
            m = r["modelos"][k][split]
            ratio = m["riesgo_medio_predicho"] / m["tasa_observada"]
            ax.scatter([ratio], [i + dy], s=42, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label=(t["val"] if split == "validation" else t["test"]) if i == 0 else None)
    ax.axvline(1.0, color=INK_2, linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10], ["×0.5", "×1", "×2", "×5", "×10"])
    ax.set_yticks(range(len(keys)), [t["model"][k] + (f" ({t['selected']})" if k == sel else "") for k in keys])
    ax.set_xlabel(t["calib_x"], color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    _title(ax, t["calib_title"], 11)
    return _save(fig, lang, "scania_calibration.png")


def scania_visit_cost(r: dict, lang: str) -> Path | None:
    t = T[lang]
    raw, inv = "lightgbm_escala_cruda", "lightgbm"
    if raw not in r["modelos"] or "sensibilidad" not in r["modelos"][inv]["validation"]:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE, sharey=True)
    for ax, split in zip(axes, ("validation", "test")):
        _style(ax, "y")
        for key, color, label in ((inv, BLUE, "LightGBM, " + t["scale_inv"].lower()), (raw, ORANGE, "LightGBM, " + t["scale_raw"].lower())):
            rows = r["modelos"][key][split]["sensibilidad"]
            ks = [row["k"] for row in rows]
            ax.plot(ks, [row["ahorro_pct"] for row in rows], color=color, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, label=label)
        ax.axhline(0, color=INK_2, linewidth=1)
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 5, 10], ["×0.5", "×1", "×2", "×5", "×10"])
        ax.set_title(t["val"] if split == "validation" else t["test"], loc="left", color=INK_2, fontsize=10)
        ax.set_xlabel(t["visit_x"], color=INK_2, fontsize=9)
    axes[0].set_ylabel(t["visit_y"], color=INK_2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    _title(fig, t["visit_title"])
    fig.tight_layout()
    return _save(fig, lang, "scania_visit_cost.png")


def scania_confusion(r: dict, lang: str) -> Path:
    t = T[lang]
    m = np.array(r["clasificador"]["test"]["costo_esperado_minimo"]["matriz_confusion"])
    rows = m / m.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5), facecolor=SURFACE)
    ax.imshow(rows, cmap=SEQ, vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{rows[i, j]:.0%}\n({m[i, j]})", ha="center", va="center", fontsize=9,
                    color="#ffffff" if rows[i, j] > 0.5 else INK)
    ax.set_xticks(range(5), range(5))
    ax.set_yticks(range(5), range(5))
    ax.set_xlabel(t["conf_x"], color=INK_2, fontsize=9)
    ax.set_ylabel(t["conf_y"], color=INK_2, fontsize=9)
    ax.tick_params(length=0, colors=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _title(ax, t["conf_title"], 10)
    return _save(fig, lang, "scania_confusion.png")


def scania_shap(lang: str, top: int = 15) -> Path | None:
    path = REPORTS_DIR / "scania" / "shap_time_to_failure.csv"
    if not path.exists():
        return None
    t = T[lang]
    imp = pd.read_csv(path).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style(ax)
    ax.barh(imp["feature"], imp["mean_abs_shap"], color=BLUE, height=0.62)
    ax.set_xlabel(t["shap_x"], color=INK_2, fontsize=9)
    _title(ax, t["shap_title"])
    return _save(fig, lang, "scania_shap.png")


# ------------------------------------------------------------------ APS
APS_ORDER = ["logistic", "random_forest", "lightgbm", "xgboost", "catboost", "mlp", "ensemble"]


def aps_models(r: dict, lang: str) -> Path:
    t = T[lang]
    keys = [k for k in APS_ORDER if k in r["models"]][::-1]
    sel = r["selected_by_oof"]
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=SURFACE)
    _style(ax)
    for i, k in enumerate(keys):
        m = r["models"][k]["test"]
        c = BLUE if k == sel else GRAY
        ax.plot(m["cost_ci95"], [i, i], color=c, linewidth=2, solid_capstyle="round")
        ax.scatter([m["cost"]], [i], s=50, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(m["cost_ci95"][1] + 250, i, _fmt(m["cost"], lang), va="center", fontsize=8.5, color=INK if k == sel else INK_2)
    win = r["ida2016_podium"][0]["cost"]
    ax.axvline(win, color=ORANGE, linewidth=1.5)
    ax.set_ylim(-0.9, len(keys) - 0.5)
    ax.text(win + 120, -0.75, f"{t['podium']}: {_fmt(win, lang)}", color=INK_2, fontsize=8.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _fmt(v, lang)))
    ax.set_yticks(range(len(keys)), [t["model"][k] + (f" ({t['selected']})" if k == sel else "") for k in keys])
    ax.set_xlabel(t["aps_x"], color=INK_2, fontsize=9)
    _title(ax, t["aps_title"], 11)
    return _save(fig, lang, "aps_models.png")


# ------------------------------------------------------------------ C-MAPSS
CMAPSS_ORDER = ["ridge", "random_forest", "lightgbm", "xgboost", "cnn", "gru", "ensemble"]


def cmapss_models(r: dict, lang: str) -> Path:
    t = T[lang]
    subsets = list(r["subsets"])
    fig, axes = plt.subplots(1, len(subsets), figsize=(12, 4.2), facecolor=SURFACE, sharey=True)
    axes = np.atleast_1d(axes)
    keys = [k for k in CMAPSS_ORDER if any(k in r["subsets"][s]["models"] for s in subsets)][::-1]
    for ax, s in zip(axes, subsets):
        _style(ax)
        d = r["subsets"][s]
        for i, k in enumerate(keys):
            if k not in d["models"]:
                continue
            v = d["models"][k]["test"]["rmse"]
            c = BLUE if k == d["selected_by_cv"] else GRAY
            ax.barh(i, v, color=c, height=0.62)
            ax.text(v + 0.6, i, _fmt(v, lang, 1), va="center", fontsize=8, color=INK if c == BLUE else INK_2)
        ax.axvline(d["baseline_constant"]["rmse"], color=ORANGE, linewidth=1.5)
        ax.set_ylim(-1.0, len(keys) - 0.5)
        ax.text(d["baseline_constant"]["rmse"] - 1, -0.8, f"{t['constant']} {_fmt(d['baseline_constant']['rmse'], lang, 1)}",
                color=INK_2, fontsize=8, ha="right")
        ax.set_title(f"{s} ({d['conditions']} cond., {d['fault_modes']} mode{'s' if d['fault_modes'] > 1 and lang == 'en' else ''})"
                     if lang == "en" else f"{s} ({d['conditions']} cond., {d['fault_modes']} modo{'s' if d['fault_modes'] > 1 else ''})",
                     loc="left", color=INK_2, fontsize=9.5)
        ax.set_xlabel(t["rmse"], color=INK_2, fontsize=9)
    axes[0].set_yticks(range(len(keys)), [t["model"][k] for k in keys])
    _title(fig, t["cmapss_title"])
    fig.tight_layout()
    return _save(fig, lang, "cmapss_models.png")


def cmapss_predictions(r: dict, lang: str) -> Path | None:
    path = REPORTS_DIR / "cmapss" / "test_predictions.csv"
    if not path.exists():
        return None
    t = T[lang]
    preds = pd.read_csv(path)
    subsets = list(r["subsets"])
    fig, axes = plt.subplots(1, len(subsets), figsize=(12, 3.6), facecolor=SURFACE, sharex=True, sharey=True)
    for ax, s in zip(np.atleast_1d(axes), subsets):
        _style(ax, "both")
        d = preds[preds["subset"] == s]
        sel = r["subsets"][s]["selected_by_cv"]
        ax.plot([0, 200], [0, 200], color=INK_2, linewidth=1)
        ax.scatter(d["rul_true"], d[sel], s=12, color=BLUE, alpha=0.75, edgecolor="none")
        ax.set_title(f"{s}: {t['model'][sel]}", loc="left", color=INK_2, fontsize=9.5)
        ax.set_xlabel(t["true_rul"], color=INK_2, fontsize=9)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)
    np.atleast_1d(axes)[0].set_ylabel(t["pred_rul"], color=INK_2, fontsize=9)
    _title(fig, t["pred_title"], 11)
    fig.tight_layout()
    return _save(fig, lang, "cmapss_predictions.png")


# ------------------------------------------------------------------ MetroPT-3
def _detector_label(key: str, d: dict, lang: str) -> str:
    t = T[lang]
    name = t["det"][d["detector"]]
    return name if d["detector"] == "lps_rule" else f"{name}, {t['view'][d['view']]}"


def metropt_detectors(r: dict, lang: str) -> Path:
    t = T[lang]
    items = list(r["detectors"].items())[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor=SURFACE)
    _style(ax)
    for i, (key, d) in enumerate(items):
        ax.plot([d["chance_detected_2h"], d["detected_2h"]], [i, i], color=GRID, linewidth=2, zorder=1)
        ax.scatter([d["chance_detected_2h"]], [i], s=40, color=GRAY, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label=t["chance"] if i == 0 else None)
        ax.scatter([d["detected_2h"]], [i], s=50, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label=t["detected"] if i == 0 else None)
        ax.text(4.25, i, f"{_fmt(d['false_alarms_per_week'], lang, 1)} {t['fa_week']}", va="center", fontsize=8.5, color=INK_2)
    ax.set_xlim(-0.2, 5.6)
    ax.set_xticks(range(5))
    ax.set_yticks(range(len(items)), [_detector_label(k, d, lang) for k, d in items])
    ax.set_xlabel(t["metro_x"], color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.4, -0.12), ncol=2)
    _title(ax, t["metro_title"], 11)
    return _save(fig, lang, "metropt_detectors.png")


def metropt_timeline(r: dict, lang: str, key: str = "autoencoder__absolute") -> Path | None:
    path = PROCESSED / "metropt_scores.parquet"
    if not path.exists():
        return None
    t = T[lang]
    s = pd.read_parquet(path)[key].dropna()
    s = s.resample("1h").max()  # las horas sin datos quedan en NaN y cortan la linea: no se dibujan puentes sobre los huecos
    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor=SURFACE)
    _style(ax, "y")
    for i, f in enumerate(r["failures"]):
        ax.axvspan(pd.Timestamp(f["start"]), pd.Timestamp(f["end"]), color=ORANGE, alpha=0.35,
                   linewidth=0, label=t["failure"] if i == 0 else None)
    ax.plot(s.index, s.clip(lower=1e-2), color=BLUE, linewidth=0.8)
    ax.axhline(1.0, color=INK_2, linewidth=1, label=t["threshold"])
    ax.set_yscale("log")
    ax.set_ylabel(t["timeline_y"], color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _title(ax, t["timeline_title"], 11)
    return _save(fig, lang, "metropt_timeline.png")


# ------------------------------------------------------------------ resumen
def overview(sc: dict | None, aps: dict | None, cm: dict | None, mp: dict | None, lang: str) -> Path | None:
    if not all((sc, aps, cm, mp)):
        return None
    t = T[lang]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), facecolor=SURFACE)
    (a, b), (c, d) = axes

    # SCANIA: riesgo medio predicho del mismo LightGBM (escala cruda vs invariante) contra la tasa observada, en test
    _style(a, "y")
    raw, sel = sc["modelos"]["lightgbm_escala_cruda"]["test"], sc["modelos"]["lightgbm"]["test"]
    vals = [raw["riesgo_medio_predicho"] * 100, sel["riesgo_medio_predicho"] * 100]
    a.bar([0, 1], vals, color=[ORANGE, BLUE], width=0.55)
    a.axhline(sel["tasa_observada"] * 100, color=INK_2, linewidth=1)
    a.text(0.5, sel["tasa_observada"] * 100 + 0.1, f"{t['observed']} {_fmt(sel['tasa_observada'] * 100, lang, 1)}%",
           va="bottom", ha="center", fontsize=8.5, color=INK_2)
    for x, v in zip([0, 1], vals):
        a.text(x, v + 0.2, f"{_fmt(v, lang, 1)}%", ha="center", fontsize=9, color=INK)
    a.set_xticks([0, 1], t["before_after"])
    a.set_ylabel("LightGBM, " + t["predicted_risk"] + " (%)", color=INK_2, fontsize=9)
    _title(a, t["overview"][0], 10.5)

    # APS: costo en test del elegido y del podio
    _style(b, "y")
    labels = [t["model"][aps["selected_by_oof"]], "IDA 1", "IDA 2", "IDA 3"]
    vals = [aps["models"][aps["selected_by_oof"]]["test"]["cost"]] + [p["cost"] for p in aps["ida2016_podium"]]
    b.bar(range(4), vals, color=[BLUE, GRAY, GRAY, GRAY], width=0.55)
    ci = aps["models"][aps["selected_by_oof"]]["test"]["cost_ci95"]
    b.plot([0, 0], ci, color=INK_2, linewidth=1.5)
    for x, v in enumerate(vals):
        b.text(x + 0.05, v + 250, _fmt(v, lang), ha="left", fontsize=8.5, color=INK if x == 0 else INK_2)
    b.set_xticks(range(4), labels)
    b.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _fmt(v, lang)))
    _title(b, t["overview"][1], 10.5)

    # C-MAPSS: RMSE del elegido vs constante por subconjunto
    _style(c, "y")
    subsets = list(cm["subsets"])
    x = np.arange(len(subsets))
    base = [cm["subsets"][s]["baseline_constant"]["rmse"] for s in subsets]
    best = [cm["subsets"][s]["selected_test"]["rmse"] for s in subsets]
    c.bar(x - 0.18, base, width=0.34, color=GRAY, label=t["constant"])
    c.bar(x + 0.18, best, width=0.34, color=BLUE, label=t["selected"])
    for xi, v in zip(x, best):
        c.text(xi + 0.18, v + 0.8, _fmt(v, lang, 1), ha="center", fontsize=8.5, color=INK)
    c.set_xticks(x, subsets)
    c.set_ylabel(t["rmse"], color=INK_2, fontsize=9)
    c.legend(frameon=False, fontsize=8.5, loc="upper left")
    _title(c, t["overview"][2], 10.5)

    # MetroPT: detectadas vs azar, vistas absolutas
    _style(d, "y")
    keys = [k for k in mp["detectors"] if k == "lps_rule" or k.endswith("__absolute")]
    x = np.arange(len(keys))
    d.bar(x - 0.18, [mp["detectors"][k]["chance_detected_2h"] for k in keys], width=0.34, color=GRAY, label=t["chance"])
    d.bar(x + 0.18, [mp["detectors"][k]["detected_2h"] for k in keys], width=0.34, color=BLUE, label=t["detected"])
    d.set_xticks(x, [t["det"][mp["detectors"][k]["detector"]].split(" (")[0].replace(" ", "\n", 1) for k in keys], fontsize=7.5)
    d.set_yticks(range(5))
    d.legend(frameon=False, fontsize=8.5, loc="upper left")
    _title(d, t["overview"][3], 10.5)
    fig.tight_layout(h_pad=2.5)
    return _save(fig, lang, "overview.png")


def make_all() -> list[Path]:
    sc, aps, cm, mp = _load("scania"), _load("aps"), _load("cmapss"), _load("metropt")
    out = []
    for lang in ("en", "es"):
        if sc:
            out += [scania_scale(sc, lang), scania_models(sc, lang), scania_calibration(sc, lang),
                    scania_visit_cost(sc, lang), scania_confusion(sc, lang), scania_shap(lang)]
        if aps:
            out.append(aps_models(aps, lang))
        if cm:
            out += [cmapss_models(cm, lang), cmapss_predictions(cm, lang)]
        if mp:
            out += [metropt_detectors(mp, lang), metropt_timeline(mp, lang)]
        out.append(overview(sc, aps, cm, mp, lang))
    from src.models.analysis_figures import make_all_analysis

    out += make_all_analysis()
    return [p for p in out if p is not None]


if __name__ == "__main__":
    for p in make_all():
        print(p.relative_to(ROOT))
