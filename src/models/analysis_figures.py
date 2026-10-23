"""Figuras de analisis (exploracion, diagnostico y robustez) de los cuatro datasets, en ingles y espanol.

Todas salen de los ``results.json`` y CSV versionados en ``reports/``; ninguna necesita los datos crudos.
Las llama ``make_figures.make_all``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.models.make_figures import (AQUA, BLUE, GRAY, GRID, INK, INK_2, ORANGE, REPORTS_DIR, SURFACE, T,  # noqa: E402
                                     _fmt, _save, _style, _title)

A = {
    "en": {
        "km_title": "Component X survival in the training fleet (Kaplan-Meier)",
        "km_all": "All 23,550 trucks", "km_by": "By {spec} (log-rank χ² {stat})", "time": "Time units since the start of the study",
        "survival": "Share without a repair", "trucks": "trucks",
        "lead_title": "How early the model sees a failure: held-out training trucks, every readout",
        "lead_x": "Time units left before the repair", "lead_risk": "Mean predicted risk", "lead_alarm": "Share of readouts with an alarm",
        "healthy": "trucks never repaired", "healthy_note": "thin horizontal lines: trucks never repaired",
        "val_saving": "Validation saving (%)",
        "roc_title": "Any-risk discrimination on the test set", "fpr": "False positive rate", "tpr": "True positive rate",
        "recall": "Recall", "precision": "Precision", "base": "base rate",
        "cost_title": "Where the official test cost comes from", "cost_x": "Official cost on the test set",
        "fa": "false alarms", "missed": "missed failures", "window": "failure caught in the wrong window",
        "nothing": "Recommend nothing",
        "paired_title": "Saving of each model minus the selected one, same trucks resampled (95% interval)",
        "paired_x": "Difference in saving (percentage points)", "p_better": "P(better)",
        "seed_title": "Same model, same data, another seed",
        "seed_sel": "Held-out cost / 1,000 trucks (selection score)", "seed_test": "Test saving (%)",
        "cut_title": "Where the evaluated readout falls in each truck's life",
        "cut_x": "time_step of the evaluated readout", "cut_y": "Share of trucks",
        "train_random": "Training, one random readout", "validation": "Validation", "test": "Test",
        "thr_title": "APS: cost per truck as the alarm threshold moves (selected ensemble)",
        "thr_x": "Alarm threshold on the predicted probability (log scale)", "thr_y": "Cost per truck",
        "oof": "Out-of-fold (training)", "chosen": "threshold fixed on out-of-fold",
        "apr_title": "APS: precision-recall on the test set",
        "miss_title": "APS: failure rate by number of missing values per truck (training)",
        "miss_x": "Missing values in the truck's record", "miss_y": "Failure rate",
        "shap_title": "APS: which anonymised variables drive the LightGBM reference model",
        "shap_x": "Mean |SHAP| summed over the variable's columns",
        "lc_title": "C-MAPSS: validation loss of the networks (final fit; marker = best epoch restored)",
        "lc_y": "Validation MSE (scaled RUL)", "epoch": "Epoch",
        "err_title": "C-MAPSS: test error by true remaining life (selected model)",
        "err_y": "Prediction − truth (cycles)", "late": "late warning (costlier)", "early": "early warning",
        "sens_title": "C-MAPSS FD001: sensors drift only in the last ~150 cycles (training engines)",
        "sens_x": "Cycles left before failure", "sens_y": "Mean sensor value (z-score within condition)", "cap": "RUL cap (125)",
        "daily_title": "MetroPT-3: daily duty cycle and oil temperature, February-August 2020",
        "duty": "Share of the day under load", "oil": "Mean oil temperature (°C)",
        "fit": "fit", "calib": "threshold", "unreported": "29 Mar: full day under load, no report",
        "zoom_title": "MetroPT-3: the four reported air leaks (shaded) and the 36 h before them",
        "hours": "Hours from the reported start", "res": "Minimum reservoir pressure (bar)", "ae": "Autoencoder score / threshold",
        "load10": "Share of the 10 min under load",
        "leak": "Leak",
        "op_title": "MetroPT-3: detection above chance against false alarms, as the threshold moves",
        "op_x": "False alarms per week (log scale)", "op_y": "Leaks detected minus expected by chance",
        "op_note": "each point is a threshold percentile from 95% to 99.95%",
    },
    "es": {
        "km_title": "Supervivencia del componente X en la flota de entrenamiento (Kaplan-Meier)",
        "km_all": "Los 23.550 camiones", "km_by": "Por {spec} (log-rank χ² {stat})", "time": "Unidades de tiempo desde el inicio del estudio",
        "survival": "Fracción sin reparación", "trucks": "camiones",
        "lead_title": "Con cuánta anticipación ve el modelo una falla: camiones retenidos del entrenamiento, todas sus lecturas",
        "lead_x": "Unidades de tiempo que faltan para la reparación", "lead_risk": "Riesgo medio predicho",
        "lead_alarm": "Fracción de lecturas con alarma", "healthy": "camiones nunca reparados",
        "healthy_note": "líneas finas horizontales: camiones nunca reparados", "val_saving": "Ahorro en validación (%)",
        "roc_title": "Discriminación de «algún riesgo» en test", "fpr": "Tasa de falsos positivos", "tpr": "Tasa de verdaderos positivos",
        "recall": "Recall", "precision": "Precisión", "base": "tasa base",
        "cost_title": "De dónde sale el costo oficial en test", "cost_x": "Costo oficial en test",
        "fa": "falsas alarmas", "missed": "fallas no detectadas", "window": "falla detectada en la ventana equivocada",
        "nothing": "No recomendar nada",
        "paired_title": "Ahorro de cada modelo menos el del elegido, mismos camiones remuestreados (intervalo 95%)",
        "paired_x": "Diferencia de ahorro (puntos porcentuales)", "p_better": "P(mejor)",
        "seed_title": "Mismo modelo, mismos datos, otra semilla",
        "seed_sel": "Costo retenidos / 1.000 camiones (puntaje de selección)", "seed_test": "Ahorro en test (%)",
        "cut_title": "Dónde cae la lectura evaluada en la vida de cada camión",
        "cut_x": "time_step de la lectura evaluada", "cut_y": "Fracción de camiones",
        "train_random": "Entrenamiento, una lectura al azar", "validation": "Validación", "test": "Test",
        "thr_title": "APS: costo por camión a medida que se mueve el umbral de alarma (ensamble elegido)",
        "thr_x": "Umbral de alarma sobre la probabilidad predicha (escala log)", "thr_y": "Costo por camión",
        "oof": "Fuera de partición (entrenamiento)", "chosen": "umbral fijado fuera de partición",
        "apr_title": "APS: precisión-recall en test",
        "miss_title": "APS: tasa de falla según la cantidad de valores faltantes del camión (entrenamiento)",
        "miss_x": "Valores faltantes en el registro del camión", "miss_y": "Tasa de falla",
        "shap_title": "APS: qué variables anonimizadas mueven el LightGBM de referencia",
        "shap_x": "|SHAP| medio sumado sobre las columnas de la variable",
        "lc_title": "C-MAPSS: pérdida de validación de las redes (ajuste final; marcador = mejor época restaurada)",
        "lc_y": "MSE de validación (RUL escalada)", "epoch": "Época",
        "err_title": "C-MAPSS: error en test según la vida restante real (modelo elegido)",
        "err_y": "Predicción − verdad (ciclos)", "late": "aviso tardío (más caro)", "early": "aviso temprano",
        "sens_title": "C-MAPSS FD001: los sensores solo derivan en los últimos ~150 ciclos (motores de entrenamiento)",
        "sens_x": "Ciclos que faltan para la falla", "sens_y": "Valor medio del sensor (z-score por condición)", "cap": "techo de RUL (125)",
        "daily_title": "MetroPT-3: ciclo de trabajo diario y temperatura del aceite, febrero-agosto 2020",
        "duty": "Fracción del día en carga", "oil": "Temperatura media del aceite (°C)",
        "fit": "ajuste", "calib": "umbral", "unreported": "29 mar: día completo en carga, sin reporte",
        "zoom_title": "MetroPT-3: las cuatro fugas reportadas (sombreado) y las 36 h previas",
        "hours": "Horas desde el inicio reportado", "res": "Presión mínima del depósito (bar)", "ae": "Puntaje autoencoder / umbral",
        "load10": "Fracción de los 10 min en carga",
        "leak": "Fuga",
        "op_title": "MetroPT-3: detección sobre el azar contra falsas alarmas, moviendo el umbral",
        "op_x": "Falsas alarmas por semana (escala log)", "op_y": "Fugas detectadas menos las esperadas por azar",
        "op_note": "cada punto es un percentil del umbral entre 95% y 99,95%",
    },
}


def _load(name: str) -> dict | None:
    path = REPORTS_DIR / name / "results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _legend(ax, **kw):
    ax.legend(frameon=False, fontsize=8.5, **kw)


# ------------------------------------------------------------------ SCANIA
def scania_survival(r: dict, lang: str) -> Path:
    a, km = A[lang], r["analisis"]["kaplan_meier"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), facecolor=SURFACE, sharey=True)
    t = km["overall"]["t"]
    _style(axes[0], "y")
    axes[0].plot(t, km["overall"]["s"], color=BLUE, linewidth=2)
    axes[0].set_title(f"{a['km_all']} ({_fmt(km['events'], lang)} {'reparaciones' if lang == 'es' else 'repairs'})",
                      loc="left", color=INK_2, fontsize=10)
    _style(axes[1], "y")
    spec = km["by_spec"]["spec"]
    for c, color in zip(km["by_spec"]["curves"], [BLUE, ORANGE, AQUA, GRAY]):
        axes[1].plot(t, c["s"], color=color, linewidth=2, label=f"{c['category']} ({_fmt(c['trucks'], lang)} {a['trucks']})")
    axes[1].set_title(a["km_by"].format(spec=spec, stat=_fmt(km["logrank"][spec]["statistic"], lang)), loc="left", color=INK_2, fontsize=10)
    _legend(axes[1], loc="lower left")
    for ax in axes:
        ax.set_xlabel(a["time"], color=INK_2, fontsize=9)
    axes[0].set_ylabel(a["survival"], color=INK_2, fontsize=9)
    _title(fig, a["km_title"])
    fig.tight_layout()
    return _save(fig, lang, "scania_survival.png")


def scania_lead_time(r: dict, lang: str) -> Path:
    a, lt = A[lang], r["analisis"]["lead_time"]
    names = T[lang]["model"]
    x = np.arange(len(lt["bins"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    keys = list(lt["models"])
    colors = {r["seleccionado"]: BLUE, "lightgbm": AQUA, "lightgbm_escala_cruda": ORANGE}
    for ax, field, hfield, label in ((axes[0], "mean_risk", "healthy_mean_risk", a["lead_risk"]),
                                     (axes[1], "alarm_share", "healthy_alarm_share", a["lead_alarm"])):
        _style(ax, "y")
        for k in keys:
            m = lt["models"][k]
            color = colors.get(k, GRAY)
            ax.plot(x, m[field], color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE, label=names.get(k, k))
            ax.axhline(m[hfield], color=color, linewidth=1, alpha=0.6)
        ax.set_xticks(x, lt["bins"])
        ax.set_xlabel(a["lead_x"], color=INK_2, fontsize=9)
        ax.set_ylabel(label, color=INK_2, fontsize=9)
        ax.set_ylim(0, None)
        ax.text(len(x) - 1, ax.get_ylim()[1] * 0.02, a["healthy_note"], ha="right", va="bottom", fontsize=7.5, color=INK_2)
    _legend(axes[0], loc="upper right")
    _title(fig, a["lead_title"], 11)
    fig.tight_layout()
    return _save(fig, lang, "scania_lead_time.png")


def _curve_keys(r: dict) -> list[str]:
    sel = r["seleccionado"]
    return [k for k in dict.fromkeys(["logistic", "random_forest", "lightgbm", "xgboost", "catboost", "mlp", "ensemble",
                                      "lightgbm_escala_cruda", sel]) if k in r["analisis"]["curves"]["test"]]


def scania_roc_pr(r: dict, lang: str) -> Path:
    a, cv = A[lang], r["analisis"]["curves"]["test"]
    names, sel = T[lang]["model"], r["seleccionado"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), facecolor=SURFACE)
    _style(axes[0], "both")
    _style(axes[1], "both")
    for k in _curve_keys(r):
        color = BLUE if k == sel else ORANGE if k == "lightgbm_escala_cruda" else AQUA if k == "xgboost" else GRAY
        lw, z = (2.2, 3) if color != GRAY else (1, 1)
        label = f"{names[k]} ({cv[k]['auc']:.3f})" if color != GRAY else None
        axes[0].plot(cv[k]["roc"]["x"], cv[k]["roc"]["y"], color=color, linewidth=lw, zorder=z, label=label)
        axes[1].plot(cv[k]["pr"]["x"], cv[k]["pr"]["y"], color=color, linewidth=lw, zorder=z,
                     label=f"{names[k]} (AP {cv[k]['ap']:.3f})" if color != GRAY else None)
    axes[0].plot([0, 1], [0, 1], color=INK_2, linewidth=1)
    axes[1].axhline(cv["base_rate"], color=INK_2, linewidth=1)
    axes[1].text(1, cv["base_rate"], f" {a['base']}", va="bottom", ha="right", fontsize=8, color=INK_2)
    axes[0].set_xlabel(a["fpr"], color=INK_2, fontsize=9)
    axes[0].set_ylabel(a["tpr"], color=INK_2, fontsize=9)
    axes[1].set_xlabel(a["recall"], color=INK_2, fontsize=9)
    axes[1].set_ylabel(a["precision"], color=INK_2, fontsize=9)
    axes[1].set_ylim(0, max(0.3, max(max(cv[k]["pr"]["y"][1:] or [0]) for k in _curve_keys(r))))
    _legend(axes[0], loc="lower right")
    _legend(axes[1], loc="upper right")
    _title(fig, a["roc_title"] + (" (gris: otros modelos)" if lang == "es" else " (gray: other models)"), 11)
    fig.tight_layout()
    return _save(fig, lang, "scania_roc_pr.png")


def scania_cost_breakdown(r: dict, lang: str) -> Path:
    a, cb = A[lang], r["analisis"]["cost_breakdown"]["test"]
    names, sel = T[lang]["model"], r["seleccionado"]
    keys = [k for k in ["siempre_sano", "logistic", "random_forest", "lightgbm", "xgboost", "catboost", "mlp", "ensemble",
                        "lightgbm_escala_cruda"] if k in cb][::-1]
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    _style(ax)
    for i, k in enumerate(keys):
        left = 0
        for part, color in (("false_alarms", BLUE), ("missed", ORANGE), ("wrong_window", AQUA)):
            v = cb[k][part]
            ax.barh(i, v, left=left, color=color, height=0.62, edgecolor=SURFACE, linewidth=1.5,
                    label=a[{"false_alarms": "fa", "missed": "missed", "wrong_window": "window"}[part]] if i == 0 else None)
            left += v
        ax.text(left + 400, i, _fmt(cb[k]["total"], lang), va="center", fontsize=8.5, color=INK if k == sel else INK_2)
    labels = [a["nothing"] if k == "siempre_sano" else names[k] + (f" ({T[lang]['selected']})" if k == sel else "") for k in keys]
    ax.set_yticks(range(len(keys)), labels)
    ax.set_xlabel(a["cost_x"], color=INK_2, fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _fmt(v, lang)))
    _legend(ax, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3)
    _title(ax, a["cost_title"])
    return _save(fig, lang, "scania_cost_breakdown.png")


def scania_paired(r: dict, lang: str) -> Path:
    a, pb = A[lang], r["analisis"]["paired_bootstrap"]
    names, sel = T[lang]["model"], r["seleccionado"]
    keys = [k for k in _curve_keys(r) if k != sel][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), facecolor=SURFACE, sharey=True, sharex=True)
    for ax, split in zip(axes, ("validation", "test")):
        _style(ax)
        for i, k in enumerate(keys):
            d = pb[split][k]
            color = ORANGE if k == "lightgbm_escala_cruda" else GRAY
            ax.plot(d["ci95"], [i, i], color=color, linewidth=2)
            ax.scatter([d["mean_diff"]], [i], s=46, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.text(1.02, i, f"{a['p_better']} {_fmt(d['p_better'], lang, 2)}", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=7.5, color=INK_2)
        ax.axvline(0, color=BLUE, linewidth=1.5)
        ax.set_title(A[lang][split], loc="left", color=INK_2, fontsize=10)
        ax.set_xlabel(a["paired_x"], color=INK_2, fontsize=9)
    axes[0].set_yticks(range(len(keys)), [names[k] for k in keys])
    _title(fig, a["paired_title"] + f" — {names[sel]}", 10.5)
    fig.tight_layout(w_pad=6)
    return _save(fig, lang, "scania_paired_bootstrap.png")


def scania_seed_spread(r: dict, lang: str) -> Path | None:
    spread = r["analisis"].get("seed_spread")
    if not spread:
        return None
    a, names = A[lang], T[lang]["model"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), facecolor=SURFACE)
    keys = list(spread)
    colors = {"mlp": BLUE, "xgboost": ORANGE}
    for ax, field, label in ((axes[0], "seleccion", a["seed_sel"]), (axes[1], "validation", a["val_saving"]),
                             (axes[2], "test", a["seed_test"])):
        _style(ax)
        for i, k in enumerate(keys):
            vals = [row[field] for row in spread[k]]
            ax.scatter(vals, np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), s=36, color=colors.get(k, GRAY),
                       edgecolor=SURFACE, linewidth=1, zorder=3)
            ax.plot([min(vals), max(vals)], [i, i], color=GRID, linewidth=6, zorder=1, solid_capstyle="round")
        ax.set_yticks(range(len(keys)), [names[k] for k in keys])
        ax.set_ylim(-0.6, len(keys) - 0.4)
        ax.set_xlabel(label, color=INK_2, fontsize=9)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    _title(fig, a["seed_title"] + (f" ({len(spread[keys[0]])} semillas)" if lang == "es" else f" ({len(spread[keys[0]])} seeds)"), 11)
    fig.tight_layout()
    return _save(fig, lang, "scania_seed_spread.png")


def scania_cut_timing(r: dict, lang: str) -> Path:
    a, ct = A[lang], r["analisis"]["cut_timing"]
    edges = np.array(ct["edges"])
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
    _style(ax, "y")
    for k, color in (("train_random", GRAY), ("validation", BLUE), ("test", ORANGE)):
        ax.stairs(ct[k]["share"], edges, color=color, linewidth=2,
                  label=f"{a[k]} ({'mediana' if lang == 'es' else 'median'} {_fmt(ct[k]['median'], lang)})")
    ax.set_xlabel(a["cut_x"], color=INK_2, fontsize=9)
    ax.set_ylabel(a["cut_y"], color=INK_2, fontsize=9)
    _legend(ax, loc="upper right")
    _title(ax, a["cut_title"], 11)
    return _save(fig, lang, "scania_cut_timing.png")


# ------------------------------------------------------------------ APS
def aps_threshold(r: dict, lang: str) -> Path:
    a, an = A[lang], r["analysis"]
    sel = r["selected_by_oof"]
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURFACE)
    _style(ax, "y")
    for key, color, label in (("oof", GRAY, a["oof"]), ("test", BLUE, A[lang]["test"])):
        c = an["threshold_curve"][key]
        ax.plot(c["x"], c["y"], color=color, linewidth=2, label=label)
    thr = r["models"][sel]["threshold_from_oof"]
    ax.axvline(thr, color=ORANGE, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_ylim(0, 3.0)
    ax.text(thr * 1.08, 2.75, f"{a['chosen']} ({_fmt(thr, lang, 3)})", fontsize=8.5, color=INK_2)
    ax.set_xlabel(a["thr_x"], color=INK_2, fontsize=9)
    ax.set_ylabel(a["thr_y"], color=INK_2, fontsize=9)
    _legend(ax, loc="upper left")
    _title(ax, a["thr_title"], 11)
    return _save(fig, lang, "aps_threshold.png")


def aps_pr(r: dict, lang: str) -> Path:
    a, an, names = A[lang], r["analysis"], T[lang]["model"]
    sel = r["selected_by_oof"]
    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=SURFACE)
    _style(ax, "both")
    for k, c in an["pr_curves"].items():
        color = BLUE if k == sel else ORANGE if k == "mlp" else GRAY
        ax.plot(c["x"], c["y"], color=color, linewidth=2.2 if color != GRAY else 1, zorder=3 if color != GRAY else 1,
                label=f"{names[k]} (PR-AUC {r['models'][k]['test']['pr_auc']:.3f})" if color != GRAY else None)
    ax.axhline(an["base_rate_test"], color=INK_2, linewidth=1)
    ax.set_xlabel(a["recall"], color=INK_2, fontsize=9)
    ax.set_ylabel(a["precision"], color=INK_2, fontsize=9)
    _legend(ax, loc="lower left")
    _title(ax, a["apr_title"] + (" (gris: otros modelos)" if lang == "es" else " (gray: other models)"), 11)
    return _save(fig, lang, "aps_pr.png")


def aps_missingness(r: dict, lang: str) -> Path:
    a, m = A[lang], r["analysis"]["missingness"]
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    _style(ax, "y")
    x = np.arange(len(m["bins"]))
    ax.bar(x, np.array(m["failure_rate"]) * 100, color=BLUE, width=0.6)
    for xi, v, n in zip(x, m["failure_rate"], m["trucks"]):
        ax.text(xi, v * 100, f"{_fmt(v * 100, lang, 1)}%\n({_fmt(n, lang)})", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(x, m["bins"])
    ax.set_ylim(0, max(np.array(m["failure_rate"]) * 100) * 1.3)
    ax.set_xlabel(a["miss_x"], color=INK_2, fontsize=9)
    ax.set_ylabel(a["miss_y"] + " (%)", color=INK_2, fontsize=9)
    _title(ax, a["miss_title"], 11)
    return _save(fig, lang, "aps_missingness.png")


def aps_shap(r: dict, lang: str) -> Path:
    a, rows = A[lang], r["analysis"]["shap_by_variable"][::-1]
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    _style(ax)
    ax.barh([d["variable"] for d in rows], [d["mean_abs_shap"] for d in rows], color=BLUE, height=0.62)
    ax.set_xlabel(a["shap_x"], color=INK_2, fontsize=9)
    _title(ax, a["shap_title"], 11)
    return _save(fig, lang, "aps_shap.png")


# ------------------------------------------------------------------ C-MAPSS
def cmapss_learning_curves(r: dict, lang: str) -> Path | None:
    a, subsets = A[lang], list(r["subsets"])
    if not any("history" in r["subsets"][s]["models"].get("gru", {}) for s in subsets):
        return None
    fig, axes = plt.subplots(1, len(subsets), figsize=(12, 3.4), facecolor=SURFACE)
    for ax, s in zip(np.atleast_1d(axes), subsets):
        _style(ax, "y")
        for kind, color in (("cnn", ORANGE), ("gru", BLUE)):
            h = r["subsets"][s]["models"].get(kind, {}).get("history")
            if not h:
                continue
            ep = np.arange(1, len(h["val_loss"]) + 1)
            ax.plot(ep, h["val_loss"], color=color, linewidth=1.6, label=T[lang]["model"][kind])
            best = int(np.argmin(h["val_loss"]))
            ax.scatter([ep[best]], [h["val_loss"][best]], s=40, color=color, edgecolor=SURFACE, zorder=3)
        ax.set_yscale("log")
        ax.set_title(s, loc="left", color=INK_2, fontsize=10)
        ax.set_xlabel(a["epoch"], color=INK_2, fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel(a["lc_y"], color=INK_2, fontsize=9)
    _legend(np.atleast_1d(axes)[0], loc="upper right")
    _title(fig, a["lc_title"], 11)
    fig.tight_layout()
    return _save(fig, lang, "cmapss_learning_curves.png")


def cmapss_error_by_rul(r: dict, lang: str) -> Path | None:
    path = REPORTS_DIR / "cmapss" / "test_predictions.csv"
    if not path.exists():
        return None
    a, preds, subsets = A[lang], pd.read_csv(path), list(r["subsets"])
    fig, axes = plt.subplots(1, len(subsets), figsize=(12, 3.6), facecolor=SURFACE, sharey=True)
    for ax, s in zip(np.atleast_1d(axes), subsets):
        _style(ax, "y")
        d = preds[preds["subset"] == s]
        sel = r["subsets"][s]["selected_by_cv"]
        err = d[sel] - d["rul_true"]
        ax.axhspan(0, 100, color=ORANGE, alpha=0.07, linewidth=0)
        ax.scatter(d["rul_true"], err, s=10, color=BLUE, alpha=0.7, edgecolor="none")
        ax.axhline(0, color=INK_2, linewidth=1)
        ax.axvline(125, color=GRAY, linewidth=1)
        ax.set_ylim(-110, 70)
        ax.set_title(f"{s}: {T[lang]['model'][sel]}", loc="left", color=INK_2, fontsize=9.5)
        ax.set_xlabel(T[lang]["true_rul"], color=INK_2, fontsize=9)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel(a["err_y"], color=INK_2, fontsize=9)
    first.text(2, 60, a["late"], fontsize=8, color=INK_2, va="top")
    first.text(2, -100, a["early"], fontsize=8, color=INK_2)
    _title(fig, a["err_title"], 11)
    fig.tight_layout()
    return _save(fig, lang, "cmapss_error_by_rul.png")


def cmapss_sensor_trends(r: dict, lang: str, subset: str = "FD001") -> Path | None:
    tr = r["subsets"].get(subset, {}).get("sensor_trends")
    if not tr:
        return None
    a = A[lang]
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
    _style(ax, "y")
    colors = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#6250d6"]
    for (sensor, vals), color in zip(tr["sensors"].items(), colors):
        ax.plot(tr["rul"], vals, color=color, linewidth=1.8, label=f"{sensor} (|r| {_fmt(tr['abs_corr_with_rul'][sensor], lang, 2)})")
    ax.axvline(125, color=INK_2, linewidth=1)
    ax.text(127, ax.get_ylim()[1] * 0.9, a["cap"], fontsize=8, color=INK_2)
    ax.invert_xaxis()
    ax.set_xlabel(a["sens_x"], color=INK_2, fontsize=9)
    ax.set_ylabel(a["sens_y"], color=INK_2, fontsize=9)
    _legend(ax, loc="upper left", ncol=2)
    _title(ax, a["sens_title"], 11)
    return _save(fig, lang, "cmapss_sensor_trends.png")


# ------------------------------------------------------------------ MetroPT-3
def metropt_daily(r: dict, lang: str) -> Path:
    a, d = A[lang], r["daily"]
    dates = pd.to_datetime(d["date"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), facecolor=SURFACE, sharex=True)
    for ax, field, label in ((axes[0], "DV_eletric_frac", a["duty"]), (axes[1], "Oil_temperature_mean", a["oil"])):
        _style(ax, "y")
        ax.axvspan(pd.Timestamp(r["protocol"]["fit_period"][0]), pd.Timestamp(r["protocol"]["fit_period"][1]), color=GRAY, alpha=0.12, linewidth=0)
        ax.axvspan(pd.Timestamp(r["protocol"]["calibration_period"][0]), pd.Timestamp(r["protocol"]["calibration_period"][1]),
                   color=GRAY, alpha=0.25, linewidth=0)
        for f in r["failures"]:
            ax.axvspan(pd.Timestamp(f["start"]), pd.Timestamp(f["end"]) + pd.Timedelta(hours=12), color=ORANGE, alpha=0.4, linewidth=0)
        ax.plot(dates, d[field], color=BLUE, linewidth=1.4, marker="o", markersize=2.5)
        ax.set_ylabel(label, color=INK_2, fontsize=9)
    peak = dates[np.argmax(np.where(dates < pd.Timestamp("2020-04-01"), d["DV_eletric_frac"], -1))]
    axes[0].annotate(a["unreported"], xy=(peak, 1.0), xytext=(pd.Timestamp("2020-02-05"), 0.72), fontsize=8.5, color=INK_2,
                     arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8})
    axes[0].text(pd.Timestamp(r["protocol"]["fit_period"][0]) + pd.Timedelta(days=2), 0.93, a["fit"], fontsize=8, color=INK_2)
    axes[0].text(pd.Timestamp(r["protocol"]["calibration_period"][0]) + pd.Timedelta(days=1), 0.93, a["calib"], fontsize=8, color=INK_2)
    axes[0].set_ylim(0, 1.05)
    _title(fig, a["daily_title"], 11)
    fig.tight_layout()
    return _save(fig, lang, "metropt_daily.png")


def metropt_failures(r: dict, lang: str) -> Path | None:
    path = REPORTS_DIR / "metropt" / "failure_windows.csv"
    if not path.exists():
        return None
    a, w = A[lang], pd.read_csv(path)
    ids = sorted(w["failure"].unique())
    rows = [("DV_eletric_frac", a["load10"]), ("Reservoirs_min", a["res"]), ("autoencoder_score_over_threshold", a["ae"])]
    fig, axes = plt.subplots(len(rows), len(ids), figsize=(12, 6.4), facecolor=SURFACE, sharey="row")
    for j, fid in enumerate(ids):
        d = w[w["failure"] == fid].sort_values("hours_from_start")
        f = next(x for x in r["failures"] if int(x["id"]) == int(fid))
        dur = (pd.Timestamp(f["end"]) - pd.Timestamp(f["start"])).total_seconds() / 3600
        # Ventanas sin datos quedan como huecos en la linea, no como puentes.
        h = d["hours_from_start"].to_numpy()
        gap = np.r_[False, np.diff(h) > 0.5]
        for i, (col, label) in enumerate(rows):
            ax = axes[i, j]
            _style(ax, "y")
            ax.axvspan(0, dur, color=ORANGE, alpha=0.25, linewidth=0)
            vals = d[col].to_numpy(dtype=float).copy()
            vals[gap] = np.nan
            ax.plot(h, vals, color=BLUE, linewidth=1)
            if col.startswith("autoencoder"):
                ax.axhline(1.0, color=INK_2, linewidth=1)
                ax.set_yscale("log")
            if j == 0:
                ax.set_ylabel(label, color=INK_2, fontsize=8.5)
            if i == 0:
                ax.set_title(f"{a['leak']} {fid}", loc="left", color=INK_2, fontsize=10)
            if i == len(rows) - 1:
                ax.set_xlabel(a["hours"], color=INK_2, fontsize=8.5)
    _title(fig, a["zoom_title"], 11)
    fig.tight_layout()
    return _save(fig, lang, "metropt_failures.png")


def metropt_operating(r: dict, lang: str) -> Path:
    a = A[lang]
    det_names = T[lang]["det"]
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=SURFACE)
    _style(ax, "both")
    colors = {"pca": GRAY, "isolation_forest": AQUA, "autoencoder": BLUE, "duty_zscore": ORANGE}
    for key, d in r["detectors"].items():
        curve = d.get("operating_curve")
        if not curve or d["view"] != "absolute" or d["detector"] == "duty_zscore":  # nunca alarma: no hay curva que dibujar
            continue
        xs = [max(c["false_alarms_per_week"], 0.05) for c in curve]
        ys = [c["detected_2h"] - c["chance_detected_2h"] for c in curve]
        ax.plot(xs, ys, color=colors.get(d["detector"], GRAY), linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                label=det_names[d["detector"]])
    lps = r["detectors"].get("lps_rule")
    if lps:
        ax.scatter([lps["false_alarms_per_week"]], [lps["detected_2h"] - lps["chance_detected_2h"]], s=70, marker="D", color=INK,
                   label=det_names["lps_rule"], zorder=4)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel(a["op_x"], color=INK_2, fontsize=9)
    ax.set_ylabel(a["op_y"], color=INK_2, fontsize=9)
    ax.text(0.99, 0.02, a["op_note"], transform=ax.transAxes, ha="right", fontsize=8, color=INK_2)
    _legend(ax, loc="upper left")
    _title(ax, a["op_title"], 11)
    return _save(fig, lang, "metropt_operating.png")


def make_all_analysis() -> list[Path]:
    sc, aps, cm, mp = _load("scania"), _load("aps"), _load("cmapss"), _load("metropt")
    out: list[Path | None] = []
    for lang in ("en", "es"):
        if sc and "analisis" in sc:
            out += [scania_survival(sc, lang), scania_lead_time(sc, lang), scania_roc_pr(sc, lang), scania_cost_breakdown(sc, lang),
                    scania_paired(sc, lang), scania_seed_spread(sc, lang), scania_cut_timing(sc, lang)]
        if aps and "analysis" in aps:
            out += [aps_threshold(aps, lang), aps_pr(aps, lang), aps_missingness(aps, lang), aps_shap(aps, lang)]
        if cm:
            out += [cmapss_learning_curves(cm, lang), cmapss_error_by_rul(cm, lang), cmapss_sensor_trends(cm, lang)]
        if mp and "daily" in mp:
            out += [metropt_daily(mp, lang), metropt_failures(mp, lang), metropt_operating(mp, lang)]
    return [p for p in out if p is not None]
