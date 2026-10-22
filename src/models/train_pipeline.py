"""Pipeline de mantenimiento predictivo sobre SCANIA Component X (datos reales).

1. Diagnostico de escala: cada variable trae un factor de escala distinto en train, validacion y test. Se mide
   (factor por variable y validacion adversarial) y se construye un conjunto de features invariante a ese factor.
2. Clasificador de riesgo (clases 0-4 del reto): seis familias de modelos y un ensamble, entrenados con las
   mismas filas. La decision de mantenimiento es la de menor costo esperado bajo la matriz oficial.
3. Seleccion de modelo SIN mirar validacion ni test: costo esperado en camiones retenidos de train, ponderando
   cada lectura por 1 / (lecturas del camion), que es exactamente el protocolo de validacion y test (una lectura
   al azar por camion). Validacion y test se evaluan una vez, con todos los modelos, e intervalo por bootstrap.
4. Tiempo hasta la falla (camiones que fallaron) y supervivencia en un landmark (Cox, Weibull AFT, bosque de
   supervivencia aleatorio). SHAP del modelo de tiempo restante.

Particiones por vehiculo: 70% ajuste, 15% parada temprana, 15% seleccion (estratificadas por falla).

    python -m src.models.train_pipeline [--force]
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data import scania_loader as S
from src.features import engineering as E
from src.models import calibration as K
from src.models import cost_decision as C
from src.models.evaluation import bootstrap
from src.models.zoo import MODEL_NAMES, TabularModel

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = PROCESSED_DIR / "models"
REPORTS_DIR = ROOT / "reports" / "scania"

SPEC_COLUMNS = [f"Spec_{i}" for i in range(8)]
NEG_RATE = 0.05  # fraccion de lecturas clase 0 que se usa para entrenar (se compensa con pesos)
LANDMARK = 50.0  # tiempo (unidades del dataset) en que se fija el riesgo para los modelos de supervivencia
SEED = 42
SPLIT_FRACTIONS = {"early_stop": 0.15, "select": 0.15}
ENSEMBLE = ["lightgbm", "xgboost", "catboost"]
N_BOOT = 1000
LGB_COMMON = dict(
    learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.3,
    min_child_samples=20, random_state=SEED, verbosity=-1, n_jobs=-1,
)


# ----------------------------------------------------------------------------- datos
def spec_categories(train_specs: pl.DataFrame) -> dict[str, list[str]]:
    return {c: sorted(train_specs[c].unique().to_list()) for c in SPEC_COLUMNS}


def with_specs(frame: pd.DataFrame, specs: pl.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Une las especificaciones como columnas categoricas con categorias fijadas en train."""
    spec_pdf = specs.to_pandas().set_index("vehicle_id")
    out = frame.join(spec_pdf, on="vehicle_id", how="left")
    for c in SPEC_COLUMNS:
        out[c] = pd.Categorical(out[c], categories=categories[c])
    return out


def vehicle_split(vehicle_ids, strata, test_size: float = 0.2, seed: int = SEED):
    """Split reproducible por vehiculo, estratificado (para mantener la tasa de fallas)."""
    rng = np.random.default_rng(seed)
    ids = np.asarray(vehicle_ids)
    strata = np.asarray(strata)
    test = []
    for s in np.unique(strata):
        group = ids[strata == s]
        rng.shuffle(group)
        test.extend(group[: max(1, int(len(group) * test_size))])
    test_set = set(test)
    return {v for v in ids if v not in test_set}, test_set


def three_way_split(vehicle_ids, strata, seed: int = SEED) -> dict[str, set]:
    """Ajuste / parada temprana / seleccion, por vehiculo y estratificado."""
    rest, select = vehicle_split(vehicle_ids, strata, SPLIT_FRACTIONS["select"], seed)
    ids, strata = np.asarray(vehicle_ids), np.asarray(strata)
    keep = np.isin(ids, list(rest))
    share = SPLIT_FRACTIONS["early_stop"] / (1 - SPLIT_FRACTIONS["select"])
    fit, early = vehicle_split(ids[keep], strata[keep], share, seed + 1)
    return {"fit": fit, "early_stop": early, "select": select}


def load_features(split: str, force: bool = False) -> pl.DataFrame:
    cache = PROCESSED_DIR / f"features_{split}.parquet"
    if cache.exists() and not force:
        feats = pl.read_parquet(cache)
        if any(c.startswith("rel_rate_") for c in feats.columns):
            return feats
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feats = E.engineer_features(S.read_readouts(split))
    feats.write_parquet(cache)
    return feats


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ("vehicle_id", "time_to_failure", "class_label", "duration", "event", "weight")]


def clean(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Infinitos a NaN (cocientes con denominador casi nulo) para que todos los modelos los traten como faltantes."""
    out = frame[cols].copy()
    num = [c for c in cols if c not in SPEC_COLUMNS]
    out[num] = out[num].replace([np.inf, -np.inf], np.nan)
    return out


# ------------------------------------------------------------------- diagnostico de escala
def one_random_readout(feats: pl.DataFrame, seed: int = SEED) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return (feats.with_columns(pl.Series("_u", rng.random(feats.height))).sort("_u")
            .group_by("vehicle_id").first().drop("_u").sort("vehicle_id"))


def scale_factors(train_rand: pl.DataFrame, others: dict[str, pl.DataFrame]) -> list[dict]:
    """Por variable: mediana de ``life_rate`` en cada conjunto dividida por la de train (mediana sobre sus bins)."""
    life = [c for c in train_rand.columns if c.startswith("life_rate_")]
    groups: dict[str, list[str]] = {}
    for c in life:
        groups.setdefault(c[len("life_rate_"):].rsplit("_", 1)[0], []).append(c)
    rows = []
    for var, cols in groups.items():
        row = {"variable": var, "bins": len(cols)}
        for name, frame in others.items():
            ratios = []
            for c in cols:
                base = train_rand[c].drop_nulls().drop_nans().median()
                other = frame[c].drop_nulls().drop_nans().median()
                if base and other and base > 0:
                    ratios.append(other / base)
            row[name] = round(float(np.median(ratios)), 3) if ratios else None
        rows.append(row)
    return rows


def adversarial_auc(a: pd.DataFrame, b: pd.DataFrame, seed: int = SEED) -> dict:
    """AUC (5 particiones) de un LightGBM que intenta distinguir filas de ``a`` y ``b``; 0,5 = indistinguibles."""
    X = pd.concat([a, b], ignore_index=True)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, colsample_bytree=0.3, verbosity=-1, random_state=seed)
    p = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    model.fit(X, y)
    gain = pd.Series(model.booster_.feature_importance("gain"), index=X.columns)
    top = (gain / gain.sum()).sort_values(ascending=False).head(5).round(3)
    return {"auc": round(float(roc_auc_score(y, p)), 3), "top_features": top.to_dict()}


# ------------------------------------------------------------------- clasificador
def sample_rows(labeled: pl.DataFrame, neg_rate: float = NEG_RATE, seed: int = SEED) -> pl.DataFrame:
    """Todas las lecturas con riesgo (clase > 0) y una fraccion de las sanas, con su peso."""
    rng = np.random.default_rng(seed)
    keep = (labeled["class_label"].to_numpy() > 0) | (rng.random(labeled.height) < neg_rate)
    sampled = labeled.filter(pl.Series(keep))
    weight = np.where(sampled["class_label"].to_numpy() > 0, 1.0, 1.0 / neg_rate)
    return sampled.with_columns(pl.Series("weight", weight))


def protocol_weights(vehicle_ids: np.ndarray) -> np.ndarray:
    """1 / (lecturas del camion): pondera las lecturas como si se eligiera una al azar por camion."""
    ids = pd.Series(vehicle_ids)
    return (1.0 / ids.map(ids.value_counts())).to_numpy()


def evaluate_decisions(y_true, proba: np.ndarray) -> dict:
    """Costo oficial de tres reglas: todo sano, argmax y minimo costo esperado."""
    rules = {
        "siempre_sano": np.zeros(len(y_true), dtype=int),
        "argmax": proba.argmax(axis=1),
        "costo_esperado_minimo": C.bayes_decision(proba),
    }
    out = {}
    for name, pred in rules.items():
        matrix = C.confusion(y_true, pred)
        out[name] = {
            "costo_total": C.cost_from_confusion(matrix),
            **C.alarm_metrics(y_true, pred),
            "matriz_confusion": matrix.tolist(),
        }
    return out


def saving_pct(y, proba) -> float:
    cost = np.asarray(S.COST_MATRIX)
    policy = cost[y, C.bayes_decision(proba)].sum()
    return 100.0 * (1.0 - policy / cost[y, 0].sum())


def selection_metrics(y, proba, weight) -> dict:
    """Costo esperado por 1.000 camiones bajo el protocolo de una lectura al azar por camion."""
    cost = np.asarray(S.COST_MATRIX, dtype=float)
    w = weight / weight.sum()
    risk = 1.0 - proba[:, 0]
    return {
        "costo_por_1000_camiones": round(1000 * float((w * cost[y, C.bayes_decision(proba)]).sum()), 2),
        "costo_por_1000_no_hacer_nada": round(1000 * float((w * cost[y, 0]).sum()), 2),
        "riesgo_medio_predicho": round(float((w * risk).sum()), 4),
        "tasa_observada": round(float((w * (y > 0)).sum()), 4),
    }


def split_metrics(y, proba, vehicle_ids) -> dict:
    ev = evaluate_decisions(y, proba)
    best = ev["costo_esperado_minimo"]
    ci = bootstrap(saving_pct, y, proba, n=N_BOOT, seed=SEED)
    return {
        "costo": best["costo_total"], "costo_no_hacer_nada": ev["siempre_sano"]["costo_total"],
        "costo_argmax": ev["argmax"]["costo_total"],
        "ahorro_pct": round(ci["point"], 2), "ahorro_ic95": [round(v, 2) for v in ci["ci95"]],
        "recall_alarma": round(best["recall_alarma"], 4), "tasa_falsa_alarma": round(best["tasa_falsa_alarma"], 4),
        "auc_algun_riesgo": round(float(roc_auc_score(y > 0, 1.0 - proba[:, 0])), 4),
        "riesgo_medio_predicho": round(float((1.0 - proba[:, 0]).mean()), 4),
        "tasa_observada": round(float((y > 0).mean()), 4),
        "costo_visita_equilibrio": K.break_even(y, proba),
        "sensibilidad": [{"k": row["k"], "ahorro_pct": round(row["ahorro_pct"], 2)} for row in K.sensitivity(y, proba)],
    }


class MeanEnsemble:
    """Promedio simple de probabilidades de varios modelos ya entrenados."""

    def __init__(self, models: list):
        self.models = models

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


# ------------------------------------------------------------------------- tiempo restante
def train_time_to_failure(train_pdf: pd.DataFrame, hold_pdf: pd.DataFrame, cols: list[str]):
    model = lgb.LGBMRegressor(objective="regression_l1", n_estimators=2000, **LGB_COMMON)
    model.fit(
        train_pdf[cols], np.log1p(train_pdf["time_to_failure"]),
        eval_X=(hold_pdf[cols],), eval_y=(np.log1p(hold_pdf["time_to_failure"]),),
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


# ------------------------------------------------------------------------------ supervivencia
def landmark_table(readouts: pl.DataFrame, tte: pl.DataFrame, specs: pl.DataFrame, landmark: float = LANDMARK):
    """Una fila por vehiculo en el landmark: uso temprano por contador + especificaciones.

    Solo entra el uso observado hasta ``landmark``; el vehiculo se modela con entrada tardia
    (``entry = landmark``), asi que no se condiciona sobre informacion posterior.
    """
    single = [cols[0] for cols in E.variable_groups(readouts.columns).values() if len(cols) == 1]
    early = readouts.filter(pl.col("time_step") <= landmark).sort(E.ID_COLUMNS).group_by("vehicle_id", maintain_order=True).last()
    rates = early.select(
        "vehicle_id", *[(pl.col(c) / pl.col("time_step").clip(lower_bound=1.0)).log1p().alias(f"usage_{c}") for c in single]
    )
    df = tte.join(rates, on="vehicle_id", how="inner").join(specs, on="vehicle_id", how="inner")
    usage = [c for c in df.columns if c.startswith("usage_")]
    # Un contador nulo en la lectura del landmark se imputa con la mediana de la flota.
    df = df.with_columns([pl.col(c).fill_nan(None).fill_null(pl.col(c).median()) for c in usage])
    return df.filter(pl.col("duration") > landmark).with_columns(pl.lit(landmark).alias("entry"))


def cox_design(table: pl.DataFrame, categories: dict[str, list[str]], min_count: int = 100) -> pd.DataFrame:
    pdf = table.to_pandas()
    for c in SPEC_COLUMNS:
        counts = pdf[c].value_counts()
        rare = counts[counts < min_count].index
        pdf[c] = pdf[c].where(~pdf[c].isin(rare), "rare")
    return pdf


def survival_matrices(design: pd.DataFrame, train_ids: set, hold_ids: set, use_specs: bool = True, use_usage: bool = True):
    usage_cols = [c for c in design.columns if c.startswith("usage_")]
    parts = []
    if use_usage:
        parts.append(design[usage_cols])
    if use_specs:
        parts.append(pd.get_dummies(design[SPEC_COLUMNS], drop_first=True, dtype=float))
    X = pd.concat(parts, axis=1)
    X = X.loc[:, X.nunique() > 1]
    covariates = list(X.columns)
    mask_tr = design["vehicle_id"].isin(train_ids).to_numpy()
    mask_ho = design["vehicle_id"].isin(hold_ids).to_numpy() & ~mask_tr
    train_x, hold_x = X[mask_tr].copy(), X[mask_ho].copy()
    mu, sd = train_x.mean(), train_x.std().replace(0, 1)
    train_x, hold_x = (train_x - mu) / sd, (hold_x - mu) / sd
    meta = design[["duration", "event", "entry"]]
    return train_x, hold_x, meta[mask_tr], meta[mask_ho], covariates, mu, sd


def fit_cox(design: pd.DataFrame, train_ids: set, hold_ids: set, use_specs: bool, use_usage: bool) -> dict:
    train_x, hold_x, m_tr, m_ho, covariates, mu, sd = survival_matrices(design, train_ids, hold_ids, use_specs, use_usage)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(pd.concat([train_x, m_tr], axis=1), duration_col="duration", event_col="event", entry_col="entry")
    risk = cph.predict_partial_hazard(hold_x)
    c_index = concordance_index(m_ho["duration"], -risk, m_ho["event"])
    return {"c_index": float(c_index), "n_covariates": len(covariates), "model": cph, "mu": mu, "sd": sd}


def fit_weibull_aft(design: pd.DataFrame, train_ids: set, hold_ids: set) -> dict:
    train_x, hold_x, m_tr, m_ho, covariates, *_ = survival_matrices(design, train_ids, hold_ids)
    aft = WeibullAFTFitter(penalizer=0.1)
    aft.fit(pd.concat([train_x, m_tr], axis=1), duration_col="duration", event_col="event", entry_col="entry")
    median = aft.predict_median(hold_x)  # mayor mediana = menos riesgo
    return {"c_index": float(concordance_index(m_ho["duration"], np.asarray(median), m_ho["event"])), "n_covariates": len(covariates)}


def fit_random_survival_forest(design: pd.DataFrame, train_ids: set, hold_ids: set) -> dict:
    """Todos los vehiculos entran en el mismo landmark, asi que la entrada tardia equivale a medir el tiempo desde ahi."""
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv

    train_x, hold_x, m_tr, m_ho, covariates, *_ = survival_matrices(design, train_ids, hold_ids)
    y_tr = Surv.from_arrays(m_tr["event"].astype(bool).to_numpy(), (m_tr["duration"] - m_tr["entry"]).to_numpy())
    rsf = RandomSurvivalForest(n_estimators=200, min_samples_leaf=30, max_features="sqrt", max_samples=0.5,
                               n_jobs=-1, random_state=SEED, low_memory=True)
    rsf.fit(train_x.to_numpy(), y_tr)
    risk = rsf.predict(hold_x.to_numpy())
    return {"c_index": float(concordance_index(m_ho["duration"], -risk, m_ho["event"])), "n_covariates": len(covariates)}


RAW_SCALE_KEY = "lightgbm_escala_cruda"
ALL_PROBABILITIES = PROCESSED_DIR / "scania_all_model_probabilities.parquet"


def export_predictions(selected: str, source: Path = ALL_PROBABILITIES, out: Path = REPORTS_DIR / "predictions.csv") -> Path:
    """Probabilidades por camion de validacion y test para la pagina, con 6 decimales: ``p0..p4`` del modelo
    elegido, ``l0..l4`` de LightGBM con features invariantes y ``r0..r4`` del mismo LightGBM con features en escala
    cruda (la version anterior del proyecto). Los dos LightGBM solo difieren en las features."""
    allp = pd.read_parquet(source)
    cols = [f"p{k}" for k in range(5)]
    frame = allp[allp["model"] == selected].set_index(["split", "vehicle_id"])[["class_label", *cols]]
    for key, prefix in (("lightgbm", "l"), (RAW_SCALE_KEY, "r")):
        part = allp[allp["model"] == key].set_index(["split", "vehicle_id"])[cols]
        part.columns = [f"{prefix}{k}" for k in range(5)]
        frame = frame.join(part)
    frame = frame.reset_index()
    order = {"validation": 0, "test": 1}
    frame = frame.sort_values(["split", "vehicle_id"], key=lambda c: c.map(order) if c.name == "split" else c)
    prob_cols = [c for c in frame.columns if c[0] in "plr" and c[1:].isdigit()]
    frame[prob_cols] = frame[prob_cols].round(6)
    frame.to_csv(out, index=False)
    return out


# ------------------------------------------------------------------------------- main
def run(force_features: bool = False, models: list[str] | None = None) -> dict:
    models = models or MODEL_NAMES
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    tte = S.read_tte()
    train_specs = S.read_specs("train")
    cats = spec_categories(train_specs)

    # --- features y etiquetas de train
    feats_tr = load_features("train", force_features)
    readouts_tr = S.read_readouts("train")
    labeled = S.label_train_readouts(readouts_tr.select(E.ID_COLUMNS), tte).select(
        "vehicle_id", "time_step", "time_to_failure", "class_label"
    )
    full = feats_tr.join(labeled, on=["vehicle_id", "time_step"], how="inner")
    split_feats = {s: E.last_readout_per_vehicle(load_features(s, force_features)) for s in ("validation", "test")}

    all_cols = [c for c in feats_tr.columns if c != "vehicle_id"]
    invariant_cols = E.scale_invariant(all_cols)
    results: dict = {"datos": {
        "vehiculos_train": int(tte.height), "reparaciones_train": int(tte["event"].sum()),
        "lecturas_train": int(readouts_tr.height), "features_todas": len(all_cols) + len(SPEC_COLUMNS),
        "features_invariantes": len(invariant_cols) + len(SPEC_COLUMNS),
    }}

    # --- 1. diagnostico de escala por conjunto
    train_rand = one_random_readout(feats_tr)
    factors = scale_factors(train_rand, {"validation": split_feats["validation"], "test": split_feats["test"]})
    adv = {}
    for label, cols in (("todas", all_cols), ("invariantes", invariant_cols)):
        frames = {k: clean(v.to_pandas(), cols) for k, v in
                  (("train", train_rand), ("validation", split_feats["validation"]), ("test", split_feats["test"]))}
        adv[label] = {f"{a}_vs_{b}": adversarial_auc(frames[a], frames[b])
                      for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    results["escala"] = {"factor_por_variable": factors, "validacion_adversarial": adv}
    print("[Escala] AUC adversarial:", {k: {p: v["auc"] for p, v in d.items()} for k, d in adv.items()}, flush=True)

    # --- 2. particiones y filas de entrenamiento
    parts = three_way_split(tte["vehicle_id"].to_numpy(), tte["event"].to_numpy())
    print("[Split] vehiculos " + " ".join(f"{k}={len(v)}" for k, v in parts.items()), flush=True)
    fit_pdf = with_specs(sample_rows(full.filter(pl.col("vehicle_id").is_in(list(parts["fit"])))).to_pandas(), train_specs, cats)
    es_pdf = with_specs(sample_rows(full.filter(pl.col("vehicle_id").is_in(list(parts["early_stop"]))), seed=SEED + 1).to_pandas(),
                        train_specs, cats)
    sel_pdf = with_specs(full.filter(pl.col("vehicle_id").is_in(list(parts["select"]))).to_pandas(), train_specs, cats)
    sel_w = protocol_weights(sel_pdf["vehicle_id"].to_numpy())
    sel_y = sel_pdf["class_label"].to_numpy()

    eval_frames = {}
    for split in ("validation", "test"):
        frame = split_feats[split].join(S.read_labels(split), on="vehicle_id", how="inner")
        eval_frames[split] = with_specs(frame.to_pandas(), S.read_specs(split), cats)

    feature_sets = {"invariantes": invariant_cols + SPEC_COLUMNS, "todas": all_cols + SPEC_COLUMNS}
    candidates = [(m, "invariantes") for m in models] + [("lightgbm", "todas")]
    fitted, probas = {}, {}
    results["modelos"] = {}
    print(f"[Clasificador] filas ajuste={len(fit_pdf)} parada={len(es_pdf)} seleccion={len(sel_pdf)}", flush=True)
    for name, fset in candidates:
        key = name if fset == "invariantes" else f"{name}_escala_cruda"
        cols = feature_sets[fset]
        t0 = time.time()
        model = TabularModel(name, 5, categorical=SPEC_COLUMNS, seed=SEED, checkpoint_dir=MODELS_DIR / "scania")
        model.fit(clean(fit_pdf, cols), fit_pdf["class_label"], fit_pdf["weight"],
                  clean(es_pdf, cols), es_pdf["class_label"], es_pdf["weight"])
        fitted[key] = model
        probas[key] = {"select": model.predict_proba(clean(sel_pdf, cols))}
        for split, frame in eval_frames.items():
            probas[key][split] = model.predict_proba(clean(frame, cols))
        results["modelos"][key] = {"familia": name, "features": fset, "info": model.info,
                                   "segundos": round(time.time() - t0, 1)}
        print(f"[Clasificador] {key} entrenado en {time.time() - t0:.0f}s {model.info}", flush=True)

    if all(m in fitted for m in ENSEMBLE):
        fitted["ensemble"] = MeanEnsemble([fitted[m] for m in ENSEMBLE])
        probas["ensemble"] = {k: np.mean([probas[m][k] for m in ENSEMBLE], axis=0) for k in probas[ENSEMBLE[0]]}
        results["modelos"]["ensemble"] = {"familia": "ensemble", "features": "invariantes", "miembros": ENSEMBLE}

    for key, p in probas.items():
        entry = results["modelos"][key]
        entry["seleccion"] = selection_metrics(sel_y, p["select"], sel_w)
        for split, frame in eval_frames.items():
            entry[split] = split_metrics(frame["class_label"].to_numpy(), p[split], frame["vehicle_id"].to_numpy())
        print(f"[{key}] seleccion {entry['seleccion']['costo_por_1000_camiones']:.1f}/1000 | "
              f"val ahorro {entry['validation']['ahorro_pct']:.1f}% {entry['validation']['ahorro_ic95']} "
              f"riesgo {entry['validation']['riesgo_medio_predicho']:.3f} | test ahorro {entry['test']['ahorro_pct']:.1f}% "
              f"{entry['test']['ahorro_ic95']} riesgo {entry['test']['riesgo_medio_predicho']:.3f}", flush=True)

    # Probabilidades de todos los modelos en validacion y test (no versionadas): permiten rehacer tablas y figuras
    # sin reentrenar.
    pd.DataFrame([
        {"model": key, "split": split, "vehicle_id": int(v), "class_label": int(y), **{f"p{k}": float(p[k]) for k in range(5)}}
        for key, pr in probas.items() for split, frame in eval_frames.items()
        for v, y, p in zip(frame["vehicle_id"], frame["class_label"], pr[split])
    ]).to_parquet(PROCESSED_DIR / "scania_all_model_probabilities.parquet")

    invariant_keys = [k for k in results["modelos"] if not k.endswith("_escala_cruda")]
    selected = min(invariant_keys, key=lambda k: results["modelos"][k]["seleccion"]["costo_por_1000_camiones"])
    results["seleccionado"] = selected
    print(f"[Seleccion] {selected}", flush=True)

    # --- detalle del modelo seleccionado (lo que usan la pagina y la API)
    clf_block, by_split = {"modelo": selected}, {}
    for split, frame in eval_frames.items():
        y_split, proba = frame["class_label"].to_numpy(), probas[selected][split]
        by_split[split] = (y_split, proba)
        block = evaluate_decisions(y_split, proba)
        block.update({
            "n_vehiculos": int(len(frame)),
            "auc_algun_riesgo": float(roc_auc_score(y_split > 0, 1.0 - proba[:, 0])),
            "sensibilidad_costo_visita": K.sensitivity(y_split, proba),
            "costo_visita_equilibrio": K.break_even(y_split, proba),
            "calibracion_tabla": K.calibration_table(y_split, proba),
            "calibracion_global": K.calibration_error(y_split, proba),
        })
        clf_block[split] = block

    # Platt con protocolo fijado de antemano: se ajusta en un conjunto y se evalua en el OTRO, en las dos direcciones.
    platt = {s: K.fit_platt(by_split[s][1], by_split[s][0]) for s in by_split}
    clf_block["platt"] = platt
    for target, source in (("test", "validation"), ("validation", "test")):
        y_t, proba_t = by_split[target]
        proba_p = K.apply_platt(proba_t, **platt[source])
        clf_block[target]["recalibrado_con_el_otro_conjunto"] = {
            "ajustado_en": source, "parametros": platt[source],
            **{k: v for k, v in evaluate_decisions(y_t, proba_p).items() if k == "costo_esperado_minimo"},
            "sensibilidad_costo_visita": K.sensitivity(y_t, proba_p),
            "costo_visita_equilibrio": K.break_even(y_t, proba_p),
            "calibracion_tabla": K.calibration_table(y_t, proba_p),
            "calibracion_global": K.calibration_error(y_t, proba_p),
        }
    results["clasificador"] = clf_block
    export_predictions(selected)

    # --- 3. tiempo restante (solo camiones con reparacion observada), con las features invariantes
    ttf_cols = feature_sets["invariantes"]
    failed = full.filter(pl.col("time_to_failure").is_not_null())
    failed_ids = np.array(sorted(set(failed["vehicle_id"].to_list())))
    ftr, fho = vehicle_split(failed_ids, np.zeros(len(failed_ids)), test_size=0.25)
    ftr_pdf = with_specs(failed.filter(pl.col("vehicle_id").is_in(list(ftr))).to_pandas(), train_specs, cats)
    fho_pdf = with_specs(failed.filter(pl.col("vehicle_id").is_in(list(fho))).to_pandas(), train_specs, cats)
    ttf = train_time_to_failure(ftr_pdf.assign(**clean(ftr_pdf, ttf_cols)), fho_pdf.assign(**clean(fho_pdf, ttf_cols)), ttf_cols)
    pred = np.expm1(ttf.predict(clean(fho_pdf, ttf_cols))).clip(min=0)
    truth = fho_pdf["time_to_failure"].to_numpy()
    baseline = np.full_like(truth, np.median(ftr_pdf["time_to_failure"]))
    near = truth <= 48
    results["tiempo_restante"] = {
        "camiones_fallados_train": len(ftr), "camiones_fallados_holdout": len(fho),
        "mae": float(np.abs(pred - truth).mean()), "mae_baseline_mediana": float(np.abs(baseline - truth).mean()),
        "mae_ultimas_48": float(np.abs(pred[near] - truth[near]).mean()),
        "mae_ultimas_48_baseline": float(np.abs(baseline[near] - truth[near]).mean()),
        "lecturas_holdout": int(len(truth)), "lecturas_ultimas_48": int(near.sum()),
    }
    print(f"[Tiempo restante] MAE={results['tiempo_restante']['mae']:.1f} (baseline {results['tiempo_restante']['mae_baseline_mediana']:.1f})")

    # --- 4. supervivencia con entrada tardia
    train_ids = parts["fit"] | parts["early_stop"]
    hold_ids = parts["select"]
    design = cox_design(landmark_table(readouts_tr, tte, train_specs), cats)
    cox = {}
    for name, (sp, us) in {"especificaciones": (True, False), "uso_temprano": (False, True), "ambos": (True, True)}.items():
        fit = fit_cox(design, train_ids, hold_ids, use_specs=sp, use_usage=us)
        cox[name] = {"c_index": round(fit["c_index"], 4), "n_covariables": fit["n_covariates"]}
        if name == "ambos":
            joblib.dump({k: fit[k] for k in ("model", "mu", "sd")}, MODELS_DIR / "coxph_landmark.joblib")
    survival_models = {"cox": cox["ambos"]["c_index"]}
    survival_models["weibull_aft"] = round(fit_weibull_aft(design, train_ids, hold_ids)["c_index"], 4)
    survival_models["random_survival_forest"] = round(fit_random_survival_forest(design, train_ids, hold_ids)["c_index"], 4)
    results["supervivencia"] = {"landmark": LANDMARK, "vehiculos_en_riesgo": int(len(design)), **cox,
                                "modelos_covariables_ambas": survival_models}
    print(f"[Supervivencia] C-index: {survival_models}", flush=True)

    # --- 5. SHAP del modelo de tiempo restante
    import shap

    sample = clean(fho_pdf, ttf_cols).sample(n=min(1000, len(fho_pdf)), random_state=SEED)
    shap_values = shap.TreeExplainer(ttf).shap_values(sample)
    imp = pd.DataFrame({"feature": ttf_cols, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp.to_csv(REPORTS_DIR / "shap_time_to_failure.csv", index=False)
    results["shap_top"] = imp.head(10).round(4).to_dict(orient="records")

    # --- artefactos para la API: el modelo seleccionado y sus columnas
    joblib.dump(fitted[selected], MODELS_DIR / "risk_classifier.joblib")
    joblib.dump(ttf, MODELS_DIR / "time_to_failure.joblib")
    (MODELS_DIR / "feature_columns.json").write_text(json.dumps(feature_sets["invariantes"]), encoding="utf-8")
    readout_columns = [c for c in readouts_tr.columns if c != "vehicle_id"]
    (MODELS_DIR / "readout_columns.json").write_text(json.dumps(readout_columns), encoding="utf-8")
    (MODELS_DIR / "spec_categories.json").write_text(json.dumps(cats), encoding="utf-8")
    pooled = K.fit_platt(np.vstack([by_split[s][1] for s in by_split]), np.concatenate([by_split[s][0] for s in by_split]))
    results["clasificador"]["platt_para_api"] = pooled
    (MODELS_DIR / "platt.json").write_text(json.dumps(pooled), encoding="utf-8")
    (REPORTS_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resultados en {REPORTS_DIR / 'results.json'}")
    return results


if __name__ == "__main__":
    import sys

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(force_features="--force" in sys.argv, models=args or None)
