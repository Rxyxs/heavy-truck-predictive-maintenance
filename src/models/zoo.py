"""Catalogo comun de clasificadores tabulares, con una sola interfaz para todos.

Cada modelo se entrena con ``fit(X_tr, y_tr, w_tr, X_va, y_va, w_va)`` y devuelve probabilidades por clase
con ``predict_proba``. El conjunto ``*_va`` es SOLO para la parada temprana (arboles por boosting y redes);
nunca es el conjunto con que se informa el desempeno. Los modelos que no lo necesitan lo ignoran.

Los hiperparametros estan fijados de antemano y son los mismos en todos los datasets salvo que el pipeline
pase ``params`` explicitos; no se ajustan mirando validacion ni prueba.

Familias:

* ``logistic``       regresion logistica L2 sobre log1p(|x|) con signo, imputacion por mediana con indicadores de faltante.
* ``random_forest``  bosque aleatorio sobre los valores imputados.
* ``lightgbm``, ``xgboost``, ``catboost``  boosting de arboles; manejan faltantes y categoricas de forma nativa.
* ``mlp``            red densa en PyTorch (``src.models.nets``): minimo 100 epocas, parada temprana y checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

MODEL_NAMES = ["logistic", "random_forest", "lightgbm", "xgboost", "catboost", "mlp"]
LABELS = {
    "logistic": "Logistic regression", "random_forest": "Random forest", "lightgbm": "LightGBM",
    "xgboost": "XGBoost", "catboost": "CatBoost", "mlp": "MLP (PyTorch)", "ensemble": "Ensemble (GBM mean)",
}
SEED = 42


def signed_log1p(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    """Peso por fila inversamente proporcional a la frecuencia de su clase (suma total = n)."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    per_class = len(y) / (len(classes) * counts)
    return per_class[np.searchsorted(classes, y)]


def make_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    """Numericas: log1p con signo (si ``scale``), mediana + indicador de faltante, estandarizacion.
    Categoricas: one-hot, agrupando las categorias con menos de 50 filas."""
    steps = []
    if scale:
        steps.append(("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")))
    steps.append(("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)))
    if scale:
        steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(steps), numeric)]
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50, sparse_output=False),
                             categorical))
    return ColumnTransformer(transformers, sparse_threshold=0.0)


def _as_object(X: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    out = X.copy()
    for c in categorical:
        out[c] = out[c].astype(str)
    return out


class TabularModel:
    """Envoltorio comun. ``n_classes`` = 2 para binario (devuelve dos columnas), 5 para SCANIA Component X."""

    def __init__(self, name: str, n_classes: int, categorical: list[str] | None = None, seed: int = SEED,
                 params: dict | None = None, checkpoint_dir: Path | None = None):
        if name not in MODEL_NAMES:
            raise ValueError(f"modelo desconocido: {name}")
        self.name, self.n_classes, self.seed = name, n_classes, seed
        self.categorical = list(categorical or [])
        self.params = dict(params or {})
        self.checkpoint_dir = checkpoint_dir
        self.info: dict = {}

    # ------------------------------------------------------------------ entrenamiento
    def fit(self, X_tr: pd.DataFrame, y_tr, w_tr=None, X_va: pd.DataFrame | None = None, y_va=None, w_va=None):
        y_tr = np.asarray(y_tr).astype(int)
        w_tr = np.ones(len(y_tr)) if w_tr is None else np.asarray(w_tr, dtype=float)
        if X_va is not None:
            y_va = np.asarray(y_va).astype(int)
            w_va = np.ones(len(y_va)) if w_va is None else np.asarray(w_va, dtype=float)
        self.columns_ = list(X_tr.columns)
        self.numeric_ = [c for c in self.columns_ if c not in self.categorical]
        getattr(self, f"_fit_{self.name}")(X_tr, y_tr, w_tr, X_va, y_va, w_va)
        return self

    def _fit_logistic(self, X_tr, y_tr, w_tr, *_):
        clf = LogisticRegression(C=self.params.get("C", 0.1), max_iter=3000, tol=1e-4)
        self.model_ = Pipeline([("prep", make_preprocessor(self.numeric_, self.categorical, scale=True)), ("clf", clf)])
        self.model_.fit(X_tr, y_tr, clf__sample_weight=w_tr)
        self.info = {"iterations": int(np.max(clf.n_iter_))}

    def _fit_random_forest(self, X_tr, y_tr, w_tr, *_):
        clf = RandomForestClassifier(
            n_estimators=self.params.get("n_estimators", 400), min_samples_leaf=self.params.get("min_samples_leaf", 3),
            max_features="sqrt", n_jobs=-1, random_state=self.seed,
        )
        self.model_ = Pipeline([("prep", make_preprocessor(self.numeric_, self.categorical, scale=False)), ("clf", clf)])
        self.model_.fit(X_tr, y_tr, clf__sample_weight=w_tr)
        self.info = {"trees": clf.n_estimators}

    def _fit_lightgbm(self, X_tr, y_tr, w_tr, X_va, y_va, w_va):
        import lightgbm as lgb

        params = dict(n_estimators=3000, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.3, min_child_samples=20, random_state=self.seed, verbosity=-1, n_jobs=-1)
        params.update(self.params.get("lightgbm", {}))
        objective = "binary" if self.n_classes == 2 else "multiclass"
        extra = {} if self.n_classes == 2 else {"num_class": self.n_classes}
        self.model_ = lgb.LGBMClassifier(objective=objective, **extra, **params)
        self.model_.fit(X_tr, y_tr, sample_weight=w_tr, eval_X=(X_va,), eval_y=(y_va,), eval_sample_weight=[w_va],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        self.info = {"best_iteration": int(self.model_.best_iteration_)}

    def _fit_xgboost(self, X_tr, y_tr, w_tr, X_va, y_va, w_va):
        import xgboost as xgb

        params = dict(n_estimators=3000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.5,
                      min_child_weight=1.0, tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
                      early_stopping_rounds=100, random_state=self.seed, n_jobs=-1)
        params.update(self.params.get("xgboost", {}))
        if self.n_classes == 2:
            params.update(objective="binary:logistic", eval_metric="logloss")
        else:
            params.update(objective="multi:softprob", num_class=self.n_classes, eval_metric="mlogloss")
        self.model_ = xgb.XGBClassifier(**params)
        self.model_.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], sample_weight_eval_set=[w_va],
                        verbose=False)
        self.info = {"best_iteration": int(self.model_.best_iteration)}

    def _fit_catboost(self, X_tr, y_tr, w_tr, X_va, y_va, w_va):
        from catboost import CatBoostClassifier, Pool

        params = dict(iterations=3000, learning_rate=0.05, depth=6, random_seed=self.seed, thread_count=-1,
                      od_type="Iter", od_wait=100, use_best_model=True, verbose=False, allow_writing_files=False)
        params.update(self.params.get("catboost", {}))
        params["loss_function"] = "Logloss" if self.n_classes == 2 else "MultiClass"
        self.model_ = CatBoostClassifier(**params)
        train = Pool(_as_object(X_tr, self.categorical), y_tr, weight=w_tr, cat_features=self.categorical or None)
        valid = Pool(_as_object(X_va, self.categorical), y_va, weight=w_va, cat_features=self.categorical or None)
        self.model_.fit(train, eval_set=valid)
        self.info = {"best_iteration": int(self.model_.get_best_iteration())}

    def _fit_mlp(self, X_tr, y_tr, w_tr, X_va, y_va, w_va):
        import torch

        from src.models import nets

        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.prep_ = make_preprocessor(self.numeric_, self.categorical, scale=True).fit(X_tr)
        A_tr = self.prep_.transform(X_tr).astype(np.float32)
        A_va = self.prep_.transform(X_va).astype(np.float32)
        cfg = nets.TrainConfig(**{"seed": self.seed, **self.params.get("mlp_train", {})})
        if self.checkpoint_dir is not None:
            cfg.checkpoint_path = Path(self.checkpoint_dir) / f"mlp_seed{self.seed}.pt"
        nets.set_seed(self.seed)
        self.net_ = nets.MLP(A_tr.shape[1], self.n_classes, **self.params.get("mlp_arch", {}))
        loss = torch.nn.CrossEntropyLoss(reduction="none")
        hist = nets.fit_network(self.net_, loss, A_tr, y_tr, w_tr, A_va, y_va, w_va, cfg, y_dtype=torch.long)
        self.info = {"best_epoch": hist.best_epoch, "epochs_run": hist.epochs_run,
                     "val_loss_best": float(min(hist.val_loss)), "n_inputs": int(A_tr.shape[1])}
        self.history_ = hist

    # ------------------------------------------------------------------ prediccion
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.columns_]
        if self.name == "mlp":
            from scipy.special import softmax

            from src.models import nets

            logits = nets.predict(self.net_, self.prep_.transform(X).astype(np.float32))
            return softmax(logits, axis=1)
        if self.name == "catboost":
            return self.model_.predict_proba(_as_object(X, self.categorical))
        return self.model_.predict_proba(X)


class MeanEnsemble:
    """Promedio simple de probabilidades de varios modelos ya entrenados (semillas de una familia, o familias)."""

    def __init__(self, models: list):
        self.models = models

    @property
    def info(self) -> dict:
        return {"members": len(self.models)}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)
