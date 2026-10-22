"""Catalogo de modelos y bucle de entrenamiento de las redes, sobre datos sinteticos pequenos."""
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score

from src.models import nets
from src.models.zoo import MODEL_NAMES, TabularModel, class_balanced_weights, signed_log1p

FAST = {"mlp_train": {"min_epochs": 5, "max_epochs": 8, "patience": 2}, "n_estimators": 50,
        "lightgbm": {"n_estimators": 200}, "xgboost": {"n_estimators": 200}, "catboost": {"iterations": 200}}


def make_data(n=1200, n_classes=2, seed=0, with_category=False):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"x{i}" for i in range(6)])
    X.loc[rng.random(n) < 0.1, "x3"] = np.nan  # faltantes
    logit = 1.5 * X["x0"].fillna(0) - X["x1"] + 0.5 * X["x2"]
    if with_category:
        X["spec"] = pd.Categorical(rng.choice(["a", "b", "c"], n), categories=["a", "b", "c"])
        logit = logit + (X["spec"] == "a") * 1.0
    if n_classes == 2:
        y = (logit + rng.logistic(size=n) > 0).astype(int).to_numpy()
    else:
        y = np.digitize(logit + rng.normal(0, 0.5, n), np.quantile(logit, np.linspace(0, 1, n_classes + 1)[1:-1]))
    return X, y


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_learns_a_binary_signal(name):
    X, y = make_data()
    model = TabularModel(name, 2, params=FAST).fit(X[:800], y[:800], None, X[800:1000], y[800:1000], None)
    p = model.predict_proba(X[1000:])
    assert p.shape == (200, 2) and np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert roc_auc_score(y[1000:], p[:, 1]) > 0.75


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_handles_five_classes_and_a_categorical_column(name):
    X, y = make_data(n_classes=5, with_category=True)
    model = TabularModel(name, 5, categorical=["spec"], params=FAST)
    model.fit(X[:800], y[:800], np.ones(800), X[800:1000], y[800:1000], np.ones(200))
    p = model.predict_proba(X[1000:])
    assert p.shape == (200, 5) and np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert (p.argmax(axis=1) == y[1000:]).mean() > 0.3  # azar = 0.2


def test_unknown_model_fails_loudly():
    with pytest.raises(ValueError):
        TabularModel("svm", 2)


def test_class_balanced_weights_give_each_class_the_same_total():
    y = np.array([0] * 90 + [1] * 10)
    w = class_balanced_weights(y)
    assert w[y == 0].sum() == pytest.approx(w[y == 1].sum()) and w.sum() == pytest.approx(len(y))


def test_signed_log1p_keeps_the_sign_and_order():
    x = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    out = signed_log1p(x)
    assert (np.sign(out) == np.sign(x)).all() and (np.diff(out) > 0).all()


# --------------------------------------------------------------------------- bucle de entrenamiento
def _toy_regression(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, 3)).astype(np.float32)
    y = (X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.1, 400)).astype(np.float32).reshape(-1, 1)
    return X, y


def test_training_runs_at_least_min_epochs_and_restores_the_best_checkpoint(tmp_path):
    X, y = _toy_regression()
    model = torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    cfg = nets.TrainConfig(min_epochs=12, max_epochs=40, patience=1, batch_size=64, checkpoint_path=tmp_path / "best.pt")
    loss = torch.nn.MSELoss(reduction="none")
    hist = nets.fit_network(model, loss, X[:300], y[:300], None, X[300:], y[300:], None, cfg)
    assert hist.epochs_run >= 12  # la paciencia de 1 no puede cortar antes del minimo
    assert hist.val_loss[hist.best_epoch - 1] == pytest.approx(min(hist.val_loss))
    final = nets.evaluate_loss(model, loss, torch.as_tensor(X[300:]), torch.as_tensor(y[300:]), torch.ones(100))
    assert final == pytest.approx(min(hist.val_loss), rel=1e-4)  # el modelo quedo en el mejor estado, no en el ultimo
    saved = torch.load(tmp_path / "best.pt")
    assert all(torch.equal(saved[k], v) for k, v in model.state_dict().items())


def test_training_is_reproducible_with_the_same_seed():
    X, y = _toy_regression()
    outs = []
    for _ in range(2):
        nets.set_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
        nets.fit_network(model, torch.nn.MSELoss(reduction="none"), X[:300], y[:300], None, X[300:], y[300:], None,
                         nets.TrainConfig(min_epochs=3, max_epochs=3, batch_size=64, seed=0))
        outs.append(nets.predict(model, X[300:]))
    assert np.allclose(outs[0], outs[1])
