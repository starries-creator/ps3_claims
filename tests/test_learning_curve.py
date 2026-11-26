
"""Tests for PS4 Exercise 2 learning curve."""

import numpy as np
import pandas as pd
import pytest

from analyses.learning_curve import (
    fit_learning_curve,
    load_ex1_report,
    plot_learning_curve,
)
from analyses.lgbm_monotonicity import FEATURES
from analyses.lgbm_baseline import TARGET, WEIGHT, TWEEDIE_POWER


@pytest.fixture(scope="module")
def fitted_example():
    """Fit a small constrained model on synthetic policy data."""

    rng = np.random.default_rng(42)
    n = 400

    frame = pd.DataFrame(
        {
            "VehBrand": rng.choice(["B1", "B2"], n),
            "VehGas": rng.choice(["Diesel", "Regular"], n),
            "Region": rng.choice(["R11", "R21"], n),
            "Area": rng.choice(["A", "B"], n),
            "DrivAge": rng.integers(0, 4, n),
            "VehAge": rng.integers(0, 3, n),
            "VehPower": rng.integers(4, 8, n),
            "BonusMalus": rng.integers(50, 150, n),
            "Density": rng.uniform(10, 1000, n),
            WEIGHT: rng.uniform(0.1, 1.0, n),
        }
    )

    frame[TARGET] = (
        10
        + frame["BonusMalus"] * 0.5
        + rng.uniform(0, 20, n)
    )

    train = frame.iloc[:320].copy()
    test = frame.iloc[320:].copy()

    # Test-only categories must not affect fitted preprocessing.
    test["VehBrand"] = "B99"

    parameters = {
        "learning_rate": 0.03,
        "n_estimators": 10,
        "n_jobs": 1,
    }

    pipeline, names, constraints = fit_learning_curve(
        train,
        test,
        parameters,
    )

    return pipeline, names, constraints, train, test


def test_ex1_selected_parameters():
    report = load_ex1_report()

    parameters = report["models"]["constrained"][
        "selected_parameters"
    ]

    assert "learning_rate" in parameters
    assert "n_estimators" in parameters
    assert parameters["learning_rate"] > 0
    assert parameters["n_estimators"] > 0


def test_monotonicity_constraint(fitted_example):
    pipeline, names, constraints, _, _ = fitted_example

    assert len(names) == len(constraints)

    bonus_index = names.index("numeric__BonusMalus")

    assert constraints[bonus_index] == 1
    assert all(
        value == 0
        for index, value in enumerate(constraints)
        if index != bonus_index
    )

    model = pipeline.named_steps["model"]

    assert list(model.get_params()["monotone_constraints"]) == constraints


def test_training_only_preprocessing(fitted_example):
    pipeline, names, _, train, test = fitted_example

    preprocessor = pipeline.named_steps["preprocess"]

    X_train = preprocessor.transform(train[FEATURES])
    X_test = preprocessor.transform(test[FEATURES])

    assert X_train.shape[1] == X_test.shape[1]

    assert not any("VehBrand_B99" in name for name in names)


def test_evaluation_history(fitted_example):
    pipeline, _, _, _, _ = fitted_example

    model = pipeline.named_steps["model"]
    history = model.evals_result_

    assert "train" in history
    assert "test" in history

    assert len(history["train"]["tweedie"]) == 10
    assert len(history["test"]["tweedie"]) == 10

    assert np.isfinite(history["train"]["tweedie"]).all()
    assert np.isfinite(history["test"]["tweedie"]).all()


def test_weighted_evaluation_metric(fitted_example):
    pipeline, _, _, train, test = fitted_example

    model = pipeline.named_steps["model"]

    for name, frame in [("train", train), ("test", test)]:

        predictions = pipeline.predict(frame[FEATURES])

        y = frame[TARGET].to_numpy(dtype=float)
        w = frame[WEIGHT].to_numpy(dtype=float)
        p = TWEEDIE_POWER

        losses = (
            -y * predictions ** (1 - p) / (1 - p)
            + predictions ** (2 - p) / (2 - p)
        )

        expected = np.average(losses, weights=w)

        recorded = model.evals_result_[name]["tweedie"][-1]

        assert recorded == pytest.approx(expected, rel=1e-5)


def test_predictions_are_monotonic(fitted_example):
    pipeline, _, _, train, _ = fitted_example

    base = train.iloc[[0]][FEATURES]

    grid = pd.concat([base] * 101, ignore_index=True)

    grid["BonusMalus"] = np.arange(50, 151)

    predictions = pipeline.predict(grid)

    assert (np.diff(predictions) >= -1e-8).all()


def test_plot_output(fitted_example, tmp_path):
    pipeline, _, _, _, _ = fitted_example

    model = pipeline.named_steps["model"]

    output = tmp_path / "learning_curve.png"

    plot_learning_curve(model, output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_parameters_remain_fixed(fitted_example):
    pipeline, _, _, _, _ = fitted_example

    model = pipeline.named_steps["model"]

    assert model.get_params()["learning_rate"] == 0.03
    assert model.get_params()["n_estimators"] == 10
    assert model.n_estimators_ == 10


def test_test_outcomes_do_not_change_fitted_model(fitted_example):
    pipeline, _, _, train, test = fitted_example

    changed_test = test.copy()
    changed_test[TARGET] = changed_test[TARGET] * 2

    parameters = {
        "learning_rate": 0.03,
        "n_estimators": 10,
        "n_jobs": 1,
    }

    new_pipeline, _, _ = fit_learning_curve(
        train,
        changed_test,
        parameters,
    )

    original_predictions = pipeline.predict(test[FEATURES])
    new_predictions = new_pipeline.predict(test[FEATURES])

    np.testing.assert_allclose(
        original_predictions,
        new_predictions,
        rtol=1e-8,
        atol=1e-8,
    )
