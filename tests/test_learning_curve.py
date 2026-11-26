
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


def test_cli_forwards_ex1_metrics_path(monkeypatch, tmp_path):
    """The CLI must forward an explicitly supplied Ex1 report path."""
    from analyses import ps3_script
    from analyses import learning_curve

    frame = pd.DataFrame()
    report_path = tmp_path / "custom_ex1_metrics.json"
    results_dir = tmp_path / "outputs"

    captured = {}

    def fake_run(data, output_dir, ex1_metrics_path=None):
        captured["frame"] = data
        captured["results_dir"] = output_dir
        captured["ex1_metrics_path"] = ex1_metrics_path
        return None, {}

    monkeypatch.setattr(
        ps3_script,
        "load_input_data",
        lambda *args: frame,
    )

    monkeypatch.setattr(
        learning_curve,
        "run_learning_curve",
        fake_run,
    )

    ps3_script.main(
        [
            "ps4-ex2",
            "--results-dir",
            str(results_dir),
            "--ex1-metrics-path",
            str(report_path),
        ]
    )

    assert captured["frame"] is frame
    assert captured["results_dir"] == results_dir
    assert captured["ex1_metrics_path"] == report_path


def test_ex1_report_defaults_to_results_dir(monkeypatch, tmp_path):
    """Ex2 should read the Ex1 report from its results directory."""
    from analyses import learning_curve

    report_path = tmp_path / "ps4_ex1_monotonicity_metrics.json"
    report_path.write_text("{}", encoding="utf-8")

    captured = {}

    class ReportLoaded(Exception):
        pass

    def fake_load(path):
        captured["path"] = path
        raise ReportLoaded

    monkeypatch.setattr(
        learning_curve,
        "load_ex1_report",
        fake_load,
    )

    with pytest.raises(ReportLoaded):
        learning_curve.run_learning_curve(
            pd.DataFrame(),
            results_dir=tmp_path,
        )

    assert captured["path"] == report_path


def test_missing_ex1_report_has_clear_error(tmp_path):
    """A missing Ex1 report should produce an informative error."""
    from analyses.learning_curve import run_learning_curve

    with pytest.raises(
        FileNotFoundError,
        match="Exercise 1 report not found",
    ):
        run_learning_curve(
            pd.DataFrame(),
            results_dir=tmp_path,
        )


def test_plot_uses_one_based_boosting_rounds(
    fitted_example,
    tmp_path,
    monkeypatch,
):
    """Both plotted curves must use rounds 1 through n."""
    import lightgbm as lgb

    pipeline, _, _, _, _ = fitted_example
    model = pipeline.named_steps["model"]

    original_plot_metric = lgb.plot_metric
    captured = {}

    def capture_plot(*args, **kwargs):
        ax = original_plot_metric(*args, **kwargs)
        captured["ax"] = ax
        return ax

    monkeypatch.setattr(
        lgb,
        "plot_metric",
        capture_plot,
    )

    plot_learning_curve(
        model,
        tmp_path / "learning_curve.png",
    )

    lines = captured["ax"].get_lines()

    assert len(lines) == 2

    expected_rounds = np.arange(1, 11)

    for line in lines:
        np.testing.assert_array_equal(
            line.get_xdata(),
            expected_rounds,
        )
