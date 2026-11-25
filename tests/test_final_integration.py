"""Contracts for the reproducible final report and member adapters."""

import json

import numpy as np
import pandas as pd
import pytest

from analyses import lgbm_monotonicity, claims_evaluation
from analyses.model_comparison import save_comparison, write_metrics_csv
from analyses.ps3_script import load_model_data
from analyses import ps3_script


def test_final_report_keeps_relative_error_and_verification_source(tmp_path):
    frame = pd.DataFrame({
        "IDpol": [0, 1, 7, 12], "PurePremium": [0., 4., 1., 2.],
        "ClaimAmountCut": [0., 4., .5, 2.], "Exposure": [.25, 1., .5, 1.],
        "sample": ["train", "train", "test", "test"],
    })
    train, test = claims_evaluation.sample_frames(frame)
    metrics = claims_evaluation.model_metrics(
        "tweedie_glm", train, test,
        pd.Series([1., 4.], index=train.index),
        pd.Series([2., 2.], index=test.index),
        parameters={}, selection_method="fixture",
    )
    metrics["verification_status"] = "locally_trained"
    claims_evaluation.write_model_metrics(metrics, tmp_path)
    report = pd.read_csv(write_metrics_csv(tmp_path))
    assert report.loc[0, "relative_claim_amount_error"] == pytest.approx(.2)
    assert report.loc[0, "verification_status"] == "locally_trained"
    assert metrics["train_actual_claim_amount"] == 4.


def test_report_rejects_mixed_dataset_fingerprints(tmp_path):
    frame = pd.DataFrame({
        "IDpol": [0, 1, 7, 12], "PurePremium": [0., 4., 1., 2.],
        "ClaimAmountCut": [0., 4., .5, 2.], "Exposure": [.25, 1., .5, 1.],
        "sample": ["train", "train", "test", "test"],
    })
    train, test = claims_evaluation.sample_frames(frame)
    for name, fingerprint in [("tweedie_glm", "a" * 64), ("lightgbm_tuned", "b" * 64)]:
        metrics = claims_evaluation.model_metrics(
            name, train, test, pd.Series(2., index=train.index),
            pd.Series(2., index=test.index), parameters={}, selection_method="fixture",
        )
        metrics["data_fingerprint"] = fingerprint
        claims_evaluation.write_model_metrics(metrics, tmp_path)
    with pytest.raises(ValueError, match="different data"):
        write_metrics_csv(tmp_path)


@pytest.fixture
def synthetic_claims():
    rng = np.random.default_rng(12)
    n = 180
    return pd.DataFrame({
        **{c: rng.choice(["a", "b"], n) for c in claims_evaluation.CATEGORICALS},
        "IDpol": np.arange(n), "BonusMalus": rng.uniform(50, 100, n),
        "Density": rng.uniform(1, 200, n), "Exposure": rng.uniform(.1, 1, n),
        "ClaimAmountCut": rng.uniform(0, 10, n),
    })


def test_e_adapter_runs_original_cv_and_shared_scoring(tmp_path, synthetic_claims):
    frame = load_model_data(synthetic_claims)
    metrics, predictions = claims_evaluation.run_tuned_lightgbm(frame, tmp_path)
    train, test = claims_evaluation.sample_frames(frame)
    cv = json.loads((tmp_path / "lgbm_tuning_cv.json").read_text())
    assert len(cv["candidate_results"]) == 6
    assert all(len(r["fold_weighted_tweedie_deviance"]) == 3 for r in cv["candidate_results"])
    assert predictions.index.equals(test.index)
    assert metrics["split_fingerprint"] == claims_evaluation.split_fingerprint(train, test)
    assert metrics["test_deviance"] == pytest.approx(claims_evaluation.evaluate_sample(test, predictions)["deviance"])


def test_ex1_handoff_returns_both_fitted_pipelines(tmp_path, synthetic_claims, monkeypatch):
    monkeypatch.setattr(
        lgbm_monotonicity, "PARAMETER_GRID", [{"n_estimators": 8, "learning_rate": 0.05}]
    )
    models, _, _, constrained_predictions = lgbm_monotonicity.run_monotonicity_comparison(
        synthetic_claims, tmp_path
    )
    _, test = claims_evaluation.sample_frames(load_model_data(synthetic_claims))
    saved = pd.read_csv(tmp_path / "ps4_ex1_test_predictions.csv").set_index("IDpol")

    assert set(models) == {"constrained_lgbm", "unconstrained_lgbm"}
    assert "monotone_constraints" not in models["unconstrained_lgbm"].named_steps["model"].get_params()
    pd.testing.assert_index_equal(saved.index, test.index)
    pd.testing.assert_series_equal(constrained_predictions, saved["pp_t_lgbm_constrained"])
    features = test[lgbm_monotonicity.FEATURES]
    for variant in ("constrained", "unconstrained"):
        pipeline = models[f"{variant}_lgbm"]
        np.testing.assert_allclose(pipeline.predict(features), saved[f"pp_t_lgbm_{variant}"])


def test_offline_summary_preserves_prediction_verified_gini(tmp_path):
    frame = pd.DataFrame({
        "IDpol": [0, 1, 7, 12], "PurePremium": [0., 4., 1., 2.],
        "ClaimAmountCut": [0., 4., .5, 2.], "Exposure": [.25, 1., .5, 1.],
        "sample": ["train", "train", "test", "test"],
    })
    train, test = claims_evaluation.sample_frames(frame)
    predicted = pd.Series([1., 2.], index=test.index, name="PurePremiumPredicted")
    metrics = claims_evaluation.model_metrics(
        "tweedie_glm", train, test, pd.Series(2., index=train.index),
        predicted, parameters={}, selection_method="fixture",
    )
    claims_evaluation.write_model_metrics(metrics, tmp_path / "evaluated")
    predicted.to_csv(tmp_path / "tweedie_glm_predictions.csv")
    save_comparison(train, test, tmp_path)
    summary = pd.read_csv(write_metrics_csv(tmp_path))
    assert summary.loc[0, "test_gini"] == pytest.approx(.3)


def test_partial_comparison_retains_other_models_published_gini(tmp_path):
    frame = pd.DataFrame({
        "IDpol": [0, 1, 7, 12], "PurePremium": [0., 4., 1., 2.],
        "ClaimAmountCut": [0., 4., .5, 2.], "Exposure": [.25, 1., .5, 1.],
        "sample": ["train", "train", "test", "test"],
    })
    train, test = claims_evaluation.sample_frames(frame)
    for model, values in [("tweedie_glm", [1., 2.]), ("lightgbm_initial", [2., 1.])]:
        pred = pd.Series(values, index=test.index, name="PurePremiumPredicted")
        metrics = claims_evaluation.model_metrics(
            model, train, test, pd.Series(2., index=train.index), pred,
            parameters={}, selection_method="fixture",
        )
        claims_evaluation.write_model_metrics(metrics, tmp_path / "evaluated")
        pred.to_csv(tmp_path / f"{model}_predictions.csv")
    save_comparison(train, test, tmp_path)
    (tmp_path / "lightgbm_initial_predictions.csv").unlink()
    save_comparison(train, test, tmp_path)
    report = pd.read_csv(tmp_path / "metrics.csv").set_index("model")
    assert report.loc["lightgbm_initial", "test_gini"] == pytest.approx(-.3)


def test_cli_bounds_remote_read_and_restores_socket_timeout(monkeypatch):
    import socket

    original = socket.getdefaulttimeout()
    seen = []
    def loader(frame):
        seen.append(socket.getdefaulttimeout())
        return pd.DataFrame({"IDpol": [0, 7], "sample": ["train", "test"]})
    monkeypatch.setattr(ps3_script, "load_model_data", loader)
    ps3_script.main(["check-data", "--network-timeout", "300"])
    assert seen == [300.]
    assert socket.getdefaulttimeout() == original
