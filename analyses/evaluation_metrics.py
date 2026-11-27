"""Compare the Exercise 1 models on matched, exposure-weighted policies."""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc

from analyses.claims_evaluation import (
    TARGET,
    WEIGHT,
    evaluation_arrays,
    prepare_model_data,
    sample_frames,
    split_fingerprint,
)
from analyses.lgbm_baseline import DEFAULT_PARAMETERS
from analyses.lgbm_monotonicity import FEATURES, _fit_pipeline, _predict
from analyses.model_comparison import lorenz_curve
from ps3.evaluation import evaluate_predictions

COLUMNS = {name: f"pp_t_lgbm_{name}" for name in ("unconstrained", "constrained")}
TOLERANCE = 1e-8


def align_predictions(test, predictions):
    """Require identical policy sets and reindex predictions to test policy order."""
    if "IDpol" not in predictions or any(
        c not in predictions for c in COLUMNS.values()
    ):
        raise ValueError("Prediction file must contain IDpol and both model columns")
    if (
        not test.index.is_unique
        or test.index.hasnans
        or predictions["IDpol"].isna().any()
    ):
        raise ValueError("Policy IDs must be unique and nonmissing")
    indexed = predictions.set_index("IDpol", verify_integrity=False)
    if (
        not indexed.index.is_unique
        or len(indexed) != len(test)
        or not indexed.index.sort_values().equals(test.index.sort_values())
    ):
        raise ValueError("Prediction policy IDs must match test policy IDs exactly")
    return indexed.loc[test.index, list(COLUMNS.values())]


def compare_predictions(test, predictions):
    """Score both predictions and use the PS3 policy-count Lorenz Gini."""
    scores = {}
    for name, column in COLUMNS.items():
        y, pred, weights = evaluation_arrays(test, predictions[column])
        result = evaluate_predictions(y, pred, weights)["value"]
        x, claims = lorenz_curve(y, pred, weights)
        scores[name] = pd.concat([result, pd.Series({"Gini": 1 - 2 * auc(x, claims)})])
    return pd.DataFrame(scores).rename_axis("metric")


def run_evaluation_metrics(
    frame, results_dir="results", ex1_metrics_path=None, predictions_path=None
):
    """Refit locked Ex1 models if needed; validate and save matched comparisons."""
    results_dir = Path(results_dir)
    ex1_path = (
        Path(ex1_metrics_path)
        if ex1_metrics_path
        else results_dir / "ps4_ex1_monotonicity_metrics.json"
    )
    handoff = json.loads(ex1_path.read_text(encoding="utf-8"))
    data = prepare_model_data(frame)
    train, test = sample_frames(data)
    fingerprint = split_fingerprint(train, test)
    if (
        fingerprint != handoff["split_fingerprint"]
        or len(train) != handoff["train_rows"]
        or len(test) != handoff["test_rows"]
        or handoff["target_definition"] != "ClaimAmountCut / Exposure"
        or handoff["weights"] != "Exposure"
        or handoff["tweedie_power"] != 1.5
    ):
        raise ValueError("Data, split or scoring conventions do not match Ex1 handoff")
    if data[FEATURES].isna().any().any():
        raise ValueError("Model features must not contain missing values")
    numeric = data.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Numeric data must be finite")
    if predictions_path is None:
        candidate = ex1_path.parent / "ps4_ex1_test_predictions.csv"
        predictions_path = candidate if candidate.is_file() else None
    parameters = {
        name: {**DEFAULT_PARAMETERS, **handoff["models"][name]["selected_parameters"]}
        for name in COLUMNS
    }
    if predictions_path is not None:
        raw_predictions = pd.read_csv(predictions_path)
        source = f"Ex1 prediction file: {predictions_path}"
    else:
        raw_predictions = pd.DataFrame({"IDpol": test.index})
        for name, constrained in (("unconstrained", False), ("constrained", True)):
            print(f"Refitting {name} LightGBM with locked Ex1 parameters", flush=True)
            model, names, constraints = _fit_pipeline(
                train[FEATURES],
                train[TARGET].to_numpy(),
                train[WEIGHT].to_numpy(),
                handoff["models"][name]["selected_parameters"],
                constrained,
            )
            expected_names = handoff["models"][name]["transformed_feature_names"]
            expected_constraints = handoff["models"][name][
                "monotone_constraints_in_feature_order"
            ]
            if names != expected_names or constraints != expected_constraints:
                raise ValueError(
                    "Refitted feature order or constraint differs from Ex1"
                )
            raw_predictions[COLUMNS[name]] = _predict(model, test[FEATURES])
        source = "Refitted Ex1 pipelines on full canonical training sample with locked parameters; no CV"
    predictions = align_predictions(test, raw_predictions)
    metrics = compare_predictions(test, predictions)
    checks = {}
    for name in COLUMNS:
        recorded = handoff["models"][name]["test_weighted_tweedie_deviance"]
        actual = float(metrics.loc["Tweedie deviance", name])
        checks[name] = {
            "ex1": recorded,
            "recomputed": actual,
            "difference": actual - recorded,
        }
        if not np.isclose(actual, recorded, rtol=TOLERANCE, atol=TOLERANCE):
            raise ValueError(
                f"{name} deviance differs from Ex1: {actual} vs {recorded}; inspect data, model provenance and package versions"
            )
    payload = {
        "exercise": "PS4 Ex3 - Evaluation metrics",
        "split_fingerprint": fingerprint,
        "training_rows": len(train),
        "test_rows": len(test),
        "target": "PurePremium",
        "target_definition": "ClaimAmountCut / Exposure",
        "weight": "Exposure",
        "tweedie_power": 1.5,
        "prediction_source": source,
        "prediction_alignment": "unique, nonmissing IDpol; exact test ID set; reindexed by IDpol",
        "model_parameters": parameters,
        "constrained_feature": "BonusMalus (+1); all other transformed features 0",
        "metric_definitions": {
            "Bias": "sum(w*(prediction-actual))/sum(w*actual); positive means overprediction",
            "Tweedie deviance": "sklearn weighted mean Tweedie deviance, power 1.5",
            "MAE": "sum(w*abs(actual-prediction))/sum(w)",
            "RMSE": "sqrt(sum(w*(actual-prediction)^2)/sum(w))",
            "Gini": "PS3 unnormalized 1 - 2*AUC of claims Lorenz curve by policy count; tied predictions grouped",
        },
        "metrics": metrics.to_dict(),
        "ex1_deviance_crosscheck": checks,
        "deviance_tolerance": {"relative": TOLERANCE, "absolute": TOLERANCE},
        "versions": {
            name: version(name)
            for name in ["numpy", "pandas", "scikit-learn", "lightgbm"]
        },
        "reproduction": "python -m analyses.ps3_script ps4-ex3 --data data/final_shared_normalized_20261120.parquet --results-dir results --ex1-metrics-path results/ps4_ex1_monotonicity_metrics.json",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(results_dir / "ps4_ex3_metrics.csv")
    (results_dir / "ps4_ex3_metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metrics, payload
