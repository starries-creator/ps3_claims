
"""PS4 Exercise 2: Learning curve for constrained LightGBM."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_tweedie_deviance
from sklearn.pipeline import Pipeline

from analyses.claims_evaluation import prepare_model_data
from analyses.lgbm_baseline import (
    DEFAULT_PARAMETERS,
    TARGET,
    WEIGHT,
    TWEEDIE_POWER,
    policy_split_fingerprint,
)
from analyses.lgbm_monotonicity import (
    FEATURES,
    BONUS_MALUS_CONSTRAINT,
    _new_preprocessor,
)


EX1_METRICS_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "ps4_ex1_monotonicity_metrics.json"
)


def load_ex1_report(path=EX1_METRICS_PATH):
    """Load the parameters selected in Exercise 1."""
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def fit_learning_curve(train, test, selected_parameters):
    """Refit constrained LightGBM and record its learning curve."""

    X_train = train[FEATURES]
    X_test = test[FEATURES]

    y_train = train[TARGET].to_numpy(dtype=float)
    y_test = test[TARGET].to_numpy(dtype=float)

    w_train = train[WEIGHT].to_numpy(dtype=float)
    w_test = test[WEIGHT].to_numpy(dtype=float)

    # Fit preprocessing using the training sample only.
    preprocessor = _new_preprocessor()

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out().tolist()

    # Apply the increasing constraint to BonusMalus.
    matches = [
        index
        for index, name in enumerate(feature_names)
        if name.rsplit("__", 1)[-1] == "BonusMalus"
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one transformed BonusMalus feature"
        )

    constraints = [0] * len(feature_names)
    constraints[matches[0]] = BONUS_MALUS_CONSTRAINT

    # Reuse the parameters selected through Ex1 cross-validation.
    parameters = dict(DEFAULT_PARAMETERS)
    parameters.update(selected_parameters)

    parameters["monotone_constraints"] = constraints
    parameters["metric"] = "tweedie"

    constrained_lgbm = lgb.LGBMRegressor(**parameters)

    # The test sample is used for evaluation only.
    # No early stopping or further parameter selection.
    constrained_lgbm.fit(
        X_train_transformed,
        y_train,
        sample_weight=w_train,
        eval_set=[
            (X_train_transformed, y_train),
            (X_test_transformed, y_test),
        ],
        eval_names=["train", "test"],
        eval_sample_weight=[w_train, w_test],
        eval_metric="tweedie",
    )

    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", constrained_lgbm),
        ]
    )

    return pipeline, feature_names, constraints


def plot_learning_curve(model, output_path):
    """Plot the training and test metrics over boosting iterations."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ax = lgb.plot_metric(
        model,
        metric="tweedie",
        figsize=(9, 5.2),
        title="Constrained LightGBM Learning Curve",
        xlabel="Boosting Iteration",
        ylabel="Exposure-weighted Tweedie Metric",
    )

    ax.grid(alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=180)

    plt.close(ax.figure)

    return output_path


def run_learning_curve(
    frame,
    results_dir="results",
    ex1_metrics_path=EX1_METRICS_PATH,
):
    """Run Exercise 2 and save the learning curve and metrics."""

    report = load_ex1_report(ex1_metrics_path)

    selected_parameters = report["models"]["constrained"][
        "selected_parameters"
    ]

    # Use the same fixed split as PS3 and PS4 Ex1.
    data = prepare_model_data(frame)

    train = data.loc[data["sample"].eq("train")]
    test = data.loc[data["sample"].eq("test")]

    if train.empty or test.empty:
        raise ValueError("Both train and test samples are required")

    split_fingerprint = policy_split_fingerprint(data)

    if split_fingerprint != report["split_fingerprint"]:
        raise ValueError("The sample split differs from Exercise 1")

    pipeline, feature_names, constraints = fit_learning_curve(
        train,
        test,
        selected_parameters,
    )

    # Verify consistency with the Exercise 1 preprocessing.
    expected_features = report["preprocessing"][
        "transformed_feature_order"
    ]

    expected_constraints = report["preprocessing"][
        "monotone_constraints_in_feature_order"
    ]

    if feature_names != expected_features:
        raise ValueError("Feature order differs from Exercise 1")

    if constraints != expected_constraints:
        raise ValueError("Monotonicity constraints differ from Exercise 1")

    model = pipeline.named_steps["model"]

    history = model.evals_result_

    train_history = [
        float(value) for value in history["train"]["tweedie"]
    ]

    test_history = [
        float(value) for value in history["test"]["tweedie"]
    ]

    expected_iterations = int(
        model.get_params()["n_estimators"]
    )

    if (
        len(train_history) != expected_iterations
        or len(test_history) != expected_iterations
    ):
        raise RuntimeError("Unexpected evaluation history length")

    # Calculate the final deviance using the shared PS3 definition.
    train_predictions = pipeline.predict(train[FEATURES])
    test_predictions = pipeline.predict(test[FEATURES])

    train_deviance = float(
        mean_tweedie_deviance(
            train[TARGET],
            train_predictions,
            sample_weight=train[WEIGHT],
            power=TWEEDIE_POWER,
        )
    )

    test_deviance = float(
        mean_tweedie_deviance(
            test[TARGET],
            test_predictions,
            sample_weight=test[WEIGHT],
            power=TWEEDIE_POWER,
        )
    )

    # The minimum test metric is descriptive only.
    minimum_test_iteration = int(np.argmin(test_history)) + 1

    metrics = {
        "exercise": "PS4 Ex2 - Learning curve",
        "model": "Constrained LightGBM",
        "selected_parameters": selected_parameters,
        "parameters": model.get_params(),
        "parameter_source": "PS4 Ex1 training-only cross-validation",
        "split": report["split"],
        "split_fingerprint": split_fingerprint,
        "train_rows": len(train),
        "test_rows": len(test),
        "target": TARGET,
        "weights": WEIGHT,
        "tweedie_power": TWEEDIE_POWER,
        "transformed_feature_names": feature_names,
        "monotone_constraints": constraints,
        "evaluation_metric": "tweedie",
        "n_iterations": len(train_history),
        "train_history": train_history,
        "test_history": test_history,
        "initial_train_metric": train_history[0],
        "initial_test_metric": test_history[0],
        "final_train_metric": train_history[-1],
        "final_test_metric": test_history[-1],
        "minimum_test_metric": min(test_history),
        "minimum_test_iteration": minimum_test_iteration,
        "train_weighted_tweedie_deviance": train_deviance,
        "test_weighted_tweedie_deviance": test_deviance,
        "train_test_deviance_gap": test_deviance - train_deviance,
        "ex1_train_deviance": report["models"]["constrained"][
            "train_weighted_tweedie_deviance"
        ],
        "ex1_test_deviance": report["models"]["constrained"][
            "test_weighted_tweedie_deviance"
        ],
        "selection_method": (
            "Parameters fixed before observing test curves; "
            "no early stopping or test-based model selection"
        ),
    }

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    plot_learning_curve(
        model,
        results_dir / "learning_curve.png",
    )

    metrics_path = results_dir / "learning_curve_metrics.json"

    metrics_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("PS4 Exercise 2 completed.")
    print("Training rows:", len(train))
    print("Test rows:", len(test))
    print("Selected parameters:", selected_parameters)
    print("Boosting iterations:", len(train_history))
    print("Final train deviance:", train_deviance)
    print("Final test deviance:", test_deviance)
    print("Minimum test metric iteration:", minimum_test_iteration)
    print("Saved:", results_dir / "learning_curve.png")
    print("Saved:", metrics_path)

    return pipeline, metrics
