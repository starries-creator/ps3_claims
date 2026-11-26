
"""PS4 Ex4: Compare LightGBM partial dependence profiles."""

from __future__ import annotations

import json
from pathlib import Path

import dalex as dx
import numpy as np
import pandas as pd

from analyses.claims_evaluation import prepare_model_data
from analyses.lgbm_baseline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    WEIGHT,
    policy_split_fingerprint,
)
from analyses.lgbm_monotonicity import _fit_pipeline


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

DATA_PATH = Path("data/ps4_claims.parquet")
METRICS_PATH = Path(
    "results/ps4_ex1_monotonicity_metrics.json"
)
OUTPUT_DIR = Path("figures/ps4_ex4_pdp")

BACKGROUND_SIZE = 100
RANDOM_STATE = 42
NUMERIC_GRID_POINTS = 21


def run_pdp_comparison() -> None:
    """Compare constrained and unconstrained LightGBM PDPs."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 2. Load canonical data
    # --------------------------------------------------

    print("Loading canonical PS4 data...", flush=True)

    raw = pd.read_parquet(DATA_PATH)
    data = prepare_model_data(raw)

    train = data.loc[data["sample"] == "train"].copy()
    test = data.loc[data["sample"] == "test"].copy()

    print("Training rows:", len(train), flush=True)
    print("Test rows:", len(test), flush=True)

    # Check consistency with Ex1 handoff.
    fingerprint = policy_split_fingerprint(data)

    handoff = json.loads(
        METRICS_PATH.read_text(encoding="utf-8")
    )

    if fingerprint != handoff["split_fingerprint"]:
        raise ValueError(
            "Data split does not match the Ex1 handoff."
        )

    if len(train) != handoff["train_rows"]:
        raise ValueError("Unexpected training sample size.")

    if len(test) != handoff["test_rows"]:
        raise ValueError("Unexpected test sample size.")

    print("Canonical data validation passed.", flush=True)

    # --------------------------------------------------
    # 3. Load selected model parameters
    # --------------------------------------------------

    parameters_u = handoff["models"]["unconstrained"][
        "selected_parameters"
    ]

    parameters_c = handoff["models"]["constrained"][
        "selected_parameters"
    ]

    if parameters_u != parameters_c:
        raise ValueError(
            "Model specifications differ. A constraint-only "
            "comparison requires matching parameters."
        )

    X_train = train[FEATURES].copy()
    y_train = train[TARGET].to_numpy(dtype=float)
    w_train = train[WEIGHT].to_numpy(dtype=float)

    # --------------------------------------------------
    # 4. Fit both LightGBM models
    # --------------------------------------------------

    print(
        "Fitting unconstrained LightGBM...",
        flush=True,
    )

    model_u, names_u, _ = _fit_pipeline(
        X_train,
        y_train,
        w_train,
        parameters_u,
        False,
    )

    print(
        "Fitting constrained LightGBM...",
        flush=True,
    )

    model_c, names_c, constraints = _fit_pipeline(
        X_train,
        y_train,
        w_train,
        parameters_c,
        True,
    )

    if names_u != names_c:
        raise ValueError(
            "Transformed feature orders do not match."
        )

    constrained_features = [
        name
        for name, value in zip(names_c, constraints)
        if value == 1
    ]

    if constrained_features != ["numeric__BonusMalus"]:
        raise ValueError(
            "Unexpected monotonicity constraint position."
        )

    print("Both models fitted successfully.", flush=True)

    # --------------------------------------------------
    # 5. Prepare identical background samples
    # --------------------------------------------------

    background = X_train.sample(
        n=min(BACKGROUND_SIZE, len(X_train)),
        random_state=RANDOM_STATE,
    ).copy()

    # Important:
    # DALEX replaces column values when calculating PDPs.
    # Keep categorical columns as object, not pandas category.
    # This avoids incompatible categorical assignments.
    for col in CATEGORICAL_FEATURES:
        background[col] = background[col].astype("object")

    # Confirm that both models accept the same background.
    for model_name, model in [
        ("Unconstrained", model_u),
        ("Constrained", model_c),
    ]:
        predictions = np.asarray(
            model.predict(background),
            dtype=float,
        )

        if not np.isfinite(predictions).all():
            raise ValueError(
                f"{model_name} returned non-finite predictions."
            )

    print(
        "Shared background sample prepared:",
        len(background),
        flush=True,
    )

    # --------------------------------------------------
    # 6. Create identical feature grids
    # --------------------------------------------------

    grids: dict[str, list] = {}

    for col in NUMERIC_FEATURES:
        values = X_train[col].to_numpy(dtype=float)

        low, high = np.quantile(
            values,
            [0.01, 0.99],
        )

        grids[col] = np.unique(
            np.linspace(
                low,
                high,
                NUMERIC_GRID_POINTS,
            )
        ).tolist()

    for col in CATEGORICAL_FEATURES:
        # Keep observed category values in their original types.
        grids[col] = (
            X_train[col]
            .value_counts()
            .head(15)
            .index.tolist()
        )

    print("Shared feature grids prepared.", flush=True)

    # --------------------------------------------------
    # 7. Create DALEX explainers
    # --------------------------------------------------

    explainer_u = dx.Explainer(
        model_u,
        data=background.copy(),
        label="Unconstrained LightGBM",
        model_type="regression",
        verbose=False,
        precalculate=False,
    )

    explainer_c = dx.Explainer(
        model_c,
        data=background.copy(),
        label="Constrained LightGBM",
        model_type="regression",
        verbose=False,
        precalculate=False,
    )

    explainers = {
        "Unconstrained": explainer_u,
        "Constrained": explainer_c,
    }

    print("DALEX explainers created.", flush=True)

    # --------------------------------------------------
    # 8. Calculate and save PDPs
    # --------------------------------------------------

    profile_groups = [
        (
            "numeric",
            NUMERIC_FEATURES,
            "numerical",
        ),
        (
            "categorical",
            CATEGORICAL_FEATURES,
            "categorical",
        ),
    ]

    for group_name, variables, variable_type in profile_groups:

        shared_grids = {
            name: grids[name]
            for name in variables
        }

        profiles = {}

        for model_name, explainer in explainers.items():
            print(
                f"Computing {group_name} PDP: {model_name}",
                flush=True,
            )

            profile = explainer.model_profile(
                type="partial",
                variables=variables,
                variable_type=variable_type,
                N=None,
                variable_splits=shared_grids,
                center=False,
                verbose=False,
            )

            profiles[model_name] = profile

        # Combine results from both models.
        comparison = pd.concat(
            [
                profiles["Unconstrained"].result,
                profiles["Constrained"].result,
            ],
            ignore_index=True,
        )

        csv_path = (
            OUTPUT_DIR / f"{group_name}_pdp_values.csv"
        )

        comparison.to_csv(
            csv_path,
            index=False,
        )

        # Create a comparison plot using DALEX.
        figure = profiles["Unconstrained"].plot(
            objects=profiles["Constrained"],
            center=False,
            facet_ncol=3,
            title=f"LightGBM {group_name.title()} PDP Comparison",
            y_title="Predicted PurePremium",
            show=False,
        )

        html_path = (
            OUTPUT_DIR / f"{group_name}_pdp.html"
        )

        figure.write_html(
            str(html_path),
            include_plotlyjs=True,
        )

        print(
            f"Saved {group_name} PDP results.",
            flush=True,
        )

    # --------------------------------------------------
    # 9. Save settings for reproducibility
    # --------------------------------------------------

    metadata = {
        "exercise": "PS4 Ex4 - Partial Dependence",
        "split_fingerprint": fingerprint,
        "training_rows": len(train),
        "test_rows": len(test),
        "background_size": len(background),
        "background_seed": RANDOM_STATE,
        "raw_feature_order": FEATURES,
        "model_parameters": parameters_u,
        "monotone_feature": "BonusMalus",
        "monotone_constraint": 1,
        "tweedie_power": 1.5,
        "target": "PurePremium = ClaimAmountCut / Exposure",
        "training_weight": "Exposure",
        "pdp_background": "Shared unweighted training sample",
        "numeric_grid": "Training-set 1st to 99th percentiles",
        "numeric_grid_points": NUMERIC_GRID_POINTS,
        "categorical_grid": "Up to 15 most common training levels",
        "feature_grids": grids,
    }

    settings_path = OUTPUT_DIR / "pdp_settings.json"

    settings_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("PDP analysis complete.", flush=True)
    print("Results saved in:", OUTPUT_DIR, flush=True)


if __name__ == "__main__":
    run_pdp_comparison()