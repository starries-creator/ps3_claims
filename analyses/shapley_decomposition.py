"""PS4 Ex5: compare local Shapley explanations on a held-out policy."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analyses.claims_evaluation import (
    CATEGORICALS,
    PREDICTORS,
    TARGET,
    WEIGHT,
    fit_benchmark,
    sample_frames,
)
from analyses.lgbm_baseline import policy_split_fingerprint
from analyses.lgbm_monotonicity import FEATURES, _fit_pipeline

BACKGROUND_SIZE = 1_000
RANDOM_STATE = 42
SHAP_PATHS = 25
EX1_REPORT_NAME = "ps4_ex1_monotonicity_metrics.json"


def _fit_initial_glm(train):
    from dask_ml.preprocessing import Categorizer

    categorizer = Categorizer(columns=CATEGORICALS)
    X_train = categorizer.fit_transform(train[PREDICTORS])
    glm = fit_benchmark(X_train, train[TARGET], train[WEIGHT].to_numpy())
    return glm, categorizer


def _summarize(explanation):
    # In DALEX, B=0 is the mean over paths; B>0 contains individual paths.
    averaged = explanation.result.loc[explanation.result["B"].eq(0)]
    return {
        "baseline": float(explanation.intercept),
        "prediction": float(explanation.prediction),
        "contributions": averaged.set_index("variable_name")["contribution"]
        .reindex(PREDICTORS)
        .to_dict(),
    }


def run_shapley_decomposition(frame, results_dir="results", ex1_metrics_path=None):
    """Refit the two models from their original settings and explain one policy."""
    import dalex

    results_dir = Path(results_dir)
    report_path = (
        Path(ex1_metrics_path)
        if ex1_metrics_path is not None
        else results_dir / EX1_REPORT_NAME
    )
    with report_path.open(encoding="utf-8") as file:
        report = json.load(file)

    fingerprint = policy_split_fingerprint(frame)
    if fingerprint is None or fingerprint != report["split_fingerprint"]:
        raise ValueError("The fixed policy split does not match the Exercise 1 report")

    train, test = sample_frames(frame)
    background = train[PREDICTORS].sample(
        n=min(BACKGROUND_SIZE, len(train)), random_state=RANDOM_STATE
    )
    observation = test[PREDICTORS].iloc[[0]]
    policy_id = int(observation.index[0])

    glm, categorizer = _fit_initial_glm(train)
    constrained_parameters = report["models"]["constrained"]["selected_parameters"]
    constrained_lgbm, _, _ = _fit_pipeline(
        train[FEATURES].reset_index(drop=True),
        train[TARGET].to_numpy(dtype=float),
        train[WEIGHT].to_numpy(dtype=float),
        constrained_parameters,
        monotone_bonusmalus=True,
    )

    glm_explainer = dalex.Explainer(
        glm, data=categorizer.transform(background), label="Initial GLM", verbose=False
    )
    lgbm_explainer = dalex.Explainer(
        constrained_lgbm, data=background, label="Constrained LightGBM", verbose=False
    )
    glm_shap = glm_explainer.predict_parts(
        categorizer.transform(observation),
        type="shap",
        B=SHAP_PATHS,
        random_state=RANDOM_STATE,
    )
    lgbm_shap = lgbm_explainer.predict_parts(
        observation, type="shap", B=SHAP_PATHS, random_state=RANDOM_STATE
    )

    # Plot both decompositions on a shared axis without overriding their baselines.
    figure = glm_shap.plot(
        lgbm_shap, show=False, title=f"Shapley decompositions for policy {policy_id}"
    )
    figure.update_layout(
        xaxis2_title="Predicted pure premium (contributions from each baseline)"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        results_dir / "ps4_ex5_shapley.html",
        include_plotlyjs="cdn",
        div_id="ps4-ex5-shapley",
    )

    summary = {
        "policy_id": policy_id,
        "split_fingerprint": fingerprint,
        "background_rows": len(background),
        "background_baseline": "Unweighted mean prediction on the shared training background",
        "random_state": RANDOM_STATE,
        "shap_paths": SHAP_PATHS,
        "target": TARGET,
        "fit_weight": WEIGHT,
        "constrained_parameters": constrained_parameters,
        "models": {
            "initial_glm": _summarize(glm_shap),
            "constrained_lgbm": _summarize(lgbm_shap),
        },
    }
    (results_dir / "ps4_ex5_shapley.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    import sys

    from analyses.ps3_script import main

    main(["ps4-ex5", *sys.argv[1:]])
