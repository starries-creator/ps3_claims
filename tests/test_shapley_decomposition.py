"""Regression checks for the PS4 Ex5 SHAP comparison."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer

from analyses.claims_evaluation import CATEGORICALS, TARGET, prepare_model_data, sample_frames
from analyses.lgbm_baseline import policy_split_fingerprint
from analyses import shapley_decomposition as ex5


@pytest.fixture
def prepared_data():
    ids = np.arange(1, 121)
    frame = pd.DataFrame({name: ids % 3 for name in CATEGORICALS})
    frame["IDpol"] = ids
    frame["BonusMalus"] = 50 + ids % 30
    frame["Density"] = 100 + 10 * ids
    frame["Exposure"] = 0.5
    frame["ClaimAmountCut"] = 1.5 * frame["BonusMalus"] + frame["Density"] / 20
    return prepare_model_data(frame)


def test_ex5_explains_one_test_policy_and_writes_both_models(prepared_data, tmp_path, monkeypatch):
    report = {
        "split_fingerprint": policy_split_fingerprint(prepared_data),
        "models": {"constrained": {"selected_parameters": {"n_estimators": 3, "n_jobs": 1}}},
    }
    report_path = tmp_path / "ex1.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    def fit_test_glm(train):
        # Use a deterministic stand-in; the full-data run validates the PS3 GLM.
        model = LinearRegression().fit(train[ex5.PREDICTORS], train[TARGET])
        return model, FunctionTransformer().fit(train[ex5.PREDICTORS])

    monkeypatch.setattr(ex5, "_fit_initial_glm", fit_test_glm)
    train, test = sample_frames(prepared_data)
    results_dir = tmp_path / "ex5"
    summary = ex5.run_shapley_decomposition(prepared_data, results_dir, report_path)

    assert summary["policy_id"] == test.index[0]
    assert summary["background_rows"] == len(train)
    assert set(summary["models"]) == {"initial_glm", "constrained_lgbm"}
    for model in summary["models"].values():
        assert set(model["contributions"]) == set(ex5.PREDICTORS)
        assert model["baseline"] + sum(model["contributions"].values()) == pytest.approx(
            model["prediction"]
        )
    plot = (results_dir / "ps4_ex5_shapley.html").read_text(encoding="utf-8")
    assert "Initial GLM" in plot and "Constrained LightGBM" in plot
    assert json.loads((results_dir / "ps4_ex5_shapley.json").read_text()) == summary


def test_ex5_rejects_a_report_for_a_different_split(prepared_data, tmp_path):
    report_path = tmp_path / ex5.EX1_REPORT_NAME
    report_path.write_text(json.dumps({"split_fingerprint": "not-the-ex1-split"}))

    with pytest.raises(ValueError, match="split does not match"):
        ex5.run_shapley_decomposition(prepared_data, tmp_path)
    assert not (tmp_path / "ps4_ex5_shapley.json").exists()
