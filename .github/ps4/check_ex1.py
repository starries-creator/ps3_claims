"""Runtime acceptance checks for the PS4 Ex1 entry point on fixed synthetic policies."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_tweedie_deviance

from analyses import lgbm_monotonicity as ex1
from analyses.part_c import prepare_model_data


@pytest.fixture(scope="module")
def exercise(tmp_path_factory):
    rng = np.random.default_rng(42)
    n = 420
    premium = rng.uniform(1, 30, n)
    exposure = rng.uniform(.1, 1, n)
    features = {
        **{name: rng.choice(["a", "b", "c"], n) for name in ex1.CATEGORICAL_FEATURES},
        "IDpol": np.arange(1, n + 1),
        "BonusMalus": rng.integers(50, 131, n),
        "Density": rng.uniform(1, 250, n),
        "Exposure": exposure,
        "ClaimAmountCut": premium * exposure,
    }
    frame = pd.DataFrame(features)
    results = tmp_path_factory.mktemp("ps4-ex1")
    data = prepare_model_data(frame)
    train = data.loc[data["sample"].eq("train")]
    calls = []
    select = ex1._select_parameters

    def verify_training_only(X, y, weights, folds, constrained):
        pd.testing.assert_frame_equal(X, train[ex1.FEATURES].reset_index(drop=True))
        np.testing.assert_allclose(y, train["ClaimAmountCut"] / train["Exposure"])
        np.testing.assert_allclose(weights, train["Exposure"])
        for fit, validation in folds:
            assert not set(fit).intersection(validation)
            assert set(fit).union(validation) == set(range(len(train)))
        calls.append(constrained)
        return select(X, y, weights, folds, constrained)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ex1, "_select_parameters", verify_training_only)
        patch.setattr(ex1, "PARAMETER_GRID", [
            {"n_estimators": 12, "learning_rate": .05},
            {"n_estimators": 22, "learning_rate": .05},
        ])
        pipeline, cv, handoff, _ = ex1.run_monotonicity_analysis(frame, results)
    return data, pipeline, cv, handoff, results, calls


def test_exposure_weighted_bonusmalus_plot(exercise):
    data, _, _, _, results, _ = exercise
    train = data.loc[data["sample"].eq("train")]
    grouped = train.groupby(train["BonusMalus"].floordiv(10) * 10).sum(numeric_only=True)
    saved = pd.read_csv(results / "bonusmalus_weighted_claims.csv").set_index("BonusMalus_band_start")
    assert (results / "bonusmalus_weighted_claims.png").stat().st_size > 0
    np.testing.assert_allclose(saved["exposure_weighted_pure_premium"], grouped["ClaimAmountCut"] / grouped["Exposure"])


def test_monotonicity_after_preprocessing(exercise):
    data, pipeline, _, handoff, _, _ = exercise
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    constraints = pipeline.named_steps["model"].get_params()["monotone_constraints"]
    expected = [int(name.rsplit("__", 1)[-1] == "BonusMalus") for name in names]
    assert sum(expected) == 1
    assert list(constraints) == expected == handoff["preprocessing"]["monotone_constraints_in_feature_order"]
    for _, row in data[ex1.FEATURES].iloc[:6].iterrows():
        grid = pd.DataFrame([row.to_dict()] * 9)
        grid["BonusMalus"] = np.arange(50, 131, 10)
        assert (np.diff(pipeline.predict(grid[ex1.FEATURES])) >= -1e-10).all()


def test_training_only_selection_and_test_comparison(exercise):
    data, pipeline, cv, _, results, calls = exercise
    assert sorted(calls) == [False, True]
    test = data.loc[data["sample"].eq("test")].set_index("IDpol")
    saved = pd.read_csv(results / "ps4_ex1_test_predictions.csv").set_index("IDpol")
    pd.testing.assert_index_equal(saved.index, test.index)
    np.testing.assert_allclose(saved["pp_t_lgbm_constrained"], pipeline.predict(test[ex1.FEATURES]))
    for variant in ("constrained", "unconstrained"):
        model = cv["models"][variant]
        best = min(model["candidate_results"], key=lambda record: record["mean_weighted_tweedie_deviance"])
        assert model["selected_parameters"] == best["parameters"]
        score = mean_tweedie_deviance(
            test["ClaimAmountCut"] / test["Exposure"], saved[f"pp_t_lgbm_{variant}"],
            sample_weight=test["Exposure"], power=1.5,
        )
        assert score == pytest.approx(model["test_weighted_tweedie_deviance"])
