import numpy as np
import pandas as pd
import pytest
from analyses.evaluation_metrics import align_predictions, compare_predictions


def test_align_by_policy_id_and_compare():
    test = pd.DataFrame(
        {"PurePremium": [1.0, 3.0], "Exposure": [1.0, 2.0]},
        index=pd.Index([10, 20], name="IDpol"),
    )
    predictions = pd.DataFrame(
        {
            "IDpol": [20, 10],
            "pp_t_lgbm_unconstrained": [3.0, 1.0],
            "pp_t_lgbm_constrained": [2.0, 2.0],
        }
    )
    aligned = align_predictions(test, predictions)
    assert aligned["pp_t_lgbm_unconstrained"].tolist() == [1.0, 3.0]
    metrics = compare_predictions(test, aligned)
    assert metrics.loc["MAE", "unconstrained"] == 0
    assert metrics.loc["MAE", "constrained"] == pytest.approx(1)
    assert np.isfinite(metrics.to_numpy()).all()


@pytest.mark.parametrize("ids", [[10, 10], [10, 30], [10], [10, None]])
def test_reject_bad_policy_ids(ids):
    test = pd.DataFrame(index=pd.Index([10, 20], name="IDpol"))
    predictions = pd.DataFrame(
        {"IDpol": ids, "pp_t_lgbm_unconstrained": 1.0, "pp_t_lgbm_constrained": 1.0}
    )
    with pytest.raises(ValueError):
        align_predictions(test, predictions)


def test_gini_uses_policy_count_and_claim_amount():
    test = pd.DataFrame(
        {"PurePremium": [1.0, 3.0], "Exposure": [1.0, 2.0]},
        index=pd.Index([10, 20], name="IDpol"),
    )
    predictions = pd.DataFrame(
        {"pp_t_lgbm_unconstrained": [1.0, 3.0], "pp_t_lgbm_constrained": [3.0, 1.0]},
        index=test.index,
    )
    metrics = compare_predictions(test, predictions)
    assert metrics.loc["Gini", "unconstrained"] == pytest.approx(5 / 14)
    assert metrics.loc["Gini", "constrained"] == pytest.approx(-5 / 14)


def test_run_evaluation_with_matched_predictions(tmp_path):
    import json
    from analyses.claims_evaluation import (
        prepare_model_data,
        sample_frames,
        split_fingerprint,
    )
    from analyses.evaluation_metrics import run_evaluation_metrics
    from sklearn.metrics import mean_tweedie_deviance

    n = 40
    raw = pd.DataFrame(
        {
            "IDpol": np.arange(1, n + 1),
            "ClaimAmountCut": np.arange(1, n + 1, dtype=float),
            "Exposure": np.full(n, 0.5),
            "VehBrand": "B1",
            "VehGas": "Regular",
            "Region": "R11",
            "Area": "A",
            "DrivAge": 1,
            "VehAge": 1,
            "VehPower": 4,
            "BonusMalus": 50,
            "Density": 100,
        }
    )
    train, test = sample_frames(prepare_model_data(raw))
    predictions = pd.DataFrame(
        {
            "IDpol": test.index[::-1],
            "pp_t_lgbm_unconstrained": 10.0,
            "pp_t_lgbm_constrained": 20.0,
        }
    )
    predictions.to_csv(tmp_path / "pred.csv", index=False)
    models = {}
    for name, pred in [("unconstrained", 10.0), ("constrained", 20.0)]:
        score = mean_tweedie_deviance(
            test["PurePremium"],
            np.full(len(test), pred),
            sample_weight=test["Exposure"],
            power=1.5,
        )
        models[name] = {
            "selected_parameters": {},
            "test_weighted_tweedie_deviance": score,
        }
    handoff = {
        "split_fingerprint": split_fingerprint(train, test),
        "train_rows": len(train),
        "test_rows": len(test),
        "target_definition": "ClaimAmountCut / Exposure",
        "weights": "Exposure",
        "tweedie_power": 1.5,
        "models": models,
    }
    (tmp_path / "ex1.json").write_text(json.dumps(handoff))
    metrics, report = run_evaluation_metrics(
        raw, tmp_path, tmp_path / "ex1.json", tmp_path / "pred.csv"
    )
    assert metrics.loc["Tweedie deviance", "constrained"] == pytest.approx(
        models["constrained"]["test_weighted_tweedie_deviance"]
    )
    assert report["test_rows"] == len(test)
    assert (tmp_path / "ps4_ex3_metrics.csv").is_file()
    assert (tmp_path / "ps4_ex3_metrics.json").is_file()
