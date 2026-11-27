import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_tweedie_deviance

from ps3.evaluation import evaluate_predictions


def test_weighted_metrics_and_structure():
    y = np.array([1.0, 4.0, 2.0])
    pred = np.array([2.0, 2.0, 3.0])
    w = np.array([1.0, 2.0, 3.0])
    result = evaluate_predictions(y, pred, w)
    assert list(result.index) == ["Bias", "Tweedie deviance", "MAE", "RMSE"]
    assert list(result.columns) == ["value"]
    assert result.loc["Bias", "value"] == pytest.approx(0)
    assert result.loc["MAE", "value"] == pytest.approx(8 / 6)
    assert result.loc["RMSE", "value"] == pytest.approx(np.sqrt(12 / 6))
    assert result.loc["Tweedie deviance", "value"] == pytest.approx(
        mean_tweedie_deviance(y, pred, sample_weight=w, power=1.5)
    )


def test_perfect_predictions():
    result = evaluate_predictions([1, 2], [1, 2], [1, 3])
    assert np.allclose(result["value"], 0)


@pytest.mark.parametrize(
    "y,p,w",
    [
        ([1, 2], [1], [1, 1]),
        ([[1], [2]], [1, 2], [1, 1]),
        ([1, np.nan], [1, 2], [1, 1]),
        ([1, 2], [1, np.inf], [1, 1]),
        ([1, 2], [1, 0], [1, 1]),
        ([-1, 2], [1, 2], [1, 1]),
        ([1, 2], [1, 2], [-1, 2]),
        ([1, 2], [1, 2], [0, 0]),
        ([0, 0], [1, 2], [1, 1]),
    ],
)
def test_invalid_inputs(y, p, w):
    with pytest.raises(ValueError):
        evaluate_predictions(y, p, w)


def test_pandas_index_must_match():
    with pytest.raises(ValueError, match="index"):
        evaluate_predictions(
            pd.Series([1, 2], index=[1, 2]),
            pd.Series([1, 2], index=[2, 1]),
            pd.Series([1, 1], index=[1, 2]),
        )


def test_bias_is_signed_and_relative_to_weighted_actual():
    result = evaluate_predictions([1, 3], [2, 4], [1, 3])
    assert result.loc["Bias", "value"] == pytest.approx(4 / 10)
