"""Exposure-weighted metrics for nonnegative Tweedie outcomes."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_tweedie_deviance


def evaluate_predictions(actual, predicted, sample_weight):
    """Return weighted metrics for Tweedie power 1.5, indexed by metric name.

    Bias is relative to weighted actual outcomes; positive means overprediction.
    Indexed pandas inputs must have identical indexes in identical order.
    """
    indexed = [
        x.index for x in (actual, predicted, sample_weight) if isinstance(x, pd.Series)
    ]
    if indexed and any(not indexed[0].equals(index) for index in indexed[1:]):
        raise ValueError("Pandas index values and order must match")
    try:
        y, pred, weight = [
            np.asarray(x, dtype=float) for x in (actual, predicted, sample_weight)
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError("Inputs must be numeric one-dimensional arrays") from exc
    if any(x.ndim != 1 for x in (y, pred, weight)):
        raise ValueError("Inputs must be one-dimensional")
    if not len(y) or len(y) != len(pred) or len(y) != len(weight):
        raise ValueError("Inputs must have equal nonzero lengths")
    if any(not np.isfinite(x).all() for x in (y, pred, weight)):
        raise ValueError("Inputs must be finite")
    if (weight < 0).any() or weight.sum() <= 0 or not np.isfinite(weight.sum()):
        raise ValueError("Weights must be nonnegative with finite positive total")
    if (y < 0).any() or (pred <= 0).any():
        raise ValueError(
            "Tweedie power 1.5 requires nonnegative actuals and positive predictions"
        )
    actual_total = np.dot(weight, y)
    if not np.isfinite(actual_total) or actual_total <= 0:
        raise ValueError(
            "Relative bias requires a finite positive weighted actual total"
        )
    values = [
        np.dot(weight, pred - y) / actual_total,
        mean_tweedie_deviance(y, pred, sample_weight=weight, power=1.5),
        mean_absolute_error(y, pred, sample_weight=weight),
        np.sqrt(np.average((y - pred) ** 2, weights=weight)),
    ]
    if not np.isfinite(values).all():
        raise ValueError("Metrics must be finite")
    return pd.DataFrame(
        {"value": values},
        index=pd.Index(["Bias", "Tweedie deviance", "MAE", "RMSE"], name="metric"),
    )
