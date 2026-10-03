from __future__ import annotations

from math import inf, isfinite
from typing import Union

import pandas as pd


def pinball_loss(
    returns: pd.Series[float],
    bounds: pd.Series[float],
    alpha: float = 0.05,
) -> pd.Series[float]:
    if not 0 < alpha < 1:
        raise ValueError("The target error rate must be between zero and one.")
    if not returns.index.equals(bounds.index):
        raise ValueError("Returns and bounds must have the same index.")
    if not returns.map(isfinite).all():
        raise ValueError("Observed returns must be finite.")

    residuals = returns - bounds
    weights = alpha - (residuals < 0).astype(float)
    losses = residuals * weights

    # Every non-finite bound receives infinite loss, including missing bounds.
    losses = losses.mask(~bounds.map(isfinite), inf)
    return losses.rename("pinball_loss")


def local_coverage_error(
    breaches: pd.Series[bool],
    alpha: float = 0.05,
    window: int = 250,
) -> pd.Series[float]:
    if not 0 < alpha < 1 or window < 1:
        raise ValueError("Use an error rate between zero and one and a positive window.")

    # Full trailing windows only; shorter sequences have no local estimate.
    local_breach_rate = breaches.rolling(window=window).mean()
    errors = (local_breach_rate - alpha).abs()
    return errors.rename("local_coverage_error")


# Evaluate against the fixed target, rather than the recorded adaptive alpha.
def summarize_forecasts(
    forecasts: pd.DataFrame,
    alpha: float = 0.05,
    coverage_window: int = 250,
) -> dict[str, Union[float, int]]:
    losses = pinball_loss(forecasts["return"], forecasts["bound"], alpha=alpha)
    coverage_errors = local_coverage_error(
        forecasts["breach"], alpha=alpha, window=coverage_window,
    )
    non_finite_count = sum(not isfinite(bound) for bound in forecasts["bound"])

    return {
        "mean_pinball_loss": float(losses.mean()),
        "breach_rate": float(forecasts["breach"].mean()),
        "mean_local_coverage_error": float(coverage_errors.mean()),
        "non_finite_count": non_finite_count,
    }
