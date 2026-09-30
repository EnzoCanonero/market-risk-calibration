from __future__ import annotations

from math import isfinite, sqrt
from statistics import variance as sample_variance

import pandas as pd


def _validate_inputs(returns: pd.Series[float], warmup: int) -> None:
    if not 2 <= warmup < len(returns):
        raise ValueError("Warm-up must contain at least two returns and leave a forecast.")
    if not returns.map(isfinite).all():
        raise ValueError("Returns must be finite, with no missing values.")


# Input returns are chronological; warm-up positions have no forecast.
def expanding_volatility(
    returns: pd.Series[float], warmup: int = 1000
) -> pd.Series[float]:
    _validate_inputs(returns, warmup)

    observed_volatility = returns.expanding(min_periods=warmup).std(ddof=1)
    forecasts = observed_volatility.shift(1)
    return forecasts.rename("volatility")


# Seed with sample variance, then update assuming a zero forecast mean.
def ewma_volatility(
    returns: pd.Series[float], warmup: int = 1000
) -> pd.Series[float]:
    _validate_inputs(returns, warmup)

    variance = sample_variance(returns.iloc[:warmup])
    forecasts = pd.Series(float("nan"), index=returns.index, name="volatility")
    forecasts.iloc[warmup] = sqrt(variance)

    for position in range(warmup + 1, len(returns)):
        previous_return = returns.iloc[position - 1]
        variance = 0.94 * variance + 0.06 * previous_return**2
        forecasts.iloc[position] = sqrt(variance)

    return forecasts
