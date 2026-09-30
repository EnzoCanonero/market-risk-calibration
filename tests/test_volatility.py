from __future__ import annotations

from math import sqrt
from typing import Callable

import pandas as pd
import pytest

from market_risk_calibration.volatility import expanding_volatility, ewma_volatility


def test_expanding_volatility_matches_sample_variance() -> None:
    returns = pd.Series(
        [-0.01, 0.0, 0.01, 0.02, -0.03, 0.04],
        index=pd.date_range("2020-01-01", periods=6),
    )

    forecasts = expanding_volatility(returns, warmup=3)

    expected = pd.Series(
        [float("nan")] * 3 + [sqrt(0.0001), sqrt(0.0005 / 3), sqrt(0.00037)],
        index=returns.index,
        name="volatility",
    )
    pd.testing.assert_series_equal(forecasts, expected)


def test_ewma_volatility_matches_variance_updates() -> None:
    returns = pd.Series(
        [-0.01, 0.0, 0.01, 0.02, -0.03, 0.04],
        index=pd.date_range("2020-01-01", periods=6),
    )

    forecasts = ewma_volatility(returns, warmup=3)

    expected = pd.Series(
        [float("nan")] * 3 + [sqrt(0.0001), sqrt(0.000118), sqrt(0.00016492)],
        index=returns.index,
        name="volatility",
    )
    pd.testing.assert_series_equal(forecasts, expected)


@pytest.mark.parametrize("forecast", [expanding_volatility, ewma_volatility])
def test_forecasts_use_only_past_returns(
    forecast: Callable[[pd.Series[float], int], pd.Series[float]],
) -> None:
    returns = pd.Series([-0.01, 0.0, 0.01, 0.02, -0.03, 0.04])
    changed_returns = returns.copy()
    changed_returns.iloc[4:] = 0.5

    original = forecast(returns, 3)
    changed = forecast(changed_returns, 3)

    # The forecast for position 4 must also precede its observed return.
    pd.testing.assert_series_equal(original.iloc[:5], changed.iloc[:5])
