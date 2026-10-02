from __future__ import annotations

from math import inf

import pandas as pd
import pytest

from market_risk_calibration.sequential import sequential_forecasts
from market_risk_calibration.volatility import ewma_volatility


@pytest.mark.parametrize(
    "gamma,quantiles,bounds,alphas,finite",
    [
        (0.0, [1.0, 2.0, 2.0], [-0.02, -0.02, -0.04], [0.5, 0.5, 0.5], [True, True, True]),
        (0.25, [1.0, 3.0, 2.0], [-0.02, -0.03, -0.04], [0.5, 0.375, 0.5], [True, True, True]),
        (2.0, [1.0, inf, 2.0], [-0.02, -inf, -0.04], [0.5, -0.5, 0.5], [True, False, True]),
    ],
)
def test_forecasts_match_manual_sequence(
    gamma: float,
    quantiles: list[float],
    bounds: list[float],
    alphas: list[float],
    finite: list[bool],
) -> None:
    dates = pd.date_range("2020-01-01", periods=9, name="date")
    returns = pd.Series(
        [0.01, -0.01, -0.10, -0.02, -0.03, 0.01, -0.04, -0.02, 0.03],
        index=dates,
    )
    volatility = pd.Series(
        [float("nan"), float("nan"), 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.02],
        index=dates,
    )

    forecasts = sequential_forecasts(
        returns, volatility, warmup=2, calibration_size=4, window=3,
        alpha=0.5, gamma=gamma,
    )

    # Calibration scores are [10, 1, 3, -1]; only [1, 3, -1] initialize the window.
    expected = pd.DataFrame({
        "volatility": [0.02, 0.01, 0.02],
        "quantile": quantiles,
        "bound": bounds,
        "alpha": alphas,
        "return": [-0.04, -0.02, 0.03],
        "score": [2.0, 2.0, -1.5],
        "breach": [True, False, False],
        "finite": finite,
    }, index=dates[6:])
    pd.testing.assert_frame_equal(forecasts, expected)


def test_future_returns_do_not_change_issued_forecasts() -> None:
    returns = pd.Series(
        [-0.01, 0.0, 0.01, 0.02, -0.03, 0.04, -0.02, 0.01, -0.04, 0.02],
        index=pd.date_range("2020-01-01", periods=10),
    )
    changed_returns = returns.copy()
    changed_returns.iloc[7:] = 0.5

    volatility = ewma_volatility(returns, warmup=3)
    changed_volatility = ewma_volatility(changed_returns, warmup=3)
    original = sequential_forecasts(
        returns, volatility, warmup=3, calibration_size=3, window=2,
        alpha=0.5, gamma=0.25,
    )
    changed = sequential_forecasts(
        changed_returns, changed_volatility, warmup=3, calibration_size=3, window=2,
        alpha=0.5, gamma=0.25,
    )

    # Evaluation starts at position 6; include the forecast for changed position 7.
    forecast_columns = ["volatility", "quantile", "bound", "alpha"]
    pd.testing.assert_frame_equal(
        original[forecast_columns].iloc[:2], changed[forecast_columns].iloc[:2]
    )
