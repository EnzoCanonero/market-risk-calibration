from math import inf, isnan

import pandas as pd
import pytest

from market_risk_calibration.metrics import (
    local_coverage_error,
    pinball_loss,
    summarize_forecasts,
)


def test_pinball_loss_matches_manual_calculation() -> None:
    dates = pd.date_range("2020-01-01", periods=3)
    returns = pd.Series([-0.03, 0.0, 0.01], index=dates)
    bounds = pd.Series([-0.01, -0.02, 0.01], index=dates)

    losses = pinball_loss(returns, bounds, alpha=0.25)

    expected = pd.Series([0.015, 0.005, 0.0], index=dates, name="pinball_loss")
    pd.testing.assert_series_equal(losses, expected)


def test_summary_uses_fixed_target_and_complete_coverage_windows() -> None:
    dates = pd.date_range("2020-01-01", periods=4)
    forecasts = pd.DataFrame({
        "return": [-0.01, -0.02, -0.03, 0.0],
        "bound": [-0.01, -0.01, -0.01, -0.01],
        "breach": [False, True, True, False],
        "alpha": [0.25, 0.20, 0.15, 0.20],
    }, index=dates)

    errors = local_coverage_error(forecasts["breach"], alpha=0.25, window=2)
    summary = summarize_forecasts(forecasts, alpha=0.25, coverage_window=2)

    # The three complete windows have breach rates 1/2, 1 and 1/2.
    expected = pd.Series(
        [float("nan"), 0.25, 0.75, 0.25], index=dates, name="local_coverage_error",
    )
    pd.testing.assert_series_equal(errors, expected)
    assert summary == pytest.approx({
        "mean_pinball_loss": 0.00625,
        "breach_rate": 0.5,
        "mean_local_coverage_error": 5 / 12,
        "non_finite_count": 0,
    })


def test_infinite_bounds_remain_in_loss_and_breach_summaries() -> None:
    forecasts = pd.DataFrame({
        "return": [0.0, 0.0, 0.0],
        "bound": [-inf, inf, -0.02],
        "breach": [False, True, False],
    })

    losses = pinball_loss(forecasts["return"], forecasts["bound"])
    summary = summarize_forecasts(forecasts)

    assert losses.tolist() == pytest.approx([inf, inf, 0.001])
    assert summary["mean_pinball_loss"] == inf
    assert summary["breach_rate"] == pytest.approx(1 / 3)
    assert summary["non_finite_count"] == 2
    assert isnan(summary["mean_local_coverage_error"])
