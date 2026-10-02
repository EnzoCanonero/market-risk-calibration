from __future__ import annotations

from math import isfinite
from typing import Union

import pandas as pd

from market_risk_calibration.calibration import ScoreCalibrator


# Returns and past-only volatility forecasts share the same chronological index.
def sequential_forecasts(
    returns: pd.Series[float],
    volatility: pd.Series[float],
    *,
    warmup: int = 1000,
    calibration_size: int = 1000,
    window: int = 500,
    alpha: float = 0.05,
    gamma: float = 0.0,
) -> pd.DataFrame:
    if not returns.index.equals(volatility.index):
        raise ValueError("Returns and volatility must have the same index.")

    evaluation_start = warmup + calibration_size
    if warmup < 0 or not 1 <= window <= calibration_size or evaluation_start >= len(returns):
        raise ValueError("The window must fit calibration, with observations left to evaluate.")

    active_volatility = volatility.iloc[warmup:]
    if not active_volatility.map(isfinite).all() or (active_volatility <= 0).any():
        raise ValueError("Volatility after warm-up must be positive and finite.")

    # Initialize from genuine past forecasts, using only the most recent window.
    calibration_returns = returns.iloc[warmup:evaluation_start]
    calibration_volatility = volatility.iloc[warmup:evaluation_start]
    calibration_scores = -calibration_returns / calibration_volatility
    initial_scores = calibration_scores.iloc[-window:].tolist()
    calibrator = ScoreCalibrator(initial_scores, alpha=alpha, gamma=gamma)

    records: list[dict[str, Union[float, bool]]] = []
    for position in range(evaluation_start, len(returns)):
        scale = volatility.iloc[position]
        quantile = calibrator.quantile()
        bound = -scale * quantile
        adaptive_alpha = calibrator.alpha

        # Observe the return only after computing the forecast and its state.
        observed_return = returns.iloc[position]
        score = -observed_return / scale
        breach = calibrator.update(score)

        records.append({
            "volatility": scale,
            "quantile": quantile,
            "bound": bound,
            "alpha": adaptive_alpha,
            "return": observed_return,
            "score": score,
            "breach": breach,
            "finite": isfinite(bound),
        })

    return pd.DataFrame(records, index=returns.index[evaluation_start:])
