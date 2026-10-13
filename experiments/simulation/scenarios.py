from __future__ import annotations

from math import isfinite, sqrt
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.stats import t as student_t


DEGREES_OF_FREEDOM = 8
# A Student-t with eight degrees of freedom has variance 8/6.
UNIT_VARIANCE_SCALE = sqrt(6 / 8)


def _alternating_values(
    length: int, change_points: Sequence[int], first: float, second: float
) -> NDArray[np.float64]:
    if length < 1:
        raise ValueError("Length must be positive.")

    values = np.full(length, first, dtype=float)
    previous_change = 0
    for regime, change_point in enumerate(change_points):
        if not previous_change < change_point < length:
            raise ValueError("Change points must be increasing and inside the series.")
        # A change point is the first observation of the new regime.
        value = second if regime % 2 == 0 else first
        values[change_point:] = value
        previous_change = change_point
    return values


# Empty change_points gives the stationary control with volatility one.
def generate_scale_scenario(
    length: int,
    *,
    seed: int,
    change_points: Sequence[int] = (),
    high_scale: float = 1.0,
) -> pd.DataFrame:
    if not isfinite(high_scale) or high_scale <= 0:
        raise ValueError("Scale must be positive and finite.")

    volatility = _alternating_values(length, change_points, first=1.0, second=high_scale)

    # The same seed preserves innovations across scale magnitudes and schedules.
    rng = np.random.default_rng(seed)
    innovations = rng.standard_t(DEGREES_OF_FREEDOM, size=length)
    innovations = innovations * UNIT_VARIANCE_SCALE
    returns = volatility * innovations
    lower_quantile = float(student_t.ppf(0.05, df=DEGREES_OF_FREEDOM))
    lower_quantile = lower_quantile * UNIT_VARIANCE_SCALE

    return pd.DataFrame({
        "return": returns,
        "true_volatility": volatility,
        "oracle_quantile": volatility * lower_quantile,
    })


# Evaluate bounds against the known distribution, including infinite bounds.
def student_t_cdf(
    bounds: pd.Series[float], volatility: pd.Series[float]
) -> pd.Series[float]:
    if not bounds.index.equals(volatility.index):
        raise ValueError("Bounds and volatility must have the same index.")
    if not volatility.map(isfinite).all() or (volatility <= 0).any():
        raise ValueError("Volatility must be positive and finite.")

    distribution_scale = volatility * UNIT_VARIANCE_SCALE
    probabilities = student_t.cdf(
        bounds.to_numpy(), df=DEGREES_OF_FREEDOM, scale=distribution_scale.to_numpy()
    )
    return pd.Series(probabilities, index=bounds.index, name="breach_probability")


# One formula serves both vector evaluation and scalar quantile calculation.
def _mixture_cdf(bounds: ArrayLike, jump_probability: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(bounds, dtype=float)
    probability = np.asarray(jump_probability, dtype=float)
    normalization = np.sqrt(1 + 16 * probability * (1 - probability))
    unscaled_bounds = values * normalization
    no_jump_cdf = norm.cdf(unscaled_bounds - 4 * probability)
    jump_cdf = norm.cdf(unscaled_bounds + 4 * (1 - probability))
    probabilities = (1 - probability) * no_jump_cdf + probability * jump_cdf
    return np.asarray(probabilities, dtype=float)


def _mixture_quantile(jump_probability: float) -> float:
    def quantile_error(bound: float) -> float:
        probability = float(_mixture_cdf(bound, jump_probability))
        return probability - 0.05

    # Both fixed regimes have their 5% quantile inside this bracket.
    return float(brentq(quantile_error, -10.0, 10.0))


# Jump probabilities alternate between 3% and 8%; both regimes have variance one.
def generate_shape_scenario(
    length: int,
    *,
    seed: int,
    change_points: Sequence[int] = (),
) -> pd.DataFrame:
    jump_probability = _alternating_values(length, change_points, first=0.03, second=0.08)

    # The same seed preserves normal and uniform draws across schedules.
    rng = np.random.default_rng(seed)
    normal_draws = rng.standard_normal(length)
    uniform_draws = rng.random(length)
    jumps = uniform_draws < jump_probability
    centered_returns = normal_draws - 4 * (jumps - jump_probability)
    normalization = np.sqrt(1 + 16 * jump_probability * (1 - jump_probability))
    returns = centered_returns / normalization

    first_quantile = _mixture_quantile(0.03)
    second_quantile = _mixture_quantile(0.08)
    oracle_quantile = np.where(jump_probability == 0.03, first_quantile, second_quantile)

    return pd.DataFrame({
        "return": returns,
        "true_volatility": np.ones(length),
        "oracle_quantile": oracle_quantile,
        "jump_probability": jump_probability,
    })


def shape_cdf(
    bounds: pd.Series[float], jump_probability: pd.Series[float]
) -> pd.Series[float]:
    if not bounds.index.equals(jump_probability.index):
        raise ValueError("Bounds and jump probabilities must have the same index.")
    if not jump_probability.between(0, 1).all():
        raise ValueError("Jump probabilities must be finite and between zero and one.")

    probabilities = _mixture_cdf(bounds.to_numpy(), jump_probability.to_numpy())
    return pd.Series(probabilities, index=bounds.index, name="breach_probability")
