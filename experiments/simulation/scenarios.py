from __future__ import annotations

from math import isfinite, sqrt
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


DEGREES_OF_FREEDOM = 8
# A Student-t with eight degrees of freedom has variance 8/6.
UNIT_VARIANCE_SCALE = sqrt(6 / 8)


# Empty change_points gives the stationary control with volatility one.
def generate_scale_scenario(
    length: int,
    *,
    seed: int,
    change_points: Sequence[int] = (),
    high_scale: float = 1.0,
) -> pd.DataFrame:
    if length < 1 or not isfinite(high_scale) or high_scale <= 0:
        raise ValueError("Length and scale must be positive, with a finite scale.")

    volatility = np.ones(length)
    previous_change = 0
    for regime, change_point in enumerate(change_points):
        if not previous_change < change_point < length:
            raise ValueError("Change points must be increasing and inside the series.")
        # A change point is the first observation of the new regime.
        scale = high_scale if regime % 2 == 0 else 1.0
        volatility[change_point:] = scale
        previous_change = change_point

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
