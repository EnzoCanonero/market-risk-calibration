from collections import deque
from math import ceil, inf, isfinite
from typing import Sequence


# Initial scores are oldest first and set the window; gamma=0 gives rolling.
class ScoreCalibrator:
    def __init__(
        self,
        initial_scores: Sequence[float],
        alpha: float = 0.05,
        gamma: float = 0.0,
    ) -> None:
        if len(initial_scores) == 0 or not all(isfinite(s) for s in initial_scores):
            raise ValueError("Initial scores must be nonempty and finite.")
        if not 0 < alpha < 1:
            raise ValueError("The target error rate must be between zero and one.")
        if not isfinite(gamma) or gamma < 0:
            raise ValueError("The adaptation step must be finite and nonnegative.")

        self._scores: deque[float] = deque(initial_scores, maxlen=len(initial_scores))
        self.target_alpha = alpha
        self.alpha = alpha
        self.gamma = gamma

    # Use an order statistic, without interpolation, including infinite endpoints.
    def quantile(self) -> float:
        count = len(self._scores)
        rank = ceil((count + 1) * (1 - self.alpha))
        if rank > count:
            return inf
        if rank < 1:
            return -inf

        ordered_scores = sorted(self._scores)
        return ordered_scores[rank - 1]

    def update(self, score: float) -> bool:
        if not isfinite(score):
            raise ValueError("The observed score must be finite.")

        # Determine the breach before adding the observed score to the window.
        threshold = self.quantile()
        breach = score > threshold
        error = 1.0 if breach else 0.0

        self._scores.append(score)
        self.alpha += self.gamma * (self.target_alpha - error)
        return breach
