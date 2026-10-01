from math import inf

from market_risk_calibration.calibration import ScoreCalibrator


def test_rolling_uses_conformal_rank_and_recent_scores() -> None:
    calibrator = ScoreCalibrator([-1.0, -4.0, -3.0, -2.0], alpha=0.5)

    # ceil((4 + 1) * 0.5) = 3: select the third score, not an interpolated median.
    assert calibrator.quantile() == -2.0

    assert calibrator.update(-3.0) is False
    assert calibrator.quantile() == -3.0

    # This breaches the old threshold, although it equals the next threshold.
    assert calibrator.update(-2.5) is True
    assert calibrator.quantile() == -2.5
    assert calibrator.alpha == 0.5


def test_aci_adjusts_after_breaches_and_ties() -> None:
    calibrator = ScoreCalibrator([-1.0, 0.0, 1.0], alpha=0.5, gamma=0.25)

    assert calibrator.quantile() == 0.0
    assert calibrator.update(0.0) is False
    assert calibrator.alpha == 0.625

    assert calibrator.update(2.0) is True
    assert calibrator.alpha == 0.5
    assert calibrator.quantile() == 1.0


def test_aci_recovers_from_infinite_thresholds_without_clipping() -> None:
    # A deliberately large step makes both endpoint transitions easy to check.
    calibrator = ScoreCalibrator([-1.0, 0.0, 1.0], alpha=0.5, gamma=2.0)

    assert calibrator.update(2.0) is True
    assert calibrator.alpha == -0.5
    assert calibrator.quantile() == inf

    assert calibrator.update(-2.0) is False
    assert calibrator.alpha == 0.5
    assert calibrator.quantile() == 1.0

    assert calibrator.update(1.0) is False
    assert calibrator.alpha == 1.5
    assert calibrator.quantile() == -inf

    assert calibrator.update(0.0) is True
    assert calibrator.alpha == 0.5
    assert calibrator.quantile() == 0.0
