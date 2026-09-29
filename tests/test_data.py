from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from market_risk_calibration.data import load_market_returns


def _write_snapshot(directory: Path, rows: list[str]) -> Path:
    content = "French daily factors\n\n,Mkt-RF,SMB,HML,RF\n"
    content += "\n".join(rows)
    content += "\n\nCopyright Fama and French\n"
    path = directory / "french_daily.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("factors.csv", content)
    return path


def test_market_returns_include_rf_and_preserve_saturday(tmp_path: Path) -> None:
    path = _write_snapshot(tmp_path, [
        "19260709,0.99,0,0,0.01",
        "19260710,-2.01,0,0,0.01",
        "19260712,0,0,0,0.01",
    ])

    returns = load_market_returns(path)

    expected = pd.DataFrame({
        "date": pd.to_datetime(["1926-07-09", "1926-07-10", "1926-07-12"]),
        "return": [0.01, -0.02, 0.0001],
    })
    pd.testing.assert_frame_equal(returns, expected)


@pytest.mark.parametrize("second_date", ["20200102", "20200101"])
def test_duplicate_or_disordered_dates_are_rejected(
    tmp_path: Path, second_date: str
) -> None:
    path = _write_snapshot(tmp_path, [
        "20200102,1,0,0,0.01",
        f"{second_date},1,0,0,0.01",
    ])

    with pytest.raises(ValueError, match="dates must be"):
        load_market_returns(path)


@pytest.mark.parametrize(
    "market,rf",
    [("-99.99", "0.01"), ("1", "-999"), ("1", "")],
)
def test_missing_target_inputs_are_rejected(
    tmp_path: Path, market: str, rf: str
) -> None:
    path = _write_snapshot(tmp_path, [f"20200102,{market},0,0,{rf}"])

    with pytest.raises(ValueError):
        load_market_returns(path)
