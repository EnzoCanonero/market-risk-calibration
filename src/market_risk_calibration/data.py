import csv
from io import StringIO
from math import isfinite
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import pandas as pd


# The official ZIP contains one CSV with a preamble, daily table and footer.
def _read_daily_factors(path: Union[str, Path]) -> pd.DataFrame:
    with ZipFile(path) as archive:
        csv_files = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_files) != 1:
            raise ValueError("Expected one CSV file in the French snapshot ZIP.")
        text = archive.read(csv_files[0]).decode("utf-8-sig")

    reader = csv.reader(StringIO(text))
    expected_header = ["", "Mkt-RF", "SMB", "HML", "RF"]
    for row in reader:
        fields = [field.strip() for field in row]
        if fields == expected_header:
            break
    else:
        raise ValueError("The French three-factor table header was not found.")

    daily_rows: list[list[str]] = []
    for row in reader:
        fields = [field.strip() for field in row]
        if not any(fields):
            continue
        if len(fields) == 1 and fields[0].startswith("Copyright"):
            break
        daily_rows.append(fields)

    if not daily_rows:
        raise ValueError("The snapshot contains no daily observations.")

    columns = ["date", "Mkt-RF", "SMB", "HML", "RF"]
    return pd.DataFrame(daily_rows, columns=columns)


# Return the official daily market series with decimal, not percent, returns.
def load_market_returns(path: Union[str, Path]) -> pd.DataFrame:
    factors = _read_daily_factors(path)

    # Reject malformed dates rather than silently skipping or reordering rows.
    if not factors["date"].str.fullmatch(r"[0-9]{8}").all():
        raise ValueError("Every observation must have an eight-digit YYYYMMDD date.")
    dates = pd.to_datetime(factors["date"], format="%Y%m%d", errors="raise")
    if dates.duplicated().any():
        raise ValueError("Observation dates must be unique.")
    if not dates.is_monotonic_increasing:
        raise ValueError("Observation dates must be in increasing order.")

    # Only Mkt-RF and RF enter the target; never fill missing returns with zero.
    for column in ["Mkt-RF", "RF"]:
        values = pd.to_numeric(factors[column], errors="raise")
        invalid = ~values.map(isfinite) | values.isin([-99.99, -999.0])
        if invalid.any():
            raise ValueError(f"{column} contains non-finite values or French missing codes.")
        factors[column] = values

    # Both source columns are percentages; add RF before converting once.
    market_returns = (factors["Mkt-RF"] + factors["RF"]) / 100.0
    return pd.DataFrame({"date": dates, "return": market_returns})
