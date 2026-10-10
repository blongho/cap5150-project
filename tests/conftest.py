import pandas as pd
import pytest

from drift.data import EXPECTED_PERMISSIONS, FIRST_PERMISSION, LAST_PERMISSION

PERMS = [FIRST_PERMISSION] + [f"P{i}" for i in range(EXPECTED_PERMISSIONS - 2)] + [LAST_PERMISSION]


@pytest.fixture
def perms():
    return PERMS


@pytest.fixture
def raw_frame():
    """Small KronoDroid-shaped frame: hash, date, label, then 166 permission flags."""
    rows = [
        ("a", "1/5/2012", 1),
        ("b", "3/2/2014", 0),
        ("c", "7/9/2015", 1),
        ("c", "7/9/2015", 1),     # exact duplicate -> keep one
        ("d", "2/2/2016", 1),
        ("d", "2/2/2016", 0),     # same hash, both labels -> drop both
        ("e", "1/1/1980", 0),     # out-of-range year -> drop
        ("f", "not a date", 1),   # unparseable -> drop
    ]
    df = pd.DataFrame(rows, columns=["sha256", "HighestModDate", "Malware"])
    flags = pd.DataFrame(0, index=df.index, columns=PERMS)
    flags.iloc[:, 0] = 2  # a count, should become a binary flag
    return pd.concat([df[["sha256", "HighestModDate", "Malware"]], flags], axis=1)
