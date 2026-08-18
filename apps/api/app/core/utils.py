from datetime import (
    datetime,
    timezone,
)
from typing import Any

import pandas as pd


def utc_now_iso() -> str:
    """
    Return the current UTC datetime
    in ISO 8601 format.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


def normalize_column_name(
    column_name: str,
) -> str:
    """
    Normalize a column name so DataLens
    can compare columns between datasets.
    """

    return (
        str(
            column_name
        )
        .strip()
        .lower()
        .replace(
            " ",
            "_",
        )
        .replace(
            "-",
            "_",
        )
    )


def to_python_number(
    value: Any,
) -> Any:
    """
    Convert pandas / NumPy scalar values
    to JSON-friendly Python values.
    """

    if pd.isna(
        value
    ):
        return None

    if isinstance(
        value,
        bool,
    ):
        return bool(
            value
        )

    try:
        numeric_value = float(
            value
        )

        if numeric_value.is_integer():
            return int(
                numeric_value
            )

        return round(
            numeric_value,
            6,
        )

    except (
        TypeError,
        ValueError,
    ):
        return value