from typing import Any

import pandas as pd

from app.core.utils import (
    to_python_number,
)
from app.profiling.types import (
    infer_analytical_type,
)


# ============================================================
# QUANTITATIVE STATISTICS
# ============================================================

def build_quantitative_statistics(
    series: pd.Series,
) -> dict[str, Any]:
    values = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .dropna()
    )

    if values.empty:
        return {}

    return {
        "count": int(
            values.count()
        ),

        "mean": to_python_number(
            values.mean()
        ),

        "median": to_python_number(
            values.median()
        ),

        "std": to_python_number(
            values.std()
        ),

        "min": to_python_number(
            values.min()
        ),

        "q1": to_python_number(
            values.quantile(
                0.25
            )
        ),

        "q3": to_python_number(
            values.quantile(
                0.75
            )
        ),

        "max": to_python_number(
            values.max()
        ),
    }


# ============================================================
# CATEGORICAL STATISTICS
# ============================================================

def build_categorical_statistics(
    series: pd.Series,
) -> dict[str, Any]:
    values = (
        series
        .dropna()
    )

    if values.empty:
        return {}

    value_counts = (
        values
        .value_counts()
    )

    total = int(
        value_counts.sum()
    )

    top_values = []

    for (
        value,
        count,
    ) in (
        value_counts
        .head(10)
        .items()
    ):
        top_values.append(
            {
                "value": str(
                    value
                ),

                "count": int(
                    count
                ),

                "percentage": round(
                    (
                        int(count)
                        / total
                        * 100
                    ),
                    2,
                ),
            }
        )

    mode = (
        value_counts.index[0]
        if not value_counts.empty
        else None
    )

    return {
        "count": int(
            values.count()
        ),

        "unique": int(
            values.nunique()
        ),

        "mode": (
            str(
                mode
            )
            if mode is not None
            else None
        ),

        "mode_count": (
            int(
                value_counts.iloc[0]
            )
            if not value_counts.empty
            else 0
        ),

        "top_values":
            top_values,
    }


# ============================================================
# TEMPORAL STATISTICS
# ============================================================

def build_temporal_statistics(
    series: pd.Series,
    subtype: str | None,
) -> dict[str, Any]:
    values = (
        series
        .dropna()
    )

    if values.empty:
        return {}

    # --------------------------------------------------------
    # YEAR
    # --------------------------------------------------------

    if subtype == "year":
        numeric_values = (
            pd.to_numeric(
                values,
                errors="coerce",
            )
            .dropna()
        )

        if numeric_values.empty:
            return {}

        start = int(
            numeric_values.min()
        )

        end = int(
            numeric_values.max()
        )

        return {
            "count": int(
                numeric_values.count()
            ),

            "start":
                start,

            "end":
                end,

            "unique_periods": int(
                numeric_values.nunique()
            ),

            "range_years":
                end - start,
        }

    # --------------------------------------------------------
    # DATETIME
    # --------------------------------------------------------

    if subtype == "datetime":
        datetime_values = (
            pd.to_datetime(
                values,
                errors="coerce",
            )
            .dropna()
        )

        if datetime_values.empty:
            return {}

        return {
            "count": int(
                datetime_values.count()
            ),

            "start": (
                datetime_values
                .min()
                .isoformat()
            ),

            "end": (
                datetime_values
                .max()
                .isoformat()
            ),

            "unique_periods": int(
                datetime_values.nunique()
            ),
        }

    return {}


# ============================================================
# TEXT STATISTICS
# ============================================================

def build_text_statistics(
    series: pd.Series,
) -> dict[str, Any]:
    values = (
        series
        .dropna()
        .astype(str)
    )

    if values.empty:
        return {}

    lengths = (
        values
        .str
        .len()
    )

    return {
        "count": int(
            values.count()
        ),

        "unique": int(
            values.nunique()
        ),

        "average_length": (
            to_python_number(
                lengths.mean()
            )
        ),

        "min_length": int(
            lengths.min()
        ),

        "max_length": int(
            lengths.max()
        ),
    }


# ============================================================
# IDENTIFIER STATISTICS
# ============================================================

def build_identifier_statistics(
    series: pd.Series,
) -> dict[str, Any]:
    values = (
        series
        .dropna()
    )

    if values.empty:
        return {}

    count = int(
        values.count()
    )

    unique = int(
        values.nunique()
    )

    uniqueness_rate = (
        unique / count
        if count > 0
        else 0
    )

    return {
        "count":
            count,

        "unique":
            unique,

        "uniqueness_rate": round(
            uniqueness_rate,
            4,
        ),
    }


# ============================================================
# STATISTICS DISPATCHER
# ============================================================

def build_column_statistics(
    series: pd.Series,
    analytical_type: dict[str, Any],
) -> dict[str, Any]:
    column_type = (
        analytical_type[
            "type"
        ]
    )

    subtype = (
        analytical_type[
            "subtype"
        ]
    )

    if column_type == "quantitative":
        return (
            build_quantitative_statistics(
                series
            )
        )

    if column_type == "categorical":
        return (
            build_categorical_statistics(
                series
            )
        )

    if column_type == "temporal":
        return (
            build_temporal_statistics(
                series,
                subtype,
            )
        )

    if column_type == "text":
        return (
            build_text_statistics(
                series
            )
        )

    if column_type == "identifier":
        return (
            build_identifier_statistics(
                series
            )
        )

    return {}


# ============================================================
# DATASET PROFILE
# ============================================================

def build_dataset_profile(
    dataset_id: str,
    filename: str,
    content_type: str | None,
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    rows = int(
        dataframe.shape[0]
    )

    columns = int(
        dataframe.shape[1]
    )

    missing_cells = int(
        dataframe
        .isna()
        .sum()
        .sum()
    )

    total_cells = (
        rows
        * columns
    )

    missing_rate = (
        missing_cells
        / total_cells
        if total_cells > 0
        else 0
    )

    duplicate_rows = int(
        dataframe
        .duplicated()
        .sum()
    )

    duplicate_rate = (
        duplicate_rows
        / rows
        if rows > 0
        else 0
    )

    column_profiles = []

    candidate_keys = []

    # ========================================================
    # COLUMN PROFILES
    # ========================================================

    for column_name in dataframe.columns:
        series = (
            dataframe[
                column_name
            ]
        )

        non_null_count = int(
            series
            .notna()
            .sum()
        )

        column_missing = int(
            series
            .isna()
            .sum()
        )

        column_missing_rate = (
            column_missing
            / rows
            if rows > 0
            else 0
        )

        unique_count = int(
            series.nunique(
                dropna=True
            )
        )

        is_constant = (
            unique_count <= 1
        )

        # ----------------------------------------------------
        # SIMPLE CANDIDATE KEY
        # ----------------------------------------------------

        is_candidate_key = (
            rows > 0
            and column_missing == 0
            and unique_count == rows
        )

        if is_candidate_key:
            candidate_keys.append(
                str(
                    column_name
                )
            )

        analytical_type = (
            infer_analytical_type(
                str(
                    column_name
                ),
                series,
            )
        )

        statistics = (
            build_column_statistics(
                series,
                analytical_type,
            )
        )

        column_profiles.append(
            {
                "name": str(
                    column_name
                ),

                "pandas_dtype": str(
                    series.dtype
                ),

                "type": (
                    analytical_type[
                        "type"
                    ]
                ),

                "subtype": (
                    analytical_type[
                        "subtype"
                    ]
                ),

                "type_inference_reason": (
                    analytical_type[
                        "reason"
                    ]
                ),

                "non_null":
                    non_null_count,

                "missing":
                    column_missing,

                "missing_rate": round(
                    column_missing_rate,
                    4,
                ),

                "unique":
                    unique_count,

                "is_constant":
                    is_constant,

                "is_candidate_key":
                    is_candidate_key,

                "statistics":
                    statistics,
            }
        )

    return {
        "dataset_id":
            dataset_id,

        "filename":
            filename,

        "content_type":
            content_type,

        "rows":
            rows,

        "columns":
            columns,

        "column_names": [
            str(
                column
            )
            for column
            in dataframe.columns
        ],

        "missing_cells":
            missing_cells,

        "missing_rate": round(
            missing_rate,
            4,
        ),

        "duplicate_rows":
            duplicate_rows,

        "duplicate_rate": round(
            duplicate_rate,
            4,
        ),

        "candidate_keys":
            candidate_keys,

        "column_profiles":
            column_profiles,
    }