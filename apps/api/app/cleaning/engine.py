from collections import Counter
from typing import Any

import pandas as pd

from pandas.api.types import (
    is_object_dtype,
    is_string_dtype,
)

from app.core.utils import (
    normalize_column_name,
)


# ============================================================
# CLEANING POLICY
# ============================================================

# These values are considered sufficiently explicit
# to be normalized automatically as missing values.
SAFE_MISSING_MARKERS = {
    "",
    "null",
    "none",
    "nan",
    "<na>",
}


# These values can mean "missing", but can also
# have legitimate business meaning.
#
# DataLens detects them but does NOT change them
# automatically.
AMBIGUOUS_MISSING_MARKERS = {
    "na",
    "n/a",
    "-",
    "--",
    "missing",
    "not available",
}


YEAR_NAME_HINTS = {
    "year",
    "annee",
    "année",
}


DATE_NAME_HINTS = {
    "date",
    "datetime",
    "timestamp",
    "created_at",
    "updated_at",
}


# ============================================================
# QUALITY SNAPSHOT
# ============================================================

def build_quality_snapshot(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build a lightweight quality snapshot used
    to compare the dataset before and after
    cleaning.
    """

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

    duplicate_rows = int(
        dataframe
        .duplicated()
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

    duplicate_rate = (
        duplicate_rows
        / rows
        if rows > 0
        else 0
    )

    entirely_empty_columns = [
        str(
            column
        )
        for column
        in dataframe.columns
        if dataframe[
            column
        ]
        .isna()
        .all()
    ]

    return {
        "rows":
            rows,

        "columns":
            columns,

        "total_cells":
            total_cells,

        "missing_cells":
            missing_cells,

        "missing_rate":
            round(
                missing_rate,
                4,
            ),

        "duplicate_rows":
            duplicate_rows,

        "duplicate_rate":
            round(
                duplicate_rate,
                4,
            ),

        "entirely_empty_columns":
            entirely_empty_columns,

        "dtypes":
            {
                str(
                    column
                ):
                    str(
                        dtype
                    )
                for (
                    column,
                    dtype,
                ) in dataframe.dtypes.items()
            },
    }


# ============================================================
# HELPERS
# ============================================================

def is_string_like_series(
    series: pd.Series,
) -> bool:
    """
    Return True for pandas string/object columns.
    """

    return bool(
        is_object_dtype(
            series
        )
        or is_string_dtype(
            series
        )
    )


# ============================================================
# COLUMN NAME CLEANING
# ============================================================

def clean_column_names(
    dataframe: pd.DataFrame,
    operations: list[dict[str, Any]],
    review_items: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Safely remove leading/trailing whitespace
    from column names.

    If this would create duplicate column names,
    DataLens does not perform the change.
    """

    original_names = [
        str(
            column
        )
        for column
        in dataframe.columns
    ]

    cleaned_names = [
        name.strip()
        for name
        in original_names
    ]

    changes = [
        {
            "before":
                before,

            "after":
                after,
        }
        for (
            before,
            after,
        ) in zip(
            original_names,
            cleaned_names,
            strict=True,
        )
        if before != after
    ]

    if not changes:
        return dataframe

    # --------------------------------------------------------
    # Collision protection
    # --------------------------------------------------------

    if (
        len(
            cleaned_names
        )
        != len(
            set(
                cleaned_names
            )
        )
    ):
        review_items.append(
            {
                "type":
                    "column_name_collision",

                "status":
                    "review_required",

                "automatic_change":
                    False,

                "reason":
                    (
                        "Trimming column names would "
                        "create duplicate column names."
                    ),

                "proposed_changes":
                    changes,
            }
        )

        return dataframe

    result = (
        dataframe.copy()
    )

    result.columns = (
        cleaned_names
    )

    operations.append(
        {
            "operation":
                "trim_column_names",

            "automatic":
                True,

            "reversible":
                True,

            "affected_columns":
                len(
                    changes
                ),

            "reason":
                (
                    "Leading or trailing whitespace "
                    "was detected in column names."
                ),

            "changes":
                changes,
        }
    )

    return result


# ============================================================
# STRING WHITESPACE
# ============================================================

def trim_string_values(
    dataframe: pd.DataFrame,
    operations: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Remove leading and trailing whitespace
    from string values.
    """

    result = (
        dataframe.copy()
    )

    for column_name in result.columns:
        series = (
            result[
                column_name
            ]
        )

        if not is_string_like_series(
            series
        ):
            continue

        affected = 0

        examples = []

        def clean_value(
            value: Any,
        ) -> Any:
            nonlocal affected

            if not isinstance(
                value,
                str,
            ):
                return value

            cleaned = (
                value.strip()
            )

            if cleaned != value:
                affected += 1

                if len(
                    examples
                ) < 5:
                    examples.append(
                        {
                            "before":
                                value,

                            "after":
                                cleaned,
                        }
                    )

            return cleaned

        result[
            column_name
        ] = series.map(
            clean_value
        )

        if affected > 0:
            operations.append(
                {
                    "operation":
                        "trim_whitespace",

                    "column":
                        str(
                            column_name
                        ),

                    "automatic":
                        True,

                    "reversible":
                        True,

                    "affected_values":
                        affected,

                    "reason":
                        (
                            "Leading or trailing "
                            "whitespace was detected."
                        ),

                    "examples":
                        examples,
                }
            )

    return result


# ============================================================
# MISSING VALUE NORMALIZATION
# ============================================================

def normalize_missing_values(
    dataframe: pd.DataFrame,
    operations: list[dict[str, Any]],
    review_items: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Normalize safe missing-value markers.

    Ambiguous markers are reported but preserved.
    """

    result = (
        dataframe.copy()
    )

    for column_name in result.columns:
        series = (
            result[
                column_name
            ]
        )

        if not is_string_like_series(
            series
        ):
            continue

        safe_markers: Counter[
            str
        ] = Counter()

        ambiguous_markers: Counter[
            str
        ] = Counter()

        affected = 0

        def normalize_value(
            value: Any,
        ) -> Any:
            nonlocal affected

            if not isinstance(
                value,
                str,
            ):
                return value

            normalized = (
                value
                .strip()
                .casefold()
            )

            # ------------------------------------------------
            # Safe marker
            # ------------------------------------------------

            if (
                normalized
                in SAFE_MISSING_MARKERS
            ):
                safe_markers[
                    value
                ] += 1

                affected += 1

                return pd.NA

            # ------------------------------------------------
            # Ambiguous marker
            # ------------------------------------------------

            if (
                normalized
                in AMBIGUOUS_MISSING_MARKERS
            ):
                ambiguous_markers[
                    value
                ] += 1

            return value

        result[
            column_name
        ] = series.map(
            normalize_value
        )

        if affected > 0:
            operations.append(
                {
                    "operation":
                        "normalize_missing_values",

                    "column":
                        str(
                            column_name
                        ),

                    "automatic":
                        True,

                    "reversible":
                        True,

                    "affected_values":
                        affected,

                    "reason":
                        (
                            "Explicit missing-value "
                            "markers were detected."
                        ),

                    "markers":
                        dict(
                            safe_markers
                        ),
                }
            )

        if ambiguous_markers:
            review_items.append(
                {
                    "type":
                        "ambiguous_missing_markers",

                    "column":
                        str(
                            column_name
                        ),

                    "status":
                        "review_required",

                    "automatic_change":
                        False,

                    "reason":
                        (
                            "Some values resemble "
                            "missing-value markers, but "
                            "they could have legitimate "
                            "business meaning."
                        ),

                    "values":
                        dict(
                            ambiguous_markers
                        ),

                    "suggested_action":
                        (
                            "Review these values before "
                            "normalizing them."
                        ),
                }
            )

    return result


# ============================================================
# ENTIRELY EMPTY ROWS
# ============================================================

def remove_entirely_empty_rows(
    dataframe: pd.DataFrame,
    operations: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Remove rows for which every column is missing.

    This is a conservative automatic operation because
    a row containing no observed value carries no usable
    analytical information.

    RAW data remains preserved outside this function.
    """

    empty_row_mask = (
        dataframe
        .isna()
        .all(
            axis=1
        )
    )

    affected_rows = int(
        empty_row_mask.sum()
    )

    if affected_rows == 0:
        return dataframe

    result = (
        dataframe
        .loc[
            ~empty_row_mask
        ]
        .reset_index(
            drop=True
        )
    )

    operations.append(
        {
            "operation":
                "remove_entirely_empty_rows",

            "automatic":
                True,

            "reversible":
                True,

            "affected_rows":
                affected_rows,

            "reason":
                (
                    "Rows contained no observed "
                    "value in any column."
                ),
        }
    )

    return result


# ============================================================
# EXACT DUPLICATE REVIEW
# ============================================================

def detect_exact_duplicate_rows(
    dataframe: pd.DataFrame,
    review_items: list[dict[str, Any]],
) -> None:
    """
    Detect exact duplicate rows without removing them.

    Identical non-empty rows can represent legitimate
    repeated business events, especially in transactional
    datasets. DataLens therefore reports them for review
    instead of deleting them automatically.
    """

    duplicate_mask = (
        dataframe
        .duplicated(
            keep="first"
        )
    )

    duplicate_rows = int(
        duplicate_mask.sum()
    )

    if duplicate_rows == 0:
        return

    review_items.append(
        {
            "type":
                "exact_duplicate_rows",

            "status":
                "review_required",

            "automatic_change":
                False,

            "duplicate_rows":
                duplicate_rows,

            "reason":
                (
                    "Some non-empty rows are strictly "
                    "identical across every column. "
                    "They may represent legitimate "
                    "repeated business events."
                ),

            "suggested_action":
                (
                    "Review the dataset grain and "
                    "business meaning before deciding "
                    "whether any duplicate rows should "
                    "be removed."
                ),
        }
    )


# ============================================================
# BOOLEAN CONVERSION
# ============================================================

def try_boolean_conversion(
    series: pd.Series,
) -> pd.Series | None:
    """
    Convert a text column to Boolean only when
    every non-null value is exactly true/false.
    """

    non_null = (
        series
        .dropna()
    )

    if non_null.empty:
        return None

    normalized = (
        non_null
        .astype(str)
        .str
        .strip()
        .str
        .casefold()
    )

    distinct = set(
        normalized.tolist()
    )

    if not distinct:
        return None

    if not distinct.issubset(
        {
            "true",
            "false",
        }
    ):
        return None

    mapping = {
        "true":
            True,

        "false":
            False,
    }

    def convert(
        value: Any,
    ) -> Any:
        if pd.isna(
            value
        ):
            return pd.NA

        normalized_value = (
            str(
                value
            )
            .strip()
            .casefold()
        )

        return mapping[
            normalized_value
        ]

    return (
        series
        .map(
            convert
        )
        .astype(
            "boolean"
        )
    )


# ============================================================
# YEAR CONVERSION
# ============================================================

def try_year_conversion(
    column_name: str,
    series: pd.Series,
) -> pd.Series | None:
    """
    Convert a text column to integer year only
    when the column name and every value make
    the conversion unambiguous.
    """

    normalized_name = (
        normalize_column_name(
            column_name
        )
    )

    if (
        normalized_name
        not in YEAR_NAME_HINTS
    ):
        return None

    numeric = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
    )

    original_non_null = (
        series.notna()
    )

    if (
        numeric[
            original_non_null
        ]
        .isna()
        .any()
    ):
        return None

    non_null_numeric = (
        numeric
        .dropna()
    )

    if non_null_numeric.empty:
        return None

    valid_range = (
        (
            non_null_numeric
            >= 1800
        )
        &
        (
            non_null_numeric
            <= 2200
        )
    )

    if not valid_range.all():
        return None

    integer_values = (
        (
            non_null_numeric
            % 1
        )
        == 0
    )

    if not integer_values.all():
        return None

    return (
        numeric
        .astype(
            "Int64"
        )
    )


# ============================================================
# DATETIME DETECTION
# ============================================================

def looks_like_date_column(
    column_name: str,
) -> bool:
    normalized_name = (
        normalize_column_name(
            column_name
        )
    )

    if (
        normalized_name
        in DATE_NAME_HINTS
    ):
        return True

    return any(
        normalized_name.endswith(
            suffix
        )
        for suffix in (
            "_date",
            "_datetime",
            "_timestamp",
        )
    )


# ============================================================
# DATETIME CONVERSION
# ============================================================

def try_datetime_conversion(
    column_name: str,
    series: pd.Series,
) -> pd.Series | None:
    """
    Convert to datetime only when:
    - the column name strongly indicates a date;
    - all non-null values parse successfully.
    """

    if not looks_like_date_column(
        column_name
    ):
        return None

    non_null = (
        series
        .dropna()
    )

    if non_null.empty:
        return None

    parsed_non_null = (
        pd.to_datetime(
            non_null,
            errors="coerce",
            format="mixed",
        )
    )

    if (
        parsed_non_null
        .isna()
        .any()
    ):
        return None

    return (
        pd.to_datetime(
            series,
            errors="coerce",
            format="mixed",
        )
    )


# ============================================================
# NUMERIC TYPE CANDIDATE
# ============================================================

def detect_numeric_conversion_candidate(
    column_name: str,
    series: pd.Series,
    review_items: list[dict[str, Any]],
) -> None:
    """
    Detect numeric-looking text without
    automatically converting it.

    Example:
        "00123"

    could represent:
        - a quantity;
        - an identifier;
        - a postal code;
        - a product code.

    Therefore human review is required.
    """

    non_null = (
        series
        .dropna()
    )

    if non_null.empty:
        return

    strings = (
        non_null
        .astype(str)
        .str
        .strip()
    )

    parsed = (
        pd.to_numeric(
            strings,
            errors="coerce",
        )
    )

    if (
        parsed
        .isna()
        .any()
    ):
        return

    review_items.append(
        {
            "type":
                "numeric_type_candidate",

            "column":
                str(
                    column_name
                ),

            "status":
                "review_required",

            "automatic_change":
                False,

            "reason":
                (
                    "All non-null values can be "
                    "parsed as numbers, but the "
                    "semantic meaning of the "
                    "column is unknown."
                ),

            "suggested_action":
                (
                    "Review whether this column "
                    "should be converted to numeric."
                ),
        }
    )


# ============================================================
# SAFE TYPE CONVERSIONS
# ============================================================

def apply_safe_type_conversions(
    dataframe: pd.DataFrame,
    operations: list[dict[str, Any]],
    review_items: list[dict[str, Any]],
) -> pd.DataFrame:
    result = (
        dataframe.copy()
    )

    for column_name in result.columns:
        series = (
            result[
                column_name
            ]
        )

        if not is_string_like_series(
            series
        ):
            continue

        original_dtype = str(
            series.dtype
        )

        # ----------------------------------------------------
        # BOOLEAN
        # ----------------------------------------------------

        boolean_result = (
            try_boolean_conversion(
                series
            )
        )

        if boolean_result is not None:
            result[
                column_name
            ] = boolean_result

            operations.append(
                {
                    "operation":
                        "safe_type_conversion",

                    "column":
                        str(
                            column_name
                        ),

                    "automatic":
                        True,

                    "reversible":
                        True,

                    "from_dtype":
                        original_dtype,

                    "to_dtype":
                        str(
                            boolean_result.dtype
                        ),

                    "reason":
                        (
                            "Every non-null value was "
                            "an unambiguous true/false "
                            "value."
                        ),
                }
            )

            continue

        # ----------------------------------------------------
        # YEAR
        # ----------------------------------------------------

        year_result = (
            try_year_conversion(
                str(
                    column_name
                ),
                series,
            )
        )

        if year_result is not None:
            result[
                column_name
            ] = year_result

            operations.append(
                {
                    "operation":
                        "safe_type_conversion",

                    "column":
                        str(
                            column_name
                        ),

                    "automatic":
                        True,

                    "reversible":
                        True,

                    "from_dtype":
                        original_dtype,

                    "to_dtype":
                        str(
                            year_result.dtype
                        ),

                    "reason":
                        (
                            "The column represents "
                            "a year and every non-null "
                            "value is a plausible "
                            "integer year."
                        ),
                }
            )

            continue

        # ----------------------------------------------------
        # DATETIME
        # ----------------------------------------------------

        datetime_result = (
            try_datetime_conversion(
                str(
                    column_name
                ),
                series,
            )
        )

        if datetime_result is not None:
            result[
                column_name
            ] = datetime_result

            operations.append(
                {
                    "operation":
                        "safe_type_conversion",

                    "column":
                        str(
                            column_name
                        ),

                    "automatic":
                        True,

                    "reversible":
                        True,

                    "from_dtype":
                        original_dtype,

                    "to_dtype":
                        str(
                            datetime_result.dtype
                        ),

                    "reason":
                        (
                            "The column name indicates "
                            "a date and every non-null "
                            "value was parsed "
                            "successfully."
                        ),
                }
            )

            continue

        # ----------------------------------------------------
        # OTHER NUMERIC-LOOKING TEXT
        # ----------------------------------------------------

        detect_numeric_conversion_candidate(
            str(
                column_name
            ),
            series,
            review_items,
        )

    return result


# ============================================================
# ADDITIONAL QUALITY WARNINGS
# ============================================================

def detect_quality_review_items(
    dataframe: pd.DataFrame,
    review_items: list[dict[str, Any]],
) -> None:
    """
    Detect potentially important quality issues
    that should not be automatically modified.
    """

    rows = int(
        dataframe.shape[0]
    )

    if rows == 0:
        review_items.append(
            {
                "type":
                    "empty_dataset",

                "status":
                    "review_required",

                "automatic_change":
                    False,

                "reason":
                    (
                        "The dataset contains "
                        "no data rows."
                    ),
            }
        )

        return

    for column_name in dataframe.columns:
        series = (
            dataframe[
                column_name
            ]
        )

        missing_count = int(
            series
            .isna()
            .sum()
        )

        missing_rate = (
            missing_count
            / rows
        )

        # ----------------------------------------------------
        # Fully empty column
        # ----------------------------------------------------

        if missing_rate == 1:
            review_items.append(
                {
                    "type":
                        "entirely_empty_column",

                    "column":
                        str(
                            column_name
                        ),

                    "status":
                        "review_required",

                    "automatic_change":
                        False,

                    "missing_rate":
                        1.0,

                    "reason":
                        (
                            "The column contains "
                            "only missing values."
                        ),

                    "suggested_action":
                        (
                            "Review whether this "
                            "column should be removed."
                        ),
                }
            )

            continue

        # ----------------------------------------------------
        # Very sparse column
        # ----------------------------------------------------

        if missing_rate >= 0.80:
            review_items.append(
                {
                    "type":
                        "high_missing_rate",

                    "column":
                        str(
                            column_name
                        ),

                    "status":
                        "review_required",

                    "automatic_change":
                        False,

                    "missing_rate":
                        round(
                            missing_rate,
                            4,
                        ),

                    "reason":
                        (
                            "At least 80% of values "
                            "are missing."
                        ),
                }
            )


# ============================================================
# CLEANING PIPELINE
# ============================================================

def clean_dataset(
    raw_dataframe: pd.DataFrame,
    filename: str,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    list[dict[str, Any]],
]:
    """
    DataLens Cleaning Engine v0.2.

    Policy:
    - preserve RAW data;
    - apply only conservative automatic changes;
    - automatically remove entirely empty rows;
    - never automatically remove exact non-empty duplicates;
    - report ambiguous changes for review;
    - produce a complete before/after report.
    """

    before = (
        build_quality_snapshot(
            raw_dataframe
        )
    )

    operations: list[
        dict[str, Any]
    ] = []

    review_items: list[
        dict[str, Any]
    ] = []

    cleaned = (
        raw_dataframe
        .copy(
            deep=True
        )
    )

    # ========================================================
    # 1. COLUMN NAMES
    # ========================================================

    cleaned = (
        clean_column_names(
            cleaned,
            operations,
            review_items,
        )
    )

    # ========================================================
    # 2. STRING WHITESPACE
    # ========================================================

    cleaned = (
        trim_string_values(
            cleaned,
            operations,
        )
    )

    # ========================================================
    # 3. MISSING VALUES
    # ========================================================

    cleaned = (
        normalize_missing_values(
            cleaned,
            operations,
            review_items,
        )
    )

    # ========================================================
    # 4. ENTIRELY EMPTY ROWS
    # ========================================================

    cleaned = (
        remove_entirely_empty_rows(
            cleaned,
            operations,
        )
    )

    # ========================================================
    # 5. SAFE TYPE CONVERSIONS
    # ========================================================

    cleaned = (
        apply_safe_type_conversions(
            cleaned,
            operations,
            review_items,
        )
    )

    # ========================================================
    # 6. EXACT DUPLICATE REVIEW
    # ========================================================

    detect_exact_duplicate_rows(
        cleaned,
        review_items,
    )

    # ========================================================
    # 7. NON-DESTRUCTIVE QUALITY WARNINGS
    # ========================================================

    detect_quality_review_items(
        cleaned,
        review_items,
    )

    # ========================================================
    # AFTER SNAPSHOT
    # ========================================================

    after = (
        build_quality_snapshot(
            cleaned
        )
    )

    affected_values = sum(
        int(
            operation.get(
                "affected_values",
                0,
            )
        )
        for operation
        in operations
    )

    affected_rows = sum(
        int(
            operation.get(
                "affected_rows",
                0,
            )
        )
        for operation
        in operations
    )

    affected_columns = sum(
        int(
            operation.get(
                "affected_columns",
                0,
            )
        )
        for operation
        in operations
    )

    report = {
        "report_version":
            "0.2",

        "filename":
            filename,

        "status":
            "cleaned",

        "policy":
            {
                "mode":
                    "conservative",

                "raw_preserved":
                    True,

                "ambiguous_changes_applied":
                    False,

                "entirely_empty_rows_removed":
                    True,

                "exact_duplicates_removed":
                    False,
            },

        "before":
            before,

        "after":
            after,

        "summary":
            {
                "operations_count":
                    len(
                        operations
                    ),

                "review_items_count":
                    len(
                        review_items
                    ),

                "affected_values":
                    affected_values,

                "affected_rows":
                    affected_rows,

                "affected_columns":
                    affected_columns,

                "rows_removed":
                    (
                        before[
                            "rows"
                        ]
                        -
                        after[
                            "rows"
                        ]
                    ),

                "missing_cells_change":
                    (
                        after[
                            "missing_cells"
                        ]
                        -
                        before[
                            "missing_cells"
                        ]
                    ),

                "raw_preserved":
                    True,

                "automatic_policy":
                    "conservative",
            },

        "operations":
            operations,

        "review_required":
            review_items,
    }

    # `operations` is also returned separately
    # because it becomes our transformation history.

    return (
        cleaned,
        report,
        operations,
    )