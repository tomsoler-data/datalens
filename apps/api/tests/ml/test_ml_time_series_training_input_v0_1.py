from __future__ import annotations


import ast
from pathlib import Path


import numpy as np
import pandas as pd


from app.ml.contracts import (
    MLTimeHoldoutSplitContract,
)


from app.ml.time_series_contracts import (
    MLTimeSeriesForecastingContract,
)


from app.ml.time_series_training_input import (
    ML_TIME_SERIES_TRAINING_INPUT_RULE_VERSION,
    MLTimeSeriesTrainingInputError,
    validate_and_extract_time_series,
)


# ============================================================
# HELPERS
# ============================================================


def make_contract(
    *,
    lookback: int = 3,
) -> MLTimeSeriesForecastingContract:

    return (
        MLTimeSeriesForecastingContract(
            workflow_id=
                "workflow:time-series-input",
            dataset_id=
                "dataset:time-series-input",
            target_column=
                "revenue",
            lookback=
                lookback,
            split=
                MLTimeHoldoutSplitContract(
                    time_column=
                        "order_date",
                    test_size=
                        0.20,
                ),
        )
    )


def require_input_error(
    callback,
    *,
    label: str,
) -> None:

    try:
        callback()

    except MLTimeSeriesTrainingInputError:
        return


    raise AssertionError(
        (
            "Expected forecasting input failure: "
            f"{label}"
        )
    )


# ============================================================
# 1. CANONICAL CHRONOLOGICAL ORDER
# ============================================================


frame = pd.DataFrame(
    {
        "order_date":
            pd.to_datetime(
                [
                    "2026-03-01",
                    "2026-01-01",
                    "2026-04-01",
                    "2026-02-01",
                    "2026-05-01",
                ]
            ),

        "revenue":
            [
                30.0,
                10.0,
                40.0,
                20.0,
                50.0,
            ],

        "ignored_column":
            [
                "c",
                "a",
                "d",
                "b",
                "e",
            ],
    },
    index=[
        100,
        101,
        102,
        103,
        104,
    ],
)


validated = (
    validate_and_extract_time_series(
        dataframe=
            frame,
        contract=
            make_contract(),
    )
)


assert (
    validated.row_count
    ==
    5
)


assert (
    validated.time_values.tolist()
    ==
    pd.to_datetime(
        [
            "2026-01-01",
            "2026-02-01",
            "2026-03-01",
            "2026-04-01",
            "2026-05-01",
        ]
    ).tolist()
)


assert (
    validated.target_values.tolist()
    ==
    [
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
    ]
)


assert (
    validated.source_row_positions.tolist()
    ==
    [
        1,
        3,
        0,
        2,
        4,
    ]
)


assert (
    validated.rule_version
    ==
    ML_TIME_SERIES_TRAINING_INPUT_RULE_VERSION
)


print(
    "[PASS] source rows canonicalize into deterministic chronological order"
)


# ============================================================
# 2. SOURCE ISOLATION
# ============================================================


frame.loc[
    101,
    "revenue",
] = 999999.0


frame.loc[
    101,
    "order_date",
] = pd.Timestamp(
    "2099-01-01"
)


assert (
    validated.target_values.iloc[
        0
    ]
    ==
    10.0
)


assert (
    validated.time_values.iloc[
        0
    ]
    ==
    pd.Timestamp(
        "2026-01-01"
    )
)


assert (
    validated.source_row_positions.flags.writeable
    is False
)


print(
    "[PASS] validated series is isolated from later source mutations"
)


# ============================================================
# 3. NO IMPLICIT DATE PARSING
# ============================================================


string_dates = pd.DataFrame(
    {
        "order_date":
            [
                "2026-01-01",
                "2026-02-01",
                "2026-03-01",
                "2026-04-01",
            ],

        "revenue":
            [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
    }
)


require_input_error(
    lambda:
        validate_and_extract_time_series(
            dataframe=
                string_dates,
            contract=
                make_contract(),
        ),
    label=
        "implicit string-date parsing",
)


print(
    "[PASS] time axis must already be validated datetime data"
)


# ============================================================
# 4. NO MISSING TIMESTAMPS
# ============================================================


missing_time = pd.DataFrame(
    {
        "order_date":
            pd.to_datetime(
                [
                    "2026-01-01",
                    None,
                    "2026-03-01",
                    "2026-04-01",
                ]
            ),

        "revenue":
            [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
    }
)


require_input_error(
    lambda:
        validate_and_extract_time_series(
            dataframe=
                missing_time,
            contract=
                make_contract(),
        ),
    label=
        "missing timestamp",
)


print(
    "[PASS] missing timestamps fail closed"
)


# ============================================================
# 5. ONE OBSERVATION PER TIMESTAMP
# ============================================================


duplicate_time = pd.DataFrame(
    {
        "order_date":
            pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-02-01",
                    "2026-02-01",
                    "2026-03-01",
                    "2026-04-01",
                ]
            ),

        "revenue":
            [
                10.0,
                20.0,
                25.0,
                30.0,
                40.0,
            ],
    }
)


require_input_error(
    lambda:
        validate_and_extract_time_series(
            dataframe=
                duplicate_time,
            contract=
                make_contract(),
        ),
    label=
        "duplicate timestamps",
)


print(
    "[PASS] duplicate timestamps require explicit Preparation aggregation"
)


# ============================================================
# 6. TARGET MUST BE NUMERIC / FINITE / COMPLETE
# ============================================================


non_numeric = pd.DataFrame(
    {
        "order_date":
            pd.date_range(
                "2026-01-01",
                periods=4,
                freq="D",
            ),

        "revenue":
            [
                "10",
                "20",
                "30",
                "40",
            ],
    }
)


require_input_error(
    lambda:
        validate_and_extract_time_series(
            dataframe=
                non_numeric,
            contract=
                make_contract(),
        ),
    label=
        "string target",
)


boolean_target = pd.DataFrame(
    {
        "order_date":
            pd.date_range(
                "2026-01-01",
                periods=4,
                freq="D",
            ),

        "revenue":
            [
                True,
                False,
                True,
                False,
            ],
    }
)


require_input_error(
    lambda:
        validate_and_extract_time_series(
            dataframe=
                boolean_target,
            contract=
                make_contract(),
        ),
    label=
        "boolean target",
)


missing_target = pd.DataFrame(
    {
        "order_date":
            pd.date_range(
                "2026-01-01",
                periods=4,
                freq="D",
            ),

        "revenue":
            [
                10.0,
                np.nan,
                30.0,
                40.0,
            ],
    }
)


require_input_error(
    lambda:
        validate_and_extract_time_series(
            dataframe=
                missing_target,
            contract=
                make_contract(),
        ),
    label=
        "missing target",
)


non_finite = pd.DataFrame(
    {
        "order_date":
            pd.date_range(
                "2026-01-01",
                periods=4,
                freq="D",
            ),

        "revenue":
            [
                10.0,
                20.0,
                np.inf,
                40.0,
            ],
    }
)


require_input_error(
    lambda:
        validate_and_extract_time_series(
            dataframe=
                non_finite,
            contract=
                make_contract(),
        ),
    label=
        "non-finite target",
)


constant_target = pd.DataFrame(
    {
        "order_date":
            pd.date_range(
                "2026-01-01",
                periods=4,
                freq="D",
            ),

        "revenue":
            [
                10.0,
                10.0,
                10.0,
                10.0,
            ],
    }
)


require_input_error(
    lambda:
        validate_and_extract_time_series(
            dataframe=
                constant_target,
            contract=
                make_contract(),
        ),
    label=
        "constant target",
)


print(
    "[PASS] target authority enforces numeric finite non-constant observations"
)


# ============================================================
# 7. ENOUGH OBSERVATIONS FOR LOOKBACK
# ============================================================


too_short = pd.DataFrame(
    {
        "order_date":
            pd.date_range(
                "2026-01-01",
                periods=3,
                freq="D",
            ),

        "revenue":
            [
                10.0,
                20.0,
                30.0,
            ],
    }
)


require_input_error(
    lambda:
        validate_and_extract_time_series(
            dataframe=
                too_short,
            contract=
                make_contract(
                    lookback=3
                ),
        ),
    label=
        "rows <= lookback",
)


print(
    "[PASS] series must contain at least lookback + 1 observations"
)


# ============================================================
# 8. REQUIRED COLUMNS
# ============================================================


missing_target_column = pd.DataFrame(
    {
        "order_date":
            pd.date_range(
                "2026-01-01",
                periods=4,
                freq="D",
            ),

        "sales":
            [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
    }
)


require_input_error(
    lambda:
        validate_and_extract_time_series(
            dataframe=
                missing_target_column,
            contract=
                make_contract(),
        ),
    label=
        "missing target column",
)


missing_time_column = pd.DataFrame(
    {
        "date":
            pd.date_range(
                "2026-01-01",
                periods=4,
                freq="D",
            ),

        "revenue":
            [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
    }
)


require_input_error(
    lambda:
        validate_and_extract_time_series(
            dataframe=
                missing_time_column,
            contract=
                make_contract(),
        ),
    label=
        "missing time column",
)


print(
    "[PASS] declared time and target columns are authoritative"
)


# ============================================================
# 9. NO HIDDEN FEATURES / AGGREGATION
# ============================================================


assert (
    not hasattr(
        validated,
        "feature_values",
    )
)


assert (
    validated.target_values.name
    ==
    "revenue"
)


assert (
    len(
        validated.target_values
    )
    ==
    5
)


print(
    "[PASS] v0.1 input remains univariate with no hidden aggregation"
)


# ============================================================
# 10. STATIC TORCH-FREE SURFACE
# ============================================================


source_path = Path(
    "app/ml/time_series_training_input.py"
)


source = source_path.read_text(
    encoding="utf-8"
)


tree = ast.parse(
    source
)


imports = []


for node in ast.walk(
    tree
):

    if isinstance(
        node,
        ast.Import,
    ):

        imports.extend(
            alias.name
            for alias in node.names
        )


    elif isinstance(
        node,
        ast.ImportFrom,
    ):

        if node.module:

            imports.append(
                node.module
            )


assert not any(
    name
    ==
    "torch"
    or
    name.startswith(
        "torch."
    )
    for name in imports
)


assert not any(
    name.startswith(
        "app.deep_learning"
    )
    for name in imports
)


print(
    "[PASS] time-series input authority is torch-free"
)


print()
print("=" * 80)
print("DL-5-A2-V1 FINAL VERDICT")
print("=" * 80)
print()

print(
    "Datetime authority                         PASS"
)

print(
    "Numeric target authority                   PASS"
)

print(
    "Chronological canonicalization             PASS"
)

print(
    "Original row-position provenance           PASS"
)

print(
    "Duplicate timestamp rejection              PASS"
)

print(
    "No implicit aggregation                    PASS"
)

print(
    "Minimum lookback feasibility               PASS"
)

print(
    "Source isolation                           PASS"
)

print(
    "Univariate surface                         PASS"
)

print(
    "Torch-free boundary                        PASS"
)

print()
print(
    "DL-5-A2-V1 - TIME-SERIES INPUT AUTHORITY: PASS"
)
