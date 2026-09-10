from __future__ import annotations


import ast
from pathlib import Path


import numpy as np
import pandas as pd


from app.ml.contracts import (
    MLTimeHoldoutSplitContract,
)


from app.ml.splitting import (
    resolve_ml_feature_holdout_partition,
)


from app.ml.time_series_contracts import (
    MLTimeSeriesForecastingContract,
)


from app.ml.time_series_holdout import (
    ML_TIME_SERIES_HOLDOUT_RULE_VERSION,
    MLTimeSeriesHoldoutError,
    resolve_time_series_holdout,
)


from app.ml.time_series_training_input import (
    MLValidatedUnivariateTimeSeries,
    validate_and_extract_time_series,
)


# ============================================================
# HELPERS
# ============================================================


def make_contract(
    *,
    lookback: int = 4,
    test_size: float = 0.25,
) -> MLTimeSeriesForecastingContract:

    return (
        MLTimeSeriesForecastingContract(
            workflow_id=
                "workflow:forecast-holdout",
            dataset_id=
                "dataset:forecast-holdout",
            target_column=
                "revenue",
            lookback=
                lookback,
            split=
                MLTimeHoldoutSplitContract(
                    time_column=
                        "order_date",
                    test_size=
                        test_size,
                ),
        )
    )


def make_validated_series(
    *,
    rows: int = 20,
    contract: (
        MLTimeSeriesForecastingContract
        |
        None
    ) = None,
) -> MLValidatedUnivariateTimeSeries:

    effective_contract = (
        contract
        if contract is not None
        else make_contract()
    )


    canonical = pd.DataFrame(
        {
            "order_date":
                pd.date_range(
                    "2026-01-01",
                    periods=rows,
                    freq="D",
                ),

            "revenue":
                [
                    float(
                        100
                        +
                        index
                    )
                    for index
                    in range(
                        rows
                    )
                ],
        }
    )


    if rows == 20:

        source_order = [
            5,
            0,
            10,
            15,
            1,
            6,
            11,
            16,
            2,
            7,
            12,
            17,
            3,
            8,
            13,
            18,
            4,
            9,
            14,
            19,
        ]

        canonical = (
            canonical.iloc[
                source_order
            ]
            .reset_index(
                drop=True
            )
        )


    return (
        validate_and_extract_time_series(
            dataframe=
                canonical,

            contract=
                effective_contract,
        )
    )


def require_holdout_error(
    callback,
    *,
    label: str,
) -> None:

    try:
        callback()

    except MLTimeSeriesHoldoutError:
        return


    raise AssertionError(
        (
            "Expected forecasting holdout failure: "
            f"{label}"
        )
    )


# ============================================================
# 1. SHARED HOLDOUT AUTHORITY PARITY
# ============================================================


contract = make_contract()

series = make_validated_series(
    contract=contract
)


partition = (
    resolve_time_series_holdout(
        series=
            series,

        contract=
            contract,
    )
)


canonical_frame = pd.DataFrame(
    {
        contract.time_column:
            series.time_values
    }
)


alignment_features = pd.DataFrame(
    index=canonical_frame.index
)


shared = (
    resolve_ml_feature_holdout_partition(
        x=
            alignment_features,

        dataframe=
            canonical_frame,

        contract=
            contract,
    )
)


assert (
    partition.train_positions
    ==
    shared.train_positions
)


assert (
    partition.test_positions
    ==
    shared.test_positions
)


assert (
    partition.rule_version
    ==
    ML_TIME_SERIES_HOLDOUT_RULE_VERSION
)


print(
    "[PASS] forecasting holdout reuses exact shared Model Lab split positions"
)


# ============================================================
# 2. EXPECTED CHRONOLOGICAL POPULATIONS
# ============================================================


assert (
    partition.source_row_count
    ==
    20
)


assert (
    partition.train_positions
    ==
    tuple(
        range(
            15
        )
    )
)


assert (
    partition.test_positions
    ==
    tuple(
        range(
            15,
            20,
        )
    )
)


assert (
    partition.train_rows
    ==
    15
)


assert (
    partition.test_rows
    ==
    5
)


assert (
    partition.train_end_time
    ==
    pd.Timestamp(
        "2026-01-15"
    )
)


assert (
    partition.test_start_time
    ==
    pd.Timestamp(
        "2026-01-16"
    )
)


assert (
    partition.train_end_time
    <
    partition.test_start_time
)


print(
    "[PASS] TRAIN is strictly historical and TEST is strictly future"
)


# ============================================================
# 3. TARGET-BLIND SPLIT IDENTITY
# ============================================================


mutated_targets = (
    series.target_values
    .copy(
        deep=True
    )
)


mutated_targets.iloc[:] = (
    np.arange(
        len(
            mutated_targets
        ),
        dtype=np.float64,
    )
    *
    999999.0
)


target_mutated_series = (
    MLValidatedUnivariateTimeSeries(
        time_values=
            series.time_values
            .copy(
                deep=True
            ),

        target_values=
            mutated_targets,

        source_row_positions=
            series.source_row_positions
            .copy(),

        rule_version=
            series.rule_version,
    )
)


target_mutated_partition = (
    resolve_time_series_holdout(
        series=
            target_mutated_series,

        contract=
            contract,
    )
)


assert (
    target_mutated_partition.train_positions
    ==
    partition.train_positions
)


assert (
    target_mutated_partition.test_positions
    ==
    partition.test_positions
)


print(
    "[PASS] outer split identity is independent of target values"
)


# ============================================================
# 4. SOURCE-ROW PROVENANCE
# ============================================================


expected_train_source = tuple(
    int(
        series.source_row_positions[
            position
        ]
    )
    for position
    in partition.train_positions
)


expected_test_source = tuple(
    int(
        series.source_row_positions[
            position
        ]
    )
    for position
    in partition.test_positions
)


assert (
    partition.train_source_row_positions
    ==
    expected_train_source
)


assert (
    partition.test_source_row_positions
    ==
    expected_test_source
)


assert not (
    set(
        partition.train_source_row_positions
    )
    &
    set(
        partition.test_source_row_positions
    )
)


assert (
    set(
        (
            *partition.train_source_row_positions,
            *partition.test_source_row_positions,
        )
    )
    ==
    set(
        range(
            20
        )
    )
)


print(
    "[PASS] original Preparation row-position provenance is preserved"
)


# ============================================================
# 5. FIRST TEST TARGET MAY USE HISTORICAL TRAIN CONTEXT
# ============================================================


first_test_target_position = (
    partition.test_positions[
        0
    ]
)


context_start = (
    first_test_target_position
    -
    contract.lookback
)


context_positions = tuple(
    range(
        context_start,
        first_test_target_position,
    )
)


assert (
    context_positions
    ==
    (
        11,
        12,
        13,
        14,
    )
)


assert all(
    position
    in
    partition.train_positions

    for position
    in context_positions
)


assert (
    first_test_target_position
    not in
    context_positions
)


print(
    "[PASS] first future target can later use TRAIN-only historical context"
)


# ============================================================
# 6. TRAIN MUST SUPPORT AT LEAST ONE LOOKBACK WINDOW
# ============================================================


short_contract = (
    make_contract(
        lookback=6,
        test_size=0.40,
    )
)


short_series = (
    make_validated_series(
        rows=10,
        contract=
            short_contract,
    )
)


require_holdout_error(
    lambda:
        resolve_time_series_holdout(
            series=
                short_series,

            contract=
                short_contract,
        ),
    label=
        "TRAIN shorter than lookback + 1",
)


print(
    "[PASS] holdout fails closed when TRAIN cannot form one forecasting sample"
)


# ============================================================
# 7. NON-CANONICAL SERIES FAILS CLOSED
# ============================================================


non_canonical_times = (
    series.time_values
    .copy(
        deep=True
    )
)


non_canonical_times.iloc[
    [0, 1]
] = (
    non_canonical_times.iloc[
        [1, 0]
    ].to_numpy()
)


non_canonical_series = (
    MLValidatedUnivariateTimeSeries(
        time_values=
            non_canonical_times,

        target_values=
            series.target_values
            .copy(
                deep=True
            ),

        source_row_positions=
            series.source_row_positions
            .copy(),

        rule_version=
            series.rule_version,
    )
)


require_holdout_error(
    lambda:
        resolve_time_series_holdout(
            series=
                non_canonical_series,

            contract=
                contract,
        ),
    label=
        "non-canonical chronology",
)


print(
    "[PASS] tampered non-canonical chronology fails closed"
)


# ============================================================
# 8. INVALID SOURCE-POSITION PROVENANCE FAILS CLOSED
# ============================================================


invalid_source_positions = (
    series.source_row_positions
    .copy()
)


invalid_source_positions.setflags(
    write=True
)


invalid_source_positions[
    -1
] = (
    invalid_source_positions[
        -2
    ]
)


invalid_provenance_series = (
    MLValidatedUnivariateTimeSeries(
        time_values=
            series.time_values
            .copy(
                deep=True
            ),

        target_values=
            series.target_values
            .copy(
                deep=True
            ),

        source_row_positions=
            invalid_source_positions,

        rule_version=
            series.rule_version,
    )
)


require_holdout_error(
    lambda:
        resolve_time_series_holdout(
            series=
                invalid_provenance_series,

            contract=
                contract,
        ),
    label=
        "invalid source-row permutation",
)


print(
    "[PASS] invalid source-row provenance fails closed"
)


# ============================================================
# 9. STATIC TORCH-FREE SURFACE
# ============================================================


source_path = Path(
    "app/ml/time_series_holdout.py"
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
    "[PASS] forecasting holdout authority is torch-free"
)


print()
print("=" * 80)
print("DL-5-A3-V1 FINAL VERDICT")
print("=" * 80)
print()

print(
    "Shared chronological authority             PASS"
)

print(
    "Target-blind outer split                   PASS"
)

print(
    "Strict TRAIN-before-TEST boundary          PASS"
)

print(
    "Exact observation population coverage      PASS"
)

print(
    "Original row-position provenance           PASS"
)

print(
    "TRAIN lookback feasibility                 PASS"
)

print(
    "Future-window context policy ready         PASS"
)

print(
    "Tamper guards                              PASS"
)

print(
    "Torch-free boundary                        PASS"
)

print()
print(
    "DL-5-A3-V1 - FORECAST HOLDOUT AUTHORITY: PASS"
)
