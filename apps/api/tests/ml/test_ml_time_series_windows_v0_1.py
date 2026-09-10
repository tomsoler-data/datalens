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


from app.ml.time_series_holdout import (
    MLTimeSeriesHoldoutPartition,
    resolve_time_series_holdout,
)


from app.ml.time_series_training_input import (
    validate_and_extract_time_series,
)


from app.ml.time_series_windows import (
    ML_TIME_SERIES_EVALUATION_PROTOCOL,
    ML_TIME_SERIES_WINDOWS_RULE_VERSION,
    MLTimeSeriesWindowError,
    build_time_series_supervised_windows,
)


# ============================================================
# HELPERS
# ============================================================


def make_contract(
    *,
    lookback: int = 4,
) -> MLTimeSeriesForecastingContract:

    return (
        MLTimeSeriesForecastingContract(
            workflow_id=
                "workflow:forecast-windows",
            dataset_id=
                "dataset:forecast-windows",
            target_column=
                "revenue",
            lookback=
                lookback,
            split=
                MLTimeHoldoutSplitContract(
                    time_column=
                        "order_date",
                    test_size=
                        0.25,
                ),
        )
    )


def make_series_and_partition():
    contract = (
        make_contract()
    )


    canonical = pd.DataFrame(
        {
            "order_date":
                pd.date_range(
                    "2026-01-01",
                    periods=20,
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
                        20
                    )
                ],
        }
    )


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


    source = (
        canonical.iloc[
            source_order
        ]
        .reset_index(
            drop=True
        )
    )


    series = (
        validate_and_extract_time_series(
            dataframe=
                source,
            contract=
                contract,
        )
    )


    partition = (
        resolve_time_series_holdout(
            series=
                series,
            contract=
                contract,
        )
    )


    return (
        contract,
        series,
        partition,
    )


def require_window_error(
    callback,
    *,
    label: str,
) -> None:

    try:
        callback()

    except MLTimeSeriesWindowError:
        return


    raise AssertionError(
        (
            "Expected forecast-window failure: "
            f"{label}"
        )
    )


# ============================================================
# 1. COMMON SUPERVISED POPULATION
# ============================================================


(
    contract,
    series,
    partition,
) = make_series_and_partition()


windows = (
    build_time_series_supervised_windows(
        series=
            series,
        partition=
            partition,
        contract=
            contract,
    )
)


assert (
    windows.lookback
    ==
    4
)


assert (
    windows.forecast_horizon
    ==
    1
)


assert (
    windows.evaluation_protocol
    ==
    ML_TIME_SERIES_EVALUATION_PROTOCOL
)


assert (
    windows.evaluation_protocol
    ==
    "one_step_rolling_origin_observed_history"
)


assert (
    windows.rule_version
    ==
    ML_TIME_SERIES_WINDOWS_RULE_VERSION
)


print(
    "[PASS] common framework-neutral forecasting population created"
)


# ============================================================
# 2. EXACT TRAIN WINDOW POPULATION
# ============================================================


assert (
    windows.train.sample_count
    ==
    11
)


assert (
    windows.train.lookback
    ==
    4
)


assert (
    windows.train.inputs.shape
    ==
    (
        11,
        4,
    )
)


assert (
    windows.train.targets.shape
    ==
    (
        11,
    )
)


assert (
    windows.train.target_positions
    ==
    tuple(
        range(
            4,
            15,
        )
    )
)


np.testing.assert_array_equal(
    windows.train.inputs[
        0
    ],
    np.array(
        [
            100.0,
            101.0,
            102.0,
            103.0,
        ],
        dtype=np.float64,
    ),
)


assert (
    windows.train.targets[
        0
    ]
    ==
    104.0
)


np.testing.assert_array_equal(
    windows.train.inputs[
        -1
    ],
    np.array(
        [
            110.0,
            111.0,
            112.0,
            113.0,
        ],
        dtype=np.float64,
    ),
)


assert (
    windows.train.targets[
        -1
    ]
    ==
    114.0
)


print(
    "[PASS] TRAIN windows use only historical TRAIN observations"
)


# ============================================================
# 3. EXACT TEST WINDOW POPULATION
# ============================================================


assert (
    windows.test.sample_count
    ==
    5
)


assert (
    windows.test.target_positions
    ==
    (
        15,
        16,
        17,
        18,
        19,
    )
)


assert (
    windows.test.context_positions[
        0
    ]
    ==
    (
        11,
        12,
        13,
        14,
    )
)


np.testing.assert_array_equal(
    windows.test.inputs[
        0
    ],
    np.array(
        [
            111.0,
            112.0,
            113.0,
            114.0,
        ],
        dtype=np.float64,
    ),
)


assert (
    windows.test.targets[
        0
    ]
    ==
    115.0
)


print(
    "[PASS] first TEST target uses historical TRAIN context only"
)


# ============================================================
# 4. ONE-STEP ROLLING-ORIGIN OBSERVED HISTORY
# ============================================================


assert (
    windows.test.context_positions[
        1
    ]
    ==
    (
        12,
        13,
        14,
        15,
    )
)


assert (
    15
    in
    windows.test.context_positions[
        1
    ]
)


assert (
    windows.test.target_positions[
        0
    ]
    ==
    15
)


assert (
    windows.test.target_positions[
        1
    ]
    ==
    16
)


np.testing.assert_array_equal(
    windows.test.inputs[
        1
    ],
    np.array(
        [
            112.0,
            113.0,
            114.0,
            115.0,
        ],
        dtype=np.float64,
    ),
)


assert (
    windows.test.targets[
        1
    ]
    ==
    116.0
)


print(
    "[PASS] later TEST forecasts use only already-observed historical TEST values"
)


# ============================================================
# 5. NO CURRENT/FUTURE TARGET LEAKAGE
# ============================================================


for batch in (
    windows.train,
    windows.test,
):

    assert (
        len(
            batch.context_positions
        )
        ==
        batch.sample_count
    )


    assert (
        len(
            batch.target_positions
        )
        ==
        batch.sample_count
    )


    for (
        context,
        target_position,
    ) in zip(
        batch.context_positions,
        batch.target_positions,
    ):

        assert (
            len(
                context
            )
            ==
            contract.lookback
        )


        assert all(
            position
            <
            target_position
            for position
            in context
        )


        assert (
            target_position
            not in
            context
        )


print(
    "[PASS] no current target or future observation can enter any window"
)


# ============================================================
# 6. TARGET / CONTEXT PROVENANCE
# ============================================================


source_positions = (
    series.source_row_positions
)


for batch in (
    windows.train,
    windows.test,
):

    for sample_index in range(
        batch.sample_count
    ):

        target_position = (
            batch.target_positions[
                sample_index
            ]
        )


        expected_target_source = int(
            source_positions[
                target_position
            ]
        )


        assert (
            batch.target_source_row_positions[
                sample_index
            ]
            ==
            expected_target_source
        )


        expected_context_source = tuple(
            int(
                source_positions[
                    position
                ]
            )
            for position
            in batch.context_positions[
                sample_index
            ]
        )


        assert (
            batch.context_source_row_positions[
                sample_index
            ]
            ==
            expected_context_source
        )


print(
    "[PASS] context and target Preparation row provenance is exact"
)


# ============================================================
# 7. TARGET TIMES
# ============================================================


assert (
    windows.train.target_times[
        0
    ]
    ==
    pd.Timestamp(
        "2026-01-05"
    )
)


assert (
    windows.test.target_times[
        0
    ]
    ==
    pd.Timestamp(
        "2026-01-16"
    )
)


assert (
    windows.test.target_times[
        -1
    ]
    ==
    pd.Timestamp(
        "2026-01-20"
    )
)


print(
    "[PASS] every supervised target preserves its forecast timestamp"
)


# ============================================================
# 8. OUTPUT ARRAYS ARE IMMUTABLE AND ISOLATED
# ============================================================


assert (
    windows.train.inputs.flags.writeable
    is False
)


assert (
    windows.train.targets.flags.writeable
    is False
)


assert (
    windows.test.inputs.flags.writeable
    is False
)


assert (
    windows.test.targets.flags.writeable
    is False
)


before_train_value = float(
    windows.train.inputs[
        0,
        0,
    ]
)


before_test_value = float(
    windows.test.inputs[
        0,
        0,
    ]
)


series.target_values.iloc[
    0
] = 999999.0


assert (
    windows.train.inputs[
        0,
        0
    ]
    ==
    before_train_value
)


assert (
    windows.test.inputs[
        0,
        0
    ]
    ==
    before_test_value
)


print(
    "[PASS] supervised windows are immutable and isolated from source mutation"
)


# ============================================================
# 9. TAMPERED NON-PREFIX HOLDOUT FAILS CLOSED
# ============================================================


(
    tamper_contract,
    tamper_series,
    good_partition,
) = make_series_and_partition()


tampered_train = tuple(
    list(
        range(
            14
        )
    )
    +
    [
        15,
    ]
)


tampered_test = (
    14,
    16,
    17,
    18,
    19,
)


source_row_positions = (
    tamper_series.source_row_positions
)


tampered_partition = (
    MLTimeSeriesHoldoutPartition(
        source_row_count=
            20,

        train_positions=
            tampered_train,

        test_positions=
            tampered_test,

        train_source_row_positions=
            tuple(
                int(
                    source_row_positions[
                        position
                    ]
                )
                for position
                in tampered_train
            ),

        test_source_row_positions=
            tuple(
                int(
                    source_row_positions[
                        position
                    ]
                )
                for position
                in tampered_test
            ),

        train_end_time=
            good_partition.train_end_time,

        test_start_time=
            good_partition.test_start_time,
    )
)


require_window_error(
    lambda:
        build_time_series_supervised_windows(
            series=
                tamper_series,

            partition=
                tampered_partition,

            contract=
                tamper_contract,
        ),
    label=
        "non-prefix holdout",
)


print(
    "[PASS] tampered non-prefix TRAIN/TEST holdout fails closed"
)


# ============================================================
# 10. STATIC TORCH-FREE SURFACE
# ============================================================


source_path = Path(
    "app/ml/time_series_windows.py"
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
    "[PASS] supervised forecasting-window authority is torch-free"
)


print()
print("=" * 80)
print("DL-5-A4-V1 FINAL VERDICT")
print("=" * 80)
print()

print(
    "Framework-neutral sample population        PASS"
)

print(
    "Exact TRAIN lookback windows               PASS"
)

print(
    "Exact TEST lookback windows                PASS"
)

print(
    "One-step rolling-origin protocol           PASS"
)

print(
    "No current/future leakage                  PASS"
)

print(
    "Context provenance                         PASS"
)

print(
    "Target provenance                          PASS"
)

print(
    "Forecast timestamps                        PASS"
)

print(
    "Immutable NumPy outputs                    PASS"
)

print(
    "Torch-free boundary                        PASS"
)

print()
print(
    "DL-5-A4-V1 - SUPERVISED FORECAST WINDOWS: PASS"
)
