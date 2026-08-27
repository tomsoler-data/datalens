import math

import pandas as pd

from app.execution.requested_executor import (
    REQUESTED_ANALYSIS_EXECUTOR_RULE_VERSION,
    execute_revenue_moving_average,
)
from app.planning.schemas import (
    RequestedAnalysisPlan,
    RequestedAnalysisResolution,
    RequestedColumnMatch,
)


def match(
    concept: str,
    column: str,
) -> RequestedColumnMatch:
    return RequestedColumnMatch(
        concept=
            concept,

        dataset_id=
            "dataset:transactions",

        dataset_filename=
            "transactions.csv",

        column=
            column,

        analysis_kind=
            "time_series",

        match_score=
            100,

        reasons=[
            "test"
        ],
    )


def plan(
    *,
    granularity: str,
    window: int,
) -> RequestedAnalysisPlan:
    return RequestedAnalysisPlan(
        request_id=
            "request:moving-average",

        request_text=
            "revenue moving average",

        evidence_quote=
            "revenue moving average",

        source_filename=
            "brief.pdf",

        source_locator=
            "page 1",

        page_number=
            1,

        source_chunk_id=
            "chunk:moving-average",

        evidence_unit_id=
            1,

        kind=
            "revenue_moving_average",

        status=
            "ready",

        resolution=
            RequestedAnalysisResolution(
                resolution_type=
                    "time_series_parameters",

                time_granularity=
                    granularity,

                moving_average_window=
                    window,
            ),

        target_family=
            "time_series",

        matched_columns=[
            match(
                "time",
                "date",
            ),
            match(
                "amount",
                "amount",
            ),
        ],

        required_dataset_ids=[
            "dataset:transactions"
        ],

        required_dataset_filenames=[
            "transactions.csv"
        ],

        required_operations=[
            "resolved"
        ],

        reasons=[
            "test"
        ],

        blockers=[],
    )


source = {
    "dataset_id":
        "dataset:transactions",

    "filename":
        "transactions.csv",

    "is_derived":
        False,

    "dataframe":
        pd.DataFrame(
            {
                "date": [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-08",
                    "2024-02-01",
                ],

                "amount": [
                    10.0,
                    20.0,
                    30.0,
                    40.0,
                ],
            }
        ),
}


certificate = {
    "dataset_id":
        "derived:transactions:monthly:date:amount",

    "filename":
        "transactions__monthly_amount.derived",

    "is_derived":
        True,

    "derivation_type":
        "monthly_additive_measure",

    "provenance": {
        "fact_dataset_id":
            "dataset:transactions",

        "operation":
            "groupby_sum",

        "source_time_column":
            "date",

        "source_measure_column":
            "amount",

        "target_time_column":
            "month",

        "target_measure_column":
            "sum_amount",

        "aggregation":
            "sum",

        "grain":
            "month",
    },

    "dataframe":
        pd.DataFrame(
            {
                "month": [
                    "2024-01-01",
                    "2024-02-01",
                ],

                "sum_amount": [
                    60.0,
                    40.0,
                ],
            }
        ),
}


datasets = [
    source,
    certificate,
]


print(
    "===== DATALENS DYNAMIC MOVING AVERAGE v0.1 ====="
)
print()


# ============================================================
# WEEK
# ============================================================

weekly = (
    execute_revenue_moving_average(
        request=
            plan(
                granularity=
                    "week",

                window=
                    2,
            ),

        datasets=
            datasets,
    )
)


assert (
    weekly.execution_status
    ==
    "complete"
)

assert (
    weekly.analytical_grain
    ==
    "week"
)

assert (
    weekly.result
    is not None
)


weekly_metrics = (
    weekly.result.metrics
)

weekly_chart = (
    weekly.result.chart_data
)


assert (
    weekly_metrics[
        "aggregation_period"
    ]
    ==
    "week"
)

assert (
    weekly_metrics[
        "moving_average_window"
    ]
    ==
    2
)

assert (
    weekly_metrics[
        "valid_observations"
    ]
    ==
    4
)

assert (
    weekly_metrics[
        "period_count"
    ]
    ==
    5
)

assert (
    weekly_metrics[
        "additive_measure_certified"
    ]
    is True
)


weekly_values = [
    float(
        point[
            "value"
        ]
    )

    for point
    in weekly_chart
]


weekly_moving = [
    float(
        point[
            "moving_average"
        ]
    )

    for point
    in weekly_chart
]


assert (
    weekly_values
    ==
    [
        30.0,
        30.0,
        0.0,
        0.0,
        40.0,
    ]
)


expected_moving = [
    30.0,
    30.0,
    15.0,
    0.0,
    20.0,
]


assert all(
    math.isclose(
        actual,
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    for (
        actual,
        expected,
    )
    in zip(
        weekly_moving,
        expected_moving,
    )
)


assert math.isclose(
    float(
        weekly_metrics[
            "total_revenue"
        ]
    ),
    100.0,
)


print(
    "[PASS] weekly aggregation uses actual calendar weeks"
)

print(
    "[PASS] missing weeks are zero-filled"
)

print(
    "[PASS] rolling window=2 is calculated on continuous weeks"
)

print(
    "[PASS] total revenue remains 100"
)


# ============================================================
# MONTH
# ============================================================

monthly = (
    execute_revenue_moving_average(
        request=
            plan(
                granularity=
                    "month",

                window=
                    2,
            ),

        datasets=
            datasets,
    )
)


assert (
    monthly.execution_status
    ==
    "complete"
)

assert (
    monthly.result
    is not None
)


monthly_chart = (
    monthly.result.chart_data
)


assert [
    float(
        point[
            "value"
        ]
    )
    for point
    in monthly_chart
] == [
    60.0,
    40.0,
]


assert [
    float(
        point[
            "moving_average"
        ]
    )
    for point
    in monthly_chart
] == [
    60.0,
    50.0,
]


print(
    "[PASS] monthly aggregation remains compatible with legacy result"
)

print(
    "[PASS] monthly rolling window uses resolved value"
)


# ============================================================
# NO ADDITIVE CERTIFICATE -> FAIL CLOSED
# ============================================================

uncertified = (
    execute_revenue_moving_average(
        request=
            plan(
                granularity=
                    "week",

                window=
                    2,
            ),

        datasets=[
            source
        ],
    )
)


assert (
    uncertified.execution_status
    ==
    "needs_information"
)


print(
    "[PASS] arbitrary monetary SUM remains blocked without additive certificate"
)


# ============================================================
# READY PLAN WITHOUT RESOLUTION -> FAIL CLOSED
# ============================================================

unresolved_payload = (
    plan(
        granularity=
            "month",

        window=
            3,
    )
    .model_dump()
)

unresolved_payload[
    "resolution"
] = None


unresolved = (
    execute_revenue_moving_average(
        request=
            RequestedAnalysisPlan(
                **unresolved_payload
            ),

        datasets=
            datasets,
    )
)


assert (
    unresolved.execution_status
    ==
    "needs_information"
)


print(
    "[PASS] executor refuses silent month/window defaults"
)

print(
    "[PASS] source calculations remain server-owned"
)

print(
    "[PASS] executor rule version "
    +
    REQUESTED_ANALYSIS_EXECUTOR_RULE_VERSION
)

print()
print(
    "PASS - dynamic moving-average execution v0.1"
)
