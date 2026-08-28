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


DATASET_ID = (
    "combine:test-revenue"
)


def match(
    concept: str,
    column: str,
) -> RequestedColumnMatch:
    return RequestedColumnMatch(
        concept=
            concept,

        dataset_id=
            DATASET_ID,

        dataset_filename=
            "orders.csv",

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


def plan() -> RequestedAnalysisPlan:
    return RequestedAnalysisPlan(
        request_id=
            "request:derived-revenue",

        request_text=
            "?volution du chiffre d?affaires / moyenne mobile.",

        evidence_quote=
            "?volution du chiffre d?affaires / moyenne mobile.",

        source_filename=
            "workspace:analysis-follow-up",

        source_locator=
            "follow_up_prompt",

        page_number=
            None,

        source_chunk_id=
            "follow-up:test",

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
                    "month",

                moving_average_window=
                    2,
            ),

        target_family=
            "time_series",

        matched_columns=[
            match(
                "time",
                "order_date",
            ),

            match(
                "amount",
                "unit_price",
            ),
        ],

        required_dataset_ids=[
            DATASET_ID
        ],

        required_dataset_filenames=[
            "orders.csv"
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
        DATASET_ID,

    "filename":
        "orders.csv",

    "is_derived":
        False,

    "dataframe":
        pd.DataFrame(
            {
                "order_date": [
                    "2024-01-02",
                    "2024-01-18",
                    "2024-02-03",
                    "2024-02-20",
                ],

                "quantity": [
                    2,
                    1,
                    3,
                    2,
                ],

                "unit_price": [
                    10.0,
                    5.0,
                    4.0,
                    6.0,
                ],
            }
        ),
}


certificate = {
    "dataset_id":
        "derived:test:monthly:order_date:gross_amount",

    "filename":
        "orders__monthly_gross_amount.derived",

    "is_derived":
        True,

    "derivation_type":
        "monthly_additive_measure",

    "provenance": {
        "fact_dataset_id":
            DATASET_ID,

        "operation":
            "groupby_sum",

        "source_time_column":
            "order_date",

        "source_measure_column":
            "gross_amount",

        "source_measure_derivation": {
            "operation":
                "analytical_line_amount_derivation",

            "derived_column":
                "gross_amount",

            "source_quantity_column":
                "quantity",

            "source_unit_price_column":
                "unit_price",

            "formula":
                "quantity * unit_price",

            "valid_count":
                4,

            "missing_count":
                0,

            "analytical_only":
                True,

            "safety_policy":
                "test strict derivation",
        },

        "target_time_column":
            "month",

        "target_measure_column":
            "sum_gross_amount",

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

                "sum_gross_amount": [
                    25.0,
                    24.0,
                ],

                "event_count": [
                    2,
                    2,
                ],
            }
        ),
}


print(
    "=== DATALENS CERTIFIED DERIVED REVENUE "
    "MOVING AVERAGE v0.1 ==="
)

print()


execution = (
    execute_revenue_moving_average(
        request=
            plan(),

        datasets=[
            source,
            certificate,
        ],
    )
)


assert (
    execution.execution_status
    ==
    "complete"
)

assert (
    execution.result
    is not None
)


metrics = (
    execution.result.metrics
)

chart = (
    execution.result.chart_data
)


assert (
    metrics[
        "measure_column"
    ]
    ==
    "gross_amount"
)

assert (
    metrics[
        "measure_source_mode"
    ]
    ==
    "derived_line_amount"
)

assert (
    metrics[
        "planner_amount_column"
    ]
    ==
    "unit_price"
)

assert (
    metrics[
        "source_quantity_column"
    ]
    ==
    "quantity"
)


values = [
    float(
        point[
            "value"
        ]
    )

    for point
    in chart
]


moving = [
    float(
        point[
            "moving_average"
        ]
    )

    for point
    in chart
]


assert (
    values
    ==
    [
        25.0,
        24.0,
    ]
)


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
        moving,
        [
            25.0,
            24.5,
        ],
    )
)


assert math.isclose(
    float(
        metrics[
            "total_revenue"
        ]
    ),
    49.0,
)


# SUM(unit_price) would incorrectly be 25.
# Revenue must be quantity * unit_price = 49.

assert not math.isclose(
    float(
        metrics[
            "total_revenue"
        ]
    ),
    float(
        source[
            "dataframe"
        ][
            "unit_price"
        ].sum()
    ),
)


assert (
    execution.variables[
        "value"
    ]
    ==
    "gross_amount"
)


print(
    "[PASS] planner unit_price binds to certified gross_amount lineage"
)

print(
    "[PASS] revenue is recomputed as quantity ? unit_price"
)

print(
    "[PASS] SUM(unit_price) is never substituted for revenue"
)

print(
    "[PASS] monthly moving average uses certified event revenue"
)


bad_certificate = {
    **certificate,

    "provenance": {
        **certificate[
            "provenance"
        ],

        "source_measure_derivation": {
            **certificate[
                "provenance"
            ][
                "source_measure_derivation"
            ],

            "source_unit_price_column":
                "other_price",

            "formula":
                "quantity * other_price",
        },
    },
}


blocked = (
    execute_revenue_moving_average(
        request=
            plan(),

        datasets=[
            source,
            bad_certificate,
        ],
    )
)


assert (
    blocked.execution_status
    ==
    "needs_information"
)


print(
    "[PASS] mismatched derivation provenance remains fail-closed"
)

print(
    "[PASS] executor rule version "
    +
    REQUESTED_ANALYSIS_EXECUTOR_RULE_VERSION
)

print()
print(
    "PASS - certified derived revenue moving average v0.1"
)
