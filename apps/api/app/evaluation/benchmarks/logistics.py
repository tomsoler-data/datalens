from __future__ import annotations

import numpy as np
import pandas as pd

from app.evaluation.registry_schemas import (
    BenchmarkDatasetSpec,
    SemanticBenchmarkSuite,
)

from app.evaluation.schemas import (
    SemanticColumnBenchmarkCase,
    SemanticFieldExpectation,
    SemanticPairBenchmarkCase,
)


# ============================================================
# IDENTIFIERS
# ============================================================

LOGISTICS_BENCHMARK_ID = (
    "semantic:logistics:regression:v0.1"
)


LOGISTICS_DATASET_ID = (
    "logistics:0001"
)


LOGISTICS_FILENAME = (
    "synthetic_logistics.csv"
)


# ============================================================
# COLUMN NAMES
# ============================================================

EXPECTED_DELIVERIES = (
    "Expected deliveries"
)


COMPLETED_DELIVERIES = (
    "Completed deliveries"
)


LATE_DELIVERIES = (
    "Late deliveries"
)


TRANSIT_TIME = (
    "Transit time (hours)"
)


FREIGHT_COST = (
    "Freight cost"
)


PACKAGE_WEIGHT = (
    "Package weight (kg)"
)


DISTANCE_TRAVELLED = (
    "Distance travelled (km)"
)


ON_TIME_DELIVERY_RATE = (
    "On-time delivery rate (%)"
)


# ============================================================
# SYNTHETIC FIXTURE
# ============================================================

def build_logistics_benchmark_dataframe(
) -> pd.DataFrame:
    n = 168


    index = np.arange(
        n
    )


    expected = (
        180
        +
        (
            index
            %
            28
        )
        *
        3
    ).astype(
        int
    )


    late = (
        4
        +
        (
            index
            %
            9
        )
    ).astype(
        int
    )


    incomplete = (
        index
        %
        4
        ==
        0
    ).astype(
        int
    )


    completed = (
        expected
        -
        incomplete
    ).astype(
        int
    )


    transit_time = (
        8.0
        +
        (
            index
            %
            13
        )
        *
        1.5
    )


    distance = (
        120.0
        +
        (
            index
            %
            21
        )
        *
        18.0
    )


    package_weight = (
        25.0
        +
        (
            index
            %
            17
        )
        *
        2.75
    )


    freight_cost = (
        140.0
        +
        distance
        *
        0.65
        +
        package_weight
        *
        1.8
    )


    on_time_rate = (
        (
            completed
            -
            late
        )
        /
        completed
        *
        100.0
    )


    return pd.DataFrame(
        {
            EXPECTED_DELIVERIES:
                expected,

            COMPLETED_DELIVERIES:
                completed,

            LATE_DELIVERIES:
                late,

            TRANSIT_TIME:
                transit_time,

            FREIGHT_COST:
                freight_cost,

            PACKAGE_WEIGHT:
                package_weight,

            DISTANCE_TRAVELLED:
                distance,

            ON_TIME_DELIVERY_RATE:
                on_time_rate,
        }
    )


# ============================================================
# COLUMN CASES
#
# These expectations originated in Logistics Holdout #2.
#
# They are now regression knowledge because the holdout
# failures were inspected and used to develop Semantic
# System S2.
# ============================================================

def build_logistics_column_cases(
) -> list[
    SemanticColumnBenchmarkCase
]:
    return [
        SemanticColumnBenchmarkCase(
            case_id=
                "expected_deliveries",

            dataset_id=
                LOGISTICS_DATASET_ID,

            column=
                EXPECTED_DELIVERIES,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "expected",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "measure_kind",

                    accepted_values=[
                        "count",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "unit_kind",

                    accepted_values=[
                        "count",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "completed_deliveries",

            dataset_id=
                LOGISTICS_DATASET_ID,

            column=
                COMPLETED_DELIVERIES,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "completed",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "measure_kind",

                    accepted_values=[
                        "count",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "unit_kind",

                    accepted_values=[
                        "count",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "late_deliveries",

            dataset_id=
                LOGISTICS_DATASET_ID,

            column=
                LATE_DELIVERIES,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "late",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "measure_kind",

                    accepted_values=[
                        "count",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "unit_kind",

                    accepted_values=[
                        "count",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "transit_time",

            dataset_id=
                LOGISTICS_DATASET_ID,

            column=
                TRANSIT_TIME,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "measure_kind",

                    accepted_values=[
                        "duration",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "unit_kind",

                    accepted_values=[
                        "duration",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "freight_cost",

            dataset_id=
                LOGISTICS_DATASET_ID,

            column=
                FREIGHT_COST,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "measure_kind",

                    accepted_values=[
                        "currency",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "unit_kind",

                    accepted_values=[
                        "currency",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "on_time_delivery_rate",

            dataset_id=
                LOGISTICS_DATASET_ID,

            column=
                ON_TIME_DELIVERY_RATE,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "measure_kind",

                    accepted_values=[
                        "percentage",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "unit_kind",

                    accepted_values=[
                        "percent",
                    ],
                ),
            ],
        ),
    ]


# ============================================================
# PAIR CASES
# ============================================================

def build_logistics_pair_cases(
) -> list[
    SemanticPairBenchmarkCase
]:
    return [
        SemanticPairBenchmarkCase(
            case_id=
                "expected_completed_deliveries",

            left_dataset_id=
                LOGISTICS_DATASET_ID,

            right_dataset_id=
                LOGISTICS_DATASET_ID,

            left_column=
                EXPECTED_DELIVERIES,

            right_column=
                COMPLETED_DELIVERIES,

            same_concept_family=
                True,

            same_domain=
                True,

            distinct_variants=
                True,

            compatible_units=
                True,

            derived_gap_compatible=
                True,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "completed_late_deliveries",

            left_dataset_id=
                LOGISTICS_DATASET_ID,

            right_dataset_id=
                LOGISTICS_DATASET_ID,

            left_column=
                COMPLETED_DELIVERIES,

            right_column=
                LATE_DELIVERIES,

            same_concept_family=
                True,

            same_domain=
                True,

            distinct_variants=
                True,

            compatible_units=
                True,

            derived_gap_compatible=
                True,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "transit_freight",

            left_dataset_id=
                LOGISTICS_DATASET_ID,

            right_dataset_id=
                LOGISTICS_DATASET_ID,

            left_column=
                TRANSIT_TIME,

            right_column=
                FREIGHT_COST,

            compatible_units=
                False,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "freight_weight",

            left_dataset_id=
                LOGISTICS_DATASET_ID,

            right_dataset_id=
                LOGISTICS_DATASET_ID,

            left_column=
                FREIGHT_COST,

            right_column=
                PACKAGE_WEIGHT,

            compatible_units=
                False,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "distance_transit",

            left_dataset_id=
                LOGISTICS_DATASET_ID,

            right_dataset_id=
                LOGISTICS_DATASET_ID,

            left_column=
                DISTANCE_TRAVELLED,

            right_column=
                TRANSIT_TIME,

            compatible_units=
                False,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "rate_late_deliveries",

            left_dataset_id=
                LOGISTICS_DATASET_ID,

            right_dataset_id=
                LOGISTICS_DATASET_ID,

            left_column=
                ON_TIME_DELIVERY_RATE,

            right_column=
                LATE_DELIVERIES,

            compatible_units=
                False,

            derived_gap_compatible=
                False,
        ),
    ]


# ============================================================
# SUITE
# ============================================================

def build_logistics_semantic_benchmark(
) -> SemanticBenchmarkSuite:
    return SemanticBenchmarkSuite(
        benchmark_id=
            LOGISTICS_BENCHMARK_ID,

        name=
            "Logistics semantic regression",

        domain=
            "logistics",

        split=
            "regression",

        description=(
            "Regression benchmark derived from the second "
            "frozen out-of-domain Logistics holdout after "
            "its errors were inspected and used to improve "
            "Semantic System S2. It covers delivery states, "
            "counts, transit duration, freight cost, weight, "
            "distance and delivery performance."
        ),

        datasets=[
            BenchmarkDatasetSpec(
                dataset_id=
                    LOGISTICS_DATASET_ID,

                filename=
                    LOGISTICS_FILENAME,
            ),
        ],

        column_cases=
            build_logistics_column_cases(),

        pair_cases=
            build_logistics_pair_cases(),

        safety_critical_fields=[
            "compatible_units",
            "derived_gap_compatible",
        ],

        tags=[
            "logistics",
            "supply_chain",
            "deliveries",
            "transport",
            "regression",
            "former_holdout",
            "generalization_history",
            "dimensional_safety",
        ],

        benchmark_version=
            "logistics_semantic_regression_v0.1",
    )
