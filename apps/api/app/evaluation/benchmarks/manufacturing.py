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

MANUFACTURING_BENCHMARK_ID = (
    "semantic:manufacturing:regression:v0.1"
)


MANUFACTURING_DATASET_ID = (
    "manufacturing:0001"
)


MANUFACTURING_FILENAME = (
    "synthetic_manufacturing.csv"
)


# ============================================================
# COLUMN NAMES
# ============================================================

PLANNED_PRODUCTION = (
    "Planned production units"
)


ACTUAL_PRODUCTION = (
    "Actual production units"
)


REJECTED_UNITS = (
    "Rejected units"
)


DEFECT_RATE = (
    "Defect rate (%)"
)


DOWNTIME = (
    "Downtime (minutes)"
)


MAINTENANCE_COST = (
    "Maintenance cost"
)


ENERGY_CONSUMPTION = (
    "Energy consumption (kWh)"
)


# ============================================================
# SYNTHETIC FIXTURE
# ============================================================

def build_manufacturing_benchmark_dataframe(
) -> pd.DataFrame:
    n = 144


    index = np.arange(
        n
    )


    planned = (
        900
        +
        (
            index
            %
            24
        )
        *
        8
    ).astype(
        int
    )


    production_loss = (
        15
        +
        (
            index
            %
            7
        )
        *
        3
    )


    actual = (
        planned
        -
        production_loss
    ).astype(
        int
    )


    rejected = (
        8
        +
        (
            index
            %
            11
        )
    ).astype(
        int
    )


    defect_rate = (
        rejected
        /
        actual
        *
        100.0
    )


    downtime = (
        12
        +
        (
            index
            %
            9
        )
        *
        4
    ).astype(
        float
    )


    maintenance_cost = (
        1200.0
        +
        downtime
        *
        18.0
        +
        (
            index
            %
            5
        )
        *
        75.0
    )


    energy_consumption = (
        actual
        *
        2.4
        +
        downtime
        *
        3.0
    )


    return pd.DataFrame(
        {
            PLANNED_PRODUCTION:
                planned,

            ACTUAL_PRODUCTION:
                actual,

            REJECTED_UNITS:
                rejected,

            DEFECT_RATE:
                defect_rate,

            DOWNTIME:
                downtime,

            MAINTENANCE_COST:
                maintenance_cost,

            ENERGY_CONSUMPTION:
                energy_consumption,
        }
    )


# ============================================================
# COLUMN CASES
#
# These cases originated in Manufacturing Holdout #1.
# They are now regression knowledge because their failures
# were inspected and used to improve Semantic System S1.
# ============================================================

def build_manufacturing_column_cases(
) -> list[
    SemanticColumnBenchmarkCase
]:
    return [
        SemanticColumnBenchmarkCase(
            case_id=
                "planned_production",

            dataset_id=
                MANUFACTURING_DATASET_ID,

            column=
                PLANNED_PRODUCTION,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "planned",
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
                "actual_production",

            dataset_id=
                MANUFACTURING_DATASET_ID,

            column=
                ACTUAL_PRODUCTION,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "actual",
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
                "rejected_units",

            dataset_id=
                MANUFACTURING_DATASET_ID,

            column=
                REJECTED_UNITS,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "rejected",
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
                "defect_rate",

            dataset_id=
                MANUFACTURING_DATASET_ID,

            column=
                DEFECT_RATE,

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

        SemanticColumnBenchmarkCase(
            case_id=
                "downtime",

            dataset_id=
                MANUFACTURING_DATASET_ID,

            column=
                DOWNTIME,

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
                "maintenance_cost",

            dataset_id=
                MANUFACTURING_DATASET_ID,

            column=
                MAINTENANCE_COST,

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
    ]


# ============================================================
# PAIR CASES
# ============================================================

def build_manufacturing_pair_cases(
) -> list[
    SemanticPairBenchmarkCase
]:
    return [
        SemanticPairBenchmarkCase(
            case_id=
                "planned_actual_production",

            left_dataset_id=
                MANUFACTURING_DATASET_ID,

            right_dataset_id=
                MANUFACTURING_DATASET_ID,

            left_column=
                PLANNED_PRODUCTION,

            right_column=
                ACTUAL_PRODUCTION,

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
                "defect_rate_rejected_units",

            left_dataset_id=
                MANUFACTURING_DATASET_ID,

            right_dataset_id=
                MANUFACTURING_DATASET_ID,

            left_column=
                DEFECT_RATE,

            right_column=
                REJECTED_UNITS,

            compatible_units=
                False,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "downtime_maintenance_cost",

            left_dataset_id=
                MANUFACTURING_DATASET_ID,

            right_dataset_id=
                MANUFACTURING_DATASET_ID,

            left_column=
                DOWNTIME,

            right_column=
                MAINTENANCE_COST,

            compatible_units=
                False,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "maintenance_energy",

            left_dataset_id=
                MANUFACTURING_DATASET_ID,

            right_dataset_id=
                MANUFACTURING_DATASET_ID,

            left_column=
                MAINTENANCE_COST,

            right_column=
                ENERGY_CONSUMPTION,

            compatible_units=
                False,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "production_defect_rate",

            left_dataset_id=
                MANUFACTURING_DATASET_ID,

            right_dataset_id=
                MANUFACTURING_DATASET_ID,

            left_column=
                ACTUAL_PRODUCTION,

            right_column=
                DEFECT_RATE,

            compatible_units=
                False,

            derived_gap_compatible=
                False,
        ),
    ]


# ============================================================
# SUITE
# ============================================================

def build_manufacturing_semantic_benchmark(
) -> SemanticBenchmarkSuite:
    return SemanticBenchmarkSuite(
        benchmark_id=
            MANUFACTURING_BENCHMARK_ID,

        name=
            "Manufacturing semantic regression",

        domain=
            "manufacturing",

        split=
            "regression",

        description=(
            "Regression benchmark derived from the first "
            "frozen Manufacturing holdout after its errors "
            "were inspected and used to improve Semantic "
            "System S1. It covers production planning, "
            "actual output, rejected units, defect rate, "
            "downtime, maintenance cost and energy use."
        ),

        datasets=[
            BenchmarkDatasetSpec(
                dataset_id=
                    MANUFACTURING_DATASET_ID,

                filename=
                    MANUFACTURING_FILENAME,
            ),
        ],

        column_cases=
            build_manufacturing_column_cases(),

        pair_cases=
            build_manufacturing_pair_cases(),

        safety_critical_fields=[
            "compatible_units",
            "derived_gap_compatible",
        ],

        tags=[
            "manufacturing",
            "production",
            "quality",
            "maintenance",
            "energy",
            "regression",
            "former_holdout",
            "generalization_history",
        ],

        benchmark_version=
            "manufacturing_semantic_regression_v0.1",
    )
