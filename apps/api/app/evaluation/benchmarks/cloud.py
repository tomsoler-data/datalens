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

CLOUD_BENCHMARK_ID = (
    "semantic:cloud:regression:v0.1"
)


CLOUD_DATASET_ID = (
    "cloud:0001"
)


CLOUD_FILENAME = (
    "synthetic_cloud_infrastructure.csv"
)


# ============================================================
# COLUMN NAMES
# ============================================================

REQUESTED_CPU_CORES = (
    "Requested CPU cores"
)


ALLOCATED_CPU_CORES = (
    "Allocated CPU cores"
)


FAILED_JOBS = (
    "Failed jobs"
)


QUEUE_TIME = (
    "Queue time (seconds)"
)


COMPUTE_COST = (
    "Compute cost"
)


CPU_UTILIZATION = (
    "CPU utilization (%)"
)


PROVISIONED_STORAGE = (
    "Provisioned storage (GB)"
)


USED_STORAGE = (
    "Used storage (GB)"
)


# ============================================================
# SYNTHETIC FIXTURE
# ============================================================

def build_cloud_benchmark_dataframe(
) -> pd.DataFrame:
    n = 192


    index = np.arange(
        n
    )


    requested_cores = (
        32
        +
        (
            index
            %
            16
        )
        *
        4
    ).astype(
        int
    )


    allocation_gap = (
        index
        %
        5
    ).astype(
        int
    )


    allocated_cores = (
        requested_cores
        -
        allocation_gap
    ).astype(
        int
    )


    failed_jobs = (
        index
        %
        7
    ).astype(
        int
    )


    queue_time = (
        8.0
        +
        (
            index
            %
            15
        )
        *
        2.5
    )


    cpu_utilization = (
        48.0
        +
        (
            index
            %
            19
        )
        *
        2.0
    )


    cpu_utilization = np.minimum(
        cpu_utilization,
        94.0,
    )


    provisioned_storage = (
        500.0
        +
        (
            index
            %
            24
        )
        *
        25.0
    )


    used_storage = (
        provisioned_storage
        *
        (
            0.55
            +
            (
                index
                %
                10
            )
            *
            0.025
        )
    )


    compute_cost = (
        80.0
        +
        allocated_cores
        *
        3.2
        +
        used_storage
        *
        0.08
    )


    return pd.DataFrame(
        {
            REQUESTED_CPU_CORES:
                requested_cores,

            ALLOCATED_CPU_CORES:
                allocated_cores,

            FAILED_JOBS:
                failed_jobs,

            QUEUE_TIME:
                queue_time,

            COMPUTE_COST:
                compute_cost,

            CPU_UTILIZATION:
                cpu_utilization,

            PROVISIONED_STORAGE:
                provisioned_storage,

            USED_STORAGE:
                used_storage,
        }
    )


# ============================================================
# COLUMN CASES
#
# These expectations originated in Cloud Holdout #3.
#
# Cloud is now regression knowledge because its first frozen
# result was inspected and used to develop:
#
# - Quantity Semantics v0.1
# - Semantic Normalizer v0.4
# - Semantic Profile Comparator v0.3
# ============================================================

def build_cloud_column_cases(
) -> list[
    SemanticColumnBenchmarkCase
]:
    return [
        SemanticColumnBenchmarkCase(
            case_id=
                "requested_cpu_cores",

            dataset_id=
                CLOUD_DATASET_ID,

            column=
                REQUESTED_CPU_CORES,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "requested",
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
                "allocated_cpu_cores",

            dataset_id=
                CLOUD_DATASET_ID,

            column=
                ALLOCATED_CPU_CORES,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "allocated",
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
                "failed_jobs",

            dataset_id=
                CLOUD_DATASET_ID,

            column=
                FAILED_JOBS,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "failed",
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
                "queue_time",

            dataset_id=
                CLOUD_DATASET_ID,

            column=
                QUEUE_TIME,

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
                "compute_cost",

            dataset_id=
                CLOUD_DATASET_ID,

            column=
                COMPUTE_COST,

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
                "cpu_utilization",

            dataset_id=
                CLOUD_DATASET_ID,

            column=
                CPU_UTILIZATION,

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
                "provisioned_storage",

            dataset_id=
                CLOUD_DATASET_ID,

            column=
                PROVISIONED_STORAGE,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "provisioned",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "used_storage",

            dataset_id=
                CLOUD_DATASET_ID,

            column=
                USED_STORAGE,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "used",
                    ],
                ),
            ],
        ),
    ]


# ============================================================
# PAIR CASES
# ============================================================

def build_cloud_pair_cases(
) -> list[
    SemanticPairBenchmarkCase
]:
    return [
        SemanticPairBenchmarkCase(
            case_id=
                "requested_allocated_cpu",

            left_dataset_id=
                CLOUD_DATASET_ID,

            right_dataset_id=
                CLOUD_DATASET_ID,

            left_column=
                REQUESTED_CPU_CORES,

            right_column=
                ALLOCATED_CPU_CORES,

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
                "provisioned_used_storage",

            left_dataset_id=
                CLOUD_DATASET_ID,

            right_dataset_id=
                CLOUD_DATASET_ID,

            left_column=
                PROVISIONED_STORAGE,

            right_column=
                USED_STORAGE,

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
                "failed_jobs_cpu_utilization",

            left_dataset_id=
                CLOUD_DATASET_ID,

            right_dataset_id=
                CLOUD_DATASET_ID,

            left_column=
                FAILED_JOBS,

            right_column=
                CPU_UTILIZATION,

            compatible_units=
                False,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "queue_time_compute_cost",

            left_dataset_id=
                CLOUD_DATASET_ID,

            right_dataset_id=
                CLOUD_DATASET_ID,

            left_column=
                QUEUE_TIME,

            right_column=
                COMPUTE_COST,

            compatible_units=
                False,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "storage_compute_cost",

            left_dataset_id=
                CLOUD_DATASET_ID,

            right_dataset_id=
                CLOUD_DATASET_ID,

            left_column=
                USED_STORAGE,

            right_column=
                COMPUTE_COST,

            compatible_units=
                False,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "allocated_cpu_utilization",

            left_dataset_id=
                CLOUD_DATASET_ID,

            right_dataset_id=
                CLOUD_DATASET_ID,

            left_column=
                ALLOCATED_CPU_CORES,

            right_column=
                CPU_UTILIZATION,

            compatible_units=
                False,

            derived_gap_compatible=
                False,
        ),
    ]


# ============================================================
# SUITE
# ============================================================

def build_cloud_semantic_benchmark(
) -> SemanticBenchmarkSuite:
    return SemanticBenchmarkSuite(
        benchmark_id=
            CLOUD_BENCHMARK_ID,

        name=
            "Cloud infrastructure semantic regression",

        domain=
            "cloud_infrastructure",

        split=
            "regression",

        description=(
            "Regression benchmark derived from the third "
            "frozen out-of-domain Cloud Infrastructure "
            "holdout after its errors were inspected and "
            "used to develop Quantity Semantics v0.1."
        ),

        datasets=[
            BenchmarkDatasetSpec(
                dataset_id=
                    CLOUD_DATASET_ID,

                filename=
                    CLOUD_FILENAME,
            ),
        ],

        column_cases=
            build_cloud_column_cases(),

        pair_cases=
            build_cloud_pair_cases(),

        safety_critical_fields=[
            "compatible_units",
            "derived_gap_compatible",
        ],

        tags=[
            "cloud",
            "infrastructure",
            "compute",
            "storage",
            "capacity",
            "regression",
            "former_holdout",
            "generalization_history",
            "quantity_semantics",
            "dimensional_safety",
        ],

        benchmark_version=
            "cloud_semantic_regression_v0.1",
    )
