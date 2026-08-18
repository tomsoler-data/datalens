from __future__ import annotations

from app.evaluation.benchmarks.clinical_lab import (
    CLINICAL_LAB_DATASET_ID,
    CLINICAL_LAB_FILENAME,
    build_clinical_lab_column_cases,
    build_clinical_lab_pair_cases,
)

from app.evaluation.registry_schemas import (
    BenchmarkDatasetSpec,
    SemanticBenchmarkSuite,
)


# ============================================================
# IDENTIFIERS
# ============================================================

CLINICAL_LAB_DEVELOPMENT_BENCHMARK_ID = (
    "semantic:clinical_lab:development:v0.1"
)


CLINICAL_LAB_DEVELOPMENT_BENCHMARK_VERSION = (
    "clinical_lab_semantic_development_v0.1"
)


# ============================================================
# DEVELOPMENT SUITE
#
# The historical Clinical Lab Holdout #6 remains frozen in:
#
#     clinical_lab.py
#
# Its freeze artifact and first-run S4 snapshot must never be
# overwritten.
#
# Clinical Lab has now been inspected and can therefore be
# used as development data for S4.1.
# ============================================================

def build_clinical_lab_development_benchmark(
) -> SemanticBenchmarkSuite:
    return SemanticBenchmarkSuite(
        benchmark_id=
            CLINICAL_LAB_DEVELOPMENT_BENCHMARK_ID,

        name=
            "Clinical laboratory semantic development",

        domain=
            "clinical_lab_operations",

        split=
            "development",

        description=(
            "Development benchmark derived from the historical "
            "Clinical Lab Holdout #6 after its first independent "
            "DataLens S4 execution was frozen and inspected. "
            "The original holdout definition, freeze artifact "
            "and first-run result remain preserved separately."
        ),

        datasets=[
            BenchmarkDatasetSpec(
                dataset_id=
                    CLINICAL_LAB_DATASET_ID,

                filename=
                    CLINICAL_LAB_FILENAME,
            ),
        ],

        column_cases=
            build_clinical_lab_column_cases(),

        pair_cases=
            build_clinical_lab_pair_cases(),

        safety_critical_fields=[
            "compatible_units",
            "derived_gap_compatible",
        ],

        tags=[
            "clinical_lab",
            "development",
            "s4_1",
            "former_holdout",
            "quantity_family",
            "semantic_coverage",
            "same_dimension_different_quantity",
            "lexical_generalization",
            "semantic_generalization",
            "count",
            "currency",
            "duration",
            "percentage",
            "mass",
            "semantic_safety",
        ],

        benchmark_version=
            CLINICAL_LAB_DEVELOPMENT_BENCHMARK_VERSION,
    )
