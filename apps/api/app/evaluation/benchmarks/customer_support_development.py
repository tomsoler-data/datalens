from __future__ import annotations

from app.evaluation.benchmarks.customer_support import (
    CUSTOMER_SUPPORT_DATASET_ID,
    CUSTOMER_SUPPORT_FILENAME,
    build_customer_support_column_cases,
    build_customer_support_pair_cases,
)

from app.evaluation.registry_schemas import (
    BenchmarkDatasetSpec,
    SemanticBenchmarkSuite,
)


# ============================================================
# IDENTIFIERS
# ============================================================

CUSTOMER_SUPPORT_DEVELOPMENT_BENCHMARK_ID = (
    "semantic:customer_support:development:v0.1"
)


CUSTOMER_SUPPORT_DEVELOPMENT_BENCHMARK_VERSION = (
    "customer_support_semantic_development_v0.1"
)


# ============================================================
# DEVELOPMENT SUITE
#
# IMPORTANT
#
# The original Customer Support holdout remains frozen in:
#
#     customer_support.py
#
# and its historical S3 experiment snapshot must remain
# untouched.
#
# This suite reuses the same dataset and assertions because
# Customer Support has now been inspected and used during S4
# development. It can therefore no longer count as an
# independent holdout.
# ============================================================

def build_customer_support_development_benchmark(
) -> SemanticBenchmarkSuite:
    return SemanticBenchmarkSuite(
        benchmark_id=
            CUSTOMER_SUPPORT_DEVELOPMENT_BENCHMARK_ID,

        name=
            "Customer support semantic development",

        domain=
            "customer_support",

        split=
            "development",

        description=(
            "Development benchmark derived from the historical "
            "Customer Support Holdout #5 after that holdout was "
            "inspected and used to develop DataLens Semantic "
            "System S4. The historical holdout definition and "
            "snapshot remain frozen separately for provenance."
        ),

        datasets=[
            BenchmarkDatasetSpec(
                dataset_id=
                    CUSTOMER_SUPPORT_DATASET_ID,

                filename=
                    CUSTOMER_SUPPORT_FILENAME,
            ),
        ],

        column_cases=
            build_customer_support_column_cases(),

        pair_cases=
            build_customer_support_pair_cases(),

        safety_critical_fields=[
            "compatible_units",
            "derived_gap_compatible",
        ],

        tags=[
            "customer_support",
            "development",
            "s4",
            "lexical_generalization",
            "conceptual_generalization",
            "paraphrase",
            "shared_capabilities",
            "count",
            "currency",
            "duration",
            "percentage",
            "semantic_safety",
            "former_holdout",
        ],

        benchmark_version=
            CUSTOMER_SUPPORT_DEVELOPMENT_BENCHMARK_VERSION,
    )
