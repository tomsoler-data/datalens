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

ECOMMERCE_BENCHMARK_ID = (
    "semantic:ecommerce:regression:v0.1"
)


ECOMMERCE_DATASET_ID = (
    "commerce:0001"
)


ECOMMERCE_FILENAME = (
    "synthetic_commerce.csv"
)


# ============================================================
# SYNTHETIC FIXTURE
# ============================================================

def build_ecommerce_benchmark_dataframe(
) -> pd.DataFrame:
    n = 120


    retail_price = np.linspace(
        25.0,
        95.0,
        n,
    )


    wholesale_price = (
        retail_price
        *
        0.62
    )


    units_ordered = np.tile(
        np.arange(
            5,
            25,
        ),
        6,
    )


    units_shipped = (
        units_ordered
        -
        (
            np.arange(
                n
            )
            %
            4
            ==
            0
        ).astype(
            int
        )
    )


    website_sessions = np.linspace(
        1000,
        9000,
        n,
    ).astype(
        int
    )


    customer_tenure = np.tile(
        np.arange(
            1,
            61,
        ),
        2,
    )


    sales_amount = (
        retail_price
        *
        units_shipped
    )


    return pd.DataFrame(
        {
            "Retail price":
                retail_price,

            "Wholesale price":
                wholesale_price,

            "Units ordered":
                units_ordered,

            "Units shipped":
                units_shipped,

            "Website sessions":
                website_sessions,

            "Customer tenure (months)":
                customer_tenure,

            "Sales amount":
                sales_amount,
        }
    )


# ============================================================
# COLUMN CASES
# ============================================================

def build_ecommerce_column_cases(
) -> list[
    SemanticColumnBenchmarkCase
]:
    return [
        SemanticColumnBenchmarkCase(
            case_id=
                "retail_price",

            dataset_id=
                ECOMMERCE_DATASET_ID,

            column=
                "Retail price",

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "semantic_group",

                    accepted_values=[
                        "price",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "retail",
                    ],
                ),
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
                "wholesale_price",

            dataset_id=
                ECOMMERCE_DATASET_ID,

            column=
                "Wholesale price",

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "semantic_group",

                    accepted_values=[
                        "price",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "wholesale",
                    ],
                ),
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
                "units_ordered",

            dataset_id=
                ECOMMERCE_DATASET_ID,

            column=
                "Units ordered",

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "semantic_group",

                    accepted_values=[
                        "units",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "ordered",
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
                "units_shipped",

            dataset_id=
                ECOMMERCE_DATASET_ID,

            column=
                "Units shipped",

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "semantic_group",

                    accepted_values=[
                        "units",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "shipped",
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
                "website_sessions",

            dataset_id=
                ECOMMERCE_DATASET_ID,

            column=
                "Website sessions",

            expectations=[
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
                "customer_tenure",

            dataset_id=
                ECOMMERCE_DATASET_ID,

            column=
                "Customer tenure (months)",

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
                "sales_amount",

            dataset_id=
                ECOMMERCE_DATASET_ID,

            column=
                "Sales amount",

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

def build_ecommerce_pair_cases(
) -> list[
    SemanticPairBenchmarkCase
]:
    return [
        SemanticPairBenchmarkCase(
            case_id=
                "retail_wholesale",

            left_dataset_id=
                ECOMMERCE_DATASET_ID,

            right_dataset_id=
                ECOMMERCE_DATASET_ID,

            left_column=
                "Retail price",

            right_column=
                "Wholesale price",

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
                "ordered_shipped",

            left_dataset_id=
                ECOMMERCE_DATASET_ID,

            right_dataset_id=
                ECOMMERCE_DATASET_ID,

            left_column=
                "Units ordered",

            right_column=
                "Units shipped",

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
                "tenure_sales",

            left_dataset_id=
                ECOMMERCE_DATASET_ID,

            right_dataset_id=
                ECOMMERCE_DATASET_ID,

            left_column=
                "Customer tenure (months)",

            right_column=
                "Sales amount",

            same_concept_family=
                False,

            same_domain=
                False,

            compatible_units=
                False,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "sessions_sales",

            left_dataset_id=
                ECOMMERCE_DATASET_ID,

            right_dataset_id=
                ECOMMERCE_DATASET_ID,

            left_column=
                "Website sessions",

            right_column=
                "Sales amount",

            same_concept_family=
                False,

            same_domain=
                False,

            compatible_units=
                False,

            derived_gap_compatible=
                False,
        ),
    ]


# ============================================================
# SUITE
# ============================================================

def build_ecommerce_semantic_benchmark(
) -> SemanticBenchmarkSuite:
    return SemanticBenchmarkSuite(
        benchmark_id=
            ECOMMERCE_BENCHMARK_ID,

        name=
            "E-commerce semantic regression",

        domain=
            "commerce",

        split=
            "regression",

        description=(
            "Regression benchmark for semantic column "
            "normalization, semantic pair comparison and "
            "derived-gap compatibility in an e-commerce "
            "dataset."
        ),

        datasets=[
            BenchmarkDatasetSpec(
                dataset_id=
                    ECOMMERCE_DATASET_ID,

                filename=
                    ECOMMERCE_FILENAME,
            ),
        ],

        column_cases=
            build_ecommerce_column_cases(),

        pair_cases=
            build_ecommerce_pair_cases(),

        safety_critical_fields=[
            "compatible_units",
            "derived_gap_compatible",
        ],

        tags=[
            "commerce",
            "semantic_normalization",
            "derived_gap",
            "units",
            "variants",
        ],

        benchmark_version=
            "ecommerce_semantic_regression_v0.1",
    )
