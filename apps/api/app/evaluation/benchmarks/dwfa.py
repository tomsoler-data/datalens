from __future__ import annotations

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

DWFA_BENCHMARK_ID = (
    "semantic:dwfa:regression:v0.1"
)


WATER_DATASET_ID = (
    "dataset:0001"
)

MORTALITY_DATASET_ID = (
    "dataset:0002"
)

POLITICAL_DATASET_ID = (
    "dataset:0003"
)

POPULATION_DATASET_ID = (
    "dataset:0004"
)

REGION_DATASET_ID = (
    "dataset:0005"
)


# ============================================================
# FILENAMES
# ============================================================

WATER_FILENAME = (
    "BasicAndSafelyManagedDrinkingWaterServices.csv"
)

MORTALITY_FILENAME = (
    "MortalityRateAttributedToWater.csv"
)

POLITICAL_FILENAME = (
    "PoliticalStability.csv"
)

POPULATION_FILENAME = (
    "Population.csv"
)

REGION_FILENAME = (
    "RegionCountry.csv"
)


# ============================================================
# COLUMN CASES
# ============================================================

def build_dwfa_column_cases(
) -> list[
    SemanticColumnBenchmarkCase
]:
    return [
        SemanticColumnBenchmarkCase(
            case_id=
                "basic_water",

            dataset_id=
                WATER_DATASET_ID,

            column=(
                "Population using at least basic "
                "drinking-water services (%)"
            ),

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "concept",

                    accepted_values=[
                        "drinking_water_access",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "domain",

                    accepted_values=[
                        "water_access",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "semantic_group",

                    accepted_values=[
                        "population_drinking_water_services",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "basic",
                    ],
                ),
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
                "safe_water",

            dataset_id=
                WATER_DATASET_ID,

            column=(
                "Population using safely managed "
                "drinking-water services (%)"
            ),

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "concept",

                    accepted_values=[
                        "drinking_water_access",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "domain",

                    accepted_values=[
                        "water_access",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "semantic_group",

                    accepted_values=[
                        "population_drinking_water_services",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "safely_managed",
                    ],
                ),
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
                "mortality_rate",

            dataset_id=
                MORTALITY_DATASET_ID,

            column=(
                "Mortality rate attributed to exposure "
                "to unsafe WASH services"
            ),

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "domain",

                    accepted_values=[
                        "health",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "measure_kind",

                    accepted_values=[
                        "rate",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "unit_kind",

                    accepted_values=[
                        "rate",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "wash_deaths",

            dataset_id=
                MORTALITY_DATASET_ID,

            column=
                "WASH deaths",

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "domain",

                    accepted_values=[
                        "health",
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
                "political_stability",

            dataset_id=
                POLITICAL_DATASET_ID,

            column=
                "Political_Stability",

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "concept",

                    accepted_values=[
                        "political_stability",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "domain",

                    accepted_values=[
                        "governance",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "semantic_group",

                    accepted_values=[
                        "political_stability",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "population",

            dataset_id=
                POPULATION_DATASET_ID,

            column=
                "Population",

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "concept",

                    accepted_values=[
                        "population",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "domain",

                    accepted_values=[
                        "demography",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "semantic_group",

                    accepted_values=[
                        "population_count",
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
                "region_display",

            dataset_id=
                REGION_DATASET_ID,

            column=
                "REGION (DISPLAY)",

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "concept",

                    accepted_values=[
                        "region",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "domain",

                    accepted_values=[
                        "geography",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "semantic_group",

                    accepted_values=[
                        "geography",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "measure_kind",

                    accepted_values=[
                        "category",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "unit_kind",

                    accepted_values=[
                        "category",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "country_display",

            dataset_id=
                REGION_DATASET_ID,

            column=
                "COUNTRY (DISPLAY)",

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "concept",

                    accepted_values=[
                        "country",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "domain",

                    accepted_values=[
                        "geography",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "semantic_group",

                    accepted_values=[
                        "geography",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "measure_kind",

                    accepted_values=[
                        "category",
                    ],
                ),
                SemanticFieldExpectation(
                    field=
                        "unit_kind",

                    accepted_values=[
                        "category",
                    ],
                ),
            ],
        ),
    ]


# ============================================================
# PAIR CASES
# ============================================================

def build_dwfa_pair_cases(
) -> list[
    SemanticPairBenchmarkCase
]:
    basic = (
        "Population using at least basic "
        "drinking-water services (%)"
    )


    safe = (
        "Population using safely managed "
        "drinking-water services (%)"
    )


    mortality = (
        "Mortality rate attributed to exposure "
        "to unsafe WASH services"
    )


    return [
        SemanticPairBenchmarkCase(
            case_id=
                "basic_safe_water",

            left_dataset_id=
                WATER_DATASET_ID,

            right_dataset_id=
                WATER_DATASET_ID,

            left_column=
                basic,

            right_column=
                safe,

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
                "mortality_deaths",

            left_dataset_id=
                MORTALITY_DATASET_ID,

            right_dataset_id=
                MORTALITY_DATASET_ID,

            left_column=
                mortality,

            right_column=
                "WASH deaths",

            same_concept_family=
                True,

            same_domain=
                True,

            distinct_variants=
                True,

            compatible_units=
                False,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "basic_political",

            left_dataset_id=
                WATER_DATASET_ID,

            right_dataset_id=
                POLITICAL_DATASET_ID,

            left_column=
                basic,

            right_column=
                "Political_Stability",

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
                "safe_political",

            left_dataset_id=
                WATER_DATASET_ID,

            right_dataset_id=
                POLITICAL_DATASET_ID,

            left_column=
                safe,

            right_column=
                "Political_Stability",

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
                "political_population",

            left_dataset_id=
                POLITICAL_DATASET_ID,

            right_dataset_id=
                POPULATION_DATASET_ID,

            left_column=
                "Political_Stability",

            right_column=
                "Population",

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
                "mortality_basic",

            left_dataset_id=
                MORTALITY_DATASET_ID,

            right_dataset_id=
                WATER_DATASET_ID,

            left_column=
                mortality,

            right_column=
                basic,

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

def build_dwfa_semantic_benchmark(
) -> SemanticBenchmarkSuite:
    return SemanticBenchmarkSuite(
        benchmark_id=
            DWFA_BENCHMARK_ID,

        name=
            "DWFA semantic regression",

        domain=
            "public_health_water",

        split=
            "regression",

        description=(
            "Multi-dataset semantic regression benchmark "
            "covering water access, mortality, governance, "
            "population and geographic reference data."
        ),

        datasets=[
            BenchmarkDatasetSpec(
                dataset_id=
                    WATER_DATASET_ID,

                filename=
                    WATER_FILENAME,
            ),
            BenchmarkDatasetSpec(
                dataset_id=
                    MORTALITY_DATASET_ID,

                filename=
                    MORTALITY_FILENAME,
            ),
            BenchmarkDatasetSpec(
                dataset_id=
                    POLITICAL_DATASET_ID,

                filename=
                    POLITICAL_FILENAME,
            ),
            BenchmarkDatasetSpec(
                dataset_id=
                    POPULATION_DATASET_ID,

                filename=
                    POPULATION_FILENAME,
            ),
            BenchmarkDatasetSpec(
                dataset_id=
                    REGION_DATASET_ID,

                filename=
                    REGION_FILENAME,
            ),
        ],

        column_cases=
            build_dwfa_column_cases(),

        pair_cases=
            build_dwfa_pair_cases(),

        safety_critical_fields=[
            "compatible_units",
            "derived_gap_compatible",
        ],

        tags=[
            "water",
            "health",
            "governance",
            "population",
            "geography",
            "multi_dataset",
            "derived_gap",
            "cross_dataset",
        ],

        benchmark_version=
            "dwfa_semantic_regression_v0.1",
    )
