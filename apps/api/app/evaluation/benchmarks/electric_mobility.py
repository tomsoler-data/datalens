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

ELECTRIC_MOBILITY_BENCHMARK_ID = (
    "semantic:electric_mobility:regression:v0.1"
)


ELECTRIC_MOBILITY_DATASET_ID = (
    "electric_mobility:0001"
)


ELECTRIC_MOBILITY_FILENAME = (
    "synthetic_electric_mobility.csv"
)


# ============================================================
# COLUMN NAMES
# ============================================================

TARGET_ROUTE_DISTANCE = (
    "Target route distance (km)"
)


ACTUAL_ROUTE_DISTANCE = (
    "Actual route distance (km)"
)


PLANNED_CARGO_MASS = (
    "Planned cargo mass (tonne)"
)


ACTUAL_CARGO_MASS = (
    "Actual cargo mass (kg)"
)


PLANNED_ENERGY_CONSUMPTION = (
    "Planned energy consumption (MWh)"
)


ACTUAL_ENERGY_CONSUMPTION = (
    "Actual energy consumption (kWh)"
)


CHARGING_TIME = (
    "Charging time (minutes)"
)


CHARGING_COST = (
    "Charging cost"
)


BATTERY_STATE_OF_CHARGE = (
    "Battery state of charge (%)"
)


# ============================================================
# SYNTHETIC FIXTURE
# ============================================================

def build_electric_mobility_benchmark_dataframe(
) -> pd.DataFrame:
    n = 216


    index = np.arange(
        n
    )


    target_distance = (
        120.0
        +
        (
            index
            %
            24
        )
        *
        7.5
    )


    distance_variation = (
        (
            index
            %
            9
        )
        -
        4
    ) * 1.8


    actual_distance = (
        target_distance
        +
        distance_variation
    )


    planned_cargo_tonne = (
        0.8
        +
        (
            index
            %
            18
        )
        *
        0.045
    )


    cargo_variation_kg = (
        (
            index
            %
            11
        )
        -
        5
    ) * 3.0


    actual_cargo_kg = (
        planned_cargo_tonne
        *
        1000.0
        +
        cargo_variation_kg
    )


    planned_energy_mwh = (
        0.06
        +
        target_distance
        *
        0.00042
        +
        planned_cargo_tonne
        *
        0.012
    )


    actual_energy_kwh = (
        planned_energy_mwh
        *
        1000.0
        *
        (
            0.96
            +
            (
                index
                %
                9
            )
            *
            0.01
        )
    )


    charging_time = (
        18.0
        +
        (
            index
            %
            20
        )
        *
        2.5
    )


    charging_cost = (
        4.0
        +
        actual_energy_kwh
        *
        0.21
    )


    battery_state_of_charge = (
        42.0
        +
        (
            index
            %
            22
        )
        *
        2.1
    )


    battery_state_of_charge = (
        np.minimum(
            battery_state_of_charge,
            96.0,
        )
    )


    return pd.DataFrame(
        {
            TARGET_ROUTE_DISTANCE:
                target_distance,

            ACTUAL_ROUTE_DISTANCE:
                actual_distance,

            PLANNED_CARGO_MASS:
                planned_cargo_tonne,

            ACTUAL_CARGO_MASS:
                actual_cargo_kg,

            PLANNED_ENERGY_CONSUMPTION:
                planned_energy_mwh,

            ACTUAL_ENERGY_CONSUMPTION:
                actual_energy_kwh,

            CHARGING_TIME:
                charging_time,

            CHARGING_COST:
                charging_cost,

            BATTERY_STATE_OF_CHARGE:
                battery_state_of_charge,
        }
    )


# ============================================================
# COLUMN CASES
#
# HISTORY
#
# These expectations were originally defined BEFORE the first
# DataLens S3 execution on Electric Mobility.
#
# S3 achieved:
#
#     51 / 51
#
# on that independent holdout without any semantic adaptation.
#
# The benchmark is now promoted to regression so that this
# independently demonstrated capability remains protected.
#
# The historical evidence remains stored separately in:
#
#     semantic_s3_electric_mobility_holdout.json
# ============================================================

def build_electric_mobility_column_cases(
) -> list[
    SemanticColumnBenchmarkCase
]:
    return [
        SemanticColumnBenchmarkCase(
            case_id=
                "target_route_distance",

            dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            column=
                TARGET_ROUTE_DISTANCE,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "target",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "quantity_dimension",

                    accepted_values=[
                        "distance",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "quantity_unit",

                    accepted_values=[
                        "kilometre",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "actual_route_distance",

            dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            column=
                ACTUAL_ROUTE_DISTANCE,

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
                        "quantity_dimension",

                    accepted_values=[
                        "distance",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "quantity_unit",

                    accepted_values=[
                        "kilometre",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "planned_cargo_mass",

            dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            column=
                PLANNED_CARGO_MASS,

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
                        "quantity_dimension",

                    accepted_values=[
                        "mass",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "quantity_unit",

                    accepted_values=[
                        "tonne",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "actual_cargo_mass",

            dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            column=
                ACTUAL_CARGO_MASS,

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
                        "quantity_dimension",

                    accepted_values=[
                        "mass",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "quantity_unit",

                    accepted_values=[
                        "kilogram",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "planned_energy_consumption",

            dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            column=
                PLANNED_ENERGY_CONSUMPTION,

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
                        "quantity_dimension",

                    accepted_values=[
                        "energy",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "quantity_unit",

                    accepted_values=[
                        "megawatt_hour",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "actual_energy_consumption",

            dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            column=
                ACTUAL_ENERGY_CONSUMPTION,

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
                        "quantity_dimension",

                    accepted_values=[
                        "energy",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "quantity_unit",

                    accepted_values=[
                        "kilowatt_hour",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "charging_time",

            dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            column=
                CHARGING_TIME,

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

                SemanticFieldExpectation(
                    field=
                        "quantity_dimension",

                    accepted_values=[
                        "duration",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "quantity_unit",

                    accepted_values=[
                        "minute",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "charging_cost",

            dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            column=
                CHARGING_COST,

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

                SemanticFieldExpectation(
                    field=
                        "quantity_dimension",

                    accepted_values=[
                        "currency",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "quantity_unit",

                    accepted_values=[
                        "currency",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "battery_state_of_charge",

            dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            column=
                BATTERY_STATE_OF_CHARGE,

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

                SemanticFieldExpectation(
                    field=
                        "quantity_dimension",

                    accepted_values=[
                        "proportion",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "quantity_unit",

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

def build_electric_mobility_pair_cases(
) -> list[
    SemanticPairBenchmarkCase
]:
    return [
        # ----------------------------------------------------
        # Same dimension + same unit.
        #
        # Direct subtraction is valid.
        # ----------------------------------------------------

        SemanticPairBenchmarkCase(
            case_id=
                "target_actual_route_distance",

            left_dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            right_dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            left_column=
                TARGET_ROUTE_DISTANCE,

            right_column=
                ACTUAL_ROUTE_DISTANCE,

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

        # ----------------------------------------------------
        # Same dimension + different units.
        #
        # Dimensionally compatible, but direct subtraction
        # remains blocked until DataLens has an explicit
        # conversion engine.
        # ----------------------------------------------------

        SemanticPairBenchmarkCase(
            case_id=
                "planned_actual_cargo_mass",

            left_dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            right_dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            left_column=
                PLANNED_CARGO_MASS,

            right_column=
                ACTUAL_CARGO_MASS,

            same_concept_family=
                True,

            same_domain=
                True,

            distinct_variants=
                True,

            compatible_units=
                True,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "planned_actual_energy",

            left_dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            right_dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            left_column=
                PLANNED_ENERGY_CONSUMPTION,

            right_column=
                ACTUAL_ENERGY_CONSUMPTION,

            same_concept_family=
                True,

            same_domain=
                True,

            distinct_variants=
                True,

            compatible_units=
                True,

            derived_gap_compatible=
                False,
        ),

        # ----------------------------------------------------
        # Incompatible dimensions.
        # ----------------------------------------------------

        SemanticPairBenchmarkCase(
            case_id=
                "distance_energy",

            left_dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            right_dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            left_column=
                ACTUAL_ROUTE_DISTANCE,

            right_column=
                ACTUAL_ENERGY_CONSUMPTION,

            compatible_units=
                False,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "cargo_cost",

            left_dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            right_dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            left_column=
                ACTUAL_CARGO_MASS,

            right_column=
                CHARGING_COST,

            compatible_units=
                False,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "charging_time_battery_soc",

            left_dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            right_dataset_id=
                ELECTRIC_MOBILITY_DATASET_ID,

            left_column=
                CHARGING_TIME,

            right_column=
                BATTERY_STATE_OF_CHARGE,

            compatible_units=
                False,

            derived_gap_compatible=
                False,
        ),
    ]


# ============================================================
# SUITE
# ============================================================

def build_electric_mobility_semantic_benchmark(
) -> SemanticBenchmarkSuite:
    return SemanticBenchmarkSuite(
        benchmark_id=
            ELECTRIC_MOBILITY_BENCHMARK_ID,

        name=
            "Electric mobility quantity semantic regression",

        domain=
            "electric_mobility",

        split=
            "regression",

        description=(
            "Regression benchmark promoted from the fourth "
            "frozen out-of-domain holdout. DataLens Semantic "
            "System S3 achieved 51/51 before promotion, "
            "demonstrating independent generalization of "
            "Quantity Semantics v0.1 to distance, mass, "
            "energy, duration, currency, proportions and "
            "mixed-unit compatibility."
        ),

        datasets=[
            BenchmarkDatasetSpec(
                dataset_id=
                    ELECTRIC_MOBILITY_DATASET_ID,

                filename=
                    ELECTRIC_MOBILITY_FILENAME,
            ),
        ],

        column_cases=
            build_electric_mobility_column_cases(),

        pair_cases=
            build_electric_mobility_pair_cases(),

        safety_critical_fields=[
            "compatible_units",
            "derived_gap_compatible",
        ],

        tags=[
            "electric_mobility",
            "distance",
            "mass",
            "energy",
            "duration",
            "currency",
            "quantity_semantics",
            "unit_conversion_boundary",
            "regression",
            "former_holdout",
            "independent_generalization_success",
            "dimensional_safety",
        ],

        benchmark_version=
            "electric_mobility_semantic_regression_v0.1",
    )
