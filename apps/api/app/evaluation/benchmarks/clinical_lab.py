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

CLINICAL_LAB_BENCHMARK_ID = (
    "semantic:clinical_lab:holdout:v0.1"
)


CLINICAL_LAB_DATASET_ID = (
    "clinical_lab:0001"
)


CLINICAL_LAB_FILENAME = (
    "synthetic_clinical_lab_operations.csv"
)


CLINICAL_LAB_BENCHMARK_VERSION = (
    "clinical_lab_semantic_holdout_v0.1"
)


# ============================================================
# COLUMNS
# ============================================================

SPECIMENS_RECEIVED = (
    "Specimens received"
)


SAMPLES_PROCESSED = (
    "Samples processed"
)


TECHNICIANS_SCHEDULED = (
    "Technicians scheduled"
)


LAB_STAFF_ON_DUTY = (
    "Lab staff on duty"
)


ESTIMATED_TEST_CHARGE = (
    "Estimated test charge"
)


FINAL_LABORATORY_BILL = (
    "Final laboratory bill"
)


TARGET_RESULT_TURNAROUND = (
    "Target result turnaround (minutes)"
)


ACTUAL_REPORTING_TIME = (
    "Actual reporting time (minutes)"
)


ANALYZER_UNITS_ONLINE = (
    "Analyzer units online"
)


REAGENT_SPEND = (
    "Reagent spend"
)


QUALITY_PASS_RATE = (
    "Quality pass rate (%)"
)


SLA_COMPLIANCE = (
    "SLA compliance (%)"
)


PLANNED_SAMPLE_MASS = (
    "Planned sample mass (grams)"
)


MEASURED_SPECIMEN_WEIGHT = (
    "Measured specimen weight (kg)"
)


ANALYZER_DOWNTIME = (
    "Analyzer downtime (minutes)"
)


REAGENT_MASS_USED = (
    "Reagent mass used (kg)"
)


# ============================================================
# SYNTHETIC HOLDOUT DATA
#
# IMPORTANT
#
# The values are deterministic.
#
# No random sampling is used so the first S4 holdout
# execution can be reproduced exactly.
#
# Several variables deliberately share the same mathematical
# dimension while representing different business quantities.
#
# This directly tests:
#
#     SameDimension
#         does NOT imply
#     SameQuantityFamily
# ============================================================

def build_clinical_lab_benchmark_dataframe(
) -> pd.DataFrame:
    n = 240


    index = np.arange(
        n
    )


    # --------------------------------------------------------
    # Laboratory specimen flow
    # --------------------------------------------------------

    specimens_received = (
        90
        +
        (
            index
            %
            41
        )
    ).astype(
        int
    )


    samples_processed = (
        specimens_received
        -
        (
            index
            %
            7
        )
    ).astype(
        int
    )


    # --------------------------------------------------------
    # Laboratory workforce
    # --------------------------------------------------------

    technicians_scheduled = (
        12
        +
        (
            index
            %
            9
        )
    ).astype(
        int
    )


    staff_absence = (
        (
            index
            %
            5
        )
        ==
        0
    ).astype(
        int
    )


    lab_staff_on_duty = (
        technicians_scheduled
        -
        staff_absence
    ).astype(
        int
    )


    # --------------------------------------------------------
    # Laboratory service amount
    # --------------------------------------------------------

    estimated_test_charge = (
        45.0
        +
        (
            index
            %
            31
        )
        *
        2.75
    )


    final_laboratory_bill = (
        estimated_test_charge
        *
        (
            0.92
            +
            (
                index
                %
                9
            )
            *
            0.018
        )
    )


    # --------------------------------------------------------
    # Result turnaround
    # --------------------------------------------------------

    target_result_turnaround = (
        30.0
        +
        (
            index
            %
            8
        )
        *
        5.0
    )


    actual_reporting_time = (
        target_result_turnaround
        *
        (
            0.78
            +
            (
                index
                %
                11
            )
            *
            0.035
        )
    )


    # --------------------------------------------------------
    # Independent count quantity
    # --------------------------------------------------------

    analyzer_units_online = (
        3
        +
        (
            index
            %
            6
        )
    ).astype(
        int
    )


    # --------------------------------------------------------
    # Independent currency quantity
    # --------------------------------------------------------

    reagent_spend = (
        120.0
        +
        (
            index
            %
            29
        )
        *
        5.2
    )


    # --------------------------------------------------------
    # Independent proportions
    # --------------------------------------------------------

    quality_pass_rate = (
        98.0
        -
        (
            index
            %
            14
        )
        *
        0.45
    )


    quality_pass_rate = (
        np.clip(
            quality_pass_rate,
            88.0,
            99.5,
        )
    )


    sla_compliance = (
        99.0
        -
        (
            actual_reporting_time
            /
            target_result_turnaround
        )
        *
        8.5
    )


    sla_compliance = (
        np.clip(
            sla_compliance,
            82.0,
            99.0,
        )
    )


    # --------------------------------------------------------
    # Same underlying mass, different lexical formulation and
    # different physical units.
    # --------------------------------------------------------

    planned_sample_mass = (
        400.0
        +
        (
            index
            %
            17
        )
        *
        25.0
    )


    measured_specimen_weight = (
        (
            planned_sample_mass
            /
            1000.0
        )
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


    # --------------------------------------------------------
    # Independent duration quantity
    # --------------------------------------------------------

    analyzer_downtime = (
        5.0
        +
        (
            index
            %
            15
        )
        *
        2.0
    )


    # --------------------------------------------------------
    # Independent mass quantity
    # --------------------------------------------------------

    reagent_mass_used = (
        0.5
        +
        (
            index
            %
            12
        )
        *
        0.08
    )


    return pd.DataFrame(
        {
            SPECIMENS_RECEIVED:
                specimens_received,

            SAMPLES_PROCESSED:
                samples_processed,

            TECHNICIANS_SCHEDULED:
                technicians_scheduled,

            LAB_STAFF_ON_DUTY:
                lab_staff_on_duty,

            ESTIMATED_TEST_CHARGE:
                estimated_test_charge,

            FINAL_LABORATORY_BILL:
                final_laboratory_bill,

            TARGET_RESULT_TURNAROUND:
                target_result_turnaround,

            ACTUAL_REPORTING_TIME:
                actual_reporting_time,

            ANALYZER_UNITS_ONLINE:
                analyzer_units_online,

            REAGENT_SPEND:
                reagent_spend,

            QUALITY_PASS_RATE:
                quality_pass_rate,

            SLA_COMPLIANCE:
                sla_compliance,

            PLANNED_SAMPLE_MASS:
                planned_sample_mass,

            MEASURED_SPECIMEN_WEIGHT:
                measured_specimen_weight,

            ANALYZER_DOWNTIME:
                analyzer_downtime,

            REAGENT_MASS_USED:
                reagent_mass_used,
        }
    )


# ============================================================
# COLUMN CASES
#
# These expectations are frozen BEFORE the first execution.
# ============================================================

def build_clinical_lab_column_cases(
) -> list[
    SemanticColumnBenchmarkCase
]:
    return [
        SemanticColumnBenchmarkCase(
            case_id=
                "specimens_received",

            dataset_id=
                CLINICAL_LAB_DATASET_ID,

            column=
                SPECIMENS_RECEIVED,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "received",
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
                "samples_processed",

            dataset_id=
                CLINICAL_LAB_DATASET_ID,

            column=
                SAMPLES_PROCESSED,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "processed",
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
                "technicians_scheduled",

            dataset_id=
                CLINICAL_LAB_DATASET_ID,

            column=
                TECHNICIANS_SCHEDULED,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "scheduled",
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
                "lab_staff_on_duty",

            dataset_id=
                CLINICAL_LAB_DATASET_ID,

            column=
                LAB_STAFF_ON_DUTY,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "on_duty",
                        "duty",
                        "active",
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
                "estimated_test_charge",

            dataset_id=
                CLINICAL_LAB_DATASET_ID,

            column=
                ESTIMATED_TEST_CHARGE,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "estimated",
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
                "final_laboratory_bill",

            dataset_id=
                CLINICAL_LAB_DATASET_ID,

            column=
                FINAL_LABORATORY_BILL,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "final",
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
                "target_result_turnaround",

            dataset_id=
                CLINICAL_LAB_DATASET_ID,

            column=
                TARGET_RESULT_TURNAROUND,

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
                "actual_reporting_time",

            dataset_id=
                CLINICAL_LAB_DATASET_ID,

            column=
                ACTUAL_REPORTING_TIME,

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
                "analyzer_units_online",

            dataset_id=
                CLINICAL_LAB_DATASET_ID,

            column=
                ANALYZER_UNITS_ONLINE,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "online",
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
                "reagent_spend",

            dataset_id=
                CLINICAL_LAB_DATASET_ID,

            column=
                REAGENT_SPEND,

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
                "quality_pass_rate",

            dataset_id=
                CLINICAL_LAB_DATASET_ID,

            column=
                QUALITY_PASS_RATE,

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
                "sla_compliance",

            dataset_id=
                CLINICAL_LAB_DATASET_ID,

            column=
                SLA_COMPLIANCE,

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
                "planned_sample_mass",

            dataset_id=
                CLINICAL_LAB_DATASET_ID,

            column=
                PLANNED_SAMPLE_MASS,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "planned",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "measured_specimen_weight",

            dataset_id=
                CLINICAL_LAB_DATASET_ID,

            column=
                MEASURED_SPECIMEN_WEIGHT,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "measured",
                        "actual",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "analyzer_downtime",

            dataset_id=
                CLINICAL_LAB_DATASET_ID,

            column=
                ANALYZER_DOWNTIME,

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
                "reagent_mass_used",

            dataset_id=
                CLINICAL_LAB_DATASET_ID,

            column=
                REAGENT_MASS_USED,

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
# S3 PAIR CASES
#
# IMPORTANT
#
# Unlike previous development probes, many NEGATIVE pairs
# deliberately have compatible mathematical dimensions.
#
# Example:
#
# specimen count
# staff count
#
# compatible dimension = True
# same concept family = False
#
# This prevents a dimensional firewall alone from solving the
# holdout.
# ============================================================

def build_clinical_lab_pair_cases(
) -> list[
    SemanticPairBenchmarkCase
]:
    return [
        # ----------------------------------------------------
        # POSITIVE — lexical paraphrases
        # ----------------------------------------------------

        SemanticPairBenchmarkCase(
            case_id=
                "specimens_received_samples_processed",

            left_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            right_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            left_column=
                SPECIMENS_RECEIVED,

            right_column=
                SAMPLES_PROCESSED,

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
                "technicians_scheduled_staff_on_duty",

            left_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            right_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            left_column=
                TECHNICIANS_SCHEDULED,

            right_column=
                LAB_STAFF_ON_DUTY,

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
                "estimated_charge_final_bill",

            left_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            right_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            left_column=
                ESTIMATED_TEST_CHARGE,

            right_column=
                FINAL_LABORATORY_BILL,

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
                "target_turnaround_actual_reporting",

            left_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            right_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            left_column=
                TARGET_RESULT_TURNAROUND,

            right_column=
                ACTUAL_REPORTING_TIME,

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
                "planned_mass_measured_weight",

            left_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            right_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            left_column=
                PLANNED_SAMPLE_MASS,

            right_column=
                MEASURED_SPECIMEN_WEIGHT,

            same_concept_family=
                True,

            same_domain=
                True,

            distinct_variants=
                True,

            compatible_units=
                True,

            # Different physical units:
            # grams vs kilograms.
            #
            # The current system has no conversion engine.
            derived_gap_compatible=
                False,
        ),

        # ----------------------------------------------------
        # NEGATIVE — SAME DIMENSION, DIFFERENT QUANTITY
        # ----------------------------------------------------

        SemanticPairBenchmarkCase(
            case_id=
                "specimens_staff",

            left_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            right_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            left_column=
                SPECIMENS_RECEIVED,

            right_column=
                TECHNICIANS_SCHEDULED,

            same_concept_family=
                False,

            same_domain=
                True,

            compatible_units=
                True,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "samples_analyzers",

            left_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            right_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            left_column=
                SAMPLES_PROCESSED,

            right_column=
                ANALYZER_UNITS_ONLINE,

            same_concept_family=
                False,

            same_domain=
                True,

            compatible_units=
                True,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "staff_analyzers",

            left_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            right_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            left_column=
                LAB_STAFF_ON_DUTY,

            right_column=
                ANALYZER_UNITS_ONLINE,

            same_concept_family=
                False,

            same_domain=
                True,

            compatible_units=
                True,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "estimated_charge_reagent_spend",

            left_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            right_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            left_column=
                ESTIMATED_TEST_CHARGE,

            right_column=
                REAGENT_SPEND,

            same_concept_family=
                False,

            same_domain=
                True,

            compatible_units=
                True,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "final_bill_reagent_spend",

            left_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            right_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            left_column=
                FINAL_LABORATORY_BILL,

            right_column=
                REAGENT_SPEND,

            same_concept_family=
                False,

            same_domain=
                True,

            compatible_units=
                True,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "turnaround_analyzer_downtime",

            left_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            right_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            left_column=
                TARGET_RESULT_TURNAROUND,

            right_column=
                ANALYZER_DOWNTIME,

            same_concept_family=
                False,

            same_domain=
                True,

            compatible_units=
                True,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "reporting_time_analyzer_downtime",

            left_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            right_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            left_column=
                ACTUAL_REPORTING_TIME,

            right_column=
                ANALYZER_DOWNTIME,

            same_concept_family=
                False,

            same_domain=
                True,

            compatible_units=
                True,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "quality_rate_sla",

            left_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            right_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            left_column=
                QUALITY_PASS_RATE,

            right_column=
                SLA_COMPLIANCE,

            same_concept_family=
                False,

            same_domain=
                True,

            compatible_units=
                True,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "sample_mass_reagent_mass",

            left_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            right_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            left_column=
                PLANNED_SAMPLE_MASS,

            right_column=
                REAGENT_MASS_USED,

            same_concept_family=
                False,

            same_domain=
                True,

            compatible_units=
                True,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "specimen_weight_reagent_mass",

            left_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            right_dataset_id=
                CLINICAL_LAB_DATASET_ID,

            left_column=
                MEASURED_SPECIMEN_WEIGHT,

            right_column=
                REAGENT_MASS_USED,

            same_concept_family=
                False,

            same_domain=
                True,

            compatible_units=
                True,

            derived_gap_compatible=
                False,
        ),
    ]


# ============================================================
# S4 QUANTITY-FAMILY HOLDOUT CASES
#
# These are the frozen S4 ground-truth relations.
#
# Do NOT change them after the first S4 execution.
# ============================================================

def build_clinical_lab_s4_quantity_family_cases(
) -> list[
    tuple[
        str,
        str,
        str,
        bool,
    ]
]:
    return [
        (
            "specimens_received_samples_processed",
            SPECIMENS_RECEIVED,
            SAMPLES_PROCESSED,
            True,
        ),

        (
            "technicians_scheduled_staff_on_duty",
            TECHNICIANS_SCHEDULED,
            LAB_STAFF_ON_DUTY,
            True,
        ),

        (
            "estimated_charge_final_bill",
            ESTIMATED_TEST_CHARGE,
            FINAL_LABORATORY_BILL,
            True,
        ),

        (
            "target_turnaround_actual_reporting",
            TARGET_RESULT_TURNAROUND,
            ACTUAL_REPORTING_TIME,
            True,
        ),

        (
            "planned_mass_measured_weight",
            PLANNED_SAMPLE_MASS,
            MEASURED_SPECIMEN_WEIGHT,
            True,
        ),

        (
            "specimens_staff",
            SPECIMENS_RECEIVED,
            TECHNICIANS_SCHEDULED,
            False,
        ),

        (
            "samples_analyzers",
            SAMPLES_PROCESSED,
            ANALYZER_UNITS_ONLINE,
            False,
        ),

        (
            "staff_analyzers",
            LAB_STAFF_ON_DUTY,
            ANALYZER_UNITS_ONLINE,
            False,
        ),

        (
            "estimated_charge_reagent_spend",
            ESTIMATED_TEST_CHARGE,
            REAGENT_SPEND,
            False,
        ),

        (
            "final_bill_reagent_spend",
            FINAL_LABORATORY_BILL,
            REAGENT_SPEND,
            False,
        ),

        (
            "turnaround_analyzer_downtime",
            TARGET_RESULT_TURNAROUND,
            ANALYZER_DOWNTIME,
            False,
        ),

        (
            "reporting_time_analyzer_downtime",
            ACTUAL_REPORTING_TIME,
            ANALYZER_DOWNTIME,
            False,
        ),

        (
            "quality_rate_sla",
            QUALITY_PASS_RATE,
            SLA_COMPLIANCE,
            False,
        ),

        (
            "sample_mass_reagent_mass",
            PLANNED_SAMPLE_MASS,
            REAGENT_MASS_USED,
            False,
        ),

        (
            "specimen_weight_reagent_mass",
            MEASURED_SPECIMEN_WEIGHT,
            REAGENT_MASS_USED,
            False,
        ),
    ]


# ============================================================
# HOLDOUT SUITE
# ============================================================

def build_clinical_lab_semantic_benchmark(
) -> SemanticBenchmarkSuite:
    return SemanticBenchmarkSuite(
        benchmark_id=
            CLINICAL_LAB_BENCHMARK_ID,

        name=
            "Clinical laboratory semantic Holdout #6",

        domain=
            "clinical_lab_operations",

        split=
            "holdout",

        description=(
            "Sixth independent semantic holdout, frozen before "
            "the first DataLens S4 execution. It tests semantic "
            "quantity-family generalization in clinical "
            "laboratory operations. The benchmark deliberately "
            "contains positive lexical paraphrases and negative "
            "pairs that share the same mathematical dimension, "
            "so dimensional compatibility alone cannot solve it."
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
            "holdout",
            "s4",
            "independent",
            "quantity_family",
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
            CLINICAL_LAB_BENCHMARK_VERSION,
    )
