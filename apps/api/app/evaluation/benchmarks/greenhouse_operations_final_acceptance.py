from __future__ import annotations


import pandas as pd


from app.evaluation.registry_schemas import (
    BenchmarkDatasetSpec,
    SemanticBenchmarkSuite,
)

from app.evaluation.schemas import (
    SemanticPairBenchmarkCase,
)


GREENHOUSE_FINAL_ACCEPTANCE_BENCHMARK_ID = (
    "semantic:greenhouse_operations:"
    "final_acceptance:v0.1"
)

GREENHOUSE_FINAL_ACCEPTANCE_BENCHMARK_VERSION = (
    "greenhouse_operations_semantic_"
    "final_acceptance_v0.1"
)

GREENHOUSE_FINAL_ACCEPTANCE_DATASET_ID = (
    "greenhouse_operations:final_acceptance:0001"
)

GREENHOUSE_FINAL_ACCEPTANCE_FILENAME = (
    "synthetic_greenhouse_operations_"
    "final_acceptance.csv"
)


COLUMNS = [
    "Seedlings transplanted",
    "Harvested produce mass (kg)",
    "Marketable produce mass (kg)",
    "Target canopy temperature (C)",
    "Measured canopy temperature (C)",
    "Planned irrigation volume (liters)",
    "Delivered irrigation volume (liters)",
    "Target nutrient conductivity (mS/cm)",
    "Measured nutrient conductivity (mS/cm)",
    "Lamps scheduled",
    "Lamps operational",
    "Estimated energy charge (USD)",
    "Final energy invoice (USD)",
    "Planned labor hours",
    "Actual labor hours",
    "Crop loss rate (%)",
    "Marketable yield rate (%)",
    "Ventilation runtime (minutes)",
]


ROWS = [[1240, 812.4, 764.1, 23.5, 23.8, 4100, 4055, 2.2, 2.18, 48, 47, 684.0, 701.5, 126, 131, 4.8, 91.6, 420], [1275, 829.1, 781.0, 23.5, 23.4, 4150, 4178, 2.2, 2.23, 48, 48, 691.5, 695.2, 128, 127, 4.2, 92.4, 395], [1210, 795.8, 738.4, 23.0, 24.1, 3980, 3915, 2.15, 2.09, 46, 44, 662.2, 709.8, 122, 136, 6.7, 88.9, 510], [1315, 856.2, 817.5, 23.0, 22.9, 4300, 4282, 2.15, 2.16, 50, 50, 718.4, 714.9, 132, 130, 3.6, 94.1, 365], [1290, 842.7, 799.3, 23.5, 23.7, 4210, 4245, 2.2, 2.25, 49, 48, 704.7, 721.0, 130, 134, 4.4, 92.8, 438], [1185, 776.3, 709.2, 23.0, 24.4, 3890, 3812, 2.15, 2.04, 45, 42, 648.9, 703.6, 119, 139, 7.5, 87.6, 552], [1330, 871.5, 836.0, 23.5, 23.3, 4370, 4392, 2.2, 2.21, 51, 51, 729.3, 726.1, 134, 132, 3.2, 94.8, 348], [1255, 819.6, 770.8, 23.0, 23.6, 4060, 4094, 2.15, 2.19, 47, 46, 676.8, 689.4, 125, 129, 5.1, 91.1, 447], [1285, 838.9, 794.7, 23.5, 23.2, 4190, 4168, 2.2, 2.17, 49, 49, 699.6, 697.3, 129, 128, 4.0, 93.0, 379], [1205, 789.4, 731.6, 23.0, 24.0, 3960, 3888, 2.15, 2.08, 46, 43, 657.1, 700.5, 121, 135, 6.4, 89.3, 526], [1320, 862.8, 824.9, 23.5, 23.5, 4330, 4341, 2.2, 2.22, 50, 50, 722.0, 719.8, 133, 131, 3.4, 94.4, 356], [1235, 806.7, 755.5, 23.0, 23.9, 4020, 3976, 2.15, 2.11, 47, 45, 669.4, 692.7, 124, 132, 5.6, 90.5, 472]]


PAIR_CASES = [{'case_id': 'greenhouse:pair:001', 'left_column': 'Target canopy temperature (C)', 'right_column': 'Measured canopy temperature (C)', 'same_concept': True, 'same_concept_family': True, 'same_domain': True, 'distinct_variants': True, 'compatible_units': True, 'derived_gap_compatible': True}, {'case_id': 'greenhouse:pair:002', 'left_column': 'Planned irrigation volume (liters)', 'right_column': 'Delivered irrigation volume (liters)', 'same_concept': True, 'same_concept_family': True, 'same_domain': True, 'distinct_variants': True, 'compatible_units': True, 'derived_gap_compatible': True}, {'case_id': 'greenhouse:pair:003', 'left_column': 'Target nutrient conductivity (mS/cm)', 'right_column': 'Measured nutrient conductivity (mS/cm)', 'same_concept': True, 'same_concept_family': True, 'same_domain': True, 'distinct_variants': True, 'compatible_units': True, 'derived_gap_compatible': True}, {'case_id': 'greenhouse:pair:004', 'left_column': 'Lamps scheduled', 'right_column': 'Lamps operational', 'same_concept': True, 'same_concept_family': True, 'same_domain': True, 'distinct_variants': True, 'compatible_units': True, 'derived_gap_compatible': True}, {'case_id': 'greenhouse:pair:005', 'left_column': 'Estimated energy charge (USD)', 'right_column': 'Final energy invoice (USD)', 'same_concept': True, 'same_concept_family': True, 'same_domain': True, 'distinct_variants': True, 'compatible_units': True, 'derived_gap_compatible': True}, {'case_id': 'greenhouse:pair:006', 'left_column': 'Planned labor hours', 'right_column': 'Actual labor hours', 'same_concept': True, 'same_concept_family': True, 'same_domain': True, 'distinct_variants': True, 'compatible_units': True, 'derived_gap_compatible': True}, {'case_id': 'greenhouse:pair:007', 'left_column': 'Harvested produce mass (kg)', 'right_column': 'Marketable produce mass (kg)', 'same_concept': False, 'same_concept_family': True, 'same_domain': True, 'distinct_variants': True, 'compatible_units': True, 'derived_gap_compatible': True}, {'case_id': 'greenhouse:pair:008', 'left_column': 'Crop loss rate (%)', 'right_column': 'Marketable yield rate (%)', 'same_concept': False, 'same_concept_family': True, 'same_domain': True, 'distinct_variants': True, 'compatible_units': True, 'derived_gap_compatible': False}, {'case_id': 'greenhouse:pair:009', 'left_column': 'Seedlings transplanted', 'right_column': 'Harvested produce mass (kg)', 'same_concept': False, 'same_concept_family': False, 'same_domain': True, 'distinct_variants': False, 'compatible_units': False, 'derived_gap_compatible': False}, {'case_id': 'greenhouse:pair:010', 'left_column': 'Ventilation runtime (minutes)', 'right_column': 'Actual labor hours', 'same_concept': False, 'same_concept_family': False, 'same_domain': True, 'distinct_variants': False, 'compatible_units': True, 'derived_gap_compatible': False}, {'case_id': 'greenhouse:pair:011', 'left_column': 'Planned irrigation volume (liters)', 'right_column': 'Harvested produce mass (kg)', 'same_concept': False, 'same_concept_family': False, 'same_domain': True, 'distinct_variants': False, 'compatible_units': False, 'derived_gap_compatible': False}, {'case_id': 'greenhouse:pair:012', 'left_column': 'Measured nutrient conductivity (mS/cm)', 'right_column': 'Measured canopy temperature (C)', 'same_concept': False, 'same_concept_family': False, 'same_domain': True, 'distinct_variants': False, 'compatible_units': False, 'derived_gap_compatible': False}, {'case_id': 'greenhouse:pair:013', 'left_column': 'Final energy invoice (USD)', 'right_column': 'Harvested produce mass (kg)', 'same_concept': False, 'same_concept_family': False, 'same_domain': True, 'distinct_variants': False, 'compatible_units': False, 'derived_gap_compatible': False}, {'case_id': 'greenhouse:pair:014', 'left_column': 'Lamps operational', 'right_column': 'Ventilation runtime (minutes)', 'same_concept': False, 'same_concept_family': False, 'same_domain': True, 'distinct_variants': False, 'compatible_units': False, 'derived_gap_compatible': False}, {'case_id': 'greenhouse:pair:015', 'left_column': 'Seedlings transplanted', 'right_column': 'Lamps scheduled', 'same_concept': False, 'same_concept_family': False, 'same_domain': True, 'distinct_variants': False, 'compatible_units': True, 'derived_gap_compatible': False}, {'case_id': 'greenhouse:pair:016', 'left_column': 'Estimated energy charge (USD)', 'right_column': 'Planned labor hours', 'same_concept': False, 'same_concept_family': False, 'same_domain': True, 'distinct_variants': False, 'compatible_units': False, 'derived_gap_compatible': False}, {'case_id': 'greenhouse:pair:017', 'left_column': 'Crop loss rate (%)', 'right_column': 'Measured nutrient conductivity (mS/cm)', 'same_concept': False, 'same_concept_family': False, 'same_domain': True, 'distinct_variants': False, 'compatible_units': False, 'derived_gap_compatible': False}, {'case_id': 'greenhouse:pair:018', 'left_column': 'Marketable yield rate (%)', 'right_column': 'Marketable produce mass (kg)', 'same_concept': False, 'same_concept_family': True, 'same_domain': True, 'distinct_variants': True, 'compatible_units': False, 'derived_gap_compatible': False}]


def build_greenhouse_final_acceptance_dataframe(
) -> pd.DataFrame:
    return pd.DataFrame(
        ROWS,
        columns=COLUMNS,
    )


def build_greenhouse_final_acceptance_pair_cases(
) -> list[
    SemanticPairBenchmarkCase
]:
    return [
        SemanticPairBenchmarkCase(
            case_id=case["case_id"],
            left_dataset_id=
                GREENHOUSE_FINAL_ACCEPTANCE_DATASET_ID,
            right_dataset_id=
                GREENHOUSE_FINAL_ACCEPTANCE_DATASET_ID,
            left_column=case["left_column"],
            right_column=case["right_column"],
            same_concept=case["same_concept"],
            same_concept_family=
                case["same_concept_family"],
            same_domain=case["same_domain"],
            distinct_variants=
                case["distinct_variants"],
            compatible_units=
                case["compatible_units"],
            derived_gap_compatible=
                case["derived_gap_compatible"],
        )

        for case
        in PAIR_CASES
    ]


def build_greenhouse_final_acceptance_benchmark(
) -> SemanticBenchmarkSuite:
    return SemanticBenchmarkSuite(
        benchmark_id=
            GREENHOUSE_FINAL_ACCEPTANCE_BENCHMARK_ID,
        name=(
            "Commercial greenhouse semantic "
            "final acceptance holdout"
        ),
        domain=
            "commercial_greenhouse_operations",
        split=
            "holdout",
        description=(
            "Independent final acceptance holdout "
            "frozen before QLoRA adaptation. "
            "It evaluates same-metric state changes, "
            "related-but-distinct quantities, compatible "
            "units without semantic equivalence, and "
            "derived-gap safety."
        ),
        datasets=[
            BenchmarkDatasetSpec(
                dataset_id=
                    GREENHOUSE_FINAL_ACCEPTANCE_DATASET_ID,
                filename=
                    GREENHOUSE_FINAL_ACCEPTANCE_FILENAME,
            ),
        ],
        column_cases=[],
        pair_cases=
            build_greenhouse_final_acceptance_pair_cases(),
        safety_critical_fields=[
            "compatible_units",
            "derived_gap_compatible",
        ],
        tags=[
            "greenhouse",
            "final_acceptance",
            "holdout",
            "independent",
            "pre_training_freeze",
            "same_metric_different_state",
            "related_distinct_metric",
            "unit_compatibility",
            "derived_gap_safety",
        ],
        benchmark_version=
            GREENHOUSE_FINAL_ACCEPTANCE_BENCHMARK_VERSION,
    )
