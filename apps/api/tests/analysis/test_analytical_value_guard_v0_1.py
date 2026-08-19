from __future__ import annotations

import pandas as pd

from app.analysis_prioritization import (
    ANALYSIS_PRIORITIZATION_RULE_VERSION,
    ANALYTICAL_VALUE_GUARD_RULE_VERSION,
    prioritize_analysis_discovery,
)

from app.discovery.schemas import (
    AnalysisDiscoveryReport,
    DiscoveredAnalysis,
    DiscoveredVariable,
)


DATASET_ID = "dataset:test"


def variable(
    *,
    column: str,
    role: str,
    analysis_kind: str,
    semantic_role: str,
) -> DiscoveredVariable:
    return DiscoveredVariable(
        dataset_id=
            DATASET_ID,

        dataset_filename=
            "test.csv",

        column=
            column,

        role=
            role,

        analysis_kind=
            analysis_kind,

        semantic_role=
            semantic_role,

        concepts=[],
    )


def candidate(
    *,
    analysis_id: str,
    family: str,
    variables: list[
        DiscoveredVariable
    ],
    observed_signals: dict | None = None,
    score: float = 90.0,
) -> DiscoveredAnalysis:
    return DiscoveredAnalysis(
        analysis_id=
            analysis_id,

        scope=
            "single_dataset",

        family=
            family,

        title=
            analysis_id,

        priority_score=
            score,

        readiness=
            "executable_now",

        datasets=[
            "test.csv"
        ],

        dataset_ids=[
            DATASET_ID
        ],

        variables=
            variables,

        chart_type=
            "test",

        execution_strategy=
            "test",

        why_interesting=[],

        limitations=[],

        relationship_status=
            None,

        relationship_score=
            None,

        join_keys={},

        observed_signals=
            observed_signals
            or {},

        redundancy_key=
            analysis_id,
    )


def discovery(
    candidates: list[
        DiscoveredAnalysis
    ],
) -> AnalysisDiscoveryReport:
    return AnalysisDiscoveryReport(
        objective=
            None,

        dataset_count=
            1,

        candidate_count=
            len(
                candidates
            ),

        single_dataset_candidate_count=
            len(
                candidates
            ),

        cross_dataset_candidate_count=
            0,

        candidates=
            candidates,

        relationships=[],

        discovery_notes=[],
    )


def dataset_record(
    dataframe: pd.DataFrame,
) -> dict:
    return {
        "dataset_id":
            DATASET_ID,

        "filename":
            "test.csv",

        "dataframe":
            dataframe,
    }


def group_candidate(
    *,
    group_column: str,
    value_column: str = "quantity",
) -> DiscoveredAnalysis:
    return candidate(
        analysis_id=
            f"group:{group_column}",

        family=
            "group_comparison",

        variables=[
            variable(
                column=
                    group_column,

                role=
                    "group",

                analysis_kind=
                    "categorical",

                semantic_role=
                    "category",
            ),

            variable(
                column=
                    value_column,

                role=
                    "value",

                analysis_kind=
                    "quantitative",

                semantic_role=
                    "measure",
            ),
        ],
    )


def test_high_cardinality_first_name_is_deferred(
) -> None:
    first_names = [
        f"Name_{index % 16}"
        for index
        in range(
            40
        )
    ]


    dataframe = pd.DataFrame(
        {
            "first_name":
                first_names,

            "quantity":
                [
                    index % 5 + 1
                    for index
                    in range(
                        40
                    )
                ],
        }
    )


    report = (
        prioritize_analysis_discovery(
            discovery(
                [
                    group_candidate(
                        group_column=
                            "first_name"
                    )
                ]
            ),

            datasets=[
                dataset_record(
                    dataframe
                )
            ],
        )
    )


    assert report.selected_count == 0
    assert report.deferred_count == 1

    assert (
        report.decisions[
            0
        ].reason_code
        ==
        "record_label_dimension"
    )


    print(
        "High-cardinality first_name group comparison deferred: PASS"
    )


def test_high_cardinality_product_name_is_not_special_cased_but_is_guarded(
) -> None:
    dataframe = pd.DataFrame(
        {
            "product_name":
                [
                    f"Product_{index % 14}"
                    for index
                    in range(
                        40
                    )
                ],

            "quantity":
                [
                    index % 4 + 1
                    for index
                    in range(
                        40
                    )
                ],
        }
    )


    report = (
        prioritize_analysis_discovery(
            discovery(
                [
                    group_candidate(
                        group_column=
                            "product_name"
                    )
                ]
            ),

            datasets=[
                dataset_record(
                    dataframe
                )
            ],
        )
    )


    assert report.deferred_count == 1

    assert (
        report.decisions[
            0
        ].reason_code
        ==
        "record_label_dimension"
    )


    print(
        "Record-label guard is generic rather than first_name-specific: PASS"
    )


def test_low_cardinality_business_dimension_remains_selected(
) -> None:
    dataframe = pd.DataFrame(
        {
            "department":
                [
                    "Sales",
                    "Engineering",
                    "Finance",
                    "HR",
                ]
                *
                10,

            "quantity":
                [
                    index % 5 + 1
                    for index
                    in range(
                        40
                    )
                ],
        }
    )


    report = (
        prioritize_analysis_discovery(
            discovery(
                [
                    group_candidate(
                        group_column=
                            "department"
                    )
                ]
            ),

            datasets=[
                dataset_record(
                    dataframe
                )
            ],
        )
    )


    assert report.selected_count == 1


    print(
        "Low-cardinality business grouping remains selectable: PASS"
    )


def test_fragmented_non_name_dimension_is_deferred(
) -> None:
    groups = [
        f"G{index % 14}"
        for index
        in range(
            40
        )
    ]


    dataframe = pd.DataFrame(
        {
            "segment":
                groups,

            "quantity":
                [
                    index % 3 + 1
                    for index
                    in range(
                        40
                    )
                ],
        }
    )


    report = (
        prioritize_analysis_discovery(
            discovery(
                [
                    group_candidate(
                        group_column=
                            "segment"
                    )
                ]
            ),

            datasets=[
                dataset_record(
                    dataframe
                )
            ],
        )
    )


    assert report.deferred_count == 1

    assert (
        report.decisions[
            0
        ].reason_code
        ==
        "fragmented_group_dimension"
    )


    print(
        "Fragmented grouping deferred using observed group sizes: PASS"
    )


def test_sparse_categorical_structure_is_deferred(
) -> None:
    categorical_candidate = candidate(
        analysis_id=
            "categorical:sparse",

        family=
            "categorical_association",

        variables=[
            variable(
                column=
                    "left_category",

                role=
                    "x",

                analysis_kind=
                    "categorical",

                semantic_role=
                    "category",
            ),

            variable(
                column=
                    "right_category",

                role=
                    "y",

                analysis_kind=
                    "categorical",

                semantic_role=
                    "category",
            ),
        ],

        observed_signals={
            "valid_observations":
                40,

            "left_levels":
                10,

            "right_levels":
                8,
        },
    )


    report = (
        prioritize_analysis_discovery(
            discovery(
                [
                    categorical_candidate
                ]
            )
        )
    )


    assert report.deferred_count == 1

    assert (
        report.decisions[
            0
        ].reason_code
        ==
        "sparse_categorical_structure"
    )


    print(
        "Sparse categorical structure deferred before execution: PASS"
    )


def test_dense_categorical_structure_remains_selectable(
) -> None:
    categorical_candidate = candidate(
        analysis_id=
            "categorical:dense",

        family=
            "categorical_association",

        variables=[
            variable(
                column=
                    "segment",

                role=
                    "x",

                analysis_kind=
                    "categorical",

                semantic_role=
                    "category",
            ),

            variable(
                column=
                    "channel",

                role=
                    "y",

                analysis_kind=
                    "categorical",

                semantic_role=
                    "category",
            ),
        ],

        observed_signals={
            "valid_observations":
                120,

            "left_levels":
                3,

            "right_levels":
                4,
        },
    )


    report = (
        prioritize_analysis_discovery(
            discovery(
                [
                    categorical_candidate
                ]
            )
        )
    )


    assert report.selected_count == 1


    print(
        "Dense categorical structure remains selectable: PASS"
    )


def test_no_dataframe_keeps_backward_compatible_prioritization(
) -> None:
    candidate_without_data = (
        group_candidate(
            group_column=
                "first_name"
        )
    )


    report = (
        prioritize_analysis_discovery(
            discovery(
                [
                    candidate_without_data
                ]
            )
        )
    )


    assert report.selected_count == 1


    print(
        "Missing DataFrame leaves legacy prioritization behavior intact: PASS"
    )


def test_component_versions(
) -> None:
    assert (
        ANALYSIS_PRIORITIZATION_RULE_VERSION
        ==
        "analysis_prioritization_v0.1"
    )


    assert (
        ANALYTICAL_VALUE_GUARD_RULE_VERSION
        ==
        "analytical_value_guard_v0.1"
    )


    print(
        "Prioritization and Analytical Value Guard versions: PASS"
    )


def main() -> None:
    print(
        "=== DATALENS ANALYTICAL VALUE GUARD v0.1 ==="
    )

    print()


    test_high_cardinality_first_name_is_deferred()

    test_high_cardinality_product_name_is_not_special_cased_but_is_guarded()

    test_low_cardinality_business_dimension_remains_selected()

    test_fragmented_non_name_dimension_is_deferred()

    test_sparse_categorical_structure_is_deferred()

    test_dense_categorical_structure_remains_selectable()

    test_no_dataframe_keeps_backward_compatible_prioritization()

    test_component_versions()


    print()

    print(
        "Analytical Value Guard v0.1: PASS"
    )


if __name__ == "__main__":
    main()
