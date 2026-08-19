from __future__ import annotations

from types import (
    SimpleNamespace,
)

import pandas as pd

from app.preparation.analysis_output_selection import (
    ANALYSIS_OUTPUT_SELECTION_RULE_VERSION,
    AnalysisOutputSelectionBlockedError,
    evaluate_analysis_output_selection,
    require_analysis_output_selection,
)

from app.preparation.preparation_artifact_store import (
    put_preparation_artifact,
    reset_preparation_artifact_store_for_tests,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
    PreparationStageStatus,
)


# ============================================================
# CONSTANTS
# ============================================================


WORKFLOW_ID = (
    "workflow-analysis-output-selection-test"
)


# ============================================================
# DATAFRAMES
# ============================================================


def sales_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "sale_id": [
                    1,
                    2,
                    3,
                ],

                "customer_id": [
                    "c1",
                    "c1",
                    "c2",
                ],

                "amount": [
                    10.0,
                    20.0,
                    30.0,
                ],
            }
        )
    )


def customers_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "customer_id": [
                    "c1",
                    "c2",
                ],

                "segment": [
                    "A",
                    "B",
                ],
            }
        )
    )


def summary_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "customer_id": [
                    "c1",
                    "c2",
                ],

                "total_amount": [
                    30.0,
                    30.0,
                ],
            }
        )
    )


def joined_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "sale_id": [
                    1,
                    2,
                    3,
                ],

                "customer_id": [
                    "c1",
                    "c1",
                    "c2",
                ],

                "amount": [
                    10.0,
                    20.0,
                    30.0,
                ],

                "segment": [
                    "A",
                    "A",
                    "B",
                ],
            }
        )
    )


# ============================================================
# SESSION
# ============================================================


def preparation_session(
    *,
    combine_status: PreparationStageStatus = (
        PreparationStageStatus.PASSED
    ),
):
    return (
        SimpleNamespace(
            workflow_id=(
                WORKFLOW_ID
            ),

            revision=(
                12
            ),

            selected_analysis_dataset_ids=[
                "sales",
                "customers",
            ],

            snapshot=(
                SimpleNamespace(
                    stages=[
                        SimpleNamespace(
                            stage=(
                                PreparationStage.COMBINE
                            ),

                            status=(
                                combine_status
                            ),
                        ),
                    ]
                )
            ),
        )
    )


# ============================================================
# ARTIFACT FIXTURE
# ============================================================


def put_standard_artifacts() -> None:
    put_preparation_artifact(
        workflow_id=(
            WORKFLOW_ID
        ),

        dataset_id=(
            "sales"
        ),

        dataset_filename=(
            "sales.csv"
        ),

        stage=(
            "clean"
        ),

        dataframe=(
            sales_frame()
        ),

        parent_dataset_ids=[
            "sales",
        ],

        evidence_refs=[
            "cleaning:sales",
        ],
    )


    put_preparation_artifact(
        workflow_id=(
            WORKFLOW_ID
        ),

        dataset_id=(
            "customers"
        ),

        dataset_filename=(
            "customers.csv"
        ),

        stage=(
            "clean"
        ),

        dataframe=(
            customers_frame()
        ),

        parent_dataset_ids=[
            "customers",
        ],

        evidence_refs=[
            "cleaning:customers",
        ],
    )


    put_preparation_artifact(
        workflow_id=(
            WORKFLOW_ID
        ),

        dataset_id=(
            "sales_summary"
        ),

        dataset_filename=(
            "sales_summary.csv"
        ),

        stage=(
            "transform"
        ),

        dataframe=(
            summary_frame()
        ),

        parent_dataset_ids=[
            "sales",
        ],

        evidence_refs=[
            "transformation:aggregate",
        ],
    )


    put_preparation_artifact(
        workflow_id=(
            WORKFLOW_ID
        ),

        dataset_id=(
            "sales_customers"
        ),

        dataset_filename=(
            "sales_customers.csv"
        ),

        stage=(
            "combine"
        ),

        dataframe=(
            joined_frame()
        ),

        parent_dataset_ids=[
            "sales",
            "customers",
        ],

        evidence_refs=[
            "join:validated",
        ],
    )


# ============================================================
# 1. VERSION
# ============================================================


def test_version() -> None:
    assert (
        ANALYSIS_OUTPUT_SELECTION_RULE_VERSION
        ==
        "analysis_output_selection_v0.1"
    )


    print(
        "Analysis output selection version: PASS"
    )


# ============================================================
# 2. ROOT DATASET
# ============================================================


def test_root_dataset_selectable() -> None:
    reset_preparation_artifact_store_for_tests()

    put_standard_artifacts()


    report = (
        require_analysis_output_selection(
            session=(
                preparation_session()
            ),

            requested_dataset_ids=[
                "sales",
            ],
        )
    )


    assert (
        report.valid
    )


    assert (
        report
        .selected_analysis_output_dataset_ids
        ==
        [
            "sales",
        ]
    )


    print(
        "Preparation root can remain an analysis output: PASS"
    )


# ============================================================
# 3. TRANSFORM DERIVED OUTPUT
# ============================================================


def test_transform_output_selectable() -> None:
    reset_preparation_artifact_store_for_tests()

    put_standard_artifacts()


    report = (
        require_analysis_output_selection(
            session=(
                preparation_session()
            ),

            requested_dataset_ids=[
                "sales_summary",
            ],
        )
    )


    assert (
        report.valid
    )


    assert (
        "sales_summary"
        in
        report.available_dataset_ids
    )


    candidate = next(
        candidate

        for candidate
        in report.candidates

        if (
            candidate.dataset_id
            ==
            "sales_summary"
        )
    )


    assert (
        candidate.lineage_root_dataset_ids
        ==
        [
            "sales",
        ]
    )


    print(
        "Validated TRANSFORM output selectable: PASS"
    )


# ============================================================
# 4. COMBINE OUTPUT
# ============================================================


def test_combine_output_selectable() -> None:
    reset_preparation_artifact_store_for_tests()

    put_standard_artifacts()


    report = (
        require_analysis_output_selection(
            session=(
                preparation_session()
            ),

            requested_dataset_ids=[
                "sales_customers",
            ],
        )
    )


    assert (
        report.valid
    )


    candidate = next(
        candidate

        for candidate
        in report.candidates

        if (
            candidate.dataset_id
            ==
            "sales_customers"
        )
    )


    assert (
        candidate.lineage_root_dataset_ids
        ==
        [
            "customers",
            "sales",
        ]
    )


    print(
        "Validated COMBINE output selectable: PASS"
    )


# ============================================================
# 5. MULTIPLE OUTPUTS
# ============================================================


def test_multiple_outputs_selectable() -> None:
    reset_preparation_artifact_store_for_tests()

    put_standard_artifacts()


    report = (
        require_analysis_output_selection(
            session=(
                preparation_session()
            ),

            requested_dataset_ids=[
                "sales_summary",
                "sales_customers",
                "sales_summary",
            ],
        )
    )


    assert (
        report
        .selected_analysis_output_dataset_ids
        ==
        [
            "sales_summary",
            "sales_customers",
        ]
    )


    print(
        "Multiple final analytical outputs selectable: PASS"
    )


# ============================================================
# 6. INVENTED DATASET
# ============================================================


def test_invented_dataset_rejected() -> None:
    reset_preparation_artifact_store_for_tests()

    put_standard_artifacts()


    report = (
        evaluate_analysis_output_selection(
            session=(
                preparation_session()
            ),

            requested_dataset_ids=[
                "invented_dataset",
            ],
        )
    )


    assert not (
        report.valid
    )


    assert (
        report.invalid_requested_dataset_ids
        ==
        [
            "invented_dataset",
        ]
    )


    try:
        require_analysis_output_selection(
            session=(
                preparation_session()
            ),

            requested_dataset_ids=[
                "invented_dataset",
            ],
        )


    except AnalysisOutputSelectionBlockedError:
        pass


    else:
        raise AssertionError(
            (
                "Invented analysis output must be rejected."
            )
        )


    print(
        "Invented analysis output rejected: PASS"
    )


# ============================================================
# 7. UNAUTHORIZED LINEAGE
# ============================================================


def test_unauthorized_lineage_rejected() -> None:
    reset_preparation_artifact_store_for_tests()

    put_standard_artifacts()


    put_preparation_artifact(
        workflow_id=(
            WORKFLOW_ID
        ),

        dataset_id=(
            "rogue_output"
        ),

        dataset_filename=(
            "rogue_output.csv"
        ),

        stage=(
            "transform"
        ),

        dataframe=(
            pd.DataFrame(
                {
                    "x": [
                        1,
                    ],
                }
            )
        ),

        parent_dataset_ids=[
            "private_dataset",
        ],

        evidence_refs=[
            "synthetic:test",
        ],
    )


    report = (
        evaluate_analysis_output_selection(
            session=(
                preparation_session()
            ),

            requested_dataset_ids=[
                "rogue_output",
            ],
        )
    )


    assert not (
        report.valid
    )


    assert (
        report.invalid_requested_dataset_ids
        ==
        [
            "rogue_output",
        ]
    )


    candidate = next(
        candidate

        for candidate
        in report.candidates

        if (
            candidate.dataset_id
            ==
            "rogue_output"
        )
    )


    assert not (
        candidate.selectable
    )


    print(
        "Unauthorized artifact lineage rejected: PASS"
    )


# ============================================================
# 8. COMBINE MUST BE RESOLVED
# ============================================================


def test_unresolved_combine_rejected() -> None:
    reset_preparation_artifact_store_for_tests()

    put_standard_artifacts()


    report = (
        evaluate_analysis_output_selection(
            session=(
                preparation_session(
                    combine_status=(
                        PreparationStageStatus
                        .REVIEW_REQUIRED
                    )
                )
            ),

            requested_dataset_ids=[
                "sales_customers",
            ],
        )
    )


    assert not (
        report.valid
    )


    assert (
        "sales_customers"
        in
        report.invalid_requested_dataset_ids
    )


    print(
        "Unresolved COMBINE blocks output selection: PASS"
    )


# ============================================================
# 9. EMPTY SELECTION
# ============================================================


def test_empty_selection_rejected() -> None:
    reset_preparation_artifact_store_for_tests()

    put_standard_artifacts()


    report = (
        evaluate_analysis_output_selection(
            session=(
                preparation_session()
            ),

            requested_dataset_ids=[],
        )
    )


    assert not (
        report.valid
    )


    assert (
        report
        .selected_analysis_output_dataset_ids
        ==
        []
    )


    print(
        "Empty analysis output selection rejected: PASS"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        (
            "=== DATALENS ANALYSIS OUTPUT "
            "SELECTION v0.1 ==="
        )
    )

    print()


    test_version()

    test_root_dataset_selectable()

    test_transform_output_selectable()

    test_combine_output_selectable()

    test_multiple_outputs_selectable()

    test_invented_dataset_rejected()

    test_unauthorized_lineage_rejected()

    test_unresolved_combine_rejected()

    test_empty_selection_rejected()


    print()

    print(
        "Analysis Output Selection v0.1: PASS"
    )


if __name__ == "__main__":
    main()