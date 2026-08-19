from __future__ import annotations


from typing import (
    List,
    Optional,
)


from app.preparation.preparation_session import (
    create_preparation_session,
    reset_preparation_session_store_for_tests,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
    PreparationStageRecord,
    PreparationStageStatus,
    evaluate_preparation_workflow,
)


# ============================================================
# DATASETS
# ============================================================


ROOT_DATASET_ID = (
    "dataset:0001"
)


CLEANED_DATASET_ID = (
    "dataset:clean:abc"
)


# ============================================================
# STAGE FACTORY
# ============================================================


def stage(
    *,
    name: PreparationStage,
    status: PreparationStageStatus,
    required: bool,
    dataset_ids: Optional[
        List[
            str
        ]
    ] = None,
    blocking_reasons: Optional[
        List[
            str
        ]
    ] = None,
) -> PreparationStageRecord:
    return (
        PreparationStageRecord(
            stage=
                name,

            status=
                status,

            required=
                required,

            dataset_ids=(
                list(
                    dataset_ids
                )
                if (
                    dataset_ids
                    is not None
                )
                else []
            ),

            evidence_refs=[],

            blocking_reasons=(
                list(
                    blocking_reasons
                )
                if (
                    blocking_reasons
                    is not None
                )
                else []
            ),

            details={},
        )
    )


# ============================================================
# COMMON PASSED ROOT STAGES
# ============================================================


def passed_root_stages() -> List[
    PreparationStageRecord
]:
    return [
        stage(
            name=
                PreparationStage.IMPORT,

            status=
                PreparationStageStatus.PASSED,

            required=
                True,

            dataset_ids=[
                ROOT_DATASET_ID,
            ],
        ),

        stage(
            name=
                PreparationStage.UNDERSTAND,

            status=
                PreparationStageStatus.PASSED,

            required=
                True,

            dataset_ids=[
                ROOT_DATASET_ID,
            ],
        ),

        stage(
            name=
                PreparationStage.QUALITY,

            status=
                PreparationStageStatus.PASSED,

            required=
                True,

            dataset_ids=[
                ROOT_DATASET_ID,
            ],
        ),
    ]


# ============================================================
# READY DERIVED OUTPUT
# ============================================================


def test_derived_analysis_output_can_be_ready(
) -> None:
    stages = (
        passed_root_stages()
        +
        [
            stage(
                name=
                    PreparationStage.CLEAN,

                status=
                    PreparationStageStatus.PASSED,

                required=
                    False,

                dataset_ids=[
                    CLEANED_DATASET_ID,
                ],
            ),

            stage(
                name=
                    PreparationStage.TRANSFORM,

                status=
                    PreparationStageStatus.SKIPPED,

                required=
                    False,
            ),

            stage(
                name=
                    PreparationStage.COMBINE,

                status=
                    PreparationStageStatus.SKIPPED,

                required=
                    False,
            ),

            stage(
                name=
                    PreparationStage.VALIDATE,

                status=
                    PreparationStageStatus.PASSED,

                required=
                    True,

                dataset_ids=[
                    CLEANED_DATASET_ID,
                ],
            ),
        ]
    )


    snapshot = (
        evaluate_preparation_workflow(
            workflow_id=
                "prep:test-derived-output",

            selected_analysis_dataset_ids=[
                ROOT_DATASET_ID,
            ],

            analysis_output_dataset_ids=[
                CLEANED_DATASET_ID,
            ],

            stages=
                stages,
        )
    )


    assert (
        snapshot
        .selected_analysis_dataset_ids
        ==
        [
            ROOT_DATASET_ID,
        ]
    )


    assert (
        snapshot
        .analysis_output_dataset_ids
        ==
        [
            CLEANED_DATASET_ID,
        ]
    )


    assert (
        snapshot
        .validated_analysis_dataset_ids
        ==
        [
            CLEANED_DATASET_ID,
        ]
    )


    assert (
        snapshot.blocking_reasons
        ==
        []
    )


    assert (
        snapshot.next_stage
        is None
    )


    assert (
        snapshot.ready_for_analysis
        is True
    )


# ============================================================
# WRONG VALIDATED OUTPUT
# ============================================================


def test_root_dataset_does_not_replace_final_output_validation(
) -> None:
    stages = (
        passed_root_stages()
        +
        [
            stage(
                name=
                    PreparationStage.CLEAN,

                status=
                    PreparationStageStatus.PASSED,

                required=
                    False,

                dataset_ids=[
                    CLEANED_DATASET_ID,
                ],
            ),

            stage(
                name=
                    PreparationStage.TRANSFORM,

                status=
                    PreparationStageStatus.SKIPPED,

                required=
                    False,
            ),

            stage(
                name=
                    PreparationStage.COMBINE,

                status=
                    PreparationStageStatus.SKIPPED,

                required=
                    False,
            ),

            stage(
                name=
                    PreparationStage.VALIDATE,

                status=
                    PreparationStageStatus.PASSED,

                required=
                    True,

                dataset_ids=[
                    ROOT_DATASET_ID,
                ],
            ),
        ]
    )


    snapshot = (
        evaluate_preparation_workflow(
            workflow_id=
                "prep:test-output-mismatch",

            selected_analysis_dataset_ids=[
                ROOT_DATASET_ID,
            ],

            analysis_output_dataset_ids=[
                CLEANED_DATASET_ID,
            ],

            stages=
                stages,
        )
    )


    assert (
        snapshot.ready_for_analysis
        is False
    )


    assert (
        snapshot.next_stage
        ==
        PreparationStage.VALIDATE
    )


    assert (
        snapshot.blocking_reasons
        ==
        [
            (
                "validate: analysis output datasets "
                "are missing from validated outputs: "
                "['dataset:clean:abc']"
            )
        ]
    )


# ============================================================
# EMPTY FINAL OUTPUT DURING PREPARATION
# ============================================================


def test_empty_analysis_output_is_valid_during_preparation(
) -> None:
    stages = (
        passed_root_stages()
        +
        [
            stage(
                name=
                    PreparationStage.CLEAN,

                status=
                    PreparationStageStatus.SKIPPED,

                required=
                    False,
            ),

            stage(
                name=
                    PreparationStage.TRANSFORM,

                status=
                    PreparationStageStatus.SKIPPED,

                required=
                    False,
            ),

            stage(
                name=
                    PreparationStage.COMBINE,

                status=
                    PreparationStageStatus.SKIPPED,

                required=
                    False,
            ),

            stage(
                name=
                    PreparationStage.VALIDATE,

                status=
                    PreparationStageStatus.NOT_STARTED,

                required=
                    True,
            ),
        ]
    )


    snapshot = (
        evaluate_preparation_workflow(
            workflow_id=
                "prep:test-empty-output",

            selected_analysis_dataset_ids=[
                ROOT_DATASET_ID,
            ],

            analysis_output_dataset_ids=[],

            stages=
                stages,
        )
    )


    assert (
        snapshot
        .analysis_output_dataset_ids
        ==
        []
    )


    assert (
        snapshot.ready_for_analysis
        is False
    )


    assert (
        snapshot.next_stage
        ==
        PreparationStage.VALIDATE
    )


# ============================================================
# EMPTY FINAL OUTPUT AFTER VALIDATE PASSED
# ============================================================


def test_empty_output_cannot_be_ready_after_validate_passes(
) -> None:
    stages = (
        passed_root_stages()
        +
        [
            stage(
                name=
                    PreparationStage.CLEAN,

                status=
                    PreparationStageStatus.SKIPPED,

                required=
                    False,
            ),

            stage(
                name=
                    PreparationStage.TRANSFORM,

                status=
                    PreparationStageStatus.SKIPPED,

                required=
                    False,
            ),

            stage(
                name=
                    PreparationStage.COMBINE,

                status=
                    PreparationStageStatus.SKIPPED,

                required=
                    False,
            ),

            stage(
                name=
                    PreparationStage.VALIDATE,

                status=
                    PreparationStageStatus.PASSED,

                required=
                    True,

                dataset_ids=[],
            ),
        ]
    )


    snapshot = (
        evaluate_preparation_workflow(
            workflow_id=
                "prep:test-empty-output-after-validation",

            selected_analysis_dataset_ids=[
                ROOT_DATASET_ID,
            ],

            analysis_output_dataset_ids=[],

            stages=
                stages,
        )
    )


    assert (
        snapshot.ready_for_analysis
        is False
    )


    assert (
        snapshot.next_stage
        ==
        PreparationStage.VALIDATE
    )


    assert (
        snapshot.blocking_reasons
        ==
        [
            (
                "validate: no final analysis output "
                "dataset is selected."
            )
        ]
    )


# ============================================================
# LEGACY COMPATIBILITY
# ============================================================


def test_legacy_workflow_contract_still_works(
) -> None:
    stages = (
        passed_root_stages()
        +
        [
            stage(
                name=
                    PreparationStage.CLEAN,

                status=
                    PreparationStageStatus.SKIPPED,

                required=
                    False,
            ),

            stage(
                name=
                    PreparationStage.TRANSFORM,

                status=
                    PreparationStageStatus.SKIPPED,

                required=
                    False,
            ),

            stage(
                name=
                    PreparationStage.COMBINE,

                status=
                    PreparationStageStatus.SKIPPED,

                required=
                    False,
            ),

            stage(
                name=
                    PreparationStage.VALIDATE,

                status=
                    PreparationStageStatus.PASSED,

                required=
                    True,

                dataset_ids=[
                    ROOT_DATASET_ID,
                ],
            ),
        ]
    )


    snapshot = (
        evaluate_preparation_workflow(
            workflow_id=
                "prep:test-legacy",

            selected_analysis_dataset_ids=[
                ROOT_DATASET_ID,
            ],

            stages=
                stages,
        )
    )


    assert (
        snapshot
        .analysis_output_dataset_ids
        ==
        [
            ROOT_DATASET_ID,
        ]
    )


    assert (
        snapshot
        .validated_analysis_dataset_ids
        ==
        [
            ROOT_DATASET_ID,
        ]
    )


    assert (
        snapshot.ready_for_analysis
        is True
    )


    assert (
        snapshot.next_stage
        is None
    )


# ============================================================
# SESSION CREATION
# ============================================================


def test_new_session_accepts_empty_analysis_output_scope(
) -> None:
    reset_preparation_session_store_for_tests()


    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                ROOT_DATASET_ID,
            ]
        )
    )


    assert (
        session
        .selected_analysis_dataset_ids
        ==
        [
            ROOT_DATASET_ID,
        ]
    )


    assert (
        session
        .analysis_output_dataset_ids
        ==
        []
    )


    assert (
        session
        .snapshot
        .selected_analysis_dataset_ids
        ==
        [
            ROOT_DATASET_ID,
        ]
    )


    assert (
        session
        .snapshot
        .analysis_output_dataset_ids
        ==
        []
    )


    assert (
        session
        .snapshot
        .ready_for_analysis
        is False
    )


    assert (
        session
        .snapshot
        .next_stage
        ==
        PreparationStage.IMPORT
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        "=== DATALENS ANALYSIS OUTPUT WORKFLOW v0.1 ==="
    )


    test_derived_analysis_output_can_be_ready()

    print(
        "Derived CLEAN output can become READY: PASS"
    )


    test_root_dataset_does_not_replace_final_output_validation()

    print(
        "Root dataset cannot replace final output: PASS"
    )


    test_empty_analysis_output_is_valid_during_preparation()

    print(
        "Empty output scope during Preparation: PASS"
    )


    test_empty_output_cannot_be_ready_after_validate_passes()

    print(
        "Empty output after VALIDATE PASSED is blocked: PASS"
    )


    test_legacy_workflow_contract_still_works()

    print(
        "Legacy workflow compatibility: PASS"
    )


    test_new_session_accepts_empty_analysis_output_scope()

    print(
        "New session with empty final output scope: PASS"
    )


    print()
    print(
        "Analysis Output Workflow v0.1: PASS"
    )


if __name__ == "__main__":
    main()