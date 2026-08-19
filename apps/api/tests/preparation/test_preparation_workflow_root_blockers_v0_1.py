from __future__ import annotations

from app.preparation.preparation_workflow import (
    PreparationStage,
    PreparationStageRecord,
    PreparationStageStatus,
    evaluate_preparation_workflow,
)


# ============================================================
# FIXTURES
# ============================================================


DATASET_IDS = [
    "dataset:0001",
    "dataset:0002",
    "dataset:0003",
]


CLEAN_REVIEW_REASON = (
    "Le plan conserve 1 problème protégé nécessitant "
    "une revue sémantique ou analyste."
)


def stage(
    *,
    name: PreparationStage,
    status: PreparationStageStatus,
    required: bool,
    dataset_ids: list[str] | None = None,
    blocking_reasons: list[str] | None = None,
) -> PreparationStageRecord:
    return PreparationStageRecord(
        stage=name,
        status=status,
        required=required,
        dataset_ids=(
            list(dataset_ids)
            if dataset_ids is not None
            else []
        ),
        evidence_refs=[],
        blocking_reasons=(
            list(blocking_reasons)
            if blocking_reasons is not None
            else []
        ),
        details={},
    )


def base_passed_stages() -> list[PreparationStageRecord]:
    return [
        stage(
            name=PreparationStage.IMPORT,
            status=PreparationStageStatus.PASSED,
            required=True,
            dataset_ids=DATASET_IDS,
        ),
        stage(
            name=PreparationStage.UNDERSTAND,
            status=PreparationStageStatus.PASSED,
            required=True,
            dataset_ids=DATASET_IDS,
        ),
        stage(
            name=PreparationStage.QUALITY,
            status=PreparationStageStatus.PASSED,
            required=True,
            dataset_ids=DATASET_IDS,
        ),
    ]


# ============================================================
# ROOT BLOCKER — CLEAN REVIEW
# ============================================================


def test_clean_review_is_the_only_root_blocker() -> None:
    snapshot = evaluate_preparation_workflow(
        workflow_id="prep:test-root-clean",
        selected_analysis_dataset_ids=DATASET_IDS,
        stages=[
            *base_passed_stages(),
            stage(
                name=PreparationStage.CLEAN,
                status=PreparationStageStatus.REVIEW_REQUIRED,
                required=False,
                dataset_ids=DATASET_IDS,
                blocking_reasons=[
                    CLEAN_REVIEW_REASON,
                ],
            ),
            stage(
                name=PreparationStage.TRANSFORM,
                status=PreparationStageStatus.SKIPPED,
                required=False,
                dataset_ids=DATASET_IDS,
            ),
            stage(
                name=PreparationStage.COMBINE,
                status=PreparationStageStatus.SKIPPED,
                required=False,
                dataset_ids=DATASET_IDS,
            ),
            stage(
                name=PreparationStage.VALIDATE,
                status=PreparationStageStatus.NOT_STARTED,
                required=True,
                dataset_ids=[],
            ),
        ],
    )

    print()
    print("=== CLEAN ROOT BLOCKER ===")
    print(
        "Next stage:",
        snapshot.next_stage,
    )
    print(
        "Blocking reasons:",
        snapshot.blocking_reasons,
    )
    print(
        "Ready for analysis:",
        snapshot.ready_for_analysis,
    )

    assert (
        snapshot.next_stage
        ==
        PreparationStage.CLEAN
    )

    assert snapshot.blocking_reasons == [
        f"clean: {CLEAN_REVIEW_REASON}"
    ]

    assert not any(
        reason.startswith(
            "validate:"
        )
        for reason in snapshot.blocking_reasons
    )

    assert (
        snapshot.ready_for_analysis
        is False
    )


# ============================================================
# VALIDATION OUTPUT MISMATCH
# ============================================================


def test_missing_validated_dataset_is_reported_only_after_validation_passes() -> None:
    snapshot = evaluate_preparation_workflow(
        workflow_id="prep:test-validation-mismatch",
        selected_analysis_dataset_ids=DATASET_IDS,
        stages=[
            *base_passed_stages(),
            stage(
                name=PreparationStage.CLEAN,
                status=PreparationStageStatus.PASSED,
                required=False,
                dataset_ids=DATASET_IDS,
            ),
            stage(
                name=PreparationStage.TRANSFORM,
                status=PreparationStageStatus.SKIPPED,
                required=False,
                dataset_ids=DATASET_IDS,
            ),
            stage(
                name=PreparationStage.COMBINE,
                status=PreparationStageStatus.SKIPPED,
                required=False,
                dataset_ids=DATASET_IDS,
            ),
            stage(
                name=PreparationStage.VALIDATE,
                status=PreparationStageStatus.PASSED,
                required=True,
                dataset_ids=[
                    "dataset:0001",
                ],
            ),
        ],
    )

    print()
    print("=== VALIDATION OUTPUT MISMATCH ===")
    print(
        "Next stage:",
        snapshot.next_stage,
    )
    print(
        "Blocking reasons:",
        snapshot.blocking_reasons,
    )
    print(
        "Validated datasets:",
        snapshot.validated_analysis_dataset_ids,
    )

    expected_missing = (
        "validate: selected analysis datasets "
        "are missing from validated outputs: "
        "['dataset:0002', 'dataset:0003']"
    )

    assert snapshot.blocking_reasons == [
        expected_missing
    ]

    assert (
        snapshot.next_stage
        ==
        PreparationStage.VALIDATE
    )

    assert (
        snapshot.ready_for_analysis
        is False
    )


# ============================================================
# READY CASE
# ============================================================


def test_ready_workflow_has_no_blocker() -> None:
    snapshot = evaluate_preparation_workflow(
        workflow_id="prep:test-ready",
        selected_analysis_dataset_ids=DATASET_IDS,
        stages=[
            *base_passed_stages(),
            stage(
                name=PreparationStage.CLEAN,
                status=PreparationStageStatus.PASSED,
                required=False,
                dataset_ids=DATASET_IDS,
            ),
            stage(
                name=PreparationStage.TRANSFORM,
                status=PreparationStageStatus.SKIPPED,
                required=False,
                dataset_ids=DATASET_IDS,
            ),
            stage(
                name=PreparationStage.COMBINE,
                status=PreparationStageStatus.SKIPPED,
                required=False,
                dataset_ids=DATASET_IDS,
            ),
            stage(
                name=PreparationStage.VALIDATE,
                status=PreparationStageStatus.PASSED,
                required=True,
                dataset_ids=DATASET_IDS,
            ),
        ],
    )

    print()
    print("=== READY WORKFLOW ===")
    print(
        "Blocking reasons:",
        snapshot.blocking_reasons,
    )
    print(
        "Ready for analysis:",
        snapshot.ready_for_analysis,
    )

    assert snapshot.blocking_reasons == []
    assert snapshot.next_stage is None
    assert snapshot.ready_for_analysis is True


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print()
    print("========================================")
    print("DataLens Preparation Root Blockers v0.1")
    print("========================================")

    test_clean_review_is_the_only_root_blocker()
    test_missing_validated_dataset_is_reported_only_after_validation_passes()
    test_ready_workflow_has_no_blocker()

    print()
    print("========================================")
    print("PASS - preparation root blockers v0.1")
    print("========================================")


if __name__ == "__main__":
    main()
