from __future__ import annotations


from app.preparation.final_validation import (
    FINAL_PREPARATION_VALIDATION_RULE_VERSION,
    FinalPreparationValidationBlockedError,
    evaluate_final_preparation_validation,
    require_final_preparation_validation,
)

from app.preparation.preparation_session import (
    create_preparation_session,
    get_preparation_session,
    record_optional_stage_signal,
    record_required_stage_signal,
    reset_preparation_session_store_for_tests,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
)


# ============================================================
# CONSTANTS
# ============================================================


DATASET_ID = (
    "dataset:0001"
)


# ============================================================
# RESET
# ============================================================


def reset_state(
) -> None:
    reset_preparation_session_store_for_tests()


# ============================================================
# SESSION HELPERS
# ============================================================


def create_quality_ready_session(
):
    reset_state()


    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                DATASET_ID
            ]
        )
    )


    for stage in [
        PreparationStage.IMPORT,
        PreparationStage.UNDERSTAND,
        PreparationStage.QUALITY,
    ]:
        record_required_stage_signal(
            workflow_id=
                session.workflow_id,

            stage=
                stage,

            completed=
                True,

            dataset_ids=[
                DATASET_ID
            ],

            evidence_refs=[
                (
                    "test:"
                    f"{stage.value}"
                )
            ],

            blocking_reasons=[],
        )


    return (
        get_preparation_session(
            session.workflow_id
        )
    )


def explicitly_skip_clean(
    workflow_id: str,
    *,
    dataset_ids: list[str] | None = None,
):
    return (
        record_optional_stage_signal(
            workflow_id=
                workflow_id,

            stage=
                PreparationStage.CLEAN,

            required=
                False,

            completed=
                False,

            review_required=
                False,

            blocked=
                False,

            dataset_ids=(
                dataset_ids
                if dataset_ids is not None
                else [
                    DATASET_ID
                ]
            ),

            evidence_refs=[
                "cleaning_plan:test"
            ],

            blocking_reasons=[],
        )
    )


def pass_clean(
    workflow_id: str,
):
    return (
        record_optional_stage_signal(
            workflow_id=
                workflow_id,

            stage=
                PreparationStage.CLEAN,

            required=
                True,

            completed=
                True,

            review_required=
                False,

            blocked=
                False,

            dataset_ids=[
                DATASET_ID
            ],

            evidence_refs=[
                "cleaning_plan:test",
                "cleaning_execution:test",
            ],

            blocking_reasons=[],
        )
    )


def review_clean(
    workflow_id: str,
):
    return (
        record_optional_stage_signal(
            workflow_id=
                workflow_id,

            stage=
                PreparationStage.CLEAN,

            required=
                True,

            completed=
                False,

            review_required=
                True,

            blocked=
                False,

            dataset_ids=[
                DATASET_ID
            ],

            evidence_refs=[
                "cleaning_plan:test",
            ],

            blocking_reasons=[
                "Analyst review remains."
            ],
        )
    )


# ============================================================
# CHECK HELPERS
# ============================================================


def check_by_code(
    report,
    code: str,
):
    return next(
        check

        for check
        in report.checks

        if (
            check.code
            ==
            code
        )
    )


# ============================================================
# VERSION
# ============================================================


def test_version(
) -> None:
    assert (
        FINAL_PREPARATION_VALIDATION_RULE_VERSION
        ==
        "final_preparation_validation_v0.1"
    )


    print(
        "Final Preparation Validation v0.1 version: PASS"
    )


# ============================================================
# DEFAULT CLEAN SKIP CANNOT BYPASS VALIDATION
# ============================================================


def test_default_clean_skip_cannot_bypass_validation(
) -> None:
    session = (
        create_quality_ready_session()
    )


    report = (
        evaluate_final_preparation_validation(
            session
        )
    )


    cleaning_check = (
        check_by_code(
            report,
            "cleaning_plan_evaluated",
        )
    )


    assert (
        report.passed
        is False
    )

    assert (
        cleaning_check.passed
        is False
    )


    print(
        "Default CLEAN state cannot bypass deterministic "
        "cleaning evaluation: PASS"
    )


# ============================================================
# EXPLICIT CLEAN SKIP
# ============================================================


def test_explicit_clean_skip_passes_legacy_root_contract(
) -> None:
    session = (
        create_quality_ready_session()
    )


    explicitly_skip_clean(
        session.workflow_id
    )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    report = (
        require_final_preparation_validation(
            current
        )
    )


    assert (
        report.passed
        is True
    )


    assert (
        check_by_code(
            report,
            "cleaning_plan_evaluated",
        )
        .passed
        is True
    )


    assert (
        check_by_code(
            report,
            "clean_skipped_dataset_scope",
        )
        .passed
        is True
    )


    print(
        "Explicit CLEAN skip passes legacy root validation: PASS"
    )


# ============================================================
# CLEAN PASSED
# ============================================================


def test_clean_passed_passes_legacy_root_contract(
) -> None:
    session = (
        create_quality_ready_session()
    )


    pass_clean(
        session.workflow_id
    )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    report = (
        require_final_preparation_validation(
            current
        )
    )


    assert (
        report.passed
        is True
    )


    assert (
        check_by_code(
            report,
            "clean_stage_resolved",
        )
        .passed
        is True
    )


    assert (
        check_by_code(
            report,
            "clean_dataset_scope",
        )
        .passed
        is True
    )


    print(
        "Completed CLEAN stage passes legacy root validation: PASS"
    )


# ============================================================
# CLEAN REVIEW
# ============================================================


def test_clean_review_blocks_legacy_root_contract(
) -> None:
    session = (
        create_quality_ready_session()
    )


    review_clean(
        session.workflow_id
    )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    report = (
        evaluate_final_preparation_validation(
            current
        )
    )


    assert (
        report.passed
        is False
    )


    assert (
        check_by_code(
            report,
            "clean_stage_resolved",
        )
        .passed
        is False
    )


    print(
        "CLEAN review requirement blocks legacy root validation: PASS"
    )


# ============================================================
# REQUIRE RAISES WHEN BLOCKED
# ============================================================


def test_require_raises_when_validation_is_blocked(
) -> None:
    session = (
        create_quality_ready_session()
    )


    try:
        require_final_preparation_validation(
            session
        )

    except FinalPreparationValidationBlockedError as error:
        assert (
            error.report.passed
            is False
        )

        assert (
            error.report.failed_check_count
            >
            0
        )

    else:
        raise AssertionError(
            "Blocked legacy validation did not raise "
            "FinalPreparationValidationBlockedError."
        )


    print(
        "Legacy require gate raises on blocked validation: PASS"
    )


# ============================================================
# EXPLICIT SKIP DATASET SCOPE
# ============================================================


def test_explicit_clean_skip_requires_root_dataset_scope(
) -> None:
    session = (
        create_quality_ready_session()
    )


    explicitly_skip_clean(
        session.workflow_id,
        dataset_ids=[
            "dataset:wrong"
        ],
    )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    report = (
        evaluate_final_preparation_validation(
            current
        )
    )


    scope_check = (
        check_by_code(
            report,
            "clean_skipped_dataset_scope",
        )
    )


    assert (
        report.passed
        is False
    )

    assert (
        scope_check.passed
        is False
    )


    print(
        "Explicit CLEAN skip must cover Preparation roots: PASS"
    )


# ============================================================
# RULE VERSION IN REPORT
# ============================================================


def test_report_exposes_rule_version(
) -> None:
    session = (
        create_quality_ready_session()
    )


    report = (
        evaluate_final_preparation_validation(
            session
        )
    )


    assert (
        report.rule_version
        ==
        FINAL_PREPARATION_VALIDATION_RULE_VERSION
    )


    print(
        "Legacy validation report exposes rule version: PASS"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:
    print(
        "=== DATALENS FINAL PREPARATION VALIDATION v0.1 ==="
    )

    print()


    test_version()

    test_default_clean_skip_cannot_bypass_validation()

    test_explicit_clean_skip_passes_legacy_root_contract()

    test_clean_passed_passes_legacy_root_contract()

    test_clean_review_blocks_legacy_root_contract()

    test_require_raises_when_validation_is_blocked()

    test_explicit_clean_skip_requires_root_dataset_scope()

    test_report_exposes_rule_version()


    print()

    print(
        "Final Preparation Validation v0.1: PASS"
    )


if __name__ == "__main__":
    main()