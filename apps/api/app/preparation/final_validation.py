from __future__ import annotations

from typing import (
    Dict,
    List,
    Set,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.preparation.preparation_session import (
    PreparationSessionView,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
    PreparationStageRecord,
    PreparationStageStatus,
)


# ============================================================
# VERSION
# ============================================================


FINAL_PREPARATION_VALIDATION_RULE_VERSION = (
    "final_preparation_validation_v0.1"
)


# ============================================================
# STRICT MODEL
# ============================================================


class StrictFinalValidationModel(
    BaseModel,
):
    model_config = ConfigDict(
        extra="forbid"
    )


# ============================================================
# CHECK
# ============================================================


class FinalPreparationValidationCheck(
    StrictFinalValidationModel,
):
    code: str

    passed: bool

    message: str


# ============================================================
# REPORT
# ============================================================


class FinalPreparationValidationReport(
    StrictFinalValidationModel,
):
    workflow_id: str

    session_revision: int

    passed: bool

    selected_analysis_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    check_count: int

    passed_check_count: int

    failed_check_count: int

    checks: List[
        FinalPreparationValidationCheck
    ] = Field(
        default_factory=list
    )

    blocking_reasons: List[
        str
    ] = Field(
        default_factory=list
    )

    notes: List[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        FINAL_PREPARATION_VALIDATION_RULE_VERSION
    )


# ============================================================
# ERROR
# ============================================================


class FinalPreparationValidationBlockedError(
    RuntimeError,
):
    def __init__(
        self,
        *,
        report: FinalPreparationValidationReport,
    ) -> None:
        self.report = (
            report
        )

        super().__init__(
            (
                "Final preparation validation "
                "did not pass."
            )
        )


# ============================================================
# HELPERS
# ============================================================


def _stage_map(
    session: PreparationSessionView,
) -> Dict[
    PreparationStage,
    PreparationStageRecord,
]:
    output: Dict[
        PreparationStage,
        PreparationStageRecord,
    ] = {}


    for record in (
        session.snapshot.stages
    ):
        if (
            record.stage
            in output
        ):
            raise ValueError(
                (
                    "Preparation workflow contains "
                    "a duplicate stage: "
                    f"{record.stage.value}"
                )
            )


        output[
            record.stage
        ] = record


    return output


def _has_evidence_prefix(
    record: PreparationStageRecord,
    prefix: str,
) -> bool:
    return any(
        str(
            evidence
        ).startswith(
            prefix
        )

        for evidence
        in record.evidence_refs
    )


def _contains_selected_datasets(
    *,
    record: PreparationStageRecord,

    selected_dataset_ids: List[
        str
    ],
) -> bool:
    selected: Set[
        str
    ] = set(
        selected_dataset_ids
    )


    actual: Set[
        str
    ] = set(
        record.dataset_ids
    )


    return (
        selected
        .issubset(
            actual
        )
    )


def _check(
    *,
    code: str,

    passed: bool,

    success_message: str,

    failure_message: str,
) -> FinalPreparationValidationCheck:
    return (
        FinalPreparationValidationCheck(
            code=
                code,

            passed=
                passed,

            message=(
                success_message

                if passed

                else
                failure_message
            ),
        )
    )


# ============================================================
# EVALUATION
# ============================================================


def evaluate_final_preparation_validation(
    session: PreparationSessionView,
) -> FinalPreparationValidationReport:
    """
    Validate whether a server-owned Preparation Session can
    enter the final VALIDATE=PASSED state.

    This policy does not mutate session state.

    Required stages:
        IMPORT
        UNDERSTAND
        QUALITY

        must be PASSED.

    Optional stages:
        CLEAN
        TRANSFORM
        COMBINE

        must be PASSED or SKIPPED.

    CLEAN has an additional guard:
        the deterministic cleaning planner must have actually
        evaluated the dataset.

    This prevents the default initial SKIPPED value from
    bypassing preparation.

    The VALIDATE stage itself is intentionally ignored here.
    """

    stage_by_name = (
        _stage_map(
            session
        )
    )


    selected_dataset_ids = list(
        session
        .selected_analysis_dataset_ids
    )


    checks: List[
        FinalPreparationValidationCheck
    ] = []


    # ========================================================
    # SESSION DATASET SCOPE
    # ========================================================

    checks.append(
        _check(
            code=
                "selected_datasets_present",

            passed=(
                len(
                    selected_dataset_ids
                )
                >
                0
            ),

            success_message=(
                "At least one analysis dataset "
                "is selected."
            ),

            failure_message=(
                "No analysis dataset is selected."
            ),
        )
    )


    # ========================================================
    # REQUIRED STAGES
    # ========================================================

    required_stages = [
        PreparationStage.IMPORT,
        PreparationStage.UNDERSTAND,
        PreparationStage.QUALITY,
    ]


    for stage in (
        required_stages
    ):
        record = (
            stage_by_name.get(
                stage
            )
        )


        exists = (
            record
            is not None
        )


        checks.append(
            _check(
                code=(
                    f"{stage.value}"
                    "_stage_exists"
                ),

                passed=
                    exists,

                success_message=(
                    f"{stage.value} stage exists."
                ),

                failure_message=(
                    f"{stage.value} stage "
                    "is missing."
                ),
            )
        )


        if (
            record is None
        ):
            continue


        passed = (
            record.status
            ==
            PreparationStageStatus.PASSED
        )


        checks.append(
            _check(
                code=(
                    f"{stage.value}"
                    "_stage_passed"
                ),

                passed=
                    passed,

                success_message=(
                    f"{stage.value} stage "
                    "is PASSED."
                ),

                failure_message=(
                    f"{stage.value} stage "
                    f"is {record.status.value}; "
                    "PASSED is required."
                ),
            )
        )


        dataset_scope_valid = (
            _contains_selected_datasets(
                record=
                    record,

                selected_dataset_ids=
                    selected_dataset_ids,
            )
        )


        checks.append(
            _check(
                code=(
                    f"{stage.value}"
                    "_dataset_scope"
                ),

                passed=
                    dataset_scope_valid,

                success_message=(
                    f"{stage.value} stage "
                    "covers every selected dataset."
                ),

                failure_message=(
                    f"{stage.value} stage "
                    "does not cover every selected "
                    "analysis dataset."
                ),
            )
        )


    # ========================================================
    # OPTIONAL STAGES
    # ========================================================

    optional_stages = [
        PreparationStage.CLEAN,
        PreparationStage.TRANSFORM,
        PreparationStage.COMBINE,
    ]


    allowed_optional_statuses = {
        PreparationStageStatus.PASSED,
        PreparationStageStatus.SKIPPED,
    }


    for stage in (
        optional_stages
    ):
        record = (
            stage_by_name.get(
                stage
            )
        )


        exists = (
            record
            is not None
        )


        checks.append(
            _check(
                code=(
                    f"{stage.value}"
                    "_stage_exists"
                ),

                passed=
                    exists,

                success_message=(
                    f"{stage.value} stage exists."
                ),

                failure_message=(
                    f"{stage.value} stage "
                    "is missing."
                ),
            )
        )


        if (
            record is None
        ):
            continue


        resolved = (
            record.status
            in
            allowed_optional_statuses
        )


        checks.append(
            _check(
                code=(
                    f"{stage.value}"
                    "_stage_resolved"
                ),

                passed=
                    resolved,

                success_message=(
                    f"{stage.value} stage "
                    f"is {record.status.value}."
                ),

                failure_message=(
                    f"{stage.value} stage "
                    f"is {record.status.value}; "
                    "PASSED or SKIPPED is required."
                ),
            )
        )


        if (
            record.status
            ==
            PreparationStageStatus.PASSED
        ):
            dataset_scope_valid = (
                _contains_selected_datasets(
                    record=
                        record,

                    selected_dataset_ids=
                        selected_dataset_ids,
                )
            )


            checks.append(
                _check(
                    code=(
                        f"{stage.value}"
                        "_dataset_scope"
                    ),

                    passed=
                        dataset_scope_valid,

                    success_message=(
                        f"{stage.value} stage "
                        "covers every selected dataset."
                    ),

                    failure_message=(
                        f"{stage.value} stage "
                        "does not cover every selected "
                        "analysis dataset."
                    ),
                )
            )


    # ========================================================
    # CLEAN MUST HAVE BEEN EXPLICITLY EVALUATED
    # ========================================================

    clean_record = (
        stage_by_name.get(
            PreparationStage.CLEAN
        )
    )


    cleaning_plan_evaluated = (
        clean_record
        is not None
        and
        _has_evidence_prefix(
            clean_record,
            "cleaning_plan:",
        )
    )


    checks.append(
        _check(
            code=
                "cleaning_plan_evaluated",

            passed=
                cleaning_plan_evaluated,

            success_message=(
                "The deterministic cleaning planner "
                "evaluated the selected dataset scope."
            ),

            failure_message=(
                "The CLEAN stage has not been "
                "explicitly evaluated by the "
                "deterministic cleaning planner."
            ),
        )
    )


    # ========================================================
    # EXPLICIT SKIP MUST STILL COVER DATASET SCOPE
    # ========================================================

    if (
        clean_record
        is not None
        and
        clean_record.status
        ==
        PreparationStageStatus.SKIPPED
        and
        cleaning_plan_evaluated
    ):
        clean_scope_valid = (
            _contains_selected_datasets(
                record=
                    clean_record,

                selected_dataset_ids=
                    selected_dataset_ids,
            )
        )


        checks.append(
            _check(
                code=
                    "clean_skipped_dataset_scope",

                passed=
                    clean_scope_valid,

                success_message=(
                    "Skipped CLEAN stage was evaluated "
                    "for every selected dataset."
                ),

                failure_message=(
                    "Skipped CLEAN stage does not cover "
                    "every selected dataset."
                ),
            )
        )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    failed_checks = [
        check

        for check
        in checks

        if not (
            check.passed
        )
    ]


    passed = (
        len(
            failed_checks
        )
        ==
        0
    )


    blocking_reasons = [
        check.message

        for check
        in failed_checks
    ]


    return (
        FinalPreparationValidationReport(
            workflow_id=
                session.workflow_id,

            session_revision=
                session.revision,

            passed=
                passed,

            selected_analysis_dataset_ids=
                selected_dataset_ids,

            check_count=
                len(
                    checks
                ),

            passed_check_count=
                sum(
                    check.passed

                    for check
                    in checks
                ),

            failed_check_count=
                len(
                    failed_checks
                ),

            checks=
                checks,

            blocking_reasons=
                blocking_reasons,

            notes=[
                (
                    "Final validation is computed "
                    "from server-owned preparation "
                    "state."
                ),

                (
                    "The client cannot submit a "
                    "VALIDATE=PASSED status."
                ),

                (
                    "The CLEAN stage must have been "
                    "explicitly evaluated by the "
                    "deterministic cleaning planner."
                ),

                (
                    "Transformation and combination "
                    "are optional in v0.1 and must "
                    "be PASSED or SKIPPED."
                ),
            ],

            rule_version=(
                FINAL_PREPARATION_VALIDATION_RULE_VERSION
            ),
        )
    )


# ============================================================
# REQUIRE
# ============================================================


def require_final_preparation_validation(
    session: PreparationSessionView,
) -> FinalPreparationValidationReport:
    report = (
        evaluate_final_preparation_validation(
            session
        )
    )


    if not (
        report.passed
    ):
        raise (
            FinalPreparationValidationBlockedError(
                report=
                    report
            )
        )


    return (
        report
    )