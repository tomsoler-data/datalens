from __future__ import annotations

from typing import (
    List,
    Optional,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
    PreparationStageRecord,
    PreparationStageStatus,
    PreparationWorkflowSnapshot,
    evaluate_preparation_workflow,
)


# ============================================================
# VERSION
# ============================================================


PREPARATION_ORCHESTRATOR_RULE_VERSION = (
    "preparation_orchestrator_v0.1"
)


# ============================================================
# STRICT INPUT MODEL
# ============================================================


class StrictPreparationInputModel(
    BaseModel,
):
    """
    Base model for every preparation orchestration input.

    Unknown fields are forbidden.

    This is important because the client must never be able
    to inject fields such as:

        status = "passed"

    Stage statuses are derived exclusively by the backend.
    """

    model_config = ConfigDict(
        extra="forbid"
    )


# ============================================================
# REQUIRED STAGE SIGNAL
# ============================================================


class RequiredPreparationStageSignal(
    StrictPreparationInputModel,
):
    """
    Input signal for mandatory preparation stages:

    - Import
    - Understand
    - Quality

    The orchestrator, not the caller, determines the final
    PreparationStageStatus.
    """

    completed: bool = False

    dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    evidence_refs: List[
        str
    ] = Field(
        default_factory=list
    )

    blocking_reasons: List[
        str
    ] = Field(
        default_factory=list
    )


# ============================================================
# OPTIONAL STAGE SIGNAL
# ============================================================


class OptionalPreparationStageSignal(
    StrictPreparationInputModel,
):
    """
    Input signal for optional preparation stages:

    - Clean
    - Transform
    - Combine

    required=False
        -> SKIPPED

    required=True + completed=True
        -> PASSED

    required=True + review_required=True
        -> REVIEW_REQUIRED

    required=True + blocked=True
        -> BLOCKED

    required=True with no outcome
        -> NOT_STARTED
    """

    required: bool = False

    completed: bool = False

    review_required: bool = False

    blocked: bool = False

    dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    evidence_refs: List[
        str
    ] = Field(
        default_factory=list
    )

    blocking_reasons: List[
        str
    ] = Field(
        default_factory=list
    )


# ============================================================
# VALIDATION STAGE SIGNAL
# ============================================================


class ValidationPreparationStageSignal(
    StrictPreparationInputModel,
):
    """
    Final preparation validation.

    completed=False
        -> NOT_STARTED

    completed=True + passed=True
        -> PASSED

    completed=True + passed=False
        -> BLOCKED
    """

    completed: bool = False

    passed: bool = False

    dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    evidence_refs: List[
        str
    ] = Field(
        default_factory=list
    )

    blocking_reasons: List[
        str
    ] = Field(
        default_factory=list
    )


# ============================================================
# ORCHESTRATION INPUT
# ============================================================


class PreparationOrchestrationInput(
    StrictPreparationInputModel,
):
    """
    Strict API/orchestration contract.

    selected_analysis_dataset_ids
        Immutable Preparation root datasets.

    analysis_output_dataset_ids
        Final materialized datasets selected for analytical
        execution.

        None is retained only for backward compatibility with
        legacy callers that predate the explicit final-output
        scope.

        An explicit empty list means that Preparation is still
        in progress and no final output has been selected yet.

    The client provides facts produced by preparation engines.

    The client cannot provide PreparationStageStatus values.
    """

    workflow_id: str

    selected_analysis_dataset_ids: List[
        str
    ]

    analysis_output_dataset_ids: Optional[
        List[
            str
        ]
    ] = None

    import_stage: RequiredPreparationStageSignal

    understand_stage: RequiredPreparationStageSignal

    quality_stage: RequiredPreparationStageSignal

    clean_stage: OptionalPreparationStageSignal = Field(
        default_factory=
            OptionalPreparationStageSignal
    )

    transform_stage: OptionalPreparationStageSignal = Field(
        default_factory=
            OptionalPreparationStageSignal
    )

    combine_stage: OptionalPreparationStageSignal = Field(
        default_factory=
            OptionalPreparationStageSignal
    )

    validate_stage: ValidationPreparationStageSignal


# ============================================================
# SIGNAL VALIDATION
# ============================================================


def _validate_required_signal(
    *,
    stage: PreparationStage,
    signal: RequiredPreparationStageSignal,
) -> None:
    if (
        signal.completed
        and
        signal.blocking_reasons
    ):
        raise ValueError(
            (
                f"{stage.value}: completed stage "
                "cannot simultaneously contain "
                "blocking reasons."
            )
        )


def _validate_optional_signal(
    *,
    stage: PreparationStage,
    signal: OptionalPreparationStageSignal,
) -> None:
    # ========================================================
    # NOT REQUIRED
    # ========================================================

    if not (
        signal.required
    ):
        if (
            signal.completed
            or
            signal.review_required
            or
            signal.blocked
        ):
            raise ValueError(
                (
                    f"{stage.value}: optional stage "
                    "cannot have execution state when "
                    "required=False."
                )
            )

        if (
            signal.blocking_reasons
        ):
            raise ValueError(
                (
                    f"{stage.value}: skipped stage "
                    "cannot contain blocking reasons."
                )
            )

        return

    # ========================================================
    # MUTUALLY EXCLUSIVE STATES
    # ========================================================

    state_count = sum(
        [
            bool(
                signal.completed
            ),

            bool(
                signal.review_required
            ),

            bool(
                signal.blocked
            ),
        ]
    )

    if (
        state_count
        >
        1
    ):
        raise ValueError(
            (
                f"{stage.value}: completed, "
                "review_required and blocked "
                "are mutually exclusive."
            )
        )

    if (
        signal.completed
        and
        signal.blocking_reasons
    ):
        raise ValueError(
            (
                f"{stage.value}: completed stage "
                "cannot contain blocking reasons."
            )
        )


def _validate_validation_signal(
    signal: ValidationPreparationStageSignal,
) -> None:
    if (
        signal.passed
        and
        not signal.completed
    ):
        raise ValueError(
            (
                "validate: passed=True requires "
                "completed=True."
            )
        )

    if (
        signal.completed
        and
        signal.passed
        and
        signal.blocking_reasons
    ):
        raise ValueError(
            (
                "validate: successful validation "
                "cannot contain blocking reasons."
            )
        )


# ============================================================
# REQUIRED STAGE MAPPING
# ============================================================


def _required_stage_record(
    *,
    stage: PreparationStage,
    signal: RequiredPreparationStageSignal,
) -> PreparationStageRecord:
    _validate_required_signal(
        stage=
            stage,

        signal=
            signal,
    )

    if (
        signal.completed
    ):
        status = (
            PreparationStageStatus.PASSED
        )

        blocking_reasons = []

    elif (
        signal.blocking_reasons
    ):
        status = (
            PreparationStageStatus.BLOCKED
        )

        blocking_reasons = list(
            signal.blocking_reasons
        )

    else:
        status = (
            PreparationStageStatus.NOT_STARTED
        )

        blocking_reasons = []

    return (
        PreparationStageRecord(
            stage=
                stage,

            status=
                status,

            required=
                True,

            dataset_ids=
                list(
                    signal.dataset_ids
                ),

            evidence_refs=
                list(
                    signal.evidence_refs
                ),

            blocking_reasons=
                blocking_reasons,

            details={
                "derived_by": (
                    PREPARATION_ORCHESTRATOR_RULE_VERSION
                ),

                "input_signal": (
                    "required_stage"
                ),
            },
        )
    )


# ============================================================
# OPTIONAL STAGE MAPPING
# ============================================================


def _optional_stage_record(
    *,
    stage: PreparationStage,
    signal: OptionalPreparationStageSignal,
) -> PreparationStageRecord:
    _validate_optional_signal(
        stage=
            stage,

        signal=
            signal,
    )

    # ========================================================
    # SKIPPED
    # ========================================================

    if not (
        signal.required
    ):
        return (
            PreparationStageRecord(
                stage=
                    stage,

                status=
                    PreparationStageStatus.SKIPPED,

                required=
                    False,

                dataset_ids=
                    list(
                        signal.dataset_ids
                    ),

                evidence_refs=
                    list(
                        signal.evidence_refs
                    ),

                blocking_reasons=[],

                details={
                    "derived_by": (
                        PREPARATION_ORCHESTRATOR_RULE_VERSION
                    ),

                    "required":
                        False,
                },
            )
        )

    # ========================================================
    # PASSED
    # ========================================================

    if (
        signal.completed
    ):
        status = (
            PreparationStageStatus.PASSED
        )

        blocking_reasons = []

    # ========================================================
    # BLOCKED
    # ========================================================

    elif (
        signal.blocked
    ):
        status = (
            PreparationStageStatus.BLOCKED
        )

        blocking_reasons = (
            list(
                signal.blocking_reasons
            )
            or
            [
                (
                    f"{stage.value} stage "
                    "is blocked."
                )
            ]
        )

    # ========================================================
    # REVIEW REQUIRED
    # ========================================================

    elif (
        signal.review_required
    ):
        status = (
            PreparationStageStatus
            .REVIEW_REQUIRED
        )

        blocking_reasons = (
            list(
                signal.blocking_reasons
            )
            or
            [
                (
                    f"{stage.value} stage "
                    "requires human review."
                )
            ]
        )

    # ========================================================
    # NOT STARTED
    # ========================================================

    else:
        status = (
            PreparationStageStatus.NOT_STARTED
        )

        blocking_reasons = []

    return (
        PreparationStageRecord(
            stage=
                stage,

            status=
                status,

            required=
                False,

            dataset_ids=
                list(
                    signal.dataset_ids
                ),

            evidence_refs=
                list(
                    signal.evidence_refs
                ),

            blocking_reasons=
                blocking_reasons,

            details={
                "derived_by": (
                    PREPARATION_ORCHESTRATOR_RULE_VERSION
                ),

                "required":
                    True,
            },
        )
    )


# ============================================================
# VALIDATION STAGE MAPPING
# ============================================================


def _validation_stage_record(
    signal: ValidationPreparationStageSignal,
) -> PreparationStageRecord:
    _validate_validation_signal(
        signal
    )

    # ========================================================
    # NOT STARTED
    # ========================================================

    if not (
        signal.completed
    ):
        status = (
            PreparationStageStatus.NOT_STARTED
        )

        blocking_reasons = []

    # ========================================================
    # PASSED
    # ========================================================

    elif (
        signal.passed
    ):
        status = (
            PreparationStageStatus.PASSED
        )

        blocking_reasons = []

    # ========================================================
    # FAILED / BLOCKED
    # ========================================================

    else:
        status = (
            PreparationStageStatus.BLOCKED
        )

        blocking_reasons = (
            list(
                signal.blocking_reasons
            )
            or
            [
                (
                    "Final preparation validation "
                    "did not pass."
                )
            ]
        )

    return (
        PreparationStageRecord(
            stage=
                PreparationStage.VALIDATE,

            status=
                status,

            required=
                True,

            dataset_ids=
                list(
                    signal.dataset_ids
                ),

            evidence_refs=
                list(
                    signal.evidence_refs
                ),

            blocking_reasons=
                blocking_reasons,

            details={
                "derived_by": (
                    PREPARATION_ORCHESTRATOR_RULE_VERSION
                ),

                "validation_passed":
                    bool(
                        signal.passed
                    ),
            },
        )
    )


# ============================================================
# PUBLIC API
# ============================================================


def orchestrate_preparation(
    orchestration_input: PreparationOrchestrationInput,
) -> PreparationWorkflowSnapshot:
    """
    Build the global preparation state from deterministic
    subsystem signals.

    The caller never provides PreparationStageStatus directly.

    Instead, the orchestrator derives:

        PASSED
        SKIPPED
        REVIEW_REQUIRED
        BLOCKED
        NOT_STARTED

    from explicit engine outcomes.

    Unknown input fields are rejected by Pydantic.

    The orchestration contract preserves two scopes:

        PREPARATION ROOTS
            selected_analysis_dataset_ids

        FINAL ANALYTICAL OUTPUTS
            analysis_output_dataset_ids

    This function NEVER:

    - mutates a DataFrame;
    - cleans data;
    - transforms data;
    - joins data;
    - approves a human decision;
    - executes analysis.

    Its only responsibility is orchestration and determining
    whether the selected final output may cross the boundary:

        PREPARATION
            ↓
        FINAL OUTPUT
            ↓
        VALIDATE
            ↓
        READY FOR ANALYSIS
    """

    workflow_id = (
        orchestration_input
        .workflow_id
        .strip()
    )

    if not (
        workflow_id
    ):
        raise ValueError(
            (
                "Preparation orchestration "
                "workflow_id cannot be empty."
            )
        )

    stages = [
        _required_stage_record(
            stage=
                PreparationStage.IMPORT,

            signal=
                orchestration_input
                .import_stage,
        ),

        _required_stage_record(
            stage=
                PreparationStage.UNDERSTAND,

            signal=
                orchestration_input
                .understand_stage,
        ),

        _required_stage_record(
            stage=
                PreparationStage.QUALITY,

            signal=
                orchestration_input
                .quality_stage,
        ),

        _optional_stage_record(
            stage=
                PreparationStage.CLEAN,

            signal=
                orchestration_input
                .clean_stage,
        ),

        _optional_stage_record(
            stage=
                PreparationStage.TRANSFORM,

            signal=
                orchestration_input
                .transform_stage,
        ),

        _optional_stage_record(
            stage=
                PreparationStage.COMBINE,

            signal=
                orchestration_input
                .combine_stage,
        ),

        _validation_stage_record(
            orchestration_input
            .validate_stage
        ),
    ]

    return (
        evaluate_preparation_workflow(
            workflow_id=
                workflow_id,

            stages=
                stages,

            selected_analysis_dataset_ids=(
                orchestration_input
                .selected_analysis_dataset_ids
            ),

            analysis_output_dataset_ids=(
                orchestration_input
                .analysis_output_dataset_ids
            ),
        )
    )