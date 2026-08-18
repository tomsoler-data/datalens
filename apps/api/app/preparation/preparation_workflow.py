from __future__ import annotations

from enum import Enum

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
)

from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# VERSION
# ============================================================


PREPARATION_WORKFLOW_RULE_VERSION = (
    "preparation_workflow_v0.1"
)


# ============================================================
# STAGES
# ============================================================


class PreparationStage(
    str,
    Enum,
):
    IMPORT = (
        "import"
    )

    UNDERSTAND = (
        "understand"
    )

    QUALITY = (
        "quality"
    )

    CLEAN = (
        "clean"
    )

    TRANSFORM = (
        "transform"
    )

    COMBINE = (
        "combine"
    )

    VALIDATE = (
        "validate"
    )


# ============================================================
# STATUS
# ============================================================


class PreparationStageStatus(
    str,
    Enum,
):
    NOT_STARTED = (
        "not_started"
    )

    REVIEW_REQUIRED = (
        "review_required"
    )

    BLOCKED = (
        "blocked"
    )

    PASSED = (
        "passed"
    )

    SKIPPED = (
        "skipped"
    )


# ============================================================
# STAGE POLICY
# ============================================================


class PreparationStagePolicy(
    BaseModel,
):
    stage: PreparationStage

    required: bool

    order: int


STAGE_POLICIES = [
    PreparationStagePolicy(
        stage=
            PreparationStage.IMPORT,

        required=
            True,

        order=
            1,
    ),

    PreparationStagePolicy(
        stage=
            PreparationStage.UNDERSTAND,

        required=
            True,

        order=
            2,
    ),

    PreparationStagePolicy(
        stage=
            PreparationStage.QUALITY,

        required=
            True,

        order=
            3,
    ),

    PreparationStagePolicy(
        stage=
            PreparationStage.CLEAN,

        required=
            False,

        order=
            4,
    ),

    PreparationStagePolicy(
        stage=
            PreparationStage.TRANSFORM,

        required=
            False,

        order=
            5,
    ),

    PreparationStagePolicy(
        stage=
            PreparationStage.COMBINE,

        required=
            False,

        order=
            6,
    ),

    PreparationStagePolicy(
        stage=
            PreparationStage.VALIDATE,

        required=
            True,

        order=
            7,
    ),
]


STAGE_POLICY_BY_STAGE = {
    policy.stage:
        policy

    for policy
    in STAGE_POLICIES
}


# ============================================================
# STAGE RECORD
# ============================================================


class PreparationStageRecord(
    BaseModel,
):
    """
    State of one preparation stage.

    This object stores orchestration evidence only.

    It does NOT contain executable cleaning,
    transformation or join operations.
    """

    stage: PreparationStage

    status: PreparationStageStatus

    required: bool

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

    details: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )


# ============================================================
# WORKFLOW SNAPSHOT
# ============================================================


class PreparationWorkflowSnapshot(
    BaseModel,
):
    workflow_id: str

    stage_count: int

    resolved_stage_count: int

    passed_stage_count: int

    skipped_stage_count: int

    review_required_count: int

    blocked_stage_count: int

    not_started_count: int

    # ========================================================
    # PREPARATION ROOT SCOPE
    # ========================================================

    selected_analysis_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    # ========================================================
    # FINAL ANALYTICAL OUTPUT SCOPE
    # ========================================================

    analysis_output_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    # ========================================================
    # VALIDATED OUTPUT SCOPE
    # ========================================================

    validated_analysis_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    next_stage: Optional[
        PreparationStage
    ] = None

    ready_for_analysis: bool

    blocking_reasons: List[
        str
    ] = Field(
        default_factory=list
    )

    stages: List[
        PreparationStageRecord
    ] = Field(
        default_factory=list
    )

    notes: List[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        PREPARATION_WORKFLOW_RULE_VERSION
    )


# ============================================================
# NORMALIZATION
# ============================================================


def _normalize_dataset_ids(
    values: List[
        str
    ],
) -> List[
    str
]:
    output = []

    seen: Set[
        str
    ] = set()

    for raw_value in values:
        value = (
            raw_value.strip()
        )

        if not value:
            raise ValueError(
                (
                    "Preparation workflow "
                    "dataset_id cannot be empty."
                )
            )

        if (
            value
            in seen
        ):
            continue

        seen.add(
            value
        )

        output.append(
            value
        )

    return output


# ============================================================
# STAGE STRUCTURE VALIDATION
# ============================================================


def _validate_stage_structure(
    stages: List[
        PreparationStageRecord
    ],
) -> Dict[
    PreparationStage,
    PreparationStageRecord,
]:
    if not (
        stages
    ):
        raise ValueError(
            (
                "Preparation Workflow requires "
                "stage records."
            )
        )

    index: Dict[
        PreparationStage,
        PreparationStageRecord,
    ] = {}

    for record in stages:
        if (
            record.stage
            in index
        ):
            raise ValueError(
                (
                    "Duplicate preparation stage: "
                    f"{record.stage.value}"
                )
            )

        policy = (
            STAGE_POLICY_BY_STAGE.get(
                record.stage
            )
        )

        if (
            policy is None
        ):
            raise ValueError(
                (
                    "Unsupported preparation stage: "
                    f"{record.stage}"
                )
            )

        if (
            record.required
            !=
            policy.required
        ):
            raise ValueError(
                (
                    "Preparation stage required flag "
                    "does not match workflow policy: "
                    f"{record.stage.value}. "
                    f"Expected required="
                    f"{policy.required}."
                )
            )

        if (
            record.required
            and
            record.status
            ==
            PreparationStageStatus.SKIPPED
        ):
            raise ValueError(
                (
                    "Required preparation stage "
                    "cannot be SKIPPED: "
                    f"{record.stage.value}"
                )
            )

        index[
            record.stage
        ] = record

    expected_stages = {
        policy.stage

        for policy
        in STAGE_POLICIES
    }

    observed_stages = set(
        index.keys()
    )

    missing = (
        expected_stages
        -
        observed_stages
    )

    if (
        missing
    ):
        raise ValueError(
            (
                "Preparation workflow is missing "
                "stage records: "
                f"{sorted(stage.value for stage in missing)}"
            )
        )

    extra = (
        observed_stages
        -
        expected_stages
    )

    if (
        extra
    ):
        raise ValueError(
            (
                "Preparation workflow contains "
                "unsupported stages: "
                f"{sorted(stage.value for stage in extra)}"
            )
        )

    return index


# ============================================================
# STAGE ORDER
# ============================================================


def _ordered_records(
    stage_index: Dict[
        PreparationStage,
        PreparationStageRecord,
    ],
) -> List[
    PreparationStageRecord
]:
    return [
        stage_index[
            policy.stage
        ]

        for policy
        in sorted(
            STAGE_POLICIES,
            key=
                lambda policy:
                    policy.order,
        )
    ]


# ============================================================
# RESOLUTION
# ============================================================


def _is_resolved(
    record: PreparationStageRecord,
) -> bool:
    return (
        record.status
        in {
            PreparationStageStatus.PASSED,
            PreparationStageStatus.SKIPPED,
        }
    )


def _is_blocking_status(
    record: PreparationStageRecord,
) -> bool:
    return (
        record.status
        in {
            PreparationStageStatus.NOT_STARTED,
            PreparationStageStatus.REVIEW_REQUIRED,
            PreparationStageStatus.BLOCKED,
        }
    )


# ============================================================
# NEXT STAGE
# ============================================================


def _find_next_stage(
    records: List[
        PreparationStageRecord
    ],
) -> Optional[
    PreparationStage
]:
    for record in records:
        if not (
            _is_resolved(
                record
            )
        ):
            return (
                record.stage
            )

    return None


# ============================================================
# BLOCKING REASONS
# ============================================================


def _build_blocking_reasons(
    records: List[
        PreparationStageRecord
    ],
) -> List[
    str
]:
    """
    Return only the blocking reasons for the first unresolved
    stage in workflow order.

    Downstream unresolved stages are consequences of the current
    preparation state, not independent root causes. Keeping only
    the first unresolved stage makes the snapshot actionable while
    preserving the strict readiness gate.
    """

    root_record: Optional[
        PreparationStageRecord
    ] = None

    for record in records:
        if (
            _is_blocking_status(
                record
            )
        ):
            root_record = (
                record
            )

            break

    if (
        root_record is None
    ):
        return []

    if (
        root_record.blocking_reasons
    ):
        return [
            (
                f"{root_record.stage.value}: "
                f"{reason}"
            )

            for reason
            in root_record.blocking_reasons
        ]

    if (
        root_record.status
        ==
        PreparationStageStatus.NOT_STARTED
    ):
        return [
            (
                f"{root_record.stage.value}: "
                "stage has not been resolved."
            )
        ]

    if (
        root_record.status
        ==
        PreparationStageStatus.REVIEW_REQUIRED
    ):
        return [
            (
                f"{root_record.stage.value}: "
                "human review is still required."
            )
        ]

    if (
        root_record.status
        ==
        PreparationStageStatus.BLOCKED
    ):
        return [
            (
                f"{root_record.stage.value}: "
                "stage is blocked."
            )
        ]

    return []


# ============================================================
# PUBLIC API
# ============================================================


def evaluate_preparation_workflow(
    *,
    workflow_id: str,
    stages: List[
        PreparationStageRecord
    ],
    selected_analysis_dataset_ids: List[
        str
    ],
    analysis_output_dataset_ids: Optional[
        List[
            str
        ]
    ] = None,
) -> PreparationWorkflowSnapshot:
    """
    Evaluate the complete preparation workflow.

    This function is an orchestration/readiness gate only.

    It NEVER:

    - mutates data;
    - executes cleaning;
    - executes transformations;
    - executes joins;
    - performs statistical analysis;
    - changes human decisions.

    Dataset scopes:

    selected_analysis_dataset_ids
        Immutable Preparation root datasets.

    analysis_output_dataset_ids
        Final materialized datasets explicitly selected for
        analytical execution.

    Compatibility:

    analysis_output_dataset_ids=None
        Legacy callers. Preparation roots are treated as the
        final analytical outputs.

    analysis_output_dataset_ids=[]
        Current Preparation workflow with no final analytical
        output selected yet.

    A workflow can become READY FOR ANALYSIS only when:

    - every required stage is PASSED;
    - every optional stage is either PASSED or SKIPPED;
    - no REVIEW_REQUIRED stage remains;
    - no BLOCKED stage remains;
    - no NOT_STARTED stage remains;
    - at least one Preparation root dataset exists;
    - at least one final analysis output is selected;
    - every final analysis output appears in the PASSED
      VALIDATE stage.
    """

    normalized_workflow_id = (
        workflow_id.strip()
    )

    if not (
        normalized_workflow_id
    ):
        raise ValueError(
            (
                "workflow_id cannot "
                "be empty."
            )
        )

    # ========================================================
    # PREPARATION ROOT SCOPE
    # ========================================================

    selected_ids = (
        _normalize_dataset_ids(
            selected_analysis_dataset_ids
        )
    )

    if not (
        selected_ids
    ):
        raise ValueError(
            (
                "At least one analysis dataset "
                "must be explicitly selected."
            )
        )

    # ========================================================
    # FINAL ANALYSIS OUTPUT SCOPE
    #
    # None:
    #     legacy caller — roots are also analytical outputs.
    #
    # []:
    #     explicit current state — final analytical output has
    #     not been selected yet.
    # ========================================================

    analysis_output_ids = (
        list(
            selected_ids
        )
        if (
            analysis_output_dataset_ids
            is None
        )
        else
        _normalize_dataset_ids(
            analysis_output_dataset_ids
        )
    )

    # ========================================================
    # STAGE STRUCTURE
    # ========================================================

    stage_index = (
        _validate_stage_structure(
            stages
        )
    )

    ordered = (
        _ordered_records(
            stage_index
        )
    )

    # ========================================================
    # NORMALIZE STAGE DATASET IDS
    # ========================================================

    normalized_records: List[
        PreparationStageRecord
    ] = []

    for record in ordered:
        normalized_records.append(
            record.model_copy(
                update={
                    "dataset_ids":
                        _normalize_dataset_ids(
                            record.dataset_ids
                        )
                }
            )
        )

    stage_index = {
        record.stage:
            record

        for record
        in normalized_records
    }

    # ========================================================
    # GLOBAL BLOCKING REASONS
    # ========================================================

    blocking_reasons = (
        _build_blocking_reasons(
            normalized_records
        )
    )

    # ========================================================
    # VALIDATED ANALYSIS DATASETS
    # ========================================================

    validation_stage = (
        stage_index[
            PreparationStage.VALIDATE
        ]
    )

    validated_analysis_dataset_ids = (
        list(
            validation_stage
            .dataset_ids
        )
        if (
            validation_stage.status
            ==
            PreparationStageStatus.PASSED
        )
        else
        []
    )

    validated_set = set(
        validated_analysis_dataset_ids
    )

    analysis_output_set = set(
        analysis_output_ids
    )

    missing_validation = (
        analysis_output_set
        -
        validated_set
    )

    # ========================================================
    # FINAL OUTPUT READINESS ERRORS
    # ========================================================

    if (
        validation_stage.status
        ==
        PreparationStageStatus.PASSED
        and
        not analysis_output_ids
    ):
        blocking_reasons.append(
            (
                "validate: no final analysis output "
                "dataset is selected."
            )
        )

    elif (
        validation_stage.status
        ==
        PreparationStageStatus.PASSED
        and
        missing_validation
    ):
        blocking_reasons.append(
            (
                "validate: analysis output datasets "
                "are missing from validated outputs: "
                f"{sorted(missing_validation)}"
            )
        )

    # ========================================================
    # STAGE COUNTS
    # ========================================================

    passed_stage_count = sum(
        1

        for record
        in normalized_records

        if (
            record.status
            ==
            PreparationStageStatus.PASSED
        )
    )

    skipped_stage_count = sum(
        1

        for record
        in normalized_records

        if (
            record.status
            ==
            PreparationStageStatus.SKIPPED
        )
    )

    review_required_count = sum(
        1

        for record
        in normalized_records

        if (
            record.status
            ==
            PreparationStageStatus.REVIEW_REQUIRED
        )
    )

    blocked_stage_count = sum(
        1

        for record
        in normalized_records

        if (
            record.status
            ==
            PreparationStageStatus.BLOCKED
        )
    )

    not_started_count = sum(
        1

        for record
        in normalized_records

        if (
            record.status
            ==
            PreparationStageStatus.NOT_STARTED
        )
    )

    resolved_stage_count = sum(
        1

        for record
        in normalized_records

        if (
            _is_resolved(
                record
            )
        )
    )

    # ========================================================
    # READY FOR ANALYSIS
    # ========================================================

    all_stages_resolved = (
        resolved_stage_count
        ==
        len(
            normalized_records
        )
    )

    required_stages_passed = all(
        (
            record.status
            ==
            PreparationStageStatus.PASSED
        )

        for record
        in normalized_records

        if record.required
    )

    optional_stages_resolved = all(
        (
            record.status
            in {
                PreparationStageStatus.PASSED,
                PreparationStageStatus.SKIPPED,
            }
        )

        for record
        in normalized_records

        if not record.required
    )

    analysis_output_selected = bool(
        analysis_output_ids
    )

    analysis_output_datasets_validated = (
        analysis_output_selected
        and
        not missing_validation
    )

    ready_for_analysis = (
        all_stages_resolved
        and
        required_stages_passed
        and
        optional_stages_resolved
        and
        analysis_output_datasets_validated
        and
        len(
            blocking_reasons
        )
        ==
        0
    )

    # ========================================================
    # NEXT STAGE
    # ========================================================

    next_stage = (
        _find_next_stage(
            normalized_records
        )
    )

    # All stages may technically be resolved while:
    #
    # - no final analytical output has been selected;
    # - or a selected final output is absent from VALIDATE.
    #
    # In both cases the workflow boundary remains VALIDATE.
    if (
        next_stage is None
        and
        (
            not analysis_output_selected
            or
            missing_validation
        )
    ):
        next_stage = (
            PreparationStage.VALIDATE
        )

    # ========================================================
    # RESULT
    # ========================================================

    return (
        PreparationWorkflowSnapshot(
            workflow_id=
                normalized_workflow_id,

            stage_count=
                len(
                    normalized_records
                ),

            resolved_stage_count=
                resolved_stage_count,

            passed_stage_count=
                passed_stage_count,

            skipped_stage_count=
                skipped_stage_count,

            review_required_count=
                review_required_count,

            blocked_stage_count=
                blocked_stage_count,

            not_started_count=
                not_started_count,

            selected_analysis_dataset_ids=
                selected_ids,

            analysis_output_dataset_ids=
                analysis_output_ids,

            validated_analysis_dataset_ids=
                validated_analysis_dataset_ids,

            next_stage=
                next_stage,

            ready_for_analysis=
                ready_for_analysis,

            blocking_reasons=
                blocking_reasons,

            stages=
                normalized_records,

            notes=[
                (
                    "Preparation Workflow v0.1 is "
                    "an orchestration and readiness "
                    "gate only."
                ),

                (
                    "Import, Understand, Quality and "
                    "Validate are mandatory stages."
                ),

                (
                    "Clean, Transform and Combine may "
                    "be SKIPPED when they are not "
                    "required for the selected data."
                ),

                (
                    "Every stage must be resolved "
                    "before analysis can begin."
                ),

                (
                    "REVIEW_REQUIRED, BLOCKED and "
                    "NOT_STARTED stages prevent "
                    "READY FOR ANALYSIS."
                ),

                (
                    "selected_analysis_dataset_ids "
                    "represent the immutable Preparation "
                    "root scope."
                ),

                (
                    "analysis_output_dataset_ids "
                    "represent the final datasets selected "
                    "for analytical execution."
                ),

                (
                    "Every final analysis output dataset "
                    "must appear explicitly in the PASSED "
                    "Validate stage."
                ),

                (
                    "The workflow never bypasses "
                    "cleaning, transformation or "
                    "join approval layers."
                ),
            ],

            rule_version=
                PREPARATION_WORKFLOW_RULE_VERSION,
        )
    )