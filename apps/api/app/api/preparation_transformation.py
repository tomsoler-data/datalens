from __future__ import annotations

from typing import (
    Any,
)

from fastapi import (
    APIRouter,
    HTTPException,
)

from pydantic import (
    BaseModel,
    Field,
)

from app.preparation.post_transformation_validation import (
    PostTransformationValidationReport,
    validate_transformation_execution,
)

from app.preparation.preparation_artifact_store import (
    PreparationArtifactDatasetNotFoundError,
    PreparationArtifactWorkflowNotFoundError,
    PreparationDatasetArtifact,
    get_preparation_artifact,
    list_preparation_artifacts,
)

from app.preparation.preparation_session import (
    PreparationSessionNotFoundError,
    get_preparation_session,
    record_optional_stage_signal,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
    PreparationStageStatus,
)

from app.preparation.transformation_approval import (
    ApprovedTransformationPlan,
    TransformationApprovalCommand,
    apply_transformation_approvals,
)

from app.preparation.transformation_artifacts import (
    TransformationArtifactMaterializationReport,
    materialize_transformation_artifacts,
)

from app.preparation.transformation_contracts import (
    AggregateIntent,
    BinNumericIntent,
    CastIntent,
    DeriveArithmeticIntent,
    ExtractDatePartIntent,
    TransformationIntent,
    TransformationOperation,
    TransformationPlan,
)

from app.preparation.transformation_executor import (
    TransformationExecutionError,
    TransformationExecutionReport,
    execute_transformation_plan,
)

from app.preparation.transformation_planner import (
    plan_transformations,
)


# ============================================================
# VERSION
# ============================================================

PREPARATION_TRANSFORMATION_API_VERSION = (
    "preparation_transformation_api_v0.1"
)


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/preparation",
    tags=[
        "preparation",
    ],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class TransformationPlanRequest(
    BaseModel,
):
    """
    Browser-facing request.

    Dataset identity is server-owned.

    The browser supplies:
    - workflow_id;
    - dataset_id;
    - transformation parameters.

    dataset_filename inside individual intents is optional
    from the browser perspective and is always reconciled
    against the server-owned artifact.
    """

    workflow_id: str = Field(
        min_length=1,
    )

    dataset_id: str = Field(
        min_length=1,
    )

    intents: list[
        dict[
            str,
            Any,
        ]
    ] = Field(
        default_factory=list,
    )


class TransformationApplyRequest(
    BaseModel,
):
    workflow_id: str = Field(
        min_length=1,
    )

    dataset_id: str = Field(
        min_length=1,
    )

    intents: list[
        dict[
            str,
            Any,
        ]
    ] = Field(
        default_factory=list,
    )

    approval_commands: list[
        TransformationApprovalCommand
    ] = Field(
        default_factory=list,
    )


# ============================================================
# RESPONSE MODELS
# ============================================================

class TransformationMaterializationView(
    BaseModel,
):
    workflow_id: str

    source_dataset_id: str

    persisted_dataset_ids: list[
        str
    ]

    derived_dataset_ids: list[
        str
    ]

    artifact_count: int

    source_data_changed: bool

    materialization_kind: str

    bridge_version: str


class TransformationApplyResponse(
    BaseModel,
):
    status: str

    plan: TransformationPlan

    approved_plan: (
        ApprovedTransformationPlan
        | None
    ) = None

    execution: (
        TransformationExecutionReport
        | None
    ) = None

    validation: (
        PostTransformationValidationReport
        | None
    ) = None

    materialization: (
        TransformationMaterializationView
        | None
    ) = None

    notes: list[
        str
    ] = Field(
        default_factory=list,
    )

    api_version: str = (
        PREPARATION_TRANSFORMATION_API_VERSION
    )


# ============================================================
# NORMALIZATION
# ============================================================

def _required_text(
    value: str,
    *,
    field_name: str,
) -> str:

    normalized = (
        value.strip()
    )


    if not normalized:

        raise ValueError(
            f"{field_name} cannot be empty."
        )


    return normalized


# ============================================================
# SESSION HELPERS
# ============================================================

def _stage_record(
    *,
    session,
    stage: PreparationStage,
):

    for record in (
        session.snapshot.stages
    ):

        if (
            record.stage
            ==
            stage
        ):

            return (
                record
            )


    raise RuntimeError(
        "Preparation session snapshot is missing "
        f"stage={stage.value}."
    )


def _require_transform_precondition(
    *,
    workflow_id: str,
    dataset_id: str,
):
    """
    TRANSFORM may start only after CLEAN has been resolved.

    CLEAN may be:
    - PASSED;
    - SKIPPED.

    REVIEW_REQUIRED / BLOCKED / NOT_STARTED cannot be
    bypassed by calling the Transformation API directly.
    """

    session = (
        get_preparation_session(
            workflow_id
        )
    )


    if (
        dataset_id
        not in
        session.selected_analysis_dataset_ids
    ):

        raise HTTPException(
            status_code=403,

            detail={
                "error":
                    "transformation_dataset_not_selected",

                "message":
                    (
                        "Transformation dataset is not part "
                        "of the server-owned analytical "
                        "dataset selection."
                    ),

                "workflow_id":
                    workflow_id,

                "dataset_id":
                    dataset_id,
            },
        )


    clean_stage = (
        _stage_record(
            session=(
                session
            ),

            stage=(
                PreparationStage.CLEAN
            ),
        )
    )


    if (
        clean_stage.status
        not in {
            PreparationStageStatus.PASSED,
            PreparationStageStatus.SKIPPED,
        }
    ):

        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "clean_stage_not_resolved",

                "message":
                    (
                        "CLEAN must be PASSED or SKIPPED "
                        "before TRANSFORM can begin."
                    ),

                "workflow_id":
                    workflow_id,

                "dataset_id":
                    dataset_id,

                "clean_status":
                    clean_stage.status.value,
            },
        )


    return (
        session
    )


# ============================================================
# ARTIFACT SOURCE
# ============================================================

def _load_transform_source(
    *,
    workflow_id: str,
    dataset_id: str,
) -> PreparationDatasetArtifact:

    _require_transform_precondition(
        workflow_id=(
            workflow_id
        ),

        dataset_id=(
            dataset_id
        ),
    )


    artifact = (
        get_preparation_artifact(
            workflow_id=(
                workflow_id
            ),

            dataset_id=(
                dataset_id
            ),
        )
    )


    if (
        artifact.stage
        not in {
            "source",
            "clean",
        }
    ):

        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "transformation_source_stage_invalid",

                "message":
                    (
                        "TRANSFORM requires the current "
                        "SOURCE or CLEAN artifact."
                    ),

                "workflow_id":
                    workflow_id,

                "dataset_id":
                    dataset_id,

                "artifact_stage":
                    artifact.stage,
            },
        )


    return (
        artifact
    )


# ============================================================
# INTENT PARSING
# ============================================================

_INTENT_MODEL_BY_OPERATION = {
    TransformationOperation
    .DERIVE_ARITHMETIC
    .value:
        DeriveArithmeticIntent,

    TransformationOperation
    .CAST
    .value:
        CastIntent,

    TransformationOperation
    .BIN_NUMERIC
    .value:
        BinNumericIntent,

    TransformationOperation
    .EXTRACT_DATE_PART
    .value:
        ExtractDatePartIntent,

    TransformationOperation
    .AGGREGATE
    .value:
        AggregateIntent,
}


def _parse_transformation_intents(
    *,
    raw_intents: list[
        dict[
            str,
            Any,
        ]
    ],
    dataset_id: str,
    dataset_filename: str,
) -> list[
    TransformationIntent
]:
    """
    Parse the heterogeneous intent union explicitly by
    operation.

    The browser cannot redirect an intent to another dataset
    by modifying dataset_id or dataset_filename.
    """

    parsed: list[
        TransformationIntent
    ] = []


    seen_request_ids: set[
        str
    ] = set()


    for (
        index,
        raw_intent,
    ) in enumerate(
        raw_intents,
        start=1,
    ):

        if not isinstance(
            raw_intent,
            dict,
        ):

            raise ValueError(
                "Transformation intent must be an object. "
                f"index={index}"
            )


        operation_value = (
            str(
                raw_intent.get(
                    "operation"
                )
                or
                ""
            )
            .strip()
        )


        if not operation_value:

            raise ValueError(
                "Transformation intent is missing operation. "
                f"index={index}"
            )


        model_class = (
            _INTENT_MODEL_BY_OPERATION
            .get(
                operation_value
            )
        )


        if (
            model_class
            is None
        ):

            raise ValueError(
                "Unsupported Transformation operation: "
                f"{operation_value}"
            )


        supplied_dataset_id = (
            raw_intent.get(
                "dataset_id"
            )
        )


        if (
            supplied_dataset_id
            is not None
            and
            str(
                supplied_dataset_id
            ).strip()
            !=
            dataset_id
        ):

            raise ValueError(
                "Transformation intent dataset_id does not "
                "match the server-selected dataset. "
                f"index={index}"
            )


        supplied_filename = (
            raw_intent.get(
                "dataset_filename"
            )
        )


        if (
            supplied_filename
            is not None
            and
            str(
                supplied_filename
            ).strip()
            !=
            dataset_filename
        ):

            raise ValueError(
                "Transformation intent dataset_filename does "
                "not match the server-owned artifact. "
                f"index={index}"
            )


        normalized_payload = dict(
            raw_intent
        )


        normalized_payload[
            "dataset_id"
        ] = (
            dataset_id
        )


        normalized_payload[
            "dataset_filename"
        ] = (
            dataset_filename
        )


        intent = (
            model_class.model_validate(
                normalized_payload
            )
        )


        request_id = (
            intent.request_id.strip()
        )


        if not request_id:

            raise ValueError(
                "Transformation request_id cannot be empty. "
                f"index={index}"
            )


        if (
            request_id
            in seen_request_ids
        ):

            raise ValueError(
                "Duplicate Transformation request_id: "
                f"{request_id}"
            )


        seen_request_ids.add(
            request_id
        )


        parsed.append(
            intent
        )


    return (
        parsed
    )


# ============================================================
# DERIVED OUTPUT COLLISION GUARD
# ============================================================

def _require_derived_output_ids_available(
    *,
    workflow_id: str,
    source_dataset_id: str,
    intents: list[
        TransformationIntent
    ],
) -> None:
    """
    Aggregate outputs create new logical datasets.

    They may never silently overwrite an existing Preparation
    artifact.
    """

    existing_ids = {
        artifact.dataset_id

        for artifact in
        list_preparation_artifacts(
            workflow_id=(
                workflow_id
            )
        )
    }


    requested_output_ids: set[
        str
    ] = set()


    for intent in intents:

        if not isinstance(
            intent,
            AggregateIntent,
        ):

            continue


        output_dataset_id = (
            intent
            .output_dataset_id
            .strip()
        )


        if not output_dataset_id:

            raise ValueError(
                "Aggregate Transformation output_dataset_id "
                "cannot be empty."
            )


        if (
            output_dataset_id
            ==
            source_dataset_id
        ):

            raise ValueError(
                "Aggregate Transformation output_dataset_id "
                "cannot overwrite its source dataset."
            )


        if (
            output_dataset_id
            in requested_output_ids
        ):

            raise ValueError(
                "Duplicate aggregate output_dataset_id: "
                f"{output_dataset_id}"
            )


        if (
            output_dataset_id
            in existing_ids
        ):

            raise ValueError(
                "Aggregate Transformation output_dataset_id "
                "already exists in the Preparation Artifact "
                "Store: "
                f"{output_dataset_id}"
            )


        requested_output_ids.add(
            output_dataset_id
        )


# ============================================================
# EMPTY PLAN
# ============================================================

def _empty_transformation_plan(
    *,
    dataset_id: str,
    dataset_filename: str,
) -> TransformationPlan:

    return (
        TransformationPlan(
            dataset_id=(
                dataset_id
            ),

            dataset_filename=(
                dataset_filename
            ),

            request_count=(
                0
            ),

            step_count=(
                0
            ),

            validated_count=(
                0
            ),

            review_required_count=(
                0
            ),

            human_approval_required_count=(
                0
            ),

            ready_for_approval=(
                False
            ),

            steps=[],

            notes=[
                (
                    "No Transformation intent was requested. "
                    "TRANSFORM is skipped."
                ),
            ],
        )
    )


# ============================================================
# SESSION — PLAN
# ============================================================

def _record_transformation_plan_stage(
    *,
    workflow_id: str,
    dataset_id: str,
    plan: TransformationPlan,
) -> None:

    evidence_refs = [
        (
            "transformation_plan:"
            f"{plan.rule_version}"
        ),

        (
            "transformation_requests:"
            f"{plan.request_count}"
        ),

        (
            "transformation_steps:"
            f"{plan.step_count}"
        ),

        (
            "transformation_review_required:"
            f"{plan.review_required_count}"
        ),
    ]


    # ========================================================
    # NOTHING REQUESTED
    # ========================================================

    if (
        plan.request_count
        ==
        0
    ):

        record_optional_stage_signal(
            workflow_id=(
                workflow_id
            ),

            stage=(
                PreparationStage.TRANSFORM
            ),

            required=(
                False
            ),

            completed=(
                False
            ),

            review_required=(
                False
            ),

            blocked=(
                False
            ),

            dataset_ids=[
                dataset_id,
            ],

            evidence_refs=(
                evidence_refs
            ),

            blocking_reasons=[],
        )


        return


    # ========================================================
    # PLANNER BLOCKED
    # ========================================================

    if not (
        plan.ready_for_approval
    ):

        record_optional_stage_signal(
            workflow_id=(
                workflow_id
            ),

            stage=(
                PreparationStage.TRANSFORM
            ),

            required=(
                True
            ),

            completed=(
                False
            ),

            review_required=(
                False
            ),

            blocked=(
                True
            ),

            dataset_ids=[
                dataset_id,
            ],

            evidence_refs=(
                evidence_refs
            ),

            blocking_reasons=[
                (
                    "Transformation plan is not ready "
                    "for approval."
                ),
            ],
        )


        return


    # ========================================================
    # APPROVABLE PLAN
    # ========================================================

    record_optional_stage_signal(
        workflow_id=(
            workflow_id
        ),

        stage=(
            PreparationStage.TRANSFORM
        ),

        required=(
            True
        ),

        completed=(
            False
        ),

        review_required=(
            plan
            .human_approval_required_count
            >
            0
        ),

        blocked=(
            False
        ),

        dataset_ids=[
            dataset_id,
        ],

        evidence_refs=(
            evidence_refs
        ),

        blocking_reasons=(
            [
                (
                    "Transformation plan contains actions "
                    "that require explicit analyst approval."
                )
            ]
            if (
                plan
                .human_approval_required_count
                >
                0
            )
            else []
        ),
    )


# ============================================================
# SESSION — APPROVAL UNRESOLVED
# ============================================================

def _record_transformation_approval_unresolved(
    *,
    workflow_id: str,
    dataset_id: str,
    approved_plan: ApprovedTransformationPlan,
) -> None:

    blocked = (
        approved_plan
        .blocked_dependency_count
        >
        0
    )


    evidence_refs = [
        (
            "transformation_approval:"
            f"{approved_plan.rule_version}"
        ),

        (
            "transformation_pending:"
            f"{approved_plan.pending_count}"
        ),

        (
            "transformation_deferred:"
            f"{approved_plan.deferred_count}"
        ),

        (
            "transformation_blocked_dependencies:"
            f"{approved_plan.blocked_dependency_count}"
        ),
    ]


    reasons: list[
        str
    ] = []


    if (
        approved_plan.pending_count
        >
        0
    ):

        reasons.append(
            (
                "One or more Transformation decisions "
                "are still pending."
            )
        )


    if (
        approved_plan.deferred_count
        >
        0
    ):

        reasons.append(
            (
                "One or more Transformation decisions "
                "were deferred."
            )
        )


    if (
        approved_plan.blocked_dependency_count
        >
        0
    ):

        reasons.append(
            (
                "One or more Transformation steps are "
                "blocked by unresolved dependencies."
            )
        )


    record_optional_stage_signal(
        workflow_id=(
            workflow_id
        ),

        stage=(
            PreparationStage.TRANSFORM
        ),

        required=(
            True
        ),

        completed=(
            False
        ),

        review_required=(
            not blocked
        ),

        blocked=(
            blocked
        ),

        dataset_ids=[
            dataset_id,
        ],

        evidence_refs=(
            evidence_refs
        ),

        blocking_reasons=(
            reasons
        ),
    )


# ============================================================
# SESSION — REJECTED / NO EXECUTION
# ============================================================

def _record_transformation_skipped_after_approval(
    *,
    workflow_id: str,
    dataset_id: str,
    approved_plan: ApprovedTransformationPlan,
) -> None:

    record_optional_stage_signal(
        workflow_id=(
            workflow_id
        ),

        stage=(
            PreparationStage.TRANSFORM
        ),

        required=(
            False
        ),

        completed=(
            False
        ),

        review_required=(
            False
        ),

        blocked=(
            False
        ),

        dataset_ids=[
            dataset_id,
        ],

        evidence_refs=[
            (
                "transformation_approval:"
                f"{approved_plan.rule_version}"
            ),

            (
                "transformation_executable_steps:0"
            ),

            (
                "transformation_rejected:"
                f"{approved_plan.rejected_count}"
            ),
        ],

        blocking_reasons=[],
    )


# ============================================================
# SESSION — VALIDATION FAILED
# ============================================================

def _record_transformation_validation_failed(
    *,
    workflow_id: str,
    dataset_id: str,
    approved_plan: ApprovedTransformationPlan,
    execution: TransformationExecutionReport,
    validation: PostTransformationValidationReport,
) -> None:

    record_optional_stage_signal(
        workflow_id=(
            workflow_id
        ),

        stage=(
            PreparationStage.TRANSFORM
        ),

        required=(
            True
        ),

        completed=(
            False
        ),

        review_required=(
            False
        ),

        blocked=(
            True
        ),

        dataset_ids=[
            dataset_id,
        ],

        evidence_refs=[
            (
                "transformation_approval:"
                f"{approved_plan.rule_version}"
            ),

            (
                "transformation_execution:"
                f"{execution.rule_version}"
            ),

            (
                "transformation_validation:"
                f"{validation.rule_version}"
            ),

            (
                "transformation_validation_failed_checks:"
                f"{validation.failed_check_count}"
            ),
        ],

        blocking_reasons=[
            (
                "Post-transformation validation did not "
                "authorize the result for downstream use."
            ),
        ],
    )


# ============================================================
# SESSION — PASSED
# ============================================================

def _record_transformation_passed(
    *,
    workflow_id: str,
    dataset_id: str,
    approved_plan: ApprovedTransformationPlan,
    execution: TransformationExecutionReport,
    validation: PostTransformationValidationReport,
    materialization: (
        TransformationArtifactMaterializationReport
    ),
) -> None:

    dataset_ids = [
        dataset_id,
    ]


    for derived_dataset_id in (
        materialization
        .derived_dataset_ids
    ):

        if (
            derived_dataset_id
            not in
            dataset_ids
        ):

            dataset_ids.append(
                derived_dataset_id
            )


    record_optional_stage_signal(
        workflow_id=(
            workflow_id
        ),

        stage=(
            PreparationStage.TRANSFORM
        ),

        required=(
            True
        ),

        completed=(
            True
        ),

        review_required=(
            False
        ),

        blocked=(
            False
        ),

        dataset_ids=(
            dataset_ids
        ),

        evidence_refs=[
            (
                "transformation_approval:"
                f"{approved_plan.rule_version}"
            ),

            (
                "transformation_execution:"
                f"{execution.rule_version}"
            ),

            (
                "transformation_validation:"
                f"{validation.rule_version}"
            ),

            (
                "transformation_materialization:"
                f"{materialization.bridge_version}"
            ),

            (
                "transformation_materialization_kind:"
                f"{materialization.materialization_kind}"
            ),
        ],

        blocking_reasons=[],
    )


# ============================================================
# RESPONSE VIEW
# ============================================================

def _materialization_view(
    report: (
        TransformationArtifactMaterializationReport
    ),
) -> TransformationMaterializationView:

    return (
        TransformationMaterializationView(
            workflow_id=(
                report.workflow_id
            ),

            source_dataset_id=(
                report.source_dataset_id
            ),

            persisted_dataset_ids=list(
                report.persisted_dataset_ids
            ),

            derived_dataset_ids=list(
                report.derived_dataset_ids
            ),

            artifact_count=(
                report.artifact_count
            ),

            source_data_changed=(
                report.source_data_changed
            ),

            materialization_kind=(
                report.materialization_kind
            ),

            bridge_version=(
                report.bridge_version
            ),
        )
    )


# ============================================================
# PLAN ROUTE
# ============================================================

@router.post(
    "/transformation-plan",
    response_model=(
        TransformationPlan
    ),
)
def build_preparation_transformation_plan(
    request: TransformationPlanRequest,
) -> TransformationPlan:
    """
    Build a deterministic Transformation plan directly from
    the current server-owned Preparation artifact.

    No CSV upload.
    No DataFrame mutation.
    No Transformation execution.
    """

    workflow_id = (
        _required_text(
            request.workflow_id,
            field_name="workflow_id",
        )
    )


    dataset_id = (
        _required_text(
            request.dataset_id,
            field_name="dataset_id",
        )
    )


    try:

        artifact = (
            _load_transform_source(
                workflow_id=(
                    workflow_id
                ),

                dataset_id=(
                    dataset_id
                ),
            )
        )


        intents = (
            _parse_transformation_intents(
                raw_intents=(
                    request.intents
                ),

                dataset_id=(
                    dataset_id
                ),

                dataset_filename=(
                    artifact
                    .dataset_filename
                ),
            )
        )


        # ====================================================
        # EXPLICIT "NO TRANSFORMATION"
        # ====================================================

        if not intents:

            plan = (
                _empty_transformation_plan(
                    dataset_id=(
                        dataset_id
                    ),

                    dataset_filename=(
                        artifact
                        .dataset_filename
                    ),
                )
            )


            _record_transformation_plan_stage(
                workflow_id=(
                    workflow_id
                ),

                dataset_id=(
                    dataset_id
                ),

                plan=(
                    plan
                ),
            )


            return (
                plan
            )


        _require_derived_output_ids_available(
            workflow_id=(
                workflow_id
            ),

            source_dataset_id=(
                dataset_id
            ),

            intents=(
                intents
            ),
        )


        plan = (
            plan_transformations(
                dataframe=(
                    artifact
                    .dataframe
                ),

                dataset_id=(
                    dataset_id
                ),

                dataset_filename=(
                    artifact
                    .dataset_filename
                ),

                intents=(
                    intents
                ),
            )
        )


        _record_transformation_plan_stage(
            workflow_id=(
                workflow_id
            ),

            dataset_id=(
                dataset_id
            ),

            plan=(
                plan
            ),
        )


        return (
            plan
        )


    except PreparationSessionNotFoundError as error:

        raise HTTPException(
            status_code=404,

            detail={
                "error":
                    "preparation_session_not_found",

                "message":
                    str(
                        error
                    ),

                "workflow_id":
                    workflow_id,
            },
        ) from error


    except (
        PreparationArtifactWorkflowNotFoundError,
        PreparationArtifactDatasetNotFoundError,
    ) as error:

        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "preparation_artifact_missing",

                "message":
                    str(
                        error
                    ),

                "workflow_id":
                    workflow_id,

                "dataset_id":
                    dataset_id,
            },
        ) from error


    except HTTPException:

        raise


    except (
        ValueError,
        KeyError,
    ) as error:

        raise HTTPException(
            status_code=422,

            detail=str(
                error
            ),
        ) from error


    except TypeError as error:

        raise HTTPException(
            status_code=500,

            detail=(
                "Transformation planning received invalid "
                f"internal state: {error}"
            ),
        ) from error


    except RuntimeError as error:

        raise HTTPException(
            status_code=500,

            detail=(
                "Transformation planning workflow "
                f"synchronization failed: {error}"
            ),
        ) from error


# ============================================================
# APPLY ROUTE
# ============================================================

@router.post(
    "/transformation-apply",
    response_model=(
        TransformationApplyResponse
    ),
)
def apply_preparation_transformation(
    request: TransformationApplyRequest,
) -> TransformationApplyResponse:
    """
    Rebuild and execute Transformation exclusively from the
    server-owned Preparation artifact.

    Trust sequence:

        PreparationSession
            ↓
        PreparationArtifactStore
            ↓
        Transformation Planner
            ↓
        Transformation Approval
            ↓
        Transformation Executor
            ↓
        PostTransformationValidation
            ↓
        Transformation Artifact Store
            ↓
        TRANSFORM = PASSED

    The browser cannot submit:
    - a DataFrame;
    - a TransformationPlan;
    - an ApprovedTransformationPlan;
    - an execution result;
    - a validation result.
    """

    workflow_id = (
        _required_text(
            request.workflow_id,
            field_name="workflow_id",
        )
    )


    dataset_id = (
        _required_text(
            request.dataset_id,
            field_name="dataset_id",
        )
    )


    try:

        artifact = (
            _load_transform_source(
                workflow_id=(
                    workflow_id
                ),

                dataset_id=(
                    dataset_id
                ),
            )
        )


        intents = (
            _parse_transformation_intents(
                raw_intents=(
                    request.intents
                ),

                dataset_id=(
                    dataset_id
                ),

                dataset_filename=(
                    artifact
                    .dataset_filename
                ),
            )
        )


        # ====================================================
        # NO TRANSFORMATION
        # ====================================================

        if not intents:

            plan = (
                _empty_transformation_plan(
                    dataset_id=(
                        dataset_id
                    ),

                    dataset_filename=(
                        artifact
                        .dataset_filename
                    ),
                )
            )


            _record_transformation_plan_stage(
                workflow_id=(
                    workflow_id
                ),

                dataset_id=(
                    dataset_id
                ),

                plan=(
                    plan
                ),
            )


            return (
                TransformationApplyResponse(
                    status=(
                        "skipped"
                    ),

                    plan=(
                        plan
                    ),

                    approved_plan=(
                        None
                    ),

                    execution=(
                        None
                    ),

                    validation=(
                        None
                    ),

                    materialization=(
                        None
                    ),

                    notes=[
                        (
                            "No Transformation was requested. "
                            "The current Preparation artifact "
                            "remains unchanged."
                        ),
                    ],
                )
            )


        _require_derived_output_ids_available(
            workflow_id=(
                workflow_id
            ),

            source_dataset_id=(
                dataset_id
            ),

            intents=(
                intents
            ),
        )


        # ====================================================
        # REBUILD PLAN SERVER-SIDE
        # ====================================================

        plan = (
            plan_transformations(
                dataframe=(
                    artifact
                    .dataframe
                ),

                dataset_id=(
                    dataset_id
                ),

                dataset_filename=(
                    artifact
                    .dataset_filename
                ),

                intents=(
                    intents
                ),
            )
        )


        # ====================================================
        # REBUILD APPROVAL CONTRACT SERVER-SIDE
        # ====================================================

        approved_plan = (
            apply_transformation_approvals(
                plan=(
                    plan
                ),

                commands=(
                    request
                    .approval_commands
                ),
            )
        )


        # ====================================================
        # APPROVAL STILL UNRESOLVED
        # ====================================================

        if not (
            approved_plan
            .ready_for_execution
        ):

            _record_transformation_approval_unresolved(
                workflow_id=(
                    workflow_id
                ),

                dataset_id=(
                    dataset_id
                ),

                approved_plan=(
                    approved_plan
                ),
            )


            return (
                TransformationApplyResponse(
                    status=(
                        "approval_required"
                    ),

                    plan=(
                        plan
                    ),

                    approved_plan=(
                        approved_plan
                    ),

                    execution=(
                        None
                    ),

                    validation=(
                        None
                    ),

                    materialization=(
                        None
                    ),

                    notes=[
                        (
                            "Transformation approval is not "
                            "fully resolved. No DataFrame was "
                            "executed or materialized."
                        ),
                    ],
                )
            )


        # ====================================================
        # ALL REQUESTED STEPS WERE RESOLVED AS NON-EXECUTABLE
        #
        # Example:
        # user rejected every review-required operation.
        # ====================================================

        if (
            approved_plan
            .executable_step_count
            ==
            0
        ):

            _record_transformation_skipped_after_approval(
                workflow_id=(
                    workflow_id
                ),

                dataset_id=(
                    dataset_id
                ),

                approved_plan=(
                    approved_plan
                ),
            )


            return (
                TransformationApplyResponse(
                    status=(
                        "skipped"
                    ),

                    plan=(
                        plan
                    ),

                    approved_plan=(
                        approved_plan
                    ),

                    execution=(
                        None
                    ),

                    validation=(
                        None
                    ),

                    materialization=(
                        None
                    ),

                    notes=[
                        (
                            "All Transformation decisions were "
                            "resolved without executable steps. "
                            "The current artifact remains "
                            "unchanged."
                        ),
                    ],
                )
            )


        # ====================================================
        # EXECUTION
        # ====================================================

        execution_result = (
            execute_transformation_plan(
                dataframe=(
                    artifact
                    .dataframe
                ),

                approved_plan=(
                    approved_plan
                ),

                dataset_id=(
                    dataset_id
                ),

                dataset_filename=(
                    artifact
                    .dataset_filename
                ),
            )
        )


        # ====================================================
        # INDEPENDENT POST-VALIDATION
        # ====================================================

        validation = (
            validate_transformation_execution(
                source_dataframe=(
                    artifact
                    .dataframe
                ),

                execution_result=(
                    execution_result
                ),

                approved_plan=(
                    approved_plan
                ),

                dataset_id=(
                    dataset_id
                ),

                dataset_filename=(
                    artifact
                    .dataset_filename
                ),
            )
        )


        # ====================================================
        # FAILED VALIDATION
        #
        # Never materialize.
        # ====================================================

        if not (
            validation
            .valid_for_downstream
        ):

            _record_transformation_validation_failed(
                workflow_id=(
                    workflow_id
                ),

                dataset_id=(
                    dataset_id
                ),

                approved_plan=(
                    approved_plan
                ),

                execution=(
                    execution_result
                    .report
                ),

                validation=(
                    validation
                ),
            )


            return (
                TransformationApplyResponse(
                    status=(
                        "validation_failed"
                    ),

                    plan=(
                        plan
                    ),

                    approved_plan=(
                        approved_plan
                    ),

                    execution=(
                        execution_result
                        .report
                    ),

                    validation=(
                        validation
                    ),

                    materialization=(
                        None
                    ),

                    notes=[
                        (
                            "Transformation execution was not "
                            "materialized because independent "
                            "post-validation rejected the "
                            "result."
                        ),
                    ],
                )
            )


        # ====================================================
        # MATERIALIZE BEFORE SESSION PASSED
        # ====================================================

        materialization = (
            materialize_transformation_artifacts(
                workflow_id=(
                    workflow_id
                ),

                source_dataframe=(
                    artifact
                    .dataframe
                ),

                approved_plan=(
                    approved_plan
                ),

                execution=(
                    execution_result
                ),

                validation=(
                    validation
                ),
            )
        )


        # ====================================================
        # ONLY NOW MAY TRANSFORM BECOME PASSED
        # ====================================================

        _record_transformation_passed(
            workflow_id=(
                workflow_id
            ),

            dataset_id=(
                dataset_id
            ),

            approved_plan=(
                approved_plan
            ),

            execution=(
                execution_result
                .report
            ),

            validation=(
                validation
            ),

            materialization=(
                materialization
            ),
        )


        return (
            TransformationApplyResponse(
                status=(
                    "ready"
                ),

                plan=(
                    plan
                ),

                approved_plan=(
                    approved_plan
                ),

                execution=(
                    execution_result
                    .report
                ),

                validation=(
                    validation
                ),

                materialization=(
                    _materialization_view(
                        materialization
                    )
                ),

                notes=[
                    (
                        "Transformation was planned, "
                        "authorized, executed and independently "
                        "validated server-side."
                    ),

                    (
                        "Validated Transformation outputs were "
                        "materialized before TRANSFORM was "
                        "marked PASSED."
                    ),

                    (
                        "No CSV or DataFrame was accepted from "
                        "the browser as the Transformation "
                        "execution source."
                    ),
                ],
            )
        )


    except PreparationSessionNotFoundError as error:

        raise HTTPException(
            status_code=404,

            detail={
                "error":
                    "preparation_session_not_found",

                "message":
                    str(
                        error
                    ),

                "workflow_id":
                    workflow_id,
            },
        ) from error


    except (
        PreparationArtifactWorkflowNotFoundError,
        PreparationArtifactDatasetNotFoundError,
    ) as error:

        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "preparation_artifact_missing",

                "message":
                    str(
                        error
                    ),

                "workflow_id":
                    workflow_id,

                "dataset_id":
                    dataset_id,
            },
        ) from error


    except HTTPException:

        raise


    except TransformationExecutionError as error:

        raise HTTPException(
            status_code=422,

            detail={
                "error":
                    "transformation_execution_rejected",

                "message":
                    str(
                        error
                    ),
            },
        ) from error


    except (
        ValueError,
        KeyError,
    ) as error:

        raise HTTPException(
            status_code=422,

            detail=str(
                error
            ),
        ) from error


    except TypeError as error:

        raise HTTPException(
            status_code=500,

            detail=(
                "Transformation workflow received invalid "
                f"internal state: {error}"
            ),
        ) from error


    except RuntimeError as error:

        raise HTTPException(
            status_code=500,

            detail=(
                "Transformation workflow synchronization "
                f"failed: {error}"
            ),
        ) from error