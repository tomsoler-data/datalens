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

from app.preparation.join_approval import (
    ApprovedJoinPlan,
    JoinApprovalCommand,
    apply_join_approvals,
)

from app.preparation.join_artifacts import (
    JoinArtifactMaterializationReport,
    materialize_join_artifacts,
)

from app.preparation.join_contracts import (
    JoinIntent,
    JoinPlan,
)

from app.preparation.join_executor import (
    JoinExecutionError,
    JoinExecutionReport,
    execute_join_plan,
)

from app.preparation.join_planner import (
    plan_joins,
)

from app.preparation.post_join_validation import (
    PostJoinValidationReport,
    validate_join_execution,
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


# ============================================================
# VERSION
# ============================================================

PREPARATION_COMBINATION_API_VERSION = (
    "preparation_combination_api_v0.1"
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

class CombinationPlanRequest(
    BaseModel,
):
    """
    Browser-facing COMBINE planning request.

    The browser supplies only:
    - workflow identity;
    - structured JOIN intents.

    Actual DataFrames and source filenames are resolved
    server-side from PreparationSession +
    PreparationArtifactStore.
    """

    workflow_id: str = Field(
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


class CombinationApplyRequest(
    BaseModel,
):
    workflow_id: str = Field(
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
        JoinApprovalCommand
    ] = Field(
        default_factory=list,
    )


# ============================================================
# RESPONSE MODELS
# ============================================================

class JoinMaterializationView(
    BaseModel,
):
    workflow_id: str

    output_dataset_ids: list[
        str
    ]

    artifact_count: int

    bridge_version: str


class CombinationApplyResponse(
    BaseModel,
):
    status: str

    plan: JoinPlan

    approved_plan: (
        ApprovedJoinPlan
        | None
    ) = None

    execution: (
        JoinExecutionReport
        | None
    ) = None

    validation: (
        PostJoinValidationReport
        | None
    ) = None

    materialization: (
        JoinMaterializationView
        | None
    ) = None

    notes: list[
        str
    ] = Field(
        default_factory=list,
    )

    api_version: str = (
        PREPARATION_COMBINATION_API_VERSION
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


    return (
        normalized
    )


# ============================================================
# SESSION STAGE
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


# ============================================================
# COMBINE PRECONDITION
# ============================================================

def _require_combination_precondition(
    *,
    workflow_id: str,
):
    """
    COMBINE may start only after CLEAN and TRANSFORM have been resolved.

    TRANSFORM may be:
    - PASSED;
    - SKIPPED.

    REVIEW_REQUIRED / BLOCKED / NOT_STARTED may not be
    bypassed through the COMBINE API.
    """

    session = (
        get_preparation_session(
            workflow_id
        )
    )


    # COMBINE_CLEAN_GUARD_V0_1
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
                        "before COMBINE can begin."
                    ),

                "workflow_id":
                    workflow_id,

                "clean_status":
                    clean_stage.status.value,
            },
        )


    transform_stage = (
        _stage_record(
            session=(
                session
            ),

            stage=(
                PreparationStage.TRANSFORM
            ),
        )
    )


    if (
        transform_stage.status
        not in {
            PreparationStageStatus.PASSED,
            PreparationStageStatus.SKIPPED,
        }
    ):

        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "transform_stage_not_resolved",

                "message":
                    (
                        "TRANSFORM must be PASSED or SKIPPED "
                        "before COMBINE can begin."
                    ),

                "workflow_id":
                    workflow_id,

                "transform_status":
                    transform_stage.status.value,
            },
        )


    return (
        session
    )


# ============================================================
# AUTHORIZED INPUT DATASETS
# ============================================================

def _authorized_combination_dataset_ids(
    *,
    session,
) -> set[
    str
]:
    """
    Inputs authorized for COMBINE are:

    - server-selected source analytical datasets;
    - datasets explicitly present in the resolved TRANSFORM
      stage, including validated derived outputs.

    This allows a validated aggregate / transformed artifact
    to participate in COMBINE without trusting an arbitrary
    dataset_id sent by the browser.
    """

    output = set(
        session
        .selected_analysis_dataset_ids
    )


    transform_stage = (
        _stage_record(
            session=(
                session
            ),

            stage=(
                PreparationStage.TRANSFORM
            ),
        )
    )


    for dataset_id in (
        transform_stage.dataset_ids
    ):

        normalized = (
            str(
                dataset_id
            )
            .strip()
        )


        if normalized:

            output.add(
                normalized
            )


    return (
        output
    )


# ============================================================
# SOURCE ARTIFACT
# ============================================================

def _load_combination_source_artifact(
    *,
    workflow_id: str,
    dataset_id: str,
) -> PreparationDatasetArtifact:

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
            "transform",
        }
    ):

        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "combination_source_stage_invalid",

                "message":
                    (
                        "COMBINE accepts only current SOURCE, "
                        "CLEAN or TRANSFORM artifacts in v0.1."
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
# INTENT PARSING + SERVER DATASET RESOLUTION
# ============================================================

def _build_combination_context(
    *,
    workflow_id: str,
    raw_intents: list[
        dict[
            str,
            Any,
        ]
    ],
):
    """
    Resolve all JOIN source datasets exclusively from the
    server.

    Browser-provided source filenames may never redirect JOIN
    to another artifact.

    Returns:

        session,
        source_datasets,
        parsed_intents
    """

    session = (
        _require_combination_precondition(
            workflow_id=(
                workflow_id
            )
        )
    )


    if not raw_intents:

        return (
            session,
            {},
            [],
        )


    authorized_dataset_ids = (
        _authorized_combination_dataset_ids(
            session=(
                session
            )
        )
    )


    artifacts: dict[
        str,
        PreparationDatasetArtifact,
    ] = {}


    parsed_intents: list[
        JoinIntent
    ] = []


    seen_request_ids: set[
        str
    ] = set()


    seen_output_dataset_ids: set[
        str
    ] = set()


    # ========================================================
    # EXISTING SERVER ARTIFACT IDS
    # ========================================================

    existing_dataset_ids = {
        artifact.dataset_id

        for artifact in
        list_preparation_artifacts(
            workflow_id=(
                workflow_id
            )
        )
    }


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
                "JOIN intent must be an object. "
                f"index={index}"
            )


        # ====================================================
        # SOURCE IDENTITIES
        # ====================================================

        left_dataset_id = (
            _required_text(
                str(
                    raw_intent.get(
                        "left_dataset_id"
                    )
                    or
                    ""
                ),

                field_name=(
                    "left_dataset_id"
                ),
            )
        )


        right_dataset_id = (
            _required_text(
                str(
                    raw_intent.get(
                        "right_dataset_id"
                    )
                    or
                    ""
                ),

                field_name=(
                    "right_dataset_id"
                ),
            )
        )


        for dataset_id in {
            left_dataset_id,
            right_dataset_id,
        }:

            if (
                dataset_id
                not in
                authorized_dataset_ids
            ):

                raise HTTPException(
                    status_code=403,

                    detail={
                        "error":
                            "combination_dataset_not_authorized",

                        "message":
                            (
                                "JOIN source dataset is not "
                                "authorized by the server-owned "
                                "Preparation workflow."
                            ),

                        "workflow_id":
                            workflow_id,

                        "dataset_id":
                            dataset_id,
                    },
                )


            if (
                dataset_id
                not in artifacts
            ):

                artifacts[
                    dataset_id
                ] = (
                    _load_combination_source_artifact(
                        workflow_id=(
                            workflow_id
                        ),

                        dataset_id=(
                            dataset_id
                        ),
                    )
                )


        left_artifact = (
            artifacts[
                left_dataset_id
            ]
        )


        right_artifact = (
            artifacts[
                right_dataset_id
            ]
        )


        # ====================================================
        # BROWSER FILENAME RECONCILIATION
        # ====================================================

        supplied_left_filename = (
            raw_intent.get(
                "left_dataset_filename"
            )
        )


        if (
            supplied_left_filename
            is not None
            and
            str(
                supplied_left_filename
            ).strip()
            !=
            left_artifact.dataset_filename
        ):

            raise ValueError(
                "JOIN intent left_dataset_filename does not "
                "match the server-owned artifact. "
                f"index={index}"
            )


        supplied_right_filename = (
            raw_intent.get(
                "right_dataset_filename"
            )
        )


        if (
            supplied_right_filename
            is not None
            and
            str(
                supplied_right_filename
            ).strip()
            !=
            right_artifact.dataset_filename
        ):

            raise ValueError(
                "JOIN intent right_dataset_filename does not "
                "match the server-owned artifact. "
                f"index={index}"
            )


        normalized_payload = dict(
            raw_intent
        )


        normalized_payload[
            "left_dataset_id"
        ] = (
            left_dataset_id
        )


        normalized_payload[
            "left_dataset_filename"
        ] = (
            left_artifact
            .dataset_filename
        )


        normalized_payload[
            "right_dataset_id"
        ] = (
            right_dataset_id
        )


        normalized_payload[
            "right_dataset_filename"
        ] = (
            right_artifact
            .dataset_filename
        )


        intent = (
            JoinIntent.model_validate(
                normalized_payload
            )
        )


        # ====================================================
        # REQUEST ID
        # ====================================================

        request_id = (
            _required_text(
                intent.request_id,

                field_name=(
                    "JoinIntent.request_id"
                ),
            )
        )


        if (
            request_id
            in seen_request_ids
        ):

            raise ValueError(
                "Duplicate JOIN request_id="
                f"{request_id}"
            )


        seen_request_ids.add(
            request_id
        )


        # ====================================================
        # OUTPUT ID
        # ====================================================

        output_dataset_id = (
            _required_text(
                intent.output_dataset_id,

                field_name=(
                    "JoinIntent.output_dataset_id"
                ),
            )
        )


        _required_text(
            intent.output_dataset_filename,

            field_name=(
                "JoinIntent.output_dataset_filename"
            ),
        )


        if (
            output_dataset_id
            in seen_output_dataset_ids
        ):

            raise ValueError(
                "Duplicate JOIN output_dataset_id="
                f"{output_dataset_id}"
            )


        if (
            output_dataset_id
            in existing_dataset_ids
        ):

            raise ValueError(
                "JOIN output dataset_id already exists in "
                "the Preparation Artifact Store: "
                f"{output_dataset_id}"
            )


        seen_output_dataset_ids.add(
            output_dataset_id
        )


        parsed_intents.append(
            intent
        )


    source_datasets = {
        dataset_id:
            artifact.dataframe

        for (
            dataset_id,
            artifact,
        ) in artifacts.items()
    }


    return (
        session,
        source_datasets,
        parsed_intents,
    )


# ============================================================
# EMPTY PLAN
# ============================================================

def _empty_join_plan() -> JoinPlan:

    return (
        JoinPlan(
            request_count=(
                0
            ),

            join_count=(
                0
            ),

            review_required_count=(
                0
            ),

            blocked_count=(
                0
            ),

            ready_for_approval=(
                False
            ),

            joins=[],

            notes=[
                (
                    "No JOIN intent was requested. "
                    "COMBINE is skipped."
                ),
            ],
        )
    )


# ============================================================
# PLAN DATASET SCOPE
# ============================================================

def _plan_dataset_ids(
    plan: JoinPlan,
) -> list[
    str
]:

    output: list[
        str
    ] = []


    seen: set[
        str
    ] = set()


    for join in (
        plan.joins
    ):

        for dataset_id in (
            join.left_dataset_id,
            join.right_dataset_id,
            join.output_dataset_id,
        ):

            normalized = (
                str(
                    dataset_id
                )
                .strip()
            )


            if (
                normalized
                and
                normalized
                not in seen
            ):

                seen.add(
                    normalized
                )

                output.append(
                    normalized
                )


    return (
        output
    )


def _approved_plan_dataset_ids(
    approved_plan: ApprovedJoinPlan,
) -> list[
    str
]:

    output: list[
        str
    ] = []


    seen: set[
        str
    ] = set()


    for join in (
        approved_plan.joins
    ):

        for dataset_id in (
            join.left_dataset_id,
            join.right_dataset_id,
            join.output_dataset_id,
        ):

            normalized = (
                str(
                    dataset_id
                )
                .strip()
            )


            if (
                normalized
                and
                normalized
                not in seen
            ):

                seen.add(
                    normalized
                )

                output.append(
                    normalized
                )


    return (
        output
    )


def _approved_source_dataset_ids(
    approved_plan: ApprovedJoinPlan,
) -> list[
    str
]:

    output: list[
        str
    ] = []


    seen: set[
        str
    ] = set()


    for join in (
        approved_plan.joins
    ):

        for dataset_id in (
            join.left_dataset_id,
            join.right_dataset_id,
        ):

            normalized = (
                str(
                    dataset_id
                )
                .strip()
            )


            if (
                normalized
                and
                normalized
                not in seen
            ):

                seen.add(
                    normalized
                )

                output.append(
                    normalized
                )


    return (
        output
    )


# ============================================================
# SESSION — PLAN
# ============================================================

def _record_combination_plan_stage(
    *,
    workflow_id: str,
    plan: JoinPlan,
    fallback_dataset_ids: list[
        str
    ],
) -> None:

    dataset_ids = (
        _plan_dataset_ids(
            plan
        )
        or
        list(
            fallback_dataset_ids
        )
    )


    evidence_refs = [
        (
            "join_plan:"
            f"{plan.rule_version}"
        ),

        (
            "join_requests:"
            f"{plan.request_count}"
        ),

        (
            "join_count:"
            f"{plan.join_count}"
        ),

        (
            "join_review_required:"
            f"{plan.review_required_count}"
        ),

        (
            "join_blocked:"
            f"{plan.blocked_count}"
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
                PreparationStage.COMBINE
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

            dataset_ids=(
                dataset_ids
            ),

            evidence_refs=(
                evidence_refs
            ),

            blocking_reasons=[],
        )


        return


    # ========================================================
    # DETERMINISTICALLY BLOCKED JOIN
    # ========================================================

    if (
        plan.blocked_count
        >
        0
        or
        not plan.ready_for_approval
    ):

        record_optional_stage_signal(
            workflow_id=(
                workflow_id
            ),

            stage=(
                PreparationStage.COMBINE
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

            dataset_ids=(
                dataset_ids
            ),

            evidence_refs=(
                evidence_refs
            ),

            blocking_reasons=[
                (
                    "One or more JOIN proposals were "
                    "deterministically blocked."
                ),
            ],
        )


        return


    # ========================================================
    # EVERY NON-BLOCKED JOIN REQUIRES HUMAN APPROVAL
    # ========================================================

    record_optional_stage_signal(
        workflow_id=(
            workflow_id
        ),

        stage=(
            PreparationStage.COMBINE
        ),

        required=(
            True
        ),

        completed=(
            False
        ),

        review_required=(
            True
        ),

        blocked=(
            False
        ),

        dataset_ids=(
            dataset_ids
        ),

        evidence_refs=(
            evidence_refs
        ),

        blocking_reasons=[
            (
                "JOIN execution requires explicit analyst "
                "approval."
            ),
        ],
    )


# ============================================================
# SESSION — UNRESOLVED APPROVAL
# ============================================================

def _record_combination_approval_unresolved(
    *,
    workflow_id: str,
    approved_plan: ApprovedJoinPlan,
) -> None:

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
                "One or more JOIN decisions are still "
                "pending."
            )
        )


    if (
        approved_plan.deferred_count
        >
        0
    ):

        reasons.append(
            (
                "One or more JOIN decisions were deferred."
            )
        )


    record_optional_stage_signal(
        workflow_id=(
            workflow_id
        ),

        stage=(
            PreparationStage.COMBINE
        ),

        required=(
            True
        ),

        completed=(
            False
        ),

        review_required=(
            True
        ),

        blocked=(
            False
        ),

        dataset_ids=(
            _approved_plan_dataset_ids(
                approved_plan
            )
        ),

        evidence_refs=[
            (
                "join_approval:"
                f"{approved_plan.rule_version}"
            ),

            (
                "join_pending:"
                f"{approved_plan.pending_count}"
            ),

            (
                "join_deferred:"
                f"{approved_plan.deferred_count}"
            ),
        ],

        blocking_reasons=(
            reasons
            or
            [
                (
                    "JOIN approval is not fully resolved."
                ),
            ]
        ),
    )


# ============================================================
# SESSION — ALL REJECTED
# ============================================================

def _record_combination_skipped_after_approval(
    *,
    workflow_id: str,
    approved_plan: ApprovedJoinPlan,
) -> None:

    record_optional_stage_signal(
        workflow_id=(
            workflow_id
        ),

        stage=(
            PreparationStage.COMBINE
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

        dataset_ids=(
            _approved_source_dataset_ids(
                approved_plan
            )
        ),

        evidence_refs=[
            (
                "join_approval:"
                f"{approved_plan.rule_version}"
            ),

            (
                "join_executable_count:0"
            ),

            (
                "join_rejected:"
                f"{approved_plan.rejected_count}"
            ),
        ],

        blocking_reasons=[],
    )


# ============================================================
# SESSION — VALIDATION FAILED
# ============================================================

def _record_combination_validation_failed(
    *,
    workflow_id: str,
    approved_plan: ApprovedJoinPlan,
    execution: JoinExecutionReport,
    validation: PostJoinValidationReport,
) -> None:

    record_optional_stage_signal(
        workflow_id=(
            workflow_id
        ),

        stage=(
            PreparationStage.COMBINE
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

        dataset_ids=(
            _approved_plan_dataset_ids(
                approved_plan
            )
        ),

        evidence_refs=[
            (
                "join_approval:"
                f"{approved_plan.rule_version}"
            ),

            (
                "join_execution:"
                f"{execution.rule_version}"
            ),

            (
                "join_validation:"
                f"{validation.rule_version}"
            ),

            (
                "join_validation_failed_checks:"
                f"{validation.failed_check_count}"
            ),
        ],

        blocking_reasons=[
            (
                "Post-join validation did not authorize "
                "the JOIN result for downstream use."
            ),
        ],
    )


# ============================================================
# SESSION — PASSED
# ============================================================

def _record_combination_passed(
    *,
    workflow_id: str,
    approved_plan: ApprovedJoinPlan,
    execution: JoinExecutionReport,
    validation: PostJoinValidationReport,
    materialization: JoinArtifactMaterializationReport,
) -> None:

    record_optional_stage_signal(
        workflow_id=(
            workflow_id
        ),

        stage=(
            PreparationStage.COMBINE
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

        dataset_ids=list(
            materialization
            .output_dataset_ids
        ),

        evidence_refs=[
            (
                "join_approval:"
                f"{approved_plan.rule_version}"
            ),

            (
                "join_execution:"
                f"{execution.rule_version}"
            ),

            (
                "join_validation:"
                f"{validation.rule_version}"
            ),

            (
                "join_materialization:"
                f"{materialization.bridge_version}"
            ),

            (
                "join_output_count:"
                f"{materialization.artifact_count}"
            ),
        ],

        blocking_reasons=[],
    )


# ============================================================
# MATERIALIZATION VIEW
# ============================================================

def _materialization_view(
    report: JoinArtifactMaterializationReport,
) -> JoinMaterializationView:

    return (
        JoinMaterializationView(
            workflow_id=(
                report.workflow_id
            ),

            output_dataset_ids=list(
                report.output_dataset_ids
            ),

            artifact_count=(
                report.artifact_count
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
    "/combination-plan",
    response_model=(
        JoinPlan
    ),
)
def build_preparation_combination_plan(
    request: CombinationPlanRequest,
) -> JoinPlan:
    """
    Inspect JOIN proposals against current server-owned
    Preparation artifacts.

    No CSV upload.
    No merge.
    No artifact mutation.
    """

    workflow_id = (
        _required_text(
            request.workflow_id,
            field_name="workflow_id",
        )
    )


    try:

        (
            session,
            source_datasets,
            intents,
        ) = (
            _build_combination_context(
                workflow_id=(
                    workflow_id
                ),

                raw_intents=(
                    request.intents
                ),
            )
        )


        # ====================================================
        # NO JOIN
        # ====================================================

        if not intents:

            plan = (
                _empty_join_plan()
            )


            _record_combination_plan_stage(
                workflow_id=(
                    workflow_id
                ),

                plan=(
                    plan
                ),

                fallback_dataset_ids=list(
                    session
                    .selected_analysis_dataset_ids
                ),
            )


            return (
                plan
            )


        # ====================================================
        # DETERMINISTIC JOIN PLANNER
        # ====================================================

        plan = (
            plan_joins(
                datasets=(
                    source_datasets
                ),

                intents=(
                    intents
                ),
            )
        )


        _record_combination_plan_stage(
            workflow_id=(
                workflow_id
            ),

            plan=(
                plan
            ),

            fallback_dataset_ids=list(
                source_datasets.keys()
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
                "COMBINE planning received invalid internal "
                f"state: {error}"
            ),
        ) from error


    except RuntimeError as error:

        raise HTTPException(
            status_code=500,

            detail=(
                "COMBINE planning workflow synchronization "
                f"failed: {error}"
            ),
        ) from error


# ============================================================
# APPLY ROUTE
# ============================================================

@router.post(
    "/combination-apply",
    response_model=(
        CombinationApplyResponse
    ),
)
def apply_preparation_combination(
    request: CombinationApplyRequest,
) -> CombinationApplyResponse:
    """
    Rebuild and execute JOINs from current server-owned
    Preparation artifacts.

    Trust sequence:

        PreparationSession
            ↓
        PreparationArtifactStore
            ↓
        Join Planner
            ↓
        explicit human approval
            ↓
        Join Executor
            ↓
        independent Post-Join Validation
            ↓
        Join Artifact Store
            ↓
        COMBINE = PASSED

    Browser cannot submit DataFrames, JoinPlan,
    ApprovedJoinPlan, execution results or validation results.
    """

    workflow_id = (
        _required_text(
            request.workflow_id,
            field_name="workflow_id",
        )
    )


    try:

        (
            session,
            source_datasets,
            intents,
        ) = (
            _build_combination_context(
                workflow_id=(
                    workflow_id
                ),

                raw_intents=(
                    request.intents
                ),
            )
        )


        # ====================================================
        # NO JOIN
        # ====================================================

        if not intents:

            plan = (
                _empty_join_plan()
            )


            _record_combination_plan_stage(
                workflow_id=(
                    workflow_id
                ),

                plan=(
                    plan
                ),

                fallback_dataset_ids=list(
                    session
                    .selected_analysis_dataset_ids
                ),
            )


            return (
                CombinationApplyResponse(
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
                            "No JOIN was requested. The "
                            "current Preparation artifacts "
                            "remain unchanged."
                        ),
                    ],
                )
            )


        # ====================================================
        # REBUILD PLAN SERVER-SIDE
        # ====================================================

        plan = (
            plan_joins(
                datasets=(
                    source_datasets
                ),

                intents=(
                    intents
                ),
            )
        )


        # ====================================================
        # PLANNER BLOCKED
        #
        # Never enter approval.
        # ====================================================

        if (
            not plan.ready_for_approval
            or
            plan.blocked_count
            >
            0
        ):

            _record_combination_plan_stage(
                workflow_id=(
                    workflow_id
                ),

                plan=(
                    plan
                ),

                fallback_dataset_ids=list(
                    source_datasets.keys()
                ),
            )


            return (
                CombinationApplyResponse(
                    status=(
                        "blocked"
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
                            "JOIN Planner blocked at least "
                            "one proposed combination. "
                            "No approval or execution "
                            "occurred."
                        ),
                    ],
                )
            )


        # ====================================================
        # HUMAN APPROVAL CONTRACT
        # ====================================================

        approved_plan = (
            apply_join_approvals(
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
        # PENDING / DEFERRED
        # ====================================================

        if not (
            approved_plan
            .ready_for_execution
        ):

            _record_combination_approval_unresolved(
                workflow_id=(
                    workflow_id
                ),

                approved_plan=(
                    approved_plan
                ),
            )


            return (
                CombinationApplyResponse(
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
                            "JOIN approval is not fully "
                            "resolved. No merge or artifact "
                            "materialization occurred."
                        ),
                    ],
                )
            )


        # ====================================================
        # ALL JOINS RESOLVED AS REJECTED
        # ====================================================

        if (
            approved_plan
            .executable_join_count
            ==
            0
        ):

            _record_combination_skipped_after_approval(
                workflow_id=(
                    workflow_id
                ),

                approved_plan=(
                    approved_plan
                ),
            )


            return (
                CombinationApplyResponse(
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
                            "All JOIN decisions were resolved "
                            "without an executable JOIN. "
                            "Source artifacts remain "
                            "unchanged."
                        ),
                    ],
                )
            )


        # ====================================================
        # EXECUTION
        # ====================================================

        execution_result = (
            execute_join_plan(
                datasets=(
                    source_datasets
                ),

                approved_plan=(
                    approved_plan
                ),
            )
        )


        # ====================================================
        # INDEPENDENT POST-JOIN VALIDATION
        # ====================================================

        validation = (
            validate_join_execution(
                source_datasets=(
                    source_datasets
                ),

                approved_plan=(
                    approved_plan
                ),

                execution_result=(
                    execution_result
                ),
            )
        )


        # ====================================================
        # INVALID OUTPUT
        #
        # Never materialize.
        # ====================================================

        if not (
            validation
            .valid_for_downstream
        ):

            _record_combination_validation_failed(
                workflow_id=(
                    workflow_id
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
                CombinationApplyResponse(
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
                            "JOIN execution was rejected by "
                            "independent Post-Join Validation "
                            "and was not materialized."
                        ),
                    ],
                )
            )


        # ====================================================
        # MATERIALIZE BEFORE COMBINE = PASSED
        # ====================================================

        materialization = (
            materialize_join_artifacts(
                workflow_id=(
                    workflow_id
                ),

                source_datasets=(
                    source_datasets
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
        # ONLY NOW MAY COMBINE BECOME PASSED
        # ====================================================

        _record_combination_passed(
            workflow_id=(
                workflow_id
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
            CombinationApplyResponse(
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
                        "JOIN was planned, explicitly "
                        "approved, executed and independently "
                        "validated server-side."
                    ),

                    (
                        "Validated JOIN outputs were "
                        "materialized before COMBINE was "
                        "marked PASSED."
                    ),

                    (
                        "No CSV or DataFrame was accepted "
                        "from the browser as a JOIN source."
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
            },
        ) from error


    except HTTPException:

        raise


    except JoinExecutionError as error:

        raise HTTPException(
            status_code=422,

            detail={
                "error":
                    "join_execution_rejected",

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
                "COMBINE workflow received invalid internal "
                f"state: {error}"
            ),
        ) from error


    except RuntimeError as error:

        raise HTTPException(
            status_code=500,

            detail=(
                "COMBINE workflow synchronization failed: "
                f"{error}"
            ),
        ) from error