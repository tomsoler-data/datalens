from __future__ import annotations

from typing import (
    Any,
    Dict,
)

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.preparation.preparation_combine_service import (
    PREPARATION_COMBINE_SERVICE_VERSION,
    CombineDiscovery,
    CombineExecution,
    approve_and_execute_next_combine,
    discover_next_combine,
)
from app.preparation.preparation_session import (
    PreparationSessionNotFoundError,
    PreparationSessionView,
    get_preparation_session,
)


# ============================================================
# VERSION
# ============================================================


PREPARATION_COMBINE_API_VERSION = (
    "preparation_combine_api_v0.1"
)


# ============================================================
# ROUTER
# ============================================================


router = APIRouter(
    prefix="/preparation/combine",
    tags=[
        "preparation",
    ],
)


# ============================================================
# STRICT REQUESTS
# ============================================================


class StrictPreparationCombineRequest(
    BaseModel,
):
    model_config = ConfigDict(
        extra="forbid"
    )


class PreparationCombineDiscoveryRequest(
    StrictPreparationCombineRequest,
):
    workflow_id: str = Field(
        min_length=1
    )


class PreparationCombineApprovalRequest(
    StrictPreparationCombineRequest,
):
    workflow_id: str = Field(
        min_length=1
    )

    request_id: str = Field(
        min_length=1
    )

    comment: (
        str
        | None
    ) = None


# ============================================================
# READ MODELS
# ============================================================


class PreparationCombineDiscoveryView(
    BaseModel,
):
    workflow_id: str

    active_dataset_ids: list[
        str
    ]

    reason: str

    has_candidate: bool

    ready_for_approval: bool

    intent: (
        Dict[
            str,
            Any,
        ]
        | None
    )

    plan: (
        Dict[
            str,
            Any,
        ]
        | None
    )

    service_version: str


class PreparationCombineDiscoveryResponse(
    BaseModel,
):
    discovery: PreparationCombineDiscoveryView

    session: PreparationSessionView

    api_version: str = (
        PREPARATION_COMBINE_API_VERSION
    )


class PreparationCombineExecutionResponse(
    BaseModel,
):
    workflow_id: str

    request_id: str

    output_dataset_id: str

    output_dataset_filename: str

    rows: int

    columns: int

    parent_dataset_ids: list[
        str
    ]

    validation: Dict[
        str,
        Any,
    ]

    next_discovery: PreparationCombineDiscoveryView

    session: PreparationSessionView

    service_version: str

    api_version: str = (
        PREPARATION_COMBINE_API_VERSION
    )


# ============================================================
# SERIALIZATION
# ============================================================


def _discovery_view(
    discovery:
        CombineDiscovery,
) -> PreparationCombineDiscoveryView:
    return (
        PreparationCombineDiscoveryView(
            workflow_id=
                discovery.workflow_id,

            active_dataset_ids=
                list(
                    discovery.active_dataset_ids
                ),

            reason=
                discovery.reason,

            has_candidate=
                discovery.has_candidate,

            ready_for_approval=
                discovery.ready_for_approval,

            intent=(
                discovery
                .intent
                .model_dump(
                    mode="json"
                )
                if (
                    discovery.intent
                    is not None
                )
                else None
            ),

            plan=(
                discovery
                .plan
                .model_dump(
                    mode="json"
                )
                if (
                    discovery.plan
                    is not None
                )
                else None
            ),

            service_version=
                discovery.rule_version,
        )
    )


def _execution_response(
    execution:
        CombineExecution,
) -> PreparationCombineExecutionResponse:
    return (
        PreparationCombineExecutionResponse(
            workflow_id=
                execution.workflow_id,

            request_id=
                execution.request_id,

            output_dataset_id=
                execution.output_dataset_id,

            output_dataset_filename=
                execution.output_dataset_filename,

            rows=
                execution.rows,

            columns=
                execution.columns,

            parent_dataset_ids=
                list(
                    execution.parent_dataset_ids
                ),

            validation=(
                execution
                .validation
                .model_dump(
                    mode="json"
                )
            ),

            next_discovery=(
                _discovery_view(
                    execution.next_discovery
                )
            ),

            session=
                execution.session,

            service_version=
                execution.rule_version,

            api_version=
                PREPARATION_COMBINE_API_VERSION,
        )
    )


# ============================================================
# ERROR DETAILS
# ============================================================


def _error_detail(
    *,
    code: str,
    error: Exception,
    workflow_id: str,
) -> Dict[
    str,
    Any,
]:
    return {
        "error":
            code,

        "message":
            str(
                error
            ),

        "workflow_id":
            workflow_id,

        "api_version":
            PREPARATION_COMBINE_API_VERSION,
    }


# ============================================================
# DISCOVER
# ============================================================


@router.post(
    "/discover",
    response_model=
        PreparationCombineDiscoveryResponse,
)
def discover_preparation_combine(
    request:
        PreparationCombineDiscoveryRequest,
) -> PreparationCombineDiscoveryResponse:
    """
    Discover the next deterministic join candidate.

    This is intentionally POST rather than GET because discovery
    synchronizes the server-owned COMBINE stage:

        safe candidate
            -> REVIEW_REQUIRED

        blocked candidate
            -> BLOCKED

        no candidate and no prior combine artifact
            -> SKIPPED

        no candidate after materialized combine work
            -> PASSED

    The client cannot submit:

        join keys
        join type
        cardinality
        dataset lineage
        stage status

    Those are derived by Python from server-owned artifacts.
    """

    try:
        discovery = (
            discover_next_combine(
                request.workflow_id,
                synchronize_stage=
                    True,
            )
        )

        session = (
            get_preparation_session(
                request.workflow_id
            )
        )

        return (
            PreparationCombineDiscoveryResponse(
                discovery=
                    _discovery_view(
                        discovery
                    ),

                session=
                    session,

                api_version=
                    PREPARATION_COMBINE_API_VERSION,
            )
        )


    except PreparationSessionNotFoundError as error:
        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,

            detail=(
                _error_detail(
                    code=
                        "preparation_session_not_found",

                    error=
                        error,

                    workflow_id=
                        request.workflow_id,
                )
            ),
        ) from error


    except ValueError as error:
        raise HTTPException(
            status_code=
                status.HTTP_409_CONFLICT,

            detail=(
                _error_detail(
                    code=
                        "preparation_combine_discovery_rejected",

                    error=
                        error,

                    workflow_id=
                        request.workflow_id,
                )
            ),
        ) from error


    except RuntimeError as error:
        raise HTTPException(
            status_code=
                status.HTTP_409_CONFLICT,

            detail=(
                _error_detail(
                    code=
                        "preparation_combine_discovery_failed",

                    error=
                        error,

                    workflow_id=
                        request.workflow_id,
                )
            ),
        ) from error


# ============================================================
# APPROVE + EXECUTE
# ============================================================


@router.post(
    "/approve",
    response_model=
        PreparationCombineExecutionResponse,
)
def approve_preparation_combine(
    request:
        PreparationCombineApprovalRequest,
) -> PreparationCombineExecutionResponse:
    """
    Approve exactly the current server-derived join candidate.

    Security properties:

    - the browser sends only workflow_id + request_id;
    - request_id must still match the current deterministic plan;
    - the browser cannot alter keys/cardinality/join type;
    - Join Approval authorizes execution;
    - Join Executor performs the merge;
    - Post-Join Validation must pass;
    - only then is the output materialized in Artifact Store.

    After a successful join the service immediately derives the
    next frontier. With three related datasets, the first approval
    can therefore produce another REVIEW_REQUIRED candidate.
    """

    try:
        execution = (
            approve_and_execute_next_combine(
                workflow_id=
                    request.workflow_id,

                request_id=
                    request.request_id,

                actor=
                    "user",

                comment=(
                    request.comment
                    .strip()
                    if (
                        request.comment
                        is not None
                        and
                        request.comment.strip()
                    )
                    else None
                ),
            )
        )

        return (
            _execution_response(
                execution
            )
        )


    except PreparationSessionNotFoundError as error:
        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,

            detail=(
                _error_detail(
                    code=
                        "preparation_session_not_found",

                    error=
                        error,

                    workflow_id=
                        request.workflow_id,
                )
            ),
        ) from error


    except ValueError as error:
        raise HTTPException(
            status_code=
                status.HTTP_409_CONFLICT,

            detail=(
                _error_detail(
                    code=
                        "preparation_combine_approval_rejected",

                    error=
                        error,

                    workflow_id=
                        request.workflow_id,
                )
            ),
        ) from error


    except RuntimeError as error:
        raise HTTPException(
            status_code=
                status.HTTP_409_CONFLICT,

            detail=(
                _error_detail(
                    code=
                        "preparation_combine_execution_failed",

                    error=
                        error,

                    workflow_id=
                        request.workflow_id,
                )
            ),
        ) from error
