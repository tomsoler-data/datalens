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
)

from app.preparation.preparation_orchestrator import (
    PREPARATION_ORCHESTRATOR_RULE_VERSION,
    PreparationOrchestrationInput,
    orchestrate_preparation,
)

from app.preparation.preparation_workflow import (
    PREPARATION_WORKFLOW_RULE_VERSION,
    PreparationWorkflowSnapshot,
)


# ============================================================
# VERSION
# ============================================================


PREPARATION_WORKFLOW_API_VERSION = (
    "preparation_workflow_api_v0.1"
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
# CAPABILITIES RESPONSE
# ============================================================


class PreparationWorkflowCapabilities(
    BaseModel,
):
    api_version: str

    orchestrator_version: str

    workflow_version: str

    stages: list[
        str
    ]

    client_can_set_stage_status_directly: bool

    ready_for_analysis_is_computed: bool

    unknown_fields_are_rejected: bool

    notes: list[
        str
    ]


# ============================================================
# ERROR DETAIL
# ============================================================


def _validation_error_detail(
    exc: Exception,
) -> Dict[
    str,
    Any,
]:
    return {
        "error": (
            "invalid_preparation_workflow"
        ),

        "message": str(
            exc
        ),

        "api_version": (
            PREPARATION_WORKFLOW_API_VERSION
        ),
    }


# ============================================================
# GET CAPABILITIES
# ============================================================


@router.get(
    "/workflow/capabilities",
    response_model=
        PreparationWorkflowCapabilities,
)
def get_preparation_workflow_capabilities(
) -> PreparationWorkflowCapabilities:
    """
    Describe the Preparation Workflow API contract.

    This endpoint is read-only.
    """

    return (
        PreparationWorkflowCapabilities(
            api_version=(
                PREPARATION_WORKFLOW_API_VERSION
            ),

            orchestrator_version=(
                PREPARATION_ORCHESTRATOR_RULE_VERSION
            ),

            workflow_version=(
                PREPARATION_WORKFLOW_RULE_VERSION
            ),

            stages=[
                "import",
                "understand",
                "quality",
                "clean",
                "transform",
                "combine",
                "validate",
            ],

            client_can_set_stage_status_directly=
                False,

            ready_for_analysis_is_computed=
                True,

            unknown_fields_are_rejected=
                True,

            notes=[
                (
                    "Preparation stage statuses are "
                    "derived by the backend."
                ),

                (
                    "The client provides structured "
                    "engine signals, not PASSED, "
                    "BLOCKED or REVIEW_REQUIRED "
                    "stage statuses."
                ),

                (
                    "Unknown request fields are "
                    "rejected by Pydantic."
                ),

                (
                    "READY FOR ANALYSIS is computed "
                    "by Preparation Workflow."
                ),

                (
                    "This API never cleans, transforms "
                    "or joins DataFrames itself."
                ),
            ],
        )
    )


# ============================================================
# POST EVALUATE
# ============================================================


@router.post(
    "/workflow/evaluate",
    response_model=
        PreparationWorkflowSnapshot,
)
def evaluate_preparation(
    request: PreparationOrchestrationInput,
) -> PreparationWorkflowSnapshot:
    """
    Evaluate preparation readiness from structured signals.

    The caller does NOT provide stage statuses directly.

    Example accepted facts:

        required = True
        completed = False
        review_required = True

    Backend-derived result:

        TRANSFORM = REVIEW_REQUIRED

    Unknown request fields are rejected by Pydantic before
    this function is called.

    This endpoint performs no DataFrame mutation.
    """

    try:
        return (
            orchestrate_preparation(
                request
            )
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_CONTENT
            ),

            detail=(
                _validation_error_detail(
                    exc
                )
            ),
        ) from exc