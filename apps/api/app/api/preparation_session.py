from __future__ import annotations


from typing import (
    Any,
    Dict,
    List,
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


from app.preparation.analysis_output_selection_commit import (
    commit_analysis_output_selection,
)


from app.preparation.preparation_artifact_store import (
    list_preparation_artifacts,
)


from app.preparation.preparation_session import (
    PREPARATION_SESSION_RULE_VERSION,
    PreparationSessionNotFoundError,
    PreparationSessionRevisionConflictError,
    PreparationSessionView,
    create_preparation_session,
    get_preparation_session,
)


# ============================================================
# VERSION
# ============================================================


PREPARATION_SESSION_API_VERSION = (
    "preparation_session_api_v0.2"
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
# STRICT REQUEST MODEL
# ============================================================


class StrictPreparationSessionRequest(
    BaseModel,
):
    model_config = ConfigDict(
        extra="forbid"
    )


class PreparationSessionCreateRequest(
    StrictPreparationSessionRequest,
):
    selected_analysis_dataset_ids: List[
        str
    ] = Field(
        min_length=
            1
    )


class PreparationAnalysisOutputSelectionRequest(
    StrictPreparationSessionRequest,
):
    """
    Public request for final analytical-output selection.

    The browser is allowed to choose among server-owned
    materialized Preparation artifacts.

    The browser is NOT allowed to provide:

    - stage statuses;
    - validation state;
    - READY FOR ANALYSIS;
    - Artifact Store contents;
    - lineage;
    - session revision.

    commit_analysis_output_selection() remains the server-side
    authority for validating and committing the choice.
    """

    workflow_id: str = Field(
        min_length=
            1
    )

    dataset_ids: List[
        str
    ] = Field(
        min_length=
            1
    )


# ============================================================
# CAPABILITIES
# ============================================================


class PreparationSessionCapabilities(
    BaseModel,
):
    api_version: str

    session_version: str

    storage: str

    persistent: bool

    client_can_set_workflow_id: bool

    client_can_update_stage_status: bool

    client_can_set_ready_for_analysis: bool

    client_can_select_analysis_output: bool

    notes: List[
        str
    ]


# ============================================================
# ANALYSIS OUTPUT READ MODEL
# ============================================================


class PreparationAnalysisOutputCandidate(
    BaseModel,
):
    dataset_id: str

    dataset_filename: str

    stage: str

    rows: int

    columns: int

    parent_dataset_ids: List[
        str
    ]

    evidence_refs: List[
        str
    ]

    is_root_dataset: bool

    is_selected: bool

    is_validated: bool


class PreparationAnalysisOutputCandidatesResponse(
    BaseModel,
):
    workflow_id: str

    revision: int

    selected_analysis_dataset_ids: List[
        str
    ]

    analysis_output_dataset_ids: List[
        str
    ]

    validated_analysis_dataset_ids: List[
        str
    ]

    locked: bool

    candidate_count: int

    candidates: List[
        PreparationAnalysisOutputCandidate
    ]

    api_version: str = (
        PREPARATION_SESSION_API_VERSION
    )


# ============================================================
# ERROR HELPERS
# ============================================================


def _invalid_session_detail(
    exc: Exception,
) -> Dict[
    str,
    Any,
]:
    return {
        "error": (
            "invalid_preparation_session"
        ),

        "message": str(
            exc
        ),

        "api_version": (
            PREPARATION_SESSION_API_VERSION
        ),
    }


def _not_found_detail(
    exc: Exception,
) -> Dict[
    str,
    Any,
]:
    return {
        "error": (
            "preparation_session_not_found"
        ),

        "message": str(
            exc
        ),

        "api_version": (
            PREPARATION_SESSION_API_VERSION
        ),
    }


def _analysis_output_conflict_detail(
    *,
    exc: Exception,
    workflow_id: str,
) -> Dict[
    str,
    Any,
]:
    return {
        "error": (
            "analysis_output_selection_rejected"
        ),

        "message": str(
            exc
        ),

        "workflow_id": (
            workflow_id
        ),

        "api_version": (
            PREPARATION_SESSION_API_VERSION
        ),
    }


def _revision_conflict_detail(
    *,
    exc: Exception,
    workflow_id: str,
) -> Dict[
    str,
    Any,
]:
    return {
        "error": (
            "preparation_session_revision_conflict"
        ),

        "message": str(
            exc
        ),

        "workflow_id": (
            workflow_id
        ),

        "retryable": True,

        "api_version": (
            PREPARATION_SESSION_API_VERSION
        ),
    }


# ============================================================
# VALIDATE LOCK
# ============================================================


def _analysis_output_selection_locked(
    session: PreparationSessionView,
) -> bool:
    """
    Final output selection becomes immutable after the
    server-owned VALIDATE stage has PASSED.
    """

    for stage in (
        session
        .snapshot
        .stages
    ):
        if (
            stage.stage.value
            ==
            "validate"
            and
            stage.status.value
            ==
            "passed"
        ):
            return True


    return False


# ============================================================
# CAPABILITIES
# ============================================================


@router.get(
    "/sessions/capabilities",
    response_model=
        PreparationSessionCapabilities,
)
def get_preparation_session_capabilities(
) -> PreparationSessionCapabilities:
    return (
        PreparationSessionCapabilities(
            api_version=
                PREPARATION_SESSION_API_VERSION,

            session_version=
                PREPARATION_SESSION_RULE_VERSION,

            storage=
                "in_memory",

            persistent=
                False,

            client_can_set_workflow_id=
                False,

            client_can_update_stage_status=
                False,

            client_can_set_ready_for_analysis=
                False,

            client_can_select_analysis_output=
                True,

            notes=[
                (
                    "Preparation Session state is "
                    "stored in the FastAPI process."
                ),

                (
                    "Sessions are lost when the "
                    "backend process restarts."
                ),

                (
                    "workflow_id values are generated "
                    "by the backend."
                ),

                (
                    "No public endpoint allows the "
                    "client to mark preparation stages "
                    "as PASSED."
                ),

                (
                    "No public endpoint allows the "
                    "client to set READY FOR ANALYSIS."
                ),

                (
                    "Stage updates are reserved for "
                    "backend preparation engines."
                ),

                (
                    "The client may select final "
                    "analytical outputs only among "
                    "server-owned Preparation artifacts."
                ),

                (
                    "Final output selection is "
                    "revalidated server-side against "
                    "Artifact Store lineage before "
                    "being committed."
                ),

                (
                    "A PASSED VALIDATE stage locks "
                    "the final analytical-output scope."
                ),
            ],
        )
    )


# ============================================================
# CREATE SESSION
# ============================================================


@router.post(
    "/sessions",
    response_model=
        PreparationSessionView,
    status_code=
        status.HTTP_201_CREATED,
)
def create_session(
    request:
        PreparationSessionCreateRequest,
) -> PreparationSessionView:
    """
    Create a server-owned preparation session.

    The client supplies only the Preparation-root dataset IDs.

    workflow_id is generated by the backend.

    The client cannot submit stage statuses or final
    readiness.
    """

    try:
        return (
            create_preparation_session(
                selected_analysis_dataset_ids=(
                    request
                    .selected_analysis_dataset_ids
                )
            )
        )


    except ValueError as exc:
        raise HTTPException(
            status_code=(
                status
                .HTTP_422_UNPROCESSABLE_CONTENT
            ),

            detail=(
                _invalid_session_detail(
                    exc
                )
            ),
        ) from exc


# ============================================================
# READ SESSION
# ============================================================


@router.get(
    "/sessions/{workflow_id}",
    response_model=
        PreparationSessionView,
)
def read_session(
    workflow_id: str,
) -> PreparationSessionView:
    """
    Return the current server-derived Preparation snapshot.

    This endpoint is read-only.
    """

    try:
        return (
            get_preparation_session(
                workflow_id
            )
        )


    except PreparationSessionNotFoundError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),

            detail=(
                _not_found_detail(
                    exc
                )
            ),
        ) from exc


# ============================================================
# ANALYSIS OUTPUT CANDIDATES
# ============================================================


@router.get(
    (
        "/sessions/{workflow_id}"
        "/analysis-output-candidates"
    ),
    response_model=
        PreparationAnalysisOutputCandidatesResponse,
)
def read_analysis_output_candidates(
    workflow_id: str,
) -> PreparationAnalysisOutputCandidatesResponse:
    """
    Return safe metadata for the materialized Preparation
    artifacts that may be presented to the analyst.

    No DataFrame values are exposed.

    IMPORTANT:

    Presence in this list does not let the browser authorize
    the artifact itself.

    The final POST selection still passes through
    commit_analysis_output_selection(), which performs the
    authoritative server-side lineage validation.
    """

    try:
        # ====================================================
        # SERVER-OWNED SESSION
        #
        # Read this first so an unknown Preparation session is
        # always a 404 even if Artifact Store.list() would
        # otherwise simply return an empty list.
        # ====================================================

        session = (
            get_preparation_session(
                workflow_id
            )
        )


        # ====================================================
        # SERVER-OWNED MATERIALIZED ARTIFACTS
        # ====================================================

        artifacts = (
            list_preparation_artifacts(
                workflow_id=
                    session.workflow_id
            )
        )


        root_ids = set(
            session
            .selected_analysis_dataset_ids
        )


        selected_ids = set(
            session
            .analysis_output_dataset_ids
        )


        validated_ids = set(
            session
            .snapshot
            .validated_analysis_dataset_ids
        )


        # ====================================================
        # PRESENTATION ORDER
        #
        # More final/materialized stages appear first.
        # This is UI ordering only.
        #
        # It does NOT authorize an artifact.
        # ====================================================

        stage_priority = {
            "combine":
                0,

            "transform":
                1,

            "clean":
                2,

            "source":
                3,
        }


        artifacts = sorted(
            artifacts,

            key=lambda artifact: (
                stage_priority.get(
                    str(
                        artifact.stage
                    ),
                    99,
                ),

                artifact
                .dataset_filename
                .lower(),

                artifact.dataset_id,
            ),
        )


        candidates = [
            PreparationAnalysisOutputCandidate(
                dataset_id=
                    artifact.dataset_id,

                dataset_filename=
                    artifact.dataset_filename,

                stage=
                    str(
                        artifact.stage
                    ),

                rows=
                    artifact.rows,

                columns=
                    artifact.columns,

                parent_dataset_ids=
                    list(
                        artifact
                        .parent_dataset_ids
                    ),

                evidence_refs=
                    list(
                        artifact
                        .evidence_refs
                    ),

                is_root_dataset=(
                    artifact.dataset_id
                    in
                    root_ids
                ),

                is_selected=(
                    artifact.dataset_id
                    in
                    selected_ids
                ),

                is_validated=(
                    artifact.dataset_id
                    in
                    validated_ids
                ),
            )

            for artifact
            in artifacts
        ]


        return (
            PreparationAnalysisOutputCandidatesResponse(
                workflow_id=
                    session.workflow_id,

                revision=
                    session.revision,

                selected_analysis_dataset_ids=
                    list(
                        session
                        .selected_analysis_dataset_ids
                    ),

                analysis_output_dataset_ids=
                    list(
                        session
                        .analysis_output_dataset_ids
                    ),

                validated_analysis_dataset_ids=
                    list(
                        session
                        .snapshot
                        .validated_analysis_dataset_ids
                    ),

                locked=(
                    _analysis_output_selection_locked(
                        session
                    )
                ),

                candidate_count=
                    len(
                        candidates
                    ),

                candidates=
                    candidates,

                api_version=
                    PREPARATION_SESSION_API_VERSION,
            )
        )


    except PreparationSessionNotFoundError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),

            detail=(
                _not_found_detail(
                    exc
                )
            ),
        ) from exc


    except ValueError as exc:
        raise HTTPException(
            status_code=(
                status
                .HTTP_422_UNPROCESSABLE_CONTENT
            ),

            detail=(
                _invalid_session_detail(
                    exc
                )
            ),
        ) from exc


# ============================================================
# COMMIT ANALYSIS OUTPUT SELECTION
# ============================================================


@router.post(
    "/analysis-output",
    response_model=
        PreparationSessionView,
)
def select_analysis_output(
    request:
        PreparationAnalysisOutputSelectionRequest,
) -> PreparationSessionView:
    """
    Select the final materialized dataset scope that Analysis
    will eventually consume.

    Trust boundary:

        browser chooses dataset_id
                    ↓
        server-owned Preparation session
                    ↓
        server-owned Artifact Store
                    ↓
        lineage / stage validation
                    ↓
        optimistic revision guard
                    ↓
        analysis_output_dataset_ids commit

    The browser never submits:

        PASSED
        ready_for_analysis
        validation state
        lineage
        DataFrames

    A successful request does NOT make Analysis ready.

    VALIDATE must still execute afterwards.
    """

    try:
        # ====================================================
        # AUTHORITATIVE SERVER-SIDE COMMIT
        #
        # Do NOT call record_analysis_output_selection()
        # directly here.
        #
        # This wrapper validates the requested output against
        # current Preparation Artifact Store lineage before the
        # session transaction is committed.
        # ====================================================

        commit_analysis_output_selection(
            workflow_id=
                request.workflow_id,

            requested_dataset_ids=
                request.dataset_ids,
        )


        # ====================================================
        # RETURN CURRENT SERVER STATE
        # ====================================================

        return (
            get_preparation_session(
                request.workflow_id
            )
        )


    except PreparationSessionNotFoundError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),

            detail=(
                _not_found_detail(
                    exc
                )
            ),
        ) from exc


    except PreparationSessionRevisionConflictError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_409_CONFLICT
            ),

            detail=(
                _revision_conflict_detail(
                    exc=
                        exc,

                    workflow_id=
                        request.workflow_id,
                )
            ),
        ) from exc


    except ValueError as exc:
        # State-level conflicts such as:
        #
        # - invalid output selection;
        # - output change after VALIDATE PASSED;
        # - violated Preparation invariant.
        #
        # The JSON shape itself has already been validated by
        # Pydantic before reaching this function.
        raise HTTPException(
            status_code=(
                status.HTTP_409_CONFLICT
            ),

            detail=(
                _analysis_output_conflict_detail(
                    exc=
                        exc,

                    workflow_id=
                        request.workflow_id,
                )
            ),
        ) from exc


    except RuntimeError as exc:
        # Selection / lineage guardrails are allowed to fail
        # closed with domain-level runtime errors.
        #
        # They represent a conflict with current Preparation
        # state, not a server crash.
        raise HTTPException(
            status_code=(
                status.HTTP_409_CONFLICT
            ),

            detail=(
                _analysis_output_conflict_detail(
                    exc=
                        exc,

                    workflow_id=
                        request.workflow_id,
                )
            ),
        ) from exc