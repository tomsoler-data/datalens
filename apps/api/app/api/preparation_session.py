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
    PreparationSessionCatalogItem,
    PreparationSessionNotFoundError,
    PreparationSessionRevisionConflictError,
    PreparationSessionStoreError,
    PreparationSessionView,
    create_preparation_session,
    get_preparation_session,
    list_preparation_sessions,
    archive_preparation_session,
    restore_preparation_session,
    rename_preparation_session,
)



from app.preparation.preparation_workflow_delete import (
    PreparationWorkflowDeleteConfirmationError,
    PreparationWorkflowDeleteIntegrityError,
    PreparationWorkflowDeleteNotArchivedError,
    PreparationWorkflowDeleteNotFoundError,
    PreparationWorkflowDeleteRecoveryError,
    PreparationWorkflowDeleteResult,
    PreparationWorkflowDeleteRevisionConflictError,
    delete_preparation_workflow,
)


from app.ingestion.loader import (
    build_dataset_manifest,
)

from app.ingestion.schemas import (
    MultiDatasetIngestion,
)

from app.preparation.preparation_artifact_store import (
    PreparationArtifactStoreError,
    get_preparation_artifact,
)


from app.preparation.preparation_ui_state import (
    PreparationUiStateView,
    get_preparation_ui_state,
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

    display_name: (
        str
        |
        None
    ) = Field(
        default=None,
        max_length=120,
    )


class PreparationSessionRenameRequest(
    StrictPreparationSessionRequest,
):
    display_name: str = Field(
        min_length=1,
        max_length=120,
    )



# ============================================================
# PERMANENT WORKFLOW DELETE REQUEST
# PREPARATION_WORKFLOW_DELETE_API_V0_1
# ============================================================


class PreparationWorkflowDeleteRequest(
    StrictPreparationSessionRequest,
):
    confirmation_workflow_id: str = Field(
        min_length=1,
    )

    confirmation_display_name: str = Field(
        min_length=1,
        max_length=120,
    )

    expected_revision: int = Field(
        ge=0,
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
# WORKFLOW HISTORY ? RESPONSE
# PREPARATION_SESSION_CATALOG_V0_1
# ============================================================


class PreparationSessionCatalogResponse(
    BaseModel,
):
    model_config = ConfigDict(
        extra="forbid"
    )

    count: int = Field(
        ge=0
    )

    sessions: List[
        PreparationSessionCatalogItem
    ]

    api_version: str = (
        PREPARATION_SESSION_API_VERSION
    )


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
# PERMANENT WORKFLOW DELETE ERROR
# PREPARATION_WORKFLOW_DELETE_API_V0_1
# ============================================================


def _workflow_delete_detail(
    *,
    error: str,
    exc: Exception,
    workflow_id: str,
) -> Dict[
    str,
    Any,
]:
    return {
        "error":
            error,

        "message":
            str(
                exc
            ),

        "workflow_id":
            workflow_id,

        "api_version":
            PREPARATION_SESSION_API_VERSION,
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
                "sqlite",

            persistent=
                True,

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
                    "Preparation sessions are stored "
                    "in the local DataLens SQLite "
                    "control-plane database."
                ),

                (
                    "Sessions survive FastAPI process "
                    "restarts."
                ),

                (
                    "workflow_id values remain "
                    "backend-generated."
                ),

                (
                    "Preparation stage statuses remain "
                    "backend-owned and derived from "
                    "server-side evidence."
                ),

                (
                    "Final analysis-output selection "
                    "keeps optimistic revision checks "
                    "and artifact-lineage validation."
                ),
            ],
        )
    )


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
                ),

                display_name=(
                    request.display_name
                ),
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



# ============================================================
# WORKFLOW HISTORY ? READ-ONLY CATALOG
# PREPARATION_SESSION_CATALOG_V0_1
# ============================================================


@router.get(
    "/sessions",
    response_model=
        PreparationSessionCatalogResponse,
)
def list_sessions(
) -> PreparationSessionCatalogResponse:
    """
    Return the server-owned Preparation workflow history.

    Catalog membership is defined exclusively by
    preparation_sessions.

    Historical AnalysisArtifact / ReportSelection rows that no
    longer own a Preparation session are intentionally excluded.
    """

    try:
        sessions = (
            list_preparation_sessions()
        )


        return (
            PreparationSessionCatalogResponse(
                count=
                    len(
                        sessions
                    ),

                sessions=
                    sessions,
            )
        )


    except PreparationSessionStoreError as exc:
        raise HTTPException(
            status_code=
                status
                .HTTP_500_INTERNAL_SERVER_ERROR,

            detail={
                "error":
                    "preparation_session_catalog_failed",

                "message":
                    str(
                        exc
                    ),

                "api_version":
                    PREPARATION_SESSION_API_VERSION,
            },
        ) from exc


# ============================================================
# WORKFLOW LIFECYCLE
# PREPARATION_WORKFLOW_METADATA_V0_1
# ============================================================



# ============================================================
# WORKFLOW METADATA ? RENAME
# PREPARATION_WORKFLOW_METADATA_V0_1
# ============================================================


@router.post(
    "/sessions/{workflow_id}/rename",
    response_model=
        PreparationSessionCatalogItem,
)
def rename_session(
    workflow_id: str,
    request:
        PreparationSessionRenameRequest,
) -> PreparationSessionCatalogItem:
    try:
        return (
            rename_preparation_session(
                workflow_id=
                    workflow_id,

                display_name=
                    request.display_name,
            )
        )


    except PreparationSessionNotFoundError as exc:
        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,

            detail=(
                _not_found_detail(
                    exc
                )
            ),
        ) from exc


    except ValueError as exc:
        raise HTTPException(
            status_code=
                status
                .HTTP_422_UNPROCESSABLE_CONTENT,

            detail=(
                _invalid_session_detail(
                    exc
                )
            ),
        ) from exc


    except PreparationSessionStoreError as exc:
        raise HTTPException(
            status_code=
                status
                .HTTP_500_INTERNAL_SERVER_ERROR,

            detail={
                "error":
                    "preparation_workflow_rename_failed",

                "message":
                    str(
                        exc
                    ),

                "api_version":
                    PREPARATION_SESSION_API_VERSION,
            },
        ) from exc

@router.post(
    "/sessions/{workflow_id}/archive",
    response_model=
        PreparationSessionCatalogItem,
)
def archive_session(
    workflow_id: str,
) -> PreparationSessionCatalogItem:
    """
    Archive a server-owned workflow without deleting or
    modifying its analytical state.
    """

    try:
        return (
            archive_preparation_session(
                workflow_id
            )
        )


    except PreparationSessionNotFoundError as exc:
        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,

            detail=(
                _not_found_detail(
                    exc
                )
            ),
        ) from exc


    except ValueError as exc:
        raise HTTPException(
            status_code=
                status
                .HTTP_422_UNPROCESSABLE_CONTENT,

            detail=(
                _invalid_session_detail(
                    exc
                )
            ),
        ) from exc


    except PreparationSessionStoreError as exc:
        raise HTTPException(
            status_code=
                status
                .HTTP_500_INTERNAL_SERVER_ERROR,

            detail={
                "error":
                    "preparation_workflow_archive_failed",

                "message":
                    str(
                        exc
                    ),

                "api_version":
                    PREPARATION_SESSION_API_VERSION,
            },
        ) from exc


@router.post(
    "/sessions/{workflow_id}/restore",
    response_model=
        PreparationSessionCatalogItem,
)
def restore_session(
    workflow_id: str,
) -> PreparationSessionCatalogItem:
    """
    Restore an archived workflow.

    No Preparation, AnalysisArtifact, ReportSelection or
    filesystem payload is recreated because archive never
    deleted those resources.
    """

    try:
        return (
            restore_preparation_session(
                workflow_id
            )
        )


    except PreparationSessionNotFoundError as exc:
        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,

            detail=(
                _not_found_detail(
                    exc
                )
            ),
        ) from exc


    except ValueError as exc:
        raise HTTPException(
            status_code=
                status
                .HTTP_422_UNPROCESSABLE_CONTENT,

            detail=(
                _invalid_session_detail(
                    exc
                )
            ),
        ) from exc


    except PreparationSessionStoreError as exc:
        raise HTTPException(
            status_code=
                status
                .HTTP_500_INTERNAL_SERVER_ERROR,

            detail={
                "error":
                    "preparation_workflow_restore_failed",

                "message":
                    str(
                        exc
                    ),

                "api_version":
                    PREPARATION_SESSION_API_VERSION,
            },
        ) from exc

# ============================================================
# PERMANENT WORKFLOW DELETE
# PREPARATION_WORKFLOW_DELETE_API_V0_1
# ============================================================


@router.delete(
    "/sessions/{workflow_id}",
    response_model=
        PreparationWorkflowDeleteResult,
)
def delete_session(
    workflow_id: str,
    request:
        PreparationWorkflowDeleteRequest,
) -> PreparationWorkflowDeleteResult:
    """
    Permanently destroy an archived Preparation workflow.

    Archive remains reversible.

    Delete irreversibly removes the workflow-owned SQLite
    control-plane rows and filesystem payloads.

    The caller must confirm:
    - immutable workflow_id;
    - current display name;
    - current analytical revision.
    """

    try:
        return (
            delete_preparation_workflow(
                workflow_id=
                    workflow_id,

                confirmation_workflow_id=(
                    request
                    .confirmation_workflow_id
                ),

                confirmation_display_name=(
                    request
                    .confirmation_display_name
                ),

                expected_revision=(
                    request
                    .expected_revision
                ),
            )
        )


    except PreparationWorkflowDeleteNotFoundError as exc:
        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,

            detail=(
                _workflow_delete_detail(
                    error=
                        "preparation_workflow_not_found",

                    exc=
                        exc,

                    workflow_id=
                        workflow_id,
                )
            ),
        ) from exc


    except PreparationWorkflowDeleteNotArchivedError as exc:
        raise HTTPException(
            status_code=
                status.HTTP_409_CONFLICT,

            detail=(
                _workflow_delete_detail(
                    error=(
                        "preparation_workflow_"
                        "delete_requires_archive"
                    ),

                    exc=
                        exc,

                    workflow_id=
                        workflow_id,
                )
            ),
        ) from exc


    except PreparationWorkflowDeleteConfirmationError as exc:
        raise HTTPException(
            status_code=
                status.HTTP_409_CONFLICT,

            detail=(
                _workflow_delete_detail(
                    error=(
                        "preparation_workflow_"
                        "delete_confirmation_conflict"
                    ),

                    exc=
                        exc,

                    workflow_id=
                        workflow_id,
                )
            ),
        ) from exc


    except PreparationWorkflowDeleteRevisionConflictError as exc:
        raise HTTPException(
            status_code=
                status.HTTP_409_CONFLICT,

            detail=(
                _workflow_delete_detail(
                    error=(
                        "preparation_workflow_"
                        "delete_revision_conflict"
                    ),

                    exc=
                        exc,

                    workflow_id=
                        workflow_id,
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
                _workflow_delete_detail(
                    error=(
                        "invalid_preparation_"
                        "workflow_delete_request"
                    ),

                    exc=
                        exc,

                    workflow_id=
                        workflow_id,
                )
            ),
        ) from exc


    except PreparationWorkflowDeleteIntegrityError as exc:
        raise HTTPException(
            status_code=
                status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=(
                _workflow_delete_detail(
                    error=(
                        "preparation_workflow_"
                        "delete_integrity_failure"
                    ),

                    exc=
                        exc,

                    workflow_id=
                        workflow_id,
                )
            ),
        ) from exc


    except PreparationWorkflowDeleteRecoveryError as exc:
        raise HTTPException(
            status_code=
                status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=(
                _workflow_delete_detail(
                    error=(
                        "preparation_workflow_"
                        "delete_recovery_failure"
                    ),

                    exc=
                        exc,

                    workflow_id=
                        workflow_id,
                )
            ),
        ) from exc


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
# PREPARATION UI STATE
# PREPARATION_UI_STATE_READ_V0_1
# ============================================================


@router.get(
    "/sessions/{workflow_id}/ui-state",
    response_model=
        PreparationUiStateView,
)
def read_preparation_ui_state(
    workflow_id: str,
) -> PreparationUiStateView:
    """
    Restore committed structured Preparation outputs.

    The Preparation Session remains the workflow authority.
    This endpoint contains no DataFrames and performs no
    recalculation, LLM call or cleaning execution.
    """

    try:
        # Fail closed for stale / unknown workflow IDs.
        get_preparation_session(
            workflow_id
        )

        return (
            get_preparation_ui_state(
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
    "/sessions/{workflow_id}/ingestion-view",
    response_model=
        MultiDatasetIngestion,
)
def read_preparation_ingestion_view(
    workflow_id: str,
) -> MultiDatasetIngestion:
    """
    Reconstruct the dataset metadata required by the frontend
    from server-owned Preparation artifacts.

    This endpoint is strictly read-only:

    - it does not re-run ingestion;
    - it does not require browser File objects;
    - it does not mutate Preparation artifacts;
    - it does not advance the workflow;
    - it never exposes DataFrame values.

    The immutable Preparation root dataset IDs determine which
    logical datasets belong to the restored ingestion view.
    Their current server-owned materializations are used to
    rebuild DatasetManifest objects.
    """

    try:
        session = (
            get_preparation_session(
                workflow_id
            )
        )

    except PreparationSessionNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(
                exc
            ),
        ) from exc


    root_dataset_ids = list(
        session
        .selected_analysis_dataset_ids
    )


    if not root_dataset_ids:
        raise HTTPException(
            status_code=409,
            detail=(
                "Preparation session has no immutable "
                "root datasets available for ingestion "
                "rehydration."
            ),
        )


    manifests = []


    try:
        for dataset_id in root_dataset_ids:
            artifact = (
                get_preparation_artifact(
                    workflow_id=
                        workflow_id,

                    dataset_id=
                        dataset_id,
                )
            )


            filename = (
                artifact
                .dataset_filename
            )


            if "." in filename:
                extension = (
                    "."
                    +
                    filename
                    .rsplit(
                        ".",
                        1,
                    )[
                        1
                    ]
                    .lower()
                )

            else:
                # DataLens ingestion is currently CSV-based.
                extension = ".csv"


            manifest = (
                build_dataset_manifest(
                    artifact.dataframe,

                    dataset_id=
                        artifact.dataset_id,

                    filename=
                        filename,

                    extension=
                        extension,
                )
            )


            manifests.append(
                manifest
            )

    except PreparationArtifactStoreError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "Preparation dataset context cannot be "
                "rehydrated because a required server-owned "
                "artifact is unavailable. "
                f"{exc}"
            ),
        ) from exc


    filenames = [
        manifest.filename
        for manifest
        in manifests
    ]


    duplicate_filenames = sorted(
        {
            filename
            for filename
            in filenames
            if filenames.count(
                filename
            )
            > 1
        }
    )


    warnings = []


    if duplicate_filenames:
        warnings.append(
            (
                "Duplicate filenames are present in the "
                "restored Preparation workflow: "
                +
                ", ".join(
                    duplicate_filenames
                )
                +
                ". Dataset IDs should be used to "
                "distinguish them."
            )
        )


    return (
        MultiDatasetIngestion(
            dataset_count=
                len(
                    manifests
                ),

            total_rows=
                sum(
                    manifest.row_count
                    for manifest
                    in manifests
                ),

            datasets=
                manifests,

            warnings=
                warnings,
        )
    )


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