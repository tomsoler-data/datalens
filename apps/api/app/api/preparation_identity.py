from __future__ import annotations

from pathlib import Path

from typing import (
    Any,
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

from app.preparation.dataset_identity import (
    DATASET_IDENTITY_RULE_VERSION,
    DatasetIdentityReport,
    create_surrogate_row_key,
    profile_dataset_identity,
)

from app.preparation.dataset_identity_explanation import (
    DATASET_IDENTITY_EXPLANATION_RULE_VERSION,
    DatasetIdentityExplanation,
    explain_dataset_identity_with_ai,
)

from app.preparation.preparation_identity_resolution import (
    PREPARATION_IDENTITY_RESOLUTION_VERSION,
    PreparationIdentityResolutionError,
    build_identity_resolution_request_id,
    clear_identity_resolution,
    get_current_identity_resolution,
    record_continue_without_surrogate,
)

from app.preparation.preparation_artifact_store import (
    PREPARATION_ARTIFACT_STORE_VERSION,
    PreparationArtifactDatasetNotFoundError,
    PreparationArtifactStoreError,
    PreparationArtifactWorkflowNotFoundError,
    get_preparation_artifact,
    list_preparation_artifacts,
    put_preparation_artifact,
)

from app.preparation.preparation_session import (
    PreparationSessionNotFoundError,
    PreparationSessionView,
    get_preparation_session,
    record_optional_stage_signal,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
)


# ============================================================
# VERSION
# ============================================================


PREPARATION_IDENTITY_API_VERSION = (
    "preparation_identity_api_v0.2"
)


# ============================================================
# ROUTER
# ============================================================


router = APIRouter(
    prefix="/preparation/identity",
    tags=[
        "preparation",
    ],
)


# ============================================================
# STRICT REQUESTS
# ============================================================


class StrictPreparationIdentityRequest(
    BaseModel,
):
    model_config = ConfigDict(
        extra="forbid"
    )


class PreparationIdentityInspectRequest(
    StrictPreparationIdentityRequest,
):
    workflow_id: str = Field(
        min_length=1
    )

    dataset_id: str = Field(
        min_length=1
    )

    include_ai: bool = True


class PreparationIdentityCreateSurrogateRequest(
    StrictPreparationIdentityRequest,
):
    workflow_id: str = Field(
        min_length=1
    )

    dataset_id: str = Field(
        min_length=1
    )

    request_id: str = Field(
        min_length=1
    )


class PreparationIdentityContinueRequest(
    StrictPreparationIdentityRequest,
):
    workflow_id: str = Field(
        min_length=1
    )

    dataset_id: str = Field(
        min_length=1
    )

    request_id: str = Field(
        min_length=1
    )


# ============================================================
# RESPONSES
# ============================================================


class PreparationIdentityInspectResponse(
    BaseModel,
):
    workflow_id: str
    dataset_id: str
    dataset_filename: str
    artifact_stage: str

    report: DatasetIdentityReport

    explanation: (
        DatasetIdentityExplanation
        | None
    ) = None

    ai_error: (
        str
        | None
    ) = None

    surrogate_request_id: (
        str
        | None
    ) = None

    identity_resolved: bool
    resolution_kind: (
        str
        | None
    ) = None

    can_create_surrogate: bool
    can_continue_without_surrogate: bool

    mutation_locked: bool
    mutation_lock_reason: (
        str
        | None
    ) = None

    api_version: str = (
        PREPARATION_IDENTITY_API_VERSION
    )

    identity_rule_version: str = (
        DATASET_IDENTITY_RULE_VERSION
    )

    explanation_rule_version: str = (
        DATASET_IDENTITY_EXPLANATION_RULE_VERSION
    )

    artifact_store_version: str = (
        PREPARATION_ARTIFACT_STORE_VERSION
    )

    identity_resolution_version: str = (
        PREPARATION_IDENTITY_RESOLUTION_VERSION
    )


class PreparationIdentityContinueResponse(
    BaseModel,
):
    workflow_id: str
    dataset_id: str
    dataset_filename: str
    request_id: str
    resolution_kind: str
    identity_resolved: bool = True

    api_version: str = (
        PREPARATION_IDENTITY_API_VERSION
    )

    identity_resolution_version: str = (
        PREPARATION_IDENTITY_RESOLUTION_VERSION
    )


class PreparationIdentityCreateSurrogateResponse(
    BaseModel,
):
    workflow_id: str

    source_dataset_id: str
    source_dataset_filename: str

    output_dataset_id: str
    output_dataset_filename: str

    surrogate_column: str

    rows: int
    columns: int

    parent_dataset_ids: list[
        str
    ]

    report_before: DatasetIdentityReport

    session: PreparationSessionView

    api_version: str = (
        PREPARATION_IDENTITY_API_VERSION
    )

    identity_rule_version: str = (
        DATASET_IDENTITY_RULE_VERSION
    )

    artifact_store_version: str = (
        PREPARATION_ARTIFACT_STORE_VERSION
    )


# ============================================================
# SESSION HELPERS
# ============================================================


def _stage_status(
    session: PreparationSessionView,
    stage: PreparationStage,
) -> str:
    record = next(
        (
            item
            for item
            in session.snapshot.stages
            if item.stage == stage
        ),
        None,
    )


    if (
        record is None
    ):
        raise RuntimeError(
            (
                "Preparation stage missing from "
                f"snapshot: {stage.value}"
            )
        )


    return str(
        record.status.value
    )


def _mutation_lock_reason(
    *,
    workflow_id: str,
    session: PreparationSessionView,
) -> str | None:
    validate_status = _stage_status(
        session,
        PreparationStage.VALIDATE,
    )


    if (
        validate_status
        ==
        "passed"
        or
        session.snapshot.ready_for_analysis
    ):
        return (
            "Preparation identity cannot mutate a dataset "
            "after final validation."
        )


    # ========================================================
    # IDENTITY_CLEAN_GUARD_V0_1
    #
    # Identity may inspect a dataset before CLEAN is resolved,
    # but it must not persist an identity decision or
    # materialize a TRANSFORM artifact until CLEAN is PASSED
    # or SKIPPED.
    # ========================================================

    clean_status = _stage_status(
        session,
        PreparationStage.CLEAN,
    )


    if (
        clean_status
        not in {
            "passed",
            "skipped",
        }
    ):
        return (
            "Preparation identity cannot mutate before "
            "CLEAN is resolved. "
            f"Current CLEAN status: {clean_status}."
        )


    combine_artifacts = [
        artifact
        for artifact
        in list_preparation_artifacts(
            workflow_id=
                workflow_id
        )
        if artifact.stage == "combine"
    ]


    if (
        combine_artifacts
    ):
        return (
            "Preparation identity must be resolved before "
            "materialized COMBINE artifacts exist."
        )


    combine_status = _stage_status(
        session,
        PreparationStage.COMBINE,
    )


    if (
        combine_status
        in {
            "review_required",
            "blocked",
            "passed",
        }
    ):
        return (
            "Preparation identity must be resolved before "
            "the COMBINE workflow is active or completed."
        )


    return None


# ============================================================
# REQUEST ID
# ============================================================


def _surrogate_request_id(
    *,
    workflow_id: str,
    dataset_id: str,
    dataset_filename: str,
    artifact_stage: str,
    report: DatasetIdentityReport,
) -> str:
    return (
        build_identity_resolution_request_id(
            workflow_id=
                workflow_id,

            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            artifact_stage=
                artifact_stage,

            report=
                report,
        )
    )


def _output_dataset_id(
    request_id: str,
) -> str:
    digest = (
        request_id
        .split(
            ":",
            1,
        )[
            -1
        ]
    )


    return (
        f"transform:surrogate:{digest}"
    )


def _output_dataset_filename(
    *,
    source_filename: str,
    surrogate_column: str,
) -> str:
    stem = Path(
        source_filename
    ).stem


    safe_surrogate = (
        surrogate_column
        .strip()
        .replace(
            " ",
            "_",
        )
    )


    return (
        f"{stem}__{safe_surrogate}.csv"
    )


# ============================================================
# IDENTITY REPORT
# ============================================================


def _build_report(
    *,
    workflow_id: str,
    dataset_id: str,
):
    session = get_preparation_session(
        workflow_id
    )


    artifact = get_preparation_artifact(
        workflow_id=
            workflow_id,

        dataset_id=
            dataset_id,
    )


    report = profile_dataset_identity(
        artifact.dataframe,

        dataset_id=
            artifact.dataset_id,

        dataset_filename=
            artifact.dataset_filename,
    )


    return (
        session,
        artifact,
        report,
    )


# ============================================================
# ERROR DETAIL
# ============================================================


def _error_detail(
    *,
    code: str,
    error: Exception,
    workflow_id: str,
    dataset_id: str,
) -> dict[
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

        "dataset_id":
            dataset_id,

        "api_version":
            PREPARATION_IDENTITY_API_VERSION,
    }


# ============================================================
# INSPECT
# ============================================================


@router.post(
    "/inspect",
    response_model=
        PreparationIdentityInspectResponse,
)
def inspect_preparation_identity(
    request:
        PreparationIdentityInspectRequest,
) -> PreparationIdentityInspectResponse:
    """
    Inspect row identity using a server-owned Preparation artifact.

    The browser provides only:

        workflow_id
        dataset_id
        include_ai

    Python computes the identity facts.

    The optional local LLM receives only the structured report,
    never the DataFrame or raw row values.

    LLM failure is non-blocking because identity facts remain
    available from Python.
    """

    try:
        (
            session,
            artifact,
            report,
        ) = _build_report(
            workflow_id=
                request.workflow_id,

            dataset_id=
                request.dataset_id,
        )


        lock_reason = (
            _mutation_lock_reason(
                workflow_id=
                    request.workflow_id,

                session=
                    session,
            )
        )


        request_id = (
            _surrogate_request_id(
                workflow_id=
                    request.workflow_id,

                dataset_id=
                    artifact.dataset_id,

                dataset_filename=
                    artifact.dataset_filename,

                artifact_stage=
                    artifact.stage,

                report=
                    report,
            )
            if (
                report.surrogate_key_recommended
            )
            else None
        )


        resolution = (
            get_current_identity_resolution(
                workflow_id=
                    request.workflow_id,

                dataset_id=
                    artifact.dataset_id,

                dataset_filename=
                    artifact.dataset_filename,

                artifact_stage=
                    artifact.stage,

                report=
                    report,
            )
        )


        explanation = None
        ai_error = None


        if (
            request.include_ai
        ):
            try:
                explanation = (
                    explain_dataset_identity_with_ai(
                        report
                    )
                )

            except Exception:
                ai_error = (
                    "Local model dataset-identity explanation "
                    "is unavailable."
                )


        can_create = bool(
            report.surrogate_key_recommended
            and
            report.suggested_surrogate_column
            and
            request_id
            and
            lock_reason is None
        )


        can_continue = bool(
            report.surrogate_key_recommended
            and
            request_id
            and
            lock_reason is None
        )


        return (
            PreparationIdentityInspectResponse(
                workflow_id=
                    request.workflow_id,

                dataset_id=
                    artifact.dataset_id,

                dataset_filename=
                    artifact.dataset_filename,

                artifact_stage=
                    artifact.stage,

                report=
                    report,

                explanation=
                    explanation,

                ai_error=
                    ai_error,

                surrogate_request_id=
                    request_id,

                identity_resolved=
                    resolution
                    is not None,

                resolution_kind=(
                    resolution.kind
                    if resolution
                    is not None
                    else None
                ),

                can_create_surrogate=
                    can_create,

                can_continue_without_surrogate=
                    can_continue,

                mutation_locked=
                    lock_reason
                    is not None,

                mutation_lock_reason=
                    lock_reason,
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

                    dataset_id=
                        request.dataset_id,
                )
            ),
        ) from error


    except (
        PreparationArtifactWorkflowNotFoundError,
        PreparationArtifactDatasetNotFoundError,
    ) as error:
        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,

            detail=(
                _error_detail(
                    code=
                        "preparation_identity_artifact_not_found",

                    error=
                        error,

                    workflow_id=
                        request.workflow_id,

                    dataset_id=
                        request.dataset_id,
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
                        "preparation_identity_inspection_rejected",

                    error=
                        error,

                    workflow_id=
                        request.workflow_id,

                    dataset_id=
                        request.dataset_id,
                )
            ),
        ) from error


# ============================================================
# CONTINUE WITHOUT SURROGATE
# ============================================================


@router.post(
    "/continue",
    response_model=
        PreparationIdentityContinueResponse,
)
def continue_preparation_without_surrogate(
    request:
        PreparationIdentityContinueRequest,
) -> PreparationIdentityContinueResponse:
    """
    Persist an explicit analyst decision to continue without
    creating the Python-recommended technical row identifier.

    The request_id is derived from the current deterministic
    identity report, so a changed artifact automatically makes
    an older decision stale.
    """

    try:
        (
            session,
            artifact,
            report,
        ) = _build_report(
            workflow_id=
                request.workflow_id,

            dataset_id=
                request.dataset_id,
        )


        lock_reason = (
            _mutation_lock_reason(
                workflow_id=
                    request.workflow_id,

                session=
                    session,
            )
        )


        if (
            lock_reason
            is not None
        ):
            raise ValueError(
                lock_reason
            )


        resolution = (
            record_continue_without_surrogate(
                workflow_id=
                    request.workflow_id,

                dataset_id=
                    artifact.dataset_id,

                dataset_filename=
                    artifact.dataset_filename,

                artifact_stage=
                    artifact.stage,

                report=
                    report,

                request_id=
                    request.request_id,
            )
        )


        return (
            PreparationIdentityContinueResponse(
                workflow_id=
                    request.workflow_id,

                dataset_id=
                    artifact.dataset_id,

                dataset_filename=
                    artifact.dataset_filename,

                request_id=
                    resolution.request_id,

                resolution_kind=
                    resolution.kind,
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

                    dataset_id=
                        request.dataset_id,
                )
            ),
        ) from error


    except (
        PreparationArtifactWorkflowNotFoundError,
        PreparationArtifactDatasetNotFoundError,
    ) as error:
        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,

            detail=(
                _error_detail(
                    code=
                        "preparation_identity_artifact_not_found",

                    error=
                        error,

                    workflow_id=
                        request.workflow_id,

                    dataset_id=
                        request.dataset_id,
                )
            ),
        ) from error


    except (
        PreparationIdentityResolutionError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=
                status.HTTP_409_CONFLICT,

            detail=(
                _error_detail(
                    code=
                        "preparation_identity_continue_rejected",

                    error=
                        error,

                    workflow_id=
                        request.workflow_id,

                    dataset_id=
                        request.dataset_id,
                )
            ),
        ) from error


# ============================================================
# CREATE SURROGATE
# ============================================================


@router.post(
    "/create-surrogate",
    response_model=
        PreparationIdentityCreateSurrogateResponse,
)
def create_preparation_surrogate_key(
    request:
        PreparationIdentityCreateSurrogateRequest,
) -> PreparationIdentityCreateSurrogateResponse:
    """
    Materialize exactly the technical key authorized by Python.

    Security / integrity properties:

    - client cannot provide the surrogate column name;
    - client cannot provide arbitrary transformation code;
    - client cannot provide output dataset_id;
    - request_id is derived from the current Python report;
    - report is recomputed immediately before execution;
    - stale request_id is rejected;
    - no mutation is allowed until CLEAN is PASSED or SKIPPED;
    - no mutation is allowed after COMBINE materialization or
      final VALIDATE;
    - source artifact is not mutated;
    - output is a server-owned TRANSFORM artifact.
    """

    try:
        (
            session,
            artifact,
            report,
        ) = _build_report(
            workflow_id=
                request.workflow_id,

            dataset_id=
                request.dataset_id,
        )


        lock_reason = (
            _mutation_lock_reason(
                workflow_id=
                    request.workflow_id,

                session=
                    session,
            )
        )


        if (
            lock_reason
            is not None
        ):
            raise ValueError(
                lock_reason
            )


        if (
            artifact.stage
            ==
            "combine"
        ):
            raise ValueError(
                (
                    "A surrogate row key cannot be created "
                    "on a COMBINE artifact in this Preparation "
                    "stage order."
                )
            )


        if not (
            report.surrogate_key_recommended
        ):
            raise ValueError(
                (
                    "Python did not recommend a surrogate "
                    "row identity for this dataset."
                )
            )


        surrogate_column = (
            report
            .suggested_surrogate_column
        )


        if (
            surrogate_column
            is None
        ):
            raise RuntimeError(
                (
                    "Dataset Identity recommended a surrogate "
                    "key without an authorized column name."
                )
            )


        expected_request_id = (
            _surrogate_request_id(
                workflow_id=
                    request.workflow_id,

                dataset_id=
                    artifact.dataset_id,

                dataset_filename=
                    artifact.dataset_filename,

                artifact_stage=
                    artifact.stage,

                report=
                    report,
            )
        )


        if (
            request.request_id
            !=
            expected_request_id
        ):
            raise ValueError(
                (
                    "Surrogate-key approval is stale or does "
                    "not match the current deterministic "
                    "identity report."
                )
            )


        # ====================================================
        # PRE-MATERIALIZATION SERVER RECHECK
        # IDENTITY_CLEAN_GUARD_V0_1
        # ====================================================

        latest_session = (
            get_preparation_session(
                request.workflow_id
            )
        )


        latest_lock_reason = (
            _mutation_lock_reason(
                workflow_id=
                    request.workflow_id,

                session=
                    latest_session,
            )
        )


        if (
            latest_lock_reason
            is not None
        ):
            raise ValueError(
                latest_lock_reason
            )


        transformation = (
            create_surrogate_row_key(
                artifact.dataframe,

                column_name=
                    surrogate_column,
            )
        )


        output_dataset_id = (
            _output_dataset_id(
                expected_request_id
            )
        )


        output_filename = (
            _output_dataset_filename(
                source_filename=
                    artifact.dataset_filename,

                surrogate_column=
                    surrogate_column,
            )
        )


        output_info = (
            put_preparation_artifact(
                workflow_id=
                    request.workflow_id,

                dataset_id=
                    output_dataset_id,

                dataset_filename=
                    output_filename,

                stage=
                    "transform",

                dataframe=
                    transformation.dataframe,

                parent_dataset_ids=[
                    artifact.dataset_id,
                ],

                evidence_refs=[
                    (
                        "dataset_identity:"
                        f"{DATASET_IDENTITY_RULE_VERSION}"
                    ),

                    (
                        "identity_request:"
                        f"{expected_request_id}"
                    ),

                    (
                        "surrogate_column:"
                        f"{surrogate_column}"
                    ),

                    "identity_user_approval:true",
                ],

                replace=
                    False,
            )
        )


        clear_identity_resolution(
            workflow_id=
                request.workflow_id,

            dataset_id=
                artifact.dataset_id,
        )


        updated_session = (
            record_optional_stage_signal(
                workflow_id=
                    request.workflow_id,

                stage=
                    PreparationStage.TRANSFORM,

                required=
                    True,

                completed=
                    True,

                review_required=
                    False,

                blocked=
                    False,

                dataset_ids=[
                    output_dataset_id,
                ],

                evidence_refs=[
                    (
                        "dataset_identity:"
                        f"{DATASET_IDENTITY_RULE_VERSION}"
                    ),

                    (
                        "identity_request:"
                        f"{expected_request_id}"
                    ),

                    (
                        "identity_output:"
                        f"{output_dataset_id}"
                    ),

                    "identity_user_approval:true",
                ],

                blocking_reasons=[],
            )
        )


        return (
            PreparationIdentityCreateSurrogateResponse(
                workflow_id=
                    request.workflow_id,

                source_dataset_id=
                    artifact.dataset_id,

                source_dataset_filename=
                    artifact.dataset_filename,

                output_dataset_id=
                    output_info.dataset_id,

                output_dataset_filename=
                    output_info.dataset_filename,

                surrogate_column=
                    surrogate_column,

                rows=
                    output_info.rows,

                columns=
                    output_info.columns,

                parent_dataset_ids=
                    list(
                        output_info
                        .parent_dataset_ids
                    ),

                report_before=
                    report,

                session=
                    updated_session,
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

                    dataset_id=
                        request.dataset_id,
                )
            ),
        ) from error


    except (
        PreparationArtifactWorkflowNotFoundError,
        PreparationArtifactDatasetNotFoundError,
    ) as error:
        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,

            detail=(
                _error_detail(
                    code=
                        "preparation_identity_artifact_not_found",

                    error=
                        error,

                    workflow_id=
                        request.workflow_id,

                    dataset_id=
                        request.dataset_id,
                )
            ),
        ) from error


    except PreparationArtifactStoreError as error:
        raise HTTPException(
            status_code=
                status.HTTP_409_CONFLICT,

            detail=(
                _error_detail(
                    code=
                        "preparation_identity_artifact_conflict",

                    error=
                        error,

                    workflow_id=
                        request.workflow_id,

                    dataset_id=
                        request.dataset_id,
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
                        "preparation_identity_surrogate_rejected",

                    error=
                        error,

                    workflow_id=
                        request.workflow_id,

                    dataset_id=
                        request.dataset_id,
                )
            ),
        ) from error


    except RuntimeError as error:
        raise HTTPException(
            status_code=
                status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=(
                _error_detail(
                    code=
                        "preparation_identity_internal_error",

                    error=
                        error,

                    workflow_id=
                        request.workflow_id,

                    dataset_id=
                        request.dataset_id,
                )
            ),
        ) from error
