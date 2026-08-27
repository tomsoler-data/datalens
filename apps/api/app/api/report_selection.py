from __future__ import annotations


from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Response,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from app.reporting.analysis_artifact_store import (
    AnalysisArtifactDetailListResponse,
    AnalysisArtifactListResponse,
    AnalysisArtifactNotFoundError,
    list_analysis_artifact_details,
    list_analysis_artifact_summaries,
)

from app.reporting.selected_report_pdf import (
    build_selected_report_filename,
    build_selected_report_pdf,
)

from app.reporting.report_selection_store import (
    ReportSelectionDetailResponse,
    ReportSelectionIntegrityError,
    ReportSelectionNotExecutableError,
    ReportSelectionReorderError,
    ReportSelectionState,
    add_analysis_to_report,
    get_report_selection,
    get_report_selection_details,
    remove_analysis_from_report,
    reorder_report_selection,
)


router = APIRouter()


# ============================================================
# REQUESTS
# ============================================================

class ReportSelectionMutationRequest(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    workflow_id: str = Field(
        min_length=1
    )

    analysis_id: str = Field(
        min_length=1
    )


class ReportSelectionReorderRequest(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    workflow_id: str = Field(
        min_length=1
    )

    analysis_ids: list[
        str
    ]


class ReportPdfExportRequest(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    workflow_id: str = Field(
        min_length=1
    )


# ============================================================
# ERROR MAPPING
# ============================================================

def _raise_store_error(
    error: Exception,
) -> None:
    if isinstance(
        error,
        AnalysisArtifactNotFoundError,
    ):
        raise HTTPException(
            status_code=404,
            detail={
                "error":
                    "analysis_artifact_not_found",

                "message":
                    str(
                        error
                    ),

                "workflow_id":
                    error.workflow_id,

                "analysis_id":
                    error.analysis_id,
            },
        ) from error


    if isinstance(
        error,
        ReportSelectionNotExecutableError,
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "error":
                    "analysis_not_executable_for_report",

                "message":
                    str(
                        error
                    ),
            },
        ) from error


    if isinstance(
        error,
        ReportSelectionReorderError,
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "error":
                    "report_selection_reorder_rejected",

                "message":
                    str(
                        error
                    ),
            },
        ) from error


    if isinstance(
        error,
        ReportSelectionIntegrityError,
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "error":
                    "report_selection_integrity_error",

                "message":
                    str(
                        error
                    ),
            },
        ) from error


    raise error


# ============================================================
# AVAILABLE SERVER-OWNED ANALYSES
# ============================================================

@router.get(
    "/report/analyses",
    response_model=
        AnalysisArtifactListResponse,
)
def read_report_available_analyses(
    response: Response,

    workflow_id: str = Query(
        ...,
        min_length=1,
    ),
) -> AnalysisArtifactListResponse:
    response.headers[
        "Cache-Control"
    ] = "no-store"


    return (
        list_analysis_artifact_summaries(
            workflow_id=
                workflow_id
        )
    )



@router.get(
    "/report/analyses/details",
    response_model=
        AnalysisArtifactDetailListResponse,
)
def read_report_available_analysis_details(
    response: Response,

    workflow_id: str = Query(
        ...,
        min_length=1,
    ),
) -> AnalysisArtifactDetailListResponse:
    response.headers[
        "Cache-Control"
    ] = "no-store"


    return (
        list_analysis_artifact_details(
            workflow_id=
                workflow_id
        )
    )


# ============================================================
# CURRENT SELECTION
# ============================================================

@router.get(
    "/report/selection",
    response_model=
        ReportSelectionState,
)
def read_report_selection(
    response: Response,

    workflow_id: str = Query(
        ...,
        min_length=1,
    ),
) -> ReportSelectionState:
    response.headers[
        "Cache-Control"
    ] = "no-store"


    try:
        return (
            get_report_selection(
                workflow_id=
                    workflow_id
            )
        )


    except Exception as error:
        _raise_store_error(
            error
        )

        raise


@router.get(
    "/report/selection/details",
    response_model=
        ReportSelectionDetailResponse,
)
def read_report_selection_details(
    response: Response,

    workflow_id: str = Query(
        ...,
        min_length=1,
    ),
) -> ReportSelectionDetailResponse:
    response.headers[
        "Cache-Control"
    ] = "no-store"


    try:
        return (
            get_report_selection_details(
                workflow_id=
                    workflow_id
            )
        )


    except Exception as error:
        _raise_store_error(
            error
        )

        raise


# ============================================================
# ADD / REMOVE
# ============================================================

@router.post(
    "/report/selection/add",
    response_model=
        ReportSelectionState,
)
def add_report_analysis(
    request:
        ReportSelectionMutationRequest,
) -> ReportSelectionState:
    try:
        return (
            add_analysis_to_report(
                workflow_id=
                    request.workflow_id,

                analysis_id=
                    request.analysis_id,
            )
        )


    except Exception as error:
        _raise_store_error(
            error
        )

        raise


@router.post(
    "/report/selection/remove",
    response_model=
        ReportSelectionState,
)
def remove_report_analysis(
    request:
        ReportSelectionMutationRequest,
) -> ReportSelectionState:
    try:
        return (
            remove_analysis_from_report(
                workflow_id=
                    request.workflow_id,

                analysis_id=
                    request.analysis_id,
            )
        )


    except Exception as error:
        _raise_store_error(
            error
        )

        raise


# ============================================================
# REORDER
# ============================================================

@router.post(
    "/report/selection/reorder",
    response_model=
        ReportSelectionState,
)
def reorder_report_analyses(
    request:
        ReportSelectionReorderRequest,
) -> ReportSelectionState:
    try:
        return (
            reorder_report_selection(
                workflow_id=
                    request.workflow_id,

                analysis_ids=
                    request.analysis_ids,
            )
        )


    except Exception as error:
        _raise_store_error(
            error
        )

        raise


# ============================================================
# SERVER-OWNED PDF EXPORT
# ============================================================

@router.post(
    "/report/export-pdf",
)
def export_selected_report_pdf(
    request:
        ReportPdfExportRequest,
) -> Response:
    """
    Generate the final prompt-analysis PDF from server-owned
    report selection only.

    The browser sends workflow_id and nothing else.

    The server resolves:
    - selected analysis ids;
    - their stable report order;
    - their persisted native pipeline payloads;
    - deterministic result metrics and chart data.
    """

    try:
        selection = (
            get_report_selection_details(
                workflow_id=
                    request.workflow_id
            )
        )


        if (
            selection.selected_count
            ==
            0
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "error":
                        "report_selection_empty",

                    "message":
                        (
                            "Aucune analyse exécutée n'est "
                            "sélectionnée pour le rapport."
                        ),

                    "workflow_id":
                        request.workflow_id,
                },
            )


        pdf_bytes = (
            build_selected_report_pdf(
                selection
            )
        )


    except HTTPException:
        raise


    except Exception as error:
        _raise_store_error(
            error
        )


        raise HTTPException(
            status_code=500,
            detail={
                "error":
                    "server_owned_pdf_generation_failed",

                "message":
                    (
                        "La génération locale du PDF "
                        "server-owned a échoué : "
                        f"{type(error).__name__}: {error}"
                    ),

                "workflow_id":
                    request.workflow_id,
            },
        ) from error


    return Response(
        content=
            pdf_bytes,

        media_type=
            "application/pdf",

        headers={
            "Content-Disposition":
                (
                    'attachment; filename="'
                    +
                    build_selected_report_filename()
                    +
                    '"'
                ),

            "Cache-Control":
                "no-store",

            "X-DataLens-Report-Selection-Count":
                str(
                    selection.selected_count
                ),

            "X-DataLens-Report-Selection-Revision":
                str(
                    selection.revision
                ),
        },
    )
