from __future__ import annotations


from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)


from app.api.routes import (
    load_uploaded_dataset_bundle,
)

from app.preparation.data_quality import (
    DataQualityReport,
    build_data_quality_report,
)

from app.preparation.preparation_session import (
    PreparationSessionNotFoundError,
    get_preparation_session,
    record_required_stage_signal,
)

from app.preparation.preparation_ui_state import (
    update_preparation_ui_state,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
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
# HELPERS
# ============================================================


def _dataset_ids_from_records(
    source_dataset_records: list[
        dict
    ],
) -> list[
    str
]:
    dataset_ids: list[
        str
    ] = []


    for record in (
        source_dataset_records
    ):
        dataset_id = str(
            record.get(
                "dataset_id",
                "",
            )
        ).strip()


        if not (
            dataset_id
        ):
            raise RuntimeError(
                (
                    "Preparation quality received "
                    "an internal dataset record "
                    "without dataset_id."
                )
            )


        dataset_ids.append(
            dataset_id
        )


    if not (
        dataset_ids
    ):
        raise RuntimeError(
            (
                "Preparation quality received "
                "no internal dataset records."
            )
        )


    return dataset_ids


def _validate_session_dataset_scope(
    *,
    workflow_id: str,
    uploaded_dataset_ids: list[
        str
    ],
) -> None:
    """
    Verify that the uploaded dataset scope corresponds to
    the server-owned preparation session.

    v0.1 uses the deterministic DataLens dataset IDs
    generated during ingestion:

        dataset:0001
        dataset:0002
        ...

    The browser cannot change the session's selected dataset
    IDs after creation.
    """

    session = (
        get_preparation_session(
            workflow_id
        )
    )


    expected_dataset_ids = list(
        session
        .selected_analysis_dataset_ids
    )


    if (
        uploaded_dataset_ids
        !=
        expected_dataset_ids
    ):
        raise HTTPException(
            status_code=409,

            detail={
                "error": (
                    "preparation_dataset_scope_mismatch"
                ),

                "message": (
                    "The uploaded datasets do not match "
                    "the preparation session dataset scope."
                ),

                "workflow_id": (
                    workflow_id
                ),

                "expected_dataset_ids": (
                    expected_dataset_ids
                ),

                "uploaded_dataset_ids": (
                    uploaded_dataset_ids
                ),
            },
        )


# ============================================================
# QUALITY
# ============================================================


@router.post(
    "/quality",
    response_model=
        DataQualityReport,
)
def analyze_uploaded_dataset_quality(
    dataset_files: list[
        UploadFile
    ] = File(
        ...,
    ),

    workflow_id: str = Form(
        ...,
        min_length=1,
    ),
) -> DataQualityReport:
    """
    Run the deterministic DataLens quality engine
    against one or more uploaded CSV datasets.

    Preparation Session synchronization:

    1. successful CSV ingestion
       -> IMPORT evidence;

    2. successful manifest/profile construction
       -> UNDERSTAND evidence;

    3. successful deterministic quality report
       -> QUALITY evidence.

    The endpoint:
    - reuses the standard DataLens CSV ingestion path;
    - performs deterministic quality diagnostics;
    - returns evidence-backed cleaning proposals;
    - does not modify the uploaded data;
    - does not call an LLM;
    - never accepts stage statuses from the client.

    Semantic review is represented only as a recommendation
    in the report. A later guarded AI layer may interpret
    those candidates, but Python remains responsible for
    validation and execution.
    """

    try:
        (
            dataset_ingestion,
            source_dataset_records,
        ) = load_uploaded_dataset_bundle(
            dataset_files
        )


        dataset_ids = (
            _dataset_ids_from_records(
                source_dataset_records
            )
        )


        # ====================================================
        # SESSION SCOPE
        # ====================================================

        _validate_session_dataset_scope(
            workflow_id=
                workflow_id,

            uploaded_dataset_ids=
                dataset_ids,
        )


        # ====================================================
        # IMPORT
        # ====================================================

        record_required_stage_signal(
            workflow_id=
                workflow_id,

            stage=
                PreparationStage.IMPORT,

            completed=
                True,

            dataset_ids=
                dataset_ids,

            evidence_refs=[
                (
                    "dataset_ingestion:"
                    f"{dataset_ingestion.dataset_count}"
                    "_dataset(s)"
                )
            ],

            blocking_reasons=[],
        )


        # ====================================================
        # UNDERSTAND
        # ====================================================

        record_required_stage_signal(
            workflow_id=
                workflow_id,

            stage=
                PreparationStage.UNDERSTAND,

            completed=
                True,

            dataset_ids=
                dataset_ids,

            evidence_refs=[
                (
                    "dataset_manifest_profile:"
                    f"{dataset_ingestion.total_rows}"
                    "_row(s)"
                )
            ],

            blocking_reasons=[],
        )


        # ====================================================
        # QUALITY
        # ====================================================

        quality_report = (
            build_data_quality_report(
                source_dataset_records
            )
        )


        record_required_stage_signal(
            workflow_id=
                workflow_id,

            stage=
                PreparationStage.QUALITY,

            completed=
                True,

            dataset_ids=
                dataset_ids,

            evidence_refs=[
                (
                    "data_quality:"
                    f"{quality_report.rule_version}"
                )
            ],

            blocking_reasons=[],
        )


        # PREPARATION_UI_STATE_WRITE_V0_1:QUALITY
        update_preparation_ui_state(
            workflow_id=
                workflow_id,

            quality_report=(
                quality_report.model_dump(
                    mode="json"
                )
            ),

            cleaning_plan=None,
            cleaning_execution=None,

            semantic_review=None,
            semantic_cleaning_plan=None,
            semantic_cleaning_execution=None,
            semantic_confirmation=None,

            applied_semantic_choices=[],

            confirmed_semantic_issue_ids=[],

            semantic_manual_resolutions=[],
        )


        return (
            quality_report
        )


    except PreparationSessionNotFoundError as error:
        raise HTTPException(
            status_code=404,

            detail={
                "error": (
                    "preparation_session_not_found"
                ),

                "message": str(
                    error
                ),

                "workflow_id": (
                    workflow_id
                ),
            },
        ) from error


    except HTTPException:
        raise


    except ValueError as error:
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
                "Data quality engine received "
                "an invalid internal dataset record: "
                f"{error}"
            ),
        ) from error


    except RuntimeError as error:
        raise HTTPException(
            status_code=500,

            detail=(
                "Preparation quality synchronization "
                "failed: "
                f"{error}"
            ),
        ) from error