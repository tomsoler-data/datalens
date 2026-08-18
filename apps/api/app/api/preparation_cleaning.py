from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import re
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from fastapi.responses import (
    Response,
)

from pydantic import (
    BaseModel,
    Field,
)


from app.api.routes import (
    load_uploaded_dataset_bundle,
)

from app.preparation.cleaning_engine import (
    CleaningExecutionResult,
    CleaningPlan,
    build_cleaning_plan,
    execute_cleaning_plan,
)

from app.preparation.cleaning_artifacts import (
    materialize_cleaning_execution_artifacts,
    materialize_skipped_cleaning_artifacts,
)

from app.preparation.data_quality import (
    DataQualityReport,
    build_data_quality_report,
)

from app.preparation.preparation_session import (
    PreparationSessionNotFoundError,
    get_preparation_session,
    record_optional_stage_signal,
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
# RESPONSE MODELS
# ============================================================


class DerivedDatasetPreview(
    BaseModel,
):
    dataset_id: str
    dataset_filename: str

    rows_before: int
    rows_after: int

    columns_before: int
    columns_after: int

    preview_rows: list[
        dict[
            str,
            Any,
        ]
    ] = Field(
        default_factory=list,
    )


class CleaningApplyResponse(
    BaseModel,
):
    status: str

    quality_report: DataQualityReport

    cleaning_plan: CleaningPlan

    execution: CleaningExecutionResult

    derived_datasets: list[
        DerivedDatasetPreview
    ]

    notes: list[str]


# ============================================================
# DATASET HELPERS
# ============================================================


def _dataset_frames(
    source_dataset_records: list[
        dict[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    pd.DataFrame,
]:
    frames: dict[
        str,
        pd.DataFrame,
    ] = {}


    for record in (
        source_dataset_records
    ):
        dataset_id = str(
            record.get(
                "dataset_id"
            )
            or
            ""
        ).strip()


        dataframe = record.get(
            "dataframe"
        )


        if not dataset_id:
            raise ValueError(
                "Internal dataset record "
                "is missing dataset_id."
            )


        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "Internal dataset record "
                f"{dataset_id} is missing "
                "its pandas DataFrame."
            )


        if dataset_id in frames:
            raise ValueError(
                "Duplicate internal dataset_id: "
                f"{dataset_id}"
            )


        frames[
            dataset_id
        ] = dataframe


    return frames


def _dataset_filename_map(
    source_dataset_records: list[
        dict[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    str,
]:
    return {
        str(
            record.get(
                "dataset_id"
            )
        ):
            str(
                record.get(
                    "filename"
                )
                or
                record.get(
                    "dataset_id"
                )
            )

        for record in
        source_dataset_records
    }


def _dataset_ids_from_records(
    source_dataset_records: list[
        dict[
            str,
            Any,
        ]
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
                "dataset_id"
            )
            or
            ""
        ).strip()


        if not dataset_id:
            raise RuntimeError(
                (
                    "Preparation cleaning received "
                    "an internal dataset record "
                    "without dataset_id."
                )
            )


        dataset_ids.append(
            dataset_id
        )


    if not dataset_ids:
        raise RuntimeError(
            (
                "Preparation cleaning received "
                "no internal dataset records."
            )
        )


    return dataset_ids


# ============================================================
# SESSION SCOPE
# ============================================================


def _validate_session_dataset_scope(
    *,
    workflow_id: str,

    source_dataset_records: list[
        dict[
            str,
            Any,
        ]
    ],
) -> list[
    str
]:
    """
    Verify that the uploaded datasets correspond exactly to
    the server-owned preparation session scope.

    Dataset IDs are deterministic and positional:

        dataset:0001
        dataset:0002
        ...

    Therefore order is intentionally checked as well.
    """

    session = (
        get_preparation_session(
            workflow_id
        )
    )


    uploaded_dataset_ids = (
        _dataset_ids_from_records(
            source_dataset_records
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


    return (
        uploaded_dataset_ids
    )


# ============================================================
# SESSION â€” CLEANING PLAN
# ============================================================


def _cleaning_plan_review_reasons(
    cleaning_plan: CleaningPlan,
) -> list[
    str
]:
    reasons: list[
        str
    ] = []


    if (
        cleaning_plan.action_count >
        0
    ):
        reasons.append(
            (
                "Le plan de nettoyage contient "
                f"{cleaning_plan.action_count} "
                "correction(s) dÃ©terministe(s) "
                "Ã  valider."
            )
        )


    if (
        cleaning_plan.protected_issue_count >
        0
    ):
        reasons.append(
            (
                "Le plan conserve "
                f"{cleaning_plan.protected_issue_count} "
                "problÃ¨me(s) protÃ©gÃ©(s) nÃ©cessitant "
                "une revue sÃ©mantique ou analyste."
            )
        )


    return reasons


def _record_cleaning_plan_stage(
    *,
    workflow_id: str,

    dataset_ids: list[
        str
    ],

    cleaning_plan: CleaningPlan,
) -> None:
    """
    Derive CLEAN state from the deterministic cleaning plan.

    No action required and no protected issue:
        CLEAN = SKIPPED

    Otherwise:
        CLEAN = REVIEW_REQUIRED
    """

    evidence_refs = [
        (
            "cleaning_plan:"
            f"{cleaning_plan.rule_version}"
        ),

        (
            "cleaning_plan_actions:"
            f"{cleaning_plan.action_count}"
        ),

        (
            "cleaning_plan_blocked_issues:"
            f"{cleaning_plan.protected_issue_count}"
        ),
    ]


    if (
        cleaning_plan.action_count
        ==
        0
        and
        cleaning_plan.protected_issue_count
        ==
        0
    ):
        record_optional_stage_signal(
            workflow_id=
                workflow_id,

            stage=
                PreparationStage.CLEAN,

            required=
                False,

            completed=
                False,

            review_required=
                False,

            blocked=
                False,

            dataset_ids=
                dataset_ids,

            evidence_refs=
                evidence_refs,

            blocking_reasons=[],
        )

        return


    record_optional_stage_signal(
        workflow_id=
            workflow_id,

        stage=
            PreparationStage.CLEAN,

        required=
            True,

        completed=
            False,

        review_required=
            True,

        blocked=
            False,

        dataset_ids=
            dataset_ids,

        evidence_refs=
            evidence_refs,

        blocking_reasons=(
            _cleaning_plan_review_reasons(
                cleaning_plan
            )
        ),
    )


# ============================================================
# SESSION â€” CLEANING EXECUTION
# ============================================================


def _record_cleaning_execution_stage(
    *,
    workflow_id: str,

    dataset_ids: list[
        str
    ],

    cleaning_plan: CleaningPlan,

    execution: CleaningExecutionResult,
) -> None:
    """
    Derive CLEAN after deterministic execution.

    Rules:

    - no cleaning needed
        -> SKIPPED;

    - execution contains blocked actions
        -> BLOCKED;

    - protected quality issues remain
        -> REVIEW_REQUIRED;

    - otherwise the analyst decision and deterministic
      execution are complete
        -> PASSED.

    skipped_action_count does not automatically block the
    stage: actions omitted from an explicit approval request
    remain recorded in provenance as skipped.
    """

    evidence_refs = [
        (
            "cleaning_plan:"
            f"{cleaning_plan.rule_version}"
        ),

        (
            "cleaning_execution:"
            f"{execution.rule_version}"
        ),

        (
            "cleaning_applied_actions:"
            f"{execution.applied_action_count}"
        ),

        (
            "cleaning_skipped_actions:"
            f"{execution.skipped_action_count}"
        ),

        (
            "cleaning_blocked_actions:"
            f"{execution.blocked_action_count}"
        ),
    ]


    # ========================================================
    # NOTHING REQUIRED
    # ========================================================

    if (
        cleaning_plan.action_count
        ==
        0
        and
        cleaning_plan.protected_issue_count
        ==
        0
    ):
        record_optional_stage_signal(
            workflow_id=
                workflow_id,

            stage=
                PreparationStage.CLEAN,

            required=
                False,

            completed=
                False,

            review_required=
                False,

            blocked=
                False,

            dataset_ids=
                dataset_ids,

            evidence_refs=
                evidence_refs,

            blocking_reasons=[],
        )

        return


    # ========================================================
    # EXECUTION BLOCKED
    # ========================================================

    if (
        execution.blocked_action_count >
        0
    ):
        record_optional_stage_signal(
            workflow_id=
                workflow_id,

            stage=
                PreparationStage.CLEAN,

            required=
                True,

            completed=
                False,

            review_required=
                False,

            blocked=
                True,

            dataset_ids=
                dataset_ids,

            evidence_refs=
                evidence_refs,

            blocking_reasons=[
                (
                    "Lâ€™exÃ©cution du nettoyage contient "
                    f"{execution.blocked_action_count} "
                    "action(s) bloquÃ©e(s)."
                )
            ],
        )

        return


    # ========================================================
    # PROTECTED ISSUES STILL REQUIRE REVIEW
    # ========================================================

    if (
        cleaning_plan.protected_issue_count >
        0
    ):
        record_optional_stage_signal(
            workflow_id=
                workflow_id,

            stage=
                PreparationStage.CLEAN,

            required=
                True,

            completed=
                False,

            review_required=
                True,

            blocked=
                False,

            dataset_ids=
                dataset_ids,

            evidence_refs=
                evidence_refs,

            blocking_reasons=[
                (
                    "Le nettoyage dÃ©terministe a Ã©tÃ© "
                    "exÃ©cutÃ©, mais "
                    f"{cleaning_plan.protected_issue_count} "
                    "problÃ¨me(s) protÃ©gÃ©(s) nÃ©cessitent "
                    "encore une revue sÃ©mantique ou "
                    "analyste."
                )
            ],
        )

        return


    # ========================================================
    # CLEANING RESOLVED
    # ========================================================

    record_optional_stage_signal(
        workflow_id=
            workflow_id,

        stage=
            PreparationStage.CLEAN,

        required=
            True,

        completed=
            True,

        review_required=
            False,

        blocked=
            False,

        dataset_ids=
            dataset_ids,

        evidence_refs=
            evidence_refs,

        blocking_reasons=[],
    )


# ============================================================
# PREVIEW
# ============================================================


def _json_safe_preview(
    dataframe: pd.DataFrame,
    *,
    limit: int = 5,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    preview = (
        dataframe
        .head(
            limit
        )
        .copy()
    )


    preview = preview.astype(
        object
    ).where(
        pd.notna(
            preview
        ),
        None,
    )


    rows = preview.to_dict(
        orient="records"
    )


    safe_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for row in rows:
        safe_row: dict[
            str,
            Any,
        ] = {}


        for key, value in (
            row.items()
        ):
            if isinstance(
                value,
                pd.Timestamp,
            ):
                safe_row[
                    str(
                        key
                    )
                ] = value.isoformat()

            elif hasattr(
                value,
                "item",
            ):
                try:
                    safe_row[
                        str(
                            key
                        )
                    ] = value.item()

                except (
                    ValueError,
                    TypeError,
                ):
                    safe_row[
                        str(
                            key
                        )
                    ] = str(
                        value
                    )

            else:
                safe_row[
                    str(
                        key
                    )
                ] = value


        safe_rows.append(
            safe_row
        )


    return safe_rows


# ============================================================
# APPROVED IDS
# ============================================================


def _parse_approved_action_ids(
    raw_value: str,
) -> set[
    str
]:
    try:
        parsed = json.loads(
            raw_value
        )

    except json.JSONDecodeError as error:
        raise ValueError(
            "approved_action_ids_json "
            "must be a valid JSON array."
        ) from error


    if not isinstance(
        parsed,
        list,
    ):
        raise ValueError(
            "approved_action_ids_json "
            "must contain a JSON array."
        )


    action_ids: set[
        str
    ] = set()


    for value in parsed:
        if not isinstance(
            value,
            str,
        ):
            raise ValueError(
                "Every approved action id "
                "must be a string."
            )


        normalized = value.strip()


        if not normalized:
            raise ValueError(
                "Approved action ids "
                "cannot be empty."
            )


        action_ids.add(
            normalized
        )


    return action_ids


# ============================================================
# QUALITY + PLAN
# ============================================================


def _build_quality_and_plan(
    dataset_files: list[
        UploadFile
    ],
) -> tuple[
    list[
        dict[
            str,
            Any,
        ]
    ],
    dict[
        str,
        pd.DataFrame,
    ],
    DataQualityReport,
    CleaningPlan,
]:
    (
        _,
        source_dataset_records,
    ) = load_uploaded_dataset_bundle(
        dataset_files
    )


    quality_report = (
        build_data_quality_report(
            source_dataset_records
        )
    )


    frames = _dataset_frames(
        source_dataset_records
    )


    cleaning_plan = (
        build_cleaning_plan(
            quality_report,
            frames,
        )
    )


    return (
        source_dataset_records,
        frames,
        quality_report,
        cleaning_plan,
    )


# ============================================================
# EXPORT HELPERS
# ============================================================


def _prepared_filename(
    filename: str,
) -> str:
    source_name = (
        Path(
            filename
        ).name
    )


    stem = (
        Path(
            source_name
        ).stem
    )


    safe_stem = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        stem,
    ).strip(
        "._-"
    )


    if not safe_stem:
        safe_stem = (
            "datalens_dataset"
        )


    return (
        f"{safe_stem}_prepared.csv"
    )


def _dataframe_to_csv_bytes(
    dataframe: pd.DataFrame,
) -> bytes:
    """
    Export the exact derived DataFrame used after controlled
    cleaning.

    UTF-8 with BOM is used so the result opens cleanly in
    spreadsheet software while remaining a standard CSV.
    """

    csv_text = dataframe.to_csv(
        index=False,
        lineterminator="\n",
    )


    return csv_text.encode(
        "utf-8-sig"
    )


def _build_prepared_export_response(
    *,
    source_dataset_records: list[
        dict[
            str,
            Any,
        ]
    ],

    derived_frames: dict[
        str,
        pd.DataFrame,
    ],

    execution: CleaningExecutionResult,
) -> Response:
    filename_map = (
        _dataset_filename_map(
            source_dataset_records
        )
    )


    headers = {
        "Cache-Control":
            "no-store",

        "X-DataLens-Cleaning-Rule":
            execution.rule_version,

        "X-DataLens-Applied-Actions":
            str(
                execution
                .applied_action_count
            ),
    }


    if (
        len(
            derived_frames
        )
        ==
        1
    ):
        (
            dataset_id,
            dataframe,
        ) = next(
            iter(
                derived_frames.items()
            )
        )


        filename = (
            _prepared_filename(
                filename_map.get(
                    dataset_id,
                    dataset_id,
                )
            )
        )


        headers[
            "Content-Disposition"
        ] = (
            "attachment; "
            f'filename="{filename}"'
        )


        return Response(
            content=
                _dataframe_to_csv_bytes(
                    dataframe
                ),

            media_type=
                "text/csv; charset=utf-8",

            headers=
                headers,
        )


    buffer = BytesIO()


    with ZipFile(
        buffer,
        mode="w",
        compression=
            ZIP_DEFLATED,
    ) as archive:
        for (
            dataset_id,
            dataframe,
        ) in derived_frames.items():
            filename = (
                _prepared_filename(
                    filename_map.get(
                        dataset_id,
                        dataset_id,
                    )
                )
            )


            archive.writestr(
                filename,
                _dataframe_to_csv_bytes(
                    dataframe
                ),
            )


    headers[
        "Content-Disposition"
    ] = (
        "attachment; "
        'filename="datalens_prepared_datasets.zip"'
    )


    return Response(
        content=
            buffer.getvalue(),

        media_type=
            "application/zip",

        headers=
            headers,
    )


# ============================================================
# CLEANING PLAN
# ============================================================


@router.post(
    "/cleaning-plan",
    response_model=
        CleaningPlan,
)
def build_uploaded_cleaning_plan(
    dataset_files: list[
        UploadFile
    ] = File(
        ...,
    ),

    workflow_id: str = Form(
        ...,
        min_length=1,
    ),
) -> CleaningPlan:
    """
    Build a deterministic cleaning plan from uploaded CSV files.

    This endpoint modifies no DataFrame.

    It does synchronize the server-owned preparation workflow:

    - no cleaning required -> CLEAN = SKIPPED;
    - cleaning decision required -> CLEAN = REVIEW_REQUIRED.
    """

    try:
        (
            source_dataset_records,
            _,
            _,
            cleaning_plan,
        ) = _build_quality_and_plan(
            dataset_files
        )


        dataset_ids = (
            _validate_session_dataset_scope(
                workflow_id=
                    workflow_id,

                source_dataset_records=
                    source_dataset_records,
            )
        )


        # ====================================================
        # MATERIALIZE SOURCE WHEN DETERMINISTIC CLEANING
        # HAS NOTHING TO EXECUTE
        #
        # This may still happen while protected semantic
        # issues remain. Artifact existence does NOT mean
        # CLEAN is authorized for downstream use.
        # PreparationSession remains the readiness authority.
        # ====================================================

        if (
            cleaning_plan.action_count
            ==
            0
        ):
            materialize_skipped_cleaning_artifacts(
                workflow_id=
                    workflow_id,

                source_dataset_records=
                    source_dataset_records,

                cleaning_plan=
                    cleaning_plan,
            )


        _record_cleaning_plan_stage(
            workflow_id=
                workflow_id,

            dataset_ids=
                dataset_ids,

            cleaning_plan=
                cleaning_plan,
        )


        return (
            cleaning_plan
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
                "Cleaning planner received "
                "an invalid internal dataset: "
                f"{error}"
            ),
        ) from error


    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Cleaning workflow synchronization "
                "failed: "
                f"{error}"
            ),
        ) from error


# ============================================================
# CLEANING APPLY
# ============================================================


@router.post(
    "/cleaning-apply",
    response_model=
        CleaningApplyResponse,
)
def apply_uploaded_cleaning_plan(
    dataset_files: list[
        UploadFile
    ] = File(
        ...,
    ),

    approved_action_ids_json: str = Form(
        ...,
    ),

    workflow_id: str = Form(
        ...,
        min_length=1,
    ),
) -> CleaningApplyResponse:
    """
    Recompute the deterministic plan and execute only the exact
    safe action IDs explicitly approved by the user.

    The browser never sends CLEAN = PASSED.

    The backend derives CLEAN from the execution result.
    """

    try:
        (
            source_dataset_records,
            source_frames,
            quality_report,
            cleaning_plan,
        ) = _build_quality_and_plan(
            dataset_files
        )


        dataset_ids = (
            _validate_session_dataset_scope(
                workflow_id=
                    workflow_id,

                source_dataset_records=
                    source_dataset_records,
            )
        )


        approved_action_ids = (
            _parse_approved_action_ids(
                approved_action_ids_json
            )
        )


        if (
            cleaning_plan.action_count >
            0
            and
            not approved_action_ids
        ):
            raise ValueError(
                (
                    "At least one cleaning action "
                    "must be explicitly approved "
                    "before cleaning execution."
                )
            )


        (
            derived_frames,
            execution,
        ) = execute_cleaning_plan(
            plan=
                cleaning_plan,
            dataset_frames=
                source_frames,
            approved_action_ids=
                approved_action_ids,
        )


        # ====================================================
        # MATERIALIZE THE EXACT CLEANING RESULT BEFORE
        # CHANGING THE LOGICAL WORKFLOW STATE
        #
        # Important trust boundary:
        #
        #     material artifact
        #         BEFORE
        #     PreparationSession stage update
        #
        # Therefore CLEAN can never become completed while its
        # corresponding server-owned DataFrame was lost.
        #
        # A protected semantic issue may still keep CLEAN in
        # REVIEW_REQUIRED. The artifact remains useful as the
        # input to the semantic-cleaning substage.
        # ====================================================

        if (
            cleaning_plan.action_count
            ==
            0
        ):
            materialize_skipped_cleaning_artifacts(
                workflow_id=
                    workflow_id,

                source_dataset_records=
                    source_dataset_records,

                cleaning_plan=
                    cleaning_plan,
            )

        elif (
            execution.blocked_action_count
            ==
            0
        ):
            materialize_cleaning_execution_artifacts(
                workflow_id=
                    workflow_id,

                source_dataset_records=
                    source_dataset_records,

                cleaning_plan=
                    cleaning_plan,

                derived_frames=
                    derived_frames,

                execution=
                    execution,
            )


        _record_cleaning_execution_stage(
            workflow_id=
                workflow_id,

            dataset_ids=
                dataset_ids,

            cleaning_plan=
                cleaning_plan,

            execution=
                execution,
        )


        filename_map = (
            _dataset_filename_map(
                source_dataset_records
            )
        )


        derived_datasets: list[
            DerivedDatasetPreview
        ] = []


        provenance_by_dataset = {
            item.dataset_id:
                item

            for item in
            execution.provenance
        }


        for dataset_id, dataframe in (
            derived_frames.items()
        ):
            provenance = (
                provenance_by_dataset.get(
                    dataset_id
                )
            )


            if provenance is None:
                raise RuntimeError(
                    "Missing cleaning provenance "
                    f"for dataset {dataset_id}."
                )


            derived_datasets.append(
                DerivedDatasetPreview(
                    dataset_id=
                        dataset_id,

                    dataset_filename=
                        filename_map.get(
                            dataset_id,
                            dataset_id,
                        ),

                    rows_before=
                        provenance.rows_before,

                    rows_after=
                        provenance.rows_after,

                    columns_before=
                        provenance.columns_before,

                    columns_after=
                        provenance.columns_after,

                    preview_rows=
                        _json_safe_preview(
                            dataframe
                        ),
                )
            )


        return (
            CleaningApplyResponse(
                status=
                    "ready",

                quality_report=
                    quality_report,

                cleaning_plan=
                    cleaning_plan,

                execution=
                    execution,

                derived_datasets=
                    derived_datasets,

                notes=[
                    (
                        "Cleaning was executed only "
                        "on derived DataFrame copies."
                    ),

                    (
                        "The original uploaded files "
                        "were not modified."
                    ),

                    (
                        "The CLEAN workflow stage was "
                        "derived server-side from the "
                        "cleaning plan and execution."
                    ),

                    (
                        "Derived cleaning datasets are "
                        "materialized server-side for "
                        "downstream preparation stages."
                    ),
                ],
            )
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
                "Cleaning executor received "
                "an invalid internal dataset: "
                f"{error}"
            ),
        ) from error


    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Cleaning workflow synchronization "
                "failed: "
                f"{error}"
            ),
        ) from error


# ============================================================
# CLEANING EXPORT
# ============================================================


@router.post(
    "/cleaning-export",
)
def export_uploaded_cleaning_result(
    dataset_files: list[
        UploadFile
    ] = File(
        ...,
    ),

    approved_action_ids_json: str = Form(
        ...,
    ),
) -> Response:
    """
    Rebuild and execute the deterministic cleaning plan, then
    export the exact derived dataset(s).

    Single dataset:
        <source>_prepared.csv

    Multiple datasets:
        datalens_prepared_datasets.zip

    Safety:
    - original uploads are never modified;
    - the plan is rebuilt server-side;
    - unknown action IDs are rejected;
    - only exact safe action IDs can be executed;
    - ambiguous quality issues remain untouched.

    Export is deliberately read/derive-only with respect to
    Preparation Session state. The workflow stage is changed
    by /cleaning-plan and /cleaning-apply, not by downloads.
    """

    try:
        (
            source_dataset_records,
            source_frames,
            _,
            cleaning_plan,
        ) = _build_quality_and_plan(
            dataset_files
        )


        approved_action_ids = (
            _parse_approved_action_ids(
                approved_action_ids_json
            )
        )


        (
            derived_frames,
            execution,
        ) = execute_cleaning_plan(
            plan=
                cleaning_plan,

            dataset_frames=
                source_frames,

            approved_action_ids=
                approved_action_ids,
        )


        return (
            _build_prepared_export_response(
                source_dataset_records=
                    source_dataset_records,

                derived_frames=
                    derived_frames,

                execution=
                    execution,
            )
        )


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
                "Prepared-dataset export received "
                "an invalid internal dataset: "
                f"{error}"
            ),
        ) from error
