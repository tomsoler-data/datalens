from pathlib import (
    Path,
)

import pandas as pd

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from app.analysis import (
    AnalysisPipelineError,
    CorrelationAnalysisRun,
    run_correlation_analysis,
)

from app.ingestion import (
    MultiDatasetIngestion,
    build_dataset_manifest,
)

from app.statistics.schemas import (
    AnalysisGoal,
    AnalysisMode,
    VariableKind,
)


router = APIRouter()


SUPPORTED_DATA_EXTENSIONS = {
    ".csv",
}


MAX_DATASET_FILES = 20


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health",
)
def health_check():
    return {
        "status": "ok",
        "service": "datalens-api",
    }


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_csv_upload(
    dataset_file: UploadFile,
) -> tuple[
    str,
    str,
]:
    filename = (
        dataset_file.filename
        or ""
    ).strip()

    if not filename:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded dataset must "
                "have a filename."
            ),
        )

    extension = (
        Path(
            filename
        )
        .suffix
        .lower()
    )

    if (
        extension
        not in SUPPORTED_DATA_EXTENSIONS
    ):
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported dataset format. "
                "DataLens currently accepts "
                "CSV datasets only."
            ),
        )

    return (
        filename,
        extension,
    )


# ============================================================
# CSV LOADER
# ============================================================

def load_csv_dataframe(
    dataset_file: UploadFile,
) -> pd.DataFrame:
    try:
        dataset_file.file.seek(
            0
        )

        dataframe = pd.read_csv(
            dataset_file.file,
            encoding="utf-8-sig",
        )

    except pd.errors.EmptyDataError as error:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded CSV contains "
                "no readable data."
            ),
        ) from error

    except pd.errors.ParserError as error:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded CSV could not "
                "be parsed safely."
            ),
        ) from error

    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded CSV is not "
                "valid UTF-8 text."
            ),
        ) from error

    if dataframe.empty:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded CSV contains "
                "no data rows."
            ),
        )

    if (
        len(
            dataframe.columns
        )
        == 0
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded CSV contains "
                "no columns."
            ),
        )

    return dataframe


# ============================================================
# MULTI-DATASET INGESTION
# ============================================================

@router.post(
    "/ingestion/datasets",
    response_model=
        MultiDatasetIngestion,
)
def ingest_datasets(
    dataset_files: list[
        UploadFile
    ] = File(
        ...,
        description=(
            "One or more datasets uploaded "
            "for local DataLens ingestion."
        ),
    ),
) -> MultiDatasetIngestion:
    if (
        len(
            dataset_files
        )
        == 0
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "At least one dataset must "
                "be uploaded."
            ),
        )

    if (
        len(
            dataset_files
        )
        >
        MAX_DATASET_FILES
    ):
        raise HTTPException(
            status_code=413,
            detail=(
                "Too many datasets were "
                "uploaded in one request. "
                f"The current limit is "
                f"{MAX_DATASET_FILES}."
            ),
        )

    manifests = []

    batch_warnings: list[
        str
    ] = []

    filenames: list[
        str
    ] = []

    try:
        for (
            index,
            dataset_file,
        ) in enumerate(
            dataset_files,
            start=1,
        ):
            (
                filename,
                extension,
            ) = (
                validate_csv_upload(
                    dataset_file
                )
            )

            dataframe = (
                load_csv_dataframe(
                    dataset_file
                )
            )

            dataset_id = (
                f"dataset:{index:04d}"
            )

            manifest = (
                build_dataset_manifest(
                    dataframe,
                    dataset_id=
                        dataset_id,
                    filename=
                        filename,
                    extension=
                        extension,
                )
            )

            manifests.append(
                manifest
            )

            filenames.append(
                filename
            )

    finally:
        for dataset_file in (
            dataset_files
        ):
            dataset_file.file.close()

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

    if duplicate_filenames:
        batch_warnings.append(
            (
                "Duplicate filenames were "
                "uploaded in the same batch: "
                + ", ".join(
                    duplicate_filenames
                )
                + ". Dataset IDs should be "
                "used to distinguish them."
            )
        )

    total_rows = sum(
        manifest.row_count
        for manifest
        in manifests
    )

    return MultiDatasetIngestion(
        dataset_count=
            len(
                manifests
            ),
        total_rows=
            total_rows,
        datasets=
            manifests,
        warnings=
            batch_warnings,
    )


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

@router.post(
    "/analysis/correlation",
    response_model=
        CorrelationAnalysisRun,
)
def analyze_correlation_csv(
    dataset_file: UploadFile = File(
        ...,
        description=(
            "CSV dataset analysed locally "
            "by DataLens."
        ),
    ),

    x_column: str = Form(
        ...,
        min_length=1,
    ),

    y_column: str = Form(
        ...,
        min_length=1,
    ),

    analysis_goal: AnalysisGoal = Form(
        ...,
    ),

    analysis_mode: AnalysisMode = Form(
        ...,
    ),

    x_kind: VariableKind = Form(
        ...,
    ),

    y_kind: VariableKind = Form(
        ...,
    ),

    observations_independent: (
        bool
        | None
    ) = Form(
        default=None,
    ),

    alpha: float = Form(
        default=0.05,
        gt=0.0,
        lt=1.0,
    ),

    permutation_resamples: int = Form(
        default=9999,
        ge=1,
    ),

    random_seed: int = Form(
        default=42,
    ),
) -> CorrelationAnalysisRun:
    (
        dataset_name,
        _,
    ) = (
        validate_csv_upload(
            dataset_file
        )
    )

    dataframe = (
        load_csv_dataframe(
            dataset_file
        )
    )

    missing_columns = [
        column
        for column
        in (
            x_column,
            y_column,
        )
        if column
        not in dataframe.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "One or more requested "
                    "columns do not exist in "
                    "the uploaded dataset."
                ),
                "missing_columns":
                    missing_columns,
                "available_columns": [
                    str(
                        column
                    )
                    for column
                    in dataframe.columns
                ],
            },
        )

    try:
        return (
            run_correlation_analysis(
                dataframe=
                    dataframe,

                dataset=
                    dataset_name,

                x_column=
                    x_column,

                y_column=
                    y_column,

                analysis_goal=
                    analysis_goal,

                analysis_mode=
                    analysis_mode,

                x_kind=
                    x_kind,

                y_kind=
                    y_kind,

                observations_independent=
                    observations_independent,

                alpha=
                    alpha,

                permutation_resamples=
                    permutation_resamples,

                random_seed=
                    random_seed,
            )
        )

    except AnalysisPipelineError as error:
        raise HTTPException(
            status_code=422,
            detail=str(
                error
            ),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(
                error
            ),
        ) from error

    finally:
        dataset_file.file.close()