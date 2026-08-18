from __future__ import annotations


import csv

from io import BytesIO

from pathlib import Path

from typing import Any


import pandas as pd


from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from fastapi.responses import (
    StreamingResponse,
)


from app.analysis import (
    AnalysisPipelineError,
    CorrelationAnalysisRun,
    run_correlation_analysis,
)

from app.cleaning.engine import (
    clean_dataset,
)

from app.discovery import (
    AnalysisDiscoveryReport,
    discover_analyses,
)

from app.ingestion import (
    MultiDatasetIngestion,
    build_dataset_manifest,
)

from app.planning import (
    AnalysisPlanReport,
    build_analysis_plan,
)

from app.reporting import (
    AnalysisReport,
    compose_analysis_report,
    render_analysis_report_pdf,
)

from app.statistics.schemas import (
    AnalysisGoal,
    AnalysisMode,
    VariableKind,
)


router = APIRouter()


# ============================================================
# DATASET CONFIGURATION
# ============================================================

SUPPORTED_DATA_EXTENSIONS = {
    ".csv",
}


MAX_DATASET_FILES = 20


# ============================================================
# CSV FORMAT DETECTION
# ============================================================

CSV_SAMPLE_BYTES = 64 * 1024


SUPPORTED_CSV_DELIMITERS = (
    ",",
    ";",
    "\t",
    "|",
)


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
# CSV SAMPLE READING
# ============================================================

def read_csv_sample(
    dataset_file: UploadFile,
) -> str:
    """
    Read only a small prefix of the uploaded CSV.

    DataLens should not load an entire large dataset into
    memory merely to determine its delimiter.

    The file pointer is restored to the beginning before
    returning.
    """

    try:
        dataset_file.file.seek(
            0
        )


        raw_sample = (
            dataset_file.file.read(
                CSV_SAMPLE_BYTES
            )
        )


        dataset_file.file.seek(
            0
        )


    except OSError as error:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded CSV could not "
                "be read safely."
            ),
        ) from error


    if isinstance(
        raw_sample,
        bytes,
    ):
        try:
            sample = (
                raw_sample.decode(
                    "utf-8-sig"
                )
            )


        except UnicodeDecodeError as error:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The uploaded CSV is not "
                    "valid UTF-8 text."
                ),
            ) from error


    else:
        sample = str(
            raw_sample
        )


    if not sample.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded CSV contains "
                "no readable data."
            ),
        )


    return sample


# ============================================================
# CSV DELIMITER DETECTION
# ============================================================

def detect_csv_delimiter(
    dataset_file: UploadFile,
) -> str:
    """
    Detect the delimiter used by an uploaded CSV.

    Supported delimiters:
    - comma
    - semicolon
    - tab
    - pipe

    csv.Sniffer is attempted first.

    If Sniffer cannot determine the delimiter, DataLens uses
    a deterministic fallback based on the first non-empty
    line.

    A comma is returned as the final fallback so genuine
    single-column CSV files remain valid.
    """

    sample = (
        read_csv_sample(
            dataset_file
        )
    )


    delimiters = "".join(
        SUPPORTED_CSV_DELIMITERS
    )


    try:
        dialect = (
            csv.Sniffer()
            .sniff(
                sample,
                delimiters=
                    delimiters,
            )
        )


        delimiter = (
            dialect.delimiter
        )


        if (
            delimiter
            in SUPPORTED_CSV_DELIMITERS
        ):
            return delimiter


    except csv.Error:
        pass


    first_non_empty_line = ""


    for line in (
        sample.splitlines()
    ):
        if line.strip():
            first_non_empty_line = (
                line
            )

            break


    if first_non_empty_line:
        delimiter_counts = {
            delimiter:
                first_non_empty_line.count(
                    delimiter
                )

            for delimiter
            in SUPPORTED_CSV_DELIMITERS
        }


        best_delimiter = max(
            delimiter_counts,
            key=lambda delimiter: (
                delimiter_counts[
                    delimiter
                ]
            ),
        )


        if (
            delimiter_counts[
                best_delimiter
            ]
            >
            0
        ):
            return best_delimiter


    return ","


# ============================================================
# CSV LOADING
# ============================================================

def load_csv_dataframe(
    dataset_file: UploadFile,
) -> pd.DataFrame:
    """
    Load an uploaded CSV into a pandas DataFrame after
    automatically detecting its delimiter.
    """

    delimiter = (
        detect_csv_delimiter(
            dataset_file
        )
    )


    try:
        dataset_file.file.seek(
            0
        )


        dataframe = (
            pd.read_csv(
                dataset_file.file,
                encoding=
                    "utf-8-sig",

                sep=
                    delimiter,

                low_memory=
                    False,
            )
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


    finally:
        try:
            dataset_file.file.seek(
                0
            )

        except OSError:
            pass


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
        ==
        0
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
# DATASET BUNDLE
# ============================================================

def load_uploaded_dataset_bundle(
    dataset_files: list[
        UploadFile
    ],
) -> tuple[
    MultiDatasetIngestion,
    list[
        dict[
            str,
            Any,
        ]
    ],
]:
    """
    Load, clean and profile multiple uploaded CSV datasets.

    The original uploaded file is never modified.

    The dataframe exposed to downstream analysis is the
    conservative cleaned dataframe produced by DataLens.

    Each dataset record also retains:
    - the cleaning report;
    - the transformation history;
    - the raw row count.

    This provides auditability without keeping an unnecessary
    second full dataframe in memory.
    """

    if not dataset_files:
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


    dataset_records: list[
        dict[
            str,
            Any,
        ]
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


            # =================================================
            # 1. RAW INGESTION
            # =================================================

            raw_dataframe = (
                load_csv_dataframe(
                    dataset_file
                )
            )


            raw_row_count = int(
                raw_dataframe.shape[0]
            )


            # =================================================
            # 2. CONSERVATIVE CLEANING
            # =================================================

            (
                dataframe,
                cleaning_report,
                transformations,
            ) = (
                clean_dataset(
                    raw_dataframe,
                    filename,
                )
            )


            # The source file is preserved.
            # We no longer need to retain a second large
            # dataframe in memory after the cleaning report
            # has captured the before/after state.
            del raw_dataframe


            if dataframe.empty:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Dataset '{filename}' "
                        "contains no analyzable rows "
                        "after conservative cleaning."
                    ),
                )


            dataset_id = (
                f"dataset:{index:04d}"
            )


            # =================================================
            # 3. MANIFEST FROM CLEAN DATA
            # =================================================

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


            # =================================================
            # 4. DOWNSTREAM DATASET RECORD
            # =================================================

            dataset_records.append(
                {
                    "dataset_id":
                        dataset_id,

                    "filename":
                        filename,

                    "extension":
                        extension,

                    # Discovery, relationships and execution
                    # consume the cleaned dataframe.
                    "dataframe":
                        dataframe,

                    # Audit information.
                    "raw_row_count":
                        raw_row_count,

                    "cleaning_report":
                        cleaning_report,

                    "transformations":
                        transformations,
                }
            )


            filenames.append(
                filename
            )


    finally:
        for dataset_file in (
            dataset_files
        ):
            dataset_file.file.close()


    # ========================================================
    # BATCH WARNINGS
    # ========================================================

    batch_warnings: list[
        str
    ] = []


    duplicate_filenames = sorted(
        {
            filename

            for filename
            in filenames

            if (
                filenames.count(
                    filename
                )
                >
                1
            )
        }
    )


    if duplicate_filenames:
        batch_warnings.append(
            (
                "Duplicate filenames were "
                "uploaded in the same batch: "
                +
                ", ".join(
                    duplicate_filenames
                )
                +
                ". Dataset IDs should be used "
                "to distinguish them."
            )
        )


    total_rows = sum(
        manifest.row_count

        for manifest
        in manifests
    )


    ingestion = (
        MultiDatasetIngestion(
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
    )


    return (
        ingestion,
        dataset_records,
    )


# ============================================================
# INGESTION HELPER
# ============================================================

def ingest_uploaded_datasets(
    dataset_files: list[
        UploadFile
    ],
) -> MultiDatasetIngestion:
    (
        ingestion,
        _,
    ) = (
        load_uploaded_dataset_bundle(
            dataset_files
        )
    )


    return ingestion


# ============================================================
# TEMPORARY REPORT BUILDER
# ============================================================

def build_report_from_uploads(
    *,
    dataset_files: list[
        UploadFile
    ],

    objective: (
        str
        | None
    ),
) -> AnalysisReport:
    ingestion = (
        ingest_uploaded_datasets(
            dataset_files
        )
    )


    plan = (
        build_analysis_plan(
            ingestion,

            objective=
                objective,
        )
    )


    return (
        compose_analysis_report(
            ingestion=
                ingestion,

            plan=
                plan,
        )
    )


# ============================================================
# INGESTION ENDPOINT
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
    ),
) -> MultiDatasetIngestion:
    return (
        ingest_uploaded_datasets(
            dataset_files
        )
    )


# ============================================================
# LEGACY ANALYSIS PLANNER
# ============================================================

@router.post(
    "/analysis/plan",
    response_model=
        AnalysisPlanReport,
)
def plan_analysis(
    dataset_files: list[
        UploadFile
    ] = File(
        ...,
    ),

    objective: str | None = Form(
        default=None,
    ),
) -> AnalysisPlanReport:
    ingestion = (
        ingest_uploaded_datasets(
            dataset_files
        )
    )


    return (
        build_analysis_plan(
            ingestion,

            objective=
                objective,
        )
    )


# ============================================================
# ANALYSIS DISCOVERY
# ============================================================

@router.post(
    "/analysis/discovery",
    response_model=
        AnalysisDiscoveryReport,
)
def discover_dataset_analyses(
    dataset_files: list[
        UploadFile
    ] = File(
        ...,
    ),

    objective: str | None = Form(
        default=None,
    ),
) -> AnalysisDiscoveryReport:
    (
        _,
        dataset_records,
    ) = (
        load_uploaded_dataset_bundle(
            dataset_files
        )
    )


    return (
        discover_analyses(
            datasets=
                dataset_records,

            objective=
                objective,
        )
    )


# ============================================================
# STRUCTURED REPORT
# ============================================================

@router.post(
    "/reports/analysis",
    response_model=
        AnalysisReport,
)
def create_analysis_report(
    dataset_files: list[
        UploadFile
    ] = File(
        ...,
    ),

    objective: str | None = Form(
        default=None,
    ),
) -> AnalysisReport:
    return (
        build_report_from_uploads(
            dataset_files=
                dataset_files,

            objective=
                objective,
        )
    )


# ============================================================
# PDF REPORT
# ============================================================

@router.post(
    "/reports/analysis.pdf",
)
def create_analysis_report_pdf(
    dataset_files: list[
        UploadFile
    ] = File(
        ...,
    ),

    objective: str | None = Form(
        default=None,
    ),
):
    report = (
        build_report_from_uploads(
            dataset_files=
                dataset_files,

            objective=
                objective,
        )
    )


    pdf_bytes = (
        render_analysis_report_pdf(
            report
        )
    )


    return StreamingResponse(
        BytesIO(
            pdf_bytes
        ),

        media_type=
            "application/pdf",

        headers={
            "Content-Disposition": (
                "attachment; "
                'filename="datalens_analysis_report.pdf"'
            )
        },
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


    raw_dataframe = (
        load_csv_dataframe(
            dataset_file
        )
    )


    (
        dataframe,
        _,
        _,
    ) = (
        clean_dataset(
            raw_dataframe,
            dataset_name,
        )
    )


    del raw_dataframe


    missing_columns = [
        column

        for column
        in (
            x_column,
            y_column,
        )

        if (
            column
            not in dataframe.columns
        )
    ]


    if missing_columns:
        dataset_file.file.close()


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