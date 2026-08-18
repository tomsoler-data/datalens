from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)


from app.rag import (
    DocumentIngestionReport,
    MAX_DOCUMENT_FILES,
    build_document_ingestion_report,
)

from app.rag_retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    RagSearchResponse,
    search_document_chunks,
)


router = APIRouter()


# ============================================================
# FILE READING
# ============================================================

def read_uploaded_documents(
    document_files: list[
        UploadFile
    ],
) -> list[
    tuple[
        str,
        bytes,
    ]
]:
    if not document_files:
        raise HTTPException(
            status_code=400,
            detail=(
                "Au moins un document "
                "doit être fourni."
            ),
        )


    if (
        len(
            document_files
        )
        >
        MAX_DOCUMENT_FILES
    ):
        raise HTTPException(
            status_code=413,
            detail=(
                "Trop de documents ont "
                "été fournis."
            ),
        )


    documents: list[
        tuple[
            str,
            bytes,
        ]
    ] = []


    try:
        for document_file in (
            document_files
        ):
            filename = (
                document_file.filename
                or ""
            ).strip()


            if not filename:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Un document ne "
                        "possède pas de nom "
                        "de fichier."
                    ),
                )


            document_file.file.seek(
                0
            )


            content = (
                document_file
                .file
                .read()
            )


            documents.append(
                (
                    filename,
                    content,
                )
            )


    finally:
        for document_file in (
            document_files
        ):
            document_file.file.close()


    return documents


# ============================================================
# DOCUMENT INGESTION
# ============================================================

def ingest_document_uploads(
    document_files: list[
        UploadFile
    ],
) -> DocumentIngestionReport:
    documents = (
        read_uploaded_documents(
            document_files
        )
    )


    try:
        return (
            build_document_ingestion_report(
                documents=
                    documents,
            )
        )


    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(
                error
            ),
        ) from error


# ============================================================
# DOCUMENT INSPECTION
# ============================================================

@router.post(
    "/rag/documents/inspect",
    response_model=
        DocumentIngestionReport,
)
def inspect_documents(
    document_files: list[
        UploadFile
    ] = File(
        ...,
    ),
) -> DocumentIngestionReport:
    return ingest_document_uploads(
        document_files
    )


# ============================================================
# DOCUMENT SEMANTIC SEARCH
# ============================================================

@router.post(
    "/rag/search",
    response_model=
        RagSearchResponse,
)
def search_documents(
    document_files: list[
        UploadFile
    ] = File(
        ...,
    ),

    query: str = Form(
        ...,
        min_length=1,
    ),

    top_k: int = Form(
        default=
            DEFAULT_TOP_K,

        ge=1,
        le=20,
    ),

    model: str = Form(
        default=
            DEFAULT_EMBEDDING_MODEL,

        min_length=1,
    ),
) -> RagSearchResponse:
    ingestion = (
        ingest_document_uploads(
            document_files
        )
    )


    try:
        return (
            search_document_chunks(
                ingestion=
                    ingestion,

                query=
                    query,

                top_k=
                    top_k,

                model=
                    model,
            )
        )


    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(
                error
            ),
        ) from error


    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(
                error
            ),
        ) from error