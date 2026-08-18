from fastapi import FastAPI

from fastapi.middleware.cors import (
    CORSMiddleware,
)


from app.api.analysis_run import (
    router as analysis_run_router,
)

from app.api.document_ingestion import (
    router as document_ingestion_router,
)

from app.api.preparation_quality import (
    router as preparation_quality_router,
)

from app.api.routes import (
    router as api_router,
)


app = FastAPI(
    title="DataLens API",
    description=(
        "Local-first backend API for "
        "deterministic data analysis, "
        "statistical decisions, visualization "
        "selection, dashboard composition, "
        "data-quality preparation, "
        "document retrieval and grounded "
        "AI explanations."
    ),
    version="0.6.0",
)


LOCAL_FRONTEND_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=
        LOCAL_FRONTEND_ORIGINS,
    allow_credentials=
        False,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
    allow_headers=[
        "*",
    ],
)


app.include_router(
    api_router
)


app.include_router(
    analysis_run_router
)


app.include_router(
    document_ingestion_router
)


app.include_router(
    preparation_quality_router
)
