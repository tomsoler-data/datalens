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

from app.api.preparation_cleaning import (
    router as preparation_cleaning_router,
)

from app.api.preparation_combination import (
    router as preparation_combination_router,
)

from app.api.preparation_combine import (
    router as preparation_combine_router,
)

from app.api.preparation_identity import (
    router as preparation_identity_router,
)

from app.api.preparation_output_explanation import (
    router as preparation_output_explanation_router,
)

from app.api.preparation_quality import (
    router as preparation_quality_router,
)

from app.api.preparation_semantic import (
    router as preparation_semantic_router,
)

from app.api.preparation_session import (
    router as preparation_session_router,
)

from app.api.preparation_transformation import (
    router as preparation_transformation_router,
)

from app.api.preparation_validation import (
    router as preparation_validation_router,
)

from app.api.preparation_workflow import (
    router as preparation_workflow_router,
)

from app.api.routes import (
    router as api_router,
)


# ============================================================
# APPLICATION
# ============================================================


app = FastAPI(
    title="DataLens API",

    description=(
        "Local-first backend API for deterministic data "
        "analysis, controlled preparation, guarded semantic "
        "review, document retrieval and grounded AI "
        "explanations."
    ),

    version="0.8.0",
)


# ============================================================
# CORS
# ============================================================


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


# ============================================================
# CORE API
# ============================================================


app.include_router(
    api_router
)


# ============================================================
# ANALYSIS
# ============================================================


app.include_router(
    analysis_run_router
)


# ============================================================
# DOCUMENTS / RAG
# ============================================================


app.include_router(
    document_ingestion_router
)


# ============================================================
# PREPARATION - QUALITY
# ============================================================


app.include_router(
    preparation_quality_router
)


# ============================================================
# PREPARATION - CLEANING
# ============================================================


app.include_router(
    preparation_cleaning_router
)


# ============================================================
# PREPARATION - SEMANTIC REVIEW
# ============================================================


app.include_router(
    preparation_semantic_router
)


# ============================================================
# PREPARATION - IDENTITY
# ============================================================


app.include_router(
    preparation_identity_router
)


# ============================================================
# PREPARATION - TRANSFORMATION
# ============================================================


app.include_router(
    preparation_transformation_router
)


# ============================================================
# PREPARATION - LEGACY / LOW-LEVEL COMBINATION
# ============================================================


app.include_router(
    preparation_combination_router
)


# ============================================================
# PREPARATION - CONTROLLED COMBINE WORKFLOW
# ============================================================


app.include_router(
    preparation_combine_router
)


# ============================================================
# PREPARATION - ANALYSIS OUTPUT EXPLANATION
# ============================================================


app.include_router(
    preparation_output_explanation_router
)


# ============================================================
# PREPARATION - WORKFLOW
# ============================================================


app.include_router(
    preparation_workflow_router
)


# ============================================================
# PREPARATION - SERVER-OWNED SESSION
# ============================================================


app.include_router(
    preparation_session_router
)


# ============================================================
# PREPARATION - FINAL VALIDATION
# ============================================================


app.include_router(
    preparation_validation_router
)
