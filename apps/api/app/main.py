from contextlib import asynccontextmanager

from fastapi import FastAPI

from fastapi.middleware.cors import (
    CORSMiddleware,
)


from app.api.analysis_run import (
    router as analysis_run_router,
)

from app.api.requested_resolution import (
    router as requested_resolution_router,
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

from app.api.report_selection import (
    router as report_selection_router,
)

from app.api.model_training import (
    router as model_training_router,
)

from app.api.model_lab import (
    router as model_lab_router,
)

from app.api.ml_monitoring import (
    router as ml_monitoring_router,
)


from app.api.ml_performance_monitoring import (
    router as ml_performance_monitoring_router,
)


from app.api.ml_model_health import (
    router as ml_model_health_router,
)

from app.api.routes import (
    router as api_router,
)

from app.observability.runtime_trace import (
    RuntimeTraceMiddleware,
)


# ============================================================
# WORKFLOW DELETE CRASH RECOVERY
# PREPARATION_WORKFLOW_DELETE_API_V0_1
# ============================================================


from app.preparation.preparation_workflow_delete import (
    recover_pending_workflow_deletions,
)


@asynccontextmanager
async def datalens_lifespan(
    _app: FastAPI,
):
    """
    Reconcile any interrupted permanent workflow deletion
    before DataLens begins serving requests.

    Recovery failure intentionally prevents startup because
    serving against ambiguous SQLite/filesystem state would
    violate the permanent-delete contract.
    """

    recover_pending_workflow_deletions()

    yield


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

    lifespan=
        datalens_lifespan,
)


# ============================================================
# CORS
# ============================================================


LOCAL_FRONTEND_CORS_RULE_VERSION = (
    "local_frontend_cors_v0.1"
)


LOCAL_FRONTEND_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


LOCAL_FRONTEND_METHODS = [
    "GET",
    "POST",
    "DELETE",
]


LOCAL_FRONTEND_HEADERS = [
    "Content-Type",
]


LOCAL_FRONTEND_EXPOSE_HEADERS = [
    "X-DataLens-Request-ID",
]


LOCAL_FRONTEND_ALLOW_CREDENTIALS = False


app.add_middleware(
    CORSMiddleware,

    allow_origins=
        LOCAL_FRONTEND_ORIGINS,

    allow_credentials=
        LOCAL_FRONTEND_ALLOW_CREDENTIALS,

    allow_methods=
        LOCAL_FRONTEND_METHODS,

    allow_headers=
        LOCAL_FRONTEND_HEADERS,

    expose_headers=
        LOCAL_FRONTEND_EXPOSE_HEADERS,
)


# ============================================================
# RUNTIME OBSERVABILITY
# ============================================================


app.add_middleware(
    RuntimeTraceMiddleware
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


app.include_router(
    requested_resolution_router
)


# ============================================================
# REPORT SELECTION
# ============================================================


app.include_router(
    report_selection_router
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

# ============================================================
# MODEL TRAINING
# ============================================================


app.include_router(
    model_training_router
)


# ============================================================
# MODEL LAB
# ============================================================


app.include_router(
    model_lab_router
)


# ============================================================
# ML MONITORING
# ============================================================


app.include_router(
    ml_monitoring_router
)



# ============================================================
# ML PERFORMANCE MONITORING
# ============================================================


app.include_router(
    ml_performance_monitoring_router
)



# ============================================================
# ML MODEL HEALTH
# ============================================================


app.include_router(
    ml_model_health_router
)
