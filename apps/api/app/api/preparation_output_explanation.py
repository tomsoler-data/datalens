from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.preparation.analysis_output_explanation import (
    ANALYSIS_OUTPUT_EXPLANATION_RULE_VERSION,
    AnalysisOutputExplanation,
    AnalysisOutputRecommendationFacts,
    build_analysis_output_recommendation_facts,
    explain_analysis_output_with_ai,
)

from app.preparation.preparation_artifact_store import (
    PreparationArtifactStoreError,
    PreparationArtifactWorkflowNotFoundError,
)

from app.preparation.preparation_session import (
    PreparationSessionNotFoundError,
)


PREPARATION_OUTPUT_EXPLANATION_API_VERSION = (
    "preparation_output_explanation_api_v0.1"
)


router = APIRouter(
    prefix="/preparation/analysis-output",
    tags=[
        "preparation",
    ],
)


class StrictPreparationOutputExplanationRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


class PreparationOutputExplanationRequest(
    StrictPreparationOutputExplanationRequest,
):
    workflow_id: str = Field(
        min_length=1
    )

    dataset_id: str = Field(
        min_length=1
    )

    include_ai: bool = True


class PreparationOutputExplanationResponse(BaseModel):
    workflow_id: str
    dataset_id: str
    dataset_filename: str
    facts: AnalysisOutputRecommendationFacts
    explanation: AnalysisOutputExplanation | None = None
    ai_error: str | None = None
    recommended: bool
    api_version: str = PREPARATION_OUTPUT_EXPLANATION_API_VERSION
    rule_version: str = ANALYSIS_OUTPUT_EXPLANATION_RULE_VERSION


def _error_detail(
    *,
    code: str,
    error: Exception,
    workflow_id: str,
    dataset_id: str,
) -> dict[str, Any]:
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
            PREPARATION_OUTPUT_EXPLANATION_API_VERSION,
    }


@router.post(
    "/explain",
    response_model=
        PreparationOutputExplanationResponse,
)
def explain_preparation_analysis_output(
    request: PreparationOutputExplanationRequest,
) -> PreparationOutputExplanationResponse:
    """
    Explain why a materialized Preparation artifact is or is not
    recommended as an analytical output.

    Python owns:
    - artifact existence;
    - lineage;
    - terminal/intermediate classification;
    - deterministic recommendation.

    The local model only turns those facts into analyst-facing text.
    """

    try:
        facts = build_analysis_output_recommendation_facts(
            workflow_id=request.workflow_id,
            dataset_id=request.dataset_id,
        )

        explanation = None
        ai_error = None

        if request.include_ai:
            try:
                explanation = explain_analysis_output_with_ai(
                    facts=facts
                )

            except Exception:
                ai_error = (
                    "Local model analysis-output explanation "
                    "is unavailable."
                )

        return PreparationOutputExplanationResponse(
            workflow_id=request.workflow_id,
            dataset_id=facts.dataset_id,
            dataset_filename=facts.dataset_filename,
            facts=facts,
            explanation=explanation,
            ai_error=ai_error,
            recommended=(
                facts.recommendation_status
                ==
                "recommended_terminal"
            ),
        )

    except PreparationSessionNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=_error_detail(
                code="preparation_session_not_found",
                error=error,
                workflow_id=request.workflow_id,
                dataset_id=request.dataset_id,
            ),
        ) from error

    except (
        PreparationArtifactWorkflowNotFoundError,
        PreparationArtifactStoreError,
    ) as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=_error_detail(
                code="preparation_output_artifact_not_found",
                error=error,
                workflow_id=request.workflow_id,
                dataset_id=request.dataset_id,
            ),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=_error_detail(
                code="preparation_output_candidate_not_found",
                error=error,
                workflow_id=request.workflow_id,
                dataset_id=request.dataset_id,
            ),
        ) from error
