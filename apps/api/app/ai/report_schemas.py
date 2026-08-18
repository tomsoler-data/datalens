from __future__ import annotations

from typing import (
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
)

from app.reporting.unified_schemas import (
    ReportFinding,
    UnifiedAnalysisReport,
)


SemanticLanguage = Literal[
    "fr",
    "en",
]


SemanticPriority = Literal[
    "high",
    "medium",
    "low",
]


SemanticGenerationMode = Literal[
    "llm",
    "hybrid",
    "deterministic_fallback",
]


SemanticReason = Literal[
    "meaningful_trend",
    "meaningful_gap",
    "cross_dataset_relationship",
    "distinct_concepts",
    "group_difference",
    "complementary_perspective",
    "high_analytical_signal",
    "limited_semantic_value",
    "conceptual_redundancy",
]


AssessmentSource = Literal[
    "llm",
    "fallback",
]


# ============================================================
# SINGLE-CANDIDATE LLM DRAFT
# ============================================================

class SemanticCandidateAssessmentDraft(
    BaseModel
):
    semantic_relevance: int = Field(
        ge=0,
        le=100,
    )

    semantic_priority: SemanticPriority

    reasons: list[
        SemanticReason
    ] = Field(
        min_length=1,
        max_length=3,
    )


# ============================================================
# VALIDATED CANDIDATE ASSESSMENT
# ============================================================

class SemanticCandidateAssessment(
    BaseModel
):
    candidate_key: str

    analysis_id: str

    semantic_relevance: int | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    semantic_priority: SemanticPriority

    semantic_reasons: list[
        SemanticReason
    ] = Field(
        default_factory=list,
    )

    assessment_source: AssessmentSource


# ============================================================
# FINAL SEMANTIC FINDING
# ============================================================

class SemanticReportFinding(
    BaseModel
):
    analysis_id: str

    semantic_relevance: int | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    semantic_priority: SemanticPriority

    semantic_reasons: list[
        SemanticReason
    ] = Field(
        default_factory=list,
    )

    interpretation: str

    why_it_matters: str

    method_explanation: str

    attention_points: list[
        str
    ] = Field(
        default_factory=list,
    )

    source_finding: ReportFinding


# ============================================================
# FINAL REPORT
# ============================================================

class SemanticAnalysisReport(
    BaseModel
):
    status: Literal[
        "ready"
    ] = "ready"

    language: SemanticLanguage

    objective: str | None = None

    model: str

    generation_mode: SemanticGenerationMode

    executive_summary: str

    main_findings: list[
        SemanticReportFinding
    ] = Field(
        default_factory=list,
    )

    candidate_assessments: list[
        SemanticCandidateAssessment
    ] = Field(
        default_factory=list,
    )

    candidate_analysis_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    not_selected_analysis_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    source_report: UnifiedAnalysisReport

    semantic_rule_version: str = (
        "semantic_report_v0.3"
    )
