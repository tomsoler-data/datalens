from __future__ import annotations

import json
import re

from pathlib import Path
from time import perf_counter
from typing import (
    Annotated,
    Literal,
    Union,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.ai.provider import (
    client,
)

from app.evals.benchmark_loader import (
    load_benchmark,
)

from app.evals.ollama_baseline import (
    DEFAULT_BASELINE_MODEL,
    SYSTEM_PROMPT,
    build_user_prompt,
)

from app.evals.schemas import (
    AnalyticalCandidate,
    AnalyticalEvalCase,
    EvalSplit,
    ToolCallCandidate,
)

from app.evals.scorer import (
    score_candidate,
)


# ============================================================
# VERSION
# ============================================================

OLLAMA_BASELINE_RULE_VERSION = (
    "ollama_analytical_baseline_v0.3"
)

TYPED_CANDIDATE_CONTRACT_VERSION = (
    "analytical_candidate_typed_v0.3"
)


# ============================================================
# CLOSED ANALYTICAL VOCABULARY
# ============================================================

AnalyticalIntent = Literal[
    "aggregate_metric",
    "compare_groups",
    "measure_relationship",
    "time_series_analysis",
    "distribution_analysis",
    "entity_anomaly_analysis",
    "data_quality_analysis",
]


AnalyticalFamily = Literal[
    "aggregation",
    "group_comparison",
    "association",
    "time_series",
    "distribution",
    "entity_outlier",
    "data_quality",
]


ControlledAssumption = Literal[
    "causal_claim",
    "fraud",
    "delete_outliers",
    "failure",
    "safety_failure",
    "poor_performance",
    "bad_restaurant",
    "poor_employee",
    "b2b",
]


# ============================================================
# TYPED TOOL ARGUMENTS
# ============================================================

class AggregateArguments(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    metrics: list[str] = Field(
        min_length=1,
    )

    group_by: list[str] | None = None


class BuildEntityViewArguments(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    entity: str = Field(
        min_length=1,
    )


class DeriveMetricArguments(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    inputs: list[str] = Field(
        min_length=1,
    )

    output: str = Field(
        min_length=1,
    )

    formula: str = Field(
        min_length=1,
    )


class AnalyzeDistributionArguments(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    target: str = Field(
        min_length=1,
    )


class DetectOutliersArguments(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    target: str = Field(
        min_length=1,
    )


class DetectEntityOutliersArguments(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    entity: str = Field(
        min_length=1,
    )

    metrics: list[str] = Field(
        min_length=1,
    )


class CompareGroupsArguments(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    target: str = Field(
        min_length=1,
    )

    group_by: str = Field(
        min_length=1,
    )


class MeasureAssociationArguments(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    target: str = Field(
        min_length=1,
    )

    value: str = Field(
        min_length=1,
    )


class AnalyzeTimeSeriesArguments(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    date: str = Field(
        min_length=1,
    )

    target: str = Field(
        min_length=1,
    )


# ============================================================
# TYPED TOOL CALLS
# ============================================================

class AggregateCall(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    name: Literal[
        "aggregate"
    ]

    arguments: AggregateArguments


class BuildEntityViewCall(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    name: Literal[
        "build_entity_view"
    ]

    arguments: BuildEntityViewArguments


class DeriveMetricCall(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    name: Literal[
        "derive_metric"
    ]

    arguments: DeriveMetricArguments


class AnalyzeDistributionCall(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    name: Literal[
        "analyze_distribution"
    ]

    arguments: AnalyzeDistributionArguments


class DetectOutliersCall(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    name: Literal[
        "detect_outliers"
    ]

    arguments: DetectOutliersArguments


class DetectEntityOutliersCall(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    name: Literal[
        "detect_entity_outliers"
    ]

    arguments: DetectEntityOutliersArguments


class CompareGroupsCall(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    name: Literal[
        "compare_groups"
    ]

    arguments: CompareGroupsArguments


class MeasureAssociationCall(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    name: Literal[
        "measure_association"
    ]

    arguments: MeasureAssociationArguments


class AnalyzeTimeSeriesCall(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    name: Literal[
        "analyze_time_series"
    ]

    arguments: AnalyzeTimeSeriesArguments


TypedToolCall = Annotated[
    Union[
        AggregateCall,
        BuildEntityViewCall,
        DeriveMetricCall,
        AnalyzeDistributionCall,
        DetectOutliersCall,
        DetectEntityOutliersCall,
        CompareGroupsCall,
        MeasureAssociationCall,
        AnalyzeTimeSeriesCall,
    ],
    Field(
        discriminator="name",
    ),
]


# ============================================================
# STRICT + TYPED CANDIDATE
# ============================================================

class TypedAnalyticalCandidate(
    BaseModel
):
    """
    AI Eval v0.3 contract.

    Differences from v0.2:

    - intent uses a closed DataLens vocabulary;
    - family uses a closed DataLens vocabulary;
    - assumptions use controlled tags;
    - each tool has its own typed argument contract.

    No benchmark answer is embedded in this schema.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    intent: AnalyticalIntent

    entity: str | None

    current_grain: str = Field(
        min_length=1,
    )

    target_grain: str | None

    relevant_columns: list[str] = Field(
        min_length=1,
    )

    family: AnalyticalFamily

    tool_calls: list[
        TypedToolCall
    ] = Field(
        min_length=1,
    )

    assumptions: list[
        ControlledAssumption
    ]


# ============================================================
# RESULT MODELS
# ============================================================

BaselineCaseStatus = Literal[
    "ready",
    "generation_error",
]


class TypedBaselineCaseResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    case_id: str

    domain: str

    user_request: str

    model: str

    status: BaselineCaseStatus

    inference_ms: float = Field(
        ge=0.0,
    )

    candidate: (
        AnalyticalCandidate
        | None
    )

    score_metrics: dict[
        str,
        float,
    ]

    score_diagnostics: dict[
        str,
        list[str],
    ]

    overall: float = Field(
        ge=0.0,
        le=1.0,
    )

    raw_content: (
        str
        | None
    ) = None

    error: (
        str
        | None
    ) = None


class TypedOllamaBaselineReport(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    status: Literal[
        "complete"
    ] = "complete"

    baseline_rule_version: str = (
        OLLAMA_BASELINE_RULE_VERSION
    )

    candidate_contract_version: str = (
        TYPED_CANDIDATE_CONTRACT_VERSION
    )

    model: str

    split: EvalSplit

    case_count: int = Field(
        ge=0,
    )

    generation_success_count: int = Field(
        ge=0,
    )

    generation_error_count: int = Field(
        ge=0,
    )

    total_inference_ms: float = Field(
        ge=0.0,
    )

    average_inference_ms: float = Field(
        ge=0.0,
    )

    average_metrics: dict[
        str,
        float,
    ]

    average_overall: float = Field(
        ge=0.0,
        le=1.0,
    )

    invented_column_count: int = Field(
        ge=0,
    )

    invented_tool_count: int = Field(
        ge=0,
    )

    forbidden_tool_count: int = Field(
        ge=0,
    )

    forbidden_assumption_count: int = Field(
        ge=0,
    )

    results: list[
        TypedBaselineCaseResult
    ]


# ============================================================
# EMPTY RESULT HELPERS
# ============================================================

def empty_metrics() -> dict[
    str,
    float,
]:
    return {
        "intent": 0.0,
        "entity": 0.0,
        "grain": 0.0,
        "relevant_columns": 0.0,
        "family": 0.0,
        "tool_selection": 0.0,
        "tool_arguments": 0.0,
        "safety": 0.0,
    }


def empty_diagnostics() -> dict[
    str,
    list[str],
]:
    return {
        "invented_columns": [],
        "invented_tools": [],
        "forbidden_tools_used": [],
        "forbidden_assumptions_used": [],
    }


# ============================================================
# CONVERSION
# ============================================================

def to_generic_candidate(
    typed_candidate: TypedAnalyticalCandidate,
) -> AnalyticalCandidate:
    """
    Convert the strict typed v0.3 contract to the unchanged
    provider-neutral candidate understood by scorer v0.1.
    """

    generic_tool_calls: list[
        ToolCallCandidate
    ] = []


    for call in (
        typed_candidate.tool_calls
    ):
        generic_tool_calls.append(
            ToolCallCandidate(
                name=call.name,

                arguments=(
                    call.arguments.model_dump(
                        mode="python",
                        exclude_none=True,
                    )
                ),
            )
        )


    return AnalyticalCandidate(
        intent=typed_candidate.intent,

        entity=typed_candidate.entity,

        current_grain=(
            typed_candidate.current_grain
        ),

        target_grain=(
            typed_candidate.target_grain
        ),

        relevant_columns=(
            typed_candidate.relevant_columns
        ),

        family=typed_candidate.family,

        tool_calls=generic_tool_calls,

        assumptions=list(
            typed_candidate.assumptions
        ),
    )


# ============================================================
# SINGLE CASE
# ============================================================

def run_typed_baseline_case(
    *,
    case: AnalyticalEvalCase,
    model: str = DEFAULT_BASELINE_MODEL,
) -> TypedBaselineCaseResult:
    """
    Same prompt, same model and same scorer as previous runs.

    Only the structured output schema changes.
    """

    prompt = build_user_prompt(
        case,
    )

    started_at = perf_counter()

    raw_content: str | None = None


    try:
        response = client.chat(
            model=model,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            format=(
                TypedAnalyticalCandidate
                .model_json_schema()
            ),

            options={
                "temperature": 0,
            },
        )


        inference_ms = (
            (
                perf_counter()
                - started_at
            )
            * 1000.0
        )


        raw_content = (
            response
            .message
            .content
        )


        typed_candidate = (
            TypedAnalyticalCandidate
            .model_validate_json(
                raw_content,
            )
        )


        candidate = (
            to_generic_candidate(
                typed_candidate,
            )
        )


        score = score_candidate(
            case,
            candidate,
        )


        score_payload = (
            score.as_dict()
        )


        return TypedBaselineCaseResult(
            case_id=case.case_id,

            domain=case.domain,

            user_request=case.user_request,

            model=model,

            status="ready",

            inference_ms=inference_ms,

            candidate=candidate,

            score_metrics=(
                score_payload[
                    "metrics"
                ]
            ),

            score_diagnostics=(
                score_payload[
                    "diagnostics"
                ]
            ),

            overall=score.overall,

            raw_content=raw_content,

            error=None,
        )


    except Exception as error:
        inference_ms = (
            (
                perf_counter()
                - started_at
            )
            * 1000.0
        )


        return TypedBaselineCaseResult(
            case_id=case.case_id,

            domain=case.domain,

            user_request=case.user_request,

            model=model,

            status=(
                "generation_error"
            ),

            inference_ms=inference_ms,

            candidate=None,

            score_metrics=(
                empty_metrics()
            ),

            score_diagnostics=(
                empty_diagnostics()
            ),

            overall=0.0,

            raw_content=raw_content,

            error=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
        )


# ============================================================
# REPORT HELPERS
# ============================================================

def average_metrics(
    results: list[
        TypedBaselineCaseResult
    ],
) -> dict[
    str,
    float,
]:
    if not results:
        return empty_metrics()


    metric_names = tuple(
        empty_metrics().keys()
    )


    return {
        metric_name: (
            sum(
                result
                .score_metrics
                .get(
                    metric_name,
                    0.0,
                )

                for result
                in results
            )
            / len(
                results
            )
        )

        for metric_name
        in metric_names
    }


def diagnostic_count(
    *,
    results: list[
        TypedBaselineCaseResult
    ],
    key: str,
) -> int:
    return sum(
        len(
            result
            .score_diagnostics
            .get(
                key,
                [],
            )
        )

        for result
        in results
    )


# ============================================================
# COMPLETE BASELINE
# ============================================================

def run_typed_ollama_baseline(
    *,
    benchmark_path: str | Path,
    split: EvalSplit = "validation",
    model: str = DEFAULT_BASELINE_MODEL,
) -> TypedOllamaBaselineReport:
    cases = load_benchmark(
        benchmark_path,
        split=split,
    )


    if not cases:
        raise ValueError(
            "Aucun cas d'évaluation "
            f"pour le split `{split}`."
        )


    results = [
        run_typed_baseline_case(
            case=case,
            model=model,
        )

        for case
        in cases
    ]


    successful = [
        result
        for result
        in results
        if result.status == "ready"
    ]


    failed = [
        result
        for result
        in results
        if (
            result.status
            == "generation_error"
        )
    ]


    total_inference_ms = sum(
        result.inference_ms
        for result
        in results
    )


    average_inference_ms = (
        total_inference_ms
        / len(
            results
        )
    )


    average_overall = (
        sum(
            result.overall
            for result
            in results
        )
        / len(
            results
        )
    )


    return TypedOllamaBaselineReport(
        model=model,

        split=split,

        case_count=len(
            results
        ),

        generation_success_count=len(
            successful
        ),

        generation_error_count=len(
            failed
        ),

        total_inference_ms=(
            total_inference_ms
        ),

        average_inference_ms=(
            average_inference_ms
        ),

        average_metrics=(
            average_metrics(
                results
            )
        ),

        average_overall=(
            average_overall
        ),

        invented_column_count=(
            diagnostic_count(
                results=results,
                key="invented_columns",
            )
        ),

        invented_tool_count=(
            diagnostic_count(
                results=results,
                key="invented_tools",
            )
        ),

        forbidden_tool_count=(
            diagnostic_count(
                results=results,
                key="forbidden_tools_used",
            )
        ),

        forbidden_assumption_count=(
            diagnostic_count(
                results=results,
                key=(
                    "forbidden_assumptions_used"
                ),
            )
        ),

        results=results,
    )


# ============================================================
# SAVE
# ============================================================

def safe_model_filename(
    model: str,
) -> str:
    normalized = (
        model
        .strip()
        .lower()
    )


    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        normalized,
    )


    return (
        normalized.strip(
            "_"
        )
        or "model"
    )


def save_typed_baseline_report(
    *,
    report: TypedOllamaBaselineReport,
    output_dir: str | Path,
) -> Path:
    output_directory = Path(
        output_dir,
    )


    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


    filename = (
        f"{safe_model_filename(report.model)}"
        f"_{report.split}"
        "_baseline_v0_3.json"
    )


    output_path = (
        output_directory
        / filename
    )


    output_path.write_text(
        json.dumps(
            report.model_dump(
                mode="json",
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    return output_path