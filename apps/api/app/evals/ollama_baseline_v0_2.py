from __future__ import annotations

import json
import re

from pathlib import Path

from time import perf_counter

from typing import (
    Literal,
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
    "ollama_analytical_baseline_v0.2"
)

STRICT_CANDIDATE_CONTRACT_VERSION = (
    "analytical_candidate_strict_v0.2"
)


# ============================================================
# IMPORTANT EXPERIMENTAL RULE
# ============================================================
#
# v0.1 allowed the model to omit:
#
# - intent
# - current_grain
# - target_grain
# - relevant_columns
# - family
#
# because AnalyticalCandidate defines permissive defaults.
#
# This v0.2 contract changes ONLY that aspect.
#
# Prompt:      unchanged
# Model:       unchanged
# Temperature: unchanged
# Benchmark:   unchanged
# Scorer:      unchanged
#
# ============================================================


class StrictAnalyticalCandidate(
    BaseModel
):
    """
    Strict structured-output contract used only for AI Eval v0.2.

    Important:

    `entity` and `target_grain` are nullable because null is a valid
    analytical answer.

    They are nevertheless REQUIRED fields in the generated JSON.

    The other analytical decisions are also required.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    intent: str = Field(
        min_length=1,
    )

    entity: str | None

    current_grain: str = Field(
        min_length=1,
    )

    target_grain: str | None

    relevant_columns: list[
        str
    ]

    family: str = Field(
        min_length=1,
    )

    tool_calls: list[
        ToolCallCandidate
    ]

    assumptions: list[
        str
    ]


# ============================================================
# RESULT MODELS
# ============================================================

BaselineCaseStatus = Literal[
    "ready",
    "generation_error",
]


class StrictBaselineCaseResult(
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


class StrictOllamaBaselineReport(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    status: Literal[
        "complete",
    ] = "complete"

    baseline_rule_version: str = (
        OLLAMA_BASELINE_RULE_VERSION
    )

    candidate_contract_version: str = (
        STRICT_CANDIDATE_CONTRACT_VERSION
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
        StrictBaselineCaseResult
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
# SINGLE CASE
# ============================================================

def run_strict_baseline_case(
    *,
    case: AnalyticalEvalCase,
    model: str = DEFAULT_BASELINE_MODEL,
) -> StrictBaselineCaseResult:
    """
    Run exactly the same prompt as baseline v0.1.

    The only intended experimental difference is the strict
    output schema supplied to Ollama.
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
                StrictAnalyticalCandidate
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

        # ----------------------------------------------------
        # 1. Validate the model against the strict contract.
        # ----------------------------------------------------

        strict_candidate = (
            StrictAnalyticalCandidate
            .model_validate_json(
                raw_content,
            )
        )

        # ----------------------------------------------------
        # 2. Convert to the provider-neutral candidate already
        #    understood by our unchanged scorer.
        # ----------------------------------------------------

        candidate = (
            AnalyticalCandidate
            .model_validate(
                strict_candidate.model_dump()
            )
        )

        # ----------------------------------------------------
        # 3. Same scorer as v0.1.
        # ----------------------------------------------------

        score = score_candidate(
            case,
            candidate,
        )

        score_payload = (
            score.as_dict()
        )

        return StrictBaselineCaseResult(
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

        return StrictBaselineCaseResult(
            case_id=case.case_id,
            domain=case.domain,
            user_request=case.user_request,
            model=model,
            status="generation_error",
            inference_ms=inference_ms,
            candidate=None,
            score_metrics=empty_metrics(),
            score_diagnostics=empty_diagnostics(),
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
        StrictBaselineCaseResult
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
        StrictBaselineCaseResult
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

def run_strict_ollama_baseline(
    *,
    benchmark_path: str | Path,
    split: EvalSplit = "validation",
    model: str = DEFAULT_BASELINE_MODEL,
) -> StrictOllamaBaselineReport:
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
        run_strict_baseline_case(
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

    return StrictOllamaBaselineReport(
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
                results,
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


def save_strict_baseline_report(
    *,
    report: StrictOllamaBaselineReport,
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
        "_baseline_v0_2.json"
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