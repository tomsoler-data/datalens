from __future__ import annotations

import json

from pathlib import Path
from time import perf_counter
from typing import Any

from app.ai.provider import (
    client,
)

from app.evals.benchmark_loader import (
    load_benchmark,
)

from app.evals.ollama_baseline import (
    SYSTEM_PROMPT,
    build_user_prompt,
)

from app.evals.ollama_baseline_v0_3 import (
    TYPED_CANDIDATE_CONTRACT_VERSION,
    TypedAnalyticalCandidate,
    to_generic_candidate,
)

from app.evals.scorer_v0_2 import (
    SCORER_RULE_VERSION,
    score_candidate_v0_2,
)


# ============================================================
# VERSION
# ============================================================

MULTIMODEL_EVAL_VERSION = (
    "multimodel_hard_eval_v0.5"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__,
).resolve().parents[2]


BENCHMARK_PATH = (
    BASE_DIR
    / "evals"
    / "analytical_reasoning_hard_v0_4.jsonl"
)


RESULTS_DIR = (
    BASE_DIR
    / "evals"
    / "results"
)


OUTPUT_PATH = (
    RESULTS_DIR
    / "multimodel_hard_eval_v0_5.json"
)


# ============================================================
# MODELS
#
# qwen3.5:4b is deliberately excluded:
# it failed the DataLens typed structured-output contract.
# ============================================================

MODELS = [
    "gemma3:4b",
    "qwen3:4b-instruct",
    "phi4-mini",
    "ministral-3:3b",
]


# ============================================================
# HELPERS
# ============================================================

def average(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return (
        sum(values)
        / len(values)
    )


def rounded(
    value: float,
) -> float:
    return round(
        value,
        3,
    )


# ============================================================
# SINGLE CASE
# ============================================================

def run_case(
    *,
    model: str,
    case,
) -> dict[str, Any]:
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
                    "role":
                        "system",

                    "content":
                        SYSTEM_PROMPT,
                },

                {
                    "role":
                        "user",

                    "content":
                        prompt,
                },
            ],

            format=(
                TypedAnalyticalCandidate
                .model_json_schema()
            ),

            options={
                "temperature":
                    0,
            },

            think=False,
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
            or ""
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

        score = (
            score_candidate_v0_2(
                case,
                candidate,
            )
        )

        score_payload = (
            score.as_dict()
        )

        return {
            "case_id":
                case.case_id,

            "domain":
                case.domain,

            "user_request":
                case.user_request,

            "status":
                "ready",

            "inference_ms":
                inference_ms,

            "candidate":
                candidate.model_dump(
                    mode="json",
                ),

            "metrics":
                score_payload[
                    "metrics"
                ],

            "capabilities":
                score_payload[
                    "capabilities"
                ],

            "diagnostics":
                score_payload[
                    "diagnostics"
                ],

            "overall":
                score.overall,

            "raw_content":
                raw_content,

            "error":
                None,
        }

    except Exception as error:
        inference_ms = (
            (
                perf_counter()
                - started_at
            )
            * 1000.0
        )

        return {
            "case_id":
                case.case_id,

            "domain":
                case.domain,

            "user_request":
                case.user_request,

            "status":
                "generation_error",

            "inference_ms":
                inference_ms,

            "candidate":
                None,

            "metrics":
                None,

            "capabilities":
                None,

            "diagnostics":
                None,

            "overall":
                0.0,

            "raw_content":
                raw_content,

            "error": (
                f"{type(error).__name__}: "
                f"{error}"
            ),
        }


# ============================================================
# SINGLE MODEL
# ============================================================

def run_model(
    *,
    model: str,
    cases,
) -> dict[str, Any]:
    print()
    print(
        "#" * 80
    )

    print(
        "MODEL:",
        model,
    )

    print(
        "#" * 80
    )

    print()

    results: list[
        dict[str, Any]
    ] = []


    for index, case in enumerate(
        cases,
        start=1,
    ):
        print(
            f"[{index}/{len(cases)}]",
            case.case_id,
            "|",
            case.domain,
        )

        result = run_case(
            model=model,
            case=case,
        )

        results.append(
            result,
        )

        print(
            "Status:",
            result[
                "status"
            ],
        )

        print(
            "Inference:",
            round(
                result[
                    "inference_ms"
                ],
                1,
            ),
            "ms",
        )


        if (
            result[
                "status"
            ]
            == "ready"
        ):
            candidate = (
                result[
                    "candidate"
                ]
            )

            print(
                "Intent:",
                candidate[
                    "intent"
                ],
            )

            print(
                "Entity:",
                candidate[
                    "entity"
                ],
            )

            print(
                "Current grain:",
                candidate[
                    "current_grain"
                ],
            )

            print(
                "Target grain:",
                candidate[
                    "target_grain"
                ],
            )

            print(
                "Family:",
                candidate[
                    "family"
                ],
            )

            print(
                "Tools:",
                [
                    tool_call[
                        "name"
                    ]
                    for tool_call
                    in candidate[
                        "tool_calls"
                    ]
                ],
            )

            print(
                "Overall:",
                rounded(
                    result[
                        "overall"
                    ]
                ),
            )

        else:
            print(
                "Error:",
                result[
                    "error"
                ],
            )

        print()


    # ========================================================
    # SUCCESS / FAILURE
    # ========================================================

    ready_results = [
        result
        for result
        in results
        if (
            result[
                "status"
            ]
            == "ready"
        )
    ]


    generation_errors = [
        result
        for result
        in results
        if (
            result[
                "status"
            ]
            != "ready"
        )
    ]


    # ========================================================
    # IMPORTANT
    #
    # Failed generations remain overall=0.
    #
    # This prevents a model from improving its apparent score
    # by simply failing difficult cases.
    # ========================================================

    average_overall = average(
        [
            float(
                result[
                    "overall"
                ]
            )
            for result
            in results
        ]
    )


    # ========================================================
    # CAPABILITIES
    # ========================================================

    capability_names = [
        "comprehension",
        "planning",
        "reliability",
    ]


    average_capabilities: dict[
        str,
        float
    ] = {}


    for capability_name in (
        capability_names
    ):
        values = []

        for result in results:
            capabilities = (
                result.get(
                    "capabilities"
                )
            )

            if capabilities is None:
                values.append(
                    0.0
                )

            else:
                values.append(
                    float(
                        capabilities[
                            capability_name
                        ]
                    )
                )

        average_capabilities[
            capability_name
        ] = average(
            values
        )


    # ========================================================
    # METRICS
    # ========================================================

    metric_names = [
        "intent",
        "entity",
        "grain",
        "relevant_columns",
        "family",
        "tool_selection",
        "tool_arguments",
        "plan_consistency",
        "parsimony",
        "constraint_compliance",
        "guardrails",
    ]


    average_metrics: dict[
        str,
        float
    ] = {}


    for metric_name in (
        metric_names
    ):
        values = []

        for result in results:
            metrics = (
                result.get(
                    "metrics"
                )
            )

            if metrics is None:
                values.append(
                    0.0
                )

            else:
                values.append(
                    float(
                        metrics[
                            metric_name
                        ]
                    )
                )

        average_metrics[
            metric_name
        ] = average(
            values
        )


    # ========================================================
    # LATENCY
    # ========================================================

    total_inference_ms = sum(
        float(
            result[
                "inference_ms"
            ]
        )
        for result
        in results
    )


    average_inference_ms = (
        total_inference_ms
        / len(
            results
        )
    )


    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    extra_tool_call_count = 0
    redundant_tool_call_count = 0
    consistency_issue_count = 0
    failed_guardrail_count = 0
    invented_column_count = 0
    invented_tool_count = 0


    for result in ready_results:
        diagnostics = (
            result[
                "diagnostics"
            ]
        )

        extra_tool_call_count += len(
            diagnostics[
                "extra_tool_calls"
            ]
        )

        redundant_tool_call_count += len(
            diagnostics[
                "redundant_tool_calls"
            ]
        )

        consistency_issue_count += len(
            diagnostics[
                "consistency_issues"
            ]
        )

        failed_guardrail_count += len(
            diagnostics[
                "failed_guardrails"
            ]
        )

        invented_column_count += len(
            diagnostics[
                "invented_columns"
            ]
        )

        invented_tool_count += len(
            diagnostics[
                "invented_tools"
            ]
        )


    return {
        "model":
            model,

        "case_count":
            len(
                results
            ),

        "generation_success_count":
            len(
                ready_results
            ),

        "generation_error_count":
            len(
                generation_errors
            ),

        "average_inference_ms":
            average_inference_ms,

        "average_overall":
            average_overall,

        "average_capabilities":
            average_capabilities,

        "average_metrics":
            average_metrics,

        "diagnostics": {
            "extra_tool_call_count":
                extra_tool_call_count,

            "redundant_tool_call_count":
                redundant_tool_call_count,

            "consistency_issue_count":
                consistency_issue_count,

            "failed_guardrail_count":
                failed_guardrail_count,

            "invented_column_count":
                invented_column_count,

            "invented_tool_count":
                invented_tool_count,
        },

        "results":
            results,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        "=== DATALENS MULTI-MODEL HARD EVAL v0.5 ==="
    )

    print()

    print(
        "Benchmark:",
        BENCHMARK_PATH.name,
    )

    print(
        "Candidate contract:",
        TYPED_CANDIDATE_CONTRACT_VERSION,
    )

    print(
        "Scorer:",
        SCORER_RULE_VERSION,
    )

    print(
        "Temperature: 0"
    )

    print(
        "Thinking: disabled"
    )

    print()

    print(
        "Models:"
    )

    for model in MODELS:
        print(
            "-",
            model,
        )

    print()


    # ========================================================
    # BENCHMARK
    # ========================================================

    cases = load_benchmark(
        BENCHMARK_PATH,
        split="validation",
    )


    assert len(
        cases
    ) == 6


    print(
        "Cases:",
        len(
            cases
        ),
    )


    # ========================================================
    # RUN MODELS
    # ========================================================

    model_reports = [
        run_model(
            model=model,
            cases=cases,
        )

        for model
        in MODELS
    ]


    # ========================================================
    # RANKING
    # ========================================================

    ranking = sorted(
        model_reports,
        key=lambda report: (
            report[
                "average_overall"
            ]
        ),
        reverse=True,
    )


    # ========================================================
    # PRINT TABLE
    # ========================================================

    print()

    print(
        "=" * 100
    )

    print(
        "FINAL RANKING"
    )

    print(
        "=" * 100
    )

    print(
        f"{'MODEL':<24}"
        f"{'OVERALL':>10}"
        f"{'COMPR.':>10}"
        f"{'PLAN':>10}"
        f"{'RELIAB.':>10}"
        f"{'AVG MS':>12}"
        f"{'ERRORS':>10}"
    )

    print(
        "-" * 100
    )


    for report in ranking:
        capabilities = (
            report[
                "average_capabilities"
            ]
        )

        print(
            f"{report['model']:<24}"
            f"{rounded(report['average_overall']):>10}"
            f"{rounded(capabilities['comprehension']):>10}"
            f"{rounded(capabilities['planning']):>10}"
            f"{rounded(capabilities['reliability']):>10}"
            f"{round(report['average_inference_ms'], 1):>12}"
            f"{report['generation_error_count']:>10}"
        )


    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    print()

    print(
        "=" * 100
    )

    print(
        "DIAGNOSTICS"
    )

    print(
        "=" * 100
    )


    for report in ranking:
        diagnostics = (
            report[
                "diagnostics"
            ]
        )

        print()

        print(
            report[
                "model"
            ]
        )

        print(
            "  Extra tools:",
            diagnostics[
                "extra_tool_call_count"
            ],
        )

        print(
            "  Redundant tools:",
            diagnostics[
                "redundant_tool_call_count"
            ],
        )

        print(
            "  Consistency issues:",
            diagnostics[
                "consistency_issue_count"
            ],
        )

        print(
            "  Failed guardrails:",
            diagnostics[
                "failed_guardrail_count"
            ],
        )

        print(
            "  Invented columns:",
            diagnostics[
                "invented_column_count"
            ],
        )

        print(
            "  Invented tools:",
            diagnostics[
                "invented_tool_count"
            ],
        )


    # ========================================================
    # SAVE
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    output_payload = {
        "evaluation_version":
            MULTIMODEL_EVAL_VERSION,

        "benchmark":
            str(
                BENCHMARK_PATH
            ),

        "candidate_contract_version":
            TYPED_CANDIDATE_CONTRACT_VERSION,

        "scorer_rule_version":
            SCORER_RULE_VERSION,

        "temperature":
            0,

        "thinking":
            False,

        "excluded_models": {
            "qwen3.5:4b": (
                "Failed DataLens typed structured-output "
                "compatibility test."
            ),
        },

        "ranking": [
            {
                "rank":
                    index,

                "model":
                    report[
                        "model"
                    ],

                "average_overall":
                    report[
                        "average_overall"
                    ],

                "average_capabilities":
                    report[
                        "average_capabilities"
                    ],

                "average_inference_ms":
                    report[
                        "average_inference_ms"
                    ],

                "generation_error_count":
                    report[
                        "generation_error_count"
                    ],
            }

            for index, report
            in enumerate(
                ranking,
                start=1,
            )
        ],

        "models":
            model_reports,
    }


    OUTPUT_PATH.write_text(
        json.dumps(
            output_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print()

    print(
        "Saved:",
        OUTPUT_PATH,
    )

    print()

    print(
        "Multi-model hard evaluation: PASS"
    )


if __name__ == "__main__":
    main()