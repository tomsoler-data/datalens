from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from app.evals.analytical_planner_benchmark_v0_9 import (
    ANALYTICAL_PLANNER_BENCHMARK_VERSION,
    load_analytical_planner_benchmark,
)

from app.evals.analytical_planner_model_runner_v0_9 import (
    ANALYTICAL_PLANNER_MODEL_RUNNER_VERSION,
    ANALYTICAL_PLANNER_PROMPT_VERSION,
    PLANNER_MODELS,
    analytical_planner_runner_metadata,
    run_analytical_planner_case,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__,
).resolve().parent


BENCHMARK_PATH = (
    BASE_DIR
    / "evals"
    / "analytical_planner_development_v0_9.jsonl"
)


RESULTS_DIR = (
    BASE_DIR
    / "evals"
    / "results"
    / "analytical_planner_v0_9"
)


OUTPUT_PATH = (
    RESULTS_DIR
    / "analytical_planner_validation_models_v0_9.json"
)


# ============================================================
# HELPERS
# ============================================================

def _average(
    values: list[float],
) -> float:

    if not values:
        return 0.0


    return (
        sum(
            values
        )
        / len(
            values
        )
    )


def _safe_metric(
    result: dict[str, Any],
    metric_name: str,
) -> float:

    score = (
        result.get(
            "score"
        )
    )


    if score is None:
        return 0.0


    return float(
        score[
            "metrics"
        ][
            metric_name
        ]
    )


# ============================================================
# MODEL SUMMARY
# ============================================================

def summarize_model(
    *,
    model: str,
    results: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:

    total = len(
        results
    )


    if total == 0:

        raise ValueError(
            f"No results available for model {model}."
        )


    generation_errors = sum(
        1

        for result
        in results

        if (
            result[
                "status"
            ]
            != "ready"
        )
    )


    valid_plans = sum(
        1

        for result
        in results

        if (
            result[
                "status"
            ]
            == "ready"

            and result[
                "validation"
            ][
                "valid"
            ]
        )
    )


    exact_plans = sum(
        1

        for result
        in results

        if result[
            "exact"
        ]
    )


    overall_scores = [
        (
            float(
                result[
                    "score"
                ][
                    "overall"
                ]
            )

            if (
                result.get(
                    "score"
                )
                is not None
            )

            else 0.0
        )

        for result
        in results
    ]


    requirement_scores = [
        _safe_metric(
            result,
            "requirement_coverage_f1",
        )

        for result
        in results
    ]


    intent_scores = [
        _safe_metric(
            result,
            "intent_accuracy",
        )

        for result
        in results
    ]


    family_scores = [
        _safe_metric(
            result,
            "family_accuracy",
        )

        for result
        in results
    ]


    grain_scores = [
        _safe_metric(
            result,
            "target_grain_accuracy",
        )

        for result
        in results
    ]


    sequence_scores = [
        _safe_metric(
            result,
            "tool_sequence_score",
        )

        for result
        in results
    ]


    argument_scores = [
        _safe_metric(
            result,
            "tool_argument_score",
        )

        for result
        in results
    ]


    validator_scores = [
        _safe_metric(
            result,
            "validator_acceptance",
        )

        for result
        in results
    ]


    issue_codes: list[
        str
    ] = []


    for result in results:

        if (
            result[
                "status"
            ]
            != "ready"
        ):
            continue


        for issue in (
            result[
                "validation"
            ][
                "issues"
            ]
        ):

            issue_codes.append(
                issue[
                    "code"
                ]
            )


    average_inference_ms = (
        _average(
            [
                float(
                    result[
                        "inference_ms"
                    ]
                )

                for result
                in results
            ]
        )
    )


    return {
        "model":
            model,

        "case_count":
            total,

        "average_overall":
            _average(
                overall_scores
            ),

        "exact_plan_count":
            exact_plans,

        "exact_plan_accuracy":
            (
                exact_plans
                / total
            ),

        "valid_plan_count":
            valid_plans,

        "validator_acceptance_rate":
            (
                valid_plans
                / total
            ),

        "requirement_coverage_f1":
            _average(
                requirement_scores
            ),

        "intent_accuracy":
            _average(
                intent_scores
            ),

        "family_accuracy":
            _average(
                family_scores
            ),

        "target_grain_accuracy":
            _average(
                grain_scores
            ),

        "tool_sequence_score":
            _average(
                sequence_scores
            ),

        "tool_argument_score":
            _average(
                argument_scores
            ),

        "validator_score":
            _average(
                validator_scores
            ),

        "generation_error_count":
            generation_errors,

        "validator_issue_codes":
            issue_codes,

        "average_inference_ms":
            average_inference_ms,
    }


# ============================================================
# CASE DISPLAY
# ============================================================

def print_case_result(
    result: dict[
        str,
        Any,
    ],
) -> None:

    print(
        "-" * 110
    )


    print(
        result[
            "case_id"
        ],
        "|",
        result[
            "domain"
        ],
    )


    print(
        "Status:",
        result[
            "status"
        ],
    )


    if (
        result[
            "status"
        ]
        != "ready"
    ):

        print(
            "Error:",
            result[
                "error"
            ],
        )


        print(
            "Inference:",
            round(
                float(
                    result[
                        "inference_ms"
                    ]
                ),
                1,
            ),
            "ms",
        )


        return


    candidate = (
        result[
            "candidate"
        ]
    )


    print(
        "Plans:"
    )


    for plan in (
        candidate[
            "plans"
        ]
    ):

        actions = [
            step[
                "action"
            ][
                "name"
            ]

            for step
            in plan[
                "steps"
            ]
        ]


        print(
            " ",
            plan[
                "requirement_id"
            ],
            "|",
            plan[
                "intent"
            ],
            "|",
            plan[
                "family"
            ],
            "| grain=",
            plan[
                "target_grain"
            ],
            "| tools=",
            actions,
        )


    validation = (
        result[
            "validation"
        ]
    )


    print(
        "Python validator:",
        (
            "PASS"
            if (
                validation[
                    "valid"
                ]
            )
            else "FAIL"
        ),
    )


    if (
        validation[
            "issues"
        ]
    ):

        print(
            "Validator issues:",
            [
                issue[
                    "code"
                ]

                for issue
                in validation[
                    "issues"
                ]
            ],
        )


    score = (
        result[
            "score"
        ]
    )


    metrics = (
        score[
            "metrics"
        ]
    )


    print(
        "Overall:",
        round(
            float(
                score[
                    "overall"
                ]
            ),
            3,
        ),
    )


    print(
        "Requirement coverage:",
        round(
            float(
                metrics[
                    "requirement_coverage_f1"
                ]
            ),
            3,
        ),
    )


    print(
        "Intent:",
        round(
            float(
                metrics[
                    "intent_accuracy"
                ]
            ),
            3,
        ),
    )


    print(
        "Family:",
        round(
            float(
                metrics[
                    "family_accuracy"
                ]
            ),
            3,
        ),
    )


    print(
        "Grain:",
        round(
            float(
                metrics[
                    "target_grain_accuracy"
                ]
            ),
            3,
        ),
    )


    print(
        "Tool sequence:",
        round(
            float(
                metrics[
                    "tool_sequence_score"
                ]
            ),
            3,
        ),
    )


    print(
        "Tool arguments:",
        round(
            float(
                metrics[
                    "tool_argument_score"
                ]
            ),
            3,
        ),
    )


    print(
        "Exact:",
        result[
            "exact"
        ],
    )


    print(
        "Inference:",
        round(
            float(
                result[
                    "inference_ms"
                ]
            ),
            1,
        ),
        "ms",
    )


# ============================================================
# SUMMARY DISPLAY
# ============================================================

def print_model_summary(
    summary: dict[
        str,
        Any,
    ],
) -> None:

    print()

    print(
        "=" * 110
    )


    print(
        "MODEL SUMMARY:",
        summary[
            "model"
        ],
    )


    print(
        "=" * 110
    )


    print(
        "Cases:",
        summary[
            "case_count"
        ],
    )


    print(
        "Average overall:",
        round(
            summary[
                "average_overall"
            ],
            3,
        ),
    )


    print(
        "Exact plans:",
        (
            f"{summary['exact_plan_count']}"
            "/"
            f"{summary['case_count']}"
        ),
    )


    print(
        "Validator accepted:",
        (
            f"{summary['valid_plan_count']}"
            "/"
            f"{summary['case_count']}"
        ),
    )


    print()


    print(
        "Requirement coverage:",
        round(
            summary[
                "requirement_coverage_f1"
            ],
            3,
        ),
    )


    print(
        "Intent accuracy:",
        round(
            summary[
                "intent_accuracy"
            ],
            3,
        ),
    )


    print(
        "Family accuracy:",
        round(
            summary[
                "family_accuracy"
            ],
            3,
        ),
    )


    print(
        "Target grain accuracy:",
        round(
            summary[
                "target_grain_accuracy"
            ],
            3,
        ),
    )


    print(
        "Tool sequence:",
        round(
            summary[
                "tool_sequence_score"
            ],
            3,
        ),
    )


    print(
        "Tool arguments:",
        round(
            summary[
                "tool_argument_score"
            ],
            3,
        ),
    )


    print()


    print(
        "Generation errors:",
        summary[
            "generation_error_count"
        ],
    )


    print(
        "Validator issues:",
        summary[
            "validator_issue_codes"
        ],
    )


    print(
        "Average inference:",
        round(
            summary[
                "average_inference_ms"
            ],
            1,
        ),
        "ms",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL PLANNER MODEL COMPARISON v0.9 ==="
    )


    print()


    metadata = (
        analytical_planner_runner_metadata()
    )


    print(
        "Runner:",
        ANALYTICAL_PLANNER_MODEL_RUNNER_VERSION,
    )


    print(
        "Prompt:",
        ANALYTICAL_PLANNER_PROMPT_VERSION,
    )


    print(
        "Benchmark:",
        ANALYTICAL_PLANNER_BENCHMARK_VERSION,
    )


    print(
        "Split: validation"
    )


    print(
        "Models:",
        PLANNER_MODELS,
    )


    print(
        "Temperature: 0"
    )


    print(
        "Thinking: disabled"
    )


    print()


    print(
        "IMPORTANT:"
    )


    print(
        "- expected plans are never sent to the models"
    )


    print(
        "- both models receive the exact same planner input"
    )


    print(
        "- both models use the exact same structured output schema"
    )


    print(
        "- both models are evaluated by the same Python validator"
    )


    print(
        "- this is development validation, not frozen evaluation"
    )


    print()


    # ========================================================
    # LOAD VALIDATION CASES
    # ========================================================

    cases = (
        load_analytical_planner_benchmark(
            BENCHMARK_PATH,
            split="validation",
        )
    )


    assert (
        len(
            cases
        )
        == 5
    )


    print(
        "Validation cases:",
        len(
            cases
        ),
    )


    print()


    # ========================================================
    # RUN MODELS
    # ========================================================

    all_results: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = {}


    summaries: list[
        dict[str, Any]
    ] = []


    for model in PLANNER_MODELS:

        print()

        print(
            "#" * 110
        )


        print(
            "MODEL:",
            model,
        )


        print(
            "#" * 110
        )


        model_results: list[
            dict[str, Any]
        ] = []


        for (
            index,
            case,
        ) in enumerate(
            cases,
            start=1,
        ):

            print()

            print(
                f"[{index}/{len(cases)}]",
                model,
                "->",
                case.case_id,
            )


            result = (
                run_analytical_planner_case(
                    case=case,
                    model=model,
                )
            )


            model_results.append(
                result
            )


            print_case_result(
                result
            )


        all_results[
            model
        ] = model_results


        summary = (
            summarize_model(
                model=model,
                results=model_results,
            )
        )


        summaries.append(
            summary
        )


        print_model_summary(
            summary
        )


    # ========================================================
    # COMPARISON
    # ========================================================

    ranked = sorted(
        summaries,
        key=lambda summary: (
            summary[
                "average_overall"
            ],
            summary[
                "validator_acceptance_rate"
            ],
            summary[
                "exact_plan_accuracy"
            ],
        ),
        reverse=True,
    )


    print()

    print(
        "=" * 110
    )


    print(
        "MODEL COMPARISON"
    )


    print(
        "=" * 110
    )


    for (
        rank,
        summary,
    ) in enumerate(
        ranked,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{summary['model']:<22}"
            f" overall="
            f"{summary['average_overall']:.3f}"
            f" validator="
            f"{summary['validator_acceptance_rate']:.3f}"
            f" exact="
            f"{summary['exact_plan_accuracy']:.3f}"
            f" inference="
            f"{summary['average_inference_ms']:.1f}ms"
        )


    # ========================================================
    # SAVE
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    payload = {
        "evaluation":
            ANALYTICAL_PLANNER_MODEL_RUNNER_VERSION,

        "metadata":
            metadata,

        "benchmark":
            str(
                BENCHMARK_PATH
            ),

        "split":
            "validation",

        "frozen":
            False,

        "case_count":
            len(
                cases
            ),

        "summaries":
            summaries,

        "ranking": [
            summary[
                "model"
            ]

            for summary
            in ranked
        ],

        "results":
            all_results,
    }


    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
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
        "Analytical Planner model comparison v0.9: COMPLETE"
    )


if __name__ == "__main__":
    main()