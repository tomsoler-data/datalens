from __future__ import annotations

import json

from pathlib import Path

from app.evals.decision_router_benchmark_v0_7 import (
    load_decision_router_benchmark,
)

from app.evals.decision_router_runner_v0_7 import (
    DECISION_ROUTER_RUNNER_VERSION,
    MODELS,
    PROMPT_VERSION,
    run_router_model,
    save_router_report,
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
    / "decision_router_development_v0_7.jsonl"
)


RESULTS_DIR = (
    BASE_DIR
    / "evals"
    / "results"
    / "router_v0_7"
)


COMBINED_OUTPUT_PATH = (
    RESULTS_DIR
    / "decision_router_validation_baseline_v0_7.json"
)


# ============================================================
# DISPLAY CASE
# ============================================================

def print_case_result(
    result: dict,
) -> None:

    print(
        "-" * 82
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
        "Expected:",
        result[
            "expected_decision"
        ],
        "|",
        result[
            "expected_reason"
        ],
    )


    print(
        "Status:",
        result[
            "status"
        ],
    )


    candidate = (
        result.get(
            "candidate"
        )
    )


    if candidate is None:
        print(
            "Error:",
            result[
                "error"
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

        return


    print(
        "Decision:",
        candidate[
            "decision"
        ],
    )


    print(
        "Reason:",
        candidate[
            "decision_reason"
        ],
    )


    print(
        "Clarification:",
        candidate[
            "clarification_question"
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


    diagnostics = (
        score[
            "diagnostics"
        ]
    )


    print(
        "Decision score:",
        round(
            metrics[
                "decision"
            ],
            3,
        ),
    )


    print(
        "Reason score:",
        round(
            metrics[
                "decision_reason"
            ],
            3,
        ),
    )


    if (
        metrics[
            "clarification"
        ]
        is not None
    ):
        print(
            "Clarification score:",
            round(
                metrics[
                    "clarification"
                ],
                3,
            ),
        )


    print(
        "Route quality:",
        round(
            metrics[
                "route_quality"
            ],
            3,
        ),
    )


    print(
        "Overall:",
        round(
            result[
                "overall"
            ],
            3,
        ),
    )


    print(
        "Unsafe execution:",
        diagnostics[
            "unsafe_execution"
        ],
    )


    print(
        "False abstention:",
        diagnostics[
            "false_abstention"
        ],
    )


    print(
        "Wrong abstention type:",
        diagnostics[
            "wrong_abstention_type"
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


# ============================================================
# DISPLAY MODEL SUMMARY
# ============================================================

def print_model_summary(
    report: dict,
) -> None:

    print()

    print(
        "=" * 88
    )


    print(
        "MODEL SUMMARY:",
        report[
            "model"
        ],
    )


    print(
        "=" * 88
    )


    print(
        "Overall:",
        round(
            report[
                "average_overall"
            ],
            3,
        ),
    )


    print(
        "Decision accuracy:",
        round(
            report[
                "decision_accuracy"
            ],
            3,
        ),
    )


    print(
        "Reason accuracy:",
        round(
            report[
                "reason_accuracy"
            ],
            3,
        ),
    )


    print(
        "Route quality:",
        round(
            report[
                "route_quality"
            ],
            3,
        ),
    )


    print(
        "Clarification quality:",
        round(
            report[
                "clarification_quality"
            ],
            3,
        ),
    )


    print(
        "Average inference:",
        round(
            report[
                "average_inference_ms"
            ],
            1,
        ),
        "ms",
    )


    print(
        "Generation errors:",
        report[
            "generation_error_count"
        ],
    )


    print()


    accuracy = (
        report[
            "accuracy_by_expected"
        ]
    )


    print(
        "Analyze accuracy:",
        round(
            accuracy[
                "analyze"
            ],
            3,
        ),
    )


    print(
        "Clarification accuracy:",
        round(
            accuracy[
                "needs_clarification"
            ],
            3,
        ),
    )


    print(
        "Cannot-answer accuracy:",
        round(
            accuracy[
                "cannot_answer"
            ],
            3,
        ),
    )


    print()


    diagnostics = (
        report[
            "diagnostics"
        ]
    )


    print(
        "Unsafe executions:",
        diagnostics[
            "unsafe_execution_count"
        ],
    )


    print(
        "False abstentions:",
        diagnostics[
            "false_abstention_count"
        ],
    )


    print(
        "Wrong abstention types:",
        diagnostics[
            "wrong_abstention_type_count"
        ],
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS DECISION ROUTER BASELINE v0.7 ==="
    )

    print()


    print(
        "Runner:",
        DECISION_ROUTER_RUNNER_VERSION,
    )


    print(
        "Prompt:",
        PROMPT_VERSION,
    )


    print(
        "Benchmark:",
        BENCHMARK_PATH.name,
    )


    print(
        "Split: validation"
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
        "Only the development VALIDATION split is used."
    )


    print(
        "The 9 training cases are not sent to the models."
    )


    print(
        "The frozen v0.6 benchmark is not used."
    )


    print()


    # ========================================================
    # LOAD VALIDATION ONLY
    # ========================================================

    validation_cases = (
        load_decision_router_benchmark(
            BENCHMARK_PATH,
            split="validation",
        )
    )


    assert (
        len(
            validation_cases
        )
        == 9
    )


    assert all(
        not case.frozen
        for case
        in validation_cases
    )


    print(
        "Validation cases:",
        len(
            validation_cases
        ),
    )


    print(
        "Models:",
        MODELS,
    )


    # ========================================================
    # RUN
    # ========================================================

    reports = []


    for model in MODELS:

        print()

        print(
            "#" * 88
        )


        print(
            "MODEL:",
            model,
        )


        print(
            "#" * 88
        )


        print()


        report = (
            run_router_model(
                model=model,
                cases=validation_cases,
            )
        )


        reports.append(
            report,
        )


        checkpoint_path = (
            save_router_report(
                report=report,
                output_dir=RESULTS_DIR,
            )
        )


        for result in (
            report[
                "results"
            ]
        ):
            print_case_result(
                result,
            )


        print_model_summary(
            report,
        )


        print()

        print(
            "Checkpoint:",
            checkpoint_path,
        )


    # ========================================================
    # RANKING
    # ========================================================

    ranking = sorted(
        reports,

        key=lambda report: (
            report[
                "average_overall"
            ]
        ),

        reverse=True,
    )


    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()

    print(
        "=" * 128
    )


    print(
        "DECISION ROUTER VALIDATION RANKING"
    )


    print(
        "=" * 128
    )


    print(
        f"{'MODEL':<24}"
        f"{'OVERALL':>10}"
        f"{'DECISION':>11}"
        f"{'REASON':>10}"
        f"{'ROUTE':>10}"
        f"{'CLARIFY Q':>11}"
        f"{'ANALYZE':>10}"
        f"{'CLARIFY':>10}"
        f"{'ABSTAIN':>10}"
        f"{'UNSAFE':>9}"
        f"{'ERRORS':>9}"
        f"{'AVG MS':>12}"
    )


    print(
        "-" * 128
    )


    for report in ranking:

        accuracy = (
            report[
                "accuracy_by_expected"
            ]
        )


        diagnostics = (
            report[
                "diagnostics"
            ]
        )


        print(
            f"{report['model']:<24}"
            f"{round(report['average_overall'], 3):>10}"
            f"{round(report['decision_accuracy'], 3):>11}"
            f"{round(report['reason_accuracy'], 3):>10}"
            f"{round(report['route_quality'], 3):>10}"
            f"{round(report['clarification_quality'], 3):>11}"
            f"{round(accuracy['analyze'], 3):>10}"
            f"{round(accuracy['needs_clarification'], 3):>10}"
            f"{round(accuracy['cannot_answer'], 3):>10}"
            f"{diagnostics['unsafe_execution_count']:>9}"
            f"{report['generation_error_count']:>9}"
            f"{round(report['average_inference_ms'], 1):>12}"
        )


    # ========================================================
    # CONFUSION MATRICES
    # ========================================================

    print()

    print(
        "=" * 128
    )


    print(
        "CONFUSION MATRICES"
    )


    print(
        "=" * 128
    )


    for report in ranking:

        print()

        print(
            report[
                "model"
            ]
        )


        print(
            json.dumps(
                report[
                    "confusion_matrix"
                ],
                ensure_ascii=False,
                indent=2,
            )
        )


    # ========================================================
    # SAVE COMBINED RESULT
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    combined_payload = {
        "evaluation":
            DECISION_ROUTER_RUNNER_VERSION,

        "prompt_version":
            PROMPT_VERSION,

        "benchmark":
            str(
                BENCHMARK_PATH
            ),

        "split":
            "validation",

        "frozen":
            False,

        "temperature":
            0,

        "thinking":
            False,

        "models":
            MODELS,

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

                "decision_accuracy":
                    report[
                        "decision_accuracy"
                    ],

                "reason_accuracy":
                    report[
                        "reason_accuracy"
                    ],

                "route_quality":
                    report[
                        "route_quality"
                    ],

                "clarification_quality":
                    report[
                        "clarification_quality"
                    ],

                "accuracy_by_expected":
                    report[
                        "accuracy_by_expected"
                    ],

                "diagnostics":
                    report[
                        "diagnostics"
                    ],

                "generation_error_count":
                    report[
                        "generation_error_count"
                    ],

                "average_inference_ms":
                    report[
                        "average_inference_ms"
                    ],
            }

            for index, report
            in enumerate(
                ranking,
                start=1,
            )
        ],

        "reports":
            reports,
    }


    COMBINED_OUTPUT_PATH.write_text(
        json.dumps(
            combined_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print()

    print(
        "Combined result:",
        COMBINED_OUTPUT_PATH,
    )


    print()

    print(
        "Decision Router baseline v0.7: PASS"
    )


if __name__ == "__main__":
    main()