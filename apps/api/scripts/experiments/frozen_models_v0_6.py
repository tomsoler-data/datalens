from __future__ import annotations

import json

from pathlib import Path

from app.evals.decision_benchmark_v0_6 import (
    load_decision_benchmark,
)

from app.evals.frozen_runner_v0_6 import (
    FROZEN_RUNNER_VERSION,
    MODELS,
    run_frozen_model,
    save_model_report,
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
    / "analytical_decision_frozen_v0_6.jsonl"
)


RESULTS_DIR = (
    BASE_DIR
    / "evals"
    / "results"
    / "frozen_v0_6"
)


COMBINED_OUTPUT_PATH = (
    RESULTS_DIR
    / "frozen_decision_multimodel_v0_6.json"
)


# ============================================================
# DISPLAY
# ============================================================

def print_model_case(
    result: dict,
) -> None:
    print(
        "-" * 78
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
    )

    print(
        "Status:",
        result[
            "status"
        ],
    )


    candidate = result.get(
        "candidate",
    )


    if candidate is not None:
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


        if (
            candidate[
                "decision"
            ]
            == "analyze"
        ):
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
                "Relevant columns:",
                candidate[
                    "relevant_columns"
                ],
            )

            print(
                "Tools:",
                [
                    call[
                        "name"
                    ]
                    for call
                    in candidate[
                        "tool_calls"
                    ]
                ],
            )


        score = (
            result[
                "score"
            ]
        )


        print(
            "Decision score:",
            round(
                score[
                    "metrics"
                ][
                    "decision"
                ],
                3,
            ),
        )

        print(
            "Route quality:",
            round(
                score[
                    "metrics"
                ][
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


        diagnostics = (
            score[
                "diagnostics"
            ]
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

    else:
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


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        "=== DATALENS FROZEN DECISION EVAL v0.6 ==="
    )

    print()

    print(
        "Runner:",
        FROZEN_RUNNER_VERSION,
    )

    print(
        "Benchmark:",
        BENCHMARK_PATH.name,
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
        "This benchmark is frozen."
    )

    print(
        "No expected answer is sent to the models."
    )

    print()


    # ========================================================
    # LOAD FROZEN SET
    # ========================================================

    cases = load_decision_benchmark(
        BENCHMARK_PATH,
    )


    assert len(
        cases
    ) == 15


    assert all(
        case.frozen
        for case
        in cases
    )


    print(
        "Frozen cases:",
        len(
            cases
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


        report = run_frozen_model(
            model=model,
            cases=cases,
        )


        reports.append(
            report,
        )


        # ----------------------------------------------------
        # Save immediately after each model.
        # This preserves the first frozen run even if a later
        # model or terminal session fails.
        # ----------------------------------------------------

        checkpoint_path = (
            save_model_report(
                report=report,
                output_dir=RESULTS_DIR,
            )
        )


        for result in (
            report[
                "results"
            ]
        ):
            print_model_case(
                result,
            )


        print()

        print(
            "=" * 88
        )

        print(
            "MODEL SUMMARY:",
            model,
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
            "Route quality:",
            round(
                report[
                    "route_quality"
                ],
                3,
            ),
        )

        print(
            "Analytical plan quality:",
            round(
                report[
                    "average_analytical_plan"
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

        print(
            "Analyze accuracy:",
            round(
                report[
                    "decision_accuracy_by_expected"
                ][
                    "analyze"
                ],
                3,
            ),
        )

        print(
            "Clarification accuracy:",
            round(
                report[
                    "decision_accuracy_by_expected"
                ][
                    "needs_clarification"
                ],
                3,
            ),
        )

        print(
            "Cannot-answer accuracy:",
            round(
                report[
                    "decision_accuracy_by_expected"
                ][
                    "cannot_answer"
                ],
                3,
            ),
        )

        print()

        print(
            "Unsafe executions:",
            report[
                "diagnostics"
            ][
                "unsafe_execution_count"
            ],
        )

        print(
            "False abstentions:",
            report[
                "diagnostics"
            ][
                "false_abstention_count"
            ],
        )

        print(
            "Wrong abstention types:",
            report[
                "diagnostics"
            ][
                "wrong_abstention_type_count"
            ],
        )

        print()

        print(
            "Checkpoint:",
            checkpoint_path,
        )


    # ========================================================
    # FINAL RANKING
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


    print()

    print(
        "=" * 116
    )

    print(
        "FROZEN TEST FINAL RANKING"
    )

    print(
        "=" * 116
    )


    print(
        f"{'MODEL':<24}"
        f"{'OVERALL':>10}"
        f"{'DECISION':>11}"
        f"{'ROUTE':>10}"
        f"{'PLAN':>10}"
        f"{'ANALYZE':>10}"
        f"{'CLARIFY':>10}"
        f"{'ABSTAIN':>10}"
        f"{'UNSAFE':>9}"
        f"{'AVG MS':>12}"
    )


    print(
        "-" * 116
    )


    for report in ranking:
        by_route = (
            report[
                "decision_accuracy_by_expected"
            ]
        )


        print(
            f"{report['model']:<24}"
            f"{round(report['average_overall'], 3):>10}"
            f"{round(report['decision_accuracy'], 3):>11}"
            f"{round(report['route_quality'], 3):>10}"
            f"{round(report['average_analytical_plan'], 3):>10}"
            f"{round(by_route['analyze'], 3):>10}"
            f"{round(by_route['needs_clarification'], 3):>10}"
            f"{round(by_route['cannot_answer'], 3):>10}"
            f"{report['diagnostics']['unsafe_execution_count']:>9}"
            f"{round(report['average_inference_ms'], 1):>12}"
        )


    # ========================================================
    # CONFUSION MATRICES
    # ========================================================

    print()

    print(
        "=" * 116
    )

    print(
        "DECISION CONFUSION MATRICES"
    )

    print(
        "=" * 116
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
            FROZEN_RUNNER_VERSION,

        "benchmark":
            str(
                BENCHMARK_PATH
            ),

        "frozen":
            True,

        "model_selection_before_test": (
            MODELS
        ),

        "ranking": [
            {
                "rank":
                    rank,

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

                "route_quality":
                    report[
                        "route_quality"
                    ],

                "average_analytical_plan":
                    report[
                        "average_analytical_plan"
                    ],

                "decision_accuracy_by_expected":
                    report[
                        "decision_accuracy_by_expected"
                    ],

                "diagnostics":
                    report[
                        "diagnostics"
                    ],

                "average_inference_ms":
                    report[
                        "average_inference_ms"
                    ],
            }

            for rank, report
            in enumerate(
                ranking,
                start=1,
            )
        ],

        "models":
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
        "Frozen decision evaluation: PASS"
    )


if __name__ == "__main__":
    main()