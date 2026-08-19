from __future__ import annotations

import json

from pathlib import Path

from app.evals.decision_router_benchmark_v0_7 import (
    load_decision_router_benchmark,
)

from app.evals.decision_router_runner_v0_7_1 import (
    DECISION_ROUTER_RUNNER_VERSION_V071,
    MODEL,
    PROMPT_VERSION_V071,
    run_router_train_v071,
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
    / "decision_router_multidataset_train_v0_7_2.jsonl"
)


RESULTS_DIR = (
    BASE_DIR
    / "evals"
    / "results"
    / "router_v0_7_1"
    / "multidataset_train"
)


OUTPUT_PATH = (
    RESULTS_DIR
    / "qwen3_4b_instruct_multidataset_train_v0_7_1.json"
)


# ============================================================
# CASE DISPLAY
# ============================================================

def print_case(
    result: dict,
) -> None:

    print(
        "-" * 88
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
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS QWEN MULTI-DATASET TRAIN TEST v0.7.1 ==="
    )

    print()


    print(
        "Runner:",
        DECISION_ROUTER_RUNNER_VERSION_V071,
    )


    print(
        "Prompt:",
        PROMPT_VERSION_V071,
    )


    print(
        "Model:",
        MODEL,
    )


    print(
        "Benchmark:",
        BENCHMARK_PATH.name,
    )


    print(
        "Split: train"
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
        "Prompt v0.7.1 is used unchanged."
    )


    print(
        "These 6 cases were created after the previous "
        "v0.7.1 validation failure."
    )


    print(
        "No v0.7.2 prompt exists yet."
    )


    print()


    # ========================================================
    # LOAD CHALLENGE
    # ============================================================

    cases = (
        load_decision_router_benchmark(
            BENCHMARK_PATH,
            split="train",
        )
    )


    assert (
        len(
            cases
        )
        == 6
    )


    assert all(
        not case.frozen
        for case
        in cases
    )


    assert all(
        len(
            case.datasets
        )
        == 2

        for case
        in cases
    )


    print(
        "Challenge cases:",
        len(
            cases
        ),
    )


    print()


    # ========================================================
    # RUN EXISTING v0.7.1 UNCHANGED
    # ============================================================

    report = (
        run_router_train_v071(
            cases,
        )
    )


    # ========================================================
    # DISPLAY CASES
    # ============================================================

    for result in (
        report[
            "results"
        ]
    ):

        print_case(
            result,
        )


    # ========================================================
    # SUMMARY
    # ============================================================

    print()

    print(
        "=" * 96
    )


    print(
        "MULTI-DATASET TRAIN SUMMARY — PROMPT v0.7.1"
    )


    print(
        "=" * 96
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


    print(
        "Generation errors:",
        report[
            "generation_error_count"
        ],
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


    # ========================================================
    # CONFUSION MATRIX
    # ============================================================

    print()

    print(
        "=" * 96
    )


    print(
        "CONFUSION MATRIX"
    )


    print(
        "=" * 96
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
    # IMPORTANT CASE BREAKDOWN
    # ============================================================

    print()

    print(
        "=" * 96
    )


    print(
        "CROSS-DATASET FEASIBILITY CHECK"
    )


    print(
        "=" * 96
    )


    by_id = {
        result[
            "case_id"
        ]:
            result

        for result
        in report[
            "results"
        ]
    }


    important_cases = [
        (
            "No join available",
            "router_md_v0_7_2_train_001",
        ),

        (
            "Different grains + no join",
            "router_md_v0_7_2_train_002",
        ),

        (
            "Second dataset irrelevant",
            "router_md_v0_7_2_train_003",
        ),

        (
            "Compatible join available",
            "router_md_v0_7_2_train_004",
        ),

        (
            "Join exists but no semantic link",
            "router_md_v0_7_2_train_005",
        ),

        (
            "Independent analyses",
            "router_md_v0_7_2_train_006",
        ),
    ]


    for (
        label,
        case_id,
    ) in important_cases:

        result = (
            by_id[
                case_id
            ]
        )


        candidate = (
            result.get(
                "candidate"
            )
        )


        actual = (
            candidate[
                "decision"
            ]

            if candidate
            is not None

            else "generation_error"
        )


        expected = (
            result[
                "expected_decision"
            ]
        )


        status = (
            "PASS"
            if actual == expected
            else "FAIL"
        )


        print(
            f"{label:<34}"
            f" expected={expected:<20}"
            f" actual={actual:<20}"
            f" {status}"
        )


    # ========================================================
    # SAVE
    # ============================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    OUTPUT_PATH.write_text(
        json.dumps(
            report,
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
        "Qwen multi-dataset train test v0.7.1: PASS"
    )


if __name__ == "__main__":
    main()