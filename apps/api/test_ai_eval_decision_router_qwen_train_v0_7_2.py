from __future__ import annotations

import json

from pathlib import Path

from app.evals.decision_router_benchmark_v0_7 import (
    load_decision_router_benchmark,
)

from app.evals.decision_router_runner_v0_7_2 import (
    DECISION_ROUTER_RUNNER_VERSION_V072,
    MODEL,
    PROMPT_VERSION_V072,
    run_router_cases_v072,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__,
).resolve().parent


CORE_TRAIN_PATH = (
    BASE_DIR
    / "evals"
    / "decision_router_development_v0_7.jsonl"
)


MULTIDATASET_TRAIN_PATH = (
    BASE_DIR
    / "evals"
    / "decision_router_multidataset_train_v0_7_2.jsonl"
)


RESULTS_DIR = (
    BASE_DIR
    / "evals"
    / "results"
    / "router_v0_7_2"
    / "train"
)


OUTPUT_PATH = (
    RESULTS_DIR
    / "qwen3_4b_instruct_router_combined_train_v0_7_2.json"
)


# ============================================================
# DISPLAY
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
        "Reason score:",
        round(
            score[
                "metrics"
            ][
                "decision_reason"
            ],
            3,
        ),
    )


    print(
        "Unsafe:",
        score[
            "diagnostics"
        ][
            "unsafe_execution"
        ],
    )


    print(
        "False abstention:",
        score[
            "diagnostics"
        ][
            "false_abstention"
        ],
    )


    print(
        "Wrong abstention type:",
        score[
            "diagnostics"
        ][
            "wrong_abstention_type"
        ],
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS DECISION ROUTER COMBINED TRAIN v0.7.2 ==="
    )

    print()


    print(
        "Runner:",
        DECISION_ROUTER_RUNNER_VERSION_V072,
    )


    print(
        "Prompt:",
        PROMPT_VERSION_V072,
    )


    print(
        "Model:",
        MODEL,
    )


    print(
        "Split: train only"
    )


    print(
        "Validation: NOT USED"
    )


    print()


    # ========================================================
    # LOAD ORIGINAL TRAIN
    # ========================================================

    core_train = (
        load_decision_router_benchmark(
            CORE_TRAIN_PATH,
            split="train",
        )
    )


    assert (
        len(
            core_train
        )
        == 9
    )


    # ========================================================
    # LOAD MULTI-DATASET TRAIN
    # ========================================================

    multidataset_train = (
        load_decision_router_benchmark(
            MULTIDATASET_TRAIN_PATH,
            split="train",
        )
    )


    assert (
        len(
            multidataset_train
        )
        == 6
    )


    # ========================================================
    # COMBINE
    # ========================================================

    cases = [
        *core_train,
        *multidataset_train,
    ]


    assert (
        len(
            cases
        )
        == 15
    )


    case_ids = [
        case.case_id
        for case
        in cases
    ]


    assert (
        len(
            set(
                case_ids
            )
        )
        == 15
    )


    assert all(
        not case.frozen
        for case
        in cases
    )


    print(
        "Core train:",
        len(
            core_train
        ),
    )


    print(
        "Multi-dataset train:",
        len(
            multidataset_train
        ),
    )


    print(
        "Combined train:",
        len(
            cases
        ),
    )


    print()


    # ========================================================
    # RUN
    # ========================================================

    report = (
        run_router_cases_v072(
            cases,
        )
    )


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
    # ========================================================

    print()

    print(
        "=" * 100
    )


    print(
        "COMBINED TRAIN SUMMARY v0.7.2"
    )


    print(
        "=" * 100
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
    # ========================================================

    print()

    print(
        "=" * 100
    )


    print(
        "CONFUSION MATRIX"
    )


    print(
        "=" * 100
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
    # MULTI-DATASET DETAIL
    # ========================================================

    print()

    print(
        "=" * 100
    )


    print(
        "MULTI-DATASET DETAIL"
    )


    print(
        "=" * 100
    )


    for result in report[
        "results"
    ]:

        if not (
            result[
                "case_id"
            ]
            .startswith(
                "router_md_"
            )
        ):
            continue


        candidate = (
            result.get(
                "candidate"
            )
        )


        actual_decision = (
            candidate[
                "decision"
            ]

            if candidate

            else "generation_error"
        )


        actual_reason = (
            candidate[
                "decision_reason"
            ]

            if candidate

            else None
        )


        exact = (
            actual_decision
            == result[
                "expected_decision"
            ]

            and actual_reason
            == result[
                "expected_reason"
            ]
        )


        print(
            result[
                "case_id"
            ],
            "| expected:",
            result[
                "expected_decision"
            ],
            "/",
            result[
                "expected_reason"
            ],
            "| actual:",
            actual_decision,
            "/",
            actual_reason,
            "|",
            (
                "PASS"
                if exact
                else "FAIL"
            ),
        )


    # ========================================================
    # SAVE
    # ========================================================

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
        "Decision Router combined train v0.7.2: PASS"
    )


if __name__ == "__main__":
    main()