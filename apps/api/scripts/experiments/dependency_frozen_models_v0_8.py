from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from app.evals.dataset_dependency_benchmark_v0_8 import (
    load_dataset_dependency_frozen_benchmark,
)

from app.evals.dataset_dependency_frozen_runner_v0_8 import (
    DATASET_DEPENDENCY_FROZEN_RUNNER_VERSION,
    frozen_runner_metadata,
    run_frozen_dependency_case,
    summarize_frozen_results,
)

from app.evals.dataset_dependency_extractor_v0_8 import (
    MODEL,
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
    / "dataset_dependency_frozen_v0_8.jsonl"
)


RESULTS_DIR = (
    BASE_DIR
    / "evals"
    / "results"
    / "dependency_frozen_v0_8"
)


CHECKPOINT_PATH = (
    RESULTS_DIR
    / "qwen3_4b_instruct_frozen_dependency_v0_8_checkpoint.json"
)


FINAL_OUTPUT_PATH = (
    RESULTS_DIR
    / "qwen3_4b_instruct_frozen_dependency_pipeline_v0_8.json"
)


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint() -> dict[
    str,
    dict[str, Any],
]:

    if not (
        CHECKPOINT_PATH.exists()
    ):
        return {}


    payload = json.loads(
        CHECKPOINT_PATH.read_text(
            encoding="utf-8",
        )
    )


    results = (
        payload.get(
            "results",
            [],
        )
    )


    by_case_id = {
        result[
            "case_id"
        ]:
            result

        for result
        in results
    }


    if (
        len(
            by_case_id
        )
        != len(
            results
        )
    ):
        raise ValueError(
            "Duplicate case IDs in frozen checkpoint."
        )


    return by_case_id


def save_checkpoint(
    results_by_case_id: dict[
        str,
        dict[str, Any],
    ],
) -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    payload = {
        "evaluation":
            DATASET_DEPENDENCY_FROZEN_RUNNER_VERSION,

        "metadata":
            frozen_runner_metadata(),

        "benchmark":
            str(
                BENCHMARK_PATH
            ),

        "frozen":
            True,

        "results": list(
            results_by_case_id.values()
        ),
    }


    CHECKPOINT_PATH.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# DISPLAY CASE
# ============================================================

def print_case(
    result: dict[
        str,
        Any,
    ],
) -> None:

    print(
        "-" * 104
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
        == "generation_error"
    ):

        print(
            "Generation error:",
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


    candidate = (
        result[
            "candidate"
        ]
    )


    groups = [
        requirement[
            "dataset_ids"
        ]

        for requirement
        in candidate[
            "requirements"
        ]
    ]


    print(
        "AI groups:",
        groups,
    )


    print(
        "Semantic exact:",
        result[
            "semantic_exact"
        ],
    )


    semantic_score = (
        result[
            "semantic_score"
        ]
    )


    if semantic_score is not None:

        print(
            "Semantic overall:",
            semantic_score[
                "overall"
            ],
        )


        print(
            "Dataset F1:",
            semantic_score[
                "metrics"
            ][
                "dataset_f1"
            ],
        )


        print(
            "Grouping F1:",
            semantic_score[
                "metrics"
            ][
                "pairwise_grouping_f1"
            ],
        )


    if (
        result[
            "status"
        ]
        == "gate_error"
    ):

        print(
            "Python gate error:",
            result[
                "gate_error"
            ],
        )


        print(
            "End-to-end exact:",
            False,
        )


        return


    gate = (
        result[
            "gate"
        ]
    )


    print(
        "Executable:",
        gate[
            "executable"
        ],
    )


    print(
        "Feasibilities:",
        [
            requirement[
                "feasibility"
            ]

            for requirement
            in gate[
                "requirements"
            ]
        ],
    )


    print(
        "Override:",
        gate[
            "routing_override_reason"
        ],
    )


    print(
        "Final verdict exact:",
        result[
            "final_verdict_exact"
        ],
    )


    print(
        "Structural detail exact:",
        result[
            "structural_detail_exact"
        ],
    )


    print(
        "End-to-end exact:",
        result[
            "end_to_end_exact"
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
        "=== DATALENS FROZEN DATASET DEPENDENCY PIPELINE v0.8 ==="
    )

    print()


    metadata = (
        frozen_runner_metadata()
    )


    print(
        "Runner:",
        metadata[
            "runner_version"
        ],
    )


    print(
        "Model:",
        metadata[
            "model"
        ],
    )


    print(
        "Prompt:",
        metadata[
            "prompt_version"
        ],
    )


    print(
        "Benchmark:",
        metadata[
            "benchmark_version"
        ],
    )


    print(
        "Temperature: 0"
    )


    print(
        "Thinking: disabled"
    )


    print()


    print(
        "FROZEN FIRST-RUN POLICY:"
    )


    print(
        "- benchmark expectations are never sent to Qwen"
    )


    print(
        "- completed cases are checkpointed and never rerun"
    )


    print(
        "- generation errors are final first-run results"
    )


    print(
        "- no prompt tuning is allowed from this benchmark"
    )


    print()


    # ========================================================
    # FINAL RESULT GUARD
    #
    # Prevent accidental second complete run.
    # ========================================================

    if (
        FINAL_OUTPUT_PATH.exists()
    ):

        raise RuntimeError(
            "Frozen v0.8 final result already exists. "
            "Refusing to rerun the frozen benchmark."
        )


    # ========================================================
    # LOAD FROZEN CASES
    # ========================================================

    cases = (
        load_dataset_dependency_frozen_benchmark(
            BENCHMARK_PATH
        )
    )


    assert (
        len(
            cases
        )
        == 12
    )


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


    print()


    # ========================================================
    # LOAD CHECKPOINT
    #
    # If the process was interrupted, completed cases remain
    # first-run results and are NOT inferred again.
    # ========================================================

    results_by_case_id = (
        load_checkpoint()
    )


    benchmark_ids = {
        case.case_id

        for case
        in cases
    }


    unknown_checkpoint_ids = (
        set(
            results_by_case_id
        )
        - benchmark_ids
    )


    if unknown_checkpoint_ids:

        raise ValueError(
            "Checkpoint contains unknown frozen case IDs: "
            f"{sorted(unknown_checkpoint_ids)}"
        )


    if results_by_case_id:

        print(
            "Checkpoint contains:",
            len(
                results_by_case_id
            ),
            "completed case(s)."
        )


        print(
            "They will NOT be rerun."
        )


        print()


    # ========================================================
    # FIRST-RUN INFERENCE
    # ========================================================

    for (
        index,
        case,
    ) in enumerate(
        cases,
        start=1,
    ):

        if (
            case.case_id
            in results_by_case_id
        ):

            print(
                f"[{index}/12] SKIP already checkpointed:",
                case.case_id,
            )

            continue


        print(
            f"[{index}/12] FIRST RUN:",
            case.case_id,
        )


        result = (
            run_frozen_dependency_case(
                case=case,
            )
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Save even generation errors.
        #
        # We do not retry bad first-run outputs.
        # ----------------------------------------------------

        results_by_case_id[
            case.case_id
        ] = result


        save_checkpoint(
            results_by_case_id
        )


        print_case(
            result
        )


        print()


    # ========================================================
    # ORDER RESULTS LIKE BENCHMARK
    # ========================================================

    results = [
        results_by_case_id[
            case.case_id
        ]

        for case
        in cases
    ]


    assert (
        len(
            results
        )
        == 12
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        summarize_frozen_results(
            results
        )
    )


    print()

    print(
        "=" * 112
    )


    print(
        "FROZEN DEPENDENCY PIPELINE SUMMARY v0.8"
    )


    print(
        "=" * 112
    )


    print(
        "Cases:",
        summary[
            "case_count"
        ],
    )


    print()

    print(
        "SEMANTIC EXTRACTION"
    )


    print(
        "Exact groups:",
        (
            f"{summary['semantic_exact_count']}"
            "/"
            f"{summary['case_count']}"
        ),
    )


    print(
        "Exact group accuracy:",
        round(
            summary[
                "semantic_exact_accuracy"
            ],
            3,
        ),
    )


    print(
        "Average semantic score:",
        round(
            summary[
                "semantic_average_overall"
            ],
            3,
        ),
    )


    print(
        "Dataset F1:",
        round(
            summary[
                "dataset_f1"
            ],
            3,
        ),
    )


    print(
        "Grouping F1:",
        round(
            summary[
                "grouping_f1"
            ],
            3,
        ),
    )


    print()

    print(
        "FINAL PIPELINE"
    )


    print(
        "Final verdict exact:",
        (
            f"{summary['final_verdict_exact_count']}"
            "/"
            f"{summary['case_count']}"
        ),
    )


    print(
        "Final verdict accuracy:",
        round(
            summary[
                "final_verdict_accuracy"
            ],
            3,
        ),
    )


    print(
        "Structural detail exact:",
        (
            f"{summary['structural_detail_exact_count']}"
            "/"
            f"{summary['case_count']}"
        ),
    )


    print(
        "Structural detail accuracy:",
        round(
            summary[
                "structural_detail_accuracy"
            ],
            3,
        ),
    )


    print(
        "End-to-end exact:",
        (
            f"{summary['end_to_end_exact_count']}"
            "/"
            f"{summary['case_count']}"
        ),
    )


    print(
        "End-to-end accuracy:",
        round(
            summary[
                "end_to_end_accuracy"
            ],
            3,
        ),
    )


    print()

    print(
        "RELIABILITY"
    )


    print(
        "Hallucinated datasets:",
        summary[
            "hallucinated_dataset_count"
        ],
    )


    print(
        "Missing datasets:",
        summary[
            "missing_dataset_count"
        ],
    )


    print(
        "Generation errors:",
        summary[
            "generation_error_count"
        ],
    )


    print(
        "Python gate errors:",
        summary[
            "gate_error_count"
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


    # ========================================================
    # CASE TABLE
    # ========================================================

    print()

    print(
        "=" * 112
    )


    print(
        "FROZEN CASE RESULTS"
    )


    print(
        "=" * 112
    )


    for result in results:

        print(
            f"{result['case_id']:<32}"
            f" semantic={'PASS' if result['semantic_exact'] else 'FAIL':<6}"
            f" final={'PASS' if result['final_verdict_exact'] else 'FAIL':<6}"
            f" e2e={'PASS' if result['end_to_end_exact'] else 'FAIL':<6}"
            f" status={result['status']}"
        )


    # ========================================================
    # SAVE FINAL
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    final_payload = {
        "evaluation":
            DATASET_DEPENDENCY_FROZEN_RUNNER_VERSION,

        "metadata":
            metadata,

        "benchmark":
            str(
                BENCHMARK_PATH
            ),

        "frozen":
            True,

        "first_run":
            True,

        "summary":
            summary,

        "results":
            results,
    }


    FINAL_OUTPUT_PATH.write_text(
        json.dumps(
            final_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print()

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )


    print(
        "Final result:",
        FINAL_OUTPUT_PATH,
    )


    print()

    print(
        "Frozen Dataset Dependency Pipeline v0.8: COMPLETE"
    )


if __name__ == "__main__":
    main()