from __future__ import annotations

import hashlib
import json

from pathlib import Path
from time import perf_counter
from typing import Any

from app.ai.provider import (
    client,
)

from app.evals.analytical_planner_contract_v0_9 import (
    ANALYTICAL_PLANNER_CONTRACT_VERSION,
    AnalyticalPlannerCandidate,
)

from app.evals.analytical_planner_frozen_benchmark_v1_0 import (
    ANALYTICAL_PLANNER_FROZEN_BENCHMARK_VERSION,
    FrozenAnalyticalPlannerEvalCase,
    build_planner_input_for_frozen_case,
    load_frozen_analytical_planner_benchmark,
)

from app.evals.analytical_planner_model_runner_v0_9 import (
    ANALYTICAL_PLANNER_PROMPT_VERSION,
    SYSTEM_PROMPT_V0_9,
)

from app.evals.analytical_planner_scorer_v0_9_1 import (
    ANALYTICAL_PLANNER_SCORER_VERSION,
    score_analytical_planner_candidate,
)

from app.evals.analytical_planner_validator_v0_9_1 import (
    ANALYTICAL_PLANNER_VALIDATOR_VERSION,
    validate_analytical_planner_candidate,
)


# ============================================================
# VERSION
# ============================================================

ANALYTICAL_PLANNER_FROZEN_RUNNER_VERSION = (
    "analytical_planner_frozen_runner_v1.0"
)


# ============================================================
# LOCKED CONFIGURATION
# ============================================================

FROZEN_MODEL = (
    "qwen3:4b-instruct"
)


TEMPERATURE = 0


THINKING_ENABLED = False


# ============================================================
# HISTORICAL BENCHMARK HASH
#
# This is intentionally hard-coded.
#
# Merely modifying both the JSONL and its .sha256 file must
# NOT be enough to silently redefine the frozen benchmark.
# ============================================================

EXPECTED_BENCHMARK_SHA256 = (
    "2c23fd35ae514fb8a0633fbaba1e233e"
    "9979eae2b093fa9ee5e5488b08f00c1a"
)


# ============================================================
# PATHS
# ============================================================

API_DIR = (
    Path(
        __file__,
    )
    .resolve()
    .parents[
        2
    ]
)


EVALS_DIR = (
    API_DIR
    / "evals"
)


BENCHMARK_PATH = (
    EVALS_DIR
    / "analytical_planner_frozen_v1_0.jsonl"
)


BENCHMARK_HASH_PATH = (
    EVALS_DIR
    / "analytical_planner_frozen_v1_0.sha256"
)


RESULTS_DIR = (
    EVALS_DIR
    / "results"
    / "analytical_planner_frozen_v1_0"
)


CHECKPOINT_PATH = (
    RESULTS_DIR
    / (
        "qwen3_4b_instruct_"
        "frozen_planner_v1_0_checkpoint.json"
    )
)


FINAL_PATH = (
    RESULTS_DIR
    / (
        "qwen3_4b_instruct_"
        "frozen_planner_v1_0.json"
    )
)


# ============================================================
# HASH
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    return (
        hashlib
        .sha256(
            path.read_bytes()
        )
        .hexdigest()
    )


def verify_frozen_benchmark_lock() -> str:
    """
    Verify three independent values:

    1. actual JSONL hash
    2. .sha256 lock file
    3. hard-coded historical hash

    All three must match.
    """

    if not (
        BENCHMARK_PATH.exists()
    ):

        raise FileNotFoundError(
            "Frozen planner benchmark not found: "
            f"{BENCHMARK_PATH}"
        )


    if not (
        BENCHMARK_HASH_PATH.exists()
    ):

        raise FileNotFoundError(
            "Frozen planner SHA-256 lock not found: "
            f"{BENCHMARK_HASH_PATH}"
        )


    actual_hash = (
        sha256_file(
            BENCHMARK_PATH
        )
    )


    lock_hash = (
        BENCHMARK_HASH_PATH
        .read_text(
            encoding="ascii",
        )
        .strip()
    )


    if (
        actual_hash
        != lock_hash
    ):

        raise ValueError(
            "Frozen benchmark hash does not match "
            "its SHA-256 lock file."
        )


    if (
        actual_hash
        != EXPECTED_BENCHMARK_SHA256
    ):

        raise ValueError(
            "Frozen benchmark does not match the "
            "historical SHA-256 embedded in the "
            "first-run runner.\n"
            f"Expected: {EXPECTED_BENCHMARK_SHA256}\n"
            f"Actual:   {actual_hash}"
        )


    return actual_hash


# ============================================================
# MODEL-VISIBLE PROMPT
# ============================================================

def build_frozen_planner_user_prompt(
    case: FrozenAnalyticalPlannerEvalCase,
) -> str:
    """
    Build the exact model-visible input.

    IMPORTANT:

    The following frozen benchmark fields are never exposed:

        expected
        notes
        frozen

    Only the trusted AnalyticalPlannerInput reaches the model.
    """

    planner_input = (
        build_planner_input_for_frozen_case(
            case
        )
    )


    payload = (
        planner_input.model_dump(
            mode="json",
        )
    )


    return (
        "ANALYTICAL PLANNER INPUT:\n\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n\n"
        + (
            "Construis uniquement le plan analytique "
            "nécessaire pour chaque requirement."
        )
    )


# ============================================================
# METADATA
# ============================================================

def frozen_runner_metadata(
    *,
    benchmark_hash: str,
) -> dict[str, Any]:

    return {
        "runner_version":
            ANALYTICAL_PLANNER_FROZEN_RUNNER_VERSION,

        "benchmark_version":
            ANALYTICAL_PLANNER_FROZEN_BENCHMARK_VERSION,

        "benchmark_sha256":
            benchmark_hash,

        "model":
            FROZEN_MODEL,

        "prompt_version":
            ANALYTICAL_PLANNER_PROMPT_VERSION,

        "contract_version":
            ANALYTICAL_PLANNER_CONTRACT_VERSION,

        "validator_version":
            ANALYTICAL_PLANNER_VALIDATOR_VERSION,

        "scorer_version":
            ANALYTICAL_PLANNER_SCORER_VERSION,

        "temperature":
            TEMPERATURE,

        "thinking":
            THINKING_ENABLED,

        "split":
            "test",

        "frozen":
            True,

        "historical_first_run":
            True,
    }


# ============================================================
# SINGLE FROZEN INFERENCE
# ============================================================

def run_frozen_case(
    *,
    case: FrozenAnalyticalPlannerEvalCase,
) -> dict[str, Any]:
    """
    Execute exactly one model attempt for one frozen case.

    A generation error is itself part of the historical
    first-run result and is NOT automatically retried.
    """

    planner_input = (
        build_planner_input_for_frozen_case(
            case
        )
    )


    raw_content: (
        str
        | None
    ) = None


    started_at = (
        perf_counter()
    )


    try:

        response = (
            client.chat(
                model=(
                    FROZEN_MODEL
                ),

                messages=[
                    {
                        "role":
                            "system",

                        "content":
                            SYSTEM_PROMPT_V0_9,
                    },

                    {
                        "role":
                            "user",

                        "content":
                            build_frozen_planner_user_prompt(
                                case
                            ),
                    },
                ],

                format=(
                    AnalyticalPlannerCandidate
                    .model_json_schema()
                ),

                options={
                    "temperature":
                        TEMPERATURE,
                },

                think=(
                    THINKING_ENABLED
                ),
            )
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


        candidate = (
            AnalyticalPlannerCandidate
            .model_validate_json(
                raw_content
            )
        )


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

            "status":
                "generation_error",

            "candidate":
                None,

            "validation":
                None,

            "score":
                None,

            "exact":
                False,

            "inference_ms":
                inference_ms,

            "raw_content":
                raw_content,

            "error":
                (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
        }


    # ========================================================
    # DETERMINISTIC VALIDATION
    # ========================================================

    validation = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    # ========================================================
    # FROZEN SCORING
    # ========================================================

    score = (
        score_analytical_planner_candidate(
            candidate=candidate,

            expected=(
                case.expected
            ),

            planner_input=(
                planner_input
            ),
        )
    )


    exact = (
        score.overall
        == 1.0
    )


    return {
        "case_id":
            case.case_id,

        "domain":
            case.domain,

        "status":
            "ready",

        "candidate":
            candidate.model_dump(
                mode="json",
            ),

        "validation": {
            "valid":
                validation.valid,

            "validated_requirement_ids":
                validation.validated_requirement_ids,

            "issues": [
                issue.model_dump(
                    mode="json",
                )

                for issue
                in validation.issues
            ],
        },

        "score":
            score.as_dict(),

        "exact":
            exact,

        "inference_ms":
            inference_ms,

        "raw_content":
            raw_content,

        "error":
            None,
    }


# ============================================================
# CHECKPOINT
# ============================================================

def _checkpoint_payload(
    *,
    metadata: dict[str, Any],
    results: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:

    return {
        "metadata":
            metadata,

        "completed_case_ids": [
            result[
                "case_id"
            ]

            for result
            in results
        ],

        "results":
            results,
    }


def save_checkpoint(
    *,
    metadata: dict[str, Any],
    results: list[
        dict[str, Any]
    ],
) -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    payload = (
        _checkpoint_payload(
            metadata=metadata,
            results=results,
        )
    )


    temporary_path = (
        CHECKPOINT_PATH
        .with_suffix(
            ".tmp"
        )
    )


    temporary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    temporary_path.replace(
        CHECKPOINT_PATH
    )


def load_checkpoint(
    *,
    expected_metadata: dict[str, Any],
    valid_case_ids: set[str],
) -> list[
    dict[str, Any]
]:

    if not (
        CHECKPOINT_PATH.exists()
    ):

        return []


    payload = json.loads(
        CHECKPOINT_PATH.read_text(
            encoding="utf-8",
        )
    )


    checkpoint_metadata = (
        payload.get(
            "metadata"
        )
    )


    if (
        checkpoint_metadata
        != expected_metadata
    ):

        raise ValueError(
            "Frozen checkpoint metadata does not "
            "match the locked runner configuration."
        )


    results = (
        payload.get(
            "results"
        )
    )


    if not isinstance(
        results,
        list,
    ):

        raise ValueError(
            "Invalid frozen checkpoint results."
        )


    seen: set[
        str
    ] = set()


    for result in results:

        case_id = (
            result.get(
                "case_id"
            )
        )


        if (
            case_id
            not in valid_case_ids
        ):

            raise ValueError(
                "Frozen checkpoint contains an "
                "unknown case_id: "
                f"{case_id}"
            )


        if (
            case_id
            in seen
        ):

            raise ValueError(
                "Frozen checkpoint contains a "
                "duplicate case result: "
                f"{case_id}"
            )


        seen.add(
            case_id
        )


    return results


# ============================================================
# SUMMARY
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


def summarize_frozen_results(
    results: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:

    total = len(
        results
    )


    if total == 0:

        raise ValueError(
            "Cannot summarize empty frozen results."
        )


    metric_names = [
        "requirement_coverage_f1",
        "intent_accuracy",
        "family_accuracy",
        "target_grain_accuracy",
        "tool_sequence_score",
        "tool_argument_score",
        "validator_acceptance",
        "parsimony_score",
    ]


    overall_values: list[
        float
    ] = []


    metric_values: dict[
        str,
        list[float],
    ] = {
        metric_name:
            []

        for metric_name
        in metric_names
    }


    valid_count = 0

    exact_count = 0

    generation_error_count = 0

    issue_codes: list[
        str
    ] = []

    total_extra_steps = 0


    for result in results:

        score = (
            result.get(
                "score"
            )
        )


        # ====================================================
        # Generation failures count as zero.
        # ====================================================

        if score is None:

            generation_error_count += 1


            overall_values.append(
                0.0
            )


            for metric_name in (
                metric_names
            ):

                metric_values[
                    metric_name
                ].append(
                    0.0
                )


            continue


        overall_values.append(
            float(
                score[
                    "overall"
                ]
            )
        )


        for metric_name in (
            metric_names
        ):

            metric_values[
                metric_name
            ].append(
                float(
                    score[
                        "metrics"
                    ][
                        metric_name
                    ]
                )
            )


        if (
            result[
                "validation"
            ][
                "valid"
            ]
        ):

            valid_count += 1


        if (
            result[
                "exact"
            ]
        ):

            exact_count += 1


        total_extra_steps += int(
            score[
                "diagnostics"
            ][
                "extra_step_count"
            ]
        )


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


    summary = {
        "case_count":
            total,

        "average_overall":
            _average(
                overall_values
            ),

        "exact_plan_count":
            exact_count,

        "exact_plan_accuracy":
            (
                exact_count
                / total
            ),

        "valid_plan_count":
            valid_count,

        "validator_acceptance_rate":
            (
                valid_count
                / total
            ),

        "generation_error_count":
            generation_error_count,

        "validator_issue_codes":
            issue_codes,

        "extra_step_count":
            total_extra_steps,

        "average_inference_ms":
            _average(
                [
                    float(
                        result.get(
                            "inference_ms",
                            0.0,
                        )
                    )

                    for result
                    in results
                ]
            ),
    }


    for metric_name in (
        metric_names
    ):

        summary[
            metric_name
        ] = (
            _average(
                metric_values[
                    metric_name
                ]
            )
        )


    return summary


# ============================================================
# PREFLIGHT
# ============================================================

def preflight_frozen_runner(
    *,
    require_unused_final: bool = True,
) -> dict[str, Any]:
    """
    Validate everything required for a historical first run.

    No model call occurs here.
    """

    benchmark_hash = (
        verify_frozen_benchmark_lock()
    )


    cases = (
        load_frozen_analytical_planner_benchmark(
            BENCHMARK_PATH
        )
    )


    if (
        len(
            cases
        )
        != 12
    ):

        raise ValueError(
            "Frozen planner benchmark must contain "
            "exactly 12 cases."
        )


    case_ids = {
        case.case_id

        for case
        in cases
    }


    if (
        len(
            case_ids
        )
        != 12
    ):

        raise ValueError(
            "Frozen planner benchmark contains "
            "duplicate case IDs."
        )


    # ========================================================
    # Verify all model-visible inputs can still be built.
    # ========================================================

    for case in cases:

        planner_input = (
            build_planner_input_for_frozen_case(
                case
            )
        )


        for requirement in (
            planner_input.requirements
        ):

            if (
                "join_datasets"
                in (
                    requirement
                    .allowed_analytical_tools
                )
            ):

                raise ValueError(
                    "Structural join tool leaked into "
                    "frozen planner input."
                )


    metadata = (
        frozen_runner_metadata(
            benchmark_hash=(
                benchmark_hash
            )
        )
    )


    if (
        require_unused_final
        and FINAL_PATH.exists()
    ):

        raise FileExistsError(
            "Historical frozen final result already "
            "exists. Refusing another first run:\n"
            f"{FINAL_PATH}"
        )


    checkpoint_results = (
        load_checkpoint(
            expected_metadata=(
                metadata
            ),

            valid_case_ids=(
                case_ids
            ),
        )
    )


    return {
        "benchmark_hash":
            benchmark_hash,

        "case_count":
            len(
                cases
            ),

        "model":
            FROZEN_MODEL,

        "prompt_version":
            ANALYTICAL_PLANNER_PROMPT_VERSION,

        "validator_version":
            ANALYTICAL_PLANNER_VALIDATOR_VERSION,

        "scorer_version":
            ANALYTICAL_PLANNER_SCORER_VERSION,

        "checkpoint_exists":
            CHECKPOINT_PATH.exists(),

        "checkpoint_completed_cases":
            len(
                checkpoint_results
            ),

        "final_exists":
            FINAL_PATH.exists(),
    }


# ============================================================
# DISPLAY
# ============================================================

def print_case_result(
    result: dict[str, Any],
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


    for plan in (
        result[
            "candidate"
        ][
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


    print(
        "Python validator:",
        (
            "PASS"
            if (
                result[
                    "validation"
                ][
                    "valid"
                ]
            )
            else "FAIL"
        ),
    )


    issues = [
        issue[
            "code"
        ]

        for issue
        in result[
            "validation"
        ][
            "issues"
        ]
    ]


    if issues:

        print(
            "Validator issues:",
            issues,
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
        f"{float(score['overall']):.3f}",
    )


    print(
        "Intent:",
        f"{float(metrics['intent_accuracy']):.3f}",
        "| Family:",
        f"{float(metrics['family_accuracy']):.3f}",
        "| Grain:",
        f"{float(metrics['target_grain_accuracy']):.3f}",
    )


    print(
        "Sequence:",
        f"{float(metrics['tool_sequence_score']):.3f}",
        "| Arguments:",
        f"{float(metrics['tool_argument_score']):.3f}",
        "| Parsimony:",
        f"{float(metrics['parsimony_score']):.3f}",
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
# FIRST HISTORICAL RUN
# ============================================================

def run_frozen_first_pass() -> dict[str, Any]:

    # ========================================================
    # PREFLIGHT — NO INFERENCE BEFORE THIS PASSES
    # ========================================================

    preflight = (
        preflight_frozen_runner(
            require_unused_final=True,
        )
    )


    benchmark_hash = (
        preflight[
            "benchmark_hash"
        ]
    )


    cases = (
        load_frozen_analytical_planner_benchmark(
            BENCHMARK_PATH
        )
    )


    cases_by_id = {
        case.case_id:
            case

        for case
        in cases
    }


    metadata = (
        frozen_runner_metadata(
            benchmark_hash=(
                benchmark_hash
            )
        )
    )


    # ========================================================
    # RESUME CHECKPOINT WITHOUT REPEATING COMPLETED CASES
    # ========================================================

    results = (
        load_checkpoint(
            expected_metadata=(
                metadata
            ),

            valid_case_ids=(
                set(
                    cases_by_id
                )
            ),
        )
    )


    completed_case_ids = {
        result[
            "case_id"
        ]

        for result
        in results
    }


    if results:

        print(
            "Existing historical checkpoint detected."
        )


        print(
            "Completed cases:",
            len(
                results
            ),
        )


        print(
            "Completed cases will NOT be inferred again."
        )


        print()


    # ========================================================
    # FIRST ATTEMPT FOR EACH REMAINING CASE
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
            in completed_case_ids
        ):

            print(
                f"[{index}/{len(cases)}] SKIP",
                case.case_id,
                "(already checkpointed)",
            )


            continue


        print(
            f"[{index}/{len(cases)}]",
            FROZEN_MODEL,
            "->",
            case.case_id,
        )


        result = (
            run_frozen_case(
                case=case
            )
        )


        results.append(
            result
        )


        completed_case_ids.add(
            case.case_id
        )


        # ====================================================
        # CHECKPOINT IMMEDIATELY
        # ====================================================

        save_checkpoint(
            metadata=metadata,
            results=results,
        )


        print_case_result(
            result
        )


        print(
            "Checkpoint:",
            (
                f"{len(results)}"
                "/"
                f"{len(cases)}"
            ),
        )


        print()


    # ========================================================
    # COMPLETENESS
    # ========================================================

    if (
        len(
            results
        )
        != len(
            cases
        )
    ):

        raise RuntimeError(
            "Frozen run ended without all cases."
        )


    result_ids = [
        result[
            "case_id"
        ]

        for result
        in results
    ]


    if (
        len(
            result_ids
        )
        != len(
            set(
                result_ids
            )
        )
    ):

        raise RuntimeError(
            "Frozen first-run results contain "
            "duplicate case IDs."
        )


    if (
        set(
            result_ids
        )
        != set(
            cases_by_id
        )
    ):

        raise RuntimeError(
            "Frozen first-run case coverage mismatch."
        )


    # ========================================================
    # PRESERVE BENCHMARK ORDER IN FINAL ARTIFACT
    # ========================================================

    results_by_id = {
        result[
            "case_id"
        ]:
            result

        for result
        in results
    }


    ordered_results = [
        results_by_id[
            case.case_id
        ]

        for case
        in cases
    ]


    summary = (
        summarize_frozen_results(
            ordered_results
        )
    )


    # ========================================================
    # FINAL HISTORICAL ARTIFACT
    # ========================================================

    final_payload = {
        "historical_first_run":
            True,

        "model_inference_repeated":
            False,

        "metadata":
            metadata,

        "benchmark": {
            "path":
                str(
                    BENCHMARK_PATH
                ),

            "sha256":
                benchmark_hash,

            "split":
                "test",

            "frozen":
                True,
        },

        "summary":
            summary,

        "results":
            ordered_results,

        "checkpoint_path":
            str(
                CHECKPOINT_PATH
            ),
    }


    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # EXCLUSIVE CREATION
    #
    # "x" refuses to overwrite an existing historical result.
    # ========================================================

    with FINAL_PATH.open(
        "x",
        encoding="utf-8",
    ) as handle:

        json.dump(
            final_payload,
            handle,
            ensure_ascii=False,
            indent=2,
        )


        handle.write(
            "\n"
        )


    return final_payload


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL PLANNER FROZEN FIRST RUN v1.0 ==="
    )


    print()


    print(
        "Runner:",
        ANALYTICAL_PLANNER_FROZEN_RUNNER_VERSION,
    )


    print(
        "Model:",
        FROZEN_MODEL,
    )


    print(
        "Prompt:",
        ANALYTICAL_PLANNER_PROMPT_VERSION,
    )


    print(
        "Validator:",
        ANALYTICAL_PLANNER_VALIDATOR_VERSION,
    )


    print(
        "Scorer:",
        ANALYTICAL_PLANNER_SCORER_VERSION,
    )


    print()


    print(
        "Historical benchmark SHA-256:",
        EXPECTED_BENCHMARK_SHA256,
    )


    print()


    payload = (
        run_frozen_first_pass()
    )


    summary = (
        payload[
            "summary"
        ]
    )


    print()

    print(
        "=" * 110
    )


    print(
        "FROZEN FIRST-RUN SUMMARY"
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
        f"{summary['average_overall']:.3f}",
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


    print(
        "Generation errors:",
        summary[
            "generation_error_count"
        ],
    )


    print()


    print(
        "Requirement coverage:",
        f"{summary['requirement_coverage_f1']:.3f}",
    )


    print(
        "Intent:",
        f"{summary['intent_accuracy']:.3f}",
    )


    print(
        "Family:",
        f"{summary['family_accuracy']:.3f}",
    )


    print(
        "Target grain:",
        f"{summary['target_grain_accuracy']:.3f}",
    )


    print(
        "Tool sequence:",
        f"{summary['tool_sequence_score']:.3f}",
    )


    print(
        "Tool arguments:",
        f"{summary['tool_argument_score']:.3f}",
    )


    print(
        "Validator score:",
        f"{summary['validator_acceptance']:.3f}",
    )


    print(
        "Parsimony:",
        f"{summary['parsimony_score']:.3f}",
    )


    print()


    print(
        "Validator issues:",
        summary[
            "validator_issue_codes"
        ],
    )


    print(
        "Extra steps:",
        summary[
            "extra_step_count"
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


    print()


    print(
        "Checkpoint preserved:",
        CHECKPOINT_PATH,
    )


    print(
        "Historical final result:",
        FINAL_PATH,
    )


    print()


    print(
        "IMPORTANT:"
    )


    print(
        "This first-run result is now historical."
    )


    print(
        "Do not overwrite, regenerate, or tune against it."
    )


    print()


    print(
        "Analytical Planner Frozen First Run v1.0: COMPLETE"
    )


if __name__ == "__main__":
    main()