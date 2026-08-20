from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from app.evals.analytical_planner_benchmark_v0_9 import (
    ANALYTICAL_PLANNER_BENCHMARK_VERSION,
    build_planner_input_for_case,
    load_analytical_planner_benchmark,
)

from app.evals.analytical_planner_contract_v0_9 import (
    AnalyticalPlannerCandidate,
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

ANALYTICAL_PLANNER_RESCORE_VERSION = (
    "analytical_planner_rescore_v0.9.1"
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
    / "analytical_planner_development_v0_9.jsonl"
)


SOURCE_RESULTS_PATH = (
    BASE_DIR
    / "evals"
    / "results"
    / "analytical_planner_v0_9"
    / "analytical_planner_validation_models_v0_9.json"
)


OUTPUT_PATH = (
    BASE_DIR
    / "evals"
    / "results"
    / "analytical_planner_v0_9_1"
    / "analytical_planner_validation_models_rescore_v0_9_1.json"
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


def _load_source_results() -> dict[
    str,
    Any,
]:

    if not (
        SOURCE_RESULTS_PATH.exists()
    ):

        raise FileNotFoundError(
            "Historical planner baseline result file "
            "was not found: "
            f"{SOURCE_RESULTS_PATH}"
        )


    return json.loads(
        SOURCE_RESULTS_PATH.read_text(
            encoding="utf-8",
        )
    )


# ============================================================
# RE-SCORE ONE RESULT
# ============================================================

def rescore_result(
    *,
    historical_result: dict[str, Any],
    case,
) -> dict[str, Any]:
    """
    Re-score one historical model output.

    IMPORTANT:

    No model call is performed.

    The original candidate generated during baseline v0.9 is
    parsed again and evaluated with:

        Validator v0.9.1
        Scorer v0.9.1
    """

    # ========================================================
    # HISTORICAL GENERATION FAILURE
    # ========================================================

    if (
        historical_result.get(
            "status"
        )
        != "ready"
    ):

        return {
            "case_id":
                historical_result[
                    "case_id"
                ],

            "domain":
                historical_result[
                    "domain"
                ],

            "model":
                historical_result[
                    "model"
                ],

            "status":
                historical_result[
                    "status"
                ],

            "candidate":
                historical_result.get(
                    "candidate"
                ),

            "historical_v0_9": {
                "validation":
                    historical_result.get(
                        "validation"
                    ),

                "score":
                    historical_result.get(
                        "score"
                    ),

                "exact":
                    historical_result.get(
                        "exact",
                        False,
                    ),
            },

            "rescore_v0_9_1": {
                "validation":
                    None,

                "score":
                    None,

                "exact":
                    False,
            },

            "inference_ms":
                historical_result.get(
                    "inference_ms",
                    0.0,
                ),

            "error":
                historical_result.get(
                    "error"
                ),
        }


    candidate_payload = (
        historical_result.get(
            "candidate"
        )
    )


    if candidate_payload is None:

        raise ValueError(
            "Historical ready result contains no candidate: "
            f"{historical_result['case_id']}"
        )


    # ========================================================
    # REBUILD EXACT CANDIDATE
    # ========================================================

    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            candidate_payload
        )
    )


    # ========================================================
    # REBUILD EXACT TRUSTED PLANNER INPUT
    # ========================================================

    planner_input = (
        build_planner_input_for_case(
            case
        )
    )


    # ========================================================
    # NEW VALIDATOR
    # ========================================================

    validation = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    # ========================================================
    # NEW SCORER
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
            historical_result[
                "case_id"
            ],

        "domain":
            historical_result[
                "domain"
            ],

        "model":
            historical_result[
                "model"
            ],

        "status":
            "ready",

        "candidate":
            candidate.model_dump(
                mode="json",
            ),

        # ====================================================
        # HISTORICAL RESULT — NEVER OVERWRITTEN
        # ====================================================

        "historical_v0_9": {
            "validation":
                historical_result.get(
                    "validation"
                ),

            "score":
                historical_result.get(
                    "score"
                ),

            "exact":
                historical_result.get(
                    "exact",
                    False,
                ),
        },

        # ====================================================
        # NEW RE-SCORE
        # ====================================================

        "rescore_v0_9_1": {
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
        },

        # ====================================================
        # ORIGINAL INFERENCE LATENCY
        #
        # No inference occurred during re-scoring.
        # ====================================================

        "inference_ms":
            historical_result.get(
                "inference_ms",
                0.0,
            ),

        "error":
            None,
    }


# ============================================================
# SUMMARY
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
            f"No results available for model: {model}"
        )


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


    generation_errors = (
        total
        - len(
            ready_results
        )
    )


    # ========================================================
    # HISTORICAL v0.9
    # ========================================================

    historical_scores = [
        float(
            result[
                "historical_v0_9"
            ][
                "score"
            ][
                "overall"
            ]
        )

        for result
        in ready_results

        if (
            result[
                "historical_v0_9"
            ][
                "score"
            ]
            is not None
        )
    ]


    historical_valid_count = sum(
        1

        for result
        in ready_results

        if (
            result[
                "historical_v0_9"
            ][
                "validation"
            ]
            is not None

            and result[
                "historical_v0_9"
            ][
                "validation"
            ][
                "valid"
            ]
        )
    )


    historical_exact_count = sum(
        1

        for result
        in ready_results

        if (
            result[
                "historical_v0_9"
            ][
                "exact"
            ]
        )
    )


    # ========================================================
    # v0.9.1
    # ========================================================

    new_scores = [
        float(
            result[
                "rescore_v0_9_1"
            ][
                "score"
            ][
                "overall"
            ]
        )

        for result
        in ready_results

        if (
            result[
                "rescore_v0_9_1"
            ][
                "score"
            ]
            is not None
        )
    ]


    valid_count = sum(
        1

        for result
        in ready_results

        if (
            result[
                "rescore_v0_9_1"
            ][
                "validation"
            ][
                "valid"
            ]
        )
    )


    exact_count = sum(
        1

        for result
        in ready_results

        if (
            result[
                "rescore_v0_9_1"
            ][
                "exact"
            ]
        )
    )


    # ========================================================
    # METRICS
    # ========================================================

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


    metrics: dict[
        str,
        float,
    ] = {}


    for metric_name in metric_names:

        values = [
            float(
                result[
                    "rescore_v0_9_1"
                ][
                    "score"
                ][
                    "metrics"
                ][
                    metric_name
                ]
            )

            for result
            in ready_results

            if (
                result[
                    "rescore_v0_9_1"
                ][
                    "score"
                ]
                is not None
            )
        ]


        metrics[
            metric_name
        ] = (
            _average(
                values
            )
        )


    # ========================================================
    # ISSUE CODES
    # ========================================================

    issue_codes: list[
        str
    ] = []


    for result in ready_results:

        validation = (
            result[
                "rescore_v0_9_1"
            ][
                "validation"
            ]
        )


        if validation is None:
            continue


        for issue in (
            validation[
                "issues"
            ]
        ):

            issue_codes.append(
                issue[
                    "code"
                ]
            )


    # ========================================================
    # EXTRA STEPS
    # ========================================================

    total_extra_steps = sum(
        int(
            result[
                "rescore_v0_9_1"
            ][
                "score"
            ][
                "diagnostics"
            ][
                "extra_step_count"
            ]
        )

        for result
        in ready_results

        if (
            result[
                "rescore_v0_9_1"
            ][
                "score"
            ]
            is not None
        )
    )


    historical_average = (
        _average(
            historical_scores
        )
    )


    new_average = (
        _average(
            new_scores
        )
    )


    return {
        "model":
            model,

        "case_count":
            total,

        "generation_error_count":
            generation_errors,

        "historical_v0_9": {
            "average_overall":
                historical_average,

            "valid_plan_count":
                historical_valid_count,

            "validator_acceptance_rate":
                (
                    historical_valid_count
                    / total
                ),

            "exact_plan_count":
                historical_exact_count,

            "exact_plan_accuracy":
                (
                    historical_exact_count
                    / total
                ),
        },

        "rescore_v0_9_1": {
            "average_overall":
                new_average,

            "score_delta":
                (
                    new_average
                    - historical_average
                ),

            "valid_plan_count":
                valid_count,

            "validator_acceptance_rate":
                (
                    valid_count
                    / total
                ),

            "exact_plan_count":
                exact_count,

            "exact_plan_accuracy":
                (
                    exact_count
                    / total
                ),

            **metrics,

            "validator_issue_codes":
                issue_codes,

            "extra_step_count":
                total_extra_steps,
        },
    }


# ============================================================
# DISPLAY — CASE
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


    if (
        result[
            "status"
        ]
        != "ready"
    ):

        print(
            "Historical generation status:",
            result[
                "status"
            ],
        )


        return


    old = (
        result[
            "historical_v0_9"
        ]
    )


    new = (
        result[
            "rescore_v0_9_1"
        ]
    )


    old_score = float(
        old[
            "score"
        ][
            "overall"
        ]
    )


    new_score = float(
        new[
            "score"
        ][
            "overall"
        ]
    )


    print(
        "Historical v0.9:",
        f"{old_score:.3f}",
        "| validator=",
        old[
            "validation"
        ][
            "valid"
        ],
        "| exact=",
        old[
            "exact"
        ],
    )


    print(
        "Re-score v0.9.1:",
        f"{new_score:.3f}",
        "| validator=",
        new[
            "validation"
        ][
            "valid"
        ],
        "| exact=",
        new[
            "exact"
        ],
    )


    print(
        "Delta:",
        f"{new_score - old_score:+.3f}",
    )


    metrics = (
        new[
            "score"
        ][
            "metrics"
        ]
    )


    print(
        "Sequence:",
        f"{float(metrics['tool_sequence_score']):.3f}",
        "| Arguments:",
        f"{float(metrics['tool_argument_score']):.3f}",
        "| Parsimony:",
        f"{float(metrics['parsimony_score']):.3f}",
    )


    issues = [
        issue[
            "code"
        ]

        for issue
        in new[
            "validation"
        ][
            "issues"
        ]
    ]


    if issues:

        print(
            "v0.9.1 validator issues:",
            issues,
        )


# ============================================================
# DISPLAY — SUMMARY
# ============================================================

def print_summary(
    summary: dict[str, Any],
) -> None:

    old = (
        summary[
            "historical_v0_9"
        ]
    )


    new = (
        summary[
            "rescore_v0_9_1"
        ]
    )


    print()

    print(
        "=" * 110
    )


    print(
        "MODEL:",
        summary[
            "model"
        ],
    )


    print(
        "=" * 110
    )


    print(
        "Historical v0.9 overall:",
        f"{old['average_overall']:.3f}",
    )


    print(
        "Re-score v0.9.1 overall:",
        f"{new['average_overall']:.3f}",
    )


    print(
        "Delta:",
        f"{new['score_delta']:+.3f}",
    )


    print()


    print(
        "Historical validator:",
        (
            f"{old['valid_plan_count']}"
            "/"
            f"{summary['case_count']}"
        ),
    )


    print(
        "v0.9.1 validator:",
        (
            f"{new['valid_plan_count']}"
            "/"
            f"{summary['case_count']}"
        ),
    )


    print()


    print(
        "Historical exact:",
        (
            f"{old['exact_plan_count']}"
            "/"
            f"{summary['case_count']}"
        ),
    )


    print(
        "v0.9.1 exact:",
        (
            f"{new['exact_plan_count']}"
            "/"
            f"{summary['case_count']}"
        ),
    )


    print()


    print(
        "Requirement coverage:",
        f"{new['requirement_coverage_f1']:.3f}",
    )


    print(
        "Intent:",
        f"{new['intent_accuracy']:.3f}",
    )


    print(
        "Family:",
        f"{new['family_accuracy']:.3f}",
    )


    print(
        "Target grain:",
        f"{new['target_grain_accuracy']:.3f}",
    )


    print(
        "Tool sequence:",
        f"{new['tool_sequence_score']:.3f}",
    )


    print(
        "Tool arguments:",
        f"{new['tool_argument_score']:.3f}",
    )


    print(
        "Validator score:",
        f"{new['validator_acceptance']:.3f}",
    )


    print(
        "Parsimony:",
        f"{new['parsimony_score']:.3f}",
    )


    print()


    print(
        "Validator issues:",
        new[
            "validator_issue_codes"
        ],
    )


    print(
        "Extra steps:",
        new[
            "extra_step_count"
        ],
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL PLANNER RE-SCORE v0.9.1 ==="
    )


    print()


    print(
        "Rescore:",
        ANALYTICAL_PLANNER_RESCORE_VERSION,
    )


    print(
        "Benchmark:",
        ANALYTICAL_PLANNER_BENCHMARK_VERSION,
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
        "NO MODEL INFERENCE WILL BE PERFORMED."
    )


    print(
        "Historical candidates are reused exactly as saved."
    )


    print()


    # ========================================================
    # LOAD BENCHMARK
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


    cases_by_id = {
        case.case_id:
            case

        for case
        in cases
    }


    # ========================================================
    # LOAD HISTORICAL RESULTS
    # ========================================================

    historical_payload = (
        _load_source_results()
    )


    historical_results = (
        historical_payload[
            "results"
        ]
    )


    models = list(
        historical_results.keys()
    )


    print(
        "Models:",
        models,
    )


    print(
        "Validation cases:",
        len(
            cases
        ),
    )


    print()


    all_rescored_results: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = {}


    summaries: list[
        dict[str, Any]
    ] = []


    # ========================================================
    # RE-SCORE
    # ========================================================

    for model in models:

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


        historical_model_results = (
            historical_results[
                model
            ]
        )


        assert (
            len(
                historical_model_results
            )
            == len(
                cases
            )
        )


        for historical_result in (
            historical_model_results
        ):

            case_id = (
                historical_result[
                    "case_id"
                ]
            )


            case = (
                cases_by_id.get(
                    case_id
                )
            )


            if case is None:

                raise ValueError(
                    "Historical result references an "
                    "unknown validation case: "
                    f"{case_id}"
                )


            result = (
                rescore_result(
                    historical_result=(
                        historical_result
                    ),

                    case=(
                        case
                    ),
                )
            )


            model_results.append(
                result
            )


            print_case_result(
                result
            )


        all_rescored_results[
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


        print_summary(
            summary
        )


    # ========================================================
    # RANKING
    # ========================================================

    ranked = sorted(
        summaries,

        key=lambda summary: (
            summary[
                "rescore_v0_9_1"
            ][
                "average_overall"
            ],

            summary[
                "rescore_v0_9_1"
            ][
                "validator_acceptance_rate"
            ],

            summary[
                "rescore_v0_9_1"
            ][
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
        "RE-SCORED MODEL COMPARISON"
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

        old = (
            summary[
                "historical_v0_9"
            ]
        )


        new = (
            summary[
                "rescore_v0_9_1"
            ]
        )


        print(
            f"{rank}. "
            f"{summary['model']:<22}"
            f" old={old['average_overall']:.3f}"
            f" new={new['average_overall']:.3f}"
            f" delta={new['score_delta']:+.3f}"
            f" validator="
            f"{new['validator_acceptance_rate']:.3f}"
            f" exact="
            f"{new['exact_plan_accuracy']:.3f}"
            f" parsimony="
            f"{new['parsimony_score']:.3f}"
        )


    # ========================================================
    # SAVE NEW RESULT
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    payload = {
        "rescore_version":
            ANALYTICAL_PLANNER_RESCORE_VERSION,

        "source_baseline": {
            "path":
                str(
                    SOURCE_RESULTS_PATH
                ),

            "historical":
                True,

            "model_inference_repeated":
                False,
        },

        "benchmark": {
            "version":
                ANALYTICAL_PLANNER_BENCHMARK_VERSION,

            "path":
                str(
                    BENCHMARK_PATH
                ),

            "split":
                "validation",

            "frozen":
                False,
        },

        "evaluation": {
            "validator_version":
                ANALYTICAL_PLANNER_VALIDATOR_VERSION,

            "scorer_version":
                ANALYTICAL_PLANNER_SCORER_VERSION,
        },

        "case_count":
            len(
                cases
            ),

        "models":
            models,

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
            all_rescored_results,
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
        "Historical baseline preserved:",
        SOURCE_RESULTS_PATH,
    )


    print()


    print(
        "Analytical Planner re-score v0.9.1: COMPLETE"
    )


if __name__ == "__main__":
    main()