from __future__ import annotations

import json

from pathlib import Path

from app.evals.benchmark_loader import (
    load_benchmark,
)

from app.evals.schemas import (
    AnalyticalCandidate,
)

from app.evals.scorer_v0_2 import (
    SCORER_RULE_VERSION,
    score_candidate_v0_2,
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
    / "analytical_reasoning_hard_v0_4.jsonl"
)


ORIGINAL_RESULT_PATH = (
    BASE_DIR
    / "evals"
    / "results"
    / "gemma3_4b_validation_hard_v0_4.json"
)


RESCORED_RESULT_PATH = (
    BASE_DIR
    / "evals"
    / "results"
    / (
        "gemma3_4b_validation_hard_v0_4"
        "_scorer_v0_2.json"
    )
)


# ============================================================
# HELPERS
# ============================================================

def average(
    values: list[
        float
    ],
) -> float:
    if not values:
        return 0.0

    return (
        sum(
            values,
        )
        / len(
            values,
        )
    )


def rounded_dict(
    payload: dict[
        str,
        float,
    ],
) -> dict[
    str,
    float,
]:
    return {
        key:
            round(
                value,
                3,
            )

        for key, value
        in payload.items()
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        "=== DATALENS HARD EVAL RESCORE ==="
    )

    print()

    print(
        "Source model responses:"
    )

    print(
        ORIGINAL_RESULT_PATH,
    )

    print()

    print(
        "New scorer:",
        SCORER_RULE_VERSION,
    )

    print(
        "Ollama calls: 0"
    )

    print()


    # ========================================================
    # VERIFY FILES
    # ========================================================

    if not ORIGINAL_RESULT_PATH.exists():
        raise FileNotFoundError(
            "Résultat v0.4 introuvable : "
            f"{ORIGINAL_RESULT_PATH}"
        )


    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            "Benchmark v0.4 introuvable : "
            f"{BENCHMARK_PATH}"
        )


    # ========================================================
    # LOAD BENCHMARK
    # ========================================================

    cases = load_benchmark(
        BENCHMARK_PATH,
        split="validation",
    )


    assert len(
        cases,
    ) == 6


    cases_by_id = {
        case.case_id:
            case

        for case
        in cases
    }


    # ========================================================
    # LOAD ORIGINAL GEMMA RESPONSES
    # ========================================================

    original_payload = json.loads(
        ORIGINAL_RESULT_PATH.read_text(
            encoding="utf-8",
        )
    )


    original_results = (
        original_payload[
            "results"
        ]
    )


    assert len(
        original_results,
    ) == 6


    original_case_ids = {
        result[
            "case_id"
        ]

        for result
        in original_results
    }


    assert (
        original_case_ids
        == set(
            cases_by_id,
        )
    )


    original_overall = float(
        original_payload[
            "average_overall"
        ]
    )


    print(
        "Original scorer overall:",
        round(
            original_overall,
            3,
        ),
    )

    print()


    # ========================================================
    # RESCORE
    # ========================================================

    rescored_results: list[
        dict
    ] = []


    for original_result in (
        original_results
    ):
        case_id = (
            original_result[
                "case_id"
            ]
        )


        case = (
            cases_by_id[
                case_id
            ]
        )


        candidate_payload = (
            original_result.get(
                "candidate",
            )
        )


        if candidate_payload is None:
            raise ValueError(
                "Candidate absent dans "
                f"{case_id}. "
                "Le replay ne doit pas "
                "rappeler le modèle."
            )


        candidate = (
            AnalyticalCandidate
            .model_validate(
                candidate_payload,
            )
        )


        score = score_candidate_v0_2(
            case,
            candidate,
        )


        score_payload = (
            score.as_dict()
        )


        original_case_overall = float(
            original_result[
                "overall"
            ]
        )


        delta = (
            score.overall
            - original_case_overall
        )


        rescored_results.append(
            {
                "case_id":
                    case_id,

                "domain":
                    case.domain,

                "user_request":
                    case.user_request,

                "candidate":
                    candidate.model_dump(
                        mode="json",
                    ),

                "original_scorer_overall":
                    original_case_overall,

                "rescored_overall":
                    score.overall,

                "delta":
                    delta,

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
            }
        )


    # ========================================================
    # PRINT CASES
    # ========================================================

    for result in (
        rescored_results
    ):
        print(
            "=" * 78
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
            "Question:",
            result[
                "user_request"
            ],
        )


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

        print()

        print(
            "Metrics v0.2:",
            rounded_dict(
                result[
                    "metrics"
                ]
            ),
        )

        print(
            "Capabilities:",
            rounded_dict(
                result[
                    "capabilities"
                ]
            ),
        )

        print()

        print(
            "Original overall:",
            round(
                result[
                    "original_scorer_overall"
                ],
                3,
            ),
        )

        print(
            "Rescored overall:",
            round(
                result[
                    "rescored_overall"
                ],
                3,
            ),
        )


        delta = (
            result[
                "delta"
            ]
        )


        print(
            "Delta:",
            (
                "+"
                if delta >= 0
                else ""
            )
            + str(
                round(
                    delta,
                    3,
                )
            ),
        )


        diagnostics = (
            result[
                "diagnostics"
            ]
        )


        print()

        print(
            "Extra tools:",
            diagnostics[
                "extra_tool_calls"
            ],
        )

        print(
            "Redundant tools:",
            diagnostics[
                "redundant_tool_calls"
            ],
        )

        print(
            "Consistency issues:",
            diagnostics[
                "consistency_issues"
            ],
        )

        print(
            "Required guardrails:",
            diagnostics[
                "required_guardrails"
            ],
        )

        print(
            "Passed guardrails:",
            diagnostics[
                "passed_guardrails"
            ],
        )

        print(
            "Failed guardrails:",
            diagnostics[
                "failed_guardrails"
            ],
        )

        print()


    # ========================================================
    # AVERAGE METRICS
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


    average_metrics = {
        metric_name:
            average(
                [
                    float(
                        result[
                            "metrics"
                        ][
                            metric_name
                        ]
                    )

                    for result
                    in rescored_results
                ]
            )

        for metric_name
        in metric_names
    }


    # ========================================================
    # AVERAGE CAPABILITIES
    # ========================================================

    capability_names = [
        "comprehension",
        "planning",
        "reliability",
    ]


    average_capabilities = {
        capability_name:
            average(
                [
                    float(
                        result[
                            "capabilities"
                        ][
                            capability_name
                        ]
                    )

                    for result
                    in rescored_results
                ]
            )

        for capability_name
        in capability_names
    }


    rescored_overall = average(
        [
            float(
                result[
                    "rescored_overall"
                ]
            )

            for result
            in rescored_results
        ]
    )


    overall_delta = (
        rescored_overall
        - original_overall
    )


    # ========================================================
    # GLOBAL DIAGNOSTICS
    # ========================================================

    total_extra_tool_calls = sum(
        len(
            result[
                "diagnostics"
            ][
                "extra_tool_calls"
            ]
        )

        for result
        in rescored_results
    )


    total_redundant_tool_calls = sum(
        len(
            result[
                "diagnostics"
            ][
                "redundant_tool_calls"
            ]
        )

        for result
        in rescored_results
    )


    total_consistency_issues = sum(
        len(
            result[
                "diagnostics"
            ][
                "consistency_issues"
            ]
        )

        for result
        in rescored_results
    )


    total_required_guardrails = sum(
        len(
            result[
                "diagnostics"
            ][
                "required_guardrails"
            ]
        )

        for result
        in rescored_results
    )


    total_passed_guardrails = sum(
        len(
            result[
                "diagnostics"
            ][
                "passed_guardrails"
            ]
        )

        for result
        in rescored_results
    )


    total_failed_guardrails = sum(
        len(
            result[
                "diagnostics"
            ][
                "failed_guardrails"
            ]
        )

        for result
        in rescored_results
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "=" * 78
    )

    print(
        "RESCORED HARD BENCHMARK SUMMARY"
    )

    print(
        "=" * 78
    )

    print(
        "Cases:",
        len(
            rescored_results
        ),
    )

    print(
        "Ollama calls:",
        0,
    )

    print()


    for (
        metric_name,
        metric_value,
    ) in (
        average_metrics.items()
    ):
        print(
            f"{metric_name:<24}",
            round(
                metric_value,
                3,
            ),
        )


    print()

    print(
        "=" * 78
    )

    print(
        "CAPABILITY VIEW v0.2"
    )

    print(
        "=" * 78
    )


    for (
        capability_name,
        capability_value,
    ) in (
        average_capabilities.items()
    ):
        print(
            f"{capability_name:<20}",
            round(
                capability_value,
                3,
            ),
        )


    print()

    print(
        "=" * 78
    )

    print(
        "SCORER COMPARISON"
    )

    print(
        "=" * 78
    )


    print(
        "Original scorer:",
        round(
            original_overall,
            3,
        ),
    )

    print(
        "Scorer v0.2:",
        round(
            rescored_overall,
            3,
        ),
    )

    print(
        "Delta:",
        (
            "+"
            if overall_delta >= 0
            else ""
        )
        + str(
            round(
                overall_delta,
                3,
            )
        ),
    )


    print()

    print(
        "Extra tool calls:",
        total_extra_tool_calls,
    )

    print(
        "Redundant tool calls:",
        total_redundant_tool_calls,
    )

    print(
        "Consistency issues:",
        total_consistency_issues,
    )

    print(
        "Required guardrails:",
        total_required_guardrails,
    )

    print(
        "Passed guardrails:",
        total_passed_guardrails,
    )

    print(
        "Failed guardrails:",
        total_failed_guardrails,
    )


    # ========================================================
    # TARGETED REGRESSION CHECKS
    # ========================================================

    rescored_by_id = {
        result[
            "case_id"
        ]:
            result

        for result
        in rescored_results
    }


    # --------------------------------------------------------
    # TELECOM
    #
    # Association must now be symmetric.
    # --------------------------------------------------------

    telecom = (
        rescored_by_id[
            "hard_v0_4_002"
        ]
    )


    assert (
        telecom[
            "metrics"
        ][
            "tool_arguments"
        ]
        == 1.0
    )


    # --------------------------------------------------------
    # STORE ENTITY ANALYSIS
    #
    # Gemma called build_entity_view but did not declare
    # entity / target_grain coherently.
    # --------------------------------------------------------

    store = (
        rescored_by_id[
            "hard_v0_4_001"
        ]
    )


    assert (
        store[
            "metrics"
        ][
            "plan_consistency"
        ]
        < 1.0
    )


    assert (
        "entity_analysis_without_entity"
        in store[
            "diagnostics"
        ][
            "consistency_issues"
        ]
    )


    assert (
        "entity_analysis_without_target_grain"
        in store[
            "diagnostics"
        ][
            "consistency_issues"
        ]
    )


    # --------------------------------------------------------
    # MACHINE ENTITY ANALYSIS
    # --------------------------------------------------------

    machine = (
        rescored_by_id[
            "hard_v0_4_005"
        ]
    )


    assert (
        machine[
            "metrics"
        ][
            "plan_consistency"
        ]
        < 1.0
    )


    # --------------------------------------------------------
    # CAUSAL QUESTION
    #
    # Previous scorer gave safety=1.
    # The positive guardrail must now fail.
    # --------------------------------------------------------

    causal = (
        rescored_by_id[
            "hard_v0_4_004"
        ]
    )


    assert (
        causal[
            "metrics"
        ][
            "guardrails"
        ]
        == 0.0
    )


    assert (
        "causality_not_established"
        in causal[
            "diagnostics"
        ][
            "failed_guardrails"
        ]
    )


    assert (
        causal[
            "capabilities"
        ][
            "reliability"
        ]
        < 1.0
    )


    # --------------------------------------------------------
    # PERFECT SUPPORT CASE
    #
    # Must remain perfect after scorer upgrade.
    # --------------------------------------------------------

    support = (
        rescored_by_id[
            "hard_v0_4_006"
        ]
    )


    assert (
        support[
            "rescored_overall"
        ]
        == 1.0
    )


    # ========================================================
    # SAVE
    # ========================================================

    output_payload = {
        "evaluation_name":
            (
                "hard_validation_v0.4"
                "_rescored_v0.2"
            ),

        "source_result_file":
            str(
                ORIGINAL_RESULT_PATH
            ),

        "benchmark_file":
            str(
                BENCHMARK_PATH
            ),

        "model":
            original_payload.get(
                "model",
                "unknown",
            ),

        "candidate_contract_version":
            original_payload.get(
                "candidate_contract_version"
            ),

        "original_scorer": {
            "average_overall":
                original_overall,
        },

        "new_scorer": {
            "rule_version":
                SCORER_RULE_VERSION,

            "average_metrics":
                average_metrics,

            "average_capabilities":
                average_capabilities,

            "average_overall":
                rescored_overall,
        },

        "comparison": {
            "overall_delta":
                overall_delta,
        },

        "diagnostics": {
            "extra_tool_call_count":
                total_extra_tool_calls,

            "redundant_tool_call_count":
                total_redundant_tool_calls,

            "consistency_issue_count":
                total_consistency_issues,

            "required_guardrail_count":
                total_required_guardrails,

            "passed_guardrail_count":
                total_passed_guardrails,

            "failed_guardrail_count":
                total_failed_guardrails,
        },

        "results":
            rescored_results,
    }


    RESCORED_RESULT_PATH.write_text(
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
        RESCORED_RESULT_PATH,
    )

    print()

    print(
        "Hard benchmark rescore: PASS"
    )


if __name__ == "__main__":
    main()