from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from app.evals.dataset_dependency_contract_v0_8 import (
    validate_dependency_candidate,
)

from app.evals.dataset_dependency_extractor_v0_8 import (
    DATASET_DEPENDENCY_EXTRACTOR_VERSION,
    DATASET_DEPENDENCY_PROMPT_VERSION,
    MODEL,
    extract_dataset_dependencies,
)

from app.evals.dataset_dependency_scorer_v0_8 import (
    DATASET_DEPENDENCY_SCORER_VERSION,
    score_dataset_dependency_candidate,
)

from app.evals.decision_router_benchmark_v0_7 import (
    DecisionRouterEvalCase,
    load_decision_router_benchmark,
)

from app.evals.routing_relationships_v0_8 import (
    RoutingRelationshipContext,
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
    / "dependency_extractor_v0_8"
)


OUTPUT_PATH = (
    RESULTS_DIR
    / "qwen3_4b_instruct_multidataset_train_v0_8.json"
)


# ============================================================
# EXPECTED SEMANTIC DEPENDENCIES
#
# IMPORTANT:
#
# These expectations are NEVER sent to the model.
#
# They evaluate only which datasets are required together for
# each analytical result.
# ============================================================

EXPECTED_GROUPS: dict[
    str,
    list[
        list[str]
    ],
] = {

    # --------------------------------------------------------
    # Association requires inventory + sales together.
    # --------------------------------------------------------

    "router_md_v0_7_2_train_001": [
        [
            "inventory",
            "sales",
        ],
    ],


    # --------------------------------------------------------
    # Association requires marketing + occupancy together.
    # --------------------------------------------------------

    "router_md_v0_7_2_train_002": [
        [
            "marketing",
            "occupancy",
        ],
    ],


    # --------------------------------------------------------
    # Only sales is needed.
    # --------------------------------------------------------

    "router_md_v0_7_2_train_003": [
        [
            "sales",
        ],
    ],


    # --------------------------------------------------------
    # Support and commerce are required by the same
    # association result.
    # --------------------------------------------------------

    "router_md_v0_7_2_train_004": [
        [
            "support",
            "commerce",
        ],
    ],


    # --------------------------------------------------------
    # Both employee and production information are required by
    # the same relationship analysis.
    #
    # Whether they are structurally connectable is deliberately
    # NOT the extractor's responsibility.
    # --------------------------------------------------------

    "router_md_v0_7_2_train_005": [
        [
            "employees",
            "production",
        ],
    ],


    # --------------------------------------------------------
    # Two independent results.
    # --------------------------------------------------------

    "router_md_v0_7_2_train_006": [
        [
            "sales",
        ],

        [
            "support",
        ],
    ],
}


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


def _make_reference_validation_context(
    case: DecisionRouterEvalCase,
) -> RoutingRelationshipContext:
    """
    Build a minimal structural context solely to validate that
    the AI did not invent dataset IDs.

    Relationships are intentionally empty here.

    This test evaluates semantic extraction, not join
    feasibility.
    """

    return (
        RoutingRelationshipContext
        .model_validate(
            {
                "datasets": [
                    dataset.model_dump(
                        mode="json",
                    )

                    for dataset
                    in case.datasets
                ],

                "relationships":
                    [],

                "available_tools":
                    case.available_tools,
            }
        )
    )


def _candidate_groups(
    candidate_dict: dict[
        str,
        Any,
    ],
) -> list[
    list[str]
]:

    return [
        requirement[
            "dataset_ids"
        ]

        for requirement
        in candidate_dict[
            "requirements"
        ]
    ]


# ============================================================
# SINGLE CASE
# ============================================================

def run_case(
    case: DecisionRouterEvalCase,
) -> dict[
    str,
    Any,
]:

    expected_groups = (
        EXPECTED_GROUPS[
            case.case_id
        ]
    )


    allowed_dataset_ids = {
        dataset.dataset_id

        for dataset
        in case.datasets
    }


    raw_content: (
        str
        | None
    ) = None


    inference_ms = 0.0


    try:

        (
            candidate,
            raw_content,
            inference_ms,
        ) = (
            extract_dataset_dependencies(
                case=case,
            )
        )


    except Exception as error:

        return {
            "case_id":
                case.case_id,

            "domain":
                case.domain,

            "status":
                "generation_error",

            "expected_groups":
                expected_groups,

            "candidate":
                None,

            "score":
                None,

            "reference_validation":
                False,

            "reference_validation_error":
                None,

            "inference_ms":
                inference_ms,

            "overall":
                0.0,

            "raw_content":
                raw_content,

            "error":
                (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
        }


    # ========================================================
    # SCORE SEMANTIC EXTRACTION
    # ========================================================

    score = (
        score_dataset_dependency_candidate(
            candidate=candidate,

            expected_groups=(
                expected_groups
            ),

            allowed_dataset_ids=(
                allowed_dataset_ids
            ),
        )
    )


    # ========================================================
    # PYTHON REFERENCE VALIDATION
    #
    # We validate only that dataset IDs actually exist.
    #
    # We deliberately do NOT evaluate structural feasibility
    # in this test.
    # ========================================================

    reference_validation = True

    reference_validation_error: (
        str
        | None
    ) = None


    try:

        validate_dependency_candidate(
            candidate=candidate,

            context=(
                _make_reference_validation_context(
                    case
                )
            ),
        )


    except Exception as error:

        reference_validation = (
            False
        )


        reference_validation_error = (
            f"{type(error).__name__}: "
            f"{error}"
        )


    candidate_dict = (
        candidate.model_dump(
            mode="json",
        )
    )


    return {
        "case_id":
            case.case_id,

        "domain":
            case.domain,

        "status":
            "ready",

        "expected_groups":
            expected_groups,

        "candidate":
            candidate_dict,

        "actual_groups":
            _candidate_groups(
                candidate_dict
            ),

        "score":
            score.as_dict(),

        "reference_validation":
            reference_validation,

        "reference_validation_error":
            reference_validation_error,

        "inference_ms":
            inference_ms,

        "overall":
            score.overall,

        "raw_content":
            raw_content,

        "error":
            None,
    }


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
        "-" * 92
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
        "Expected groups:",
        result[
            "expected_groups"
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

        return


    print(
        "Actual groups:",
        result[
            "actual_groups"
        ],
    )


    metrics = (
        result[
            "score"
        ][
            "metrics"
        ]
    )


    diagnostics = (
        result[
            "score"
        ][
            "diagnostics"
        ]
    )


    print(
        "Exact groups:",
        metrics[
            "exact_groups"
        ],
    )


    print(
        "Dataset F1:",
        metrics[
            "dataset_f1"
        ],
    )


    print(
        "Grouping F1:",
        metrics[
            "pairwise_grouping_f1"
        ],
    )


    print(
        "Requirement count:",
        metrics[
            "requirement_count"
        ],
    )


    print(
        "Overall:",
        result[
            "overall"
        ],
    )


    print(
        "Missing datasets:",
        diagnostics[
            "missing_dataset_ids"
        ],
    )


    print(
        "Hallucinated datasets:",
        diagnostics[
            "hallucinated_dataset_ids"
        ],
    )


    print(
        "Duplicate groups:",
        diagnostics[
            "duplicate_requirement_groups"
        ],
    )


    print(
        "Python dataset validation:",
        (
            "PASS"

            if result[
                "reference_validation"
            ]

            else "FAIL"
        ),
    )


    if not (
        result[
            "reference_validation"
        ]
    ):

        print(
            "Validation error:",
            result[
                "reference_validation_error"
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
        "=== DATALENS DATASET DEPENDENCY EXTRACTOR v0.8 ==="
    )

    print()


    print(
        "Extractor:",
        DATASET_DEPENDENCY_EXTRACTOR_VERSION,
    )


    print(
        "Prompt:",
        DATASET_DEPENDENCY_PROMPT_VERSION,
    )


    print(
        "Scorer:",
        DATASET_DEPENDENCY_SCORER_VERSION,
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
        "The model sees requests and dataset schemas only."
    )


    print(
        "Expected groups are never sent to the model."
    )


    print(
        "Relationships and available join capabilities are "
        "not used for extraction."
    )


    print()


    # ========================================================
    # LOAD CASES
    # ========================================================

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


    assert (
        set(
            EXPECTED_GROUPS
        )
        == {
            case.case_id
            for case
            in cases
        }
    )


    print(
        "Cases:",
        len(
            cases
        ),
    )


    print()


    # ========================================================
    # RUN
    # ========================================================

    results = [
        run_case(
            case
        )

        for case
        in cases
    ]


    for result in results:

        print_case(
            result
        )


    # ========================================================
    # AGGREGATE
    # ========================================================

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


    exact_group_scores = [
        float(
            result[
                "score"
            ][
                "metrics"
            ][
                "exact_groups"
            ]
        )

        if (
            result[
                "score"
            ]
            is not None
        )

        else 0.0

        for result
        in results
    ]


    dataset_f1_scores = [
        float(
            result[
                "score"
            ][
                "metrics"
            ][
                "dataset_f1"
            ]
        )

        if (
            result[
                "score"
            ]
            is not None
        )

        else 0.0

        for result
        in results
    ]


    grouping_scores = [
        float(
            result[
                "score"
            ][
                "metrics"
            ][
                "pairwise_grouping_f1"
            ]
        )

        if (
            result[
                "score"
            ]
            is not None
        )

        else 0.0

        for result
        in results
    ]


    overall_scores = [
        float(
            result[
                "overall"
            ]
        )

        for result
        in results
    ]


    hallucinated_count = sum(
        len(
            result[
                "score"
            ][
                "diagnostics"
            ][
                "hallucinated_dataset_ids"
            ]
        )

        for result
        in ready_results
    )


    missing_count = sum(
        len(
            result[
                "score"
            ][
                "diagnostics"
            ][
                "missing_dataset_ids"
            ]
        )

        for result
        in ready_results
    )


    validation_failures = sum(
        1

        for result
        in ready_results

        if not (
            result[
                "reference_validation"
            ]
        )
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


    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print(
        "=" * 100
    )


    print(
        "DEPENDENCY EXTRACTOR SUMMARY v0.8"
    )


    print(
        "=" * 100
    )


    print(
        "Average overall:",
        round(
            _average(
                overall_scores
            ),
            3,
        ),
    )


    print(
        "Exact group accuracy:",
        round(
            _average(
                exact_group_scores
            ),
            3,
        ),
    )


    print(
        "Dataset F1:",
        round(
            _average(
                dataset_f1_scores
            ),
            3,
        ),
    )


    print(
        "Grouping F1:",
        round(
            _average(
                grouping_scores
            ),
            3,
        ),
    )


    print()

    print(
        "Missing dataset references:",
        missing_count,
    )


    print(
        "Hallucinated dataset references:",
        hallucinated_count,
    )


    print(
        "Python reference validation failures:",
        validation_failures,
    )


    print(
        "Generation errors:",
        generation_errors,
    )


    print(
        "Average inference:",
        round(
            average_inference_ms,
            1,
        ),
        "ms",
    )


    # ========================================================
    # EXACT CASE TABLE
    # ========================================================

    print()

    print(
        "=" * 100
    )


    print(
        "DEPENDENCY GROUP CHECK"
    )


    print(
        "=" * 100
    )


    for result in results:

        exact = (
            result[
                "status"
            ]
            == "ready"

            and result[
                "score"
            ][
                "metrics"
            ][
                "exact_groups"
            ]
            == 1.0
        )


        print(
            f"{result['case_id']:<34}"
            f" expected={str(result['expected_groups']):<34}"
            f" actual={str(result.get('actual_groups')):<34}"
            f" {'PASS' if exact else 'FAIL'}"
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
            DATASET_DEPENDENCY_EXTRACTOR_VERSION,

        "prompt_version":
            DATASET_DEPENDENCY_PROMPT_VERSION,

        "scorer_version":
            DATASET_DEPENDENCY_SCORER_VERSION,

        "model":
            MODEL,

        "benchmark":
            str(
                BENCHMARK_PATH
            ),

        "split":
            "train",

        "case_count":
            len(
                cases
            ),

        "summary": {
            "average_overall":
                _average(
                    overall_scores
                ),

            "exact_group_accuracy":
                _average(
                    exact_group_scores
                ),

            "dataset_f1":
                _average(
                    dataset_f1_scores
                ),

            "grouping_f1":
                _average(
                    grouping_scores
                ),

            "missing_dataset_reference_count":
                missing_count,

            "hallucinated_dataset_reference_count":
                hallucinated_count,

            "reference_validation_failure_count":
                validation_failures,

            "generation_error_count":
                generation_errors,

            "average_inference_ms":
                average_inference_ms,
        },

        "results":
            results,
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
        "Dataset Dependency Extractor v0.8: PASS"
    )


if __name__ == "__main__":
    main()