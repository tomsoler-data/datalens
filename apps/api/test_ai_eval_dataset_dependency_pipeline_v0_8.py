from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from app.evals.dataset_dependency_contract_v0_8 import (
    DatasetDependencyCandidate,
    evaluate_dataset_dependencies,
)

from app.evals.decision_router_benchmark_v0_7 import (
    DecisionRouterEvalCase,
    load_decision_router_benchmark,
)

from app.evals.routing_relationships_v0_8 import (
    RoutingRelationshipContext,
)


# ============================================================
# VERSION
# ============================================================

DATASET_DEPENDENCY_PIPELINE_VERSION = (
    "dataset_dependency_pipeline_v0.8"
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
    / "decision_router_multidataset_train_v0_7_2.jsonl"
)


EXTRACTOR_RESULT_PATH = (
    BASE_DIR
    / "evals"
    / "results"
    / "dependency_extractor_v0_8"
    / "qwen3_4b_instruct_multidataset_train_v0_8.json"
)


OUTPUT_DIR = (
    BASE_DIR
    / "evals"
    / "results"
    / "dependency_pipeline_v0_8"
)


OUTPUT_PATH = (
    OUTPUT_DIR
    / "qwen3_4b_instruct_multidataset_pipeline_v0_8.json"
)


# ============================================================
# EXPECTED STRUCTURAL VERDICTS
# ============================================================

EXPECTED: dict[
    str,
    dict[str, Any],
] = {

    # --------------------------------------------------------
    # inventory + sales are required together.
    #
    # No join capability exists.
    # --------------------------------------------------------

    "router_md_v0_7_2_train_001": {
        "executable":
            False,

        "feasibilities": [
            "missing_combination_capability",
        ],

        "routing_override_reason":
            "unsupported_analysis",
    },


    # --------------------------------------------------------
    # marketing + occupancy are required together.
    #
    # Different grains and no join capability.
    # --------------------------------------------------------

    "router_md_v0_7_2_train_002": {
        "executable":
            False,

        "feasibilities": [
            "missing_combination_capability",
        ],

        "routing_override_reason":
            "unsupported_analysis",
    },


    # --------------------------------------------------------
    # Only sales is required.
    # --------------------------------------------------------

    "router_md_v0_7_2_train_003": {
        "executable":
            True,

        "feasibilities": [
            "not_required",
        ],

        "routing_override_reason":
            None,
    },


    # --------------------------------------------------------
    # support + commerce are required together.
    #
    # join_datasets exists and a validated relationship is
    # supplied by the structural context.
    # --------------------------------------------------------

    "router_md_v0_7_2_train_004": {
        "executable":
            True,

        "feasibilities": [
            "supported",
        ],

        "routing_override_reason":
            None,
    },


    # --------------------------------------------------------
    # employees + production are semantically required
    # together.
    #
    # join_datasets exists, but there is deliberately no
    # validated structural relationship between them.
    # --------------------------------------------------------

    "router_md_v0_7_2_train_005": {
        "executable":
            False,

        "feasibilities": [
            "missing_validated_relationship",
        ],

        "routing_override_reason":
            "unsupported_analysis",
    },


    # --------------------------------------------------------
    # sales and support are independent analytical results.
    #
    # Each requirement contains only one dataset.
    # --------------------------------------------------------

    "router_md_v0_7_2_train_006": {
        "executable":
            True,

        "feasibilities": [
            "not_required",
            "not_required",
        ],

        "routing_override_reason":
            None,
    },
}


# ============================================================
# LOAD SAVED EXTRACTOR RESULTS
# ============================================================

def load_saved_extractor_results() -> dict[
    str,
    dict[str, Any],
]:

    payload = json.loads(
        EXTRACTOR_RESULT_PATH.read_text(
            encoding="utf-8",
        )
    )


    results = (
        payload[
            "results"
        ]
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
            "Duplicate case IDs found in saved "
            "dependency extractor results."
        )


    return by_case_id


# ============================================================
# RELATIONSHIP DEFINITIONS
# ============================================================

def relationships_for_case(
    case_id: str,
) -> list[
    dict[str, Any]
]:
    """
    Return structural relationships that DataLens preparation
    is assumed to have validated for this test scenario.

    IMPORTANT:

    These relationships are NOT inferred from Qwen output.

    They represent deterministic preparation metadata.
    """

    # ========================================================
    # CUSTOMER COMMERCE
    #
    # support:
    #     customer_month
    #
    # commerce:
    #     customer_month
    #
    # Validated relationship:
    #     customer_id + month
    # ========================================================

    if (
        case_id
        == "router_md_v0_7_2_train_004"
    ):
        return [
            {
                "relationship_id":
                    "support_commerce_customer_month",

                "left_dataset_id":
                    "support",

                "right_dataset_id":
                    "commerce",

                "kind":
                    "join",

                "left_keys": [
                    "customer_id",
                    "month",
                ],

                "right_keys": [
                    "customer_id",
                    "month",
                ],

                "validated":
                    True,
            }
        ]


    # ========================================================
    # MANUFACTURING HR
    #
    # Deliberately no relationship:
    #
    # employee_month
    # vs
    # machine_day
    #
    # No employee -> machine mapping has been validated.
    # ========================================================

    if (
        case_id
        == "router_md_v0_7_2_train_005"
    ):
        return []


    # ========================================================
    # OTHER CASES
    #
    # Either:
    #
    # - combination capability is already absent;
    # - only one dataset is required;
    # - analyses are independent.
    #
    # No validated relationship is needed for the expected
    # structural verdict.
    # ========================================================

    return []


# ============================================================
# BUILD STRUCTURAL CONTEXT
# ============================================================

def build_context(
    case: DecisionRouterEvalCase,
) -> RoutingRelationshipContext:

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
                    relationships_for_case(
                        case.case_id
                    ),

                "available_tools":
                    case.available_tools,
            }
        )
    )


# ============================================================
# LOAD CANDIDATE
# ============================================================

def candidate_from_saved_result(
    result: dict[
        str,
        Any,
    ],
) -> DatasetDependencyCandidate:

    if (
        result[
            "status"
        ]
        != "ready"
    ):
        raise ValueError(
            "Saved extractor result is not ready for "
            f"case {result['case_id']}."
        )


    candidate_payload = (
        result[
            "candidate"
        ]
    )


    if candidate_payload is None:
        raise ValueError(
            "Saved extractor result contains no candidate "
            f"for case {result['case_id']}."
        )


    return (
        DatasetDependencyCandidate
        .model_validate(
            candidate_payload
        )
    )


# ============================================================
# SINGLE CASE
# ============================================================

def evaluate_case(
    *,
    case: DecisionRouterEvalCase,
    saved_result: dict[str, Any],
) -> dict[str, Any]:

    candidate = (
        candidate_from_saved_result(
            saved_result
        )
    )


    context = (
        build_context(
            case
        )
    )


    gate_result = (
        evaluate_dataset_dependencies(
            candidate=candidate,
            context=context,
        )
    )


    actual_feasibilities = [
        requirement.feasibility

        for requirement
        in gate_result.requirements
    ]


    expected = (
        EXPECTED[
            case.case_id
        ]
    )


    exact = (
        gate_result.executable
        == expected[
            "executable"
        ]

        and actual_feasibilities
        == expected[
            "feasibilities"
        ]

        and (
            gate_result.routing_override_reason
            == expected[
                "routing_override_reason"
            ]
        )
    )


    return {
        "case_id":
            case.case_id,

        "domain":
            case.domain,

        "candidate":
            candidate.model_dump(
                mode="json",
            ),

        "expected":
            expected,

        "actual": {
            "executable":
                gate_result.executable,

            "feasibilities":
                actual_feasibilities,

            "blocking_requirements":
                gate_result.blocking_requirements,

            "routing_override_reason":
                gate_result.routing_override_reason,
        },

        "exact":
            exact,
    }


# ============================================================
# DISPLAY
# ============================================================

def print_result(
    result: dict[
        str,
        Any,
    ],
) -> None:

    print(
        "-" * 100
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


    requirements = (
        result[
            "candidate"
        ][
            "requirements"
        ]
    )


    groups = [
        requirement[
            "dataset_ids"
        ]

        for requirement
        in requirements
    ]


    print(
        "AI dependency groups:",
        groups,
    )


    print(
        "Expected executable:",
        result[
            "expected"
        ][
            "executable"
        ],
    )


    print(
        "Actual executable:",
        result[
            "actual"
        ][
            "executable"
        ],
    )


    print(
        "Expected feasibility:",
        result[
            "expected"
        ][
            "feasibilities"
        ],
    )


    print(
        "Actual feasibility:",
        result[
            "actual"
        ][
            "feasibilities"
        ],
    )


    print(
        "Blocking requirements:",
        result[
            "actual"
        ][
            "blocking_requirements"
        ],
    )


    print(
        "Expected override:",
        result[
            "expected"
        ][
            "routing_override_reason"
        ],
    )


    print(
        "Actual override:",
        result[
            "actual"
        ][
            "routing_override_reason"
        ],
    )


    print(
        "Pipeline:",
        (
            "PASS"
            if result[
                "exact"
            ]
            else "FAIL"
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS DATASET DEPENDENCY PIPELINE v0.8 ==="
    )

    print(
        "Pipeline:",
        DATASET_DEPENDENCY_PIPELINE_VERSION,
    )

    print()


    print(
        "IMPORTANT:"
    )


    print(
        "No model inference is performed by this test."
    )


    print(
        "The saved first-run Qwen dependency extraction "
        "results are reused."
    )


    print(
        "Python now evaluates structural feasibility."
    )

    print()


    # ========================================================
    # VERIFY FILES
    # ========================================================

    if not (
        BENCHMARK_PATH.exists()
    ):
        raise FileNotFoundError(
            BENCHMARK_PATH
        )


    if not (
        EXTRACTOR_RESULT_PATH.exists()
    ):
        raise FileNotFoundError(
            EXTRACTOR_RESULT_PATH
        )


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
            EXPECTED
        )
        == {
            case.case_id
            for case
            in cases
        }
    )


    # ========================================================
    # LOAD SAVED AI RESULTS
    # ========================================================

    saved_results = (
        load_saved_extractor_results()
    )


    assert (
        set(
            saved_results
        )
        == {
            case.case_id
            for case
            in cases
        }
    )


    # ========================================================
    # EVALUATE PIPELINE
    # ========================================================

    results = [
        evaluate_case(
            case=case,

            saved_result=(
                saved_results[
                    case.case_id
                ]
            ),
        )

        for case
        in cases
    ]


    for result in results:
        print_result(
            result
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    pass_count = sum(
        1

        for result
        in results

        if result[
            "exact"
        ]
    )


    fail_count = (
        len(
            results
        )
        - pass_count
    )


    executable_count = sum(
        1

        for result
        in results

        if (
            result[
                "actual"
            ][
                "executable"
            ]
        )
    )


    blocked_count = (
        len(
            results
        )
        - executable_count
    )


    print()

    print(
        "=" * 100
    )


    print(
        "PIPELINE SUMMARY v0.8"
    )


    print(
        "=" * 100
    )


    print(
        "Cases:",
        len(
            results
        ),
    )


    print(
        "Exact pipeline verdicts:",
        (
            f"{pass_count}/"
            f"{len(results)}"
        ),
    )


    print(
        "Executable:",
        executable_count,
    )


    print(
        "Blocked:",
        blocked_count,
    )


    print(
        "Failures:",
        fail_count,
    )


    print()


    # ========================================================
    # SEMANTIC + STRUCTURAL TABLE
    # ========================================================

    print(
        "=" * 100
    )


    print(
        "SEMANTIC + STRUCTURAL CHECK"
    )


    print(
        "=" * 100
    )


    for result in results:

        requirements = (
            result[
                "candidate"
            ][
                "requirements"
            ]
        )


        groups = [
            requirement[
                "dataset_ids"
            ]

            for requirement
            in requirements
        ]


        print(
            f"{result['case_id']:<34}"
            f" groups={str(groups):<42}"
            f" executable={str(result['actual']['executable']):<7}"
            f" {'PASS' if result['exact'] else 'FAIL'}"
        )


    # ========================================================
    # HARD ASSERTION
    #
    # This integration test is deterministic.
    # ========================================================

    assert (
        fail_count
        == 0
    ), (
        f"{fail_count} dependency pipeline "
        "case(s) failed."
    )


    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    payload = {
        "pipeline_version":
            DATASET_DEPENDENCY_PIPELINE_VERSION,

        "benchmark":
            str(
                BENCHMARK_PATH
            ),

        "extractor_result":
            str(
                EXTRACTOR_RESULT_PATH
            ),

        "model_inference_performed":
            False,

        "case_count":
            len(
                results
            ),

        "summary": {
            "exact_pipeline_verdicts":
                pass_count,

            "failures":
                fail_count,

            "executable":
                executable_count,

            "blocked":
                blocked_count,
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
        "Dataset Dependency Pipeline v0.8: PASS"
    )


if __name__ == "__main__":
    main()