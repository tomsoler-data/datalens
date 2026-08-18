from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from app.evals.dataset_dependency_benchmark_v0_8 import (
    DATASET_DEPENDENCY_FROZEN_BENCHMARK_VERSION,
    DatasetDependencyFrozenCase,
)

from app.evals.dataset_dependency_contract_v0_8 import (
    DATASET_DEPENDENCY_GATE_VERSION,
    DatasetDependencyCandidate,
    evaluate_dataset_dependencies,
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

from app.evals.routing_relationships_v0_8 import (
    ROUTING_RELATIONSHIP_CONTEXT_VERSION,
    RoutingRelationshipContext,
)


# ============================================================
# VERSION
# ============================================================

DATASET_DEPENDENCY_FROZEN_RUNNER_VERSION = (
    "dataset_dependency_frozen_runner_v0.8"
)


# ============================================================
# HELPERS
# ============================================================

def _canonical_group(
    dataset_ids: list[str],
) -> tuple[str, ...]:

    return tuple(
        sorted(
            dataset_id.strip()

            for dataset_id
            in dataset_ids
        )
    )


def _candidate_groups(
    candidate: DatasetDependencyCandidate,
) -> list[
    list[str]
]:

    return [
        requirement.dataset_ids

        for requirement
        in candidate.requirements
    ]


def _expected_feasibility_by_group(
    case: DatasetDependencyFrozenCase,
) -> dict[
    tuple[str, ...],
    str,
]:

    return {
        _canonical_group(
            group
        ):
            feasibility

        for (
            group,
            feasibility,
        ) in zip(
            case
            .expected
            .expected_groups,

            case
            .expected
            .expected_feasibilities,

            strict=True,
        )
    }


# ============================================================
# STRUCTURAL CONTEXT
# ============================================================

def build_structural_context(
    case: DatasetDependencyFrozenCase,
) -> RoutingRelationshipContext:

    return (
        RoutingRelationshipContext(
            datasets=(
                case.datasets
            ),

            relationships=(
                case.relationships
            ),

            available_tools=(
                case.available_tools
            ),
        )
    )


# ============================================================
# SINGLE CASE
# ============================================================

def run_frozen_dependency_case(
    *,
    case: DatasetDependencyFrozenCase,
) -> dict[str, Any]:

    raw_content: (
        str
        | None
    ) = None


    inference_ms = 0.0


    # ========================================================
    # AI SEMANTIC EXTRACTION
    #
    # IMPORTANT:
    #
    # extract_dataset_dependencies serializes only:
    #
    # - user_request
    # - dataset schemas
    #
    # It does NOT serialize:
    #
    # - expected
    # - relationships
    # - available tools
    # - feasibility labels
    # ========================================================

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

            "inference_ms":
                inference_ms,

            "candidate":
                None,

            "semantic_score":
                None,

            "semantic_exact":
                False,

            "gate":
                None,

            "gate_error":
                None,

            "final_verdict_exact":
                False,

            "structural_detail_exact":
                False,

            "end_to_end_exact":
                False,

            "raw_content":
                raw_content,

            "error":
                (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
        }


    # ========================================================
    # SEMANTIC SCORING
    # ========================================================

    allowed_dataset_ids = {
        dataset.dataset_id

        for dataset
        in case.datasets
    }


    semantic_score = (
        score_dataset_dependency_candidate(
            candidate=candidate,

            expected_groups=(
                case
                .expected
                .expected_groups
            ),

            allowed_dataset_ids=(
                allowed_dataset_ids
            ),
        )
    )


    semantic_exact = (
        semantic_score.exact_groups
        == 1.0
    )


    # ========================================================
    # PYTHON STRUCTURAL GATE
    # ========================================================

    try:

        gate_result = (
            evaluate_dataset_dependencies(
                candidate=candidate,

                context=(
                    build_structural_context(
                        case
                    )
                ),
            )
        )


    except Exception as error:

        return {
            "case_id":
                case.case_id,

            "domain":
                case.domain,

            "status":
                "gate_error",

            "inference_ms":
                inference_ms,

            "candidate":
                candidate.model_dump(
                    mode="json",
                ),

            "semantic_score":
                semantic_score.as_dict(),

            "semantic_exact":
                semantic_exact,

            "gate":
                None,

            "gate_error":
                (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),

            "final_verdict_exact":
                False,

            "structural_detail_exact":
                False,

            "end_to_end_exact":
                False,

            "raw_content":
                raw_content,

            "error":
                None,
        }


    # ========================================================
    # FINAL VERDICT
    # ========================================================

    final_verdict_exact = (
        gate_result.executable
        == case.expected.executable

        and (
            gate_result
            .routing_override_reason
            == (
                case
                .expected
                .routing_override_reason
            )
        )
    )


    # ========================================================
    # DETAILED FEASIBILITY
    #
    # Requirement order and requirement IDs do not matter.
    #
    # We compare feasibility by canonical dependency group.
    # ========================================================

    expected_by_group = (
        _expected_feasibility_by_group(
            case
        )
    )


    actual_by_group = {
        _canonical_group(
            requirement.dataset_ids
        ):
            requirement.feasibility

        for requirement
        in gate_result.requirements
    }


    structural_detail_exact = (
        semantic_exact

        and actual_by_group
        == expected_by_group
    )


    # ========================================================
    # END TO END
    # ========================================================

    end_to_end_exact = (
        semantic_exact

        and structural_detail_exact

        and final_verdict_exact
    )


    # ========================================================
    # SERIALIZE GATE
    # ========================================================

    gate_payload = {
        "executable":
            gate_result.executable,

        "routing_override_reason":
            gate_result.routing_override_reason,

        "blocking_requirements":
            gate_result.blocking_requirements,

        "requirements": [
            {
                "requirement_id":
                    requirement.requirement_id,

                "dataset_ids":
                    requirement.dataset_ids,

                "feasibility":
                    requirement.feasibility,

                "executable":
                    requirement.executable,
            }

            for requirement
            in gate_result.requirements
        ],
    }


    return {
        "case_id":
            case.case_id,

        "domain":
            case.domain,

        "status":
            "ready",

        "inference_ms":
            inference_ms,

        "candidate":
            candidate.model_dump(
                mode="json",
            ),

        "actual_groups":
            _candidate_groups(
                candidate
            ),

        "semantic_score":
            semantic_score.as_dict(),

        "semantic_exact":
            semantic_exact,

        "gate":
            gate_payload,

        "gate_error":
            None,

        "final_verdict_exact":
            final_verdict_exact,

        "structural_detail_exact":
            structural_detail_exact,

        "end_to_end_exact":
            end_to_end_exact,

        "raw_content":
            raw_content,

        "error":
            None,
    }


# ============================================================
# SUMMARY
# ============================================================

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
            "Cannot summarize zero frozen results."
        )


    generation_errors = sum(
        1

        for result
        in results

        if (
            result[
                "status"
            ]
            == "generation_error"
        )
    )


    gate_errors = sum(
        1

        for result
        in results

        if (
            result[
                "status"
            ]
            == "gate_error"
        )
    )


    semantic_exact_count = sum(
        1

        for result
        in results

        if result[
            "semantic_exact"
        ]
    )


    final_verdict_exact_count = sum(
        1

        for result
        in results

        if result[
            "final_verdict_exact"
        ]
    )


    structural_detail_exact_count = sum(
        1

        for result
        in results

        if result[
            "structural_detail_exact"
        ]
    )


    end_to_end_exact_count = sum(
        1

        for result
        in results

        if result[
            "end_to_end_exact"
        ]
    )


    # ========================================================
    # SEMANTIC METRICS
    #
    # Errors count as zero.
    # ========================================================

    semantic_overall_values: list[
        float
    ] = []


    dataset_f1_values: list[
        float
    ] = []


    grouping_f1_values: list[
        float
    ] = []


    hallucinated_dataset_count = 0

    missing_dataset_count = 0


    for result in results:

        semantic_score = (
            result.get(
                "semantic_score"
            )
        )


        if semantic_score is None:

            semantic_overall_values.append(
                0.0
            )

            dataset_f1_values.append(
                0.0
            )

            grouping_f1_values.append(
                0.0
            )

            continue


        semantic_overall_values.append(
            float(
                semantic_score[
                    "overall"
                ]
            )
        )


        metrics = (
            semantic_score[
                "metrics"
            ]
        )


        dataset_f1_values.append(
            float(
                metrics[
                    "dataset_f1"
                ]
            )
        )


        grouping_f1_values.append(
            float(
                metrics[
                    "pairwise_grouping_f1"
                ]
            )
        )


        diagnostics = (
            semantic_score[
                "diagnostics"
            ]
        )


        hallucinated_dataset_count += len(
            diagnostics[
                "hallucinated_dataset_ids"
            ]
        )


        missing_dataset_count += len(
            diagnostics[
                "missing_dataset_ids"
            ]
        )


    def average(
        values: list[float],
    ) -> float:

        return (
            sum(
                values
            )
            / len(
                values
            )
        )


    return {
        "case_count":
            total,

        "semantic_exact_count":
            semantic_exact_count,

        "semantic_exact_accuracy":
            (
                semantic_exact_count
                / total
            ),

        "semantic_average_overall":
            average(
                semantic_overall_values
            ),

        "dataset_f1":
            average(
                dataset_f1_values
            ),

        "grouping_f1":
            average(
                grouping_f1_values
            ),

        "final_verdict_exact_count":
            final_verdict_exact_count,

        "final_verdict_accuracy":
            (
                final_verdict_exact_count
                / total
            ),

        "structural_detail_exact_count":
            structural_detail_exact_count,

        "structural_detail_accuracy":
            (
                structural_detail_exact_count
                / total
            ),

        "end_to_end_exact_count":
            end_to_end_exact_count,

        "end_to_end_accuracy":
            (
                end_to_end_exact_count
                / total
            ),

        "hallucinated_dataset_count":
            hallucinated_dataset_count,

        "missing_dataset_count":
            missing_dataset_count,

        "generation_error_count":
            generation_errors,

        "gate_error_count":
            gate_errors,

        "average_inference_ms":
            average(
                [
                    float(
                        result[
                            "inference_ms"
                        ]
                    )

                    for result
                    in results
                ]
            ),
    }


# ============================================================
# METADATA
# ============================================================

def frozen_runner_metadata() -> dict[
    str,
    Any,
]:

    return {
        "runner_version":
            DATASET_DEPENDENCY_FROZEN_RUNNER_VERSION,

        "benchmark_version":
            DATASET_DEPENDENCY_FROZEN_BENCHMARK_VERSION,

        "extractor_version":
            DATASET_DEPENDENCY_EXTRACTOR_VERSION,

        "prompt_version":
            DATASET_DEPENDENCY_PROMPT_VERSION,

        "scorer_version":
            DATASET_DEPENDENCY_SCORER_VERSION,

        "dependency_gate_version":
            DATASET_DEPENDENCY_GATE_VERSION,

        "relationship_context_version":
            ROUTING_RELATIONSHIP_CONTEXT_VERSION,

        "model":
            MODEL,

        "temperature":
            0,

        "thinking":
            False,
    }