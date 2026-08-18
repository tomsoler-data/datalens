from __future__ import annotations

import json

from pathlib import Path

from app.evals.analytical_planner_frozen_benchmark_v1_0 import (
    build_planner_input_for_frozen_case,
    load_frozen_analytical_planner_benchmark,
)

from app.planning.analytical_v1.contract import (
    AnalyticalPlannerCandidate,
)

from app.planning.analytical_v1.context import (
    build_analytical_planner_context,
)

from app.planning.analytical_v1.dataset_context import (
    DatasetContext,
)

from app.planning.analytical_v1.dependency import (
    DatasetDependencyCandidate,
)

from app.planning.analytical_v1.input import (
    build_analytical_planner_input,
)

from app.planning.analytical_v1.relationships import (
    DatasetRelationshipSpec,
    RoutingRelationshipContext,
)

from app.planning.analytical_v1.safety import (
    evaluate_analytical_planner_safety,
)

from app.planning.analytical_v1.validator import (
    validate_analytical_planner_candidate,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    Path(
        __file__
    )
    .resolve()
    .parent
)


FROZEN_BENCHMARK_PATH = (
    BASE_DIR
    / "evals"
    / "analytical_planner_frozen_v1_0.jsonl"
)


FROZEN_RESULTS_PATH = (
    BASE_DIR
    / "evals"
    / "results"
    / "analytical_planner_frozen_v1_0"
    / "qwen3_4b_instruct_frozen_planner_v1_0.json"
)


APP_DIR = (
    BASE_DIR
    / "app"
)


# ============================================================
# BUILD PRODUCTION PLANNER INPUT
# ============================================================

def build_production_planner_input(
    case,
):

    datasets = [
        DatasetContext.model_validate(
            dataset.model_dump(
                mode="json",
            )
        )

        for dataset
        in case.datasets
    ]


    relationships = [
        DatasetRelationshipSpec.model_validate(
            relationship.model_dump(
                mode="json",
            )
        )

        for relationship
        in case.relationships
    ]


    dependency_candidate = (
        DatasetDependencyCandidate
        .model_validate(
            case
            .dependency_candidate
            .model_dump(
                mode="json",
            )
        )
    )


    structural_context = (
        RoutingRelationshipContext(
            datasets=datasets,
            relationships=relationships,
            available_tools=list(
                case.available_tools
            ),
        )
    )


    planner_context = (
        build_analytical_planner_context(
            candidate=(
                dependency_candidate
            ),

            context=(
                structural_context
            ),
        )
    )


    assert (
        planner_context.ready_for_planning
    ), (
        "Production structural context unexpectedly "
        "blocked a frozen case: "
        f"{case.case_id}"
    )


    return (
        build_analytical_planner_input(
            user_request=(
                case.user_request
            ),

            planner_context=(
                planner_context
            ),

            structural_context=(
                structural_context
            ),
        )
    )


# ============================================================
# 1. NO PRODUCTION APP.EVALS IMPORTS
# ============================================================

def test_no_production_eval_imports() -> None:

    violations: list[
        tuple[
            str,
            int,
            str,
        ]
    ] = []


    for path in (
        APP_DIR.rglob(
            "*.py"
        )
    ):

        if (
            "evals"
            in path.parts
        ):

            continue


        content = (
            path.read_text(
                encoding="utf-8"
            )
        )


        for (
            line_number,
            line,
        ) in enumerate(
            content.splitlines(),
            start=1,
        ):

            if (
                "app.evals"
                in line
            ):

                violations.append(
                    (
                        str(
                            path.relative_to(
                                BASE_DIR
                            )
                        ),
                        line_number,
                        line.strip(),
                    )
                )


    assert (
        violations
        == []
    ), (
        "Production imports app.evals: "
        f"{violations}"
    )


    print(
        "Production app.evals dependency boundary: PASS"
    )


# ============================================================
# 2. LOAD FROZEN CASES
# ============================================================

def load_cases():

    cases = (
        load_frozen_analytical_planner_benchmark(
            FROZEN_BENCHMARK_PATH
        )
    )


    assert (
        len(
            cases
        )
        == 12
    )


    return cases


# ============================================================
# 3. PLANNER INPUT PARITY
# ============================================================

def test_frozen_planner_input_parity() -> None:

    cases = (
        load_cases()
    )


    for case in cases:

        historical_input = (
            build_planner_input_for_frozen_case(
                case
            )
        )


        production_input = (
            build_production_planner_input(
                case
            )
        )


        historical_payload = (
            historical_input.model_dump(
                mode="json"
            )
        )


        production_payload = (
            production_input.model_dump(
                mode="json"
            )
        )


        assert (
            production_payload
            == historical_payload
        ), (
            "Production planner input diverged from "
            "the validated eval implementation. "
            f"case_id={case.case_id}"
        )


    print(
        "Frozen planner input parity 12/12: PASS"
    )


# ============================================================
# 4. ALL FROZEN GROUND TRUTHS PASS PRODUCTION VALIDATOR
# ============================================================

def test_ground_truth_validator_parity() -> None:

    cases = (
        load_cases()
    )


    for case in cases:

        planner_input = (
            build_production_planner_input(
                case
            )
        )


        candidate = (
            AnalyticalPlannerCandidate
            .model_validate(
                case
                .expected
                .model_dump(
                    mode="json"
                )
            )
        )


        validation = (
            validate_analytical_planner_candidate(
                candidate=(
                    candidate
                ),

                planner_input=(
                    planner_input
                ),
            )
        )


        assert validation.valid, (
            "Production validator rejected frozen "
            "ground truth. "
            f"case_id={case.case_id}, "
            "issues="
            f"{[issue.code for issue in validation.issues]}"
        )


    print(
        "Frozen ground truths on production validator 12/12: PASS"
    )


# ============================================================
# HISTORICAL RESULTS
# ============================================================

def load_historical_results() -> dict:

    if not (
        FROZEN_RESULTS_PATH.exists()
    ):

        raise FileNotFoundError(
            "Historical frozen planner result not found: "
            f"{FROZEN_RESULTS_PATH}"
        )


    payload = json.loads(
        FROZEN_RESULTS_PATH.read_text(
            encoding="utf-8"
        )
    )


    return {
        result[
            "case_id"
        ]:
            result

        for result
        in payload[
            "results"
        ]
    }


def get_case_by_id(
    case_id: str,
):

    return next(
        case

        for case
        in load_cases()

        if (
            case.case_id
            == case_id
        )
    )


def candidate_from_historical_result(
    historical_result: dict,
) -> AnalyticalPlannerCandidate:

    candidate_payload = (
        historical_result.get(
            "candidate"
        )
    )


    assert (
        candidate_payload
        is not None
    )


    return (
        AnalyticalPlannerCandidate
        .model_validate(
            candidate_payload
        )
    )


# ============================================================
# 5. HISTORICAL 001
#
# sum must remain blocked.
# ============================================================

def test_historical_001_production_safety() -> None:

    case_id = (
        "planner_frozen_v1_0_001"
    )


    historical = (
        load_historical_results()
    )


    case = (
        get_case_by_id(
            case_id
        )
    )


    candidate = (
        candidate_from_historical_result(
            historical[
                case_id
            ]
        )
    )


    result = (
        evaluate_analytical_planner_safety(
            candidate=(
                candidate
            ),

            planner_input=(
                build_production_planner_input(
                    case
                )
            ),
        )
    )


    assert not (
        result.ready_for_execution
    )


    assert (
        result.blocking_stage
        == "reference_canonicalization"
    )


    assert (
        "unknown_reference"
        in result.blocking_codes
    )


    assert (
        result.execution_candidate
        is None
    )


    print(
        "Historical Frozen 001 remains safely blocked: PASS"
    )


# ============================================================
# 6. HISTORICAL 008
#
# channel becomes ad_performance.channel and the plan becomes
# executable under the production safety boundary.
#
# This does NOT alter or re-score the historical Frozen.
# ============================================================

def test_historical_008_production_safety() -> None:

    case_id = (
        "planner_frozen_v1_0_008"
    )


    historical = (
        load_historical_results()
    )


    case = (
        get_case_by_id(
            case_id
        )
    )


    candidate = (
        candidate_from_historical_result(
            historical[
                case_id
            ]
        )
    )


    result = (
        evaluate_analytical_planner_safety(
            candidate=(
                candidate
            ),

            planner_input=(
                build_production_planner_input(
                    case
                )
            ),
        )
    )


    assert (
        result.ready_for_execution
    )


    assert (
        result.blocking_stage
        == "none"
    )


    assert (
        len(
            result
            .canonicalization
            .rewrites
        )
        == 1
    )


    rewrite = (
        result
        .canonicalization
        .rewrites[
            0
        ]
    )


    assert (
        rewrite.original_reference
        == "channel"
    )


    assert (
        rewrite.canonical_reference
        == "ad_performance.channel"
    )


    assert (
        result.execution_candidate
        is not None
    )


    compare_action = (
        result
        .execution_candidate
        .plans[
            0
        ]
        .steps[
            1
        ]
        .action
    )


    assert (
        compare_action.group_by
        == "ad_performance.channel"
    )


    print(
        "Historical Frozen 008 canonicalized safely in production: PASS"
    )


# ============================================================
# 7. HISTORICAL 012
#
# References become canonical, but the real target grain
# reasoning error must remain blocked.
# ============================================================

def test_historical_012_production_safety() -> None:

    case_id = (
        "planner_frozen_v1_0_012"
    )


    historical = (
        load_historical_results()
    )


    case = (
        get_case_by_id(
            case_id
        )
    )


    candidate = (
        candidate_from_historical_result(
            historical[
                case_id
            ]
        )
    )


    result = (
        evaluate_analytical_planner_safety(
            candidate=(
                candidate
            ),

            planner_input=(
                build_production_planner_input(
                    case
                )
            ),
        )
    )


    assert (
        result.canonicalization.safe
    )


    assert (
        len(
            result
            .canonicalization
            .rewrites
        )
        == 2
    )


    assert not (
        result.ready_for_execution
    )


    assert (
        result.blocking_stage
        == "planner_validation"
    )


    assert (
        "entity_target_grain_mismatch"
        in result.blocking_codes
    )


    assert (
        "unknown_analytical_reference"
        not in result.blocking_codes
    )


    assert (
        result.execution_candidate
        is None
    )


    print(
        "Historical Frozen 012 real grain error remains blocked: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL v1 PRODUCTION PARITY ==="
    )


    print()


    test_no_production_eval_imports()

    test_frozen_planner_input_parity()

    test_ground_truth_validator_parity()

    test_historical_001_production_safety()

    test_historical_008_production_safety()

    test_historical_012_production_safety()


    print()


    print(
        "Analytical v1 production parity: PASS"
    )


if __name__ == "__main__":
    main()