from __future__ import annotations

from pathlib import Path

from app.evals.analytical_planner_benchmark_v0_9 import (
    ANALYTICAL_PLANNER_BENCHMARK_VERSION,
    build_planner_input_for_case,
    load_analytical_planner_benchmark,
)

from app.evals.analytical_planner_validator_v0_9 import (
    validate_analytical_planner_candidate,
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(
    __file__,
).resolve().parents[3]


BENCHMARK_PATH = (
    BASE_DIR
    / "evals"
    / "analytical_planner_development_v0_9.jsonl"
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL PLANNER BENCHMARK v0.9 ==="
    )


    print(
        "Benchmark:",
        ANALYTICAL_PLANNER_BENCHMARK_VERSION,
    )


    print()


    # ========================================================
    # LOAD COMPLETE BENCHMARK
    # ========================================================

    cases = (
        load_analytical_planner_benchmark(
            BENCHMARK_PATH
        )
    )


    assert (
        len(
            cases
        )
        == 10
    )


    # ========================================================
    # SPLITS
    # ========================================================

    train_cases = (
        load_analytical_planner_benchmark(
            BENCHMARK_PATH,
            split="train",
        )
    )


    validation_cases = (
        load_analytical_planner_benchmark(
            BENCHMARK_PATH,
            split="validation",
        )
    )


    assert (
        len(
            train_cases
        )
        == 5
    )


    assert (
        len(
            validation_cases
        )
        == 5
    )


    # ========================================================
    # UNIQUE DOMAINS
    # ========================================================

    domains = {
        case.domain

        for case
        in cases
    }


    assert (
        len(
            domains
        )
        == 10
    )


    # ========================================================
    # NON-FROZEN
    # ========================================================

    assert all(
        not case.frozen

        for case
        in cases
    )


    # ========================================================
    # EXPECTED PLANS PASS REAL VALIDATOR
    # ========================================================

    for case in cases:

        planner_input = (
            build_planner_input_for_case(
                case
            )
        )


        validation = (
            validate_analytical_planner_candidate(
                candidate=(
                    case.expected
                ),

                planner_input=(
                    planner_input
                ),
            )
        )


        assert (
            validation.valid
        ), (
            f"Expected plan failed validation: "
            f"{case.case_id}"
        )


    # ========================================================
    # COVERAGE
    # ========================================================

    families = {
        plan.family

        for case
        in cases

        for plan
        in case.expected.plans
    }


    required_families = {
        "aggregation",
        "group_comparison",
        "association",
        "time_series",
        "distribution",
        "entity_outlier",
    }


    assert (
        required_families
        <= families
    )


    # ========================================================
    # SPECIAL CAPABILITIES
    # ========================================================

    has_multirequirement_case = any(
        len(
            case.expected.plans
        )
        > 1

        for case
        in cases
    )


    assert (
        has_multirequirement_case
    )


    has_derived_metric_case = any(
        any(
            step.action.name
            == "derive_metric"

            for plan
            in case.expected.plans

            for step
            in plan.steps
        )

        for case
        in cases
    )


    assert (
        has_derived_metric_case
    )


    has_entity_plan = any(
        any(
            step.action.name
            == "detect_entity_outliers"

            for plan
            in case.expected.plans

            for step
            in plan.steps
        )

        for case
        in cases
    )


    assert (
        has_entity_plan
    )


    # ========================================================
    # ENCODING
    # ========================================================

    raw_bytes = (
        BENCHMARK_PATH.read_bytes()
    )


    assert not (
        raw_bytes.startswith(
            b"\xef\xbb\xbf"
        )
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        "Cases:",
        len(
            cases
        ),
    )


    print(
        "Train:",
        len(
            train_cases
        ),
    )


    print(
        "Validation:",
        len(
            validation_cases
        ),
    )


    print(
        "Domains:",
        len(
            domains
        ),
    )


    print(
        "Frozen: False"
    )


    print(
        "Expected plans validated by Python: PASS"
    )


    print(
        "Multi-requirement coverage: PASS"
    )


    print(
        "Derived metric coverage: PASS"
    )


    print(
        "Entity planner coverage: PASS"
    )


    print()


    print(
        "Families:",
        sorted(
            families
        ),
    )


    print()


    for case in cases:

        print(
            "-",
            case.case_id,
            "|",
            case.split,
            "|",
            case.domain,
            "| requirements:",
            len(
                case.expected.plans
            ),
        )


    print()

    print(
        "Analytical Planner benchmark contract v0.9: PASS"
    )


if __name__ == "__main__":
    main()