from __future__ import annotations

import hashlib

from pathlib import Path

from app.evals.analytical_planner_benchmark_v0_9 import (
    load_analytical_planner_benchmark,
)

from app.evals.analytical_planner_frozen_benchmark_v1_0 import (
    ANALYTICAL_PLANNER_FROZEN_BENCHMARK_VERSION,
    build_planner_input_for_frozen_case,
    load_frozen_analytical_planner_benchmark,
)

from app.evals.analytical_planner_validator_v0_9_1 import (
    ANALYTICAL_PLANNER_VALIDATOR_VERSION,
    validate_analytical_planner_candidate,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__,
).resolve().parent


FROZEN_BENCHMARK_PATH = (
    BASE_DIR
    / "evals"
    / "analytical_planner_frozen_v1_0.jsonl"
)


FROZEN_HASH_PATH = (
    BASE_DIR
    / "evals"
    / "analytical_planner_frozen_v1_0.sha256"
)


DEVELOPMENT_BENCHMARK_PATH = (
    BASE_DIR
    / "evals"
    / "analytical_planner_development_v0_9.jsonl"
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


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL PLANNER FROZEN BENCHMARK CONTRACT v1.0 ==="
    )


    print()


    print(
        "Benchmark:",
        ANALYTICAL_PLANNER_FROZEN_BENCHMARK_VERSION,
    )


    print(
        "Validator:",
        ANALYTICAL_PLANNER_VALIDATOR_VERSION,
    )


    print()


    # ========================================================
    # ARTIFACTS EXIST
    # ========================================================

    assert (
        FROZEN_BENCHMARK_PATH.exists()
    )


    assert (
        FROZEN_HASH_PATH.exists()
    )


    # ========================================================
    # HASH LOCK
    # ========================================================

    expected_hash = (
        FROZEN_HASH_PATH
        .read_text(
            encoding="ascii",
        )
        .strip()
    )


    actual_hash = (
        sha256_file(
            FROZEN_BENCHMARK_PATH
        )
    )


    assert (
        expected_hash
        == actual_hash
    ), (
        "Frozen planner benchmark hash mismatch. "
        "The locked benchmark appears to have changed."
    )


    print(
        "SHA-256 lock verification: PASS"
    )


    # ========================================================
    # LOAD
    # ========================================================

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


    print(
        "Case count: PASS"
    )


    # ========================================================
    # UNIQUE IDS
    # ========================================================

    case_ids = [
        case.case_id

        for case
        in cases
    ]


    assert (
        len(
            case_ids
        )
        == len(
            set(
                case_ids
            )
        )
    )


    print(
        "Unique case IDs: PASS"
    )


    # ========================================================
    # UNIQUE DOMAINS
    # ========================================================

    frozen_domains = {
        case.domain

        for case
        in cases
    }


    assert (
        len(
            frozen_domains
        )
        == 12
    )


    print(
        "Unique frozen domains: PASS"
    )


    # ========================================================
    # NO DEVELOPMENT DOMAIN OVERLAP
    # ========================================================

    development_cases = (
        load_analytical_planner_benchmark(
            DEVELOPMENT_BENCHMARK_PATH
        )
    )


    development_domains = {
        case.domain

        for case
        in development_cases
    }


    overlap = (
        frozen_domains
        & development_domains
    )


    assert (
        overlap
        == set()
    ), (
        "Frozen benchmark reuses development domains: "
        f"{sorted(overlap)}"
    )


    print(
        "No development-domain overlap: PASS"
    )


    # ========================================================
    # TEST + FROZEN
    # ========================================================

    assert all(
        case.split
        == "test"

        for case
        in cases
    )


    assert all(
        case.frozen

        for case
        in cases
    )


    print(
        "Split=test and frozen=True: PASS"
    )


    # ========================================================
    # UTF-8 WITHOUT BOM
    # ========================================================

    raw_bytes = (
        FROZEN_BENCHMARK_PATH
        .read_bytes()
    )


    assert not (
        raw_bytes.startswith(
            b"\xef\xbb\xbf"
        )
    )


    print(
        "UTF-8 without BOM: PASS"
    )


    # ========================================================
    # EXPECTED PLANS PASS REAL VALIDATOR
    # ========================================================

    for case in cases:

        planner_input = (
            build_planner_input_for_frozen_case(
                case
            )
        )


        result = (
            validate_analytical_planner_candidate(
                candidate=(
                    case.expected
                ),

                planner_input=(
                    planner_input
                ),
            )
        )


        assert result.valid, (
            "Frozen ground truth failed validator: "
            f"{case.case_id} "
            f"{[issue.code for issue in result.issues]}"
        )


    print(
        "All frozen ground truths validated by Python: PASS"
    )


    # ========================================================
    # FAMILY COVERAGE
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


    print(
        "Analytical family coverage: PASS"
    )


    # ========================================================
    # DERIVED METRIC COVERAGE
    # ========================================================

    has_derived_metric = any(
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


    assert has_derived_metric


    print(
        "Derived metric coverage: PASS"
    )


    # ========================================================
    # ENTITY-GRAIN COVERAGE
    # ========================================================

    entity_cases = [
        case

        for case
        in cases

        if any(
            plan.family
            == "entity_outlier"

            for plan
            in case.expected.plans
        )
    ]


    assert (
        len(
            entity_cases
        )
        >= 2
    )


    print(
        "Entity-grain coverage: PASS"
    )


    # ========================================================
    # MULTI-REQUIREMENT COVERAGE
    # ========================================================

    has_multi_requirement = any(
        len(
            case.expected.plans
        )
        > 1

        for case
        in cases
    )


    assert has_multi_requirement


    print(
        "Multi-requirement coverage: PASS"
    )


    # ========================================================
    # CROSS-DATASET COVERAGE
    # ========================================================

    cross_dataset_cases = [
        case

        for case
        in cases

        if any(
            len(
                requirement.dataset_ids
            )
            > 1

            for requirement
            in case.dependency_candidate.requirements
        )
    ]


    assert (
        len(
            cross_dataset_cases
        )
        >= 2
    )


    print(
        "Cross-dataset coverage: PASS"
    )


    # ========================================================
    # MULTI-HOP BRIDGE COVERAGE
    # ========================================================

    healthcare_case = next(
        case

        for case
        in cases

        if (
            case.case_id
            == "planner_frozen_v1_0_010"
        )
    )


    healthcare_input = (
        build_planner_input_for_frozen_case(
            healthcare_case
        )
    )


    requirement = (
        healthcare_input.requirements[
            0
        ]
    )


    roles = {
        dataset.dataset_id:
            dataset.role

        for dataset
        in requirement.datasets
    }


    assert (
        roles[
            "consultations"
        ]
        == "semantic"
    )


    assert (
        roles[
            "care_costs"
        ]
        == "semantic"
    )


    assert (
        roles[
            "patients"
        ]
        == "bridge"
    )


    analytical_names = {
        column[
            "qualified_name"
        ]

        for column
        in requirement.analytical_columns
    }


    assert (
        "patients.age_group"
        not in analytical_names
    )


    print(
        "Multi-hop bridge isolation: PASS"
    )


    # ========================================================
    # STRUCTURAL TOOL MUST NEVER REACH PLANNER
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

            assert (
                "join_datasets"
                not in (
                    requirement
                    .allowed_analytical_tools
                )
            )


    print(
        "join_datasets hidden from all frozen planner inputs: PASS"
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    print()


    print(
        "Cases:",
        len(
            cases
        ),
    )


    print(
        "Domains:",
        len(
            frozen_domains
        ),
    )


    print(
        "Split: test"
    )


    print(
        "Frozen: True"
    )


    print(
        "Families:",
        sorted(
            families
        ),
    )


    print(
        "SHA-256:",
        actual_hash,
    )


    print()


    for case in cases:

        print(
            "-",
            case.case_id,
            "|",
            case.domain,
            "| requirements:",
            len(
                case.expected.plans
            ),
        )


    print()


    print(
        "Analytical Planner Frozen Benchmark v1.0: PASS / LOCKED"
    )


if __name__ == "__main__":
    main()