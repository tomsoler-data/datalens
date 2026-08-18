from __future__ import annotations

from pathlib import Path

from app.evals.dataset_dependency_benchmark_v0_8 import (
    DATASET_DEPENDENCY_FROZEN_BENCHMARK_VERSION,
    load_dataset_dependency_frozen_benchmark,
)

from app.evals.dataset_dependency_contract_v0_8 import (
    DatasetDependencyCandidate,
    evaluate_dataset_dependencies,
)

from app.evals.routing_relationships_v0_8 import (
    RoutingRelationshipContext,
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(
    __file__,
).resolve().parent


BENCHMARK_PATH = (
    BASE_DIR
    / "evals"
    / "dataset_dependency_frozen_v0_8.jsonl"
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS FROZEN DEPENDENCY BENCHMARK CONTRACT v0.8 ==="
    )

    print()


    print(
        "Benchmark:",
        DATASET_DEPENDENCY_FROZEN_BENCHMARK_VERSION,
    )


    # ========================================================
    # LOAD
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
        == 12
    )


    # ========================================================
    # EXPECTED DISTRIBUTION
    # ========================================================

    executable_count = sum(
        1

        for case
        in cases

        if case.expected.executable
    )


    blocked_count = (
        len(
            cases
        )
        - executable_count
    )


    assert (
        executable_count
        == 7
    )


    assert (
        blocked_count
        == 5
    )


    # ========================================================
    # DETERMINISTIC GROUND-TRUTH VALIDATION
    #
    # Recreate the ideal semantic candidate from the benchmark
    # expectation, then verify that Python produces exactly the
    # expected structural verdict.
    #
    # This does NOT call any model.
    # ========================================================

    for case in cases:

        candidate = (
            DatasetDependencyCandidate
            .model_validate(
                {
                    "requirements": [
                        {
                            "requirement_id":
                                (
                                    "expected_"
                                    f"{index}"
                                ),

                            "dataset_ids":
                                group,
                        }

                        for index, group
                        in enumerate(
                            case
                            .expected
                            .expected_groups,
                            start=1,
                        )
                    ],
                }
            )
        )


        context = (
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


        result = (
            evaluate_dataset_dependencies(
                candidate=candidate,
                context=context,
            )
        )


        actual_feasibilities = [
            requirement.feasibility

            for requirement
            in result.requirements
        ]


        assert (
            actual_feasibilities
            == (
                case
                .expected
                .expected_feasibilities
            )
        ), (
            f"Feasibility mismatch for "
            f"{case.case_id}: "
            f"{actual_feasibilities} != "
            f"{case.expected.expected_feasibilities}"
        )


        assert (
            result.executable
            == case.expected.executable
        ), (
            f"Executable mismatch for "
            f"{case.case_id}"
        )


        assert (
            result.routing_override_reason
            == (
                case
                .expected
                .routing_override_reason
            )
        ), (
            f"Override mismatch for "
            f"{case.case_id}"
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
            domains
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
        "Split: test"
    )


    print(
        "Frozen: True"
    )


    print(
        "Ground-truth feasibility: validated by Python"
    )


    print()


    for case in cases:

        print(
            "-",
            case.case_id,
            "|",
            case.domain,
            "|",
            case.expected.expected_groups,
            "| executable:",
            case.expected.executable,
        )


    print()

    print(
        "Frozen Dependency benchmark contract v0.8: PASS"
    )


if __name__ == "__main__":
    main()