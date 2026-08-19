from __future__ import annotations

from pathlib import Path

from app.evals.decision_router_benchmark_v0_7 import (
    load_decision_router_benchmark,
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
    / "decision_router_multidataset_train_v0_7_2.jsonl"
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS MULTI-DATASET ROUTER TRAIN CONTRACT v0.7.2 ==="
    )

    print()


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


    assert all(
        not case.frozen
        for case
        in cases
    )


    assert all(
        len(
            case.datasets
        )
        == 2
        for case
        in cases
    )


    by_id = {
        case.case_id:
            case
        for case
        in cases
    }


    # ========================================================
    # EXPECTED DISTRIBUTION
    # ========================================================

    analyze_count = sum(
        1

        for case
        in cases

        if (
            case.expected.decision
            == "analyze"
        )
    )


    cannot_count = sum(
        1

        for case
        in cases

        if (
            case.expected.decision
            == "cannot_answer"
        )
    )


    assert (
        analyze_count
        == 3
    )


    assert (
        cannot_count
        == 3
    )


    # ========================================================
    # JOIN ABSENT
    # ========================================================

    no_join_case = (
        by_id[
            "router_md_v0_7_2_train_001"
        ]
    )


    assert (
        "join_datasets"
        not in no_join_case.available_tools
    )


    assert (
        no_join_case
        .expected
        .decision
        == "cannot_answer"
    )


    assert (
        no_join_case
        .expected
        .decision_reason
        == "unsupported_analysis"
    )


    # ========================================================
    # SECOND DATASET IRRELEVANT
    # ========================================================

    single_dataset_needed = (
        by_id[
            "router_md_v0_7_2_train_003"
        ]
    )


    assert (
        single_dataset_needed
        .expected
        .decision
        == "analyze"
    )


    assert (
        "join_datasets"
        not in (
            single_dataset_needed
            .available_tools
        )
    )


    # ========================================================
    # JOIN AVAILABLE
    # ========================================================

    join_available = (
        by_id[
            "router_md_v0_7_2_train_004"
        ]
    )


    assert (
        "join_datasets"
        in join_available.available_tools
    )


    assert (
        join_available
        .expected
        .decision
        == "analyze"
    )


    # ========================================================
    # JOIN EXISTS BUT SEMANTIC LINK DOES NOT
    # ========================================================

    incompatible = (
        by_id[
            "router_md_v0_7_2_train_005"
        ]
    )


    assert (
        "join_datasets"
        in incompatible.available_tools
    )


    assert (
        incompatible
        .expected
        .decision
        == "cannot_answer"
    )


    # ========================================================
    # INDEPENDENT ANALYSES
    # ========================================================

    independent = (
        by_id[
            "router_md_v0_7_2_train_006"
        ]
    )


    assert (
        independent
        .expected
        .decision
        == "analyze"
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
        "Analyze:",
        analyze_count,
    )

    print(
        "Cannot answer:",
        cannot_count,
    )

    print(
        "Frozen:",
        False,
    )

    print()


    for case in cases:

        print(
            "-",
            case.case_id,
            "|",
            case.domain,
            "|",
            case.expected.decision,
            "|",
            case.expected.decision_reason,
        )


    print()

    print(
        "Multi-dataset router train contract v0.7.2: PASS"
    )


if __name__ == "__main__":
    main()