from __future__ import annotations

from app.evals.dataset_dependency_contract_v0_8 import (
    DatasetDependencyCandidate,
)

from app.evals.dataset_dependency_scorer_v0_8 import (
    DATASET_DEPENDENCY_SCORER_VERSION,
    score_dataset_dependency_candidate,
)


# ============================================================
# HELPERS
# ============================================================

ALLOWED = {
    "sales",
    "support",
    "inventory",
    "machines",
}


def candidate(
    groups: list[
        list[str]
    ],
) -> DatasetDependencyCandidate:

    return (
        DatasetDependencyCandidate
        .model_validate(
            {
                "requirements": [
                    {
                        "requirement_id":
                            f"requirement_{index}",

                        "dataset_ids":
                            group,
                    }

                    for index, group
                    in enumerate(
                        groups,
                        start=1,
                    )
                ],
            }
        )
    )


# ============================================================
# 1. EXACT CROSS-DATASET
# ============================================================

def test_exact_cross_dataset() -> None:

    score = (
        score_dataset_dependency_candidate(
            candidate=(
                candidate(
                    [
                        [
                            "sales",
                            "support",
                        ]
                    ]
                )
            ),

            expected_groups=[
                [
                    "sales",
                    "support",
                ]
            ],

            allowed_dataset_ids=(
                ALLOWED
            ),
        )
    )


    assert (
        score.exact_groups
        == 1.0
    )


    assert (
        score.dataset_f1
        == 1.0
    )


    assert (
        score.pairwise_grouping_f1
        == 1.0
    )


    assert (
        score.requirement_count
        == 1.0
    )


    assert (
        score.overall
        == 1.0
    )


    print(
        "Exact cross-dataset dependency: PASS"
    )


# ============================================================
# 2. DATASET ORDER DOES NOT MATTER
# ============================================================

def test_dataset_order_ignored() -> None:

    score = (
        score_dataset_dependency_candidate(
            candidate=(
                candidate(
                    [
                        [
                            "support",
                            "sales",
                        ]
                    ]
                )
            ),

            expected_groups=[
                [
                    "sales",
                    "support",
                ]
            ],

            allowed_dataset_ids=(
                ALLOWED
            ),
        )
    )


    assert (
        score.overall
        == 1.0
    )


    print(
        "Dataset order ignored: PASS"
    )


# ============================================================
# 3. REQUIREMENT ORDER DOES NOT MATTER
# ============================================================

def test_requirement_order_ignored() -> None:

    score = (
        score_dataset_dependency_candidate(
            candidate=(
                candidate(
                    [
                        [
                            "support",
                        ],

                        [
                            "sales",
                        ],
                    ]
                )
            ),

            expected_groups=[
                [
                    "sales",
                ],

                [
                    "support",
                ],
            ],

            allowed_dataset_ids=(
                ALLOWED
            ),
        )
    )


    assert (
        score.overall
        == 1.0
    )


    print(
        "Requirement order ignored: PASS"
    )


# ============================================================
# 4. REQUIREMENT IDS DO NOT MATTER
# ============================================================

def test_requirement_ids_ignored() -> None:

    model_candidate = (
        DatasetDependencyCandidate
        .model_validate(
            {
                "requirements": [
                    {
                        "requirement_id":
                            "anything_the_model_wants",

                        "dataset_ids": [
                            "sales",
                            "support",
                        ],
                    }
                ],
            }
        )
    )


    score = (
        score_dataset_dependency_candidate(
            candidate=(
                model_candidate
            ),

            expected_groups=[
                [
                    "sales",
                    "support",
                ]
            ],

            allowed_dataset_ids=(
                ALLOWED
            ),
        )
    )


    assert (
        score.overall
        == 1.0
    )


    print(
        "Requirement IDs ignored: PASS"
    )


# ============================================================
# 5. INDEPENDENT ANALYSES
# ============================================================

def test_independent_analyses() -> None:

    score = (
        score_dataset_dependency_candidate(
            candidate=(
                candidate(
                    [
                        [
                            "sales",
                        ],

                        [
                            "support",
                        ],
                    ]
                )
            ),

            expected_groups=[
                [
                    "sales",
                ],

                [
                    "support",
                ],
            ],

            allowed_dataset_ids=(
                ALLOWED
            ),
        )
    )


    assert (
        score.exact_groups
        == 1.0
    )


    assert (
        score.pairwise_grouping_f1
        == 1.0
    )


    assert (
        score.overall
        == 1.0
    )


    print(
        "Independent analyses grouping: PASS"
    )


# ============================================================
# 6. MERGING INDEPENDENT RESULTS IS WRONG
# ============================================================

def test_wrong_merge_penalized() -> None:

    score = (
        score_dataset_dependency_candidate(
            candidate=(
                candidate(
                    [
                        [
                            "sales",
                            "support",
                        ]
                    ]
                )
            ),

            expected_groups=[
                [
                    "sales",
                ],

                [
                    "support",
                ],
            ],

            allowed_dataset_ids=(
                ALLOWED
            ),
        )
    )


    # Same flat datasets.
    assert (
        score.dataset_f1
        == 1.0
    )


    # But grouping is wrong.
    assert (
        score.exact_groups
        == 0.0
    )


    assert (
        score.pairwise_grouping_f1
        == 0.0
    )


    assert (
        score.requirement_count
        == 0.0
    )


    assert (
        score.overall
        < 1.0
    )


    print(
        "Wrong dependency merge penalized: PASS"
    )


# ============================================================
# 7. SPLITTING A CROSS-DATASET RESULT IS WRONG
# ============================================================

def test_wrong_split_penalized() -> None:

    score = (
        score_dataset_dependency_candidate(
            candidate=(
                candidate(
                    [
                        [
                            "sales",
                        ],

                        [
                            "support",
                        ],
                    ]
                )
            ),

            expected_groups=[
                [
                    "sales",
                    "support",
                ]
            ],

            allowed_dataset_ids=(
                ALLOWED
            ),
        )
    )


    assert (
        score.dataset_f1
        == 1.0
    )


    assert (
        score.exact_groups
        == 0.0
    )


    assert (
        score.pairwise_grouping_f1
        == 0.0
    )


    assert (
        score.requirement_count
        == 0.0
    )


    print(
        "Wrong dependency split penalized: PASS"
    )


# ============================================================
# 8. MISSING DATASET
# ============================================================

def test_missing_dataset_detected() -> None:

    score = (
        score_dataset_dependency_candidate(
            candidate=(
                candidate(
                    [
                        [
                            "sales",
                        ]
                    ]
                )
            ),

            expected_groups=[
                [
                    "sales",
                    "support",
                ]
            ],

            allowed_dataset_ids=(
                ALLOWED
            ),
        )
    )


    assert (
        score.exact_groups
        == 0.0
    )


    assert (
        score.missing_dataset_ids
        == (
            "support",
        )
    )


    assert (
        score.dataset_f1
        < 1.0
    )


    print(
        "Missing dataset detected: PASS"
    )


# ============================================================
# 9. HALLUCINATED DATASET
# ============================================================

def test_hallucinated_dataset_detected() -> None:

    score = (
        score_dataset_dependency_candidate(
            candidate=(
                candidate(
                    [
                        [
                            "sales",
                            "invented_dataset",
                        ]
                    ]
                )
            ),

            expected_groups=[
                [
                    "sales",
                ]
            ],

            allowed_dataset_ids=(
                ALLOWED
            ),
        )
    )


    assert (
        score.exact_groups
        == 0.0
    )


    assert (
        score.hallucinated_dataset_ids
        == (
            "invented_dataset",
        )
    )


    assert (
        score.dataset_f1
        < 1.0
    )


    print(
        "Hallucinated dataset detected: PASS"
    )


# ============================================================
# 10. DUPLICATE REQUIREMENT GROUP
# ============================================================

def test_duplicate_requirement_group_penalized() -> None:

    model_candidate = (
        DatasetDependencyCandidate
        .model_validate(
            {
                "requirements": [

                    {
                        "requirement_id":
                            "first",

                        "dataset_ids": [
                            "sales",
                            "support",
                        ],
                    },

                    {
                        "requirement_id":
                            "second",

                        "dataset_ids": [
                            "support",
                            "sales",
                        ],
                    },
                ],
            }
        )
    )


    score = (
        score_dataset_dependency_candidate(
            candidate=(
                model_candidate
            ),

            expected_groups=[
                [
                    "sales",
                    "support",
                ]
            ],

            allowed_dataset_ids=(
                ALLOWED
            ),
        )
    )


    assert (
        score.exact_groups
        == 0.0
    )


    assert (
        score.requirement_count
        == 0.0
    )


    assert (
        score.duplicate_requirement_groups
        == (
            (
                "sales",
                "support",
            ),
        )
    )


    print(
        "Duplicate requirement group penalized: PASS"
    )


# ============================================================
# 11. INVALID EXPECTATION REJECTED
# ============================================================

def test_invalid_expectation_rejected() -> None:

    try:

        score_dataset_dependency_candidate(
            candidate=(
                candidate(
                    [
                        [
                            "sales",
                        ]
                    ]
                )
            ),

            expected_groups=[
                [
                    "unknown_expected_dataset",
                ]
            ],

            allowed_dataset_ids=(
                ALLOWED
            ),
        )


    except ValueError as error:

        assert (
            "unknown_expected_dataset"
            in str(
                error
            )
        )


        print(
            "Invalid benchmark expectation rejected: PASS"
        )


    else:

        raise AssertionError(
            "Invalid expected dataset IDs "
            "must be rejected."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS DATASET DEPENDENCY SCORER v0.8 ==="
    )


    print(
        "Scorer:",
        DATASET_DEPENDENCY_SCORER_VERSION,
    )


    print()


    test_exact_cross_dataset()

    test_dataset_order_ignored()

    test_requirement_order_ignored()

    test_requirement_ids_ignored()

    test_independent_analyses()

    test_wrong_merge_penalized()

    test_wrong_split_penalized()

    test_missing_dataset_detected()

    test_hallucinated_dataset_detected()

    test_duplicate_requirement_group_penalized()

    test_invalid_expectation_rejected()


    print()

    print(
        "Dataset dependency scorer v0.8: PASS"
    )


if __name__ == "__main__":
    main()