from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from app.preparation.dataset_identity import (
    DATASET_IDENTITY_RULE_VERSION,
    is_technical_surrogate_column,
    profile_dataset_identity,
)

from app.preparation.preparation_combine_service import (
    MIN_AUTOMATIC_JOIN_MATCH_RATE,
    PREPARATION_COMBINE_SERVICE_VERSION,
    _candidate_seeds,
    _is_automatic_join_key,
)


# ============================================================
# HELPERS
# ============================================================


def artifact(
    dataset_id: str,
    filename: str,
    dataframe: pd.DataFrame,
):
    return SimpleNamespace(
        dataset_id=dataset_id,
        dataset_filename=filename,
        rows=int(
            len(
                dataframe
            )
        ),
    )


# ============================================================
# 1. PREFIX IDENTIFIER ID_PROD
# ============================================================


def test_id_prod_is_natural_identity() -> None:
    products = pd.DataFrame(
        {
            "id_prod": [
                "0_1",
                "0_2",
                "1_1",
            ],
            "price": [
                10.0,
                20.0,
                30.0,
            ],
            "categ": [
                0,
                0,
                1,
            ],
        }
    )

    report = profile_dataset_identity(
        products,
        dataset_id="dataset:products",
        dataset_filename="products.csv",
    )

    assert report.status == "single_key"
    assert report.preferred_candidate is not None
    assert report.preferred_candidate.columns == [
        "id_prod",
    ]
    assert report.surrogate_key_recommended is False
    assert report.suggested_surrogate_column is None

    print(
        "[PASS] id_prod recognized as natural identity"
    )


# ============================================================
# 2. TECHNICAL SURROGATE NEVER BECOMES JOIN KEY
# ============================================================


def test_surrogate_names_are_not_automatic_join_keys(
) -> None:
    for column_name in [
        "row_id",
        "datalens_row_id",
        "technical_row_id",
        "datalens_row_id_2",
        "datalens_row_id_99",
    ]:
        assert is_technical_surrogate_column(
            column_name
        ) is True

        assert _is_automatic_join_key(
            column_name
        ) is False

    assert _is_automatic_join_key(
        "client_id"
    ) is True

    assert _is_automatic_join_key(
        "id_prod"
    ) is True

    assert _is_automatic_join_key(
        "id"
    ) is False

    print(
        "[PASS] technical row ids excluded from join semantics"
    )


# ============================================================
# 3. LAPAGE-LIKE FIRST FRONTIER
# ============================================================


def test_lapage_first_frontier_uses_business_relationship(
) -> None:
    transactions = pd.DataFrame(
        {
            "row_id": [
                1,
                2,
                3,
                4,
            ],
            "id_prod": [
                "0_1",
                "0_2",
                "0_1",
                "1_1",
            ],
            "client_id": [
                "c_1",
                "c_1",
                "c_2",
                "c_3",
            ],
        }
    )

    customers = pd.DataFrame(
        {
            "client_id": [
                "c_1",
                "c_2",
                "c_3",
            ],
            "sex": [
                "f",
                "m",
                "f",
            ],
        }
    )

    products = pd.DataFrame(
        {
            "row_id": [
                1,
                2,
                3,
            ],
            "id_prod": [
                "0_1",
                "0_2",
                "1_1",
            ],
            "price": [
                10.0,
                20.0,
                30.0,
            ],
        }
    )

    artifacts = [
        artifact(
            "dataset:transactions",
            "Transactions.csv",
            transactions,
        ),
        artifact(
            "dataset:customers",
            "customers.csv",
            customers,
        ),
        artifact(
            "dataset:products",
            "products.csv",
            products,
        ),
    ]

    frames = {
        "dataset:transactions":
            transactions,

        "dataset:customers":
            customers,

        "dataset:products":
            products,
    }

    seeds = _candidate_seeds(
        artifacts=artifacts,
        frames=frames,
    )

    assert seeds

    keys = [
        seed.key_column
        for seed in seeds
    ]

    assert "row_id" not in keys
    assert "client_id" in keys
    assert "id_prod" in keys

    # Both real relationships have complete overlap. Existing
    # deterministic sorting keeps client_id first alphabetically.
    assert seeds[0].key_column == "client_id"
    assert seeds[0].expected_cardinality.value == "many_to_one"
    assert seeds[0].left_match_rate == 1.0

    print(
        "[PASS] Lapage first frontier ignores row_id"
    )


# ============================================================
# 4. LAPAGE-LIKE SECOND FRONTIER
# ============================================================


def test_lapage_second_frontier_uses_id_prod_not_row_id(
) -> None:
    combined_transactions_customers = pd.DataFrame(
        {
            "row_id": [
                1,
                2,
                3,
                4,
            ],
            "id_prod": [
                "0_1",
                "0_2",
                "0_1",
                "1_1",
            ],
            "client_id": [
                "c_1",
                "c_1",
                "c_2",
                "c_3",
            ],
            "sex": [
                "f",
                "f",
                "m",
                "f",
            ],
        }
    )

    # Deliberately include a technical row_id on products too.
    # The old bug would discover row_id as a shared *_id key.
    products = pd.DataFrame(
        {
            "row_id": [
                1,
                2,
                3,
            ],
            "id_prod": [
                "0_1",
                "0_2",
                "1_1",
            ],
            "price": [
                10.0,
                20.0,
                30.0,
            ],
        }
    )

    artifacts = [
        artifact(
            "combine:transactions_customers",
            "Transactions__customers.csv",
            combined_transactions_customers,
        ),
        artifact(
            "dataset:products",
            "products__row_id.csv",
            products,
        ),
    ]

    frames = {
        "combine:transactions_customers":
            combined_transactions_customers,

        "dataset:products":
            products,
    }

    seeds = _candidate_seeds(
        artifacts=artifacts,
        frames=frames,
    )

    assert len(
        seeds
    ) == 1

    seed = seeds[
        0
    ]

    assert seed.key_column == "id_prod"
    assert seed.expected_cardinality.value == "many_to_one"
    assert seed.left_match_rate == 1.0

    print(
        "[PASS] Lapage second frontier joins on id_prod"
    )


# ============================================================
# 5. PATHOLOGICAL LOW-OVERLAP SAME-NAME KEY
# ============================================================


def test_pathological_identifier_overlap_is_not_auto_proposed(
) -> None:
    left = pd.DataFrame(
        {
            "fake_id": [
                f"L{index}"
                for index in range(
                    100
                )
            ],
        }
    )

    right_values = [
        "L0",
        *[
            f"R{index}"
            for index in range(
                1,
                100
            )
        ],
    ]

    right = pd.DataFrame(
        {
            "fake_id":
                right_values,
        }
    )

    artifacts = [
        artifact(
            "dataset:left",
            "left.csv",
            left,
        ),
        artifact(
            "dataset:right",
            "right.csv",
            right,
        ),
    ]

    frames = {
        "dataset:left":
            left,

        "dataset:right":
            right,
    }

    seeds = _candidate_seeds(
        artifacts=artifacts,
        frames=frames,
    )

    assert MIN_AUTOMATIC_JOIN_MATCH_RATE == 0.05
    assert seeds == []

    print(
        "[PASS] pathological 1% key overlap rejected"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        "=== DATALENS LAPAGE JOIN GUARDRAILS v0.1 ==="
    )
    print()

    test_id_prod_is_natural_identity()
    test_surrogate_names_are_not_automatic_join_keys()
    test_lapage_first_frontier_uses_business_relationship()
    test_lapage_second_frontier_uses_id_prod_not_row_id()
    test_pathological_identifier_overlap_is_not_auto_proposed()

    print()
    print(
        "Dataset identity version:",
        DATASET_IDENTITY_RULE_VERSION,
    )
    print(
        "Combine service version:",
        PREPARATION_COMBINE_SERVICE_VERSION,
    )
    print(
        "PASS - Lapage join guardrails v0.1"
    )


if __name__ == "__main__":
    main()
