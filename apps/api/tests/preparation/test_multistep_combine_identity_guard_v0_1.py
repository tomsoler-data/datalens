from __future__ import annotations


from types import (
    SimpleNamespace,
)

import pandas as pd


import app.preparation.preparation_combine_service as combine_service

from app.preparation.preparation_combine_service import (
    CombineIdentityResolutionRequiredError,
    PREPARATION_COMBINE_SERVICE_VERSION,
    _candidate_seeds,
    _require_identity_frontier_resolved,
    _requires_explicit_identity_resolution,
)


# ============================================================
# HELPERS
# ============================================================


def artifact(
    *,
    dataset_id: str,
    filename: str,
    stage: str,
    dataframe: pd.DataFrame,
):
    return SimpleNamespace(
        dataset_id=
            dataset_id,

        dataset_filename=
            filename,

        stage=
            stage,

        rows=
            int(
                len(
                    dataframe
                )
            ),
    )


# ============================================================
# 1. DERIVED COMBINE ARTIFACT DOES NOT REOPEN IDENTITY REVIEW
# ============================================================


def test_combine_artifact_does_not_require_new_identity_review(
) -> None:
    combined = pd.DataFrame(
        {
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


    derived = artifact(
        dataset_id=
            "combine:transactions_customers",

        filename=
            "Transactions__customers.csv",

        stage=
            "combine",

        dataframe=
            combined,
    )


    assert (
        _requires_explicit_identity_resolution(
            derived
        )
        is False
    )


    # No identity resolution exists for the derived output.
    # The gate must nevertheless accept it because it was already
    # produced by an approved + validated COMBINE operation.
    _require_identity_frontier_resolved(
        workflow_id=
            "prep:lapage",

        artifacts=[
            derived,
        ],

        frames={
            derived.dataset_id:
                combined,
        },
    )


    print(
        "[PASS] derived COMBINE artifact does not reopen identity review"
    )


# ============================================================
# 2. INDEPENDENT SOURCE STILL REQUIRES IDENTITY RESOLUTION
# ============================================================


def test_source_artifact_still_requires_identity_resolution(
) -> None:
    transactions = pd.DataFrame(
        {
            "id_prod": [
                "0_1",
                "0_2",
                "0_1",
            ],

            "client_id": [
                "c_1",
                "c_1",
                "c_2",
            ],
        }
    )


    source = artifact(
        dataset_id=
            "dataset:transactions",

        filename=
            "Transactions.csv",

        stage=
            "source",

        dataframe=
            transactions,
    )


    assert (
        _requires_explicit_identity_resolution(
            source
        )
        is True
    )


    original_get_resolution = (
        combine_service
        .get_current_identity_resolution
    )


    try:
        combine_service.get_current_identity_resolution = (
            lambda **_: None
        )


        try:
            _require_identity_frontier_resolved(
                workflow_id=
                    "prep:lapage",

                artifacts=[
                    source,
                ],

                frames={
                    source.dataset_id:
                        transactions,
                },
            )


        except CombineIdentityResolutionRequiredError as error:
            assert (
                "Transactions.csv"
                in
                str(
                    error
                )
            )


        else:
            raise AssertionError(
                (
                    "An independently prepared source artifact "
                    "must remain blocked until identity is resolved."
                )
            )


    finally:
        combine_service.get_current_identity_resolution = (
            original_get_resolution
        )


    print(
        "[PASS] independent source identity gate remains enforced"
    )


# ============================================================
# 3. SECOND LAPAGE JOIN REMAINS ID_PROD
# ============================================================


def test_second_lapage_join_is_id_prod(
) -> None:
    combined = pd.DataFrame(
        {
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


    combined_artifact = artifact(
        dataset_id=
            "combine:transactions_customers",

        filename=
            "Transactions__customers.csv",

        stage=
            "combine",

        dataframe=
            combined,
    )


    products_artifact = artifact(
        dataset_id=
            "dataset:products",

        filename=
            "products.csv",

        stage=
            "source",

        dataframe=
            products,
    )


    seeds = _candidate_seeds(
        artifacts=[
            combined_artifact,
            products_artifact,
        ],

        frames={
            combined_artifact.dataset_id:
                combined,

            products_artifact.dataset_id:
                products,
        },
    )


    assert (
        len(
            seeds
        )
        ==
        1
    )


    seed = seeds[
        0
    ]


    assert (
        seed.key_column
        ==
        "id_prod"
    )


    assert (
        seed.expected_cardinality.value
        ==
        "many_to_one"
    )


    assert (
        seed.left_match_rate
        ==
        1.0
    )


    print(
        "[PASS] second Lapage join remains id_prod many-to-one"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        "=== DATALENS MULTI-STEP COMBINE IDENTITY GUARD v0.1 ==="
    )

    print()


    test_combine_artifact_does_not_require_new_identity_review()

    test_source_artifact_still_requires_identity_resolution()

    test_second_lapage_join_is_id_prod()


    print()

    print(
        "Combine service version:",
        PREPARATION_COMBINE_SERVICE_VERSION,
    )

    print(
        "PASS - multi-step COMBINE identity guard v0.1"
    )


if __name__ == "__main__":
    main()
