from __future__ import annotations

import pandas as pd

from app.preparation.dataset_identity import (
    DATASET_IDENTITY_RULE_VERSION,
    create_surrogate_row_key,
    profile_dataset_identity,
)


def test_single_identifier_key() -> None:
    dataframe = pd.DataFrame(
        {
            "order_id": [
                "O1",
                "O2",
                "O3",
                "O4",
            ],
            "customer_id": [
                "C1",
                "C1",
                "C2",
                "C3",
            ],
            "amount": [
                12.5,
                19.0,
                8.5,
                22.0,
            ],
        }
    )

    report = profile_dataset_identity(
        dataframe,
        dataset_id="orders",
        dataset_filename="orders.csv",
    )

    assert report.status == "single_key"
    assert report.preferred_candidate is not None
    assert report.preferred_candidate.columns == [
        "order_id",
    ]
    assert report.surrogate_key_recommended is False

    print(
        "Identifier-like unique key detected: PASS"
    )


def test_composite_identifier_key() -> None:
    dataframe = pd.DataFrame(
        {
            "order_id": [
                "O1",
                "O1",
                "O2",
                "O2",
            ],
            "line_id": [
                "L1",
                "L2",
                "L1",
                "L2",
            ],
            "amount": [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
        }
    )

    report = profile_dataset_identity(
        dataframe,
        dataset_id="order_lines",
        dataset_filename="order_lines.csv",
    )

    assert report.status == "composite_key"
    assert report.preferred_candidate is not None
    assert set(
        report.preferred_candidate.columns
    ) == {
        "order_id",
        "line_id",
    }
    assert report.preferred_candidate.unique is True
    assert report.surrogate_key_recommended is False

    print(
        "Simple composite identity detected: PASS"
    )


def test_unique_measure_does_not_become_identity_key(
) -> None:
    dataframe = pd.DataFrame(
        {
            "customer": [
                "A",
                "A",
                "B",
                "B",
            ],
            "amount": [
                10.01,
                20.02,
                30.03,
                40.04,
            ],
            "status": [
                "paid",
                "paid",
                "paid",
                "paid",
            ],
        }
    )

    report = profile_dataset_identity(
        dataframe,
        dataset_id="payments",
        dataset_filename="payments.csv",
    )

    assert "amount" in report.mechanically_unique_columns
    assert report.status == "surrogate_recommended"
    assert report.preferred_candidate is None
    assert report.surrogate_key_recommended is True
    assert report.suggested_surrogate_column == "row_id"

    print(
        "Accidentally unique measure rejected as identity key: PASS"
    )


def test_no_natural_key_recommends_surrogate(
) -> None:
    dataframe = pd.DataFrame(
        {
            "city": [
                "Paris",
                "Paris",
                "Lyon",
                "Lyon",
            ],
            "category": [
                "A",
                "A",
                "B",
                "B",
            ],
            "value": [
                10,
                10,
                20,
                20,
            ],
        }
    )

    report = profile_dataset_identity(
        dataframe,
        dataset_id="observations",
        dataset_filename="observations.csv",
    )

    assert report.status == "surrogate_recommended"
    assert report.surrogate_key_recommended is True
    assert report.suggested_surrogate_column == "row_id"

    print(
        "Missing natural identity recommends surrogate key: PASS"
    )


def test_surrogate_creation_is_non_mutating_and_unique(
) -> None:
    dataframe = pd.DataFrame(
        {
            "city": [
                "Paris",
                "Paris",
                "Lyon",
            ],
            "value": [
                10,
                10,
                20,
            ],
        }
    )

    original = dataframe.copy(
        deep=True
    )

    transformation = (
        create_surrogate_row_key(
            dataframe
        )
    )

    assert transformation.column_name == "row_id"
    assert transformation.dataframe.columns[0] == "row_id"
    assert transformation.dataframe[
        "row_id"
    ].tolist() == [
        1,
        2,
        3,
    ]
    assert transformation.dataframe[
        "row_id"
    ].is_unique
    assert (
        transformation.dataframe[
            "row_id"
        ]
        .isna()
        .sum()
        ==
        0
    )

    pd.testing.assert_frame_equal(
        dataframe,
        original,
    )

    print(
        "Surrogate row key creation is deterministic and non-mutating: PASS"
    )


def test_surrogate_name_collision() -> None:
    dataframe = pd.DataFrame(
        {
            "row_id": [
                None,
                None,
            ],
            "value": [
                1,
                1,
            ],
        }
    )

    report = profile_dataset_identity(
        dataframe,
        dataset_id="legacy_missing",
        dataset_filename="legacy_missing.csv",
    )

    assert (
        report.suggested_surrogate_column
        ==
        "datalens_row_id"
    )

    transformation = create_surrogate_row_key(
        dataframe
    )

    assert (
        transformation.column_name
        ==
        "datalens_row_id"
    )

    print(
        "Surrogate column collision handled safely: PASS"
    )


def test_surrogate_is_not_a_join_claim() -> None:
    dataframe = pd.DataFrame(
        {
            "name": [
                "A",
                "A",
                "B",
            ],
            "value": [
                1,
                1,
                2,
            ],
        }
    )

    report = profile_dataset_identity(
        dataframe,
        dataset_id="observations",
        dataset_filename="observations.csv",
    )

    assert any(
        "must not be treated as a join key"
        in reason
        for reason in report.reasons
    )

    print(
        "Surrogate identity separated from join semantics: PASS"
    )


def main() -> None:
    print(
        "=== DATALENS DATASET IDENTITY v0.1 ==="
    )

    print()

    test_single_identifier_key()
    test_composite_identifier_key()
    test_unique_measure_does_not_become_identity_key()
    test_no_natural_key_recommends_surrogate()
    test_surrogate_creation_is_non_mutating_and_unique()
    test_surrogate_name_collision()
    test_surrogate_is_not_a_join_claim()

    print()

    print(
        (
            "Dataset Identity rule version: "
            f"{DATASET_IDENTITY_RULE_VERSION}"
        )
    )

    print(
        "Dataset Identity v0.1: PASS"
    )


if __name__ == "__main__":
    main()
