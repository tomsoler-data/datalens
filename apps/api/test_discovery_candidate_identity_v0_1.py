from __future__ import annotations

import pandas as pd

from app.discovery.engine import (
    DISCOVERY_CANDIDATE_IDENTITY_RULE_VERSION,
    ColumnProfile,
    DatasetProfile,
    build_time_series_analysis_id,
    deduplicate_candidates,
    discover_time_series,
)


DATASET_ID = "dataset:test"


def column_profile(
    *,
    name: str,
    kind: str,
    semantic_role: str,
) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        kind=kind,
        analytical_subtype=None,
        semantic_role=semantic_role,
        concepts=set(),
        valid_count=40,
        missing_count=0,
        missing_ratio=0.0,
        unique_count=40,
        unique_ratio=1.0,
        numeric_variance=1.0 if kind == "quantitative" else None,
        numeric_skewness=0.0 if kind == "quantitative" else None,
    )


def build_profile(
) -> DatasetProfile:
    dataframe = pd.DataFrame(
        {
            "order_date": [
                f"2026-01-{(index % 20) + 1:02d}"
                for index
                in range(
                    40
                )
            ],

            "signup_date": [
                f"2025-12-{(index % 10) + 1:02d}"
                for index
                in range(
                    40
                )
            ],

            "quantity": [
                index % 5 + 1
                for index
                in range(
                    40
                )
            ],

            "unit_price": [
                10.0 + index
                for index
                in range(
                    40
                )
            ],
        }
    )

    columns = {
        "order_date":
            column_profile(
                name="order_date",
                kind="temporal",
                semantic_role="time",
            ),

        "signup_date":
            column_profile(
                name="signup_date",
                kind="temporal",
                semantic_role="time",
            ),

        "quantity":
            column_profile(
                name="quantity",
                kind="quantitative",
                semantic_role="measure",
            ),

        "unit_price":
            column_profile(
                name="unit_price",
                kind="quantitative",
                semantic_role="measure",
            ),
    }

    return DatasetProfile(
        dataset_id=DATASET_ID,
        filename="test.csv",
        dataframe=dataframe,
        columns=columns,
        temporal_columns=[
            "order_date",
            "signup_date",
        ],
        quantitative_columns=[
            "quantity",
            "unit_price",
        ],
        categorical_columns=[],
        entity_columns=[],
        geographic_columns=[],
        granularity_columns=[],
        repeated_measure_structure=None,
    )


def test_time_series_identity_includes_temporal_axis(
) -> None:
    first = (
        build_time_series_analysis_id(
            dataset_id=DATASET_ID,
            time_column="order_date",
            value_column="quantity",
        )
    )

    second = (
        build_time_series_analysis_id(
            dataset_id=DATASET_ID,
            time_column="signup_date",
            value_column="quantity",
        )
    )

    assert (
        first
        ==
        "dataset:test:time:order_date:quantity"
    )

    assert (
        second
        ==
        "dataset:test:time:signup_date:quantity"
    )

    assert first != second

    print(
        "Time-series identity includes temporal axis: PASS"
    )


def test_same_structural_analysis_has_stable_id(
) -> None:
    first = (
        build_time_series_analysis_id(
            dataset_id=DATASET_ID,
            time_column="Order Date",
            value_column="Unit Price",
        )
    )

    second = (
        build_time_series_analysis_id(
            dataset_id=DATASET_ID,
            time_column="Order Date",
            value_column="Unit Price",
        )
    )

    assert first == second

    assert (
        first
        ==
        "dataset:test:time:order_date:unit_price"
    )

    print(
        "Time-series candidate identity is deterministic: PASS"
    )


def test_discovery_generates_unique_ids_for_all_time_value_pairs(
) -> None:
    profile = (
        build_profile()
    )

    candidates = (
        discover_time_series(
            profile,
            objective=None,
        )
    )

    # 2 temporal columns × 2 quantitative columns.
    assert len(candidates) == 4

    analysis_ids = [
        candidate.analysis_id
        for candidate
        in candidates
    ]

    assert (
        len(
            set(
                analysis_ids
            )
        )
        ==
        4
    )

    assert set(
        analysis_ids
    ) == {
        "dataset:test:time:order_date:quantity",
        "dataset:test:time:order_date:unit_price",
        "dataset:test:time:signup_date:quantity",
        "dataset:test:time:signup_date:unit_price",
    }

    print(
        "Discovery gives every time/value pair a distinct analysis_id: PASS"
    )


def test_candidate_count_and_redundancy_contract_are_preserved(
) -> None:
    profile = (
        build_profile()
    )

    candidates = (
        discover_time_series(
            profile,
            objective=None,
        )
    )

    redundancy_keys = [
        candidate.redundancy_key
        for candidate
        in candidates
    ]

    assert (
        len(
            set(
                redundancy_keys
            )
        )
        ==
        4
    )

    deduplicated = (
        deduplicate_candidates(
            candidates
        )
    )

    assert len(deduplicated) == 4

    assert {
        candidate.redundancy_key
        for candidate
        in deduplicated
    } == {
        "time:dataset:test:order_date:quantity",
        "time:dataset:test:order_date:unit_price",
        "time:dataset:test:signup_date:quantity",
        "time:dataset:test:signup_date:unit_price",
    }

    print(
        "Candidate count and redundancy-key behavior remain unchanged: PASS"
    )


def test_identity_rule_version(
) -> None:
    assert (
        DISCOVERY_CANDIDATE_IDENTITY_RULE_VERSION
        ==
        "discovery_candidate_identity_v0.1"
    )

    print(
        "Discovery candidate identity rule version: PASS"
    )


def main() -> None:
    print(
        "=== DATALENS DISCOVERY CANDIDATE IDENTITY v0.1 ==="
    )

    print()

    test_time_series_identity_includes_temporal_axis()
    test_same_structural_analysis_has_stable_id()
    test_discovery_generates_unique_ids_for_all_time_value_pairs()
    test_candidate_count_and_redundancy_contract_are_preserved()
    test_identity_rule_version()

    print()

    print(
        "Discovery Candidate Identity v0.1: PASS"
    )


if __name__ == "__main__":
    main()
