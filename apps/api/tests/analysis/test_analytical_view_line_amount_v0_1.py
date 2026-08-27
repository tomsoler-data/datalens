from __future__ import annotations


import pandas as pd


from app.analysis.analytical_views import (
    ANALYTICAL_VIEW_RULE_VERSION,
    build_analytical_views,
)


# ============================================================
# FIXTURES
# ============================================================

def build_sales_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [
                "o1",
                "o1",
                "o2",
                "o3",
                "o3",
                "o4",
                "o5",
                "o6",
            ],
            "customer_id": [
                "c1",
                "c1",
                "c1",
                "c2",
                "c2",
                "c3",
                "c3",
                "c4",
            ],
            "product_id": [
                "p1",
                "p2",
                "p3",
                "p1",
                "p2",
                "p3",
                "p1",
                "p2",
            ],
            "order_date": [
                "2026-01-02",
                "2026-01-02",
                "2026-01-10",
                "2026-01-15",
                "2026-01-15",
                "2026-02-03",
                "2026-02-09",
                "2026-02-20",
            ],
            "quantity": [
                2,
                1,
                3,
                1,
                2,
                4,
                2,
                1,
            ],
            "unit_price": [
                10.0,
                20.0,
                5.0,
                10.0,
                20.0,
                5.0,
                10.0,
                20.0,
            ],
            "category": [
                "A",
                "B",
                "C",
                "A",
                "B",
                "C",
                "A",
                "B",
            ],
        }
    )


def build_prepared_record(
    dataframe: pd.DataFrame,
) -> dict[str, object]:
    return {
        "dataset_id":
            "combine:test-line-amount",
        "filename":
            "orders__customers__products.csv",
        "extension":
            ".csv",
        "dataframe":
            dataframe,
        "preparation_stage":
            "combine",
        "preparation_workflow_id":
            "prep:test-line-amount",
        "preparation_parent_dataset_ids": [
            "dataset:orders",
            "dataset:customers",
            "dataset:products",
        ],
        "preparation_evidence_refs": [
            "test:validated-combine",
        ],
        "analysis_input_rule_version":
            "analysis_input_handoff_v0.1",
    }


def provenance_operation(
    dataset: dict[str, object],
) -> str:
    provenance = dataset.get(
        "provenance"
    )

    if not isinstance(
        provenance,
        dict,
    ):
        return ""

    return str(
        provenance.get(
            "operation",
            "",
        )
    )


# ============================================================
# TESTS
# ============================================================

def test_strict_quantity_unit_price_derives_internal_gross_amount() -> None:
    dataframe = build_sales_frame()

    expected_total = float(
        (
            dataframe[
                "quantity"
            ]
            *
            dataframe[
                "unit_price"
            ]
        ).sum()
    )

    result = build_analytical_views(
        [
            build_prepared_record(
                dataframe
            )
        ],
        include_requested_context=False,
    )

    # The validated Preparation output must remain unchanged.
    assert (
        "gross_amount"
        not in dataframe.columns
    )

    category_views = [
        dataset
        for dataset
        in result.derived_datasets
        if (
            dataset.get(
                "derivation_type"
            )
            ==
            "categorical_additive_measure"
            and
            provenance_operation(
                dataset
            )
            ==
            "groupby_sum"
            and
            (
                dataset.get(
                    "provenance"
                )
                or {}
            ).get(
                "grain"
            )
            ==
            "category"
            and
            (
                dataset.get(
                    "provenance"
                )
                or {}
            ).get(
                "source_measure_column"
            )
            ==
            "gross_amount"
        )
    ]

    assert len(
        category_views
    ) == 1

    category_frame = category_views[
        0
    ][
        "dataframe"
    ]

    assert isinstance(
        category_frame,
        pd.DataFrame,
    )

    assert {
        "category",
        "sum_gross_amount",
        "event_count",
    }.issubset(
        set(
            category_frame.columns
        )
    )

    assert float(
        category_frame[
            "sum_gross_amount"
        ].sum()
    ) == expected_total

    lineage = (
        category_views[
            0
        ]
        .get(
            "provenance",
            {},
        )
        .get(
            "source_measure_derivation"
        )
    )

    assert isinstance(
        lineage,
        dict,
    )

    assert lineage.get(
        "operation"
    ) == (
        "analytical_line_amount_derivation"
    )

    assert lineage.get(
        "formula"
    ) == (
        "quantity * unit_price"
    )

    assert lineage.get(
        "analytical_only"
    ) is True

    customer_views = [
        dataset
        for dataset
        in result.derived_datasets
        if provenance_operation(
            dataset
        )
        ==
        "customer_behavior_materialization"
    ]

    assert len(
        customer_views
    ) == 1

    customer_frame = customer_views[
        0
    ][
        "dataframe"
    ]

    assert isinstance(
        customer_frame,
        pd.DataFrame,
    )

    assert {
        "customer_id",
        "total_spend",
        "purchase_sessions",
        "average_basket",
        "median_basket",
    }.issubset(
        set(
            customer_frame.columns
        )
    )

    assert float(
        customer_frame[
            "total_spend"
        ].sum()
    ) == expected_total

    assert (
        ANALYTICAL_VIEW_RULE_VERSION
        ==
        "analytical_view_v0.6"
    )


def test_ambiguous_quantity_pair_abstains() -> None:
    dataframe = build_sales_frame()

    dataframe[
        "qty"
    ] = dataframe[
        "quantity"
    ]

    result = build_analytical_views(
        [
            build_prepared_record(
                dataframe
            )
        ],
        include_requested_context=False,
    )

    gross_amount_views = [
        dataset
        for dataset
        in result.derived_datasets
        if (
            (
                dataset.get(
                    "provenance"
                )
                or {}
            ).get(
                "source_measure_column"
            )
            ==
            "gross_amount"
        )
    ]

    assert gross_amount_views == []


def test_existing_revenue_wins_over_internal_line_amount() -> None:
    dataframe = build_sales_frame()

    dataframe[
        "revenue"
    ] = (
        dataframe[
            "quantity"
        ]
        *
        dataframe[
            "unit_price"
        ]
    )

    result = build_analytical_views(
        [
            build_prepared_record(
                dataframe
            )
        ],
        include_requested_context=False,
    )

    source_measures = {
        str(
            (
                dataset.get(
                    "provenance"
                )
                or {}
            ).get(
                "source_measure_column",
                "",
            )
        )
        for dataset
        in result.derived_datasets
    }

    assert "revenue" in source_measures
    assert "gross_amount" not in source_measures


if __name__ == "__main__":
    test_strict_quantity_unit_price_derives_internal_gross_amount()
    print(
        "[PASS] strict quantity × unit_price creates analytical gross_amount"
    )

    test_ambiguous_quantity_pair_abstains()
    print(
        "[PASS] ambiguous quantity candidates cause safe abstention"
    )

    test_existing_revenue_wins_over_internal_line_amount()
    print(
        "[PASS] existing revenue prevents competing gross_amount derivation"
    )

    print()
    print(
        "PASS - analytical line monetary amount v0.1"
    )
