from __future__ import annotations

import pandas as pd

from app.planning.ai_analytical_planner import (
    AI_ANALYTICAL_PLANNER_RULE_VERSION,
    AIPlannerProposal,
    RawAIPlannerOutput,
    validate_ai_planner_output,
)

from app.planning.planner_catalog import (
    PLANNER_CATALOG_RULE_VERSION,
    planner_catalog_from_dataset_records,
)


SOURCE_DATASET_ID = "combine:demo"
CATEGORY_REVENUE_DATASET_ID = (
    "derived:combine_demo:category:category:gross_amount"
)
SESSION_DATASET_ID = (
    "derived:combine_demo:session:order_id:gross_amount"
)


def build_records(
    *,
    duplicate_source: bool = False,
) -> list[dict[str, object]]:
    source_frame = pd.DataFrame(
        {
            "order_id": [
                "o1",
                "o2",
                "o3",
                "o4",
                "o5",
                "o6",
            ],
            "segment": [
                "Consumer",
                "SMB",
                "Entreprise",
                "Consumer",
                "SMB",
                "Entreprise",
            ],
            "category": [
                "Accessories",
                "Electronics",
                "Office",
                "Electronics",
                "Accessories",
                "Office",
            ],
            "quantity": [
                1,
                2,
                3,
                1,
                2,
                1,
            ],
            "unit_price": [
                10.0,
                20.0,
                15.0,
                30.0,
                12.0,
                18.0,
            ],
        }
    )

    category_frame = pd.DataFrame(
        {
            "category": [
                "Accessories",
                "Electronics",
                "Office",
            ],
            "sum_gross_amount": [
                34.0,
                70.0,
                63.0,
            ],
            "event_count": [
                2,
                2,
                2,
            ],
        }
    )

    session_frame = pd.DataFrame(
        {
            "order_id": [
                "o1",
                "o2",
                "o3",
                "o4",
                "o5",
                "o6",
            ],
            "segment": [
                "Consumer",
                "SMB",
                "Entreprise",
                "Consumer",
                "SMB",
                "Entreprise",
            ],
            "category": [
                "Accessories",
                "Electronics",
                "Office",
                "Electronics",
                "Accessories",
                "Office",
            ],
            "basket_amount": [
                10.0,
                40.0,
                45.0,
                30.0,
                24.0,
                18.0,
            ],
        }
    )

    records: list[dict[str, object]] = [
        {
            "dataset_id": SOURCE_DATASET_ID,
            "filename": "orders__customers__products.csv",
            "dataframe": source_frame,
            "is_derived": False,
        },
        {
            "dataset_id": SESSION_DATASET_ID,
            "filename": (
                "orders__customers__products__sessions_"
                "gross_amount.derived"
            ),
            "dataframe": session_frame,
            "is_derived": True,
            "derivation_type": "entity_additive_measure",
            "provenance": {
                "fact_dataset_id": SOURCE_DATASET_ID,
                "operation": "entity_rollup",
                "entity_column": "order_id",
                "source_measure_column": "gross_amount",
                "target_measure_column": "basket_amount",
                "aggregation": "sum",
                "grain": "order_id",
            },
        },
        {
            "dataset_id": CATEGORY_REVENUE_DATASET_ID,
            "filename": "sales__by_category_gross_amount.derived",
            "dataframe": category_frame,
            "is_derived": True,
            "derivation_type": "categorical_additive_measure",
            "provenance": {
                "fact_dataset_id": SOURCE_DATASET_ID,
                "operation": "groupby_sum",
                "group_column": "category",
                "source_measure_column": "gross_amount",
                "target_measure_column": "sum_gross_amount",
                "aggregation": "sum",
                "grain": "category",
                "metric_semantics": (
                    "Additive monetary measure derived at fact-row grain."
                ),
            },
        },
    ]

    if duplicate_source:
        records.append(
            {
                "dataset_id": "combine:duplicate",
                "filename": "duplicate_source.csv",
                "dataframe": source_frame.copy(),
                "is_derived": False,
            }
        )

    return records


def observed_segment_category_failure() -> AIPlannerProposal:
    """
    Reproduce the live v0.25 planner failure.

    Gemma answers a categorical/categorical relationship request with a
    quantitative association against the revenue-by-category derived view.
    """

    return AIPlannerProposal(
        decision="propose",
        title="Quantitative Association",
        family="quantitative_association",
        dataset_id="combine_demo",
        analytical_grain="category",
        x_column="category",
        y_column="sum_gross_amount",
        group_column=None,
        value_column=None,
        time_column=None,
        dimension_column=None,
        entity_column=None,
        aggregation_function="sum",
        ranking_order="none",
        ranking_limit=None,
        window_operation="none",
        window_size=None,
        benchmark_reference=None,
        benchmark_operator=None,
        benchmark_selection=None,
        blockers=[
            "no_time_series",
            "no_session",
            "no_customer",
            "no_product",
        ],
        reasons=[
            "Observed small-model categorical-association wire failure."
        ],
        confidence=1.0,
    )


def main() -> None:
    catalog = planner_catalog_from_dataset_records(
        build_records()
    )

    assert PLANNER_CATALOG_RULE_VERSION == "planner_catalog_v0.3"
    assert (
        AI_ANALYTICAL_PLANNER_RULE_VERSION
        ==
        "ai_analytical_planner_v0.34"
    )

    source = next(
        dataset
        for dataset in catalog.datasets
        if dataset.dataset_id == SOURCE_DATASET_ID
    )

    kinds = {
        column.name: column.analysis_kind
        for column in source.columns
    }

    assert kinds["segment"] == "categorical"
    assert kinds["category"] == "categorical"

    print(
        "[PASS] catalog exposes segment and category as categorical"
    )

    report = validate_ai_planner_output(
        objective=(
            "Existe-t-il une relation entre le segment client "
            "et la catégorie de produit ?"
        ),
        raw_output=RawAIPlannerOutput(
            proposals=[
                observed_segment_category_failure()
            ]
        ),
        catalog=catalog,
        model="deterministic-proof",
    )

    assert report.validated_count == 1
    assert report.ambiguous_count == 0
    assert report.blocked_count == 0
    assert report.rejected_count == 0

    print(
        "[PASS] derived session view with the same categorical "
        "columns does not override the deterministic source binding"
    )

    item = report.items[0]

    assert item.validation_status == "validated"
    assert item.proposal.family == "categorical_association"
    assert item.proposal.dataset_id == SOURCE_DATASET_ID
    assert item.proposal.x_column == "segment"
    assert item.proposal.y_column == "category"
    assert item.proposal.aggregation_function == "none"
    assert item.proposal.group_column is None
    assert item.proposal.value_column is None
    assert item.proposal.dimension_column is None
    assert item.proposal.time_column is None
    assert item.proposal.entity_column is None

    assert any(
        "association catégorielle explicitement formulée"
        in normalization
        for normalization in item.normalizations
    )

    assert item.contract is not None
    assert item.contract.status == "validated"
    assert item.contract.family == "categorical_association"
    assert item.contract.aggregation is None

    bindings = {
        binding.role: binding.column
        for binding in item.contract.bindings
    }

    assert bindings == {
        "x": "segment",
        "y": "category",
    }

    print(
        "[PASS] observed segment/category wire failure is repaired "
        "to a validated categorical association"
    )

    non_association = validate_ai_planner_output(
        objective="CA par catégorie",
        raw_output=RawAIPlannerOutput(
            proposals=[
                observed_segment_category_failure()
            ]
        ),
        catalog=catalog,
        model="deterministic-proof",
    )

    assert not (
        non_association.validated_count == 1
        and
        non_association.items[0].proposal.family
        ==
        "categorical_association"
    )

    print(
        "[PASS] non-association objective is not rewritten "
        "to categorical_association"
    )

    ambiguous_catalog = planner_catalog_from_dataset_records(
        build_records(
            duplicate_source=True,
        )
    )

    ambiguous = validate_ai_planner_output(
        objective=(
            "Existe-t-il une relation entre le segment client "
            "et la catégorie de produit ?"
        ),
        raw_output=RawAIPlannerOutput(
            proposals=[
                observed_segment_category_failure()
            ]
        ),
        catalog=ambiguous_catalog,
        model="deterministic-proof",
    )

    assert not (
        ambiguous.validated_count == 1
        and
        ambiguous.items[0].proposal.family
        ==
        "categorical_association"
        and
        ambiguous.items[0].proposal.dataset_id
        in {
            SOURCE_DATASET_ID,
            "combine:duplicate",
        }
    )

    print(
        "[PASS] ambiguous source datasets do not trigger "
        "objective-first categorical binding"
    )

    print()
    print("PASS - planner categorical association objective v0.2")


if __name__ == "__main__":
    main()
