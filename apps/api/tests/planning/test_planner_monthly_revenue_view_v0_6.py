from __future__ import annotations

import pandas as pd

from app.planning.ai_analytical_planner import (
    AIPlannerProposal,
    RawAIPlannerOutput,
    objective_schema_column_mentions,
    validate_ai_planner_output,
)

from app.planning.planner_catalog import (
    PLANNER_CATALOG_RULE_VERSION,
    planner_catalog_from_dataset_records,
)


MONTHLY_DATASET_ID = (
    "derived:combine_demo:monthly:order_date:gross_amount"
)

CATEGORY_DATASET_ID = (
    "derived:combine_demo:category:category:gross_amount"
)


LINE_AMOUNT_DERIVATION = {
    "operation": "analytical_line_amount_derivation",
    "derived_column": "gross_amount",
    "source_quantity_column": "quantity",
    "source_unit_price_column": "unit_price",
    "formula": "quantity * unit_price",
    "valid_count": 6,
    "missing_count": 0,
    "analytical_only": True,
    "safety_policy": (
        "Derived only from exactly one strict quantity column and "
        "exactly one strict unit-price column."
    ),
}


METRIC_SEMANTICS = (
    "The monetary measure was derived internally at fact-row grain "
    "from one unambiguous strict quantity × unit-price pair."
)


def build_records() -> list[dict[str, object]]:
    monthly_frame = pd.DataFrame(
        {
            "month": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-02-01",
                ]
            ),
            "sum_gross_amount": [
                400.0,
                462.0,
            ],
            "event_count": [
                10,
                10,
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
                293.0,
                333.0,
                236.0,
            ],
            "event_count": [
                6,
                8,
                6,
            ],
        }
    )

    return [
        {
            "dataset_id": MONTHLY_DATASET_ID,
            "filename": "sales__monthly_gross_amount.derived",
            "dataframe": monthly_frame,
            "is_derived": True,
            "derivation_type": "monthly_additive_measure",
            "provenance": {
                "fact_dataset_id": "combine:demo",
                "operation": "groupby_sum",
                "source_time_column": "order_date",
                "source_measure_column": "gross_amount",
                "target_time_column": "month",
                "target_measure_column": "sum_gross_amount",
                "aggregation": "sum",
                "grain": "month",
                "metric_semantics": METRIC_SEMANTICS,
                "source_measure_derivation": LINE_AMOUNT_DERIVATION,
            },
        },
        {
            "dataset_id": CATEGORY_DATASET_ID,
            "filename": "sales__by_category_gross_amount.derived",
            "dataframe": category_frame,
            "is_derived": True,
            "derivation_type": "categorical_additive_measure",
            "provenance": {
                "fact_dataset_id": "combine:demo",
                "operation": "groupby_sum",
                "group_column": "category",
                "source_measure_column": "gross_amount",
                "target_measure_column": "sum_gross_amount",
                "aggregation": "sum",
                "grain": "category",
                "metric_semantics": METRIC_SEMANTICS,
                "source_measure_derivation": LINE_AMOUNT_DERIVATION,
            },
        },
    ]


def observed_malformed_gemma_proposal() -> AIPlannerProposal:
    """Reproduce the exact family/role failure observed in the live trace."""

    return AIPlannerProposal(
        decision="propose",
        title="family=group_comparison",
        family="group_comparison",
        dataset_id="combine_demo",
        analytical_grain="month",
        x_column="category",
        y_column="total_spend",
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
        blockers=[],
        reasons=[
            (
                "Observed small-model wire error reproduced from the "
                "monthly-revenue live trace."
            )
        ],
        confidence=0.9,
    )


def main() -> None:
    catalog = planner_catalog_from_dataset_records(
        build_records()
    )

    monthly_profile = next(
        dataset
        for dataset in catalog.datasets
        if dataset.dataset_id == MONTHLY_DATASET_ID
    )

    assert PLANNER_CATALOG_RULE_VERSION == "planner_catalog_v0.3"
    assert monthly_profile.is_derived is True
    assert monthly_profile.derivation_type == "monthly_additive_measure"
    assert monthly_profile.analytical_grain == "month"
    assert monthly_profile.operation == "groupby_sum"
    assert monthly_profile.aggregation == "sum"
    assert monthly_profile.source_time_column == "order_date"
    assert monthly_profile.target_time_column == "month"
    assert monthly_profile.source_measure_column == "gross_amount"
    assert monthly_profile.target_measure_column == "sum_gross_amount"
    assert "chiffre_affaires" in monthly_profile.measure_semantic_aliases
    assert "ca" in monthly_profile.measure_semantic_aliases

    print(
        "[PASS] planner catalog preserves monthly time and measure provenance"
    )

    objective = "Évolution mensuelle du chiffre d’affaires"

    mentions = set(
        objective_schema_column_mentions(
            objective=objective,
            dataset=monthly_profile,
        )
    )

    assert {
        "month",
        "sum_gross_amount",
    }.issubset(mentions)

    print(
        "[PASS] monthly cadence and revenue semantics resolve to the materialized view columns"
    )

    report = validate_ai_planner_output(
        objective=objective,
        raw_output=RawAIPlannerOutput(
            proposals=[
                observed_malformed_gemma_proposal()
            ]
        ),
        catalog=catalog,
        model="deterministic-proof",
    )

    assert report.validated_count == 1
    assert report.ambiguous_count == 0
    assert report.blocked_count == 0
    assert report.rejected_count == 0
    assert report.normalization_count >= 1

    item = report.items[0]

    assert item.validation_status == "validated"
    assert item.proposal.family == "time_series"
    assert item.proposal.dataset_id == MONTHLY_DATASET_ID
    assert item.proposal.analytical_grain == "month"
    assert item.proposal.time_column == "month"
    assert item.proposal.value_column == "sum_gross_amount"
    assert item.proposal.aggregation_function == "sum"
    assert item.proposal.x_column is None
    assert item.proposal.y_column is None
    assert item.proposal.group_column is None

    assert item.contract is not None
    assert item.contract.status == "validated"
    assert item.contract.family == "time_series"
    assert item.contract.required_dataset_ids == [
        MONTHLY_DATASET_ID
    ]

    binding_map = {
        binding.role: binding.column
        for binding in item.contract.bindings
    }

    assert binding_map == {
        "time": "month",
        "value": "sum_gross_amount",
    }

    print(
        "[PASS] observed Gemma monthly-revenue wire failure is repaired to a validated time-series contract"
    )

    vague_report = validate_ai_planner_output(
        objective="Évolution mensuelle de la performance",
        raw_output=RawAIPlannerOutput(
            proposals=[
                observed_malformed_gemma_proposal()
            ]
        ),
        catalog=catalog,
        model="deterministic-proof",
    )

    assert vague_report.validated_count == 0

    print(
        "[PASS] vague monthly business metric is not silently rewritten to revenue"
    )

    category_report = validate_ai_planner_output(
        objective="CA par catégorie",
        raw_output=RawAIPlannerOutput(
            proposals=[
                AIPlannerProposal(
                    decision="propose",
                    title="CA par catégorie",
                    family="quantitative_association",
                    dataset_id="combine_demo:category:gross_amount",
                    analytical_grain="category",
                    x_column="category",
                    y_column="sum_gross_amount",
                    group_column=None,
                    value_column=None,
                    time_column=None,
                    dimension_column=None,
                    entity_column=None,
                    aggregation_function="sum",
                    ranking_order="descending",
                    ranking_limit=None,
                    window_operation="none",
                    window_size=None,
                    benchmark_reference=None,
                    benchmark_operator=None,
                    benchmark_selection=None,
                    blockers=[],
                    reasons=[],
                    confidence=0.95,
                )
            ]
        ),
        catalog=catalog,
        model="deterministic-proof",
    )

    assert category_report.validated_count == 1
    assert category_report.items[0].proposal.family == "aggregation"
    assert category_report.items[0].proposal.dataset_id == CATEGORY_DATASET_ID

    print(
        "[PASS] existing CA-by-category deterministic repair remains valid"
    )

    print()
    print("PASS - planner monthly revenue view v0.1")


if __name__ == "__main__":
    main()
