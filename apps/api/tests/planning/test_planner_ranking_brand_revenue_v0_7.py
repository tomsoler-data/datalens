from __future__ import annotations

import pandas as pd

from app.planning.ai_analytical_planner import (
    AI_ANALYTICAL_PLANNER_RULE_VERSION,
    AIPlannerProposal,
    RawAIPlannerOutput,
    objective_schema_column_mentions,
    validate_ai_planner_output,
)

from app.planning.planner_catalog import (
    PLANNER_CATALOG_RULE_VERSION,
    planner_catalog_from_dataset_records,
)


BRAND_DATASET_ID = (
    "derived:combine_demo:category:brand:gross_amount"
)

COUNTRY_DATASET_ID = (
    "derived:combine_demo:category:country:gross_amount"
)


LINE_AMOUNT_DERIVATION = {
    "operation": "analytical_line_amount_derivation",
    "derived_column": "gross_amount",
    "source_quantity_column": "quantity",
    "source_unit_price_column": "unit_price",
    "formula": "quantity * unit_price",
    "valid_count": 20,
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


def categorical_view_record(
    *,
    dataset_id: str,
    filename: str,
    group_column: str,
    values: list[str],
    revenue: list[float],
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "filename": filename,
        "dataframe": pd.DataFrame(
            {
                group_column: values,
                "sum_gross_amount": revenue,
                "event_count": [1 for _ in values],
            }
        ),
        "is_derived": True,
        "derivation_type": "categorical_additive_measure",
        "provenance": {
            "fact_dataset_id": "combine:demo",
            "operation": "groupby_sum",
            "group_column": group_column,
            "source_measure_column": "gross_amount",
            "target_measure_column": "sum_gross_amount",
            "aggregation": "sum",
            "grain": group_column,
            "metric_semantics": METRIC_SEMANTICS,
            "source_measure_derivation": LINE_AMOUNT_DERIVATION,
        },
    }


def build_records() -> list[dict[str, object]]:
    return [
        categorical_view_record(
            dataset_id=BRAND_DATASET_ID,
            filename="sales__by_brand_gross_amount.derived",
            group_column="brand",
            values=["Acme", "Nova", "Orbit"],
            revenue=[420.0, 315.0, 127.0],
        ),
        categorical_view_record(
            dataset_id=COUNTRY_DATASET_ID,
            filename="sales__by_country_gross_amount.derived",
            group_column="country",
            values=["France", "Spain", "Italy"],
            revenue=[360.0, 285.0, 217.0],
        ),
    ]


def observed_malformed_gemma_proposal() -> AIPlannerProposal:
    """Reproduce the live Top-3-brands failure from Gemma."""

    return AIPlannerProposal(
        decision="propose",
        title="Top brands by revenue",
        family="quantitative_association",
        dataset_id="sales__by_brand_gross_amount",
        analytical_grain="brand",
        x_column="brand",
        y_column="sum_gross_amount",
        group_column=None,
        value_column=None,
        time_column=None,
        dimension_column=None,
        entity_column=None,
        aggregation_function="sum",
        ranking_order="none",
        ranking_limit=1,
        window_operation="none",
        window_size=None,
        benchmark_reference=None,
        benchmark_operator=None,
        benchmark_selection=None,
        blockers=[],
        reasons=[
            "Observed small-model wire error reproduced from the live trace."
        ],
        confidence=0.9,
    )


def main() -> None:
    catalog = planner_catalog_from_dataset_records(
        build_records()
    )

    assert PLANNER_CATALOG_RULE_VERSION == "planner_catalog_v0.3"
    assert AI_ANALYTICAL_PLANNER_RULE_VERSION == "ai_analytical_planner_v0.35"

    brand_profile = next(
        dataset
        for dataset in catalog.datasets
        if dataset.dataset_id == BRAND_DATASET_ID
    )

    brand_mentions = set(
        objective_schema_column_mentions(
            objective="Top 3 marques par chiffre d’affaires",
            dataset=brand_profile,
        )
    )

    assert {
        "brand",
        "sum_gross_amount",
    }.issubset(brand_mentions)

    print(
        "[PASS] French brand wording resolves to the server-owned brand view"
    )

    country_profile = next(
        dataset
        for dataset in catalog.datasets
        if dataset.dataset_id == COUNTRY_DATASET_ID
    )

    country_mentions = set(
        objective_schema_column_mentions(
            objective="CA par pays",
            dataset=country_profile,
        )
    )

    assert {
        "country",
        "sum_gross_amount",
    }.issubset(country_mentions)

    print(
        "[PASS] French country wording resolves to the server-owned country view"
    )

    report = validate_ai_planner_output(
        objective="Top 3 marques par chiffre d’affaires",
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
    assert report.normalization_count >= 2

    item = report.items[0]

    assert item.validation_status == "validated"
    assert item.proposal.family == "ranking"
    assert item.proposal.dataset_id == BRAND_DATASET_ID
    assert item.proposal.analytical_grain == "brand"
    assert item.proposal.dimension_column == "brand"
    assert item.proposal.value_column == "sum_gross_amount"
    assert item.proposal.aggregation_function == "sum"
    assert item.proposal.ranking_order == "descending"
    assert item.proposal.ranking_limit == 3
    assert item.proposal.x_column is None
    assert item.proposal.y_column is None
    assert item.proposal.group_column is None

    assert item.contract is not None
    assert item.contract.status == "validated"
    assert item.contract.family == "ranking"
    assert item.contract.required_dataset_ids == [
        BRAND_DATASET_ID
    ]

    binding_map = {
        binding.role: binding.column
        for binding in item.contract.bindings
    }

    assert binding_map == {
        "dimension": "brand",
        "value": "sum_gross_amount",
    }

    assert item.contract.aggregation is not None
    assert item.contract.aggregation.function == "sum"
    assert item.contract.ranking is not None
    assert item.contract.ranking.order == "descending"
    assert item.contract.ranking.limit == 3

    print(
        "[PASS] observed Gemma Top-3-brands wire failure is repaired to a validated ranking contract"
    )

    vague_report = validate_ai_planner_output(
        objective="Top 3 marques par performance",
        raw_output=RawAIPlannerOutput(
            proposals=[
                observed_malformed_gemma_proposal()
            ]
        ),
        catalog=catalog,
        model="deterministic-proof",
    )

    assert vague_report.validated_count == 0
    assert (
        vague_report.ambiguous_count > 0
        or vague_report.rejected_count > 0
    )

    print(
        "[PASS] vague ranking metric is not silently rewritten to revenue"
    )

    print()
    print("PASS - planner brand revenue ranking v0.2")


if __name__ == "__main__":
    main()
