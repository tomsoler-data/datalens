from __future__ import annotations


import pandas as pd


from app.planning.ai_analytical_planner import (
    AIPlannerProposal,
    RawAIPlannerOutput,
    compatible_datasets_for_proposal,
    validate_ai_planner_output,
)

from app.planning.planner_catalog import (
    PLANNER_CATALOG_RULE_VERSION,
    planner_catalog_from_dataset_records,
)


CATEGORY_DATASET_ID = (
    "derived:combine_demo:category:category:gross_amount"
)

PRODUCT_DATASET_ID = (
    "derived:combine_demo:entity:product_id:gross_amount"
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
    category_frame = pd.DataFrame(
        {
            "category": [
                "A",
                "B",
                "C",
            ],
            "sum_gross_amount": [
                120.0,
                240.0,
                360.0,
            ],
            "event_count": [
                2,
                2,
                2,
            ],
        }
    )


    product_frame = pd.DataFrame(
        {
            "product_id": [
                "p1",
                "p2",
                "p3",
                "p4",
            ],
            "sum_gross_amount": [
                50.0,
                70.0,
                100.0,
                500.0,
            ],
            "event_count": [
                1,
                1,
                2,
                2,
            ],
            "category": [
                "A",
                "A",
                "B",
                "C",
            ],
            "brand": [
                "x",
                "y",
                "x",
                "z",
            ],
        }
    )


    return [
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
        {
            "dataset_id": PRODUCT_DATASET_ID,
            "filename": "sales__by_product_gross_amount.derived",
            "dataframe": product_frame,
            "is_derived": True,
            "derivation_type": "entity_additive_measure",
            "provenance": {
                "fact_dataset_id": "combine:demo",
                "operation": "groupby_sum",
                "entity_column": "product_id",
                "source_measure_column": "gross_amount",
                "target_measure_column": "sum_gross_amount",
                "aggregation": "sum",
                "grain": "product_id",
                "metric_semantics": METRIC_SEMANTICS,
                "source_measure_derivation": LINE_AMOUNT_DERIVATION,
            },
        },
    ]


def canonical_ca_proposal() -> AIPlannerProposal:
    return AIPlannerProposal(
        decision="propose",
        title="CA par catégorie",
        family="aggregation",
        dataset_id=CATEGORY_DATASET_ID,
        analytical_grain="category",
        x_column=None,
        y_column=None,
        group_column="category",
        value_column="sum_gross_amount",
        time_column=None,
        dimension_column=None,
        entity_column=None,
        aggregation_function="sum",
        ranking_order="none",
        ranking_limit=None,
        window_operation="none",
        window_size=None,
        blockers=[],
        reasons=[],
        confidence=1.0,
    )


def malformed_gemma_proposal() -> AIPlannerProposal:
    return AIPlannerProposal(
        decision="propose",
        title="Quantitative Association - Category Gross Amount",
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
        blockers=[],
        reasons=[
            "Observed small-model wire error reproduced by regression test."
        ],
        confidence=0.95,
    )


def main() -> None:
    catalog = planner_catalog_from_dataset_records(
        build_records()
    )


    category_profile = next(
        dataset
        for dataset
        in catalog.datasets
        if dataset.dataset_id == CATEGORY_DATASET_ID
    )


    assert category_profile.is_derived is True
    assert category_profile.analytical_grain == "category"
    assert category_profile.operation == "groupby_sum"
    assert category_profile.aggregation == "sum"
    assert category_profile.group_column == "category"
    assert category_profile.source_measure_column == "gross_amount"
    assert category_profile.target_measure_column == "sum_gross_amount"
    assert category_profile.source_measure_formula == "quantity * unit_price"
    assert "ca" in category_profile.measure_semantic_aliases
    assert "chiffre_affaires" in category_profile.measure_semantic_aliases


    print(
        "[PASS] planner catalog preserves server-owned analytical grain and measure provenance"
    )


    canonical = canonical_ca_proposal()


    compatible = compatible_datasets_for_proposal(
        proposal=canonical,
        catalog=catalog,
    )


    assert [
        dataset.dataset_id
        for dataset
        in compatible
    ] == [
        CATEGORY_DATASET_ID
    ]


    print(
        "[PASS] exact analytical grain removes the false category/product dataset ambiguity"
    )


    canonical_report = validate_ai_planner_output(
        objective="CA par catégorie",
        raw_output=RawAIPlannerOutput(
            proposals=[
                canonical
            ]
        ),
        catalog=catalog,
        model="deterministic-proof",
    )


    assert canonical_report.validated_count == 1
    assert canonical_report.ambiguous_count == 0
    assert canonical_report.rejected_count == 0


    canonical_item = canonical_report.items[0]


    assert canonical_item.validation_status == "validated"
    assert canonical_item.proposal.dataset_id == CATEGORY_DATASET_ID
    assert canonical_item.proposal.family == "aggregation"
    assert canonical_item.proposal.group_column == "category"
    assert canonical_item.proposal.value_column == "sum_gross_amount"


    print(
        "[PASS] canonical CA by category contract validates without metric ambiguity"
    )


    repaired_report = validate_ai_planner_output(
        objective="CA par catégorie",
        raw_output=RawAIPlannerOutput(
            proposals=[
                malformed_gemma_proposal()
            ]
        ),
        catalog=catalog,
        model="deterministic-wire-repair-proof",
    )


    assert repaired_report.validated_count == 1
    assert repaired_report.ambiguous_count == 0
    assert repaired_report.rejected_count == 0


    repaired_item = repaired_report.items[0]


    assert repaired_item.validation_status == "validated"
    assert repaired_item.proposal.dataset_id == CATEGORY_DATASET_ID
    assert repaired_item.proposal.family == "aggregation"
    assert repaired_item.proposal.analytical_grain == "category"
    assert repaired_item.proposal.group_column == "category"
    assert repaired_item.proposal.value_column == "sum_gross_amount"
    assert repaired_item.proposal.aggregation_function == "sum"
    assert repaired_item.normalizations


    print(
        "[PASS] observed Gemma dataset/family wire error is repaired only from exact columns plus grain"
    )


    vague_report = validate_ai_planner_output(
        objective="performance par catégorie",
        raw_output=RawAIPlannerOutput(
            proposals=[
                canonical
            ]
        ),
        catalog=catalog,
        model="deterministic-abstention-proof",
    )


    assert vague_report.validated_count == 0
    assert vague_report.ambiguous_count == 1


    print(
        "[PASS] vague business metric still abstains instead of trusting an arbitrary measure"
    )


    assert PLANNER_CATALOG_RULE_VERSION == "planner_catalog_v0.3"


    print()
    print("PASS - analytical planner catalog grain v0.1")


if __name__ == "__main__":
    main()
