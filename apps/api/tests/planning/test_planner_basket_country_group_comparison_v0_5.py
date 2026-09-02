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


COUNTRY_DATASET_ID = (
    "derived:combine_demo:category:country:gross_amount"
)

SESSION_DATASET_ID = (
    "derived:combine_demo:session:order_id:gross_amount"
)


LINE_AMOUNT_DERIVATION = {
    "operation": "analytical_line_amount_derivation",
    "derived_column": "gross_amount",
    "source_quantity_column": "quantity",
    "source_unit_price_column": "unit_price",
    "formula": "quantity * unit_price",
    "valid_count": 8,
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
    country_frame = pd.DataFrame(
        {
            "country": [
                "France",
                "Germany",
                "Italy",
                "Spain",
            ],
            "sum_gross_amount": [
                298.0,
                266.5,
                155.0,
                142.5,
            ],
            "event_count": [
                6,
                6,
                4,
                4,
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
                "o7",
                "o8",
            ],
            "basket_amount": [
                90.0,
                66.0,
                48.0,
                45.0,
                32.0,
                24.0,
                19.5,
                12.5,
            ],
            "item_count": [
                1,
                2,
                1,
                3,
                2,
                1,
                1,
                1,
            ],
            "country": [
                "France",
                "France",
                "Germany",
                "Germany",
                "Italy",
                "Italy",
                "Spain",
                "Spain",
            ],
            "segment": [
                "Consumer",
                "Consumer",
                "SMB",
                "SMB",
                "Enterprise",
                "Enterprise",
                "Consumer",
                "SMB",
            ],
        }
    )

    return [
        {
            "dataset_id": COUNTRY_DATASET_ID,
            "filename": "sales__by_country_gross_amount.derived",
            "dataframe": country_frame,
            "is_derived": True,
            "derivation_type": "categorical_additive_measure",
            "provenance": {
                "fact_dataset_id": "combine:demo",
                "operation": "groupby_sum",
                "group_column": "country",
                "source_measure_column": "gross_amount",
                "target_measure_column": "sum_gross_amount",
                "aggregation": "sum",
                "grain": "country",
                "metric_semantics": METRIC_SEMANTICS,
                "source_measure_derivation": LINE_AMOUNT_DERIVATION,
            },
        },
        {
            "dataset_id": SESSION_DATASET_ID,
            "filename": "sales__sessions_gross_amount.derived",
            "dataframe": session_frame,
            "is_derived": True,
            "derivation_type": "entity_additive_measure",
            "provenance": {
                "fact_dataset_id": "combine:demo",
                "operation": "session_materialization",
                "entity_column": "order_id",
                "source_measure_column": "gross_amount",
                "target_measure_column": "basket_amount",
                "aggregation": "sum",
                "grain": "order_id",
                "metric_semantics": METRIC_SEMANTICS,
                "source_measure_derivation": LINE_AMOUNT_DERIVATION,
            },
        },
    ]


def observed_basket_country_failure() -> AIPlannerProposal:
    """
    Reproduce the live v0.22 failure:

    Gemma finds basket_amount but emits a quantitative association with the
    same measure in x/y, the wrong grain, and a stray sum aggregation.
    """

    return AIPlannerProposal(
        decision="propose",
        title="Montant du panier selon le pays",
        family="quantitative_association",
        dataset_id="combine_demo",
        analytical_grain="basket_amount",
        x_column="basket_amount",
        y_column="basket_amount",
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
        blockers=["none"],
        reasons=[
            "Observed small-model wire failure reproduced from the live "
            "basket-by-country trace."
        ],
        confidence=0.95,
    )


def observed_ca_country_failure() -> AIPlannerProposal:
    return AIPlannerProposal(
        decision="propose",
        title="CA par pays",
        family="group_comparison",
        dataset_id="combine_demo",
        analytical_grain="country",
        x_column=None,
        y_column=None,
        group_column="country",
        value_column="basket_amount",
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
            "Regression case: revenue by country must remain an additive "
            "country aggregation, never a basket comparison."
        ],
        confidence=0.95,
    )


def main() -> None:
    catalog = planner_catalog_from_dataset_records(
        build_records()
    )

    assert PLANNER_CATALOG_RULE_VERSION == "planner_catalog_v0.3"
    assert (
        AI_ANALYTICAL_PLANNER_RULE_VERSION
        ==
        "ai_analytical_planner_v0.35"
    )

    session_profile = next(
        dataset
        for dataset in catalog.datasets
        if dataset.dataset_id == SESSION_DATASET_ID
    )

    assert session_profile.derivation_type == "entity_additive_measure"
    assert session_profile.operation == "session_materialization"
    assert session_profile.analytical_grain == "order_id"
    assert session_profile.target_measure_column == "basket_amount"

    print(
        "[PASS] catalog exposes the server-owned session basket view"
    )

    report = validate_ai_planner_output(
        objective=(
            "Comment le montant du panier varie-t-il selon le pays ?"
        ),
        raw_output=RawAIPlannerOutput(
            proposals=[
                observed_basket_country_failure()
            ]
        ),
        catalog=catalog,
        model="deterministic-proof",
    )

    assert report.validated_count == 1
    assert report.ambiguous_count == 0
    assert report.blocked_count == 0
    assert report.rejected_count == 0

    item = report.items[0]

    assert item.validation_status == "validated"
    assert item.proposal.family == "group_comparison"
    assert item.proposal.dataset_id == SESSION_DATASET_ID
    assert item.proposal.analytical_grain == "order_id"
    assert item.proposal.group_column == "country"
    assert item.proposal.value_column == "basket_amount"
    assert item.proposal.aggregation_function == "none"
    assert item.proposal.x_column is None
    assert item.proposal.y_column is None
    assert item.proposal.dimension_column is None
    assert item.proposal.time_column is None
    assert item.proposal.entity_column is None

    assert any(
        "entity-level server-owned"
        in message
        for message in item.normalizations
    )

    assert item.contract is not None
    assert item.contract.family == "group_comparison"

    binding_map = {
        binding.role: binding.column
        for binding in item.contract.bindings
    }

    assert binding_map["group"] == "country"
    assert binding_map["value"] == "basket_amount"

    print(
        "[PASS] observed basket-by-country wire failure is repaired to a validated group comparison"
    )

    segment_report = validate_ai_planner_output(
        objective=(
            "Comment le montant du panier varie-t-il selon le segment ?"
        ),
        raw_output=RawAIPlannerOutput(
            proposals=[
                observed_basket_country_failure()
            ]
        ),
        catalog=catalog,
        model="deterministic-proof",
    )

    assert segment_report.validated_count == 1
    segment_item = segment_report.items[0]
    assert segment_item.proposal.family == "group_comparison"
    assert segment_item.proposal.dataset_id == SESSION_DATASET_ID
    assert segment_item.proposal.group_column == "segment"
    assert segment_item.proposal.value_column == "basket_amount"

    print(
        "[PASS] basket group-comparison resolution generalizes from country to segment"
    )

    ca_report = validate_ai_planner_output(
        objective="CA par pays",
        raw_output=RawAIPlannerOutput(
            proposals=[
                observed_ca_country_failure()
            ]
        ),
        catalog=catalog,
        model="deterministic-proof",
    )

    assert ca_report.validated_count == 1
    ca_item = ca_report.items[0]
    assert ca_item.proposal.family == "aggregation"
    assert ca_item.proposal.dataset_id == COUNTRY_DATASET_ID
    assert ca_item.proposal.group_column == "country"
    assert ca_item.proposal.value_column == "sum_gross_amount"
    assert ca_item.proposal.aggregation_function == "sum"

    print(
        "[PASS] CA par pays remains an additive country aggregation, not a basket comparison"
    )

    vague_report = validate_ai_planner_output(
        objective=(
            "Comment la performance varie-t-elle selon le pays ?"
        ),
        raw_output=RawAIPlannerOutput(
            proposals=[
                observed_basket_country_failure()
            ]
        ),
        catalog=catalog,
        model="deterministic-proof",
    )

    assert vague_report.validated_count == 0
    assert not any(
        item.validation_status == "validated"
        and item.proposal.dataset_id == SESSION_DATASET_ID
        and item.proposal.value_column == "basket_amount"
        for item in vague_report.items
    )

    print(
        "[PASS] vague country performance is not silently rewritten to basket_amount"
    )

    print()
    print("PASS - planner basket country group comparison v0.1")


if __name__ == "__main__":
    main()
