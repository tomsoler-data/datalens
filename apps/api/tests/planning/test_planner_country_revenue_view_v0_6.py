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


def observed_ca_country_failure() -> AIPlannerProposal:
    """
    Reproduce the live failure observed for ``CA par pays``:

    Gemma binds the country dimension to the order/session view and
    uses basket_amount, which would answer a basket-distribution question
    rather than total revenue by country.
    """

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
            "Observed small-model session-view binding failure reproduced "
            "from the live CA-par-pays trace."
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
        "ai_analytical_planner_v0.34"
    )

    country_profile = next(
        dataset
        for dataset in catalog.datasets
        if dataset.dataset_id == COUNTRY_DATASET_ID
    )

    session_profile = next(
        dataset
        for dataset in catalog.datasets
        if dataset.dataset_id == SESSION_DATASET_ID
    )

    assert country_profile.derivation_type == "categorical_additive_measure"
    assert country_profile.group_column == "country"
    assert country_profile.target_measure_column == "sum_gross_amount"
    assert session_profile.derivation_type == "entity_additive_measure"
    assert session_profile.target_measure_column == "basket_amount"

    print(
        "[PASS] catalog contains both country revenue and session basket views"
    )

    report = validate_ai_planner_output(
        objective="CA par pays",
        raw_output=RawAIPlannerOutput(
            proposals=[
                observed_ca_country_failure()
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
    assert item.proposal.family == "aggregation"
    assert item.proposal.dataset_id == COUNTRY_DATASET_ID
    assert item.proposal.analytical_grain == "country"
    assert item.proposal.group_column == "country"
    assert item.proposal.value_column == "sum_gross_amount"
    assert item.proposal.aggregation_function == "sum"
    assert item.proposal.x_column is None
    assert item.proposal.y_column is None
    assert item.proposal.dimension_column is None
    assert item.proposal.time_column is None
    assert item.proposal.entity_column is None

    assert any(
        "vue catégorielle additive server-owned"
        in normalization
        for normalization in item.normalizations
    )

    assert item.contract is not None
    assert item.contract.status == "validated"
    assert item.contract.family == "aggregation"
    assert item.contract.required_dataset_ids == [
        COUNTRY_DATASET_ID
    ]

    binding_map = {
        binding.role: binding.column
        for binding in item.contract.bindings
    }

    assert binding_map == {
        "group": "country",
        "value": "sum_gross_amount",
    }

    assert item.contract.aggregation is not None
    assert item.contract.aggregation.function == "sum"

    print(
        "[PASS] CA par pays overrides the wrong session basket binding with the unique additive country view"
    )

    vague = validate_ai_planner_output(
        objective="Performance par pays",
        raw_output=RawAIPlannerOutput(
            proposals=[
                observed_ca_country_failure()
            ]
        ),
        catalog=catalog,
        model="deterministic-proof",
    )

    vague_item = vague.items[0]

    assert not (
        vague_item.proposal.dataset_id == COUNTRY_DATASET_ID
        and
        vague_item.proposal.value_column == "sum_gross_amount"
    )

    print(
        "[PASS] vague country performance is not silently rewritten to revenue"
    )

    print()
    print("PASS - planner country revenue view v0.1")


if __name__ == "__main__":
    main()
