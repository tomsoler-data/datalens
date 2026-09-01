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


DATASET_ID = "combine:demo"


def build_records() -> list[dict[str, object]]:
    frame = pd.DataFrame(
        {
            "order_id": [
                "o1",
                "o2",
                "o3",
                "o4",
                "o5",
                "o6",
            ],
            "quantity": [
                1,
                2,
                3,
                1,
                4,
                2,
            ],
            "unit_price": [
                12.5,
                20.0,
                9.0,
                35.0,
                8.0,
                15.0,
            ],
            "country": [
                "France",
                "France",
                "Germany",
                "Italy",
                "Spain",
                "Spain",
            ],
        }
    )

    return [
        {
            "dataset_id": DATASET_ID,
            "filename": "orders.csv",
            "dataframe": frame,
            "is_derived": False,
        }
    ]


def observed_quantity_unit_price_failure() -> AIPlannerProposal:
    """
    Reproduce the live v0.23 wire failure:

    Gemma correctly identifies the quantitative_association family and the
    exact quantity/unit_price columns, but leaks aggregation_function="sum".
    The dataset spelling is also normalized from the observed underscore form.
    """

    return AIPlannerProposal(
        decision="propose",
        title="Quantitative Association",
        family="quantitative_association",
        dataset_id="combine_demo",
        analytical_grain="entity",
        x_column="quantity",
        y_column="unit_price",
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
            "Observed small-model wire failure reproduced from the live "
            "quantity versus unit-price trace."
        ],
        confidence=1.0,
    )


def invalid_mixed_type_proposal() -> AIPlannerProposal:
    """
    Structural normalization must not turn an invalid mixed-type association
    into a valid contract merely because an aggregation leak is present.
    """

    return AIPlannerProposal(
        decision="propose",
        title="Invalid mixed association",
        family="quantitative_association",
        dataset_id=DATASET_ID,
        analytical_grain="entity",
        x_column="country",
        y_column="unit_price",
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
            "Guardrail regression: categorical + quantitative must remain "
            "invalid for quantitative_association."
        ],
        confidence=0.9,
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

    profile = catalog.datasets[0]
    kinds = {
        column.name: column.analysis_kind
        for column in profile.columns
    }

    assert kinds["quantity"] == "quantitative"
    assert kinds["unit_price"] == "quantitative"

    print(
        "[PASS] catalog exposes quantity and unit_price as quantitative"
    )

    report = validate_ai_planner_output(
        objective=(
            "Existe-t-il une relation entre la quantité commandée "
            "et le prix unitaire ?"
        ),
        raw_output=RawAIPlannerOutput(
            proposals=[
                observed_quantity_unit_price_failure()
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
    assert item.proposal.family == "quantitative_association"
    assert item.proposal.dataset_id == DATASET_ID
    assert item.proposal.x_column == "quantity"
    assert item.proposal.y_column == "unit_price"
    assert item.proposal.aggregation_function == "none"
    assert item.proposal.value_column is None
    assert item.proposal.group_column is None
    assert item.proposal.time_column is None
    assert item.proposal.dimension_column is None
    assert item.proposal.entity_column is None

    assert any(
        "agrégation incompatible"
        in normalization
        for normalization in item.normalizations
    )

    assert item.contract is not None
    assert item.contract.status == "validated"
    assert item.contract.family == "quantitative_association"
    assert item.contract.aggregation is None

    binding_map = {
        binding.role: binding.column
        for binding in item.contract.bindings
    }

    assert binding_map == {
        "x": "quantity",
        "y": "unit_price",
    }

    print(
        "[PASS] observed quantity/unit-price aggregation leak is repaired to a validated quantitative association"
    )

    invalid = validate_ai_planner_output(
        objective="Analyse cette relation.",
        raw_output=RawAIPlannerOutput(
            proposals=[
                invalid_mixed_type_proposal()
            ]
        ),
        catalog=catalog,
        model="deterministic-proof",
    )

    assert invalid.validated_count == 0
    assert invalid.rejected_count == 1
    assert invalid.items[0].validation_status == "rejected"
    assert invalid.items[0].proposal.aggregation_function == "sum"

    print(
        "[PASS] invalid mixed-type association is not rescued by aggregation cleanup"
    )

    print()
    print("PASS - planner quantitative association wire v0.1")


if __name__ == "__main__":
    main()
