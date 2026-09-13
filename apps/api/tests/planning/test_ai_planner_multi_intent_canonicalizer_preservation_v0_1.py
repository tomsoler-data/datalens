from __future__ import annotations


from app.planning.ai_analytical_planner import (
    AIPlannerProposal,
    RawAIPlannerOutput,
    validate_ai_planner_output,
)

from tests.planning.test_objective_coverage_multi_intent_same_metric_v0_1 import (
    OBJECTIVE,
    build_catalog,
)


def proposal(
    *,
    title: str,
    family: str,
    dataset_id: str,
    analytical_grain: str,
    value_column: str,
    time_column: str | None = None,
    group_column: str | None = None,
    entity_column: str | None = None,
    ranking_order: str = "none",
    ranking_limit: int | None = None,
) -> AIPlannerProposal:
    return AIPlannerProposal(
        decision="propose",
        title=title,
        family=family,
        dataset_id=dataset_id,
        analytical_grain=analytical_grain,
        x_column=None,
        y_column=None,
        group_column=group_column,
        value_column=value_column,
        time_column=time_column,
        dimension_column=None,
        entity_column=entity_column,
        aggregation_function="sum",
        ranking_order=ranking_order,
        ranking_limit=ranking_limit,
        window_operation="none",
        window_size=None,
        benchmark_reference=None,
        benchmark_operator=None,
        benchmark_selection=None,
        blockers=[],
        reasons=[],
        confidence=None,
    )


def main() -> None:
    print()
    print("=" * 80)
    print("DATALENS MULTI-INTENT CANONICALIZER PRESERVATION v0.1")
    print("=" * 80)
    print()

    catalog = build_catalog()

    raw_output = RawAIPlannerOutput(
        proposals=[
            proposal(
                title="Total revenue",
                family="aggregation",
                dataset_id="derived:demo:scalar:price",
                analytical_grain="overall",
                value_column="sum_price",
            ),
            proposal(
                title="Monthly revenue",
                family="time_series",
                dataset_id="derived:demo:monthly:date:price",
                analytical_grain="month",
                value_column="sum_price",
                time_column="month",
            ),
            proposal(
                title="Revenue by category",
                family="aggregation",
                dataset_id="derived:demo:category:categ:price",
                analytical_grain="categ",
                value_column="sum_price",
                group_column="categ",
            ),
            proposal(
                title="Top 10 customers",
                family="ranking",
                dataset_id="derived:demo:customer:client_id:price",
                analytical_grain="client_id",
                value_column="total_spend",
                entity_column="client_id",
                ranking_order="descending",
                ranking_limit=10,
            ),
        ]
    )

    report = validate_ai_planner_output(
        objective=OBJECTIVE,
        raw_output=raw_output,
        catalog=catalog,
        model="qwen3.5:4b",
    )

    print("validated_count :", report.validated_count)
    print("rejected_count  :", report.rejected_count)
    print()

    for item in report.items:
        print(
            item.proposal_index,
            item.proposal.family if item.proposal else None,
            item.proposal.dataset_id if item.proposal else None,
            item.proposal.analytical_grain if item.proposal else None,
        )

    assert report.validated_count == 4
    assert report.rejected_count == 0

    assert report.items[0].proposal is not None
    assert report.items[0].proposal.family == "aggregation"
    assert report.items[0].proposal.dataset_id == "derived:demo:scalar:price"
    assert report.items[0].proposal.analytical_grain == "overall"

    assert report.items[1].proposal is not None
    assert report.items[1].proposal.family == "time_series"
    assert report.items[1].proposal.dataset_id == "derived:demo:monthly:date:price"
    assert report.items[1].proposal.analytical_grain == "month"

    assert report.items[2].proposal is not None
    assert report.items[2].proposal.family == "aggregation"
    assert report.items[2].proposal.dataset_id == "derived:demo:category:categ:price"
    assert report.items[2].proposal.analytical_grain == "categ"
    assert report.items[2].proposal.group_column == "categ"

    assert report.items[3].proposal is not None
    assert report.items[3].proposal.family == "ranking"
    assert report.items[3].proposal.dataset_id == "derived:demo:customer:client_id:price"
    assert report.items[3].proposal.analytical_grain == "client_id"
    assert report.items[3].proposal.dimension_column == "client_id"
    assert report.items[3].proposal.entity_column is None
    assert report.items[3].proposal.ranking_order == "descending"
    assert report.items[3].proposal.ranking_limit == 10

    assert report.items[3].contract is not None

    ranking_bindings = {
        binding.role: binding.column
        for binding in report.items[3].contract.bindings
    }

    assert ranking_bindings == {
        "dimension": "client_id",
        "value": "total_spend",
    }

    print()
    print(
        "PASS - multi-intent proposals survive deterministic canonicalization"
    )


if __name__ == "__main__":
    main()
