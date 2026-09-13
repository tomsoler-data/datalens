from __future__ import annotations

from types import SimpleNamespace


import app.planning.ai_analytical_planner as planner

from tests.planning.test_ai_planner_qwen_wire_adapter_v0_1 import (
    RAW_QWEN_CONTENT,
)

from tests.planning.test_objective_coverage_multi_intent_same_metric_v0_1 import (
    OBJECTIVE,
    build_catalog,
)


def main() -> None:
    print()
    print("=" * 80)
    print("DATALENS QWEN PLANNER WIRE INTEGRATION v0.1")
    print("=" * 80)
    print()

    captured: dict[
        str,
        object,
    ] = {}


    def fake_classified_llm_chat(
        chat_client,
        *,
        payload_class,
        **kwargs,
    ):
        captured.update(
            kwargs
        )

        return SimpleNamespace(
            message=SimpleNamespace(
                content=RAW_QWEN_CONTENT,
            )
        )


    original = (
        planner.classified_llm_chat
    )

    planner.classified_llm_chat = (
        fake_classified_llm_chat
    )


    try:
        raw_output = (
            planner.generate_raw_ai_plan(
                objective=OBJECTIVE,
                catalog=build_catalog(),
                model="qwen3.5:4b",
            )
        )

    finally:
        planner.classified_llm_chat = (
            original
        )


    # ========================================================
    # STRUCTURED GENERATION MODE
    # ========================================================

    assert (
        captured.get(
            "think"
        )
        is False
    )


    # ========================================================
    # QWEN RAW LIST/FENCE OUTPUT MUST REACH THE CANONICAL WIRE
    # ========================================================

    assert (
        len(
            raw_output.proposals
        )
        ==
        4
    )


    scalar = raw_output.proposals[0]
    monthly = raw_output.proposals[1]
    category = raw_output.proposals[2]
    ranking = raw_output.proposals[3]


    assert scalar.family == "aggregation"
    assert scalar.dataset_id == "derived:demo:scalar:price"
    assert scalar.analytical_grain == "overall"
    assert scalar.value_column == "sum_price"


    assert monthly.family == "time_series"
    assert monthly.dataset_id == "derived:demo:monthly:date:price"
    assert monthly.analytical_grain == "month"
    assert monthly.time_column == "month"
    assert monthly.value_column == "sum_price"


    assert category.family == "aggregation"
    assert category.dataset_id == "derived:demo:category:categ:price"
    assert category.analytical_grain == "categ"
    assert category.group_column == "categ"
    assert category.value_column == "sum_price"


    assert ranking.family == "ranking"
    assert ranking.dataset_id == "derived:demo:customer:client_id:price"
    assert ranking.analytical_grain == "client_id"
    assert ranking.entity_column == "client_id"
    assert ranking.value_column == "total_spend"
    assert ranking.ranking_order == "descending"
    assert ranking.ranking_limit == 10


    # ========================================================
    # NO BENCHMARK MAY BE INVENTED BY THE PROTOCOL LAYER
    # ========================================================

    for proposal in raw_output.proposals:
        assert proposal.benchmark_reference is None
        assert proposal.benchmark_operator is None
        assert proposal.benchmark_selection is None


    print(
        "[PASS] structured generation uses think=False"
    )

    print(
        "[PASS] Qwen fenced/list output reaches canonical planner wire"
    )

    print(
        "[PASS] four AI-selected analytical intents are preserved"
    )

    print(
        "[PASS] no benchmark is invented by protocol normalization"
    )

    print()
    print(
        "PASS - Qwen planner wire integration v0.1"
    )


if __name__ == "__main__":
    main()
