from __future__ import annotations

import json


from app.planning.ai_analytical_planner import (
    normalize_ai_planner_wire_content,
)

from tests.planning.test_objective_coverage_multi_intent_same_metric_v0_1 import (
    build_catalog,
)


RAW_PROPOSALS = [
    {
        "decision": "propose",
        "family": "aggregation",
        "dataset_id": "derived:demo:scalar:price",
        "x_column": None,
        "y_column": None,
        "group_column": None,
        "value_column": "sum_price",
        "time_column": None,
        "dimension_column": None,
        "entity_column": None,
        "aggregation_function": "sum",
        "ranking_order": None,
        "ranking_limit": None,
        "window_operation": None,
        "window_size": None,
        "benchmark_reference": None,
        "benchmark_operator": None,
        "benchmark_selection": None,
        "blockers": [],
    },
    {
        "decision": "propose",
        "family": "time_series",
        "dataset_id": "derived:demo:monthly:date:price",
        "x_column": None,
        "y_column": None,
        "group_column": None,
        "value_column": "sum_price",
        "time_column": "month",
        "dimension_column": None,
        "entity_column": None,
        "aggregation_function": "sum",
        "ranking_order": None,
        "ranking_limit": None,
        "window_operation": None,
        "window_size": None,
        "benchmark_reference": None,
        "benchmark_operator": None,
        "benchmark_selection": None,
        "blockers": [],
    },
    {
        "decision": "propose",
        "family": "aggregation",
        "dataset_id": "derived:demo:category:categ:price",
        "x_column": None,
        "y_column": None,
        "group_column": "categ",
        "value_column": "sum_price",
        "time_column": None,
        "dimension_column": None,
        "entity_column": None,
        "aggregation_function": "sum",
        "ranking_order": None,
        "ranking_limit": None,
        "window_operation": None,
        "window_size": None,
        "benchmark_reference": None,
        "benchmark_operator": None,
        "benchmark_selection": None,
        "blockers": [],
    },
    {
        "decision": "propose",
        "family": "ranking",
        "dataset_id": "derived:demo:customer:client_id:price",
        "x_column": None,
        "y_column": None,
        "group_column": None,
        "value_column": "total_spend",
        "time_column": None,
        "dimension_column": None,
        "entity_column": "client_id",
        "aggregation_function": "sum",
        "ranking_order": "descending",
        "ranking_limit": 10,
        "window_operation": None,
        "window_size": None,
        "benchmark_reference": None,
        "benchmark_operator": None,
        "benchmark_selection": None,
        "blockers": [],
    },
]


FENCE = chr(96) * 3

RAW_QWEN_CONTENT = (
    FENCE
    + "json\n"
    + json.dumps(
        RAW_PROPOSALS,
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
    + FENCE
)


def main() -> None:
    print()
    print("=" * 80)
    print("DATALENS QWEN WIRE ADAPTER v0.1")
    print("=" * 80)
    print()

    normalized = normalize_ai_planner_wire_content(
        content=RAW_QWEN_CONTENT,
        catalog=build_catalog(),
    )

    decoded = json.loads(
        normalized
    )

    assert isinstance(
        decoded,
        dict,
    )

    assert (
        "proposals"
        in
        decoded
    )

    proposals = decoded[
        "proposals"
    ]

    assert (
        len(proposals)
        ==
        4
    )


    # ========================================================
    # TRANSPORT NORMALIZATION
    # ========================================================

    assert (
        proposals[0]["ranking_order"]
        ==
        "none"
    )

    assert (
        proposals[0]["window_operation"]
        ==
        "none"
    )

    assert (
        proposals[0]["analytical_grain"]
        ==
        "overall"
    )

    assert (
        proposals[1]["analytical_grain"]
        ==
        "month"
    )

    assert (
        proposals[2]["analytical_grain"]
        ==
        "categ"
    )

    assert (
        proposals[3]["analytical_grain"]
        ==
        "client_id"
    )


    # ========================================================
    # QWEN-OWNED ANALYTICAL DECISIONS
    # ========================================================

    assert (
        proposals[0]["family"]
        ==
        "aggregation"
    )

    assert (
        proposals[0]["dataset_id"]
        ==
        "derived:demo:scalar:price"
    )

    assert (
        proposals[0]["value_column"]
        ==
        "sum_price"
    )

    assert (
        proposals[1]["family"]
        ==
        "time_series"
    )

    assert (
        proposals[1]["dataset_id"]
        ==
        "derived:demo:monthly:date:price"
    )

    assert (
        proposals[1]["time_column"]
        ==
        "month"
    )

    assert (
        proposals[2]["family"]
        ==
        "aggregation"
    )

    assert (
        proposals[2]["group_column"]
        ==
        "categ"
    )

    assert (
        proposals[3]["family"]
        ==
        "ranking"
    )

    assert (
        proposals[3]["entity_column"]
        ==
        "client_id"
    )

    assert (
        proposals[3]["value_column"]
        ==
        "total_spend"
    )

    assert (
        proposals[3]["ranking_order"]
        ==
        "descending"
    )

    assert (
        proposals[3]["ranking_limit"]
        ==
        10
    )


    # ========================================================
    # BENCHMARK MUST REMAIN ABSENT
    # ========================================================

    for proposal in proposals:
        assert proposal["benchmark_reference"] is None
        assert proposal["benchmark_operator"] is None
        assert proposal["benchmark_selection"] is None


    print(
        "[PASS] markdown/list transport is normalized"
    )

    print(
        "[PASS] server-owned analytical grain metadata is restored"
    )

    print(
        "[PASS] Qwen analytical decisions remain unchanged"
    )

    print(
        "[PASS] no benchmark decision is invented"
    )

    print()
    print(
        "PASS - Qwen wire adapter v0.1"
    )


if __name__ == "__main__":
    main()
