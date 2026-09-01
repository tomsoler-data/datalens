from __future__ import annotations

import hashlib
import inspect
import json


from app.planning.ai_analytical_planner import (
    AI_ANALYTICAL_PLANNER_RULE_VERSION,
    AIPlannerProposal,
    BENCHMARK_WIRE_FIELDS,
    SYSTEM_PROMPT,
    benchmark_wire_errors,
    build_benchmark,
    build_contract_id,
    build_user_prompt,
    clear_benchmark_wire,
)


def proposal(
    *,
    family="aggregation",
    decision="propose",
    group_column="country",
    benchmark_reference=None,
    benchmark_operator=None,
    benchmark_selection=None,
):
    return AIPlannerProposal(
        decision=decision,
        title="Benchmark wire test",
        family=family,
        dataset_id="dataset:benchmark",
        analytical_grain=None,
        x_column=None,
        y_column=None,
        group_column=group_column,
        value_column="amount",
        time_column=None,
        dimension_column=None,
        entity_column=None,
        aggregation_function="mean",
        ranking_order="none",
        ranking_limit=None,
        window_operation="none",
        window_size=None,
        benchmark_reference=(
            benchmark_reference
        ),
        benchmark_operator=(
            benchmark_operator
        ),
        benchmark_selection=(
            benchmark_selection
        ),
        blockers=[],
        reasons=[
            "benchmark wire test",
        ],
        confidence=1.0,
    )


def enum_values(
    value,
):
    result = set()


    if isinstance(
        value,
        dict,
    ):
        enum = (
            value.get(
                "enum"
            )
        )

        if isinstance(
            enum,
            list,
        ):
            result.update(
                item

                for item
                in enum

                if isinstance(
                    item,
                    str,
                )
            )


        const = (
            value.get(
                "const"
            )
        )

        if isinstance(
            const,
            str,
        ):
            result.add(
                const
            )


        for child in (
            value.values()
        ):
            result.update(
                enum_values(
                    child
                )
            )


    elif isinstance(
        value,
        list,
    ):
        for child in (
            value
        ):
            result.update(
                enum_values(
                    child
                )
            )


    return result


def legacy_contract_id(
    *,
    objective,
    item,
    proposal_index,
):
    proposal_payload = (
        item.model_dump(
            mode="json"
        )
    )


    for field in (
        BENCHMARK_WIRE_FIELDS
    ):
        proposal_payload.pop(
            field,
            None,
        )


    payload = json.dumps(
        {
            "objective":
                objective,

            "proposal":
                proposal_payload,

            "proposal_index":
                proposal_index,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


    digest = hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()[
        :16
    ]


    return (
        "ai:"
        f"{digest}:"
        f"{proposal_index:02d}"
    )


def main():
    print(
        "=== DATALENS AI PLANNER "
        "BENCHMARK WIRE v0.1 ==="
    )


    # ========================================================
    # VERSION
    # ========================================================

    assert (
        AI_ANALYTICAL_PLANNER_RULE_VERSION
        ==
        "ai_analytical_planner_v0.34"
    )

    print(
        "[PASS] planner version v0.34"
    )


    # ========================================================
    # STRICT STRUCTURED OUTPUT SCHEMA
    # ========================================================

    schema = (
        AIPlannerProposal
        .model_json_schema()
    )

    required = set(
        schema[
            "required"
        ]
    )


    assert set(
        BENCHMARK_WIRE_FIELDS
    ).issubset(
        required
    )

    print(
        "[PASS] benchmark wire keys required"
    )


    # ========================================================
    # EXACT CANONICAL VOCABULARY
    # ========================================================

    properties = (
        schema[
            "properties"
        ]
    )


    assert (
        enum_values(
            properties[
                "benchmark_reference"
            ]
        )
        ==
        {
            "overall_aggregate",
        }
    )


    assert (
        enum_values(
            properties[
                "benchmark_operator"
            ]
        )
        ==
        {
            "gt",
            "gte",
            "lt",
            "lte",
        }
    )


    assert (
        enum_values(
            properties[
                "benchmark_selection"
            ]
        )
        ==
        {
            "matching_only",
            "annotate_all",
        }
    )

    print(
        "[PASS] canonical benchmark vocabulary"
    )


    # ========================================================
    # NO BENCHMARK
    # ========================================================

    plain = (
        proposal()
    )

    assert (
        benchmark_wire_errors(
            plain
        )
        ==
        []
    )

    assert (
        build_benchmark(
            plain
        )
        is None
    )

    print(
        "[PASS] null/null/null = no benchmark"
    )


    # ========================================================
    # ACTIVE BENCHMARK
    # ========================================================

    active = (
        proposal(
            benchmark_reference=(
                "overall_aggregate"
            ),
            benchmark_operator="gt",
            benchmark_selection=(
                "matching_only"
            ),
        )
    )


    benchmark = (
        build_benchmark(
            active
        )
    )

    assert (
        benchmark
        is not None
    )

    assert (
        benchmark.reference
        ==
        "overall_aggregate"
    )

    assert (
        benchmark.operator
        ==
        "gt"
    )

    assert (
        benchmark.selection
        ==
        "matching_only"
    )

    print(
        "[PASS] wire -> canonical BenchmarkSpec"
    )


    # ========================================================
    # PARTIAL BENCHMARK FAIL-CLOSED
    # ========================================================

    partial = (
        proposal(
            benchmark_reference=(
                "overall_aggregate"
            ),
            benchmark_operator=None,
            benchmark_selection=(
                "matching_only"
            ),
        )
    )

    assert (
        benchmark_wire_errors(
            partial
        )
    )

    print(
        "[PASS] partial benchmark fails closed"
    )


    # ========================================================
    # WRONG FAMILY
    # ========================================================

    wrong_family = (
        proposal(
            family="ranking",
            benchmark_reference=(
                "overall_aggregate"
            ),
            benchmark_operator="gt",
            benchmark_selection=(
                "matching_only"
            ),
        )
    )

    assert any(
        "aggregation"
        in error

        for error
        in benchmark_wire_errors(
            wrong_family
        )
    )

    print(
        "[PASS] non-aggregation benchmark rejected"
    )


    # ========================================================
    # UNGROUPED
    # ========================================================

    ungrouped = (
        proposal(
            group_column=None,
            benchmark_reference=(
                "overall_aggregate"
            ),
            benchmark_operator="gt",
            benchmark_selection=(
                "matching_only"
            ),
        )
    )

    assert (
        benchmark_wire_errors(
            ungrouped
        )
    )

    print(
        "[PASS] ungrouped benchmark rejected"
    )


    # ========================================================
    # BLOCKED
    # ========================================================

    blocked = (
        proposal(
            decision="blocked",
            benchmark_reference=(
                "overall_aggregate"
            ),
            benchmark_operator="gt",
            benchmark_selection=(
                "matching_only"
            ),
        )
    )

    assert (
        benchmark_wire_errors(
            blocked
        )
    )

    print(
        "[PASS] blocked benchmark rejected"
    )


    # ========================================================
    # DETERMINISTIC CLEAR
    # ========================================================

    cleared = (
        clear_benchmark_wire(
            active
        )
    )

    assert (
        cleared.benchmark_reference
        is None
    )

    assert (
        cleared.benchmark_operator
        is None
    )

    assert (
        cleared.benchmark_selection
        is None
    )

    print(
        "[PASS] deterministic wire clearing"
    )


    # ========================================================
    # IDENTITY BACKWARD COMPATIBILITY
    # ========================================================

    objective = (
        "identity compatibility"
    )


    assert (
        build_contract_id(
            objective=objective,
            proposal=plain,
            proposal_index=1,
        )
        ==
        legacy_contract_id(
            objective=objective,
            item=plain,
            proposal_index=1,
        )
    )

    print(
        "[PASS] legacy no-benchmark identity preserved"
    )


    assert (
        build_contract_id(
            objective=objective,
            proposal=plain,
            proposal_index=1,
        )
        !=
        build_contract_id(
            objective=objective,
            proposal=active,
            proposal_index=1,
        )
    )

    print(
        "[PASS] active benchmark changes identity"
    )


    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    assert (
        "benchmark_reference"
        in
        SYSTEM_PROMPT
    )

    assert (
        "benchmark_operator"
        in
        SYSTEM_PROMPT
    )

    assert (
        "benchmark_selection"
        in
        SYSTEM_PROMPT
    )

    assert (
        "overall_aggregate"
        in
        SYSTEM_PROMPT
    )

    assert (
        "supérieur à la moyenne"
        in
        SYSTEM_PROMPT.lower()
    )

    assert (
        "EXEMPLE 9"
        in
        SYSTEM_PROMPT
    )

    print(
        "[PASS] SYSTEM_PROMPT benchmark wiring"
    )


    # ========================================================
    # USER PROMPT
    # ========================================================

    prompt_source = (
        inspect.getsource(
            build_user_prompt
        )
    )

    assert (
        "benchmark_reference"
        in
        prompt_source
    )

    assert (
        "benchmark_operator"
        in
        prompt_source
    )

    assert (
        "benchmark_selection"
        in
        prompt_source
    )

    assert (
        "overall_aggregate"
        in
        prompt_source
    )

    print(
        "[PASS] build_user_prompt benchmark wiring"
    )


    print()
    print(
        "PASS - AI Planner benchmark wire v0.1"
    )


if __name__ == "__main__":
    main()
