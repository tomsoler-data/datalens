from __future__ import annotations


from pydantic import (
    ValidationError,
)


from app.planning.analytical_contract import (
    ANALYTICAL_CONTRACT_RULE_VERSION,
    AggregationSpec,
    AnalyticalContract,
    BenchmarkSpec,
    VariableBinding,
    analytical_contract_json_schema,
)


def binding(
    role: str,
    column: str,
) -> VariableBinding:
    return (
        VariableBinding(
            role=
                role,

            column=
                column,

            dataset_id=
                "dataset:demo",

            dataset_filename=
                "demo.csv",
        )
    )


def grouped_aggregation(
    *,
    benchmark: BenchmarkSpec | None,
) -> AnalyticalContract:
    return (
        AnalyticalContract(
            contract_id=
                "contract:benchmark",

            origin=
                "ai_planner",

            status=
                "validated",

            title=
                "Grouped metric versus overall aggregate",

            request_text=
                "Compare grouped values with the overall value.",

            family=
                "aggregation",

            bindings=[
                binding(
                    "value",
                    "metric",
                ),

                binding(
                    "group",
                    "segment",
                ),
            ],

            aggregation=
                AggregationSpec(
                    function=
                        "mean",

                    source_role=
                        "value",

                    group_by_roles=[
                        "group",
                    ],
                ),

            benchmark=
                benchmark,
        )
    )


def main() -> None:
    print()
    print("=" * 80)
    print(
        "DATALENS CANONICAL BENCHMARK SPEC v0.1"
    )
    print("=" * 80)
    print()


    # ========================================================
    # GENERIC SPEC
    # ========================================================

    benchmark = (
        BenchmarkSpec(
            reference=
                "overall_aggregate",

            operator=
                "gt",

            selection=
                "matching_only",
        )
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
        "[PASS] generic BenchmarkSpec is representable"
    )


    # ========================================================
    # VALID GROUPED AGGREGATION
    # ========================================================

    contract = (
        grouped_aggregation(
            benchmark=
                benchmark
        )
    )


    assert (
        contract.benchmark
        is not None
    )

    assert (
        contract.aggregation
        is not None
    )

    assert (
        contract.aggregation.function
        ==
        "mean"
    )

    assert (
        contract.aggregation.group_by_roles
        ==
        [
            "group"
        ]
    )


    print(
        "[PASS] grouped aggregation may carry BenchmarkSpec"
    )


    # ========================================================
    # NO GROUPING MUST FAIL
    # ========================================================

    try:
        AnalyticalContract(
            contract_id=
                "contract:no-group",

            origin=
                "ai_planner",

            status=
                "validated",

            title=
                "Invalid ungrouped benchmark",

            request_text=
                "Compare with overall aggregate.",

            family=
                "aggregation",

            bindings=[
                binding(
                    "value",
                    "metric",
                ),
            ],

            aggregation=
                AggregationSpec(
                    function=
                        "mean",

                    source_role=
                        "value",

                    group_by_roles=[],
                ),

            benchmark=
                benchmark,
        )

    except ValidationError as error:
        assert (
            "BenchmarkSpec requires at least one grouped"
            in
            str(
                error
            )
        )

    else:
        raise AssertionError(
            "Ungrouped benchmark was accepted."
        )


    print(
        "[PASS] ungrouped benchmark is rejected"
    )


    # ========================================================
    # WRONG FAMILY MUST FAIL
    # ========================================================

    try:
        AnalyticalContract(
            contract_id=
                "contract:ranking-benchmark",

            origin=
                "ai_planner",

            status=
                "validated",

            title=
                "Unsupported ranking benchmark",

            request_text=
                "Rank and benchmark.",

            family=
                "ranking",

            bindings=[
                binding(
                    "value",
                    "metric",
                ),

                binding(
                    "group",
                    "segment",
                ),
            ],

            aggregation=
                AggregationSpec(
                    function=
                        "sum",

                    source_role=
                        "value",

                    group_by_roles=[
                        "group"
                    ],
                ),

            ranking={
                "order":
                    "descending",

                "limit":
                    5,
            },

            benchmark=
                benchmark,
        )

    except ValidationError as error:
        assert (
            "BenchmarkSpec v0.1 is supported only for aggregation"
            in
            str(
                error
            )
        )

    else:
        raise AssertionError(
            "Benchmark on unsupported family was accepted."
        )


    print(
        "[PASS] BenchmarkSpec v0.1 remains aggregation-only"
    )


    # ========================================================
    # INVALID ENUM MUST FAIL
    # ========================================================

    try:
        BenchmarkSpec(
            reference=
                "overall_aggregate",

            operator=
                "eq",
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            "Unsupported benchmark operator was accepted."
        )


    print(
        "[PASS] benchmark operator vocabulary is closed"
    )


    # ========================================================
    # BACKWARD COMPATIBILITY
    # ========================================================

    legacy_shape = (
        grouped_aggregation(
            benchmark=None
        )
    )


    assert (
        legacy_shape.benchmark
        is None
    )


    print(
        "[PASS] existing contracts remain benchmark-optional"
    )


    # ========================================================
    # STRUCTURED SCHEMA
    # ========================================================

    schema = (
        analytical_contract_json_schema()
    )


    assert (
        "benchmark"
        in
        schema[
            "properties"
        ]
    )


    definitions = (
        schema.get(
            "$defs",
            {}
        )
    )


    assert (
        "BenchmarkSpec"
        in
        definitions
    )


    print(
        "[PASS] structured contract schema exposes BenchmarkSpec"
    )


    # ========================================================
    # VERSION
    # ========================================================

    assert (
        ANALYTICAL_CONTRACT_RULE_VERSION
        ==
        "analytical_contract_v0.3"
    )


    assert (
        contract.contract_version
        ==
        "analytical_contract_v0.3"
    )


    print(
        "[PASS] Analytical Contract rule version v0.3"
    )


    print()
    print(
        "PASS - Canonical BenchmarkSpec v0.1"
    )


if __name__ == "__main__":
    main()
