from __future__ import annotations


import pandas as pd


from app.ai.tool_orchestrator import (
    AI_AGGREGATION_EXECUTION_RULE_VERSION,
    execute_aggregation_contract,
)

from app.planning.analytical_contract import (
    AggregationSpec,
    AnalyticalContract,
    BenchmarkSpec,
    VariableBinding,
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


def contract(
    *,
    operator: str = "gt",
    selection: str = "matching_only",
    with_benchmark: bool = True,
) -> AnalyticalContract:
    return (
        AnalyticalContract(
            contract_id=
                "contract:return-rate-benchmark",

            origin=
                "ai_planner",

            status=
                "validated",

            title=
                "Grouped metric versus overall aggregate",

            request_text=
                "Find groups above the overall average.",

            family=
                "aggregation",

            bindings=[
                binding(
                    "value",
                    "returned_order",
                ),

                binding(
                    "group",
                    "region",
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

            benchmark=(
                BenchmarkSpec(
                    reference=
                        "overall_aggregate",

                    operator=
                        operator,

                    selection=
                        selection,
                )

                if with_benchmark

                else
                None
            ),
        )
    )


def dataset() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "region": [
                    "A",
                    "A",
                    "B",
                    "B",
                    "C",
                    "C",
                    None,
                ],

                "returned_order": [
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    1,
                ],
            }
        )
    )


def main() -> None:
    frame = (
        dataset()
    )


    print()
    print("=" * 80)
    print(
        "DATALENS BENCHMARK DETERMINISTIC EXECUTION v0.1"
    )
    print("=" * 80)
    print()


    # ========================================================
    # GT + MATCHING ONLY
    # ========================================================

    result = (
        execute_aggregation_contract(
            contract=
                contract(),

            dataframe=
                frame,

            dataset_id=
                "dataset:demo",

            dataset_filename=
                "demo.csv",
        )
    )


    assert (
        result.execution_status
        ==
        "complete"
    )


    # The row with region=None must NOT enter the reference
    # population. Eligible grouped values are:
    #
    # A = 0.0
    # B = 1.0
    # C = 0.5
    #
    # global eligible mean = 3 / 6 = 0.5
    assert (
        result.metrics[
            "benchmark_value"
        ]
        ==
        0.5
    )


    assert (
        result.metrics[
            "pre_benchmark_result_count"
        ]
        ==
        3
    )


    assert (
        result.metrics[
            "benchmark_matching_count"
        ]
        ==
        1
    )


    assert (
        result.metrics[
            "result_count"
        ]
        ==
        1
    )


    assert (
        len(
            result.chart_data
        )
        ==
        1
    )


    assert (
        result.chart_data[
            0
        ][
            "group"
        ]
        ==
        "B"
    )


    assert (
        result.chart_data[
            0
        ][
            "value"
        ]
        ==
        1.0
    )


    assert (
        result.chart_data[
            0
        ][
            "benchmark_match"
        ]
        is True
    )


    print(
        "[PASS] gt/matching_only keeps only groups above overall aggregate"
    )

    print(
        "[PASS] overall benchmark uses the same grouped population"
    )


    # ========================================================
    # GTE
    # ========================================================

    gte_result = (
        execute_aggregation_contract(
            contract=
                contract(
                    operator=
                        "gte",
                ),

            dataframe=
                frame,

            dataset_id=
                "dataset:demo",

            dataset_filename=
                "demo.csv",
        )
    )


    assert (
        [
            row[
                "group"
            ]

            for row
            in gte_result.chart_data
        ]
        ==
        [
            "B",
            "C",
        ]
    )


    print(
        "[PASS] gte includes groups equal to the benchmark"
    )


    # ========================================================
    # ANNOTATE ALL
    # ========================================================

    annotated = (
        execute_aggregation_contract(
            contract=
                contract(
                    selection=
                        "annotate_all",
                ),

            dataframe=
                frame,

            dataset_id=
                "dataset:demo",

            dataset_filename=
                "demo.csv",
        )
    )


    assert (
        len(
            annotated.chart_data
        )
        ==
        3
    )


    match_by_group = {
        row[
            "group"
        ]:
            row[
                "benchmark_match"
            ]

        for row
        in annotated.chart_data
    }


    assert (
        match_by_group
        ==
        {
            "A":
                False,

            "B":
                True,

            "C":
                False,
        }
    )


    assert (
        all(
            row[
                "benchmark_value"
            ]
            ==
            0.5

            for row
            in annotated.chart_data
        )
    )


    print(
        "[PASS] annotate_all preserves groups and benchmark evidence"
    )


    # ========================================================
    # BACKWARD COMPATIBILITY
    # ========================================================

    legacy = (
        execute_aggregation_contract(
            contract=
                contract(
                    with_benchmark=
                        False,
                ),

            dataframe=
                frame,

            dataset_id=
                "dataset:demo",

            dataset_filename=
                "demo.csv",
        )
    )


    assert (
        len(
            legacy.chart_data
        )
        ==
        3
    )


    assert (
        "benchmark_value"
        not in
        legacy.metrics
    )


    assert (
        all(
            "benchmark_match"
            not in
            row

            for row
            in legacy.chart_data
        )
    )


    print(
        "[PASS] legacy aggregation execution remains unchanged"
    )


    # ========================================================
    # VERSION
    # ========================================================

    assert (
        AI_AGGREGATION_EXECUTION_RULE_VERSION
        ==
        "ai_tool_aggregation_v0.3"
    )


    print(
        "[PASS] aggregation execution rule version v0.3"
    )


    print()
    print(
        "PASS - Benchmark deterministic execution v0.1"
    )


if __name__ == "__main__":
    main()
