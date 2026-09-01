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


EXPECTED_VERSION = (
    "ai_tool_aggregation_v0.3"
)


DATASET_ID = (
    "dataset:0001"
)


FILENAME = (
    "orders_prepared.csv"
)


def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_segment":
                    "Premium",

                "amount":
                    100,
            },
            {
                "customer_segment":
                    "Standard",

                "amount":
                    75,
            },
            {
                "customer_segment":
                    "Premium",

                "amount":
                    140,
            },
            {
                "customer_segment":
                    "Basic",

                "amount":
                    45,
            },
            {
                "customer_segment":
                    "Standard",

                "amount":
                    90,
            },
            {
                "customer_segment":
                    "Premium",

                "amount":
                    180,
            },
            {
                "customer_segment":
                    "Basic",

                "amount":
                    55,
            },
            {
                "customer_segment":
                    "Standard",

                "amount":
                    110,
            },
            {
                "customer_segment":
                    "Premium",

                "amount":
                    160,
            },
            {
                "customer_segment":
                    "Basic",

                "amount":
                    60,
            },
            {
                "customer_segment":
                    "Standard",

                "amount":
                    95,
            },
            {
                "customer_segment":
                    "Premium",

                "amount":
                    210,
            },
        ]
    )


def contract(
    *,
    benchmark: bool,
) -> AnalyticalContract:
    return AnalyticalContract(
        contract_id=(
            "ai:test:reporting:"
            +
            (
                "benchmark"
                if benchmark
                else "normal"
            )
        ),

        origin="ai_planner",
        status="validated",

        title=(
            "Average amount per "
            "customer segment"
        ),

        request_text=(
            "Quel est le amount moyen "
            "par customer_segment ?"
            if not benchmark
            else (
                "Quels customer_segment ont "
                "un amount moyen superieur "
                "a la moyenne globale ?"
            )
        ),

        family="aggregation",

        required_dataset_ids=[
            DATASET_ID,
        ],

        required_dataset_filenames=[
            FILENAME,
        ],

        analytical_grain=(
            "customer_segment"
        ),

        bindings=[
            VariableBinding(
                role="group",
                column=(
                    "customer_segment"
                ),
                dataset_id=DATASET_ID,
                dataset_filename=FILENAME,
                analysis_kind="categorical",
            ),

            VariableBinding(
                role="value",
                column="amount",
                dataset_id=DATASET_ID,
                dataset_filename=FILENAME,
                analysis_kind="quantitative",
            ),
        ],

        aggregation=AggregationSpec(
            function="mean",
            source_role="value",
            group_by_roles=[
                "group",
            ],
            output_name=(
                "planned_metric"
            ),
        ),

        benchmark=(
            BenchmarkSpec(
                reference=(
                    "overall_aggregate"
                ),
                operator="gt",
                selection=(
                    "matching_only"
                ),
            )
            if benchmark
            else None
        ),
    )


def assert_no_mojibake(
    text: str,
) -> None:
    for invalid in (
        "r?sultat",
        "?t?",
        "calcul?s",
        "agr?g",
        "?tre",
    ):
        assert (
            invalid
            not in
            text
        ), text


def main() -> None:
    print(
        "=== DATALENS AGGREGATION REPORTING "
        "SEMANTICS v0.1 ==="
    )


    assert (
        AI_AGGREGATION_EXECUTION_RULE_VERSION
        ==
        EXPECTED_VERSION
    )


    print(
        "[PASS] aggregation executor v0.3"
    )


    frame = dataframe()


    # ========================================================
    # NORMAL AGGREGATION
    # ========================================================

    normal = execute_aggregation_contract(
        contract=contract(
            benchmark=False
        ),
        dataframe=frame,
        dataset_id=DATASET_ID,
        dataset_filename=FILENAME,
    )


    assert (
        normal.execution_status
        ==
        "complete"
    )


    assert (
        normal.execution_rule_version
        ==
        EXPECTED_VERSION
    )


    assert (
        normal.summary
        ==
        [
            (
                "3 r\u00e9sultat(s) ont "
                "\u00e9t\u00e9 calcul\u00e9s "
                "avec l'agr\u00e9gation `mean`."
            )
        ]
    ), normal.summary


    for line in normal.summary:
        assert_no_mojibake(
            line
        )


    assert (
        normal.limitations
        ==
        [
            (
                "This result is a deterministic descriptive "
                "aggregation. It does not imply statistical "
                "significance or causality."
            )
        ]
    ), normal.limitations


    assert (
        "benchmark"
        not in
        normal.limitations[
            0
        ].lower()
    )


    assert (
        normal.metrics.get(
            "benchmark_reference"
        )
        is None
    )


    assert (
        normal.metrics[
            "result_count"
        ]
        ==
        3
    )


    print(
        "[PASS] normal aggregation summary "
        "has correct Unicode"
    )

    print(
        "[PASS] normal aggregation limitation "
        "does not claim benchmark comparison"
    )


    # ========================================================
    # REAL BENCHMARK AGGREGATION
    # ========================================================

    benchmark = execute_aggregation_contract(
        contract=contract(
            benchmark=True
        ),
        dataframe=frame,
        dataset_id=DATASET_ID,
        dataset_filename=FILENAME,
    )


    assert (
        benchmark.execution_status
        ==
        "complete"
    )


    assert (
        benchmark.execution_rule_version
        ==
        EXPECTED_VERSION
    )


    for line in benchmark.summary:
        assert_no_mojibake(
            line
        )


    assert (
        benchmark.summary[
            0
        ]
        ==
        (
            "3 r\u00e9sultat(s) ont "
            "\u00e9t\u00e9 calcul\u00e9s "
            "avec l'agr\u00e9gation `mean`."
        )
    )


    assert (
        len(
            benchmark.summary
        )
        ==
        2
    )


    assert (
        "benchmark"
        in
        benchmark.summary[
            1
        ].lower()
    )


    assert (
        benchmark.limitations
        ==
        [
            (
                "This result is a deterministic descriptive "
                "aggregation and benchmark comparison. "
                "It does not imply statistical significance "
                "or causality."
            )
        ]
    ), benchmark.limitations


    assert (
        benchmark.metrics[
            "benchmark_reference"
        ]
        ==
        "overall_aggregate"
    )


    assert (
        benchmark.metrics[
            "benchmark_operator"
        ]
        ==
        "gt"
    )


    assert (
        benchmark.metrics[
            "benchmark_selection"
        ]
        ==
        "matching_only"
    )


    assert (
        benchmark.metrics[
            "result_count"
        ]
        ==
        1
    )


    assert (
        benchmark.chart_data[
            0
        ][
            "category"
        ]
        ==
        "Premium"
    )


    print(
        "[PASS] real benchmark summary "
        "has correct Unicode"
    )

    print(
        "[PASS] real benchmark limitation "
        "retains benchmark semantics"
    )

    print(
        "[PASS] benchmark deterministic result "
        "remains Premium only"
    )


    print()
    print(
        "PASS - aggregation reporting "
        "semantics v0.1"
    )


if __name__ == "__main__":
    main()
