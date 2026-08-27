from __future__ import annotations


import pandas as pd


from app.ai.native_tool_calling import (
    NATIVE_TOOL_CALLING_RULE_VERSION,
    NativeToolCallProposal,
    NATIVE_TOOL_SCHEMAS,
    expected_time_series_tool_args,
    validate_native_tool_call,
)

from app.ai.tool_orchestrator import (
    AI_TIME_SERIES_EXECUTION_RULE_VERSION,
    AI_TOOL_ORCHESTRATOR_RULE_VERSION,
    execute_validated_contract,
)

from app.planning.analytical_contract import (
    AggregationSpec,
    AnalyticalContract,
    VariableBinding,
)


def time_series_contract(
    *,
    contract_id: str,
    dataset_id: str,
    filename: str,
    time_column: str,
    value_column: str,
    function: str,
    title: str,
) -> AnalyticalContract:
    return AnalyticalContract(
        contract_id=contract_id,
        origin="ai_planner",
        status="validated",
        title=title,
        request_text=title,
        family="time_series",
        required_dataset_ids=[
            dataset_id,
        ],
        required_dataset_filenames=[
            filename,
        ],
        analytical_grain=(
            "month"
            if time_column == "month"
            else time_column
        ),
        bindings=[
            VariableBinding(
                role="time",
                column=time_column,
                dataset_id=dataset_id,
                dataset_filename=filename,
                analysis_kind="temporal",
            ),
            VariableBinding(
                role="value",
                column=value_column,
                dataset_id=dataset_id,
                dataset_filename=filename,
                analysis_kind="quantitative",
            ),
        ],
        aggregation=AggregationSpec(
            function=function,
            source_role="value",
            group_by_roles=[
                "time",
            ],
            output_name="planned_metric",
        ),
    )


def main() -> None:
    print(
        "=== DATALENS NATIVE TIME-SERIES AGGREGATIONS v0.1 ==="
    )


    # ========================================================
    # 1. NATIVE SCHEMA REQUIRES THE CANONICAL AGGREGATION
    # ========================================================

    time_schema = next(
        schema
        for schema in NATIVE_TOOL_SCHEMAS
        if (
            schema[
                "function"
            ][
                "name"
            ]
            ==
            "run_time_series"
        )
    )


    required = set(
        time_schema[
            "function"
        ][
            "parameters"
        ][
            "required"
        ]
    )


    assert (
        "aggregation_function"
        in required
    )


    print(
        "[PASS] run_time_series schema requires "
        "aggregation_function"
    )


    # ========================================================
    # 2. MONTHLY REVENUE SUM — EXACT LIVE SHAPE
    # ========================================================

    dataset_id = (
        "derived:"
        "combine_38d1ca2f75052572:"
        "monthly:order_date:gross_amount"
    )

    filename = (
        "orders__customers__products__"
        "monthly_gross_amount.derived"
    )


    sum_contract = time_series_contract(
        contract_id="ai:test:monthly-revenue:sum",
        dataset_id=dataset_id,
        filename=filename,
        time_column="month",
        value_column="sum_gross_amount",
        function="sum",
        title=(
            "Évolution mensuelle du chiffre d’affaires"
        ),
    )


    expected_sum = (
        expected_time_series_tool_args(
            sum_contract
        )
    )


    assert (
        expected_sum.aggregation_function
        ==
        "sum"
    )


    valid_sum_call = (
        NativeToolCallProposal(
            tool_name="run_time_series",
            arguments={
                "dataset_id":
                    dataset_id,

                "time_column":
                    "month",

                "value_column":
                    "sum_gross_amount",

                "aggregation_function":
                    "sum",
            },
        )
    )


    assert (
        validate_native_tool_call(
            contract=sum_contract,
            proposal=valid_sum_call,
        )
        ==
        []
    )


    wrong_sum_call = (
        NativeToolCallProposal(
            tool_name="run_time_series",
            arguments={
                "dataset_id":
                    dataset_id,

                "time_column":
                    "month",

                "value_column":
                    "sum_gross_amount",

                "aggregation_function":
                    "median",
            },
        )
    )


    assert (
        validate_native_tool_call(
            contract=sum_contract,
            proposal=wrong_sum_call,
        )
    )


    print(
        "[PASS] native validation accepts exact SUM and "
        "rejects aggregation drift"
    )


    monthly = pd.DataFrame(
        {
            "month":
                pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-02-01",
                    ]
                ),

            "sum_gross_amount":
                [
                    500.0,
                    362.0,
                ],
        }
    )


    sum_trace = (
        execute_validated_contract(
            contract=sum_contract,
            datasets=[
                {
                    "dataset_id":
                        dataset_id,

                    "filename":
                        filename,

                    "dataframe":
                        monthly,
                },
            ],
            call_index=1,
        )
    )


    assert (
        sum_trace.execution_status
        ==
        "executed"
    )

    assert (
        sum_trace.result
        is not None
    )


    sum_result = (
        sum_trace.result
    )


    assert (
        sum_result.execution_status
        ==
        "complete"
    )

    assert (
        sum_result.chart_type
        ==
        "line"
    )

    assert (
        sum_result.metrics[
            "aggregation_function"
        ]
        ==
        "sum"
    )

    assert (
        sum_result.metrics[
            "period_count"
        ]
        ==
        2
    )

    assert (
        [
            row[
                "value"
            ]
            for row
            in sum_result.chart_data
        ]
        ==
        [
            500.0,
            362.0,
        ]
    )


    print(
        "[PASS] deterministic time-series executor runs "
        "monthly revenue SUM"
    )


    # ========================================================
    # 3. MEDIAN / IQR REMAINS SUPPORTED
    # ========================================================

    median_contract = (
        time_series_contract(
            contract_id="ai:test:median",
            dataset_id="dataset:salary",
            filename="salary.csv",
            time_column="Year",
            value_column="salary",
            function="median",
            title="Évolution médiane du salaire",
        )
    )


    # Historical hand-written median tool proposals did not
    # contain aggregation_function. TimeSeriesToolArgs keeps a
    # deterministic median default for that legacy validation
    # shape, while the live JSON schema requires the field.
    legacy_median_call = (
        NativeToolCallProposal(
            tool_name="run_time_series",
            arguments={
                "dataset_id":
                    "dataset:salary",

                "time_column":
                    "Year",

                "value_column":
                    "salary",
            },
        )
    )


    assert (
        validate_native_tool_call(
            contract=median_contract,
            proposal=legacy_median_call,
        )
        ==
        []
    )


    salaries = pd.DataFrame(
        {
            "Year":
                [
                    2025,
                    2025,
                    2025,
                    2026,
                    2026,
                    2026,
                ],

            "salary":
                [
                    10.0,
                    20.0,
                    30.0,
                    20.0,
                    40.0,
                    60.0,
                ],
        }
    )


    median_trace = (
        execute_validated_contract(
            contract=median_contract,
            datasets=[
                {
                    "dataset_id":
                        "dataset:salary",

                    "filename":
                        "salary.csv",

                    "dataframe":
                        salaries,
                },
            ],
            call_index=1,
        )
    )


    assert (
        median_trace.execution_status
        ==
        "executed"
    )

    assert (
        median_trace.result
        is not None
    )


    median_result = (
        median_trace.result
    )


    assert (
        median_result.chart_type
        ==
        "line_band"
    )

    assert (
        median_result.metrics[
            "aggregation_function"
        ]
        ==
        "median"
    )

    assert (
        median_result.chart_data[
            0
        ][
            "median"
        ]
        ==
        20.0
    )

    assert (
        "q1"
        in median_result.chart_data[
            0
        ]
        and
        "q3"
        in median_result.chart_data[
            0
        ]
    )


    print(
        "[PASS] historical median/IQR time-series "
        "execution remains valid"
    )


    # ========================================================
    # 4. RULE VERSIONS
    # ========================================================

    assert (
        NATIVE_TOOL_CALLING_RULE_VERSION
        ==
        "native_tool_calling_v0.9"
    )

    assert (
        AI_TOOL_ORCHESTRATOR_RULE_VERSION
        ==
        "ai_tool_orchestrator_v0.4"
    )

    assert (
        AI_TIME_SERIES_EXECUTION_RULE_VERSION
        ==
        "ai_tool_time_series_v0.2"
    )


    print(
        "[PASS] native time-series rule versions"
    )

    print()
    print(
        "PASS - native time-series aggregations v0.1"
    )


if __name__ == "__main__":
    main()
