from __future__ import annotations


import pandas as pd


from app.ai.tool_orchestrator import (
    AI_TOOL_ORCHESTRATOR_RULE_VERSION,
    TOOL_CAPABILITIES,
    execute_validated_contract,
)

from app.planning.analytical_contract import (
    AnalyticalContract,
)


def build_contract() -> AnalyticalContract:
    return AnalyticalContract.model_validate(
        {
            "contract_id":
                "ai:test-orchestrated-entity-outlier:01",

            "contract_version":
                "analytical_contract_v0.4",

            "origin":
                "ai_planner",

            "status":
                "validated",

            "title":
                "Supplier delivery-delay outliers",

            "request_text":
                (
                    "Identify atypical suppliers "
                    "according to delivery_delay_days."
                ),

            "family":
                "entity_outlier",

            "required_dataset_ids":
                [
                    "dataset:suppliers"
                ],

            "required_dataset_filenames":
                [
                    "suppliers.csv"
                ],

            "analytical_grain":
                "supplier_id",

            "bindings":
                [
                    {
                        "role":
                            "entity",

                        "column":
                            "supplier_id",

                        "dataset_id":
                            "dataset:suppliers",

                        "dataset_filename":
                            "suppliers.csv",

                        "semantic_concept":
                            None,

                        "analysis_kind":
                            "categorical",
                    },

                    {
                        "role":
                            "value",

                        "column":
                            "delivery_delay_days",

                        "dataset_id":
                            "dataset:suppliers",

                        "dataset_filename":
                            "suppliers.csv",

                        "semantic_concept":
                            None,

                        "analysis_kind":
                            "quantitative",
                    },
                ],

            "aggregation":
                None,

            "ranking":
                None,

            "benchmark":
                None,

            "share_of_total":
                None,

            "window":
                None,

            "filters":
                [],

            "joins":
                [],

            "derived_variables":
                [],

            "required_operations":
                [
                    (
                        "Execute only through a deterministic "
                        "entity_outlier tool."
                    )
                ],

            "provenance":
                None,

            "reasons":
                [],

            "blockers":
                [],

            "planner_confidence":
                None,
        }
    )


def build_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "supplier_id":
                [
                    f"s_{index}"
                    for index
                    in range(
                        1,
                        13,
                    )
                ],

            "delivery_delay_days":
                [
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    6,
                    6,
                    40,
                ],

            "defect_rate":
                [
                    0.01,
                    0.02,
                    0.02,
                    0.03,
                    0.03,
                    0.04,
                    0.04,
                    0.05,
                    0.05,
                    0.06,
                    0.06,
                    0.07,
                ],
        }
    )


def test_orchestrator_rule_version_v0_5() -> None:
    assert (
        AI_TOOL_ORCHESTRATOR_RULE_VERSION
        ==
        "ai_tool_orchestrator_v0.5"
    )


def test_entity_outlier_capability_is_enabled() -> None:
    capability = (
        TOOL_CAPABILITIES[
            "entity_outlier"
        ]
    )

    assert capability.enabled is True

    assert (
        capability.tool_name
        ==
        "run_entity_outlier"
    )

    assert (
        capability.statistical_strategy
        ==
        "deterministic_iqr_entity_outlier"
    )


def test_entity_outlier_contract_executes_directly() -> None:
    trace = (
        execute_validated_contract(
            contract=
                build_contract(),

            datasets=[
                {
                    "dataset_id":
                        "dataset:suppliers",

                    "filename":
                        "suppliers.csv",

                    "dataframe":
                        build_dataframe(),
                }
            ],

            call_index=
                1,
        )
    )


    assert (
        trace.execution_status
        ==
        "executed"
    )

    assert (
        trace.tool_name
        ==
        "run_entity_outlier"
    )

    assert trace.errors == []

    assert trace.result is not None


    result = trace.result

    assert (
        result.family
        ==
        "entity_outlier"
    )

    assert (
        result.execution_status
        ==
        "complete"
    )

    assert (
        result.metrics[
            "entity_column"
        ]
        ==
        "supplier_id"
    )

    assert (
        result.metrics[
            "value_column"
        ]
        ==
        "delivery_delay_days"
    )

    assert (
        result.metrics[
            "flagged_entity_count"
        ]
        ==
        1
    )

    assert (
        result.chart_data[
            0
        ][
            "entity"
        ]
        ==
        "s_12"
    )

    assert (
        result.chart_data[
            0
        ][
            "value"
        ]
        ==
        40.0
    )


def test_orchestrator_does_not_execute_unselected_metric() -> None:
    dataframe = build_dataframe()

    dataframe.loc[
        dataframe.index[-1],
        "defect_rate",
    ] = 999.0


    trace = (
        execute_validated_contract(
            contract=
                build_contract(),

            datasets=[
                {
                    "dataset_id":
                        "dataset:suppliers",

                    "filename":
                        "suppliers.csv",

                    "dataframe":
                        dataframe,
                }
            ],

            call_index=
                1,
        )
    )


    assert trace.result is not None

    assert (
        trace.result.metrics[
            "value_column"
        ]
        ==
        "delivery_delay_days"
    )

    assert (
        all(
            "defect_rate"
            not in str(row)

            for row
            in trace.result.chart_data
        )
    )
