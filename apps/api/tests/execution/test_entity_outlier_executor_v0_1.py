from __future__ import annotations


import pandas as pd

import pytest


from app.execution.entity_outlier import (
    ENTITY_OUTLIER_EXECUTION_RULE_VERSION,
    execute_entity_outlier_contract,
)

from app.planning.analytical_contract import (
    AnalyticalContract,
)


def build_contract(
    *,
    entity_column: str = "machine_id",
    value_column: str = "downtime_hours",
) -> AnalyticalContract:
    return AnalyticalContract.model_validate(
        {
            "contract_id":
                "ai:test-entity-outlier-executor:01",

            "contract_version":
                "analytical_contract_v0.4",

            "origin":
                "ai_planner",

            "status":
                "validated",

            "title":
                "Machine downtime outliers",

            "request_text":
                (
                    "Identify atypical machines "
                    "according to downtime_hours."
                ),

            "family":
                "entity_outlier",

            "required_dataset_ids":
                [
                    "dataset:machines"
                ],

            "required_dataset_filenames":
                [
                    "machines.csv"
                ],

            "analytical_grain":
                "machine_id",

            "bindings":
                [
                    {
                        "role":
                            "entity",

                        "column":
                            entity_column,

                        "dataset_id":
                            "dataset:machines",

                        "dataset_filename":
                            "machines.csv",

                        "semantic_concept":
                            None,

                        "analysis_kind":
                            "categorical",
                    },

                    {
                        "role":
                            "value",

                        "column":
                            value_column,

                        "dataset_id":
                            "dataset:machines",

                        "dataset_filename":
                            "machines.csv",

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


def test_entity_outlier_executor_rule_version() -> None:
    assert (
        ENTITY_OUTLIER_EXECUTION_RULE_VERSION
        ==
        "entity_outlier_executor_v0.1"
    )


def test_generic_machine_outlier_is_detected() -> None:
    dataframe = pd.DataFrame(
        {
            "machine_id":
                [
                    f"m_{index}"
                    for index
                    in range(
                        1,
                        13,
                    )
                ],

            "downtime_hours":
                [
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    100,
                ],

            "temperature":
                [
                    50,
                    51,
                    52,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    59,
                    60,
                    61,
                ],
        }
    )


    result = (
        execute_entity_outlier_contract(
            contract=
                build_contract(),

            dataframe=
                dataframe,

            dataset_id=
                "dataset:machines",

            dataset_filename=
                "machines.csv",
        )
    )


    assert (
        result.execution_status
        ==
        "complete"
    )

    assert (
        result.family
        ==
        "entity_outlier"
    )

    assert (
        result.metrics[
            "entity_column"
        ]
        ==
        "machine_id"
    )

    assert (
        result.metrics[
            "value_column"
        ]
        ==
        "downtime_hours"
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
        "m_12"
    )

    assert (
        result.chart_data[
            0
        ][
            "value"
        ]
        ==
        100.0
    )

    assert (
        result.chart_data[
            0
        ][
            "direction"
        ]
        ==
        "high"
    )


def test_executor_uses_only_ai_selected_value_column() -> None:
    dataframe = pd.DataFrame(
        {
            "machine_id":
                [
                    f"m_{index}"
                    for index
                    in range(
                        1,
                        13,
                    )
                ],

            "downtime_hours":
                [
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                ],

            "defect_rate":
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    999,
                ],
        }
    )


    result = (
        execute_entity_outlier_contract(
            contract=
                build_contract(
                    value_column=
                        "downtime_hours"
                ),

            dataframe=
                dataframe,

            dataset_id=
                "dataset:machines",

            dataset_filename=
                "machines.csv",
        )
    )


    assert (
        result.metrics[
            "value_column"
        ]
        ==
        "downtime_hours"
    )

    assert (
        result.metrics[
            "flagged_entity_count"
        ]
        ==
        0
    )

    assert (
        result.chart_data
        ==
        []
    )


def test_repeated_entities_are_not_silently_aggregated() -> None:
    dataframe = pd.DataFrame(
        {
            "machine_id":
                [
                    "m_1",
                    "m_1",
                    "m_2",
                    "m_3",
                    "m_4",
                    "m_5",
                    "m_6",
                    "m_7",
                    "m_8",
                ],

            "downtime_hours":
                [
                    10,
                    12,
                    11,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                ],
        }
    )


    result = (
        execute_entity_outlier_contract(
            contract=
                build_contract(),

            dataframe=
                dataframe,

            dataset_id=
                "dataset:machines",

            dataset_filename=
                "machines.csv",
        )
    )


    assert (
        result.execution_status
        ==
        "needs_information"
    )

    assert (
        result.chart_data
        ==
        []
    )

    assert any(
        "never performs an implicit aggregation"
        in warning

        for warning
        in result.warnings
    )


def test_too_few_entities_are_skipped() -> None:
    dataframe = pd.DataFrame(
        {
            "machine_id":
                [
                    "m_1",
                    "m_2",
                    "m_3",
                    "m_4",
                    "m_5",
                    "m_6",
                    "m_7",
                ],

            "downtime_hours":
                [
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    100,
                ],
        }
    )


    result = (
        execute_entity_outlier_contract(
            contract=
                build_contract(),

            dataframe=
                dataframe,

            dataset_id=
                "dataset:machines",

            dataset_filename=
                "machines.csv",
        )
    )


    assert (
        result.execution_status
        ==
        "skipped"
    )

    assert (
        result.metrics[
            "valid_observations"
        ]
        ==
        7
    )


def test_missing_validated_runtime_column_is_rejected() -> None:
    dataframe = pd.DataFrame(
        {
            "machine_id":
                [
                    f"m_{index}"
                    for index
                    in range(
                        1,
                        10,
                    )
                ],

            "other_metric":
                list(
                    range(
                        1,
                        10,
                    )
                ),
        }
    )


    with pytest.raises(
        ValueError,
        match=
            "Validated runtime column is missing",
    ):
        execute_entity_outlier_contract(
            contract=
                build_contract(),

            dataframe=
                dataframe,

            dataset_id=
                "dataset:machines",

            dataset_filename=
                "machines.csv",
        )
