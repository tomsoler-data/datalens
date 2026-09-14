from __future__ import annotations


from typing import (
    Any,
)


import pandas as pd


from app.execution.schemas import (
    ExecutedAnalysis,
)

from app.planning.analytical_contract import (
    AnalyticalContract,
)


# ============================================================
# VERSION
# ============================================================


ENTITY_OUTLIER_EXECUTION_RULE_VERSION = (
    "entity_outlier_executor_v0.1"
)


# ============================================================
# CONSTANTS
# ============================================================


MIN_VALID_ENTITY_OBSERVATIONS = 8

IQR_MULTIPLIER = 1.5

MAX_ENTITY_OUTLIER_EVIDENCE_ROWS = 100


# ============================================================
# CONTRACT BINDINGS
# ============================================================


def exact_entity_outlier_bindings(
    contract: AnalyticalContract,
) -> tuple[
    str,
    str,
]:
    if (
        contract.status
        !=
        "validated"
    ):
        raise ValueError(
            (
                "Entity-outlier execution requires a "
                "contract already validated by Python."
            )
        )


    if (
        contract.family
        !=
        "entity_outlier"
    ):
        raise ValueError(
            (
                "The entity-outlier executor accepts only "
                "`entity_outlier` contracts."
            )
        )


    by_role: dict[
        str,
        list[
            str
        ],
    ] = {}


    for binding in (
        contract.bindings
    ):
        by_role.setdefault(
            binding.role,
            [],
        ).append(
            binding.column
        )


    if (
        set(
            by_role
        )
        !=
        {
            "entity",
            "value",
        }
    ):
        raise ValueError(
            (
                "An entity_outlier contract must contain "
                "only entity and value bindings."
            )
        )


    if (
        len(
            by_role[
                "entity"
            ]
        )
        !=
        1
        or
        len(
            by_role[
                "value"
            ]
        )
        !=
        1
    ):
        raise ValueError(
            (
                "An entity_outlier contract requires "
                "exactly one entity binding and exactly "
                "one value binding."
            )
        )


    return (
        by_role[
            "entity"
        ][
            0
        ],
        by_role[
            "value"
        ][
            0
        ],
    )


# ============================================================
# RESULT HELPERS
# ============================================================


def build_entity_outlier_result(
    *,
    contract: AnalyticalContract,
    dataset_id: str,
    dataset_filename: str,
    execution_status: str,
    summary: list[
        str
    ] | None = None,
    metrics: dict[
        str,
        Any
    ] | None = None,
    chart_data: list[
        dict[
            str,
            Any,
        ]
    ] | None = None,
    statistical_result: dict[
        str,
        Any,
    ] | None = None,
    warnings: list[
        str
    ] | None = None,
    limitations: list[
        str
    ] | None = None,
) -> ExecutedAnalysis:
    return ExecutedAnalysis(
        analysis_id=(
            "ai_tool:"
            f"{contract.contract_id}"
        ),

        dataset_id=
            dataset_id,

        dataset_filename=
            dataset_filename,

        title=
            contract.title,

        family=
            "entity_outlier",

        planned_readiness=
            "executable_now",

        execution_status=
            execution_status,

        chart_type=
            "bar",

        summary=(
            summary
            or []
        ),

        metrics=(
            metrics
            or {}
        ),

        chart_data=(
            chart_data
            or []
        ),

        statistical_decision=
            None,

        statistical_result=
            statistical_result,

        visualization=
            None,

        warnings=(
            warnings
            or []
        ),

        limitations=(
            limitations
            or []
        ),

        execution_rule_version=(
            ENTITY_OUTLIER_EXECUTION_RULE_VERSION
        ),
    )


# ============================================================
# CANONICAL ENTITY OUTLIER EXECUTOR
# ============================================================


def execute_entity_outlier_contract(
    *,
    contract: AnalyticalContract,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_filename: str,
) -> ExecutedAnalysis:
    (
        entity_column,
        value_column,
    ) = (
        exact_entity_outlier_bindings(
            contract
        )
    )


    if (
        contract.required_dataset_ids
        !=
        [
            dataset_id
        ]
    ):
        raise ValueError(
            (
                "Runtime dataset_id does not exactly "
                "match the validated contract."
            )
        )


    if (
        contract.required_dataset_filenames
        and
        dataset_filename
        not in
        contract.required_dataset_filenames
    ):
        raise ValueError(
            (
                "Runtime dataset filename does not match "
                "the validated contract."
            )
        )


    for column in (
        entity_column,
        value_column,
    ):
        if (
            column
            not in
            dataframe.columns
        ):
            raise ValueError(
                (
                    "Validated runtime column is missing: "
                    f"{column}"
                )
            )


    # --------------------------------------------------------
    # GRAIN SAFETY
    #
    # v0.1 deliberately refuses to aggregate repeated entity
    # rows. If the dataset is not already at the validated
    # entity grain, DataLens abstains.
    # --------------------------------------------------------

    non_null_entities = (
        dataframe[
            entity_column
        ]
        .dropna()
    )


    if (
        non_null_entities
        .duplicated()
        .any()
    ):
        return build_entity_outlier_result(
            contract=
                contract,

            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            execution_status=
                "needs_information",

            summary=[
                (
                    "Entity-outlier execution was not run "
                    "because the validated entity column "
                    "contains repeated entities."
                ),
            ],

            metrics={
                "entity_column":
                    entity_column,

                "value_column":
                    value_column,

                "method":
                    "iqr",
            },

            warnings=[
                (
                    "The v0.1 executor requires exactly "
                    "one row per entity and never performs "
                    "an implicit aggregation."
                ),
            ],

            limitations=[
                (
                    "A validated entity-level analytical "
                    "view is required before execution."
                ),
            ],
        )


    numeric = (
        pd.to_numeric(
            dataframe[
                value_column
            ],
            errors=
                "coerce",
        )
    )


    valid_mask = (
        dataframe[
            entity_column
        ]
        .notna()
        &
        numeric.notna()
    )


    valid_values = (
        numeric[
            valid_mask
        ]
        .astype(
            float
        )
    )


    valid_observations = int(
        len(
            valid_values
        )
    )


    if (
        valid_observations
        <
        MIN_VALID_ENTITY_OBSERVATIONS
    ):
        return build_entity_outlier_result(
            contract=
                contract,

            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            execution_status=
                "skipped",

            summary=[
                (
                    "Too few valid entity observations "
                    "are available for IQR outlier "
                    "detection."
                ),
            ],

            metrics={
                "entity_column":
                    entity_column,

                "value_column":
                    value_column,

                "method":
                    "iqr",

                "valid_observations":
                    valid_observations,

                "minimum_required":
                    MIN_VALID_ENTITY_OBSERVATIONS,
            },

            warnings=[
                (
                    "Entity-outlier detection requires "
                    "at least 8 valid entity observations."
                ),
            ],
        )


    if (
        int(
            valid_values.nunique()
        )
        <=
        1
    ):
        return build_entity_outlier_result(
            contract=
                contract,

            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            execution_status=
                "skipped",

            summary=[
                (
                    "The validated metric has no usable "
                    "variation for IQR outlier detection."
                ),
            ],

            metrics={
                "entity_column":
                    entity_column,

                "value_column":
                    value_column,

                "method":
                    "iqr",

                "valid_observations":
                    valid_observations,
            },
        )


    q1 = float(
        valid_values.quantile(
            0.25
        )
    )

    q3 = float(
        valid_values.quantile(
            0.75
        )
    )

    iqr = float(
        q3
        -
        q1
    )


    if (
        iqr
        <=
        0.0
    ):
        return build_entity_outlier_result(
            contract=
                contract,

            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            execution_status=
                "skipped",

            summary=[
                (
                    "The validated metric has a zero "
                    "interquartile range."
                ),
            ],

            metrics={
                "entity_column":
                    entity_column,

                "value_column":
                    value_column,

                "method":
                    "iqr",

                "valid_observations":
                    valid_observations,

                "q1":
                    q1,

                "q3":
                    q3,

                "iqr":
                    iqr,
            },
        )


    lower_bound = float(
        q1
        -
        (
            IQR_MULTIPLIER
            *
            iqr
        )
    )

    upper_bound = float(
        q3
        +
        (
            IQR_MULTIPLIER
            *
            iqr
        )
    )


    evidence: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for index in (
        dataframe.index[
            valid_mask
        ]
    ):
        entity = str(
            dataframe.at[
                index,
                entity_column,
            ]
        )

        value = float(
            numeric.at[
                index
            ]
        )


        if (
            value
            <
            lower_bound
        ):
            direction = "low"

            distance_iqr = float(
                (
                    lower_bound
                    -
                    value
                )
                /
                iqr
            )


        elif (
            value
            >
            upper_bound
        ):
            direction = "high"

            distance_iqr = float(
                (
                    value
                    -
                    upper_bound
                )
                /
                iqr
            )


        else:
            continue


        anomaly_score = float(
            1.0
            +
            distance_iqr
        )


        evidence.append(
            {
                "entity":
                    entity,

                "value":
                    value,

                "direction":
                    direction,

                "distance_iqr":
                    distance_iqr,

                "anomaly_score":
                    anomaly_score,
            }
        )


    evidence.sort(
        key=lambda row: (
            -float(
                row[
                    "anomaly_score"
                ]
            ),
            str(
                row[
                    "entity"
                ]
            ),
        )
    )


    flagged_entity_count = int(
        len(
            evidence
        )
    )


    flagged_ratio = float(
        flagged_entity_count
        /
        valid_observations
    )


    returned_evidence = (
        evidence[
            :MAX_ENTITY_OUTLIER_EVIDENCE_ROWS
        ]
    )


    metrics = {
        "method":
            "iqr",

        "iqr_multiplier":
            IQR_MULTIPLIER,

        "entity_column":
            entity_column,

        "value_column":
            value_column,

        "valid_observations":
            valid_observations,

        "entity_count":
            int(
                non_null_entities.nunique()
            ),

        "q1":
            q1,

        "q3":
            q3,

        "iqr":
            iqr,

        "lower_bound":
            lower_bound,

        "upper_bound":
            upper_bound,

        "flagged_entity_count":
            flagged_entity_count,

        "flagged_ratio":
            flagged_ratio,

        "returned_evidence_count":
            len(
                returned_evidence
            ),
    }


    summary = [
        (
            f"{flagged_entity_count} entity outlier(s) "
            f"were detected among "
            f"{valid_observations} valid entities "
            f"for `{value_column}`."
        ),
        (
            "The deterministic method used the "
            "1.5 x IQR rule."
        ),
    ]


    return build_entity_outlier_result(
        contract=
            contract,

        dataset_id=
            dataset_id,

        dataset_filename=
            dataset_filename,

        execution_status=
            "complete",

        summary=
            summary,

        metrics=
            metrics,

        chart_data=
            returned_evidence,

        statistical_result={
            "method":
                "iqr",

            "metric":
                value_column,

            "thresholds":
                {
                    "q1":
                        q1,

                    "q3":
                        q3,

                    "iqr":
                        iqr,

                    "lower_bound":
                        lower_bound,

                    "upper_bound":
                        upper_bound,
                },

            "flagged_entity_count":
                flagged_entity_count,

            "evidence":
                returned_evidence,
        },

        warnings=[],

        limitations=[
            (
                "IQR outliers are statistical signals, "
                "not automatic fraud, error or deletion "
                "labels."
            ),
            (
                "This v0.1 contract evaluates exactly "
                "one quantitative metric selected by "
                "the AI planner."
            ),
        ],
    )
