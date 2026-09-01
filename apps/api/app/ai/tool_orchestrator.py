from __future__ import annotations


from numbers import (
    Number,
)

from typing import (
    Any,
    Literal,
)


import pandas as pd

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from app.execution.executor import (
    execute_analysis_candidate,
)

from app.execution.schemas import (
    ExecutedAnalysis,
)

from app.planning.ai_analytical_planner import (
    AIPlannerReport,
)

from app.planning.analytical_contract import (
    AnalyticalContract,
)

from app.planning.schemas import (
    AnalysisCandidate,
    PlannedVariable,
)


# ============================================================
# VERSION
# ============================================================

AI_TOOL_ORCHESTRATOR_RULE_VERSION = (
    "ai_tool_orchestrator_v0.4"
)


AI_TIME_SERIES_EXECUTION_RULE_VERSION = (
    "ai_tool_time_series_v0.2"
)


AI_AGGREGATION_EXECUTION_RULE_VERSION = (
    "ai_tool_aggregation_v0.3"
)


AI_RANKING_EXECUTION_RULE_VERSION = (
    "ai_tool_ranking_v0.1"
)


# ============================================================
# TOOL VOCABULARY
# ============================================================

AIToolName = Literal[
    "run_quantitative_association",
    "run_categorical_association",
    "run_group_comparison",
    "run_time_series",
    "run_distribution",
    "run_aggregation",
    "run_ranking",
]


AIToolExecutionStatus = Literal[
    "executed",
    "not_executed",
    "not_supported",
    "rejected",
]


class AIToolCapability(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    tool_name: str

    family: str

    enabled: bool

    chart_type: (
        str
        | None
    ) = None

    statistical_strategy: (
        str
        | None
    ) = None

    reason: str


class AIToolCallTrace(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    call_index: int = Field(
        ge=1
    )

    contract_id: str

    family: str

    tool_name: (
        str
        | None
    )

    selection_source: Literal[
        "validated_contract_family"
    ] = (
        "validated_contract_family"
    )

    execution_status: (
        AIToolExecutionStatus
    )

    dataset_id: (
        str
        | None
    ) = None

    dataset_filename: (
        str
        | None
    ) = None

    arguments: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    result: (
        ExecutedAnalysis
        | None
    ) = None

    errors: list[
        str
    ] = Field(
        default_factory=list
    )

    warnings: list[
        str
    ] = Field(
        default_factory=list
    )


class AIToolOrchestrationReport(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    status: Literal[
        "ready"
    ] = "ready"

    planner: AIPlannerReport

    validated_contract_count: int

    tool_call_count: int

    executed_count: int

    not_executed_count: int

    not_supported_count: int

    rejected_count: int

    tool_calls: list[
        AIToolCallTrace
    ]

    orchestrator_notes: list[
        str
    ] = Field(
        default_factory=list
    )

    orchestrator_rule_version: str = (
        AI_TOOL_ORCHESTRATOR_RULE_VERSION
    )


# ============================================================
# CAPABILITY REGISTRY
#
# The registry is deterministic. The model does not receive
# arbitrary Python execution rights.
# ============================================================

TOOL_CAPABILITIES: dict[
    str,
    AIToolCapability,
] = {
    "quantitative_association":
        AIToolCapability(
            tool_name=(
                "run_quantitative_association"
            ),
            family=(
                "quantitative_association"
            ),
            enabled=True,
            chart_type="scatter",
            statistical_strategy=(
                "correlation_decision_engine"
            ),
            reason=(
                "The deterministic analysis executor "
                "already supports quantitative association."
            ),
        ),

    "categorical_association":
        AIToolCapability(
            tool_name=(
                "run_categorical_association"
            ),
            family=(
                "categorical_association"
            ),
            enabled=True,
            chart_type="heatmap",
            statistical_strategy=(
                "chi_square_or_fisher_decision_engine"
            ),
            reason=(
                "The deterministic analysis executor "
                "already supports categorical association."
            ),
        ),

    "group_comparison":
        AIToolCapability(
            tool_name=(
                "run_group_comparison"
            ),
            family=(
                "group_comparison"
            ),
            enabled=True,
            chart_type="boxplot",
            statistical_strategy=(
                "automatic_group_comparison_engine"
            ),
            reason=(
                "The deterministic analysis executor "
                "can produce the current group-comparison "
                "result without LLM-side calculation."
            ),
        ),

    "time_series":
        AIToolCapability(
            tool_name=(
                "run_time_series"
            ),
            family=(
                "time_series"
            ),
            enabled=True,
            chart_type="line",
            statistical_strategy=(
                "deterministic_aggregation_by_period"
            ),
            reason=(
                "The AI-native time-series boundary executes "
                "the exact validated aggregation by period. "
                "Median contracts additionally expose a "
                "deterministic Q1/Q3 band."
            ),
        ),

    "distribution":
        AIToolCapability(
            tool_name=(
                "run_distribution"
            ),
            family=(
                "distribution"
            ),
            enabled=True,
            chart_type="histogram",
            statistical_strategy=(
                "descriptive_distribution"
            ),
            reason=(
                "The deterministic analysis executor "
                "already supports distributions."
            ),
        ),

    "descriptive_metric":
        AIToolCapability(
            tool_name=(
                "run_descriptive_metric"
            ),
            family=(
                "descriptive_metric"
            ),
            enabled=False,
            reason=(
                "The generic AI tool layer does not yet "
                "have a canonical descriptive-metric tool."
            ),
        ),

    "aggregation":
        AIToolCapability(
            tool_name=(
                "run_aggregation"
            ),
            family=(
                "aggregation"
            ),
            enabled=True,
            chart_type="bar",
            statistical_strategy=(
                "deterministic_grouped_aggregation"
            ),
            reason=(
                "The canonical aggregation contract is executed "
                "directly by Python from AggregationSpec."
            ),
        ),

    "ranking":
        AIToolCapability(
            tool_name=(
                "run_ranking"
            ),
            family=(
                "ranking"
            ),
            enabled=True,
            chart_type="bar",
            statistical_strategy=(
                "deterministic_grouped_ranking"
            ),
            reason=(
                "The canonical ranking contract is executed "
                "directly by Python from AggregationSpec and "
                "RankingSpec."
            ),
        ),

    "inequality":
        AIToolCapability(
            tool_name=(
                "run_inequality"
            ),
            family=(
                "inequality"
            ),
            enabled=False,
            reason=(
                "Generic concentration/inequality execution "
                "will be migrated from the legacy requested "
                "executor in a later step."
            ),
        ),

    "data_quality":
        AIToolCapability(
            tool_name=(
                "run_data_quality"
            ),
            family=(
                "data_quality"
            ),
            enabled=False,
            reason=(
                "Data-quality checks exist elsewhere but are "
                "not yet exposed through the AI tool contract."
            ),
        ),

    "unresolved":
        AIToolCapability(
            tool_name=(
                "none"
            ),
            family=(
                "unresolved"
            ),
            enabled=False,
            reason=(
                "Unresolved plans are never executable."
            ),
        ),
}


# ============================================================
# DATASET RESOLUTION
# ============================================================

def build_dataset_map(
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    dict[
        str,
        Any,
    ],
]:
    result: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}


    for dataset in (
        datasets
    ):
        dataset_id = str(
            dataset.get(
                "dataset_id",
                "",
            )
        )


        if not dataset_id:
            continue


        result[
            dataset_id
        ] = dataset


    return result


def resolve_contract_dataset(
    *,
    contract: AnalyticalContract,
    dataset_map: dict[
        str,
        dict[
            str,
            Any,
        ],
    ],
) -> tuple[
    dict[
        str,
        Any,
    ]
    | None,
    str
    | None,
]:
    dataset_ids = list(
        contract
        .required_dataset_ids
    )


    if (
        len(
            dataset_ids
        )
        !=
        1
    ):
        return (
            None,
            (
                "AI tool execution v0.4 requires exactly "
                "one validated dataset_id."
            ),
        )


    dataset_id = (
        dataset_ids[
            0
        ]
    )


    record = (
        dataset_map.get(
            dataset_id
        )
    )


    if (
        record is None
    ):
        return (
            None,
            (
                "The validated contract references a dataset "
                f"that is not available for execution: {dataset_id}."
            ),
        )


    dataframe = (
        record.get(
            "dataframe"
        )
    )


    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        return (
            None,
            (
                "The resolved dataset does not contain an "
                "executable pandas DataFrame."
            ),
        )


    expected_filenames = (
        contract
        .required_dataset_filenames
    )


    if (
        expected_filenames
        and
        str(
            record.get(
                "filename",
                "",
            )
        )
        not in expected_filenames
    ):
        return (
            None,
            (
                "The dataset filename no longer matches the "
                "filename validated by the AI planner."
            ),
        )


    return (
        record,
        None,
    )


# ============================================================
# CONTRACT → EXISTING DETERMINISTIC CANDIDATE
#
# This remains a compatibility adapter for the families that
# are already faithfully represented by AnalysisCandidate.
#
# time_series is intentionally NOT executed through this
# compatibility path because AnalysisCandidate does not carry
# the canonical AggregationSpec. The canonical executor below
# therefore executes the exact validated aggregation directly.
# ============================================================

def normalize_candidate_role(
    *,
    family: str,
    role: str,
) -> str:
    if (
        family ==
        "distribution"
    ):
        return "value"


    return role


def contract_to_analysis_candidate(
    contract: AnalyticalContract,
    *,
    dataset_filename: str,
) -> AnalysisCandidate:
    capability = (
        TOOL_CAPABILITIES.get(
            contract.family
        )
    )


    if (
        capability is None
        or
        not capability.enabled
    ):
        raise ValueError(
            (
                "No enabled deterministic tool exists for "
                f"family `{contract.family}`."
            )
        )


    if (
        capability.chart_type
        is None
    ):
        raise ValueError(
            (
                "The enabled tool capability does not define "
                "a chart type."
            )
        )


    variables: list[
        PlannedVariable
    ] = []


    for binding in (
        contract.bindings
    ):
        role = (
            normalize_candidate_role(
                family=(
                    contract.family
                ),
                role=(
                    binding.role
                ),
            )
        )


        if (
            role
            not in {
                "x",
                "y",
                "time",
                "value",
                "group",
                "category",
            }
        ):
            raise ValueError(
                (
                    "The current deterministic executor "
                    "cannot represent binding role "
                    f"`{binding.role}` for family "
                    f"`{contract.family}`."
                )
            )


        variables.append(
            PlannedVariable(
                column=(
                    binding.column
                ),
                role=role,  # type: ignore[arg-type]
                analysis_kind=(
                    binding.analysis_kind
                    or
                    "unknown"
                ),
            )
        )


    return (
        AnalysisCandidate(
            analysis_id=(
                "ai_tool:"
                f"{contract.contract_id}"
            ),
            dataset_id=(
                contract
                .required_dataset_ids[
                    0
                ]
            ),
            dataset_filename=(
                dataset_filename
            ),
            title=(
                contract.title
            ),
            family=(
                contract.family
            ),  # type: ignore[arg-type]
            priority_score=100,
            readiness=(
                "executable_now"
            ),
            variables=(
                variables
            ),
            chart_type=(
                capability
                .chart_type
            ),  # type: ignore[arg-type]
            statistical_strategy=(
                capability
                .statistical_strategy
            ),
            reasons=[
                (
                    "The analytical family and variable roles "
                    "came from an AI-generated contract that "
                    "was validated deterministically before "
                    "tool execution."
                ),
                (
                    "Execution is delegated to the existing "
                    "deterministic Python analysis engine."
                ),
            ],
            limitations=[
                (
                    "AI tool orchestration v0.4 supports only "
                    "single-dataset contracts and does not "
                    "allow LLM-generated joins, derived "
                    "variables or arbitrary Python code."
                ),
            ],
        )
    )


# ============================================================
# CONTRACT HELPERS
# ============================================================

def contract_binding_map(
    contract: AnalyticalContract,
) -> dict[
    str,
    str,
]:
    return {
        binding.role:
            binding.column

        for binding
        in contract.bindings
    }


def native_scalar(
    value: Any,
) -> Any:
    if isinstance(
        value,
        pd.Timestamp,
    ):
        return (
            value.isoformat()
        )


    if hasattr(
        value,
        "item",
    ):
        try:
            return (
                value.item()
            )
        except Exception:
            pass


    return value


def period_sort_key(
    value: Any,
) -> tuple[
    int,
    Any,
]:
    if isinstance(
        value,
        Number,
    ) and not isinstance(
        value,
        bool,
    ):
        return (
            0,
            float(
                value
            ),
        )


    parsed = pd.to_datetime(
        value,
        errors="coerce",
    )


    if not pd.isna(
        parsed
    ):
        return (
            1,
            int(
                parsed.value
            ),
        )


    return (
        2,
        str(
            value
        ),
    )


# ============================================================
# CANONICAL AI TIME-SERIES EXECUTION
# ============================================================

SUPPORTED_TIME_SERIES_AGGREGATIONS = frozenset(
    {
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "count",
        "distinct_count",
    }
)


def execute_time_series_contract(
    *,
    contract: AnalyticalContract,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_filename: str,
) -> ExecutedAnalysis:
    """
    Execute the canonical time-series AggregationSpec.

    The contract remains authoritative:

        time binding
            -> grouping period

        value binding
            -> aggregation source

        aggregation.function
            -> deterministic reducer

    Median preserves the historical Q1/Q3 band. Other
    supported reducers return a standard period/value line.
    """

    if (
        contract.family
        !=
        "time_series"
    ):
        raise ValueError(
            (
                "The canonical time-series executor only "
                "accepts `time_series` contracts."
            )
        )


    aggregation = (
        contract.aggregation
    )


    if (
        aggregation is None
        or
        aggregation.function
        not in SUPPORTED_TIME_SERIES_AGGREGATIONS
        or
        aggregation.source_role
        !=
        "value"
        or
        list(
            aggregation.group_by_roles
        )
        !=
        [
            "time",
        ]
    ):
        raise ValueError(
            (
                "AI time-series v0.4 requires a canonical "
                "AggregationSpec with a supported function, "
                "source_role:value and "
                "group_by_roles:[time]."
            )
        )


    bindings = (
        contract_binding_map(
            contract
        )
    )


    time_column = (
        bindings.get(
            "time"
        )
    )


    value_column = (
        bindings.get(
            "value"
        )
    )


    if (
        time_column is None
        or
        value_column is None
    ):
        raise ValueError(
            (
                "The validated time-series contract "
                "must contain time and value bindings."
            )
        )


    missing_columns = [
        column

        for column
        in [
            time_column,
            value_column,
        ]

        if (
            column
            not in dataframe.columns
        )
    ]


    if (
        missing_columns
    ):
        raise ValueError(
            (
                "Validated time-series column(s) are "
                "missing at execution time: "
                +
                ", ".join(
                    missing_columns
                )
            )
        )


    function = (
        aggregation.function
    )


    working = pd.DataFrame(
        {
            "__time":
                dataframe[
                    time_column
                ],

            "__value":
                dataframe[
                    value_column
                ],
        }
    )


    working = (
        working
        .dropna(
            subset=[
                "__time",
            ]
        )
        .copy()
    )


    if (
        function
        in {
            "sum",
            "mean",
            "median",
            "min",
            "max",
        }
    ):
        working[
            "__value"
        ] = pd.to_numeric(
            working[
                "__value"
            ],
            errors="coerce",
        )


        working = (
            working
            .dropna(
                subset=[
                    "__value",
                ]
            )
        )


    else:
        working = (
            working
            .dropna(
                subset=[
                    "__value",
                ]
            )
        )


    chart_type = (
        "line_band"
        if (
            function
            ==
            "median"
        )
        else
        "line"
    )


    if (
        working.empty
    ):
        return (
            ExecutedAnalysis(
                analysis_id=(
                    "ai_tool:"
                    f"{contract.contract_id}"
                ),
                dataset_id=(
                    dataset_id
                ),
                dataset_filename=(
                    dataset_filename
                ),
                title=(
                    contract.title
                ),
                family=(
                    "time_series"
                ),
                planned_readiness=(
                    "executable_now"
                ),
                execution_status=(
                    "skipped"
                ),
                chart_type=(
                    chart_type
                ),
                summary=[
                    (
                        "Aucune observation complète "
                        "time/value n'est disponible."
                    ),
                ],
                metrics={
                    "time_column":
                        time_column,

                    "value_column":
                        value_column,

                    "aggregation_function":
                        function,

                    "valid_observations":
                        0,

                    "period_count":
                        0,
                },
                chart_data=[],
                warnings=[
                    (
                        "The validated contract could not "
                        "produce a temporal profile because "
                        "all required time/value pairs were "
                        "missing or invalid."
                    ),
                ],
                limitations=[
                    (
                        "No inferential temporal model was "
                        "executed."
                    ),
                ],
                execution_rule_version=(
                    AI_TIME_SERIES_EXECUTION_RULE_VERSION
                ),
            )
        )


    def reduce_values(
        values: pd.Series,
    ) -> float | int:
        if (
            function
            ==
            "count"
        ):
            return int(
                values.count()
            )


        if (
            function
            ==
            "distinct_count"
        ):
            return int(
                values.nunique(
                    dropna=True
                )
            )


        if (
            function
            ==
            "sum"
        ):
            return float(
                values.sum()
            )


        if (
            function
            ==
            "mean"
        ):
            return float(
                values.mean()
            )


        if (
            function
            ==
            "median"
        ):
            return float(
                values.median()
            )


        if (
            function
            ==
            "min"
        ):
            return float(
                values.min()
            )


        if (
            function
            ==
            "max"
        ):
            return float(
                values.max()
            )


        raise ValueError(
            (
                "Unsupported deterministic time-series "
                f"aggregation: {function}."
            )
        )


    grouped_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for (
        period,
        group
    ) in working.groupby(
        "__time",
        dropna=True,
        sort=False,
    ):
        values = (
            group[
                "__value"
            ]
        )


        reduced_value = (
            reduce_values(
                values
            )
        )


        row: dict[
            str,
            Any,
        ] = {
            "period":
                native_scalar(
                    period
                ),

            "value":
                native_scalar(
                    reduced_value
                ),

            "count":
                int(
                    values.count()
                ),
        }


        if (
            function
            ==
            "median"
        ):
            numeric_values = pd.to_numeric(
                values,
                errors="coerce",
            ).dropna()


            row.update(
                {
                    "median":
                        float(
                            numeric_values.median()
                        ),

                    "q1":
                        float(
                            numeric_values.quantile(
                                0.25
                            )
                        ),

                    "q3":
                        float(
                            numeric_values.quantile(
                                0.75
                            )
                        ),
                }
            )


        grouped_rows.append(
            row
        )


    grouped_rows.sort(
        key=lambda row:
            period_sort_key(
                row[
                    "period"
                ]
            )
    )


    period_count = (
        len(
            grouped_rows
        )
    )


    if (
        period_count
        <
        2
    ):
        status = (
            "descriptive_only"
        )

        warnings = [
            (
                "Only one distinct period is available; "
                "the requested temporal evolution cannot "
                "be interpreted as a trend."
            ),
        ]


    else:
        status = (
            "complete"
        )

        warnings = []


    period_values = [
        float(
            row[
                "value"
            ]
        )

        for row
        in grouped_rows

        if isinstance(
            row.get(
                "value"
            ),
            Number,
        )
    ]


    summary = [
        (
            f"{period_count} période(s) distincte(s) "
            f"ont été agrégées avec `{function}` pour "
            f"{value_column}."
        ),
    ]


    if (
        function
        ==
        "median"
    ):
        summary.append(
            (
                "Les quartiles Q1 et Q3 sont calculés "
                "pour chaque période afin de représenter "
                "la dispersion autour de la médiane."
            )
        )


    limitations = [
        (
            "This AI-native time-series execution is "
            "descriptive. It does not fit a forecasting, "
            "causal or inferential time-series model."
        ),
    ]


    if (
        function
        ==
        "median"
    ):
        limitations.append(
            (
                "Median contracts add Q1/Q3 as deterministic "
                "descriptive dispersion summaries."
            )
        )


    return (
        ExecutedAnalysis(
            analysis_id=(
                "ai_tool:"
                f"{contract.contract_id}"
            ),
            dataset_id=(
                dataset_id
            ),
            dataset_filename=(
                dataset_filename
            ),
            title=(
                contract.title
            ),
            family=(
                "time_series"
            ),
            planned_readiness=(
                "executable_now"
            ),
            execution_status=(
                status
            ),
            chart_type=(
                chart_type
            ),
            summary=(
                summary
            ),
            metrics={
                "time_column":
                    time_column,

                "value_column":
                    value_column,

                "aggregation_function":
                    function,

                "aggregation_group_role":
                    "time",

                "source_observation_count":
                    int(
                        len(
                            working
                        )
                    ),

                "valid_observations":
                    int(
                        len(
                            working
                        )
                    ),

                "period_count":
                    period_count,

                "chart_point_count":
                    period_count,

                "time_start":
                    (
                        grouped_rows[
                            0
                        ][
                            "period"
                        ]
                        if grouped_rows
                        else None
                    ),

                "time_end":
                    (
                        grouped_rows[
                            -1
                        ][
                            "period"
                        ]
                        if grouped_rows
                        else None
                    ),

                "period_value_min":
                    (
                        min(
                            period_values
                        )
                        if period_values
                        else None
                    ),

                "period_value_max":
                    (
                        max(
                            period_values
                        )
                        if period_values
                        else None
                    ),
            },
            chart_data=(
                grouped_rows
            ),
            statistical_decision=None,
            statistical_result=None,
            visualization=None,
            warnings=(
                warnings
            ),
            limitations=(
                limitations
            ),
            execution_rule_version=(
                AI_TIME_SERIES_EXECUTION_RULE_VERSION
            ),
        )
    )


# Backward-compatible executor name retained for existing
# imports/tests. It still refuses non-median contracts by name.
def execute_median_time_series_contract(
    *,
    contract: AnalyticalContract,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_filename: str,
) -> ExecutedAnalysis:
    aggregation = (
        contract.aggregation
    )


    if (
        aggregation is None
        or
        aggregation.function
        !=
        "median"
    ):
        raise ValueError(
            (
                "execute_median_time_series_contract() "
                "accepts only median contracts."
            )
        )


    return (
        execute_time_series_contract(
            contract=(
                contract
            ),
            dataframe=(
                dataframe
            ),
            dataset_id=(
                dataset_id
            ),
            dataset_filename=(
                dataset_filename
            ),
        )
    )


# ============================================================
# CANONICAL AGGREGATION / RANKING EXECUTION
# ============================================================

def aggregation_contract_columns(
    contract: AnalyticalContract,
) -> tuple[
    str | None,
    list[str],
]:
    aggregation = (
        contract.aggregation
    )


    if (
        aggregation is None
    ):
        raise ValueError(
            f"A {contract.family} contract requires AggregationSpec."
        )


    bindings = (
        contract_binding_map(
            contract
        )
    )


    source_column: str | None = (
        None
    )


    if (
        aggregation.source_role
        is not None
    ):
        source_column = (
            bindings.get(
                aggregation.source_role
            )
        )


        if (
            source_column is None
        ):
            raise ValueError(
                "The aggregation source role is not bound "
                "to an execution column."
            )


    group_columns: list[str] = []


    for role in (
        aggregation.group_by_roles
    ):
        column = (
            bindings.get(
                role
            )
        )


        if (
            column is None
        ):
            raise ValueError(
                "The aggregation group role is not bound "
                f"to an execution column: {role}."
            )


        group_columns.append(
            column
        )


    return (
        source_column,
        group_columns,
    )


def validate_runtime_columns(
    *,
    dataframe: pd.DataFrame,
    columns: list[str],
) -> None:
    missing = [
        column
        for column
        in columns
        if column not in dataframe.columns
    ]


    if missing:
        raise ValueError(
            "Validated analytical column(s) are missing at "
            "execution time: "
            + ", ".join(
                missing
            )
        )


def aggregate_contract_rows(
    *,
    contract: AnalyticalContract,
    dataframe: pd.DataFrame,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    aggregation = contract.aggregation


    if aggregation is None:
        raise ValueError(
            f"A {contract.family} contract requires AggregationSpec."
        )


    (
        source_column,
        group_columns,
    ) = aggregation_contract_columns(
        contract
    )


    required_columns = [
        *group_columns,
        *(
            [source_column]
            if source_column is not None
            else []
        ),
    ]


    validate_runtime_columns(
        dataframe=dataframe,
        columns=required_columns,
    )


    function = aggregation.function


    working = dataframe[
        required_columns
    ].copy() if required_columns else dataframe.copy()


    if group_columns:
        working = working.dropna(
            subset=group_columns
        )


    if (
        function in {
            "sum",
            "mean",
            "median",
            "min",
            "max",
        }
    ):
        if source_column is None:
            raise ValueError(
                f"Aggregation `{function}` requires a source column."
            )


        working[source_column] = pd.to_numeric(
            working[source_column],
            errors="coerce",
        )


        working = working.dropna(
            subset=[
                source_column,
            ]
        )


    elif (
        function ==
        "distinct_count"
    ):
        if source_column is None:
            raise ValueError(
                "distinct_count requires a source column."
            )


    elif (
        function !=
        "count"
    ):
        raise ValueError(
            f"Unsupported aggregation function: {function}."
        )


    source_observation_count = int(
        len(
            working
        )
    )


    def reduce_frame(
        frame: pd.DataFrame,
    ) -> float | int:
        if function == "count":
            return int(
                len(
                    frame
                )
            )


        assert source_column is not None
        series = frame[
            source_column
        ]


        if function == "distinct_count":
            return int(
                series.nunique(
                    dropna=True
                )
            )


        if function == "sum":
            return float(
                series.sum()
            )


        if function == "mean":
            return float(
                series.mean()
            )


        if function == "median":
            return float(
                series.median()
            )


        if function == "min":
            return float(
                series.min()
            )


        if function == "max":
            return float(
                series.max()
            )


        raise ValueError(
            f"Unsupported aggregation function: {function}."
        )


    rows: list[dict[str, Any]] = []


    if group_columns:
        grouper: Any = (
            group_columns[0]
            if len(group_columns) == 1
            else group_columns
        )


        for (
            key,
            group_frame,
        ) in working.groupby(
            grouper,
            dropna=False,
            sort=False,
        ):
            raw_key = (
                key
                if isinstance(key, tuple)
                else (key,)
            )


            group_values = {
                column: native_scalar(
                    raw_key[index]
                )
                for index, column
                in enumerate(
                    group_columns
                )
            }


            if len(group_columns) == 1:
                label = str(
                    native_scalar(
                        raw_key[0]
                    )
                )
            else:
                label = " · ".join(
                    str(
                        native_scalar(
                            value
                        )
                    )
                    for value
                    in raw_key
                )


            rows.append(
                {
                    "category": label,
                    "group": label,
                    "value": native_scalar(
                        reduce_frame(
                            group_frame
                        )
                    ),
                    "count": int(
                        len(
                            group_frame
                        )
                    ),
                    "group_values": group_values,
                }
            )


    else:
        rows.append(
            {
                "category": "Total",
                "group": "Total",
                "value": native_scalar(
                    reduce_frame(
                        working
                    )
                ),
                "count": int(
                    len(
                        working
                    )
                ),
                "group_values": {},
            }
        )


    metrics = {
        "aggregation_function": function,
        "aggregation_source_role": aggregation.source_role,
        "source_column": source_column,
        "group_by_columns": group_columns,
        "source_observation_count": source_observation_count,
        "valid_observations": source_observation_count,
        "group_count": len(rows),
        "result_count": len(rows),
    }


    return (
        rows,
        metrics,
    )


# ============================================================
# BENCHMARK EXECUTION
# DATALENS_BENCHMARK_EXECUTION_V0_1
# ============================================================

def overall_aggregate_contract_value(
    *,
    contract: AnalyticalContract,
    dataframe: pd.DataFrame,
) -> float | int:
    """
    Compute the benchmark reference with exactly the same:

    - source column,
    - aggregation function,
    - analytically eligible population,

    while suppressing only the grouping operation.

    Rows missing a grouping value are excluded first because they
    cannot belong to any grouped result. This prevents the global
    reference from silently using a broader population than the
    grouped metric.
    """

    aggregation = (
        contract.aggregation
    )


    if aggregation is None:
        raise ValueError(
            "Benchmark execution requires AggregationSpec."
        )


    (
        _,
        group_columns,
    ) = aggregation_contract_columns(
        contract
    )


    validate_runtime_columns(
        dataframe=
            dataframe,

        columns=
            group_columns,
    )


    if group_columns:
        reference_population = (
            dataframe
            .dropna(
                subset=
                    group_columns
            )
            .copy()
        )

    else:
        reference_population = (
            dataframe.copy()
        )


    reference_aggregation = (
        aggregation.model_copy(
            update={
                "group_by_roles":
                    [],
            }
        )
    )


    reference_contract = (
        contract.model_copy(
            update={
                "aggregation":
                    reference_aggregation,

                "benchmark":
                    None,
            }
        )
    )


    (
        reference_rows,
        _,
    ) = aggregate_contract_rows(
        contract=
            reference_contract,

        dataframe=
            reference_population,
    )


    if (
        len(
            reference_rows
        )
        !=
        1
    ):
        raise ValueError(
            "overall_aggregate benchmark must produce exactly "
            "one deterministic reference value."
        )


    reference_value = (
        reference_rows[
            0
        ][
            "value"
        ]
    )


    if (
        reference_value
        is None
    ):
        raise ValueError(
            "overall_aggregate benchmark reference is missing."
        )


    try:
        numeric_reference = float(
            reference_value
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "overall_aggregate benchmark reference is not numeric."
        ) from error


    if pd.isna(
        numeric_reference
    ):
        raise ValueError(
            "overall_aggregate benchmark reference is NaN."
        )


    if (
        isinstance(
            reference_value,
            int,
        )
        and
        not isinstance(
            reference_value,
            bool,
        )
    ):
        return int(
            reference_value
        )


    return float(
        numeric_reference
    )


def benchmark_comparison_matches(
    *,
    value: Any,
    reference: float | int,
    operator: str,
) -> bool:
    try:
        numeric_value = float(
            value
        )

        numeric_reference = float(
            reference
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Benchmark comparison requires numeric values."
        ) from error


    if (
        pd.isna(
            numeric_value
        )
        or
        pd.isna(
            numeric_reference
        )
    ):
        return False


    if operator == "gt":
        return (
            numeric_value
            >
            numeric_reference
        )


    if operator == "gte":
        return (
            numeric_value
            >=
            numeric_reference
        )


    if operator == "lt":
        return (
            numeric_value
            <
            numeric_reference
        )


    if operator == "lte":
        return (
            numeric_value
            <=
            numeric_reference
        )


    raise ValueError(
        f"Unsupported benchmark operator: {operator}."
    )


def apply_contract_benchmark(
    *,
    contract: AnalyticalContract,
    dataframe: pd.DataFrame,
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
    metrics: dict[
        str,
        Any,
    ],
) -> tuple[
    list[
        dict[
            str,
            Any,
        ]
    ],
    dict[
        str,
        Any,
    ],
]:
    benchmark = (
        contract.benchmark
    )


    if benchmark is None:
        return (
            rows,
            metrics,
        )


    if (
        benchmark.reference
        !=
        "overall_aggregate"
    ):
        raise ValueError(
            "Unsupported benchmark reference: "
            f"{benchmark.reference}."
        )


    reference_value = (
        overall_aggregate_contract_value(
            contract=
                contract,

            dataframe=
                dataframe,
        )
    )


    annotated_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for row in rows:
        matched = (
            benchmark_comparison_matches(
                value=
                    row.get(
                        "value"
                    ),

                reference=
                    reference_value,

                operator=
                    benchmark.operator,
            )
        )


        annotated_rows.append(
            {
                **row,

                "benchmark_reference":
                    benchmark.reference,

                "benchmark_value":
                    reference_value,

                "benchmark_operator":
                    benchmark.operator,

                "benchmark_match":
                    matched,
            }
        )


    matched_count = sum(
        1

        for row
        in annotated_rows

        if row[
            "benchmark_match"
        ]
    )


    if (
        benchmark.selection
        ==
        "matching_only"
    ):
        output_rows = [
            row

            for row
            in annotated_rows

            if row[
                "benchmark_match"
            ]
        ]


    elif (
        benchmark.selection
        ==
        "annotate_all"
    ):
        output_rows = (
            annotated_rows
        )


    else:
        raise ValueError(
            "Unsupported benchmark selection: "
            f"{benchmark.selection}."
        )


    benchmark_metrics = {
        **metrics,

        "benchmark_reference":
            benchmark.reference,

        "benchmark_operator":
            benchmark.operator,

        "benchmark_selection":
            benchmark.selection,

        "benchmark_value":
            reference_value,

        "pre_benchmark_result_count":
            len(
                rows
            ),

        "benchmark_matching_count":
            matched_count,

        "result_count":
            len(
                output_rows
            ),
    }


    return (
        output_rows,
        benchmark_metrics,
    )


def execute_aggregation_contract(
    *,
    contract: AnalyticalContract,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_filename: str,
) -> ExecutedAnalysis:
    if contract.family != "aggregation":
        raise ValueError(
            "The canonical aggregation executor only accepts "
            "`aggregation` contracts."
        )


    (
        base_rows,
        metrics,
    ) = aggregate_contract_rows(
        contract=
            contract,

        dataframe=
            dataframe,
    )


    aggregation = (
        contract.aggregation
    )

    assert (
        aggregation
        is not None
    )


    (
        rows,
        metrics,
    ) = apply_contract_benchmark(
        contract=
            contract,

        dataframe=
            dataframe,

        rows=
            base_rows,

        metrics=
            metrics,
    )


    benchmark = (
        contract.benchmark
    )


    if not base_rows:
        status = (
            "skipped"
        )

        summary = [
            (
                "Aucun r\u00e9sultat agr\u00e9g\u00e9 n'a pu \u00eatre calcul\u00e9."
            )
        ]


    elif benchmark is not None:
        status = (
            "complete"
        )

        summary = [
            (
                f"{len(base_rows)} r\u00e9sultat(s) ont \u00e9t\u00e9 calcul\u00e9s "
                f"avec l'agr\u00e9gation `{aggregation.function}`."
            ),
            (
                f"{metrics['benchmark_matching_count']} groupe(s) "
                "respectent le benchmark "
                f"`{benchmark.operator}` "
                f"{metrics['benchmark_value']}."
            ),
        ]


    else:
        status = (
            "complete"
        )

        summary = [
            (
                f"{len(rows)} r\u00e9sultat(s) ont \u00e9t\u00e9 calcul\u00e9s "
                f"avec l'agr\u00e9gation `{aggregation.function}`."
            )
        ]


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
            "aggregation",

        planned_readiness=
            "executable_now",

        execution_status=
            status,

        chart_type=
            "bar",

        summary=
            summary,

        metrics=
            metrics,

        chart_data=
            rows,

        statistical_decision=
            None,

        statistical_result=
            None,

        visualization=
            None,

        warnings=
            [],

        limitations=[
            (
                "This result is a deterministic descriptive "
                +
                (
                    "aggregation and benchmark comparison. "
                    if benchmark is not None
                    else "aggregation. "
                )
                +
                (
                    "It does not imply statistical significance "
                    "or causality."
                )
            )
        ],

        execution_rule_version=(
            AI_AGGREGATION_EXECUTION_RULE_VERSION
        ),
    )


def execute_ranking_contract(
    *,
    contract: AnalyticalContract,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_filename: str,
) -> ExecutedAnalysis:
    if contract.family != "ranking":
        raise ValueError(
            "The canonical ranking executor only accepts "
            "`ranking` contracts."
        )


    ranking = contract.ranking


    if ranking is None:
        raise ValueError(
            "A ranking contract requires RankingSpec."
        )


    (
        rows,
        metrics,
    ) = aggregate_contract_rows(
        contract=contract,
        dataframe=dataframe,
    )


    reverse = (
        ranking.order ==
        "descending"
    )


    ordered_rows = sorted(
        rows,
        key=lambda row: float(
            row[
                "value"
            ]
        ),
        reverse=reverse,
    )


    limited_rows = ordered_rows[
        :ranking.limit
    ]


    chart_rows = [
        {
            **row,
            "rank": index,
        }
        for index, row
        in enumerate(
            limited_rows,
            start=1,
        )
    ]


    metrics = {
        **metrics,
        "ranking_order": ranking.order,
        "ranking_limit": ranking.limit,
        "available_group_count": len(rows),
        "result_count": len(chart_rows),
        "top_category": (
            chart_rows[0]["category"]
            if chart_rows
            else None
        ),
        "top_value": (
            chart_rows[0]["value"]
            if chart_rows
            else None
        ),
    }


    if not chart_rows:
        status = "skipped"
        summary = [
            "Aucun groupe classable n'a pu être calculé."
        ]
    else:
        status = "complete"
        direction = (
            "décroissant"
            if ranking.order == "descending"
            else "croissant"
        )
        summary = [
            (
                f"{len(chart_rows)} résultat(s) ont été conservés "
                f"après classement {direction}."
            ),
            (
                f"Premier résultat : {chart_rows[0]['category']} "
                f"= {chart_rows[0]['value']}."
            ),
        ]


    return ExecutedAnalysis(
        analysis_id=(
            "ai_tool:"
            f"{contract.contract_id}"
        ),
        dataset_id=dataset_id,
        dataset_filename=dataset_filename,
        title=contract.title,
        family="ranking",
        planned_readiness="executable_now",
        execution_status=status,
        chart_type="bar",
        summary=summary,
        metrics=metrics,
        chart_data=chart_rows,
        statistical_decision=None,
        statistical_result=None,
        visualization=None,
        warnings=[],
        limitations=[
            (
                "The ranking is deterministic and reflects only "
                "the validated aggregation, ordering and limit."
            )
        ],
        execution_rule_version=(
            AI_RANKING_EXECUTION_RULE_VERSION
        ),
    )


# ============================================================
# TOOL ARGUMENT TRACE
# ============================================================

def build_tool_arguments(
    contract: AnalyticalContract,
) -> dict[
    str,
    Any,
]:
    aggregation = (
        contract.aggregation
    )


    return {
        "family":
            contract.family,

        "dataset_ids":
            list(
                contract
                .required_dataset_ids
            ),

        "analytical_grain":
            contract
            .analytical_grain,

        "variables":
            {
                binding.role:
                    binding.column

                for binding
                in contract.bindings
            },

        "aggregation":
            (
                aggregation.model_dump(
                    mode="json"
                )
                if (
                    aggregation
                    is not None
                )
                else None
            ),
    }


# ============================================================
# SINGLE CONTRACT EXECUTION
# ============================================================

def execute_validated_contract(
    *,
    contract: AnalyticalContract,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
    call_index: int,
) -> AIToolCallTrace:
    arguments = (
        build_tool_arguments(
            contract
        )
    )


    if (
        contract.status
        !=
        "validated"
    ):
        return AIToolCallTrace(
            call_index=(
                call_index
            ),
            contract_id=(
                contract.contract_id
            ),
            family=(
                contract.family
            ),
            tool_name=None,
            execution_status=(
                "not_executed"
            ),
            arguments=(
                arguments
            ),
            warnings=[
                (
                    "Only contracts promoted to `validated` "
                    "by Python may reach the tool layer."
                ),
            ],
        )


    if (
        contract.joins
        or
        contract.derived_variables
    ):
        return AIToolCallTrace(
            call_index=(
                call_index
            ),
            contract_id=(
                contract.contract_id
            ),
            family=(
                contract.family
            ),
            tool_name=None,
            execution_status=(
                "rejected"
            ),
            arguments=(
                arguments
            ),
            errors=[
                (
                    "AI tool orchestration v0.4 refuses "
                    "contracts containing joins or derived "
                    "variables."
                ),
            ],
        )


    capability = (
        TOOL_CAPABILITIES.get(
            contract.family
        )
    )


    if (
        capability is None
        or
        not capability.enabled
    ):
        return AIToolCallTrace(
            call_index=(
                call_index
            ),
            contract_id=(
                contract.contract_id
            ),
            family=(
                contract.family
            ),
            tool_name=(
                capability.tool_name
                if capability
                is not None
                else None
            ),
            execution_status=(
                "not_supported"
            ),
            arguments=(
                arguments
            ),
            warnings=[
                (
                    capability.reason
                    if capability
                    is not None
                    else (
                        "No deterministic AI tool capability "
                        "is registered for this family."
                    )
                ),
            ],
        )


    dataset_map = (
        build_dataset_map(
            datasets
        )
    )


    (
        record,
        resolution_error,
    ) = resolve_contract_dataset(
        contract=(
            contract
        ),
        dataset_map=(
            dataset_map
        ),
    )


    if (
        record is None
    ):
        return AIToolCallTrace(
            call_index=(
                call_index
            ),
            contract_id=(
                contract.contract_id
            ),
            family=(
                contract.family
            ),
            tool_name=(
                capability.tool_name
            ),
            execution_status=(
                "rejected"
            ),
            arguments=(
                arguments
            ),
            errors=[
                (
                    resolution_error
                    or
                    "Dataset resolution failed."
                )
            ],
        )


    dataframe = (
        record[
            "dataframe"
        ]
    )


    dataset_id = str(
        record.get(
            "dataset_id",
            "",
        )
    )


    dataset_filename = str(
        record.get(
            "filename",
            "",
        )
    )


    try:
        if (
            contract.family
            ==
            "time_series"
        ):
            executed = (
                execute_time_series_contract(
                    contract=(
                        contract
                    ),
                    dataframe=(
                        dataframe
                    ),
                    dataset_id=(
                        dataset_id
                    ),
                    dataset_filename=(
                        dataset_filename
                    ),
                )
            )


        elif (
            contract.family
            ==
            "aggregation"
        ):
            executed = (
                execute_aggregation_contract(
                    contract=(
                        contract
                    ),
                    dataframe=(
                        dataframe
                    ),
                    dataset_id=(
                        dataset_id
                    ),
                    dataset_filename=(
                        dataset_filename
                    ),
                )
            )


        elif (
            contract.family
            ==
            "ranking"
        ):
            executed = (
                execute_ranking_contract(
                    contract=(
                        contract
                    ),
                    dataframe=(
                        dataframe
                    ),
                    dataset_id=(
                        dataset_id
                    ),
                    dataset_filename=(
                        dataset_filename
                    ),
                )
            )


        else:
            candidate = (
                contract_to_analysis_candidate(
                    contract,
                    dataset_filename=(
                        dataset_filename
                    ),
                )
            )


            executed = (
                execute_analysis_candidate(
                    candidate,
                    dataframe,
                )
            )


    except Exception as error:
        return AIToolCallTrace(
            call_index=(
                call_index
            ),
            contract_id=(
                contract.contract_id
            ),
            family=(
                contract.family
            ),
            tool_name=(
                capability.tool_name
            ),
            execution_status=(
                "rejected"
            ),
            dataset_id=(
                dataset_id
            ),
            dataset_filename=(
                dataset_filename
            ),
            arguments=(
                arguments
            ),
            errors=[
                (
                    "The deterministic tool invocation "
                    "failed before returning a trusted "
                    f"analysis result: {error}"
                ),
            ],
        )


    return AIToolCallTrace(
        call_index=(
            call_index
        ),
        contract_id=(
            contract.contract_id
        ),
        family=(
            contract.family
        ),
        tool_name=(
            capability.tool_name
        ),
        execution_status=(
            "executed"
        ),
        dataset_id=(
            dataset_id
        ),
        dataset_filename=(
            dataset_filename
        ),
        arguments=(
            arguments
        ),
        result=(
            executed
        ),
        errors=[],
        warnings=[
            (
                "The LLM did not execute Python. "
                "It produced the analytical intent; "
                "Python selected and ran a whitelisted "
                "deterministic tool."
            ),
        ],
    )


# ============================================================
# PLANNER REPORT → TOOL ORCHESTRATION
# ============================================================

def execute_ai_planner_report(
    *,
    planner_report: AIPlannerReport,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> AIToolOrchestrationReport:
    validated_contracts = [
        item.contract

        for item
        in planner_report.items

        if (
            item.validation_status ==
            "validated"
            and
            item.contract is not None
        )
    ]


    tool_calls = [
        execute_validated_contract(
            contract=(
                contract
            ),
            datasets=(
                datasets
            ),
            call_index=(
                index
            ),
        )

        for (
            index,
            contract,
        ) in enumerate(
            validated_contracts,
            start=1,
        )
    ]


    return (
        AIToolOrchestrationReport(
            planner=(
                planner_report
            ),
            validated_contract_count=(
                len(
                    validated_contracts
                )
            ),
            tool_call_count=(
                len(
                    tool_calls
                )
            ),
            executed_count=sum(
                1

                for call
                in tool_calls

                if (
                    call.execution_status
                    ==
                    "executed"
                )
            ),
            not_executed_count=sum(
                1

                for call
                in tool_calls

                if (
                    call.execution_status
                    ==
                    "not_executed"
                )
            ),
            not_supported_count=sum(
                1

                for call
                in tool_calls

                if (
                    call.execution_status
                    ==
                    "not_supported"
                )
            ),
            rejected_count=sum(
                1

                for call
                in tool_calls

                if (
                    call.execution_status
                    ==
                    "rejected"
                )
            ),
            tool_calls=(
                tool_calls
            ),
            orchestrator_notes=[
                (
                    "The AI orchestration layer accepts "
                    "only Python-validated analytical "
                    "contracts."
                ),
                (
                    "Tool selection is deterministic at the "
                    "contract-to-executor boundary; arbitrary "
                    "Python generated by an LLM is never run."
                ),
                (
                    "time_series aggregation is executed "
                    "directly from the canonical AggregationSpec. "
                    "Supported reducers are validated before "
                    "deterministic execution; median additionally "
                    "retains the Q1/Q3 descriptive band."
                ),
            ],
        )
    )
