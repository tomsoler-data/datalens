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
    "ai_tool_orchestrator_v0.2"
)


AI_TIME_SERIES_EXECUTION_RULE_VERSION = (
    "ai_tool_time_series_median_v0.1"
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
            chart_type="line_band",
            statistical_strategy=(
                "deterministic_median_iqr_by_period"
            ),
            reason=(
                "The AI-native time-series boundary executes "
                "the validated median aggregation by period "
                "and calculates an IQR band deterministically."
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
            enabled=False,
            reason=(
                "Generic aggregation will be added as a "
                "dedicated deterministic tool instead of "
                "reusing business-specific requested code."
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
            enabled=False,
            reason=(
                "Generic ranking is not yet exposed as a "
                "canonical deterministic AI tool."
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
                "AI tool execution v0.2 requires exactly "
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
# compatibility path anymore because AnalysisCandidate does
# not carry the canonical AggregationSpec. v0.2 therefore
# executes the validated median-by-period contract directly.
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
                    "AI tool orchestration v0.2 supports only "
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

def execute_median_time_series_contract(
    *,
    contract: AnalyticalContract,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_filename: str,
) -> ExecutedAnalysis:
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
        !=
        "median"
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
                "AI time-series v0.2 currently requires "
                "aggregation=function:median, "
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


    working = pd.DataFrame(
        {
            "__time":
                dataframe[
                    time_column
                ],

            "__value":
                pd.to_numeric(
                    dataframe[
                        value_column
                    ],
                    errors="coerce",
                ),
        }
    ).dropna()


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
                    "line_band"
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
                        "median",

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
                        "all time/value pairs were missing "
                        "or non-numeric."
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


        grouped_rows.append(
            {
                "period":
                    native_scalar(
                        period
                    ),

                "median":
                    float(
                        values.median()
                    ),

                "q1":
                    float(
                        values.quantile(
                            0.25
                        )
                    ),

                "q3":
                    float(
                        values.quantile(
                            0.75
                        )
                    ),

                "count":
                    int(
                        values.count()
                    ),
            }
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
        period_count <
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


    medians = [
        float(
            row[
                "median"
            ]
        )

        for row
        in grouped_rows
    ]


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
                "line_band"
            ),
            summary=[
                (
                    f"{period_count} période(s) distincte(s) "
                    f"ont été agrégées par médiane pour "
                    f"{value_column}."
                ),
                (
                    "Les quartiles Q1 et Q3 sont calculés "
                    "pour chaque période afin de représenter "
                    "la dispersion autour de la médiane."
                ),
            ],
            metrics={
                "time_column":
                    time_column,

                "value_column":
                    value_column,

                "aggregation_function":
                    "median",

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

                "period_median_min":
                    (
                        min(
                            medians
                        )
                        if medians
                        else None
                    ),

                "period_median_max":
                    (
                        max(
                            medians
                        )
                        if medians
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
            limitations=[
                (
                    "This AI-native time-series execution "
                    "is descriptive. It does not fit a "
                    "forecasting, causal or inferential "
                    "time-series model."
                ),
                (
                    "The current native scope executes the "
                    "validated median aggregation and adds "
                    "Q1/Q3 as deterministic descriptive "
                    "dispersion summaries."
                ),
            ],
            execution_rule_version=(
                AI_TIME_SERIES_EXECUTION_RULE_VERSION
            ),
        )
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
                    "AI tool orchestration v0.2 refuses "
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
                execute_median_time_series_contract(
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
                    "time_series median aggregation is now "
                    "executed directly from the canonical "
                    "AggregationSpec instead of being lost "
                    "through the legacy AnalysisCandidate "
                    "compatibility adapter."
                ),
            ],
        )
    )
