from __future__ import annotations


from time import (
    perf_counter,
)

from typing import (
    Any,
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from app.ai.provider import (
    client,
)

from app.ai.tool_orchestrator import (
    AIToolCallTrace,
    execute_validated_contract,
)

from app.planning.analytical_contract import (
    AnalyticalContract,
)


# ============================================================
# VERSION
# ============================================================

NATIVE_TOOL_CALLING_RULE_VERSION = (
    "native_tool_calling_v0.9"
)


DEFAULT_NATIVE_TOOL_MODEL = (
    "qwen2.5:1.5b-instruct"
)


MAX_NATIVE_TOOL_ATTEMPTS = 2


# ============================================================
# GENERIC TWO-VARIABLE TOOL ARGUMENTS
# ============================================================

class TwoVariableToolArgs(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    dataset_id: str = Field(
        min_length=1
    )

    x_column: str = Field(
        min_length=1
    )

    y_column: str = Field(
        min_length=1
    )


class GroupComparisonToolArgs(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    dataset_id: str = Field(
        min_length=1
    )

    group_column: str = Field(
        min_length=1
    )

    value_column: str = Field(
        min_length=1
    )


class DistributionToolArgs(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    dataset_id: str = Field(
        min_length=1
    )

    value_column: str = Field(
        min_length=1
    )


class TimeSeriesToolArgs(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    dataset_id: str = Field(
        min_length=1
    )

    time_column: str = Field(
        min_length=1
    )

    value_column: str = Field(
        min_length=1
    )

    # The default keeps deterministic compatibility with
    # historical median-only validation fixtures. The native
    # JSON schema below still requires the field from the tool
    # model, so new live calls must copy the canonical
    # AggregationSpec explicitly.
    aggregation_function: Literal[
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "count",
        "distinct_count",
    ] = "median"


class AggregationToolArgs(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    dataset_id: str = Field(
        min_length=1
    )

    aggregation_function: Literal[
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "count",
        "distinct_count",
    ]

    source_column: (
        str
        | None
    ) = None

    group_by_columns: list[
        str
    ] = Field(
        default_factory=list
    )


class RankingToolArgs(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    dataset_id: str = Field(
        min_length=1
    )

    dimension_column: str = Field(
        min_length=1
    )

    aggregation_function: Literal[
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "count",
        "distinct_count",
    ]

    source_column: (
        str
        | None
    ) = None

    order: Literal[
        "ascending",
        "descending",
    ]

    limit: int = Field(
        ge=1,
        le=100,
    )


# Backward-compatible alias for previous imports/tests.
QuantitativeAssociationToolArgs = (
    TwoVariableToolArgs
)


# ============================================================
# NATIVE TOOL REGISTRY
# ============================================================

class NativeToolSpec(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    family: str

    tool_name: str

    argument_shape: Literal[
        "x_y",
        "group_value",
        "value",
        "time_value",
        "aggregation",
        "ranking",
    ]

    description: str


NATIVE_TOOL_SPECS: dict[
    str,
    NativeToolSpec,
] = {
    "quantitative_association":
        NativeToolSpec(
            family=(
                "quantitative_association"
            ),
            tool_name=(
                "run_quantitative_association"
            ),
            argument_shape=(
                "x_y"
            ),
            description=(
                "Execute a validated quantitative "
                "association analysis between exactly "
                "two quantitative columns from one "
                "DataLens dataset."
            ),
        ),

    "categorical_association":
        NativeToolSpec(
            family=(
                "categorical_association"
            ),
            tool_name=(
                "run_categorical_association"
            ),
            argument_shape=(
                "x_y"
            ),
            description=(
                "Execute a validated categorical "
                "association analysis between exactly "
                "two categorical or boolean columns "
                "from one DataLens dataset."
            ),
        ),

    "group_comparison":
        NativeToolSpec(
            family=(
                "group_comparison"
            ),
            tool_name=(
                "run_group_comparison"
            ),
            argument_shape=(
                "group_value"
            ),
            description=(
                "Execute a validated group comparison "
                "between one categorical or boolean "
                "grouping column and one quantitative "
                "value column from one DataLens dataset."
            ),
        ),

    "distribution":
        NativeToolSpec(
            family=(
                "distribution"
            ),
            tool_name=(
                "run_distribution"
            ),
            argument_shape=(
                "value"
            ),
            description=(
                "Execute a validated descriptive "
                "distribution analysis for exactly one "
                "quantitative value column from one "
                "DataLens dataset."
            ),
        ),

    "time_series":
        NativeToolSpec(
            family=(
                "time_series"
            ),
            tool_name=(
                "run_time_series"
            ),
            argument_shape=(
                "time_value"
            ),
            description=(
                "Execute a validated deterministic temporal "
                "aggregation for one temporal column and one "
                "value column from one DataLens dataset. "
                "The aggregation function is copied exactly "
                "from the canonical DataLens contract. Median "
                "retains the deterministic Q1/Q3 profile."
            ),
        ),

    "aggregation":
        NativeToolSpec(
            family=(
                "aggregation"
            ),
            tool_name=(
                "run_aggregation"
            ),
            argument_shape=(
                "aggregation"
            ),
            description=(
                "Execute a validated deterministic aggregation "
                "using the exact aggregation function, source "
                "column and grouping columns from the canonical "
                "DataLens contract."
            ),
        ),

    "ranking":
        NativeToolSpec(
            family=(
                "ranking"
            ),
            tool_name=(
                "run_ranking"
            ),
            argument_shape=(
                "ranking"
            ),
            description=(
                "Execute a validated deterministic ranking after "
                "the exact canonical aggregation, sort direction "
                "and top-K limit from the DataLens contract."
            ),
        ),
}


SUPPORTED_NATIVE_FAMILIES = frozenset(
    NATIVE_TOOL_SPECS.keys()
)


SUPPORTED_NATIVE_TOOLS = frozenset(
    spec.tool_name
    for spec
    in NATIVE_TOOL_SPECS.values()
)


# Backward-compatible constants used by older code/tests.
SUPPORTED_NATIVE_FAMILY = (
    "quantitative_association"
)

SUPPORTED_NATIVE_TOOL = (
    "run_quantitative_association"
)


# ============================================================
# NATIVE TOOL REQUEST TRACE
# ============================================================

NativeToolValidationStatus = Literal[
    "validated",
    "rejected",
]


class NativeToolCallProposal(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    tool_name: str

    arguments: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )


class NativeToolCallAttempt(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    attempt_index: int = Field(
        ge=1
    )

    prompt_variant: Literal[
        "standard",
        "mandatory_retry",
    ]

    tool_call_count: int = Field(
        ge=0
    )

    assistant_content: str = ""

    selected_tool_name: (
        str
        | None
    ) = None

    errors: list[
        str
    ] = Field(
        default_factory=list
    )

    prompt_construction_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    model_inference_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    response_parse_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    total_ms: float = Field(
        default=0.0,
        ge=0.0,
    )


class NativeToolCallRequestResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    proposal: (
        NativeToolCallProposal
        | None
    ) = None

    attempts: list[
        NativeToolCallAttempt
    ] = Field(
        default_factory=list
    )


class NativeToolTiming(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    prompt_construction_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    model_inference_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    response_parse_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    python_validation_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    deterministic_execution_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    total_ms: float = Field(
        default=0.0,
        ge=0.0,
    )


class NativeToolCallingReport(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    status: Literal[
        "ready"
    ] = "ready"

    model: str

    contract_id: str

    contract_family: str

    available_tools: list[
        str
    ] = Field(
        default_factory=list
    )

    expected_tool: (
        str
        | None
    ) = None

    tool_call_received: bool

    requested_tool: (
        str
        | None
    ) = None

    requested_arguments: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    validation_status: (
        NativeToolValidationStatus
    )

    validation_errors: list[
        str
    ] = Field(
        default_factory=list
    )

    attempt_count: int = Field(
        ge=0
    )

    retry_count: int = Field(
        ge=0
    )

    attempts: list[
        NativeToolCallAttempt
    ] = Field(
        default_factory=list
    )

    execution: (
        AIToolCallTrace
        | None
    ) = None

    timing: NativeToolTiming = Field(
        default_factory=
            NativeToolTiming
    )

    notes: list[
        str
    ] = Field(
        default_factory=list
    )

    native_tool_rule_version: str = (
        NATIVE_TOOL_CALLING_RULE_VERSION
    )


# ============================================================
# TOOL JSON SCHEMAS
# ============================================================

def build_two_variable_tool_schema(
    spec: NativeToolSpec,
) -> dict[
    str,
    Any,
]:
    return {
        "type":
            "function",

        "function":
            {
                "name":
                    spec.tool_name,

                "description":
                    (
                        spec.description
                        +
                        " Only use the exact dataset and "
                        "column bindings from the validated "
                        "analytical contract."
                    ),

                "parameters":
                    {
                        "type":
                            "object",

                        "required":
                            [
                                "dataset_id",
                                "x_column",
                                "y_column",
                            ],

                        "properties":
                            {
                                "dataset_id":
                                    {
                                        "type":
                                            "string",

                                        "description":
                                            (
                                                "Exact dataset_id "
                                                "from the validated "
                                                "DataLens contract."
                                            ),
                                    },

                                "x_column":
                                    {
                                        "type":
                                            "string",

                                        "description":
                                            (
                                                "Exact x column "
                                                "from the validated "
                                                "DataLens contract."
                                            ),
                                    },

                                "y_column":
                                    {
                                        "type":
                                            "string",

                                        "description":
                                            (
                                                "Exact y column "
                                                "from the validated "
                                                "DataLens contract."
                                            ),
                                    },
                            },

                        "additionalProperties":
                            False,
                    },
            },
    }


def build_group_comparison_tool_schema(
    spec: NativeToolSpec,
) -> dict[
    str,
    Any,
]:
    return {
        "type":
            "function",

        "function":
            {
                "name":
                    spec.tool_name,

                "description":
                    (
                        spec.description
                        +
                        " Only use the exact dataset and "
                        "column bindings from the validated "
                        "analytical contract."
                    ),

                "parameters":
                    {
                        "type":
                            "object",

                        "required":
                            [
                                "dataset_id",
                                "group_column",
                                "value_column",
                            ],

                        "properties":
                            {
                                "dataset_id":
                                    {
                                        "type":
                                            "string",

                                        "description":
                                            (
                                                "Exact dataset_id "
                                                "from the validated "
                                                "DataLens contract."
                                            ),
                                    },

                                "group_column":
                                    {
                                        "type":
                                            "string",

                                        "description":
                                            (
                                                "Exact categorical "
                                                "or boolean grouping "
                                                "column from the "
                                                "validated contract."
                                            ),
                                    },

                                "value_column":
                                    {
                                        "type":
                                            "string",

                                        "description":
                                            (
                                                "Exact quantitative "
                                                "value column from "
                                                "the validated "
                                                "contract."
                                            ),
                                    },
                            },

                        "additionalProperties":
                            False,
                    },
            },
    }


def build_distribution_tool_schema(
    spec: NativeToolSpec,
) -> dict[
    str,
    Any,
]:
    return {
        "type":
            "function",

        "function":
            {
                "name":
                    spec.tool_name,

                "description":
                    (
                        spec.description
                        +
                        " Only use the exact dataset and "
                        "value column from the validated "
                        "analytical contract."
                    ),

                "parameters":
                    {
                        "type":
                            "object",

                        "required":
                            [
                                "dataset_id",
                                "value_column",
                            ],

                        "properties":
                            {
                                "dataset_id":
                                    {
                                        "type":
                                            "string",

                                        "description":
                                            (
                                                "Exact dataset_id "
                                                "from the validated "
                                                "DataLens contract."
                                            ),
                                    },

                                "value_column":
                                    {
                                        "type":
                                            "string",

                                        "description":
                                            (
                                                "Exact quantitative "
                                                "value column from "
                                                "the validated "
                                                "distribution contract."
                                            ),
                                    },
                            },

                        "additionalProperties":
                            False,
                    },
            },
    }


def build_time_series_tool_schema(
    spec: NativeToolSpec,
) -> dict[
    str,
    Any,
]:
    return {
        "type":
            "function",

        "function":
            {
                "name":
                    spec.tool_name,

                "description":
                    (
                        spec.description
                        +
                        " Only use the exact dataset, "
                        "time column, value column and "
                        "aggregation function from the "
                        "validated analytical contract."
                    ),

                "parameters":
                    {
                        "type":
                            "object",

                        "required":
                            [
                                "dataset_id",
                                "time_column",
                                "value_column",
                                "aggregation_function",
                            ],

                        "properties":
                            {
                                "dataset_id":
                                    {
                                        "type":
                                            "string",

                                        "description":
                                            (
                                                "Exact dataset_id "
                                                "from the validated "
                                                "DataLens contract."
                                            ),
                                    },

                                "time_column":
                                    {
                                        "type":
                                            "string",

                                        "description":
                                            (
                                                "Exact temporal "
                                                "column from the "
                                                "validated contract."
                                            ),
                                    },

                                "value_column":
                                    {
                                        "type":
                                            "string",

                                        "description":
                                            (
                                                "Exact value column "
                                                "from the validated "
                                                "contract."
                                            ),
                                    },

                                "aggregation_function":
                                    {
                                        "type":
                                            "string",

                                        "enum":
                                            [
                                                "sum",
                                                "mean",
                                                "median",
                                                "min",
                                                "max",
                                                "count",
                                                "distinct_count",
                                            ],

                                        "description":
                                            (
                                                "Exact aggregation "
                                                "function from the "
                                                "validated canonical "
                                                "AggregationSpec."
                                            ),
                                    },
                            },

                        "additionalProperties":
                            False,
                    },
            },
    }

def build_aggregation_tool_schema(
    spec: NativeToolSpec,
) -> dict[
    str,
    Any,
]:
    return {
        "type": "function",
        "function": {
            "name": spec.tool_name,
            "description": (
                spec.description
                +
                " Copy the validated arguments exactly. "
                "source_column may be null only when the "
                "validated aggregation has no source role."
            ),
            "parameters": {
                "type": "object",
                "required": [
                    "dataset_id",
                    "aggregation_function",
                    "source_column",
                    "group_by_columns",
                ],
                "properties": {
                    "dataset_id": {
                        "type": "string",
                    },
                    "aggregation_function": {
                        "type": "string",
                        "enum": [
                            "sum",
                            "mean",
                            "median",
                            "min",
                            "max",
                            "count",
                            "distinct_count",
                        ],
                    },
                    "source_column": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ],
                    },
                    "group_by_columns": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
    }


def build_ranking_tool_schema(
    spec: NativeToolSpec,
) -> dict[
    str,
    Any,
]:
    return {
        "type": "function",
        "function": {
            "name": spec.tool_name,
            "description": (
                spec.description
                +
                " Copy every validated argument exactly."
            ),
            "parameters": {
                "type": "object",
                "required": [
                    "dataset_id",
                    "dimension_column",
                    "aggregation_function",
                    "source_column",
                    "order",
                    "limit",
                ],
                "properties": {
                    "dataset_id": {
                        "type": "string",
                    },
                    "dimension_column": {
                        "type": "string",
                    },
                    "aggregation_function": {
                        "type": "string",
                        "enum": [
                            "sum",
                            "mean",
                            "median",
                            "min",
                            "max",
                            "count",
                            "distinct_count",
                        ],
                    },
                    "source_column": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ],
                    },
                    "order": {
                        "type": "string",
                        "enum": [
                            "ascending",
                            "descending",
                        ],
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "additionalProperties": False,
            },
        },
    }


def build_native_tool_schema(
    spec: NativeToolSpec,
) -> dict[
    str,
    Any,
]:
    if (
        spec.argument_shape ==
        "x_y"
    ):
        return (
            build_two_variable_tool_schema(
                spec
            )
        )


    if (
        spec.argument_shape ==
        "group_value"
    ):
        return (
            build_group_comparison_tool_schema(
                spec
            )
        )


    if (
        spec.argument_shape ==
        "value"
    ):
        return (
            build_distribution_tool_schema(
                spec
            )
        )


    if (
        spec.argument_shape ==
        "time_value"
    ):
        return (
            build_time_series_tool_schema(
                spec
            )
        )


    if (
        spec.argument_shape ==
        "aggregation"
    ):
        return (
            build_aggregation_tool_schema(
                spec
            )
        )


    if (
        spec.argument_shape ==
        "ranking"
    ):
        return (
            build_ranking_tool_schema(
                spec
            )
        )


    raise ValueError(
        (
            "Unsupported native tool argument shape: "
            f"{spec.argument_shape}."
        )
    )


NATIVE_TOOL_SCHEMAS: list[
    dict[
        str,
        Any,
    ]
] = [
    build_native_tool_schema(
        spec
    )
    for spec
    in NATIVE_TOOL_SPECS.values()
]


# Backward-compatible constant.
QUANTITATIVE_ASSOCIATION_TOOL_SCHEMA = (
    build_native_tool_schema(
        NATIVE_TOOL_SPECS[
            "quantitative_association"
        ]
    )
)


# ============================================================
# CONTRACT HELPERS
# ============================================================

def native_tool_spec_for_contract(
    contract: AnalyticalContract,
) -> NativeToolSpec:
    spec = (
        NATIVE_TOOL_SPECS.get(
            contract.family
        )
    )


    if (
        spec is None
    ):
        raise ValueError(
            (
                "Native tool calling v0.9 does not "
                "support analytical family "
                f"`{contract.family}`."
            )
        )


    return spec


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


def expected_two_variable_tool_args(
    contract: AnalyticalContract,
) -> TwoVariableToolArgs:
    if (
        contract.status
        !=
        "validated"
    ):
        raise ValueError(
            (
                "Native tool calling requires a contract "
                "already promoted to `validated` by Python."
            )
        )


    native_tool_spec_for_contract(
        contract
    )


    if (
        len(
            contract
            .required_dataset_ids
        )
        !=
        1
    ):
        raise ValueError(
            (
                "Native tool calling v0.9 requires "
                "exactly one dataset."
            )
        )


    bindings = (
        contract_binding_map(
            contract
        )
    )


    x_column = (
        bindings.get(
            "x"
        )
    )

    y_column = (
        bindings.get(
            "y"
        )
    )


    if (
        x_column is None
        or
        y_column is None
    ):
        raise ValueError(
            (
                "The validated two-variable contract "
                "must contain x and y bindings."
            )
        )


    return (
        TwoVariableToolArgs(
            dataset_id=(
                contract
                .required_dataset_ids[
                    0
                ]
            ),
            x_column=(
                x_column
            ),
            y_column=(
                y_column
            ),
        )
    )


def expected_group_comparison_tool_args(
    contract: AnalyticalContract,
) -> GroupComparisonToolArgs:
    if (
        contract.status
        !=
        "validated"
    ):
        raise ValueError(
            (
                "Native tool calling requires a contract "
                "already promoted to `validated` by Python."
            )
        )


    if (
        contract.family
        !=
        "group_comparison"
    ):
        raise ValueError(
            (
                "Expected a group_comparison contract."
            )
        )


    native_tool_spec_for_contract(
        contract
    )


    if (
        len(
            contract
            .required_dataset_ids
        )
        !=
        1
    ):
        raise ValueError(
            (
                "Native tool calling v0.9 requires "
                "exactly one dataset."
            )
        )


    bindings = (
        contract_binding_map(
            contract
        )
    )


    group_column = (
        bindings.get(
            "group"
        )
    )

    value_column = (
        bindings.get(
            "value"
        )
    )


    if (
        group_column is None
        or
        value_column is None
    ):
        raise ValueError(
            (
                "The validated group-comparison contract "
                "must contain group and value bindings."
            )
        )


    return (
        GroupComparisonToolArgs(
            dataset_id=(
                contract
                .required_dataset_ids[
                    0
                ]
            ),
            group_column=(
                group_column
            ),
            value_column=(
                value_column
            ),
        )
    )


def expected_distribution_tool_args(
    contract: AnalyticalContract,
) -> DistributionToolArgs:
    if (
        contract.status
        !=
        "validated"
    ):
        raise ValueError(
            (
                "Native tool calling requires a contract "
                "already promoted to `validated` by Python."
            )
        )


    if (
        contract.family
        !=
        "distribution"
    ):
        raise ValueError(
            (
                "Expected a distribution contract."
            )
        )


    native_tool_spec_for_contract(
        contract
    )


    if (
        len(
            contract
            .required_dataset_ids
        )
        !=
        1
    ):
        raise ValueError(
            (
                "Native tool calling v0.9 requires "
                "exactly one dataset."
            )
        )


    bindings = (
        contract_binding_map(
            contract
        )
    )


    value_column = (
        bindings.get(
            "value"
        )
    )


    if (
        value_column is None
    ):
        raise ValueError(
            (
                "The validated distribution contract "
                "must contain a value binding."
            )
        )


    return (
        DistributionToolArgs(
            dataset_id=(
                contract
                .required_dataset_ids[
                    0
                ]
            ),
            value_column=(
                value_column
            ),
        )
    )


def expected_time_series_tool_args(
    contract: AnalyticalContract,
) -> TimeSeriesToolArgs:
    if (
        contract.status
        !=
        "validated"
    ):
        raise ValueError(
            (
                "Native tool calling requires a contract "
                "already promoted to `validated` by Python."
            )
        )


    if (
        contract.family
        !=
        "time_series"
    ):
        raise ValueError(
            (
                "Expected a time_series contract."
            )
        )


    native_tool_spec_for_contract(
        contract
    )


    if (
        len(
            contract
            .required_dataset_ids
        )
        !=
        1
    ):
        raise ValueError(
            (
                "Native tool calling v0.9 requires "
                "exactly one dataset."
            )
        )


    aggregation = (
        contract.aggregation
    )


    supported_functions = {
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "count",
        "distinct_count",
    }


    if (
        aggregation is None
        or
        aggregation.function
        not in supported_functions
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
                "Native time_series v0.9 requires a "
                "validated canonical AggregationSpec with "
                "source_role=`value`, "
                "group_by_roles=[`time`] and one supported "
                "aggregation function."
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
                "The validated time-series contract must "
                "contain time and value bindings."
            )
        )


    return (
        TimeSeriesToolArgs(
            dataset_id=(
                contract
                .required_dataset_ids[
                    0
                ]
            ),
            time_column=(
                time_column
            ),
            value_column=(
                value_column
            ),
            aggregation_function=(
                aggregation.function
            ),
        )
    )

def expected_aggregation_tool_args(
    contract: AnalyticalContract,
) -> AggregationToolArgs:
    if (
        contract.status !=
        "validated"
    ):
        raise ValueError(
            "Native tool calling requires a contract already "
            "promoted to `validated` by Python."
        )


    if contract.family != "aggregation":
        raise ValueError(
            "Expected an aggregation contract."
        )


    native_tool_spec_for_contract(
        contract
    )


    if len(contract.required_dataset_ids) != 1:
        raise ValueError(
            "Native aggregation requires exactly one dataset."
        )


    aggregation = contract.aggregation


    if aggregation is None:
        raise ValueError(
            "A validated aggregation contract requires AggregationSpec."
        )


    bindings = contract_binding_map(
        contract
    )


    source_column = (
        bindings.get(
            aggregation.source_role
        )
        if aggregation.source_role is not None
        else None
    )


    if (
        aggregation.source_role is not None
        and
        source_column is None
    ):
        raise ValueError(
            "The validated aggregation source role has no column binding."
        )


    group_by_columns: list[str] = []


    for role in aggregation.group_by_roles:
        column = bindings.get(
            role
        )


        if column is None:
            raise ValueError(
                "The validated aggregation group role has no "
                f"column binding: {role}."
            )


        group_by_columns.append(
            column
        )


    return AggregationToolArgs(
        dataset_id=contract.required_dataset_ids[0],
        aggregation_function=aggregation.function,
        source_column=source_column,
        group_by_columns=group_by_columns,
    )


def expected_ranking_tool_args(
    contract: AnalyticalContract,
) -> RankingToolArgs:
    if (
        contract.status !=
        "validated"
    ):
        raise ValueError(
            "Native tool calling requires a contract already "
            "promoted to `validated` by Python."
        )


    if contract.family != "ranking":
        raise ValueError(
            "Expected a ranking contract."
        )


    native_tool_spec_for_contract(
        contract
    )


    if len(contract.required_dataset_ids) != 1:
        raise ValueError(
            "Native ranking requires exactly one dataset."
        )


    aggregation = contract.aggregation
    ranking = contract.ranking


    if aggregation is None or ranking is None:
        raise ValueError(
            "A validated ranking contract requires both "
            "AggregationSpec and RankingSpec."
        )


    bindings = contract_binding_map(
        contract
    )


    dimension_column = (
        bindings.get(
            "dimension"
        )
        or
        bindings.get(
            "group"
        )
    )


    if dimension_column is None:
        raise ValueError(
            "The validated ranking contract requires a "
            "dimension or group binding."
        )


    source_column = (
        bindings.get(
            aggregation.source_role
        )
        if aggregation.source_role is not None
        else None
    )


    if (
        aggregation.source_role is not None
        and
        source_column is None
    ):
        raise ValueError(
            "The validated ranking aggregation source role "
            "has no column binding."
        )


    return RankingToolArgs(
        dataset_id=contract.required_dataset_ids[0],
        dimension_column=dimension_column,
        aggregation_function=aggregation.function,
        source_column=source_column,
        order=ranking.order,
        limit=ranking.limit,
    )


def expected_tool_arguments(
    contract: AnalyticalContract,
) -> (
    TwoVariableToolArgs
    | GroupComparisonToolArgs
    | DistributionToolArgs
    | TimeSeriesToolArgs
    | AggregationToolArgs
    | RankingToolArgs
):
    spec = (
        native_tool_spec_for_contract(
            contract
        )
    )


    if (
        spec.argument_shape ==
        "x_y"
    ):
        return (
            expected_two_variable_tool_args(
                contract
            )
        )


    if (
        spec.argument_shape ==
        "group_value"
    ):
        return (
            expected_group_comparison_tool_args(
                contract
            )
        )


    if (
        spec.argument_shape ==
        "value"
    ):
        return (
            expected_distribution_tool_args(
                contract
            )
        )


    if (
        spec.argument_shape ==
        "time_value"
    ):
        return (
            expected_time_series_tool_args(
                contract
            )
        )


    if (
        spec.argument_shape ==
        "aggregation"
    ):
        return (
            expected_aggregation_tool_args(
                contract
            )
        )


    if (
        spec.argument_shape ==
        "ranking"
    ):
        return (
            expected_ranking_tool_args(
                contract
            )
        )


    raise ValueError(
        (
            "Unsupported native tool argument shape: "
            f"{spec.argument_shape}."
        )
    )


# Backward-compatible helper.
def expected_quantitative_tool_args(
    contract: AnalyticalContract,
) -> QuantitativeAssociationToolArgs:
    if (
        contract.family
        !=
        "quantitative_association"
    ):
        raise ValueError(
            (
                "Expected a quantitative_association "
                "contract."
            )
        )


    return (
        expected_two_variable_tool_args(
            contract
        )
    )


# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """
You are the native function-calling router inside DataLens.

You do not calculate statistics.
You do not rewrite analytical plans.
You do not invent columns or datasets.
You do not answer the analytical question yourself.

Python has already validated the analytical contract.

Your job is to select exactly ONE matching function from the
available DataLens tools and copy the validated arguments.

Rules:

1. Select the tool whose analytical family EXACTLY matches the
   validated contract family.
2. Copy every argument from the validated contract EXACTLY.
3. Do not swap, rename, infer or transform arguments.
4. Do not return prose instead of the function call.
5. Do not calculate or summarize any statistical result.
6. Python independently validates the selected tool and every
   argument before executing anything.
""".strip()


def serialized_expected_arguments(
    contract: AnalyticalContract,
) -> list[
    tuple[
        str,
        str,
    ]
]:
    expected = (
        expected_tool_arguments(
            contract
        )
    )


    return [
        (
            key,
            str(
                value
            ),
        )
        for (
            key,
            value,
        )
        in expected
        .model_dump()
        .items()
    ]


def build_native_tool_prompt(
    contract: AnalyticalContract,
) -> str:
    spec = (
        native_tool_spec_for_contract(
            contract
        )
    )


    available = ", ".join(
        sorted(
            SUPPORTED_NATIVE_TOOLS
        )
    )


    argument_lines = "\n".join(
        f"{key}: {value}"
        for (
            key,
            value,
        )
        in serialized_expected_arguments(
            contract
        )
    )


    return (
        "VALIDATED DATALENS CONTRACT\n"
        "===========================\n"
        f"contract_id: {contract.contract_id}\n"
        f"family: {contract.family}\n"
        f"expected_tool: {spec.tool_name}\n"
        f"{argument_lines}\n\n"
        f"AVAILABLE TOOLS: {available}\n\n"
        "Select the tool that exactly matches the "
        "validated family and invoke it now using "
        "the exact argument names required by that tool."
    )


def build_mandatory_retry_prompt(
    contract: AnalyticalContract,
) -> str:
    spec = (
        native_tool_spec_for_contract(
            contract
        )
    )


    argument_lines = ",\n".join(
        (
            f'  {key}="{value}"'
        )
        for (
            key,
            value,
        )
        in serialized_expected_arguments(
            contract
        )
    )


    return (
        "MANDATORY FUNCTION CALL RETRY\n"
        "=============================\n"
        "Your previous response did not contain exactly one "
        "native tool call.\n"
        "Do not answer with text.\n"
        "The validated family requires exactly this function:\n\n"
        f"{spec.tool_name}(\n"
        f"{argument_lines}\n"
        ")\n\n"
        "Use Ollama native tool calling, not plain text."
    )


# ============================================================
# OLLAMA REQUEST HELPERS
# ============================================================

def response_content(
    response: Any,
) -> str:
    content = getattr(
        response.message,
        "content",
        "",
    )


    if (
        content is None
    ):
        return ""


    return str(
        content
    )


def make_native_tool_request(
    *,
    model: str,
    messages: list[
        dict[
            str,
            str,
        ]
    ],
    seed: int,
) -> Any:
    try:
        return (
            client.chat(
                model=(
                    model
                ),
                messages=(
                    messages
                ),
                tools=(
                    NATIVE_TOOL_SCHEMAS
                ),
                stream=False,
                options={
                    "temperature":
                        0,

                    "seed":
                        seed,

                    "num_ctx":
                        2048,
                },
            )
        )


    except Exception as error:
        raise RuntimeError(
            (
                "Ollama native tool calling failed."
            )
        ) from error


# ============================================================
# NATIVE OLLAMA TOOL CALL WITH ONE GUARDED RETRY
# ============================================================

def request_native_tool_call(
    *,
    contract: AnalyticalContract,
    model: str = (
        DEFAULT_NATIVE_TOOL_MODEL
    ),
) -> NativeToolCallRequestResult:
    initial_prompt_started_at = (
        perf_counter()
    )


    initial_prompt = (
        build_native_tool_prompt(
            contract
        )
    )


    initial_prompt_ms = (
        (
            perf_counter()
            -
            initial_prompt_started_at
        )
        *
        1000.0
    )


    attempts: list[
        NativeToolCallAttempt
    ] = []


    messages: list[
        dict[
            str,
            str,
        ]
    ] = [
        {
            "role":
                "system",

            "content":
                SYSTEM_PROMPT,
        },

        {
            "role":
                "user",

            "content":
                initial_prompt,
        },
    ]


    for attempt_index in range(
        1,
        MAX_NATIVE_TOOL_ATTEMPTS
        +
        1,
    ):
        attempt_started_at = (
            perf_counter()
        )


        prompt_variant: Literal[
            "standard",
            "mandatory_retry",
        ] = (
            "standard"
            if attempt_index ==
            1
            else
            "mandatory_retry"
        )


        if (
            attempt_index ==
            1
        ):
            prompt_construction_ms = (
                initial_prompt_ms
            )


        else:
            prompt_started_at = (
                perf_counter()
            )


            retry_prompt = (
                build_mandatory_retry_prompt(
                    contract
                )
            )


            prompt_construction_ms = (
                (
                    perf_counter()
                    -
                    prompt_started_at
                )
                *
                1000.0
            )


            messages = [
                {
                    "role":
                        "system",

                    "content":
                        SYSTEM_PROMPT,
                },

                {
                    "role":
                        "user",

                    "content":
                        retry_prompt,
                },
            ]


        inference_started_at = (
            perf_counter()
        )


        response = (
            make_native_tool_request(
                model=(
                    model
                ),
                messages=(
                    messages
                ),
                seed=(
                    41
                    +
                    attempt_index
                ),
            )
        )


        model_inference_ms = (
            (
                perf_counter()
                -
                inference_started_at
            )
            *
            1000.0
        )


        parse_started_at = (
            perf_counter()
        )


        tool_calls = (
            response
            .message
            .tool_calls
            or []
        )


        content = (
            response_content(
                response
            )
        )


        selected_tool_name = (
            (
                tool_calls[
                    0
                ]
                .function
                .name
            )
            if (
                len(
                    tool_calls
                )
                ==
                1
            )
            else None
        )


        errors = (
            []
            if (
                len(
                    tool_calls
                )
                ==
                1
            )
            else [
                (
                    "Expected exactly one native "
                    "tool call but received "
                    f"{len(tool_calls)}."
                )
            ]
        )


        proposal = None


        if (
            len(
                tool_calls
            )
            ==
            1
        ):
            call = (
                tool_calls[
                    0
                ]
            )


            proposal = (
                NativeToolCallProposal(
                    tool_name=(
                        call
                        .function
                        .name
                    ),
                    arguments=dict(
                        call
                        .function
                        .arguments
                    ),
                )
            )


        response_parse_ms = (
            (
                perf_counter()
                -
                parse_started_at
            )
            *
            1000.0
        )


        attempt = (
            NativeToolCallAttempt(
                attempt_index=(
                    attempt_index
                ),
                prompt_variant=(
                    prompt_variant
                ),
                tool_call_count=(
                    len(
                        tool_calls
                    )
                ),
                assistant_content=(
                    content
                ),
                selected_tool_name=(
                    selected_tool_name
                ),
                errors=(
                    errors
                ),
                prompt_construction_ms=(
                    prompt_construction_ms
                ),
                model_inference_ms=(
                    model_inference_ms
                ),
                response_parse_ms=(
                    response_parse_ms
                ),
                total_ms=(
                    (
                        perf_counter()
                        -
                        attempt_started_at
                    )
                    *
                    1000.0
                ),
            )
        )


        attempts.append(
            attempt
        )


        if proposal is None:
            continue


        return (
            NativeToolCallRequestResult(
                proposal=(
                    proposal
                ),
                attempts=(
                    attempts
                ),
            )
        )


    return (
        NativeToolCallRequestResult(
            proposal=None,
            attempts=(
                attempts
            ),
        )
    )


# ============================================================
# DETERMINISTIC TOOL-CALL VALIDATION
# ============================================================

def validate_native_tool_call(
    *,
    contract: AnalyticalContract,
    proposal: NativeToolCallProposal,
) -> list[
    str
]:
    errors: list[
        str
    ] = []


    try:
        expected_spec = (
            native_tool_spec_for_contract(
                contract
            )
        )


        expected = (
            expected_tool_arguments(
                contract
            )
        )


    except ValueError as error:
        return [
            str(
                error
            )
        ]


    if (
        proposal.tool_name
        !=
        expected_spec.tool_name
    ):
        errors.append(
            (
                "Native tool name does not match "
                "the whitelisted tool for the "
                "validated analytical family. "
                f"Expected `{expected_spec.tool_name}` "
                f"for `{contract.family}`."
            )
        )


    if isinstance(
        expected,
        AggregationToolArgs,
    ):
        try:
            received = (
                AggregationToolArgs
                .model_validate(
                    proposal.arguments
                )
            )


        except Exception as error:
            errors.append(
                (
                    "Native aggregation arguments do not "
                    "match the required schema: "
                    f"{error}"
                )
            )

            return errors


    elif isinstance(
        expected,
        RankingToolArgs,
    ):
        try:
            received = (
                RankingToolArgs
                .model_validate(
                    proposal.arguments
                )
            )


        except Exception as error:
            errors.append(
                (
                    "Native ranking arguments do not match "
                    "the required schema: "
                    f"{error}"
                )
            )

            return errors


    elif isinstance(
        expected,
        GroupComparisonToolArgs,
    ):
        try:
            received = (
                GroupComparisonToolArgs
                .model_validate(
                    proposal.arguments
                )
            )


        except Exception as error:
            errors.append(
                (
                    "Native group-comparison arguments "
                    "do not match the required schema: "
                    f"{error}"
                )
            )

            return errors


    elif isinstance(
        expected,
        DistributionToolArgs,
    ):
        try:
            received = (
                DistributionToolArgs
                .model_validate(
                    proposal.arguments
                )
            )


        except Exception as error:
            errors.append(
                (
                    "Native distribution arguments do not "
                    "match the required schema: "
                    f"{error}"
                )
            )

            return errors


    elif isinstance(
        expected,
        TimeSeriesToolArgs,
    ):
        try:
            received = (
                TimeSeriesToolArgs
                .model_validate(
                    proposal.arguments
                )
            )


        except Exception as error:
            errors.append(
                (
                    "Native time-series arguments do not "
                    "match the required schema: "
                    f"{error}"
                )
            )

            return errors


    else:
        try:
            received = (
                TwoVariableToolArgs
                .model_validate(
                    proposal.arguments
                )
            )


        except Exception as error:
            errors.append(
                (
                    "Native two-variable arguments do not "
                    "match the required schema: "
                    f"{error}"
                )
            )

            return errors


    expected_values = (
        expected
        .model_dump()
    )


    received_values = (
        received
        .model_dump()
    )


    for (
        key,
        expected_value,
    ) in expected_values.items():
        received_value = (
            received_values.get(
                key
            )
        )


        if (
            received_value
            !=
            expected_value
        ):
            errors.append(
                (
                    f"Native tool argument `{key}` differs "
                    "from the validated contract. "
                    f"Expected `{expected_value}`, "
                    f"received `{received_value}`."
                )
            )


    return errors


# ============================================================
# COMPLETE NATIVE TOOL FLOW
# ============================================================

def run_native_tool_call(
    *,
    contract: AnalyticalContract,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
    model: str = (
        DEFAULT_NATIVE_TOOL_MODEL
    ),
) -> NativeToolCallingReport:
    total_started_at = (
        perf_counter()
    )


    try:
        expected_spec = (
            native_tool_spec_for_contract(
                contract
            )
        )


    except ValueError as error:
        return (
            NativeToolCallingReport(
                model=(
                    model
                ),
                contract_id=(
                    contract
                    .contract_id
                ),
                contract_family=(
                    contract
                    .family
                ),
                available_tools=(
                    sorted(
                        SUPPORTED_NATIVE_TOOLS
                    )
                ),
                expected_tool=None,
                tool_call_received=False,
                requested_tool=None,
                requested_arguments={},
                validation_status=(
                    "rejected"
                ),
                validation_errors=[
                    str(
                        error
                    ),
                ],
                attempt_count=0,
                retry_count=0,
                attempts=[],
                execution=None,
                timing=(
                    NativeToolTiming(
                        total_ms=(
                            (
                                perf_counter()
                                -
                                total_started_at
                            )
                            *
                            1000.0
                        )
                    )
                ),
                notes=[
                    (
                        "The analytical family is not "
                        "available in the native tool "
                        "registry."
                    ),
                ],
            )
        )


    try:
        request_result = (
            request_native_tool_call(
                contract=(
                    contract
                ),
                model=(
                    model
                ),
            )
        )


    except Exception as error:
        return (
            NativeToolCallingReport(
                model=(
                    model
                ),
                contract_id=(
                    contract
                    .contract_id
                ),
                contract_family=(
                    contract
                    .family
                ),
                available_tools=(
                    sorted(
                        SUPPORTED_NATIVE_TOOLS
                    )
                ),
                expected_tool=(
                    expected_spec
                    .tool_name
                ),
                tool_call_received=False,
                requested_tool=None,
                requested_arguments={},
                validation_status=(
                    "rejected"
                ),
                validation_errors=[
                    str(
                        error
                    ),
                ],
                attempt_count=0,
                retry_count=0,
                attempts=[],
                execution=None,
                timing=(
                    NativeToolTiming(
                        total_ms=(
                            (
                                perf_counter()
                                -
                                total_started_at
                            )
                            *
                            1000.0
                        )
                    )
                ),
                notes=[
                    (
                        "No deterministic tool was "
                        "executed because the Ollama "
                        "native tool request failed."
                    ),
                ],
            )
        )


    attempts = (
        request_result
        .attempts
    )


    attempt_count = (
        len(
            attempts
        )
    )


    retry_count = max(
        0,
        attempt_count
        -
        1,
    )


    prompt_construction_ms = sum(
        attempt.prompt_construction_ms
        for attempt
        in attempts
    )


    model_inference_ms = sum(
        attempt.model_inference_ms
        for attempt
        in attempts
    )


    response_parse_ms = sum(
        attempt.response_parse_ms
        for attempt
        in attempts
    )


    proposal = (
        request_result
        .proposal
    )


    if (
        proposal is None
    ):
        return (
            NativeToolCallingReport(
                model=(
                    model
                ),
                contract_id=(
                    contract
                    .contract_id
                ),
                contract_family=(
                    contract
                    .family
                ),
                available_tools=(
                    sorted(
                        SUPPORTED_NATIVE_TOOLS
                    )
                ),
                expected_tool=(
                    expected_spec
                    .tool_name
                ),
                tool_call_received=False,
                requested_tool=None,
                requested_arguments={},
                validation_status=(
                    "rejected"
                ),
                validation_errors=[
                    (
                        "The native tool model returned "
                        "no single executable tool call "
                        f"after {attempt_count} attempt(s)."
                    ),
                ],
                attempt_count=(
                    attempt_count
                ),
                retry_count=(
                    retry_count
                ),
                attempts=(
                    attempts
                ),
                execution=None,
                timing=(
                    NativeToolTiming(
                        prompt_construction_ms=(
                            prompt_construction_ms
                        ),
                        model_inference_ms=(
                            model_inference_ms
                        ),
                        response_parse_ms=(
                            response_parse_ms
                        ),
                        total_ms=(
                            (
                                perf_counter()
                                -
                                total_started_at
                            )
                            *
                            1000.0
                        ),
                    )
                ),
                notes=[
                    (
                        "DataLens did not fall back to "
                        "parsing assistant prose as a "
                        "function call."
                    ),
                    (
                        "No deterministic tool was "
                        "executed because the native "
                        "function-calling boundary was "
                        "not satisfied."
                    ),
                ],
            )
        )


    validation_started_at = (
        perf_counter()
    )


    validation_errors = (
        validate_native_tool_call(
            contract=(
                contract
            ),
            proposal=(
                proposal
            ),
        )
    )


    python_validation_ms = (
        (
            perf_counter()
            -
            validation_started_at
        )
        *
        1000.0
    )


    if (
        validation_errors
    ):
        return (
            NativeToolCallingReport(
                model=(
                    model
                ),
                contract_id=(
                    contract
                    .contract_id
                ),
                contract_family=(
                    contract
                    .family
                ),
                available_tools=(
                    sorted(
                        SUPPORTED_NATIVE_TOOLS
                    )
                ),
                expected_tool=(
                    expected_spec
                    .tool_name
                ),
                tool_call_received=True,
                requested_tool=(
                    proposal
                    .tool_name
                ),
                requested_arguments=(
                    proposal
                    .arguments
                ),
                validation_status=(
                    "rejected"
                ),
                validation_errors=(
                    validation_errors
                ),
                attempt_count=(
                    attempt_count
                ),
                retry_count=(
                    retry_count
                ),
                attempts=(
                    attempts
                ),
                execution=None,
                timing=(
                    NativeToolTiming(
                        prompt_construction_ms=(
                            prompt_construction_ms
                        ),
                        model_inference_ms=(
                            model_inference_ms
                        ),
                        response_parse_ms=(
                            response_parse_ms
                        ),
                        python_validation_ms=(
                            python_validation_ms
                        ),
                        total_ms=(
                            (
                                perf_counter()
                                -
                                total_started_at
                            )
                            *
                            1000.0
                        ),
                    )
                ),
                notes=[
                    (
                        "The model emitted a native "
                        "tool call, but Python refused "
                        "to execute it."
                    ),
                ],
            )
        )


    execution_started_at = (
        perf_counter()
    )


    execution = (
        execute_validated_contract(
            contract=(
                contract
            ),
            datasets=(
                datasets
            ),
            call_index=1,
        )
    )


    deterministic_execution_ms = (
        (
            perf_counter()
            -
            execution_started_at
        )
        *
        1000.0
    )


    return (
        NativeToolCallingReport(
            model=(
                model
            ),
            contract_id=(
                contract
                .contract_id
            ),
            contract_family=(
                contract
                .family
            ),
            available_tools=(
                sorted(
                    SUPPORTED_NATIVE_TOOLS
                )
            ),
            expected_tool=(
                expected_spec
                .tool_name
            ),
            tool_call_received=True,
            requested_tool=(
                proposal
                .tool_name
            ),
            requested_arguments=(
                proposal
                .arguments
            ),
            validation_status=(
                "validated"
            ),
            validation_errors=[],
            attempt_count=(
                attempt_count
            ),
            retry_count=(
                retry_count
            ),
            attempts=(
                attempts
            ),
            execution=(
                execution
            ),
            timing=(
                NativeToolTiming(
                    prompt_construction_ms=(
                        prompt_construction_ms
                    ),
                    model_inference_ms=(
                        model_inference_ms
                    ),
                    response_parse_ms=(
                        response_parse_ms
                    ),
                    python_validation_ms=(
                        python_validation_ms
                    ),
                    deterministic_execution_ms=(
                        deterministic_execution_ms
                    ),
                    total_ms=(
                        (
                            perf_counter()
                            -
                            total_started_at
                        )
                        *
                        1000.0
                    ),
                )
            ),
            notes=[
                (
                    "The local tool model selected one "
                    "function from the native DataLens "
                    "tool catalog."
                ),
                (
                    "Python verified that the selected "
                    "tool matched the validated analytical "
                    "family and that every argument exactly "
                    "matched the validated contract."
                ),
                (
                    "A missing tool call may trigger one "
                    "stricter retry; ordinary assistant "
                    "prose is never treated as executable."
                ),
                (
                    "The deterministic DataLens executor "
                    "performed the statistical calculation."
                ),
            ],
        )
    )
