from __future__ import annotations


from typing import (
    Any,
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# ============================================================
# VERSION
# ============================================================

ANALYTICAL_CONTRACT_RULE_VERSION = (
    "analytical_contract_v0.3"
)


# ============================================================
# GENERIC ANALYTICAL VOCABULARY
# ============================================================

AnalysisFamily = Literal[
    "descriptive_metric",
    "aggregation",
    "ranking",
    "time_series",
    "quantitative_association",
    "categorical_association",
    "group_comparison",
    "distribution",
    "inequality",
    "data_quality",
    "unresolved",
]


AnalysisOrigin = Literal[
    "user_objective",
    "document_request",
    "exploratory",
    "ai_planner",
    "legacy_adapter",
]


ContractStatus = Literal[
    "proposed",
    "validated",
    "blocked",
    "ambiguous",
]


VariableRole = Literal[
    "x",
    "y",
    "time",
    "value",
    "group",
    "dimension",
    "entity",
    "weight",
]


AggregationFunction = Literal[
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "count",
    "distinct_count",
]


GroupingRole = Literal[
    "time",
    "group",
    "dimension",
]


AggregationSourceRole = Literal[
    "x",
    "y",
    "value",
    "entity",
    "dimension",
    "group",
]


SortDirection = Literal[
    "ascending",
    "descending",
]


BenchmarkReference = Literal[
    "overall_aggregate",
]


BenchmarkOperator = Literal[
    "gt",
    "gte",
    "lt",
    "lte",
]


BenchmarkSelection = Literal[
    "matching_only",
    "annotate_all",
]


FilterOperator = Literal[
    "eq",
    "neq",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
    "not_in",
    "is_null",
    "not_null",
]


WindowOperation = Literal[
    "moving_average",
    "rolling_sum",
    "rolling_median",
]


DerivedOperation = Literal[
    "aggregate",
    "ratio",
    "difference",
    "date_difference",
    "age_at_event",
    "date_trunc",
    "custom_deterministic",
]


PrimitiveValue = (
    str
    | int
    | float
    | bool
    | None
)


FilterValue = (
    PrimitiveValue
    | list[
        PrimitiveValue
    ]
)


# ============================================================
# SOURCE PROVENANCE
# ============================================================

class ContractProvenance(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    source_filename: (
        str
        | None
    ) = None

    source_locator: (
        str
        | None
    ) = None

    page_number: (
        int
        | None
    ) = None

    source_chunk_id: (
        str
        | None
    ) = None

    evidence_unit_id: (
        int
        | None
    ) = None

    evidence_quote: (
        str
        | None
    ) = None


# ============================================================
# VARIABLE BINDINGS
# ============================================================

class VariableBinding(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    role: VariableRole

    column: str = Field(
        min_length=1
    )

    dataset_id: (
        str
        | None
    ) = None

    dataset_filename: (
        str
        | None
    ) = None

    semantic_concept: (
        str
        | None
    ) = None

    analysis_kind: (
        str
        | None
    ) = None


# ============================================================
# AGGREGATION
# ============================================================

class AggregationSpec(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    function: AggregationFunction

    source_role: (
        AggregationSourceRole
        | None
    ) = None

    group_by_roles: list[
        GroupingRole
    ] = Field(
        default_factory=list
    )

    output_name: (
        str
        | None
    ) = None


    @model_validator(
        mode="after"
    )
    def validate_source_role(
        self,
    ) -> "AggregationSpec":
        if (
            self.function !=
            "count"
            and self.source_role is None
        ):
            raise ValueError(
                "source_role is required for every aggregation "
                "except a row count."
            )


        return self


# ============================================================
# RANKING
# ============================================================

class RankingSpec(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    order: SortDirection

    limit: int = Field(
        default=10,
        ge=1,
        le=100,
    )


# ============================================================
# BENCHMARK
# DATALENS_CANONICAL_BENCHMARK_SPEC_V0_1
# ============================================================

class BenchmarkSpec(
    BaseModel
):
    """
    Generic deterministic comparison applied after a grouped
    aggregation.

    `overall_aggregate` means:

    - reuse the contract aggregation function;
    - reuse the same aggregation source;
    - reuse the same analytical population;
    - suppress the grouping roles for the reference value.

    Example:

        grouped metric:
            mean(value) by group

        benchmark:
            reference = overall_aggregate
            operator = gt

    This is generic analytical vocabulary. Business concepts
    such as return rate remain outside the canonical core.
    """

    model_config = ConfigDict(
        extra="forbid"
    )


    reference: BenchmarkReference = (
        "overall_aggregate"
    )

    operator: BenchmarkOperator

    selection: BenchmarkSelection = (
        "matching_only"
    )


# ============================================================
# TIME WINDOWS
# ============================================================

class WindowSpec(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    operation: WindowOperation

    window: int = Field(
        ge=2,
        le=365,
    )

    minimum_periods: int = Field(
        default=1,
        ge=1,
        le=365,
    )


    @model_validator(
        mode="after"
    )
    def validate_minimum_periods(
        self,
    ) -> "WindowSpec":
        if (
            self.minimum_periods >
            self.window
        ):
            raise ValueError(
                "minimum_periods cannot exceed window."
            )


        return self


# ============================================================
# FILTERS
# ============================================================

class FilterSpec(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    column: str = Field(
        min_length=1
    )

    operator: FilterOperator

    value: (
        FilterValue
    ) = None


    @model_validator(
        mode="after"
    )
    def validate_filter_value(
        self,
    ) -> "FilterSpec":
        if (
            self.operator in {
                "is_null",
                "not_null",
            }
            and self.value is not None
        ):
            raise ValueError(
                "Null filters must not define a value."
            )


        if (
            self.operator in {
                "in",
                "not_in",
            }
            and not isinstance(
                self.value,
                list,
            )
        ):
            raise ValueError(
                "The 'in' and 'not_in' operators require a list value."
            )


        return self


# ============================================================
# JOIN REQUIREMENTS
# ============================================================

class JoinRequirement(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    left_dataset_id: str = Field(
        min_length=1
    )

    right_dataset_id: str = Field(
        min_length=1
    )

    left_column: str = Field(
        min_length=1
    )

    right_column: str = Field(
        min_length=1
    )

    expected_cardinality: (
        Literal[
            "one_to_one",
            "one_to_many",
            "many_to_one",
            "unknown",
        ]
    ) = "unknown"

    preserve_left_grain: bool = (
        True
    )

    required: bool = (
        True
    )


# ============================================================
# DERIVED VARIABLES
# ============================================================

class DerivedVariableSpec(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    output_column: str = Field(
        min_length=1
    )

    operation: DerivedOperation

    source_columns: list[
        str
    ] = Field(
        default_factory=list
    )

    group_by_columns: list[
        str
    ] = Field(
        default_factory=list
    )

    parameters: dict[
        str,
        PrimitiveValue,
    ] = Field(
        default_factory=dict
    )

    rationale: (
        str
        | None
    ) = None


# ============================================================
# GENERIC ANALYTICAL CONTRACT
# ============================================================

class AnalyticalContract(
    BaseModel
):
    """
    Generic analytical plan understood by DataLens.

    Important design rule:
    - business vocabulary belongs in request_text, provenance,
      semantic_concept and concrete column names;
    - DataLens core behavior is described only by generic analytical
      families, roles and deterministic operations.
    """

    model_config = ConfigDict(
        extra="forbid"
    )


    contract_id: str = Field(
        min_length=1
    )

    contract_version: str = (
        ANALYTICAL_CONTRACT_RULE_VERSION
    )

    origin: AnalysisOrigin

    status: ContractStatus

    title: str = Field(
        min_length=1
    )

    request_text: str = Field(
        min_length=1
    )

    family: AnalysisFamily

    required_dataset_ids: list[
        str
    ] = Field(
        default_factory=list
    )

    required_dataset_filenames: list[
        str
    ] = Field(
        default_factory=list
    )

    analytical_grain: (
        str
        | None
    ) = None

    bindings: list[
        VariableBinding
    ] = Field(
        default_factory=list
    )

    aggregation: (
        AggregationSpec
        | None
    ) = None

    ranking: (
        RankingSpec
        | None
    ) = None

    benchmark: (
        BenchmarkSpec
        | None
    ) = None

    window: (
        WindowSpec
        | None
    ) = None

    filters: list[
        FilterSpec
    ] = Field(
        default_factory=list
    )

    joins: list[
        JoinRequirement
    ] = Field(
        default_factory=list
    )

    derived_variables: list[
        DerivedVariableSpec
    ] = Field(
        default_factory=list
    )

    required_operations: list[
        str
    ] = Field(
        default_factory=list
    )

    provenance: (
        ContractProvenance
        | None
    ) = None

    reasons: list[
        str
    ] = Field(
        default_factory=list
    )

    blockers: list[
        str
    ] = Field(
        default_factory=list
    )

    planner_confidence: (
        float
        | None
    ) = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


    def roles(
        self,
    ) -> set[
        str
    ]:
        return {
            binding.role
            for binding
            in self.bindings
        }


    def bindings_for_role(
        self,
        role: VariableRole,
    ) -> list[
        VariableBinding
    ]:
        return [
            binding
            for binding
            in self.bindings
            if binding.role ==
            role
        ]


    @model_validator(
        mode="after"
    )
    def validate_contract(
        self,
    ) -> "AnalyticalContract":
        # Blocked and ambiguous plans are allowed to remain incomplete.
        # This is essential for safe abstention: DataLens must be able
        # to represent "I cannot execute this yet" without inventing
        # variables merely to satisfy a schema.
        if (
            self.status in {
                "blocked",
                "ambiguous",
            }
        ):
            if (
                len(
                    self.blockers
                ) ==
                0
            ):
                raise ValueError(
                    "A blocked or ambiguous contract must explain "
                    "at least one blocker."
                )


            return self


        if (
            self.family ==
            "unresolved"
        ):
            raise ValueError(
                "An unresolved analytical family cannot be proposed "
                "as an executable contract. Use status='ambiguous' "
                "or status='blocked'."
            )


        roles = (
            self.roles()
        )


        if (
            self.family ==
            "quantitative_association"
        ):
            self._require_roles(
                roles,
                {
                    "x",
                    "y",
                },
            )


        elif (
            self.family ==
            "categorical_association"
        ):
            self._require_roles(
                roles,
                {
                    "x",
                    "y",
                },
            )


        elif (
            self.family ==
            "group_comparison"
        ):
            self._require_roles(
                roles,
                {
                    "group",
                    "value",
                },
            )


        elif (
            self.family ==
            "aggregation"
        ):
            self._require_aggregation(
                roles
            )


        elif (
            self.family ==
            "ranking"
        ):
            self._require_aggregation(
                roles
            )

            if (
                self.ranking is None
            ):
                raise ValueError(
                    "A ranking contract requires ranking settings."
                )


            if not (
                {
                    "dimension",
                    "group",
                }
                &
                roles
            ):
                raise ValueError(
                    "A ranking contract requires a dimension or group binding."
                )


        # ====================================================
        # BENCHMARK INVARIANTS
        #
        # Benchmark v0.1 is deliberately a post-aggregation
        # operation. It is accepted only for grouped aggregation
        # contracts. Execution support is added separately.
        # ====================================================

        if (
            self.benchmark
            is not None
        ):
            if (
                self.family
                !=
                "aggregation"
            ):
                raise ValueError(
                    "BenchmarkSpec v0.1 is supported only for "
                    "aggregation contracts."
                )


            if (
                self.aggregation
                is None
            ):
                raise ValueError(
                    "A benchmark contract requires an "
                    "AggregationSpec."
                )


            if not (
                self.aggregation
                .group_by_roles
            ):
                raise ValueError(
                    "BenchmarkSpec requires at least one grouped "
                    "aggregation role."
                )


        if (
            self.family ==
            "time_series"
        ):
            if (
                "time"
                not in roles
            ):
                raise ValueError(
                    "A time_series contract requires a time binding."
                )


            self._require_aggregation(
                roles
            )


        elif (
            self.family ==
            "descriptive_metric"
        ):
            self._require_aggregation(
                roles
            )


        elif (
            self.family ==
            "distribution"
        ):
            if not (
                {
                    "value",
                    "group",
                    "dimension",
                    "x",
                    "y",
                }
                &
                roles
            ):
                raise ValueError(
                    "A distribution contract requires at least one "
                    "analytical variable."
                )


        elif (
            self.family ==
            "inequality"
        ):
            self._require_roles(
                roles,
                {
                    "entity",
                    "value",
                },
            )


        elif (
            self.family ==
            "data_quality"
        ):
            # Dataset-level quality checks may legitimately have no
            # variable binding.
            pass


        return self


    def _require_roles(
        self,
        roles: set[
            str
        ],
        required_roles: set[
            str
        ],
    ) -> None:
        missing = (
            required_roles -
            roles
        )


        if (
            missing
        ):
            raise ValueError(
                "Missing required role(s): "
                +
                ", ".join(
                    sorted(
                        missing
                    )
                )
                +
                "."
            )


    def _require_aggregation(
        self,
        roles: set[
            str
        ],
    ) -> None:
        if (
            self.aggregation is None
        ):
            raise ValueError(
                f"A {self.family} contract requires an aggregation specification."
            )


        source_role = (
            self.aggregation
            .source_role
        )


        if (
            source_role is not None
            and source_role
            not in roles
        ):
            raise ValueError(
                "The aggregation source_role is not bound to a variable: "
                f"{source_role}."
            )


        missing_group_roles = (
            set(
                self.aggregation
                .group_by_roles
            )
            -
            roles
        )


        if (
            missing_group_roles
        ):
            raise ValueError(
                "Aggregation group_by role(s) are not bound: "
                +
                ", ".join(
                    sorted(
                        missing_group_roles
                    )
                )
                +
                "."
            )


# ============================================================
# STRUCTURED OUTPUT SCHEMA
# ============================================================

def analytical_contract_json_schema() -> dict[
    str,
    Any,
]:
    """
    JSON Schema intended for structured LLM generation.

    The LLM proposes a contract matching this schema.
    Python/Pydantic remains the authority that validates it.
    """
    return (
        AnalyticalContract
        .model_json_schema()
    )
