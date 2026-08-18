from __future__ import annotations

from enum import Enum

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# VERSION
# ============================================================


TRANSFORMATION_CONTRACT_VERSION = (
    "transformation_contracts_v0.1"
)


# ============================================================
# ENUMS
# ============================================================


class TransformationOperation(
    str,
    Enum,
):
    DERIVE_ARITHMETIC = (
        "derive_arithmetic"
    )

    CAST = (
        "cast"
    )

    BIN_NUMERIC = (
        "bin_numeric"
    )

    EXTRACT_DATE_PART = (
        "extract_date_part"
    )

    AGGREGATE = (
        "aggregate"
    )


class TransformationStatus(
    str,
    Enum,
):
    VALIDATED = (
        "validated"
    )

    REVIEW_REQUIRED = (
        "review_required"
    )


class TransformationRisk(
    str,
    Enum,
):
    LOW = (
        "low"
    )

    MEDIUM = (
        "medium"
    )

    HIGH = (
        "high"
    )


class ArithmeticOperator(
    str,
    Enum,
):
    ADD = (
        "add"
    )

    SUBTRACT = (
        "subtract"
    )

    MULTIPLY = (
        "multiply"
    )

    DIVIDE = (
        "divide"
    )


class OperandKind(
    str,
    Enum,
):
    COLUMN = (
        "column"
    )

    LITERAL = (
        "literal"
    )


class CastTargetType(
    str,
    Enum,
):
    STRING = (
        "string"
    )

    INTEGER = (
        "integer"
    )

    FLOAT = (
        "float"
    )

    BOOLEAN = (
        "boolean"
    )

    DATETIME = (
        "datetime"
    )


class DatePart(
    str,
    Enum,
):
    YEAR = (
        "year"
    )

    MONTH = (
        "month"
    )

    DAY = (
        "day"
    )

    QUARTER = (
        "quarter"
    )

    WEEK = (
        "week"
    )

    WEEKDAY = (
        "weekday"
    )


class AggregationFunction(
    str,
    Enum,
):
    SUM = (
        "sum"
    )

    MEAN = (
        "mean"
    )

    MEDIAN = (
        "median"
    )

    MIN = (
        "min"
    )

    MAX = (
        "max"
    )

    COUNT = (
        "count"
    )

    NUNIQUE = (
        "nunique"
    )


# ============================================================
# OPERANDS
# ============================================================


class TransformationOperand(
    BaseModel,
):
    """
    Structured arithmetic operand.

    Exactly one of:

    COLUMN:
        column=<existing column>
        value=None

    LITERAL:
        column=None
        value=<numeric literal>

    The planner performs the strict validation.
    """

    kind: OperandKind

    column: Optional[
        str
    ] = None

    value: Optional[
        Any
    ] = None


# ============================================================
# DERIVED ARITHMETIC
# ============================================================


class DeriveArithmeticIntent(
    BaseModel,
):
    request_id: str

    dataset_id: str

    dataset_filename: str

    operation: TransformationOperation = (
        TransformationOperation
        .DERIVE_ARITHMETIC
    )

    output_column: str

    left: TransformationOperand

    operator: ArithmeticOperator

    right: TransformationOperand


# ============================================================
# CAST
# ============================================================


class CastIntent(
    BaseModel,
):
    request_id: str

    dataset_id: str

    dataset_filename: str

    operation: TransformationOperation = (
        TransformationOperation
        .CAST
    )

    source_column: str

    output_column: str

    target_type: CastTargetType


# ============================================================
# NUMERIC BINNING
# ============================================================


class BinNumericIntent(
    BaseModel,
):
    request_id: str

    dataset_id: str

    dataset_filename: str

    operation: TransformationOperation = (
        TransformationOperation
        .BIN_NUMERIC
    )

    source_column: str

    output_column: str

    bins: List[
        float
    ]

    labels: Optional[
        List[
            str
        ]
    ] = None

    include_lowest: bool = True

    right: bool = True


# ============================================================
# DATE PART
# ============================================================


class ExtractDatePartIntent(
    BaseModel,
):
    request_id: str

    dataset_id: str

    dataset_filename: str

    operation: TransformationOperation = (
        TransformationOperation
        .EXTRACT_DATE_PART
    )

    source_column: str

    output_column: str

    part: DatePart


# ============================================================
# AGGREGATION
# ============================================================


class AggregationMetric(
    BaseModel,
):
    source_column: str

    function: AggregationFunction

    output_column: str


class AggregateIntent(
    BaseModel,
):
    request_id: str

    dataset_id: str

    dataset_filename: str

    operation: TransformationOperation = (
        TransformationOperation
        .AGGREGATE
    )

    group_by: List[
        str
    ]

    metrics: List[
        AggregationMetric
    ]

    output_dataset_id: str

    output_dataset_filename: str


# ============================================================
# INTENT UNION
# ============================================================


TransformationIntent = Union[
    DeriveArithmeticIntent,
    CastIntent,
    BinNumericIntent,
    ExtractDatePartIntent,
    AggregateIntent,
]


# ============================================================
# PLANNED STEP
# ============================================================


class TransformationStep(
    BaseModel,
):
    """
    Deterministically validated transformation proposal.

    executable always remains False at the planning stage.
    """

    step_id: str

    request_id: str

    dataset_id: str

    dataset_filename: str

    operation: TransformationOperation

    status: TransformationStatus

    risk: TransformationRisk

    input_columns: List[
        str
    ] = Field(
        default_factory=list
    )

    output_column: Optional[
        str
    ] = None

    output_dataset_id: Optional[
        str
    ] = None

    output_dataset_filename: Optional[
        str
    ] = None

    parameters: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    rationale: str

    requires_human_approval: bool

    executable: bool = False


# ============================================================
# PLAN
# ============================================================


class TransformationPlan(
    BaseModel,
):
    dataset_id: str

    dataset_filename: str

    request_count: int

    step_count: int

    validated_count: int

    review_required_count: int

    human_approval_required_count: int

    ready_for_approval: bool

    steps: List[
        TransformationStep
    ] = Field(
        default_factory=list
    )

    notes: List[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        TRANSFORMATION_CONTRACT_VERSION
    )