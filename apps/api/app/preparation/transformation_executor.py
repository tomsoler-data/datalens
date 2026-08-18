from __future__ import annotations

from dataclasses import dataclass

from enum import Enum

import hashlib

import json

from numbers import Number

from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
)

import pandas as pd

from pydantic import (
    BaseModel,
    Field,
)

from app.preparation.transformation_approval import (
    ApprovedTransformationPlan,
    ApprovedTransformationStep,
    TransformationAuthorizationStatus,
)

from app.preparation.transformation_contracts import (
    AggregationFunction,
    ArithmeticOperator,
    CastTargetType,
    DatePart,
    OperandKind,
    TransformationOperation,
)


# ============================================================
# VERSION
# ============================================================


TRANSFORMATION_EXECUTOR_RULE_VERSION = (
    "transformation_executor_v0.1"
)


# ============================================================
# ERROR
# ============================================================


class TransformationExecutionError(
    RuntimeError,
):
    """
    Raised when an approved transformation cannot be safely
    executed.

    The executor is transactional: when this exception is
    raised, no partially transformed DataFrame is returned.
    """


# ============================================================
# STATUS
# ============================================================


class TransformationExecutionStatus(
    str,
    Enum,
):
    APPLIED = (
        "applied"
    )

    SKIPPED = (
        "skipped"
    )


# ============================================================
# STEP REPORT
# ============================================================


class TransformationStepExecution(
    BaseModel,
):
    order: int

    request_id: str

    operation: TransformationOperation

    authorization_status: (
        TransformationAuthorizationStatus
    )

    status: TransformationExecutionStatus

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

    source_rows_before: int

    source_rows_after: int

    source_columns_before: int

    source_columns_after: int

    affected_rows: int = 0

    affected_cells: int = 0

    source_fingerprint_before: str

    source_fingerprint_after: str

    output_fingerprint: Optional[
        str
    ] = None

    rationale: str


# ============================================================
# EXECUTION REPORT
# ============================================================


class TransformationExecutionReport(
    BaseModel,
):
    status: Literal[
        "success"
    ] = (
        "success"
    )

    dataset_id: str

    dataset_filename: str

    source_rows_before: int

    source_rows_after: int

    source_columns_before: int

    source_columns_after: int

    source_fingerprint_before: str

    source_fingerprint_after: str

    source_data_changed: bool

    total_step_count: int

    executable_step_count: int

    applied_count: int

    skipped_count: int

    derived_dataset_count: int

    derived_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    steps: List[
        TransformationStepExecution
    ] = Field(
        default_factory=list
    )

    notes: List[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        TRANSFORMATION_EXECUTOR_RULE_VERSION
    )


# ============================================================
# RESULT
# ============================================================


@dataclass(
    frozen=True
)
class TransformationExecutionResult:
    dataframe: pd.DataFrame

    derived_datasets: Dict[
        str,
        pd.DataFrame,
    ]

    report: TransformationExecutionReport


# ============================================================
# GENERIC HELPERS
# ============================================================


def _enum_value(
    value: Any,
) -> str:
    if hasattr(
        value,
        "value",
    ):
        return str(
            value.value
        )

    return str(
        value
    )


def _dataframe_fingerprint(
    dataframe: pd.DataFrame,
) -> str:
    """
    Deterministic fingerprint including:

    - column names;
    - dtypes;
    - index;
    - cell values.

    No mutation occurs.
    """

    metadata = {
        "columns": [
            str(
                column
            )
            for column
            in dataframe.columns
        ],

        "dtypes": [
            str(
                dtype
            )
            for dtype
            in dataframe.dtypes
        ],

        "shape": [
            int(
                dataframe.shape[
                    0
                ]
            ),
            int(
                dataframe.shape[
                    1
                ]
            ),
        ],
    }

    digest = (
        hashlib.sha256()
    )

    digest.update(
        json.dumps(
            metadata,
            sort_keys=
                True,
            ensure_ascii=
                False,
        )
        .encode(
            "utf-8"
        )
    )

    try:
        hashed = (
            pd.util
            .hash_pandas_object(
                dataframe,
                index=
                    True,
                categorize=
                    True,
            )
        )

    except TypeError:
        # Conservative fallback for unusual object columns.
        hashed = (
            pd.util
            .hash_pandas_object(
                dataframe
                .astype(
                    "string"
                ),
                index=
                    True,
                categorize=
                    True,
            )
        )

    digest.update(
        hashed
        .to_numpy()
        .tobytes()
    )

    return (
        digest.hexdigest()
    )


def _require_column(
    *,
    dataframe: pd.DataFrame,
    column: str,
) -> None:
    if (
        column
        not in
        dataframe.columns
    ):
        raise (
            TransformationExecutionError(
                (
                    "Required transformation column "
                    "does not exist at execution time: "
                    f"{column}"
                )
            )
        )


def _require_new_column(
    *,
    dataframe: pd.DataFrame,
    column: str,
) -> None:
    if (
        column
        in
        dataframe.columns
    ):
        raise (
            TransformationExecutionError(
                (
                    "Transformation Executor v0.1 "
                    "refuses to overwrite an existing "
                    f"column: {column}"
                )
            )
        )


def _require_numeric_series(
    *,
    series: pd.Series,
    column: str,
) -> None:
    if not (
        pd.api.types
        .is_numeric_dtype(
            series
        )
    ):
        raise (
            TransformationExecutionError(
                (
                    "Transformation requires a "
                    "numeric execution-time column: "
                    f"{column}"
                )
            )
        )


# ============================================================
# OPERAND RESOLUTION
# ============================================================


def _resolve_operand(
    *,
    dataframe: pd.DataFrame,
    operand: Dict[
        str,
        Any,
    ],
):
    kind = (
        _enum_value(
            operand.get(
                "kind"
            )
        )
    )

    if (
        kind
        ==
        OperandKind.COLUMN.value
    ):
        column = (
            operand.get(
                "column"
            )
        )

        if not (
            isinstance(
                column,
                str,
            )
            and
            column
        ):
            raise (
                TransformationExecutionError(
                    (
                        "COLUMN operand does not "
                        "contain a valid column."
                    )
                )
            )

        _require_column(
            dataframe=
                dataframe,

            column=
                column,
        )

        series = (
            dataframe[
                column
            ]
        )

        _require_numeric_series(
            series=
                series,

            column=
                column,
        )

        return series

    if (
        kind
        ==
        OperandKind.LITERAL.value
    ):
        value = (
            operand.get(
                "value"
            )
        )

        if (
            not isinstance(
                value,
                Number,
            )
            or
            isinstance(
                value,
                bool,
            )
        ):
            raise (
                TransformationExecutionError(
                    (
                        "LITERAL operand must "
                        "contain a numeric value."
                    )
                )
            )

        return value

    raise (
        TransformationExecutionError(
            (
                "Unsupported transformation "
                f"operand kind: {kind}"
            )
        )
    )


# ============================================================
# DERIVED ARITHMETIC
# ============================================================


def _execute_derive_arithmetic(
    *,
    dataframe: pd.DataFrame,
    step: ApprovedTransformationStep,
) -> int:
    output_column = (
        step.output_column
    )

    if not (
        output_column
    ):
        raise (
            TransformationExecutionError(
                (
                    "DERIVE_ARITHMETIC requires "
                    "output_column."
                )
            )
        )

    _require_new_column(
        dataframe=
            dataframe,

        column=
            output_column,
    )

    left = (
        _resolve_operand(
            dataframe=
                dataframe,

            operand=
                step.parameters[
                    "left"
                ],
        )
    )

    right = (
        _resolve_operand(
            dataframe=
                dataframe,

            operand=
                step.parameters[
                    "right"
                ],
        )
    )

    operator = (
        _enum_value(
            step.parameters.get(
                "operator"
            )
        )
    )

    if (
        operator
        ==
        ArithmeticOperator.ADD.value
    ):
        result = (
            left
            +
            right
        )

    elif (
        operator
        ==
        ArithmeticOperator.SUBTRACT.value
    ):
        result = (
            left
            -
            right
        )

    elif (
        operator
        ==
        ArithmeticOperator.MULTIPLY.value
    ):
        result = (
            left
            *
            right
        )

    elif (
        operator
        ==
        ArithmeticOperator.DIVIDE.value
    ):
        # ----------------------------------------------------
        # Execution-time division guardrail.
        #
        # Planner validation cannot guarantee a denominator
        # column contains no zero.
        # ----------------------------------------------------

        if isinstance(
            right,
            pd.Series,
        ):
            zero_mask = (
                right
                .notna()
                &
                (
                    right
                    ==
                    0
                )
            )

            zero_count = int(
                zero_mask.sum()
            )

            if (
                zero_count
                >
                0
            ):
                raise (
                    TransformationExecutionError(
                        (
                            "Division denominator "
                            "contains zero values. "
                            f"Affected rows: {zero_count}"
                        )
                    )
                )

        else:
            if (
                right
                ==
                0
            ):
                raise (
                    TransformationExecutionError(
                        "Division by zero."
                    )
                )

        result = (
            left
            /
            right
        )

    else:
        raise (
            TransformationExecutionError(
                (
                    "Unsupported arithmetic "
                    f"operator: {operator}"
                )
            )
        )

    dataframe[
        output_column
    ] = result

    return int(
        len(
            dataframe
        )
    )


# ============================================================
# CAST
# ============================================================


def _strict_boolean_cast(
    series: pd.Series,
) -> pd.Series:
    """
    Conservative boolean parser.

    We explicitly avoid:

        astype(bool)

    because bool("false") == True in Python.
    """

    mapping = {
        "true":
            True,

        "false":
            False,

        "1":
            True,

        "0":
            False,

        "yes":
            True,

        "no":
            False,

        "y":
            True,

        "n":
            False,
    }

    output = (
        pd.Series(
            pd.NA,
            index=
                series.index,
            dtype=
                "boolean",
        )
    )

    non_missing = (
        series.notna()
    )

    normalized = (
        series
        .loc[
            non_missing
        ]
        .astype(
            str
        )
        .str
        .strip()
        .str
        .casefold()
    )

    invalid_mask = (
        ~normalized
        .isin(
            mapping.keys()
        )
    )

    if (
        bool(
            invalid_mask.any()
        )
    ):
        invalid_values = (
            normalized
            .loc[
                invalid_mask
            ]
            .drop_duplicates()
            .tolist()
        )

        raise (
            TransformationExecutionError(
                (
                    "BOOLEAN cast encountered "
                    "unsupported values: "
                    f"{invalid_values}"
                )
            )
        )

    output.loc[
        non_missing
    ] = (
        normalized.map(
            mapping
        )
    )

    return output


def _execute_cast(
    *,
    dataframe: pd.DataFrame,
    step: ApprovedTransformationStep,
) -> int:
    source_column = (
        step.parameters.get(
            "source_column"
        )
    )

    output_column = (
        step.output_column
    )

    if not (
        isinstance(
            source_column,
            str,
        )
        and
        source_column
    ):
        raise (
            TransformationExecutionError(
                (
                    "CAST requires "
                    "source_column."
                )
            )
        )

    if not (
        output_column
    ):
        raise (
            TransformationExecutionError(
                (
                    "CAST requires "
                    "output_column."
                )
            )
        )

    _require_column(
        dataframe=
            dataframe,

        column=
            source_column,
    )

    _require_new_column(
        dataframe=
            dataframe,

        column=
            output_column,
    )

    source = (
        dataframe[
            source_column
        ]
    )

    target_type = (
        _enum_value(
            step.parameters.get(
                "target_type"
            )
        )
    )

    # ========================================================
    # STRING
    # ========================================================

    if (
        target_type
        ==
        CastTargetType.STRING.value
    ):
        result = (
            source.astype(
                "string"
            )
        )

    # ========================================================
    # FLOAT
    # ========================================================

    elif (
        target_type
        ==
        CastTargetType.FLOAT.value
    ):
        converted = (
            pd.to_numeric(
                source,
                errors=
                    "coerce",
            )
        )

        invalid_mask = (
            source.notna()
            &
            converted.isna()
        )

        if (
            bool(
                invalid_mask.any()
            )
        ):
            raise (
                TransformationExecutionError(
                    (
                        "FLOAT cast would create "
                        "missing values from "
                        "non-missing source values."
                    )
                )
            )

        result = (
            converted.astype(
                "Float64"
            )
        )

    # ========================================================
    # INTEGER
    # ========================================================

    elif (
        target_type
        ==
        CastTargetType.INTEGER.value
    ):
        converted = (
            pd.to_numeric(
                source,
                errors=
                    "coerce",
            )
        )

        invalid_mask = (
            source.notna()
            &
            converted.isna()
        )

        if (
            bool(
                invalid_mask.any()
            )
        ):
            raise (
                TransformationExecutionError(
                    (
                        "INTEGER cast would create "
                        "missing values from "
                        "non-missing source values."
                    )
                )
            )

        fractional_mask = (
            converted
            .dropna()
            .map(
                lambda value:
                    not float(
                        value
                    ).is_integer()
            )
        )

        if (
            bool(
                fractional_mask.any()
            )
        ):
            raise (
                TransformationExecutionError(
                    (
                        "INTEGER cast would lose "
                        "fractional information."
                    )
                )
            )

        result = (
            converted.astype(
                "Int64"
            )
        )

    # ========================================================
    # BOOLEAN
    # ========================================================

    elif (
        target_type
        ==
        CastTargetType.BOOLEAN.value
    ):
        result = (
            _strict_boolean_cast(
                source
            )
        )

    # ========================================================
    # DATETIME
    # ========================================================

    elif (
        target_type
        ==
        CastTargetType.DATETIME.value
    ):
        converted = (
            pd.to_datetime(
                source,
                errors=
                    "coerce",
            )
        )

        invalid_mask = (
            source.notna()
            &
            converted.isna()
        )

        if (
            bool(
                invalid_mask.any()
            )
        ):
            invalid_count = int(
                invalid_mask.sum()
            )

            raise (
                TransformationExecutionError(
                    (
                        "DATETIME cast encountered "
                        "invalid non-missing values. "
                        f"Count: {invalid_count}"
                    )
                )
            )

        result = converted

    else:
        raise (
            TransformationExecutionError(
                (
                    "Unsupported cast target: "
                    f"{target_type}"
                )
            )
        )

    dataframe[
        output_column
    ] = result

    return int(
        len(
            dataframe
        )
    )


# ============================================================
# BIN NUMERIC
# ============================================================


def _execute_bin_numeric(
    *,
    dataframe: pd.DataFrame,
    step: ApprovedTransformationStep,
) -> int:
    source_column = (
        step.parameters.get(
            "source_column"
        )
    )

    output_column = (
        step.output_column
    )

    if not (
        isinstance(
            source_column,
            str,
        )
        and
        source_column
    ):
        raise (
            TransformationExecutionError(
                (
                    "BIN_NUMERIC requires "
                    "source_column."
                )
            )
        )

    if not (
        output_column
    ):
        raise (
            TransformationExecutionError(
                (
                    "BIN_NUMERIC requires "
                    "output_column."
                )
            )
        )

    _require_column(
        dataframe=
            dataframe,

        column=
            source_column,
    )

    _require_new_column(
        dataframe=
            dataframe,

        column=
            output_column,
    )

    source = (
        dataframe[
            source_column
        ]
    )

    _require_numeric_series(
        series=
            source,

        column=
            source_column,
    )

    bins = list(
        step.parameters.get(
            "bins"
        )
        or
        []
    )

    labels = (
        step.parameters.get(
            "labels"
        )
    )

    include_lowest = bool(
        step.parameters.get(
            "include_lowest",
            True,
        )
    )

    right = bool(
        step.parameters.get(
            "right",
            True,
        )
    )

    if (
        len(
            bins
        )
        <
        2
    ):
        raise (
            TransformationExecutionError(
                (
                    "BIN_NUMERIC execution "
                    "requires at least two "
                    "boundaries."
                )
            )
        )

    result = (
        pd.cut(
            source,
            bins=
                bins,
            labels=
                labels,
            include_lowest=
                include_lowest,
            right=
                right,
        )
    )

    # --------------------------------------------------------
    # Do not silently introduce missing categories for
    # non-missing source values outside the approved bins.
    # --------------------------------------------------------

    introduced_missing = (
        source.notna()
        &
        result.isna()
    )

    if (
        bool(
            introduced_missing.any()
        )
    ):
        count = int(
            introduced_missing.sum()
        )

        examples = (
            source
            .loc[
                introduced_missing
            ]
            .drop_duplicates()
            .head(
                8
            )
            .tolist()
        )

        raise (
            TransformationExecutionError(
                (
                    "BIN_NUMERIC approved boundaries "
                    "do not cover all non-missing "
                    "source values. "
                    f"Count: {count}. "
                    f"Examples: {examples}"
                )
            )
        )

    dataframe[
        output_column
    ] = result

    return int(
        source
        .notna()
        .sum()
    )


# ============================================================
# DATE PART
# ============================================================


def _execute_extract_date_part(
    *,
    dataframe: pd.DataFrame,
    step: ApprovedTransformationStep,
) -> int:
    source_column = (
        step.parameters.get(
            "source_column"
        )
    )

    output_column = (
        step.output_column
    )

    if not (
        isinstance(
            source_column,
            str,
        )
        and
        source_column
    ):
        raise (
            TransformationExecutionError(
                (
                    "EXTRACT_DATE_PART requires "
                    "source_column."
                )
            )
        )

    if not (
        output_column
    ):
        raise (
            TransformationExecutionError(
                (
                    "EXTRACT_DATE_PART requires "
                    "output_column."
                )
            )
        )

    _require_column(
        dataframe=
            dataframe,

        column=
            source_column,
    )

    _require_new_column(
        dataframe=
            dataframe,

        column=
            output_column,
    )

    source = (
        dataframe[
            source_column
        ]
    )

    parsed = (
        pd.to_datetime(
            source,
            errors=
                "coerce",
        )
    )

    invalid_mask = (
        source.notna()
        &
        parsed.isna()
    )

    if (
        bool(
            invalid_mask.any()
        )
    ):
        invalid_count = int(
            invalid_mask.sum()
        )

        raise (
            TransformationExecutionError(
                (
                    "Date-part extraction would "
                    "silently lose invalid date "
                    "values. "
                    f"Count: {invalid_count}"
                )
            )
        )

    part = (
        _enum_value(
            step.parameters.get(
                "part"
            )
        )
    )

    if (
        part
        ==
        DatePart.YEAR.value
    ):
        result = (
            parsed.dt.year
            .astype(
                "Int64"
            )
        )

    elif (
        part
        ==
        DatePart.MONTH.value
    ):
        result = (
            parsed.dt.month
            .astype(
                "Int64"
            )
        )

    elif (
        part
        ==
        DatePart.DAY.value
    ):
        result = (
            parsed.dt.day
            .astype(
                "Int64"
            )
        )

    elif (
        part
        ==
        DatePart.QUARTER.value
    ):
        result = (
            parsed.dt.quarter
            .astype(
                "Int64"
            )
        )

    elif (
        part
        ==
        DatePart.WEEK.value
    ):
        result = (
            parsed
            .dt
            .isocalendar()
            .week
            .astype(
                "Int64"
            )
        )

    elif (
        part
        ==
        DatePart.WEEKDAY.value
    ):
        result = (
            parsed
            .dt
            .weekday
            .astype(
                "Int64"
            )
        )

    else:
        raise (
            TransformationExecutionError(
                (
                    "Unsupported date part: "
                    f"{part}"
                )
            )
        )

    dataframe[
        output_column
    ] = result

    return int(
        source
        .notna()
        .sum()
    )


# ============================================================
# AGGREGATION
# ============================================================


def _execute_aggregate(
    *,
    dataframe: pd.DataFrame,
    step: ApprovedTransformationStep,
    derived_datasets: Dict[
        str,
        pd.DataFrame,
    ],
) -> tuple[
    pd.DataFrame,
    int,
]:
    output_dataset_id = (
        step.output_dataset_id
    )

    if not (
        output_dataset_id
    ):
        raise (
            TransformationExecutionError(
                (
                    "AGGREGATE requires "
                    "output_dataset_id."
                )
            )
        )

    if (
        output_dataset_id
        in
        derived_datasets
    ):
        raise (
            TransformationExecutionError(
                (
                    "Derived dataset already "
                    "exists during execution: "
                    f"{output_dataset_id}"
                )
            )
        )

    group_by = list(
        step.parameters.get(
            "group_by"
        )
        or
        []
    )

    metrics = list(
        step.parameters.get(
            "metrics"
        )
        or
        []
    )

    if not (
        group_by
    ):
        raise (
            TransformationExecutionError(
                (
                    "AGGREGATE requires at least "
                    "one group_by column."
                )
            )
        )

    if not (
        metrics
    ):
        raise (
            TransformationExecutionError(
                (
                    "AGGREGATE requires at least "
                    "one metric."
                )
            )
        )

    for column in (
        group_by
    ):
        _require_column(
            dataframe=
                dataframe,

            column=
                column,
        )

    named_aggregations: Dict[
        str,
        pd.NamedAgg,
    ] = {}

    for metric in metrics:
        source_column = (
            metric.get(
                "source_column"
            )
        )

        function = (
            _enum_value(
                metric.get(
                    "function"
                )
            )
        )

        output_column = (
            metric.get(
                "output_column"
            )
        )

        if not (
            isinstance(
                source_column,
                str,
            )
            and
            source_column
        ):
            raise (
                TransformationExecutionError(
                    (
                        "Aggregation metric requires "
                        "source_column."
                    )
                )
            )

        if not (
            isinstance(
                output_column,
                str,
            )
            and
            output_column
        ):
            raise (
                TransformationExecutionError(
                    (
                        "Aggregation metric requires "
                        "output_column."
                    )
                )
            )

        _require_column(
            dataframe=
                dataframe,

            column=
                source_column,
        )

        if (
            output_column
            in
            named_aggregations
        ):
            raise (
                TransformationExecutionError(
                    (
                        "Duplicate aggregation "
                        "output column: "
                        f"{output_column}"
                    )
                )
            )

        if (
            function
            not in {
                AggregationFunction.SUM.value,
                AggregationFunction.MEAN.value,
                AggregationFunction.MEDIAN.value,
                AggregationFunction.MIN.value,
                AggregationFunction.MAX.value,
                AggregationFunction.COUNT.value,
                AggregationFunction.NUNIQUE.value,
            }
        ):
            raise (
                TransformationExecutionError(
                    (
                        "Unsupported aggregation "
                        f"function: {function}"
                    )
                )
            )

        named_aggregations[
            output_column
        ] = (
            pd.NamedAgg(
                column=
                    source_column,
                aggfunc=
                    function,
            )
        )

    try:
        output = (
            dataframe
            .groupby(
                group_by,
                dropna=
                    False,
                as_index=
                    False,
            )
            .agg(
                **named_aggregations
            )
        )

    except Exception as exc:
        raise (
            TransformationExecutionError(
                (
                    "Aggregation execution failed: "
                    f"{exc}"
                )
            )
        ) from exc

    derived_datasets[
        output_dataset_id
    ] = (
        output.copy(
            deep=True
        )
    )

    return (
        output,
        int(
            len(
                output
            )
        ),
    )


# ============================================================
# STEP EXECUTION
# ============================================================


def _execute_step(
    *,
    dataframe: pd.DataFrame,
    step: ApprovedTransformationStep,
    derived_datasets: Dict[
        str,
        pd.DataFrame,
    ],
) -> tuple[
    int,
    Optional[
        str
    ],
]:
    operation = (
        step.operation
    )

    if (
        operation
        ==
        TransformationOperation
        .DERIVE_ARITHMETIC
    ):
        affected_rows = (
            _execute_derive_arithmetic(
                dataframe=
                    dataframe,

                step=
                    step,
            )
        )

        return (
            affected_rows,
            None,
        )

    if (
        operation
        ==
        TransformationOperation.CAST
    ):
        affected_rows = (
            _execute_cast(
                dataframe=
                    dataframe,

                step=
                    step,
            )
        )

        return (
            affected_rows,
            None,
        )

    if (
        operation
        ==
        TransformationOperation
        .BIN_NUMERIC
    ):
        affected_rows = (
            _execute_bin_numeric(
                dataframe=
                    dataframe,

                step=
                    step,
            )
        )

        return (
            affected_rows,
            None,
        )

    if (
        operation
        ==
        TransformationOperation
        .EXTRACT_DATE_PART
    ):
        affected_rows = (
            _execute_extract_date_part(
                dataframe=
                    dataframe,

                step=
                    step,
            )
        )

        return (
            affected_rows,
            None,
        )

    if (
        operation
        ==
        TransformationOperation
        .AGGREGATE
    ):
        (
            output,
            affected_rows,
        ) = (
            _execute_aggregate(
                dataframe=
                    dataframe,

                step=
                    step,

                derived_datasets=
                    derived_datasets,
            )
        )

        return (
            affected_rows,
            _dataframe_fingerprint(
                output
            ),
        )

    raise (
        TransformationExecutionError(
            (
                "Unsupported transformation "
                f"operation: {operation}"
            )
        )
    )


# ============================================================
# PUBLIC API
# ============================================================


def execute_transformation_plan(
    *,
    dataframe: pd.DataFrame,
    approved_plan: ApprovedTransformationPlan,
    dataset_id: str,
    dataset_filename: str,
) -> TransformationExecutionResult:
    """
    Execute an ApprovedTransformationPlan transactionally.

    Safety guarantees:

    - only ApprovedTransformationPlan is accepted;
    - plan must be ready_for_execution=True;
    - source dataframe is NEVER mutated;
    - execution works on a deep copy;
    - only executable=True steps are applied;
    - PENDING / DEFERRED / BLOCKED plans are rejected through
      the ready_for_execution invariant;
    - REJECTED transformations are skipped;
    - arbitrary eval() is never used;
    - approved output columns cannot overwrite existing columns;
    - division by zero is checked at execution time;
    - date conversion failures are not silently coerced;
    - binning may not silently introduce missing categories;
    - aggregation produces a separate derived dataset;
    - if any executable step fails, no partial result is returned.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            (
                "Transformation Executor requires "
                "a pandas DataFrame."
            )
        )

    if (
        dataframe.empty
    ):
        raise ValueError(
            (
                "Transformation Executor requires "
                "at least one source row."
            )
        )

    if (
        approved_plan.dataset_id
        !=
        dataset_id
    ):
        raise ValueError(
            (
                "ApprovedTransformationPlan "
                "dataset_id does not match "
                "execution dataset."
            )
        )

    if (
        approved_plan.dataset_filename
        !=
        dataset_filename
    ):
        raise ValueError(
            (
                "ApprovedTransformationPlan "
                "dataset filename does not "
                "match execution dataset."
            )
        )

    if not (
        approved_plan
        .ready_for_execution
    ):
        raise (
            TransformationExecutionError(
                (
                    "ApprovedTransformationPlan "
                    "is not ready for execution."
                )
            )
        )

    executable_steps = [
        step

        for step
        in approved_plan.steps

        if step.executable
    ]

    if (
        len(
            executable_steps
        )
        !=
        approved_plan
        .executable_step_count
    ):
        raise (
            TransformationExecutionError(
                (
                    "ApprovedTransformationPlan "
                    "contains an inconsistent "
                    "executable_step_count."
                )
            )
        )

    # --------------------------------------------------------
    # executable=True must be explicitly authorized.
    # --------------------------------------------------------

    allowed_execution_statuses = {
        TransformationAuthorizationStatus
        .AUTOMATIC,

        TransformationAuthorizationStatus
        .APPROVED,
    }

    for step in executable_steps:
        if (
            step.authorization_status
            not in
            allowed_execution_statuses
        ):
            raise (
                TransformationExecutionError(
                    (
                        "Executable transformation "
                        "has invalid authorization "
                        "status: "
                        f"{step.request_id} / "
                        f"{step.authorization_status}"
                    )
                )
            )

    # ========================================================
    # TRANSACTION START
    # ========================================================

    source_fingerprint_before = (
        _dataframe_fingerprint(
            dataframe
        )
    )

    source_rows_before = int(
        dataframe.shape[
            0
        ]
    )

    source_columns_before = int(
        dataframe.shape[
            1
        ]
    )

    working = (
        dataframe.copy(
            deep=True
        )
    )

    derived_datasets: Dict[
        str,
        pd.DataFrame,
    ] = {}

    step_reports: List[
        TransformationStepExecution
    ] = []

    applied_count = 0

    skipped_count = 0

    # ========================================================
    # EXECUTE IN APPROVED ORDER
    # ========================================================

    for order, step in enumerate(
        approved_plan.steps,
        start=
            1,
    ):
        fingerprint_before = (
            _dataframe_fingerprint(
                working
            )
        )

        rows_before = int(
            working.shape[
                0
            ]
        )

        columns_before = int(
            working.shape[
                1
            ]
        )

        # ====================================================
        # NON-EXECUTABLE BUT RESOLVED
        #
        # Example: user REJECT.
        # ====================================================

        if not (
            step.executable
        ):
            if not (
                step.resolved
            ):
                raise (
                    TransformationExecutionError(
                        (
                            "Unresolved transformation "
                            "reached executor: "
                            f"{step.request_id}"
                        )
                    )
                )

            skipped_count += 1

            step_reports.append(
                TransformationStepExecution(
                    order=
                        order,

                    request_id=
                        step.request_id,

                    operation=
                        step.operation,

                    authorization_status=
                        step.authorization_status,

                    status=
                        TransformationExecutionStatus
                        .SKIPPED,

                    input_columns=
                        list(
                            step.input_columns
                        ),

                    output_column=
                        step.output_column,

                    output_dataset_id=
                        step.output_dataset_id,

                    output_dataset_filename=
                        step.output_dataset_filename,

                    source_rows_before=
                        rows_before,

                    source_rows_after=
                        rows_before,

                    source_columns_before=
                        columns_before,

                    source_columns_after=
                        columns_before,

                    affected_rows=
                        0,

                    affected_cells=
                        0,

                    source_fingerprint_before=
                        fingerprint_before,

                    source_fingerprint_after=
                        fingerprint_before,

                    output_fingerprint=None,

                    rationale=(
                        "Transformation was resolved "
                        "as non-executable and was "
                        "therefore skipped."
                    ),
                )
            )

            continue

        # ====================================================
        # EXECUTION
        # ====================================================

        try:
            (
                affected_rows,
                output_fingerprint,
            ) = (
                _execute_step(
                    dataframe=
                        working,

                    step=
                        step,

                    derived_datasets=
                        derived_datasets,
                )
            )

        except TransformationExecutionError:
            raise

        except Exception as exc:
            raise (
                TransformationExecutionError(
                    (
                        "Unexpected transformation "
                        "execution failure for "
                        f"{step.request_id}: {exc}"
                    )
                )
            ) from exc

        fingerprint_after = (
            _dataframe_fingerprint(
                working
            )
        )

        rows_after = int(
            working.shape[
                0
            ]
        )

        columns_after = int(
            working.shape[
                1
            ]
        )

        applied_count += 1

        affected_cells = (
            affected_rows
            if (
                step.output_column
                is not None
            )
            else
            0
        )

        step_reports.append(
            TransformationStepExecution(
                order=
                    order,

                request_id=
                    step.request_id,

                operation=
                    step.operation,

                authorization_status=
                    step.authorization_status,

                status=
                    TransformationExecutionStatus
                    .APPLIED,

                input_columns=
                    list(
                        step.input_columns
                    ),

                output_column=
                    step.output_column,

                output_dataset_id=
                    step.output_dataset_id,

                output_dataset_filename=
                    step.output_dataset_filename,

                source_rows_before=
                    rows_before,

                source_rows_after=
                    rows_after,

                source_columns_before=
                    columns_before,

                source_columns_after=
                    columns_after,

                affected_rows=
                    affected_rows,

                affected_cells=
                    affected_cells,

                source_fingerprint_before=
                    fingerprint_before,

                source_fingerprint_after=
                    fingerprint_after,

                output_fingerprint=
                    output_fingerprint,

                rationale=(
                    "Approved structured "
                    "transformation executed "
                    "successfully."
                ),
            )
        )

    # ========================================================
    # FINAL SOURCE STATE
    # ========================================================

    source_fingerprint_after = (
        _dataframe_fingerprint(
            working
        )
    )

    source_rows_after = int(
        working.shape[
            0
        ]
    )

    source_columns_after = int(
        working.shape[
            1
        ]
    )

    # --------------------------------------------------------
    # Core invariant:
    #
    # aggregation must not replace or mutate the source grain.
    # --------------------------------------------------------

    if (
        source_rows_after
        !=
        source_rows_before
    ):
        raise (
            TransformationExecutionError(
                (
                    "Transformation Executor v0.1 "
                    "detected an unexpected source "
                    "row-count change."
                )
            )
        )

    # --------------------------------------------------------
    # Verify original source itself remained untouched.
    # --------------------------------------------------------

    if (
        _dataframe_fingerprint(
            dataframe
        )
        !=
        source_fingerprint_before
    ):
        raise (
            TransformationExecutionError(
                (
                    "Source DataFrame was unexpectedly "
                    "mutated during transformation."
                )
            )
        )

    # ========================================================
    # REPORT
    # ========================================================

    report = (
        TransformationExecutionReport(
            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            source_rows_before=
                source_rows_before,

            source_rows_after=
                source_rows_after,

            source_columns_before=
                source_columns_before,

            source_columns_after=
                source_columns_after,

            source_fingerprint_before=
                source_fingerprint_before,

            source_fingerprint_after=
                source_fingerprint_after,

            source_data_changed=(
                source_fingerprint_before
                !=
                source_fingerprint_after
            ),

            total_step_count=
                len(
                    approved_plan.steps
                ),

            executable_step_count=
                approved_plan
                .executable_step_count,

            applied_count=
                applied_count,

            skipped_count=
                skipped_count,

            derived_dataset_count=
                len(
                    derived_datasets
                ),

            derived_dataset_ids=
                list(
                    derived_datasets.keys()
                ),

            steps=
                step_reports,

            notes=[
                (
                    "Transformation Executor v0.1 "
                    "works on a deep copy and never "
                    "mutates the source DataFrame."
                ),

                (
                    "Only AUTOMATIC or APPROVED "
                    "executable steps are executed."
                ),

                (
                    "Rejected but resolved steps "
                    "are skipped."
                ),

                (
                    "No arbitrary Python expression "
                    "or eval() is executed."
                ),

                (
                    "Derived columns are added to "
                    "the transformed source copy."
                ),

                (
                    "Aggregations create separate "
                    "derived datasets and do not "
                    "replace source grain."
                ),

                (
                    "Execution is transactional: "
                    "a failing step returns no "
                    "partial transformed result."
                ),
            ],
        )
    )

    return (
        TransformationExecutionResult(
            dataframe=
                working,

            derived_datasets={
                key:
                    value.copy(
                        deep=True
                    )

                for (
                    key,
                    value
                )
                in derived_datasets.items()
            },

            report=
                report,
        )
    )