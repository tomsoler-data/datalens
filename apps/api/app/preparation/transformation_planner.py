from __future__ import annotations

import math

from numbers import Number

from typing import (
    Dict,
    List,
    Set,
)

import pandas as pd

from app.preparation.transformation_contracts import (
    AggregateIntent,
    AggregationFunction,
    ArithmeticOperator,
    BinNumericIntent,
    CastIntent,
    CastTargetType,
    DatePart,
    DeriveArithmeticIntent,
    ExtractDatePartIntent,
    OperandKind,
    TransformationIntent,
    TransformationOperation,
    TransformationPlan,
    TransformationRisk,
    TransformationStatus,
    TransformationStep,
)


# ============================================================
# VERSION
# ============================================================


TRANSFORMATION_PLANNER_RULE_VERSION = (
    "transformation_planner_v0.1"
)


# ============================================================
# INTERNAL TYPE FAMILIES
# ============================================================


TYPE_NUMERIC = (
    "numeric"
)

TYPE_STRING = (
    "string"
)

TYPE_BOOLEAN = (
    "boolean"
)

TYPE_DATETIME = (
    "datetime"
)

TYPE_CATEGORICAL = (
    "categorical"
)

TYPE_UNKNOWN = (
    "unknown"
)


# ============================================================
# DATAFRAME TYPE INFERENCE
# ============================================================


def _infer_series_family(
    series: pd.Series,
) -> str:
    if (
        pd.api.types
        .is_bool_dtype(
            series
        )
    ):
        return TYPE_BOOLEAN

    if (
        pd.api.types
        .is_numeric_dtype(
            series
        )
    ):
        return TYPE_NUMERIC

    if (
        pd.api.types
        .is_datetime64_any_dtype(
            series
        )
    ):
        return TYPE_DATETIME

    if isinstance(
        series.dtype,
        pd.CategoricalDtype,
    ):
        return TYPE_CATEGORICAL

    if (
        pd.api.types
        .is_object_dtype(
            series
        )
        or
        pd.api.types
        .is_string_dtype(
            series
        )
    ):
        return TYPE_STRING

    return TYPE_UNKNOWN


def _build_virtual_schema(
    dataframe: pd.DataFrame,
) -> Dict[
    str,
    str,
]:
    """
    Build an in-memory schema used only during planning.

    Derived columns are added to this schema without being
    added to the actual DataFrame.
    """

    return {
        str(
            column
        ):
            _infer_series_family(
                dataframe[
                    column
                ]
            )

        for column
        in dataframe.columns
    }


# ============================================================
# GENERIC VALIDATION
# ============================================================


def _require_dataset_match(
    *,
    intent: TransformationIntent,
    dataset_id: str,
    dataset_filename: str,
) -> None:
    if (
        intent.dataset_id
        !=
        dataset_id
    ):
        raise ValueError(
            (
                "Transformation intent references "
                "unexpected dataset_id: "
                f"{intent.dataset_id}"
            )
        )

    if (
        intent.dataset_filename
        !=
        dataset_filename
    ):
        raise ValueError(
            (
                "Transformation intent references "
                "unexpected dataset filename: "
                f"{intent.dataset_filename}"
            )
        )


def _require_operation(
    *,
    actual: TransformationOperation,
    expected: TransformationOperation,
) -> None:
    """
    Prevent a malformed structured object from declaring
    an operation inconsistent with its contract type.
    """

    if (
        actual
        !=
        expected
    ):
        raise ValueError(
            (
                "Transformation operation does not "
                "match intent type. "
                f"Expected {expected.value}, "
                f"received {actual.value}."
            )
        )


def _require_column(
    *,
    column: str,
    schema: Dict[
        str,
        str,
    ],
) -> None:
    if (
        column
        not in schema
    ):
        raise ValueError(
            (
                "Unknown transformation column: "
                f"{column}"
            )
        )


def _require_new_output_column(
    *,
    column: str,
    schema: Dict[
        str,
        str,
    ],
) -> None:
    if not (
        column.strip()
    ):
        raise ValueError(
            (
                "Output column cannot "
                "be empty."
            )
        )

    if (
        column
        in schema
    ):
        raise ValueError(
            (
                "Transformation Planner v0.1 "
                "does not allow overwriting "
                "an existing column: "
                f"{column}"
            )
        )


def _require_numeric_column(
    *,
    column: str,
    schema: Dict[
        str,
        str,
    ],
) -> None:
    _require_column(
        column=
            column,

        schema=
            schema,
    )

    observed_family = (
        schema[
            column
        ]
    )

    if (
        observed_family
        !=
        TYPE_NUMERIC
    ):
        raise ValueError(
            (
                f"Column '{column}' must be numeric "
                "for this transformation. "
                f"Observed family: "
                f"{observed_family}"
            )
        )


# ============================================================
# DATE-LIKE VALIDATION
# ============================================================


def _date_parse_ratio(
    series: pd.Series,
) -> float:
    non_missing = (
        series
        .dropna()
    )

    if (
        non_missing.empty
    ):
        return 0.0

    parsed = (
        pd.to_datetime(
            non_missing,
            errors=
                "coerce",
        )
    )

    valid_count = int(
        parsed
        .notna()
        .sum()
    )

    return float(
        valid_count
        /
        len(
            non_missing
        )
    )


def _is_date_like(
    *,
    column: str,
    dataframe: pd.DataFrame,
    schema: Dict[
        str,
        str,
    ],
) -> bool:
    family = (
        schema[
            column
        ]
    )

    if (
        family
        ==
        TYPE_DATETIME
    ):
        return True

    # A previously derived virtual column may exist in
    # schema without existing yet in the real dataframe.
    if (
        column
        not in
        dataframe.columns
    ):
        return (
            family
            ==
            TYPE_DATETIME
        )

    if (
        family
        !=
        TYPE_STRING
    ):
        return False

    return (
        _date_parse_ratio(
            dataframe[
                column
            ]
        )
        >=
        0.80
    )


# ============================================================
# OPERAND VALIDATION
# ============================================================


def _validate_operand(
    *,
    operand,
    schema: Dict[
        str,
        str,
    ],
) -> List[
    str
]:
    if (
        operand.kind
        ==
        OperandKind.COLUMN
    ):
        if not (
            operand.column
        ):
            raise ValueError(
                (
                    "COLUMN operand requires "
                    "a column."
                )
            )

        if (
            operand.value
            is not None
        ):
            raise ValueError(
                (
                    "COLUMN operand cannot also "
                    "contain a literal value."
                )
            )

        _require_numeric_column(
            column=
                operand.column,

            schema=
                schema,
        )

        return [
            operand.column
        ]

    if (
        operand.kind
        ==
        OperandKind.LITERAL
    ):
        if (
            operand.column
            is not None
        ):
            raise ValueError(
                (
                    "LITERAL operand cannot "
                    "contain a column."
                )
            )

        if (
            not isinstance(
                operand.value,
                Number,
            )
            or
            isinstance(
                operand.value,
                bool,
            )
        ):
            raise ValueError(
                (
                    "Arithmetic literal operand "
                    "must be numeric."
                )
            )

        if (
            isinstance(
                operand.value,
                float,
            )
            and
            not math.isfinite(
                operand.value
            )
        ):
            raise ValueError(
                (
                    "Arithmetic literal must "
                    "be finite."
                )
            )

        return []

    raise ValueError(
        (
            "Unsupported operand kind: "
            f"{operand.kind}"
        )
    )


# ============================================================
# DERIVE ARITHMETIC
# ============================================================


def _plan_derive_arithmetic(
    *,
    intent: DeriveArithmeticIntent,
    schema: Dict[
        str,
        str,
    ],
) -> TransformationStep:
    _require_operation(
        actual=
            intent.operation,

        expected=
            TransformationOperation
            .DERIVE_ARITHMETIC,
    )

    _require_new_output_column(
        column=
            intent.output_column,

        schema=
            schema,
    )

    left_columns = (
        _validate_operand(
            operand=
                intent.left,

            schema=
                schema,
        )
    )

    right_columns = (
        _validate_operand(
            operand=
                intent.right,

            schema=
                schema,
        )
    )

    # --------------------------------------------------------
    # Deterministic literal division-by-zero rejection.
    # --------------------------------------------------------

    if (
        intent.operator
        ==
        ArithmeticOperator.DIVIDE
        and
        intent.right.kind
        ==
        OperandKind.LITERAL
        and
        intent.right.value
        ==
        0
    ):
        raise ValueError(
            (
                "Division by zero literal "
                "is not allowed."
            )
        )

    input_columns = list(
        dict.fromkeys(
            left_columns
            +
            right_columns
        )
    )

    # --------------------------------------------------------
    # Dividing by another column is structurally valid,
    # but zero values may exist at execution time.
    # --------------------------------------------------------

    division_by_column = (
        intent.operator
        ==
        ArithmeticOperator.DIVIDE
        and
        intent.right.kind
        ==
        OperandKind.COLUMN
    )

    if (
        division_by_column
    ):
        status = (
            TransformationStatus
            .REVIEW_REQUIRED
        )

        risk = (
            TransformationRisk
            .MEDIUM
        )

        requires_human_approval = (
            True
        )

        rationale = (
            "Arithmetic derivation is structurally "
            "valid, but division by a column can "
            "encounter zero values. Explicit review "
            "is required before execution."
        )

    else:
        status = (
            TransformationStatus
            .VALIDATED
        )

        risk = (
            TransformationRisk
            .LOW
        )

        requires_human_approval = (
            False
        )

        rationale = (
            "Arithmetic derivation uses existing "
            "numeric columns and/or finite numeric "
            "literals. No free-form expression "
            "will be evaluated."
        )

    return (
        TransformationStep(
            step_id=(
                "transform:"
                +
                intent.request_id
            ),

            request_id=
                intent.request_id,

            dataset_id=
                intent.dataset_id,

            dataset_filename=
                intent.dataset_filename,

            operation=
                intent.operation,

            status=
                status,

            risk=
                risk,

            input_columns=
                input_columns,

            output_column=
                intent.output_column,

            output_dataset_id=None,

            output_dataset_filename=None,

            parameters={
                "left":
                    intent.left
                    .model_dump(),

                "operator":
                    intent.operator
                    .value,

                "right":
                    intent.right
                    .model_dump(),
            },

            rationale=
                rationale,

            requires_human_approval=
                requires_human_approval,

            executable=
                False,
        )
    )


# ============================================================
# CAST
# ============================================================


def _plan_cast(
    *,
    intent: CastIntent,
    schema: Dict[
        str,
        str,
    ],
) -> TransformationStep:
    _require_operation(
        actual=
            intent.operation,

        expected=
            TransformationOperation.CAST,
    )

    _require_column(
        column=
            intent.source_column,

        schema=
            schema,
    )

    _require_new_output_column(
        column=
            intent.output_column,

        schema=
            schema,
    )

    source_family = (
        schema[
            intent.source_column
        ]
    )

    target_family_map = {
        CastTargetType.STRING:
            TYPE_STRING,

        CastTargetType.INTEGER:
            TYPE_NUMERIC,

        CastTargetType.FLOAT:
            TYPE_NUMERIC,

        CastTargetType.BOOLEAN:
            TYPE_BOOLEAN,

        CastTargetType.DATETIME:
            TYPE_DATETIME,
    }

    target_family = (
        target_family_map[
            intent.target_type
        ]
    )

    # --------------------------------------------------------
    # These conversions can lose information or generate
    # missing values.
    # --------------------------------------------------------

    potentially_lossy = (
        intent.target_type
        in {
            CastTargetType.INTEGER,
            CastTargetType.BOOLEAN,
            CastTargetType.DATETIME,
        }
        or
        (
            source_family
            ==
            TYPE_STRING
            and
            target_family
            ==
            TYPE_NUMERIC
        )
    )

    if (
        potentially_lossy
    ):
        status = (
            TransformationStatus
            .REVIEW_REQUIRED
        )

        risk = (
            TransformationRisk
            .MEDIUM
        )

        requires_human_approval = (
            True
        )

        rationale = (
            "The requested cast can be lossy "
            "or may create invalid values. "
            "Human approval is required before "
            "execution."
        )

    else:
        status = (
            TransformationStatus
            .VALIDATED
        )

        risk = (
            TransformationRisk
            .LOW
        )

        requires_human_approval = (
            False
        )

        rationale = (
            "The requested cast is structurally "
            "valid and considered low-risk."
        )

    return (
        TransformationStep(
            step_id=(
                "transform:"
                +
                intent.request_id
            ),

            request_id=
                intent.request_id,

            dataset_id=
                intent.dataset_id,

            dataset_filename=
                intent.dataset_filename,

            operation=
                intent.operation,

            status=
                status,

            risk=
                risk,

            input_columns=[
                intent.source_column
            ],

            output_column=
                intent.output_column,

            output_dataset_id=None,

            output_dataset_filename=None,

            parameters={
                "source_column":
                    intent.source_column,

                "target_type":
                    intent.target_type
                    .value,
            },

            rationale=
                rationale,

            requires_human_approval=
                requires_human_approval,

            executable=
                False,
        )
    )


# ============================================================
# BIN NUMERIC
# ============================================================


def _plan_bin_numeric(
    *,
    intent: BinNumericIntent,
    schema: Dict[
        str,
        str,
    ],
) -> TransformationStep:
    _require_operation(
        actual=
            intent.operation,

        expected=
            TransformationOperation
            .BIN_NUMERIC,
    )

    _require_numeric_column(
        column=
            intent.source_column,

        schema=
            schema,
    )

    _require_new_output_column(
        column=
            intent.output_column,

        schema=
            schema,
    )

    if (
        len(
            intent.bins
        )
        <
        2
    ):
        raise ValueError(
            (
                "BIN_NUMERIC requires at least "
                "two boundaries."
            )
        )

    for value in (
        intent.bins
    ):
        if not (
            math.isfinite(
                float(
                    value
                )
            )
        ):
            raise ValueError(
                (
                    "Bin boundaries must "
                    "be finite."
                )
            )

    # --------------------------------------------------------
    # Strict monotonicity.
    # --------------------------------------------------------

    for index in range(
        1,
        len(
            intent.bins
        ),
    ):
        if (
            intent.bins[
                index
            ]
            <=
            intent.bins[
                index
                -
                1
            ]
        ):
            raise ValueError(
                (
                    "Bin boundaries must be "
                    "strictly increasing."
                )
            )

    expected_label_count = (
        len(
            intent.bins
        )
        -
        1
    )

    if (
        intent.labels
        is not None
        and
        len(
            intent.labels
        )
        !=
        expected_label_count
    ):
        raise ValueError(
            (
                "Number of bin labels must "
                "equal len(bins) - 1."
            )
        )

    return (
        TransformationStep(
            step_id=(
                "transform:"
                +
                intent.request_id
            ),

            request_id=
                intent.request_id,

            dataset_id=
                intent.dataset_id,

            dataset_filename=
                intent.dataset_filename,

            operation=
                intent.operation,

            status=
                TransformationStatus
                .REVIEW_REQUIRED,

            risk=
                TransformationRisk
                .MEDIUM,

            input_columns=[
                intent.source_column
            ],

            output_column=
                intent.output_column,

            output_dataset_id=None,

            output_dataset_filename=None,

            parameters={
                "source_column":
                    intent.source_column,

                "bins":
                    list(
                        intent.bins
                    ),

                "labels": (
                    list(
                        intent.labels
                    )
                    if (
                        intent.labels
                        is not None
                    )
                    else
                    None
                ),

                "include_lowest":
                    intent.include_lowest,

                "right":
                    intent.right,
            },

            rationale=(
                "Numeric binning changes the "
                "analytical representation of a "
                "continuous variable. The boundaries "
                "are deterministic but should be "
                "explicitly approved."
            ),

            requires_human_approval=
                True,

            executable=
                False,
        )
    )


# ============================================================
# DATE PART EXTRACTION
# ============================================================


def _plan_extract_date_part(
    *,
    intent: ExtractDatePartIntent,
    dataframe: pd.DataFrame,
    schema: Dict[
        str,
        str,
    ],
) -> TransformationStep:
    _require_operation(
        actual=
            intent.operation,

        expected=
            TransformationOperation
            .EXTRACT_DATE_PART,
    )

    _require_column(
        column=
            intent.source_column,

        schema=
            schema,
    )

    _require_new_output_column(
        column=
            intent.output_column,

        schema=
            schema,
    )

    if not (
        _is_date_like(
            column=
                intent.source_column,

            dataframe=
                dataframe,

            schema=
                schema,
        )
    ):
        raise ValueError(
            (
                "EXTRACT_DATE_PART requires "
                "a datetime or strongly date-like "
                "source column: "
                f"{intent.source_column}"
            )
        )

    return (
        TransformationStep(
            step_id=(
                "transform:"
                +
                intent.request_id
            ),

            request_id=
                intent.request_id,

            dataset_id=
                intent.dataset_id,

            dataset_filename=
                intent.dataset_filename,

            operation=
                intent.operation,

            status=
                TransformationStatus
                .VALIDATED,

            risk=
                TransformationRisk
                .LOW,

            input_columns=[
                intent.source_column
            ],

            output_column=
                intent.output_column,

            output_dataset_id=None,

            output_dataset_filename=None,

            parameters={
                "source_column":
                    intent.source_column,

                "part":
                    intent.part
                    .value,
            },

            rationale=(
                "The source column is deterministically "
                "compatible with date-part extraction."
            ),

            requires_human_approval=
                False,

            executable=
                False,
        )
    )


# ============================================================
# AGGREGATION
# ============================================================


def _plan_aggregate(
    *,
    intent: AggregateIntent,
    schema: Dict[
        str,
        str,
    ],
) -> TransformationStep:
    _require_operation(
        actual=
            intent.operation,

        expected=
            TransformationOperation
            .AGGREGATE,
    )

    if not (
        intent.group_by
    ):
        raise ValueError(
            (
                "AGGREGATE requires at least "
                "one group_by column."
            )
        )

    if not (
        intent.metrics
    ):
        raise ValueError(
            (
                "AGGREGATE requires at least "
                "one metric."
            )
        )

    # --------------------------------------------------------
    # Duplicate group keys are not useful and often indicate
    # a planner/LLM mistake.
    # --------------------------------------------------------

    group_by = list(
        dict.fromkeys(
            intent.group_by
        )
    )

    if (
        len(
            group_by
        )
        !=
        len(
            intent.group_by
        )
    ):
        raise ValueError(
            (
                "Duplicate group_by columns "
                "are not allowed."
            )
        )

    for column in (
        group_by
    ):
        _require_column(
            column=
                column,

            schema=
                schema,
        )

    metric_outputs: Set[
        str
    ] = set()

    input_columns: List[
        str
    ] = list(
        group_by
    )

    metric_parameters = []

    numeric_functions = {
        AggregationFunction.SUM,
        AggregationFunction.MEAN,
        AggregationFunction.MEDIAN,
        AggregationFunction.MIN,
        AggregationFunction.MAX,
    }

    for metric in (
        intent.metrics
    ):
        _require_column(
            column=
                metric.source_column,

            schema=
                schema,
        )

        if not (
            metric.output_column
            .strip()
        ):
            raise ValueError(
                (
                    "Aggregate output column "
                    "cannot be empty."
                )
            )

        if (
            metric.output_column
            in metric_outputs
        ):
            raise ValueError(
                (
                    "Duplicate aggregate output "
                    "column: "
                    f"{metric.output_column}"
                )
            )

        if (
            metric.output_column
            in group_by
        ):
            raise ValueError(
                (
                    "Aggregate output column "
                    "cannot collide with a "
                    "group_by column."
                )
            )

        metric_outputs.add(
            metric.output_column
        )

        if (
            metric.function
            in
            numeric_functions
        ):
            _require_numeric_column(
                column=
                    metric.source_column,

                schema=
                    schema,
            )

        if (
            metric.source_column
            not in
            input_columns
        ):
            input_columns.append(
                metric.source_column
            )

        metric_parameters.append(
            {
                "source_column":
                    metric.source_column,

                "function":
                    metric.function
                    .value,

                "output_column":
                    metric.output_column,
            }
        )

    if not (
        intent.output_dataset_id
        .strip()
    ):
        raise ValueError(
            (
                "Aggregate output_dataset_id "
                "cannot be empty."
            )
        )

    # --------------------------------------------------------
    # Aggregation changes grain, therefore v0.1 forces
    # creation of a separate derived dataset.
    # --------------------------------------------------------

    if (
        intent.output_dataset_id
        ==
        intent.dataset_id
    ):
        raise ValueError(
            (
                "Aggregation must produce a "
                "new derived dataset_id in v0.1."
            )
        )

    if not (
        intent.output_dataset_filename
        .strip()
    ):
        raise ValueError(
            (
                "Aggregate output filename "
                "cannot be empty."
            )
        )

    return (
        TransformationStep(
            step_id=(
                "transform:"
                +
                intent.request_id
            ),

            request_id=
                intent.request_id,

            dataset_id=
                intent.dataset_id,

            dataset_filename=
                intent.dataset_filename,

            operation=
                intent.operation,

            status=
                TransformationStatus
                .REVIEW_REQUIRED,

            risk=
                TransformationRisk
                .MEDIUM,

            input_columns=
                input_columns,

            output_column=None,

            output_dataset_id=
                intent.output_dataset_id,

            output_dataset_filename=
                intent.output_dataset_filename,

            parameters={
                "group_by":
                    group_by,

                "metrics":
                    metric_parameters,
            },

            rationale=(
                "Aggregation changes the grain "
                "of the analytical dataset. "
                "Columns and aggregation functions "
                "are valid, but the grain change "
                "requires explicit approval."
            ),

            requires_human_approval=
                True,

            executable=
                False,
        )
    )


# ============================================================
# PUBLIC API
# ============================================================


def plan_transformations(
    *,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_filename: str,
    intents: List[
        TransformationIntent
    ],
) -> TransformationPlan:
    """
    Deterministically validate transformation intents.

    Safety guarantees:

    - never mutates the DataFrame;
    - never executes transformations;
    - never evaluates arbitrary expressions;
    - rejects unknown columns;
    - rejects output-column overwrites;
    - validates arithmetic operands;
    - validates numeric bin boundaries;
    - validates aggregation functions against types;
    - aggregation must be the final operation in v0.1.

    Derived columns are registered only in a virtual schema,
    allowing later planned operations to reference them.
    """

    if (
        dataframe is None
    ):
        raise ValueError(
            (
                "DataFrame cannot "
                "be None."
            )
        )

    if (
        dataframe.empty
    ):
        raise ValueError(
            (
                "Transformation planning "
                "requires at least one row."
            )
        )

    if not (
        dataset_id.strip()
    ):
        raise ValueError(
            (
                "dataset_id cannot "
                "be empty."
            )
        )

    if not (
        dataset_filename.strip()
    ):
        raise ValueError(
            (
                "dataset_filename cannot "
                "be empty."
            )
        )

    if not (
        intents
    ):
        raise ValueError(
            (
                "At least one transformation "
                "intent is required."
            )
        )

    # --------------------------------------------------------
    # Virtual schema — never modifies actual dataframe.
    # --------------------------------------------------------

    schema = (
        _build_virtual_schema(
            dataframe
        )
    )

    steps: List[
        TransformationStep
    ] = []

    seen_request_ids: Set[
        str
    ] = set()

    aggregate_seen = False

    for index, intent in enumerate(
        intents
    ):
        if (
            not intent.request_id
            .strip()
        ):
            raise ValueError(
                (
                    "Transformation request_id "
                    "cannot be empty."
                )
            )

        if (
            intent.request_id
            in
            seen_request_ids
        ):
            raise ValueError(
                (
                    "Duplicate transformation "
                    "request_id: "
                    f"{intent.request_id}"
                )
            )

        seen_request_ids.add(
            intent.request_id
        )

        _require_dataset_match(
            intent=
                intent,

            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,
        )

        if (
            aggregate_seen
        ):
            raise ValueError(
                (
                    "No transformation may follow "
                    "AGGREGATE in Transformation "
                    "Planner v0.1."
                )
            )

        # ====================================================
        # DERIVED ARITHMETIC
        # ====================================================

        if isinstance(
            intent,
            DeriveArithmeticIntent,
        ):
            step = (
                _plan_derive_arithmetic(
                    intent=
                        intent,

                    schema=
                        schema,
                )
            )

            # Virtual only.
            schema[
                intent.output_column
            ] = (
                TYPE_NUMERIC
            )

        # ====================================================
        # CAST
        # ====================================================

        elif isinstance(
            intent,
            CastIntent,
        ):
            step = (
                _plan_cast(
                    intent=
                        intent,

                    schema=
                        schema,
                )
            )

            target_family_map = {
                CastTargetType.STRING:
                    TYPE_STRING,

                CastTargetType.INTEGER:
                    TYPE_NUMERIC,

                CastTargetType.FLOAT:
                    TYPE_NUMERIC,

                CastTargetType.BOOLEAN:
                    TYPE_BOOLEAN,

                CastTargetType.DATETIME:
                    TYPE_DATETIME,
            }

            schema[
                intent.output_column
            ] = (
                target_family_map[
                    intent.target_type
                ]
            )

        # ====================================================
        # BIN
        # ====================================================

        elif isinstance(
            intent,
            BinNumericIntent,
        ):
            step = (
                _plan_bin_numeric(
                    intent=
                        intent,

                    schema=
                        schema,
                )
            )

            schema[
                intent.output_column
            ] = (
                TYPE_CATEGORICAL
            )

        # ====================================================
        # DATE PART
        # ====================================================

        elif isinstance(
            intent,
            ExtractDatePartIntent,
        ):
            step = (
                _plan_extract_date_part(
                    intent=
                        intent,

                    dataframe=
                        dataframe,

                    schema=
                        schema,
                )
            )

            if (
                intent.part
                ==
                DatePart.WEEKDAY
            ):
                schema[
                    intent.output_column
                ] = (
                    TYPE_CATEGORICAL
                )

            else:
                schema[
                    intent.output_column
                ] = (
                    TYPE_NUMERIC
                )

        # ====================================================
        # AGGREGATE
        # ====================================================

        elif isinstance(
            intent,
            AggregateIntent,
        ):
            step = (
                _plan_aggregate(
                    intent=
                        intent,

                    schema=
                        schema,
                )
            )

            aggregate_seen = (
                True
            )

            if (
                index
                !=
                len(
                    intents
                )
                -
                1
            ):
                raise ValueError(
                    (
                        "AGGREGATE must be the "
                        "final transformation "
                        "in v0.1."
                    )
                )

        else:
            raise TypeError(
                (
                    "Unsupported transformation "
                    "intent type: "
                    f"{type(intent).__name__}"
                )
            )

        steps.append(
            step
        )

    # ========================================================
    # COUNTS
    # ========================================================

    validated_count = sum(
        1

        for step
        in steps

        if (
            step.status
            ==
            TransformationStatus
            .VALIDATED
        )
    )

    review_required_count = sum(
        1

        for step
        in steps

        if (
            step.status
            ==
            TransformationStatus
            .REVIEW_REQUIRED
        )
    )

    human_approval_required_count = sum(
        1

        for step
        in steps

        if (
            step
            .requires_human_approval
        )
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    return (
        TransformationPlan(
            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            request_count=
                len(
                    intents
                ),

            step_count=
                len(
                    steps
                ),

            validated_count=
                validated_count,

            review_required_count=
                review_required_count,

            human_approval_required_count=
                human_approval_required_count,

            ready_for_approval=
                True,

            steps=
                steps,

            notes=[
                (
                    "Transformation Planner v0.1 "
                    "does not mutate any dataframe."
                ),

                (
                    "Arbitrary Python expressions "
                    "and eval() are not supported."
                ),

                (
                    "Arithmetic derivations use "
                    "structured operands and a "
                    "closed operator enum."
                ),

                (
                    "Derived columns exist only in "
                    "a virtual planning schema until "
                    "execution is explicitly authorized."
                ),

                (
                    "Numeric binning requires approval "
                    "because thresholds alter the "
                    "analytical representation."
                ),

                (
                    "Aggregation requires approval "
                    "because it changes dataset grain."
                ),

                (
                    "Aggregation must be the final "
                    "transformation in v0.1."
                ),

                (
                    "Every planned step remains "
                    "executable=False."
                ),
            ],
        )
    )