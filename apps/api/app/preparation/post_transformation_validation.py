from __future__ import annotations

from enum import Enum

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
)

from app.preparation.transformation_contracts import (
    AggregationFunction,
    ArithmeticOperator,
    CastTargetType,
    DatePart,
    OperandKind,
    TransformationOperation,
)

from app.preparation.transformation_executor import (
    TransformationExecutionResult,
    TransformationExecutionStatus,
    _dataframe_fingerprint,
)


# ============================================================
# VERSION
# ============================================================


POST_TRANSFORMATION_VALIDATION_RULE_VERSION = (
    "post_transformation_validation_v0.1"
)


# ============================================================
# STATUS
# ============================================================


class TransformationValidationStatus(
    str,
    Enum,
):
    PASSED = (
        "passed"
    )

    FAILED = (
        "failed"
    )

    WARNING = (
        "warning"
    )


# ============================================================
# CHECK
# ============================================================


class TransformationValidationCheck(
    BaseModel,
):
    check_id: str

    status: TransformationValidationStatus

    message: str

    request_id: Optional[
        str
    ] = None

    details: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )


# ============================================================
# STEP VALIDATION
# ============================================================


class TransformationStepValidation(
    BaseModel,
):
    request_id: str

    operation: TransformationOperation

    status: TransformationValidationStatus

    passed_count: int

    failed_count: int

    warning_count: int

    checks: List[
        TransformationValidationCheck
    ] = Field(
        default_factory=list
    )


# ============================================================
# REPORT
# ============================================================


class PostTransformationValidationReport(
    BaseModel,
):
    status: Literal[
        "passed",
        "failed",
    ]

    dataset_id: str

    dataset_filename: str

    valid_for_downstream: bool

    passed_check_count: int

    failed_check_count: int

    warning_count: int

    validated_step_count: int

    derived_dataset_count: int

    step_validations: List[
        TransformationStepValidation
    ] = Field(
        default_factory=list
    )

    checks: List[
        TransformationValidationCheck
    ] = Field(
        default_factory=list
    )

    notes: List[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        POST_TRANSFORMATION_VALIDATION_RULE_VERSION
    )


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


def _passed_check(
    *,
    check_id: str,
    message: str,
    request_id: Optional[
        str
    ] = None,
    details: Optional[
        Dict[
            str,
            Any,
        ]
    ] = None,
) -> TransformationValidationCheck:
    return (
        TransformationValidationCheck(
            check_id=
                check_id,

            status=
                TransformationValidationStatus
                .PASSED,

            message=
                message,

            request_id=
                request_id,

            details=
                details
                or
                {},
        )
    )


def _failed_check(
    *,
    check_id: str,
    message: str,
    request_id: Optional[
        str
    ] = None,
    details: Optional[
        Dict[
            str,
            Any,
        ]
    ] = None,
) -> TransformationValidationCheck:
    return (
        TransformationValidationCheck(
            check_id=
                check_id,

            status=
                TransformationValidationStatus
                .FAILED,

            message=
                message,

            request_id=
                request_id,

            details=
                details
                or
                {},
        )
    )


def _warning_check(
    *,
    check_id: str,
    message: str,
    request_id: Optional[
        str
    ] = None,
    details: Optional[
        Dict[
            str,
            Any,
        ]
    ] = None,
) -> TransformationValidationCheck:
    return (
        TransformationValidationCheck(
            check_id=
                check_id,

            status=
                TransformationValidationStatus
                .WARNING,

            message=
                message,

            request_id=
                request_id,

            details=
                details
                or
                {},
        )
    )


def _series_equal(
    left: pd.Series,
    right: pd.Series,
) -> bool:
    try:
        pd.testing.assert_series_equal(
            left.reset_index(
                drop=True
            ),
            right.reset_index(
                drop=True
            ),
            check_names=
                False,
            check_dtype=
                False,
            check_categorical=
                False,
        )

        return True

    except AssertionError:
        return False


def _categorical_series_equal(
    left: pd.Series,
    right: pd.Series,
) -> bool:
    left_normalized = (
        left
        .astype(
            "string"
        )
        .fillna(
            "<missing>"
        )
        .reset_index(
            drop=True
        )
    )

    right_normalized = (
        right
        .astype(
            "string"
        )
        .fillna(
            "<missing>"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        left_normalized
        .equals(
            right_normalized
        )
    )


def _dataframes_equal(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> bool:
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(
                drop=True
            ),
            right.reset_index(
                drop=True
            ),
            check_dtype=
                False,
            check_categorical=
                False,
            check_like=
                False,
        )

        return True

    except AssertionError:
        return False


# ============================================================
# EXECUTION STEP INDEX
# ============================================================


def _execution_step_index(
    result: TransformationExecutionResult,
):
    output = {}

    for step in (
        result.report.steps
    ):
        if (
            step.request_id
            in output
        ):
            raise ValueError(
                (
                    "Transformation execution report "
                    "contains duplicate request_id: "
                    f"{step.request_id}"
                )
            )

        output[
            step.request_id
        ] = step

    return output


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

        if (
            not isinstance(
                column,
                str,
            )
            or
            column
            not in
            dataframe.columns
        ):
            raise ValueError(
                (
                    "Validation operand references "
                    "unknown column: "
                    f"{column}"
                )
            )

        return (
            dataframe[
                column
            ]
        )

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
            raise ValueError(
                (
                    "Validation literal operand "
                    "must be numeric."
                )
            )

        return value

    raise ValueError(
        (
            "Unknown operand kind during "
            f"validation: {kind}"
        )
    )


# ============================================================
# EXPECTED ARITHMETIC
# ============================================================


def _expected_arithmetic(
    *,
    dataframe: pd.DataFrame,
    step: ApprovedTransformationStep,
) -> pd.Series:
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
        return (
            left
            +
            right
        )

    if (
        operator
        ==
        ArithmeticOperator.SUBTRACT.value
    ):
        return (
            left
            -
            right
        )

    if (
        operator
        ==
        ArithmeticOperator.MULTIPLY.value
    ):
        return (
            left
            *
            right
        )

    if (
        operator
        ==
        ArithmeticOperator.DIVIDE.value
    ):
        return (
            left
            /
            right
        )

    raise ValueError(
        (
            "Unsupported arithmetic "
            f"operator: {operator}"
        )
    )


# ============================================================
# EXPECTED CAST
# ============================================================


def _expected_boolean_cast(
    series: pd.Series,
) -> pd.Series:
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

    mask = (
        series.notna()
    )

    normalized = (
        series.loc[
            mask
        ]
        .astype(
            str
        )
        .str
        .strip()
        .str
        .casefold()
    )

    output.loc[
        mask
    ] = (
        normalized.map(
            mapping
        )
    )

    return output


def _expected_cast(
    *,
    dataframe: pd.DataFrame,
    step: ApprovedTransformationStep,
) -> pd.Series:
    source_column = (
        step.parameters[
            "source_column"
        ]
    )

    source = (
        dataframe[
            source_column
        ]
    )

    target_type = (
        _enum_value(
            step.parameters[
                "target_type"
            ]
        )
    )

    if (
        target_type
        ==
        CastTargetType.STRING.value
    ):
        return (
            source.astype(
                "string"
            )
        )

    if (
        target_type
        ==
        CastTargetType.FLOAT.value
    ):
        return (
            pd.to_numeric(
                source,
                errors=
                    "coerce",
            )
            .astype(
                "Float64"
            )
        )

    if (
        target_type
        ==
        CastTargetType.INTEGER.value
    ):
        return (
            pd.to_numeric(
                source,
                errors=
                    "coerce",
            )
            .astype(
                "Int64"
            )
        )

    if (
        target_type
        ==
        CastTargetType.BOOLEAN.value
    ):
        return (
            _expected_boolean_cast(
                source
            )
        )

    if (
        target_type
        ==
        CastTargetType.DATETIME.value
    ):
        return (
            pd.to_datetime(
                source,
                errors=
                    "coerce",
            )
        )

    raise ValueError(
        (
            "Unsupported validation cast "
            f"target: {target_type}"
        )
    )


# ============================================================
# EXPECTED DATE PART
# ============================================================


def _expected_date_part(
    *,
    dataframe: pd.DataFrame,
    step: ApprovedTransformationStep,
) -> pd.Series:
    source_column = (
        step.parameters[
            "source_column"
        ]
    )

    parsed = (
        pd.to_datetime(
            dataframe[
                source_column
            ],
            errors=
                "coerce",
        )
    )

    part = (
        _enum_value(
            step.parameters[
                "part"
            ]
        )
    )

    if (
        part
        ==
        DatePart.YEAR.value
    ):
        return (
            parsed
            .dt
            .year
            .astype(
                "Int64"
            )
        )

    if (
        part
        ==
        DatePart.MONTH.value
    ):
        return (
            parsed
            .dt
            .month
            .astype(
                "Int64"
            )
        )

    if (
        part
        ==
        DatePart.DAY.value
    ):
        return (
            parsed
            .dt
            .day
            .astype(
                "Int64"
            )
        )

    if (
        part
        ==
        DatePart.QUARTER.value
    ):
        return (
            parsed
            .dt
            .quarter
            .astype(
                "Int64"
            )
        )

    if (
        part
        ==
        DatePart.WEEK.value
    ):
        return (
            parsed
            .dt
            .isocalendar()
            .week
            .astype(
                "Int64"
            )
        )

    if (
        part
        ==
        DatePart.WEEKDAY.value
    ):
        return (
            parsed
            .dt
            .weekday
            .astype(
                "Int64"
            )
        )

    raise ValueError(
        (
            "Unsupported date part during "
            f"validation: {part}"
        )
    )


# ============================================================
# EXPECTED BINNING
# ============================================================


def _expected_binning(
    *,
    dataframe: pd.DataFrame,
    step: ApprovedTransformationStep,
) -> pd.Series:
    source_column = (
        step.parameters[
            "source_column"
        ]
    )

    return (
        pd.cut(
            dataframe[
                source_column
            ],
            bins=
                list(
                    step.parameters[
                        "bins"
                    ]
                ),
            labels=
                step.parameters.get(
                    "labels"
                ),
            include_lowest=
                bool(
                    step.parameters.get(
                        "include_lowest",
                        True,
                    )
                ),
            right=
                bool(
                    step.parameters.get(
                        "right",
                        True,
                    )
                ),
        )
    )


# ============================================================
# EXPECTED AGGREGATION
# ============================================================


def _expected_aggregation(
    *,
    dataframe: pd.DataFrame,
    step: ApprovedTransformationStep,
) -> pd.DataFrame:
    group_by = list(
        step.parameters[
            "group_by"
        ]
    )

    metrics = list(
        step.parameters[
            "metrics"
        ]
    )

    named_aggregations = {}

    allowed_functions = {
        AggregationFunction.SUM.value,
        AggregationFunction.MEAN.value,
        AggregationFunction.MEDIAN.value,
        AggregationFunction.MIN.value,
        AggregationFunction.MAX.value,
        AggregationFunction.COUNT.value,
        AggregationFunction.NUNIQUE.value,
    }

    for metric in metrics:
        function = (
            _enum_value(
                metric[
                    "function"
                ]
            )
        )

        if (
            function
            not in
            allowed_functions
        ):
            raise ValueError(
                (
                    "Unsupported aggregation "
                    "during validation: "
                    f"{function}"
                )
            )

        named_aggregations[
            metric[
                "output_column"
            ]
        ] = (
            pd.NamedAgg(
                column=
                    metric[
                        "source_column"
                    ],
                aggfunc=
                    function,
            )
        )

    return (
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


# ============================================================
# SINGLE STEP VALIDATION
# ============================================================


def _validate_step(
    *,
    transformed_dataframe: pd.DataFrame,
    result: TransformationExecutionResult,
    approved_step: ApprovedTransformationStep,
    execution_step,
) -> TransformationStepValidation:
    checks: List[
        TransformationValidationCheck
    ] = []

    request_id = (
        approved_step.request_id
    )

    # ========================================================
    # EXECUTION STATUS
    # ========================================================

    expected_execution_status = (
        TransformationExecutionStatus.APPLIED
        if approved_step.executable
        else TransformationExecutionStatus.SKIPPED
    )

    if (
        execution_step.status
        ==
        expected_execution_status
    ):
        checks.append(
            _passed_check(
                check_id=(
                    f"{request_id}:execution-status"
                ),

                request_id=
                    request_id,

                message=(
                    "Execution status matches "
                    "the approved authorization."
                ),
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=(
                    f"{request_id}:execution-status"
                ),

                request_id=
                    request_id,

                message=(
                    "Execution status does not match "
                    "the approved authorization."
                ),

                details={
                    "expected":
                        expected_execution_status.value,

                    "observed":
                        execution_step.status.value,
                },
            )
        )

    # ========================================================
    # SKIPPED STEP
    # ========================================================

    if not (
        approved_step.executable
    ):
        if (
            approved_step.output_column
            and
            approved_step.output_column
            in
            transformed_dataframe.columns
        ):
            checks.append(
                _failed_check(
                    check_id=(
                        f"{request_id}:skipped-output"
                    ),

                    request_id=
                        request_id,

                    message=(
                        "A skipped transformation "
                        "unexpectedly produced its "
                        "output column."
                    ),
                )
            )

        elif (
            approved_step.output_dataset_id
            and
            approved_step.output_dataset_id
            in
            result.derived_datasets
        ):
            checks.append(
                _failed_check(
                    check_id=(
                        f"{request_id}:skipped-dataset"
                    ),

                    request_id=
                        request_id,

                    message=(
                        "A skipped transformation "
                        "unexpectedly produced its "
                        "derived dataset."
                    ),
                )
            )

        else:
            checks.append(
                _passed_check(
                    check_id=(
                        f"{request_id}:skipped-postcondition"
                    ),

                    request_id=
                        request_id,

                    message=(
                        "Skipped transformation produced "
                        "no unauthorized output."
                    ),
                )
            )

        return (
            _build_step_validation(
                approved_step=
                    approved_step,

                checks=
                    checks,
            )
        )

    # ========================================================
    # DERIVE ARITHMETIC
    # ========================================================

    if (
        approved_step.operation
        ==
        TransformationOperation
        .DERIVE_ARITHMETIC
    ):
        output_column = (
            approved_step.output_column
        )

        if (
            not output_column
            or
            output_column
            not in
            transformed_dataframe.columns
        ):
            checks.append(
                _failed_check(
                    check_id=(
                        f"{request_id}:output-column"
                    ),

                    request_id=
                        request_id,

                    message=(
                        "Expected arithmetic output "
                        "column is missing."
                    ),
                )
            )

        else:
            expected = (
                _expected_arithmetic(
                    dataframe=
                        transformed_dataframe,

                    step=
                        approved_step,
                )
            )

            observed = (
                transformed_dataframe[
                    output_column
                ]
            )

            if (
                _series_equal(
                    observed,
                    expected,
                )
            ):
                checks.append(
                    _passed_check(
                        check_id=(
                            f"{request_id}:arithmetic-values"
                        ),

                        request_id=
                            request_id,

                        message=(
                            "Derived arithmetic values "
                            "match the approved formula."
                        ),
                    )
                )

            else:
                checks.append(
                    _failed_check(
                        check_id=(
                            f"{request_id}:arithmetic-values"
                        ),

                        request_id=
                            request_id,

                        message=(
                            "Derived arithmetic values "
                            "do not match the approved "
                            "formula."
                        ),
                    )
                )

    # ========================================================
    # CAST
    # ========================================================

    elif (
        approved_step.operation
        ==
        TransformationOperation.CAST
    ):
        output_column = (
            approved_step.output_column
        )

        if (
            not output_column
            or
            output_column
            not in
            transformed_dataframe.columns
        ):
            checks.append(
                _failed_check(
                    check_id=(
                        f"{request_id}:output-column"
                    ),

                    request_id=
                        request_id,

                    message=(
                        "Expected cast output "
                        "column is missing."
                    ),
                )
            )

        else:
            expected = (
                _expected_cast(
                    dataframe=
                        transformed_dataframe,

                    step=
                        approved_step,
                )
            )

            observed = (
                transformed_dataframe[
                    output_column
                ]
            )

            if (
                _series_equal(
                    observed,
                    expected,
                )
            ):
                checks.append(
                    _passed_check(
                        check_id=(
                            f"{request_id}:cast-values"
                        ),

                        request_id=
                            request_id,

                        message=(
                            "Cast output matches "
                            "the approved conversion."
                        ),
                    )
                )

            else:
                checks.append(
                    _failed_check(
                        check_id=(
                            f"{request_id}:cast-values"
                        ),

                        request_id=
                            request_id,

                        message=(
                            "Cast output does not match "
                            "the approved conversion."
                        ),
                    )
                )

    # ========================================================
    # BIN NUMERIC
    # ========================================================

    elif (
        approved_step.operation
        ==
        TransformationOperation
        .BIN_NUMERIC
    ):
        output_column = (
            approved_step.output_column
        )

        if (
            not output_column
            or
            output_column
            not in
            transformed_dataframe.columns
        ):
            checks.append(
                _failed_check(
                    check_id=(
                        f"{request_id}:output-column"
                    ),

                    request_id=
                        request_id,

                    message=(
                        "Expected binned output "
                        "column is missing."
                    ),
                )
            )

        else:
            expected = (
                _expected_binning(
                    dataframe=
                        transformed_dataframe,

                    step=
                        approved_step,
                )
            )

            observed = (
                transformed_dataframe[
                    output_column
                ]
            )

            if (
                _categorical_series_equal(
                    observed,
                    expected,
                )
            ):
                checks.append(
                    _passed_check(
                        check_id=(
                            f"{request_id}:bin-values"
                        ),

                        request_id=
                            request_id,

                        message=(
                            "Binned values match "
                            "the approved boundaries "
                            "and labels."
                        ),
                    )
                )

            else:
                checks.append(
                    _failed_check(
                        check_id=(
                            f"{request_id}:bin-values"
                        ),

                        request_id=
                            request_id,

                        message=(
                            "Binned values do not match "
                            "the approved boundaries."
                        ),
                    )
                )

    # ========================================================
    # DATE PART
    # ========================================================

    elif (
        approved_step.operation
        ==
        TransformationOperation
        .EXTRACT_DATE_PART
    ):
        output_column = (
            approved_step.output_column
        )

        if (
            not output_column
            or
            output_column
            not in
            transformed_dataframe.columns
        ):
            checks.append(
                _failed_check(
                    check_id=(
                        f"{request_id}:output-column"
                    ),

                    request_id=
                        request_id,

                    message=(
                        "Expected date-part output "
                        "column is missing."
                    ),
                )
            )

        else:
            expected = (
                _expected_date_part(
                    dataframe=
                        transformed_dataframe,

                    step=
                        approved_step,
                )
            )

            observed = (
                transformed_dataframe[
                    output_column
                ]
            )

            if (
                _series_equal(
                    observed,
                    expected,
                )
            ):
                checks.append(
                    _passed_check(
                        check_id=(
                            f"{request_id}:date-part-values"
                        ),

                        request_id=
                            request_id,

                        message=(
                            "Extracted date-part values "
                            "match the approved rule."
                        ),
                    )
                )

            else:
                checks.append(
                    _failed_check(
                        check_id=(
                            f"{request_id}:date-part-values"
                        ),

                        request_id=
                            request_id,

                        message=(
                            "Extracted date-part values "
                            "do not match the approved rule."
                        ),
                    )
                )

    # ========================================================
    # AGGREGATION
    # ========================================================

    elif (
        approved_step.operation
        ==
        TransformationOperation
        .AGGREGATE
    ):
        output_dataset_id = (
            approved_step.output_dataset_id
        )

        if (
            not output_dataset_id
            or
            output_dataset_id
            not in
            result.derived_datasets
        ):
            checks.append(
                _failed_check(
                    check_id=(
                        f"{request_id}:derived-dataset"
                    ),

                    request_id=
                        request_id,

                    message=(
                        "Expected aggregate derived "
                        "dataset is missing."
                    ),
                )
            )

        else:
            observed = (
                result
                .derived_datasets[
                    output_dataset_id
                ]
            )

            expected = (
                _expected_aggregation(
                    dataframe=
                        transformed_dataframe,

                    step=
                        approved_step,
                )
            )

            if (
                _dataframes_equal(
                    observed,
                    expected,
                )
            ):
                checks.append(
                    _passed_check(
                        check_id=(
                            f"{request_id}:aggregate-values"
                        ),

                        request_id=
                            request_id,

                        message=(
                            "Derived aggregate dataset "
                            "matches the approved grouping "
                            "and metrics."
                        ),
                    )
                )

            else:
                checks.append(
                    _failed_check(
                        check_id=(
                            f"{request_id}:aggregate-values"
                        ),

                        request_id=
                            request_id,

                        message=(
                            "Derived aggregate dataset "
                            "does not match the approved "
                            "grouping and metrics."
                        ),
                    )
                )

            if (
                execution_step.output_fingerprint
                is None
            ):
                checks.append(
                    _failed_check(
                        check_id=(
                            f"{request_id}:output-fingerprint"
                        ),

                        request_id=
                            request_id,

                        message=(
                            "Aggregate execution report "
                            "does not contain an output "
                            "fingerprint."
                        ),
                    )
                )

            else:
                actual_fingerprint = (
                    _dataframe_fingerprint(
                        observed
                    )
                )

                if (
                    actual_fingerprint
                    ==
                    execution_step
                    .output_fingerprint
                ):
                    checks.append(
                        _passed_check(
                            check_id=(
                                f"{request_id}:output-fingerprint"
                            ),

                            request_id=
                                request_id,

                            message=(
                                "Derived dataset fingerprint "
                                "matches the execution report."
                            ),
                        )
                    )

                else:
                    checks.append(
                        _failed_check(
                            check_id=(
                                f"{request_id}:output-fingerprint"
                            ),

                            request_id=
                                request_id,

                            message=(
                                "Derived dataset fingerprint "
                                "does not match the execution "
                                "report."
                            ),
                        )
                    )

    else:
        checks.append(
            _warning_check(
                check_id=(
                    f"{request_id}:unknown-postcondition"
                ),

                request_id=
                    request_id,

                message=(
                    "No explicit postcondition exists "
                    "for this transformation operation."
                ),
            )
        )

    return (
        _build_step_validation(
            approved_step=
                approved_step,

            checks=
                checks,
        )
    )


# ============================================================
# STEP SUMMARY
# ============================================================


def _build_step_validation(
    *,
    approved_step: ApprovedTransformationStep,
    checks: List[
        TransformationValidationCheck
    ],
) -> TransformationStepValidation:
    passed_count = sum(
        check.status
        ==
        TransformationValidationStatus.PASSED

        for check
        in checks
    )

    failed_count = sum(
        check.status
        ==
        TransformationValidationStatus.FAILED

        for check
        in checks
    )

    warning_count = sum(
        check.status
        ==
        TransformationValidationStatus.WARNING

        for check
        in checks
    )

    if (
        failed_count
        >
        0
    ):
        status = (
            TransformationValidationStatus
            .FAILED
        )

    elif (
        warning_count
        >
        0
    ):
        status = (
            TransformationValidationStatus
            .WARNING
        )

    else:
        status = (
            TransformationValidationStatus
            .PASSED
        )

    return (
        TransformationStepValidation(
            request_id=
                approved_step.request_id,

            operation=
                approved_step.operation,

            status=
                status,

            passed_count=
                passed_count,

            failed_count=
                failed_count,

            warning_count=
                warning_count,

            checks=
                checks,
        )
    )


# ============================================================
# PUBLIC API
# ============================================================


def validate_transformation_execution(
    *,
    source_dataframe: pd.DataFrame,
    execution_result: TransformationExecutionResult,
    approved_plan: ApprovedTransformationPlan,
    dataset_id: str,
    dataset_filename: str,
) -> PostTransformationValidationReport:
    """
    Validate a completed Transformation Executor result.

    This function NEVER mutates data.

    It verifies:

    - approved plan readiness;
    - dataset identity;
    - BEFORE fingerprint;
    - AFTER fingerprint;
    - source row-count invariant;
    - report row/column consistency;
    - execution step count;
    - exact approved/executed step reconciliation;
    - derived-column postconditions;
    - aggregation postconditions;
    - derived dataset fingerprints.

    Any FAILED check makes valid_for_downstream=False.
    """

    transformed = (
        execution_result.dataframe
    )

    report = (
        execution_result.report
    )

    checks: List[
        TransformationValidationCheck
    ] = []

    # ========================================================
    # PLAN READY
    # ========================================================

    if (
        approved_plan.ready_for_execution
    ):
        checks.append(
            _passed_check(
                check_id=
                    "plan-ready",

                message=(
                    "Approved transformation plan "
                    "was ready for execution."
                ),
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=
                    "plan-ready",

                message=(
                    "Approved transformation plan "
                    "was not ready for execution."
                ),
            )
        )

    # ========================================================
    # DATASET IDENTITY
    # ========================================================

    identity_valid = (
        approved_plan.dataset_id
        ==
        dataset_id
        ==
        report.dataset_id
        and
        approved_plan.dataset_filename
        ==
        dataset_filename
        ==
        report.dataset_filename
    )

    if (
        identity_valid
    ):
        checks.append(
            _passed_check(
                check_id=
                    "dataset-identity",

                message=(
                    "Dataset identity is consistent "
                    "across plan, execution and "
                    "validation."
                ),
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=
                    "dataset-identity",

                message=(
                    "Dataset identity is inconsistent."
                ),
            )
        )

    # ========================================================
    # BEFORE FINGERPRINT
    # ========================================================

    actual_before_fingerprint = (
        _dataframe_fingerprint(
            source_dataframe
        )
    )

    if (
        actual_before_fingerprint
        ==
        report.source_fingerprint_before
    ):
        checks.append(
            _passed_check(
                check_id=
                    "source-before-fingerprint",

                message=(
                    "Source dataframe fingerprint "
                    "matches the executor's BEFORE "
                    "fingerprint."
                ),
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=
                    "source-before-fingerprint",

                message=(
                    "Source dataframe no longer "
                    "matches the executor's BEFORE "
                    "fingerprint."
                ),
            )
        )

    # ========================================================
    # AFTER FINGERPRINT
    # ========================================================

    actual_after_fingerprint = (
        _dataframe_fingerprint(
            transformed
        )
    )

    if (
        actual_after_fingerprint
        ==
        report.source_fingerprint_after
    ):
        checks.append(
            _passed_check(
                check_id=
                    "transformed-after-fingerprint",

                message=(
                    "Transformed dataframe fingerprint "
                    "matches the executor's AFTER "
                    "fingerprint."
                ),
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=
                    "transformed-after-fingerprint",

                message=(
                    "Transformed dataframe was changed "
                    "after executor reporting or does not "
                    "match the reported output."
                ),
            )
        )

    # ========================================================
    # ROW COUNTS
    # ========================================================

    row_counts_valid = (
        len(
            source_dataframe
        )
        ==
        report.source_rows_before
        ==
        report.source_rows_after
        ==
        len(
            transformed
        )
    )

    if (
        row_counts_valid
    ):
        checks.append(
            _passed_check(
                check_id=
                    "source-row-count",

                message=(
                    "Source row grain was preserved "
                    "through transformation."
                ),
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=
                    "source-row-count",

                message=(
                    "Source row count is inconsistent "
                    "with transformation execution."
                ),
            )
        )

    # ========================================================
    # COLUMN COUNTS
    # ========================================================

    column_counts_valid = (
        source_dataframe.shape[
            1
        ]
        ==
        report.source_columns_before
        and
        transformed.shape[
            1
        ]
        ==
        report.source_columns_after
    )

    if (
        column_counts_valid
    ):
        checks.append(
            _passed_check(
                check_id=
                    "source-column-count",

                message=(
                    "Source column counts match "
                    "the execution report."
                ),
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=
                    "source-column-count",

                message=(
                    "Source column counts do not "
                    "match the execution report."
                ),
            )
        )

    # ========================================================
    # EXPECTED NEW COLUMNS
    # ========================================================

    expected_added_columns = [
        step.output_column

        for step
        in approved_plan.steps

        if (
            step.executable
            and
            step.output_column
            is not None
        )
    ]

    expected_after_column_count = (
        source_dataframe.shape[
            1
        ]
        +
        len(
            expected_added_columns
        )
    )

    if (
        transformed.shape[
            1
        ]
        ==
        expected_after_column_count
    ):
        checks.append(
            _passed_check(
                check_id=
                    "expected-column-growth",

                message=(
                    "The number of added columns "
                    "matches executable transformation "
                    "steps."
                ),

                details={
                    "expected_added_columns":
                        expected_added_columns,
                },
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=
                    "expected-column-growth",

                message=(
                    "Unexpected source-column growth "
                    "was detected."
                ),

                details={
                    "expected_after":
                        expected_after_column_count,

                    "observed_after":
                        int(
                            transformed.shape[
                                1
                            ]
                        ),
                },
            )
        )

    # ========================================================
    # EXECUTION STEP COUNT
    # ========================================================

    if (
        len(
            report.steps
        )
        ==
        len(
            approved_plan.steps
        )
    ):
        checks.append(
            _passed_check(
                check_id=
                    "execution-step-count",

                message=(
                    "Execution report contains one "
                    "record for every approved "
                    "transformation step."
                ),
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=
                    "execution-step-count",

                message=(
                    "Execution report step count does "
                    "not match the approved plan."
                ),
            )
        )

    # ========================================================
    # DERIVED DATASET COUNT
    # ========================================================

    expected_derived_dataset_ids = {
        step.output_dataset_id

        for step
        in approved_plan.steps

        if (
            step.executable
            and
            step.output_dataset_id
            is not None
        )
    }

    actual_derived_dataset_ids = set(
        execution_result
        .derived_datasets
        .keys()
    )

    derived_match = (
        expected_derived_dataset_ids
        ==
        actual_derived_dataset_ids
        and
        len(
            actual_derived_dataset_ids
        )
        ==
        report.derived_dataset_count
    )

    if (
        derived_match
    ):
        checks.append(
            _passed_check(
                check_id=
                    "derived-datasets",

                message=(
                    "Derived datasets exactly match "
                    "the executable approved plan."
                ),
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=
                    "derived-datasets",

                message=(
                    "Derived dataset outputs do not "
                    "match the approved plan."
                ),

                details={
                    "expected":
                        sorted(
                            expected_derived_dataset_ids
                        ),

                    "observed":
                        sorted(
                            actual_derived_dataset_ids
                        ),
                },
            )
        )

    # ========================================================
    # STEP-BY-STEP RECONCILIATION
    # ========================================================

    execution_index = (
        _execution_step_index(
            execution_result
        )
    )

    step_validations: List[
        TransformationStepValidation
    ] = []

    for approved_step in (
        approved_plan.steps
    ):
        execution_step = (
            execution_index.get(
                approved_step.request_id
            )
        )

        if (
            execution_step
            is None
        ):
            missing_check = (
                _failed_check(
                    check_id=(
                        f"{approved_step.request_id}:"
                        "missing-execution-step"
                    ),

                    request_id=
                        approved_step.request_id,

                    message=(
                        "Approved transformation step "
                        "is missing from the execution "
                        "report."
                    ),
                )
            )

            step_validations.append(
                _build_step_validation(
                    approved_step=
                        approved_step,

                    checks=[
                        missing_check
                    ],
                )
            )

            continue

        step_validations.append(
            _validate_step(
                transformed_dataframe=
                    transformed,

                result=
                    execution_result,

                approved_step=
                    approved_step,

                execution_step=
                    execution_step,
            )
        )

    # ========================================================
    # EXTRA EXECUTION STEPS
    # ========================================================

    approved_request_ids = {
        step.request_id

        for step
        in approved_plan.steps
    }

    extra_execution_ids = (
        set(
            execution_index.keys()
        )
        -
        approved_request_ids
    )

    if not (
        extra_execution_ids
    ):
        checks.append(
            _passed_check(
                check_id=
                    "no-extra-execution-steps",

                message=(
                    "No unapproved transformation "
                    "steps appear in execution."
                ),
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=
                    "no-extra-execution-steps",

                message=(
                    "Execution contains transformation "
                    "steps absent from the approved plan."
                ),

                details={
                    "extra_request_ids":
                        sorted(
                            extra_execution_ids
                        ),
                },
            )
        )

    # ========================================================
    # COUNTS
    # ========================================================

    all_checks = list(
        checks
    )

    for step_validation in (
        step_validations
    ):
        all_checks.extend(
            step_validation.checks
        )

    passed_count = sum(
        check.status
        ==
        TransformationValidationStatus.PASSED

        for check
        in all_checks
    )

    failed_count = sum(
        check.status
        ==
        TransformationValidationStatus.FAILED

        for check
        in all_checks
    )

    warning_count = sum(
        check.status
        ==
        TransformationValidationStatus.WARNING

        for check
        in all_checks
    )

    valid_for_downstream = (
        failed_count
        ==
        0
    )

    return (
        PostTransformationValidationReport(
            status=(
                "passed"
                if valid_for_downstream
                else
                "failed"
            ),

            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            valid_for_downstream=
                valid_for_downstream,

            passed_check_count=
                passed_count,

            failed_check_count=
                failed_count,

            warning_count=
                warning_count,

            validated_step_count=
                len(
                    step_validations
                ),

            derived_dataset_count=
                len(
                    execution_result
                    .derived_datasets
                ),

            step_validations=
                step_validations,

            checks=
                checks,

            notes=[
                (
                    "Post-transformation Validation "
                    "never mutates source or transformed "
                    "datasets."
                ),

                (
                    "Fingerprints reconcile the source "
                    "before and transformed output after "
                    "execution."
                ),

                (
                    "Derived arithmetic values are "
                    "recomputed from approved operands."
                ),

                (
                    "Date parts and numeric bins are "
                    "recomputed from approved parameters."
                ),

                (
                    "Aggregated datasets are independently "
                    "recomputed from the transformed source."
                ),

                (
                    "A single failed validation check "
                    "sets valid_for_downstream=False."
                ),
            ],

            rule_version=
                POST_TRANSFORMATION_VALIDATION_RULE_VERSION,
        )
    )