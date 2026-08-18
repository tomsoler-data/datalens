from __future__ import annotations

from enum import Enum

import hashlib
import json

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

from app.preparation.join_approval import (
    ApprovedJoin,
    ApprovedJoinPlan,
    JoinAuthorizationStatus,
)

from app.preparation.join_contracts import (
    JoinCardinality,
    JoinType,
)

from app.preparation.join_executor import (
    JoinExecutionResult,
    JoinExecutionStatus,
)


# ============================================================
# VERSION
# ============================================================


POST_JOIN_VALIDATION_RULE_VERSION = (
    "post_join_validation_v0.1"
)


# ============================================================
# STATUS
# ============================================================


class JoinValidationStatus(
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


class JoinValidationCheck(
    BaseModel,
):
    check_id: str

    status: JoinValidationStatus

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
# JOIN VALIDATION
# ============================================================


class JoinStepValidation(
    BaseModel,
):
    request_id: str

    status: JoinValidationStatus

    passed_count: int

    failed_count: int

    warning_count: int

    checks: List[
        JoinValidationCheck
    ] = Field(
        default_factory=list
    )


# ============================================================
# REPORT
# ============================================================


class PostJoinValidationReport(
    BaseModel,
):
    status: Literal[
        "passed",
        "failed",
    ]

    valid_for_downstream: bool

    total_join_count: int

    validated_join_count: int

    passed_check_count: int

    failed_check_count: int

    warning_count: int

    output_dataset_count: int

    join_validations: List[
        JoinStepValidation
    ] = Field(
        default_factory=list
    )

    checks: List[
        JoinValidationCheck
    ] = Field(
        default_factory=list
    )

    notes: List[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        POST_JOIN_VALIDATION_RULE_VERSION
    )


# ============================================================
# FINGERPRINT
# ============================================================


def _dataframe_fingerprint(
    dataframe: pd.DataFrame,
) -> str:
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
            sort_keys=True,
            ensure_ascii=False,
        ).encode(
            "utf-8"
        )
    )

    try:
        hashed = (
            pd.util
            .hash_pandas_object(
                dataframe,
                index=True,
                categorize=True,
            )
        )

    except TypeError:
        hashed = (
            pd.util
            .hash_pandas_object(
                dataframe.astype(
                    "string"
                ),
                index=True,
                categorize=True,
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


# ============================================================
# CHECK FACTORIES
# ============================================================


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
) -> JoinValidationCheck:
    return (
        JoinValidationCheck(
            check_id=
                check_id,

            status=
                JoinValidationStatus.PASSED,

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
) -> JoinValidationCheck:
    return (
        JoinValidationCheck(
            check_id=
                check_id,

            status=
                JoinValidationStatus.FAILED,

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
) -> JoinValidationCheck:
    return (
        JoinValidationCheck(
            check_id=
                check_id,

            status=
                JoinValidationStatus.WARNING,

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


# ============================================================
# GENERIC DATAFRAME COMPARISON
# ============================================================


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
            check_dtype=False,
            check_categorical=False,
            check_like=False,
        )

        return True

    except AssertionError:
        return False


# ============================================================
# SOURCE DATASET
# ============================================================


def _require_dataset(
    *,
    datasets: Dict[
        str,
        pd.DataFrame,
    ],
    dataset_id: str,
) -> pd.DataFrame:
    dataframe = (
        datasets.get(
            dataset_id
        )
    )

    if dataframe is None:
        raise ValueError(
            (
                "Post-join validation cannot find "
                "source dataset: "
                f"{dataset_id}"
            )
        )

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            (
                "Post-join validation requires "
                "pandas DataFrames."
            )
        )

    return dataframe


# ============================================================
# NULL-SAFE JOIN HELPERS
# ============================================================


def _is_missing_value(
    value,
) -> bool:
    try:
        result = (
            pd.isna(
                value
            )
        )

        if isinstance(
            result,
            bool,
        ):
            return result

        return bool(
            result
        )

    except (
        TypeError,
        ValueError,
    ):
        return False


def _build_composite_join_key(
    *,
    dataframe: pd.DataFrame,
    columns: List[
        str
    ],
    side: str,
) -> pd.Series:
    """
    Independently reproduce DataLens join semantics:

    missing key on left
        NEVER matches
    missing key on right
    """

    values = []

    for row_number, row in enumerate(
        dataframe[
            columns
        ]
        .itertuples(
            index=False,
            name=None,
        )
    ):
        has_missing = any(
            _is_missing_value(
                value
            )
            for value
            in row
        )

        if (
            has_missing
        ):
            values.append(
                (
                    "__DATALENS_NULL__",
                    side,
                    row_number,
                )
            )

        else:
            values.append(
                (
                    "__DATALENS_VALUE__",
                    tuple(
                        row
                    ),
                )
            )

    return (
        pd.Series(
            values,
            index=
                dataframe.index,
            dtype=
                "object",
        )
    )


def _temporary_join_column(
    *,
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> str:
    base = (
        "__datalens_validation_join_key__"
    )

    candidate = (
        base
    )

    counter = (
        1
    )

    columns = (
        set(
            str(
                column
            )
            for column
            in left.columns
        )
        |
        set(
            str(
                column
            )
            for column
            in right.columns
        )
    )

    while (
        candidate
        in
        columns
    ):
        candidate = (
            f"{base}{counter}"
        )

        counter += 1

    return candidate


# ============================================================
# PANDAS CARDINALITY MODE
# ============================================================


def _pandas_validate_mode(
    cardinality: JoinCardinality,
) -> str:
    if (
        cardinality
        ==
        JoinCardinality.ONE_TO_ONE
    ):
        return (
            "one_to_one"
        )

    if (
        cardinality
        ==
        JoinCardinality.ONE_TO_MANY
    ):
        return (
            "one_to_many"
        )

    if (
        cardinality
        ==
        JoinCardinality.MANY_TO_ONE
    ):
        return (
            "many_to_one"
        )

    raise ValueError(
        (
            "MANY_TO_MANY cannot be independently "
            "validated as an executable join in v0.1."
        )
    )


# ============================================================
# SAME-NAME JOIN KEYS
# ============================================================


def _coalesce_same_named_keys(
    *,
    dataframe: pd.DataFrame,
    join: ApprovedJoin,
) -> pd.DataFrame:
    result = (
        dataframe
    )

    for key in (
        join.keys
    ):
        if (
            key.left_column
            !=
            key.right_column
        ):
            continue

        base_name = (
            key.left_column
        )

        left_name = (
            base_name
            +
            join.left_suffix
        )

        right_name = (
            base_name
            +
            join.right_suffix
        )

        if (
            left_name
            not in
            result.columns
            or
            right_name
            not in
            result.columns
        ):
            raise ValueError(
                (
                    "Unable to independently "
                    "reconcile same-name join key: "
                    f"{base_name}"
                )
            )

        left_position = (
            int(
                result.columns
                .get_loc(
                    left_name
                )
            )
        )

        right_position = (
            int(
                result.columns
                .get_loc(
                    right_name
                )
            )
        )

        insert_position = (
            min(
                left_position,
                right_position,
            )
        )

        combined = (
            result[
                left_name
            ]
            .combine_first(
                result[
                    right_name
                ]
            )
        )

        result = (
            result.drop(
                columns=[
                    left_name,
                    right_name,
                ]
            )
        )

        result.insert(
            insert_position,
            base_name,
            combined,
        )

    return (
        result
    )


# ============================================================
# INDEPENDENT JOIN RECOMPUTATION
# ============================================================


def _recompute_join_output(
    *,
    datasets: Dict[
        str,
        pd.DataFrame,
    ],
    join: ApprovedJoin,
) -> pd.DataFrame:
    """
    Recompute the join without using Join Executor.

    This is deliberately duplicated validation logic rather
    than simply trusting the executor output.
    """

    left_source = (
        _require_dataset(
            datasets=
                datasets,

            dataset_id=
                join.left_dataset_id,
        )
    )

    right_source = (
        _require_dataset(
            datasets=
                datasets,

            dataset_id=
                join.right_dataset_id,
        )
    )

    left = (
        left_source.copy(
            deep=True
        )
    )

    right = (
        right_source.copy(
            deep=True
        )
    )

    left_keys = [
        key.left_column

        for key
        in join.keys
    ]

    right_keys = [
        key.right_column

        for key
        in join.keys
    ]

    temporary_key = (
        _temporary_join_column(
            left=
                left,

            right=
                right,
        )
    )

    left[
        temporary_key
    ] = (
        _build_composite_join_key(
            dataframe=
                left,

            columns=
                left_keys,

            side=
                "left",
        )
    )

    right[
        temporary_key
    ] = (
        _build_composite_join_key(
            dataframe=
                right,

            columns=
                right_keys,

            side=
                "right",
        )
    )

    expected = (
        pd.merge(
            left,
            right,

            how=
                join.join_type.value,

            on=
                temporary_key,

            suffixes=(
                join.left_suffix,
                join.right_suffix,
            ),

            sort=False,

            copy=True,

            validate=
                _pandas_validate_mode(
                    join.detected_cardinality
                ),
        )
    )

    expected = (
        expected.drop(
            columns=[
                temporary_key
            ]
        )
    )

    expected = (
        _coalesce_same_named_keys(
            dataframe=
                expected,

            join=
                join,
        )
    )

    return (
        expected
    )


# ============================================================
# EXECUTION STEP INDEX
# ============================================================


def _execution_step_index(
    execution_result: JoinExecutionResult,
):
    output = {}

    for step in (
        execution_result
        .report
        .steps
    ):
        if (
            step.request_id
            in
            output
        ):
            raise ValueError(
                (
                    "Join execution report contains "
                    "duplicate request_id: "
                    f"{step.request_id}"
                )
            )

        output[
            step.request_id
        ] = step

    return (
        output
    )


# ============================================================
# STEP SUMMARY
# ============================================================


def _build_step_validation(
    *,
    join: ApprovedJoin,
    checks: List[
        JoinValidationCheck
    ],
) -> JoinStepValidation:
    passed_count = sum(
        check.status
        ==
        JoinValidationStatus.PASSED

        for check
        in checks
    )

    failed_count = sum(
        check.status
        ==
        JoinValidationStatus.FAILED

        for check
        in checks
    )

    warning_count = sum(
        check.status
        ==
        JoinValidationStatus.WARNING

        for check
        in checks
    )

    if (
        failed_count
        >
        0
    ):
        status = (
            JoinValidationStatus.FAILED
        )

    elif (
        warning_count
        >
        0
    ):
        status = (
            JoinValidationStatus.WARNING
        )

    else:
        status = (
            JoinValidationStatus.PASSED
        )

    return (
        JoinStepValidation(
            request_id=
                join.request_id,

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
# SINGLE JOIN VALIDATION
# ============================================================


def _validate_join(
    *,
    datasets: Dict[
        str,
        pd.DataFrame,
    ],
    approved_join: ApprovedJoin,
    execution_result: JoinExecutionResult,
    execution_step,
) -> JoinStepValidation:
    checks: List[
        JoinValidationCheck
    ] = []

    request_id = (
        approved_join.request_id
    )

    # ========================================================
    # AUTHORIZATION ↔ EXECUTION STATUS
    # ========================================================

    expected_status = (
        JoinExecutionStatus.APPLIED
        if approved_join.executable
        else
        JoinExecutionStatus.SKIPPED
    )

    if (
        execution_step.status
        ==
        expected_status
    ):
        checks.append(
            _passed_check(
                check_id=(
                    f"{request_id}:execution-status"
                ),

                request_id=
                    request_id,

                message=(
                    "Join execution status matches "
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
                    "Join execution status does not "
                    "match approved authorization."
                ),
            )
        )

    # ========================================================
    # REJECTED JOIN
    # ========================================================

    if not (
        approved_join.executable
    ):
        if (
            approved_join.output_dataset_id
            in
            execution_result.joined_datasets
        ):
            checks.append(
                _failed_check(
                    check_id=(
                        f"{request_id}:rejected-output"
                    ),

                    request_id=
                        request_id,

                    message=(
                        "Rejected join unexpectedly "
                        "produced an output dataset."
                    ),
                )
            )

        else:
            checks.append(
                _passed_check(
                    check_id=(
                        f"{request_id}:rejected-output"
                    ),

                    request_id=
                        request_id,

                    message=(
                        "Rejected join produced "
                        "no unauthorized output."
                    ),
                )
            )

        return (
            _build_step_validation(
                join=
                    approved_join,

                checks=
                    checks,
            )
        )

    # ========================================================
    # PREFLIGHT
    # ========================================================

    if (
        execution_step.preflight_passed
    ):
        checks.append(
            _passed_check(
                check_id=(
                    f"{request_id}:preflight"
                ),

                request_id=
                    request_id,

                message=(
                    "Executor reports a successful "
                    "deterministic preflight."
                ),
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=(
                    f"{request_id}:preflight"
                ),

                request_id=
                    request_id,

                message=(
                    "Executable join did not pass "
                    "preflight."
                ),
            )
        )

    # ========================================================
    # CARDINALITY
    # ========================================================

    if (
        execution_step.detected_cardinality
        ==
        approved_join.detected_cardinality
    ):
        checks.append(
            _passed_check(
                check_id=(
                    f"{request_id}:cardinality"
                ),

                request_id=
                    request_id,

                message=(
                    "Execution cardinality matches "
                    "the approved planner cardinality."
                ),
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=(
                    f"{request_id}:cardinality"
                ),

                request_id=
                    request_id,

                message=(
                    "Execution cardinality differs "
                    "from approved cardinality."
                ),
            )
        )

    # ========================================================
    # OUTPUT EXISTS
    # ========================================================

    output = (
        execution_result
        .joined_datasets
        .get(
            approved_join.output_dataset_id
        )
    )

    if (
        output is None
    ):
        checks.append(
            _failed_check(
                check_id=(
                    f"{request_id}:output-dataset"
                ),

                request_id=
                    request_id,

                message=(
                    "Expected joined dataset "
                    "is missing."
                ),
            )
        )

        return (
            _build_step_validation(
                join=
                    approved_join,

                checks=
                    checks,
            )
        )

    checks.append(
        _passed_check(
            check_id=(
                f"{request_id}:output-dataset"
            ),

            request_id=
                request_id,

            message=(
                "Expected joined dataset exists."
            ),
        )
    )

    # ========================================================
    # ROW COUNT
    # ========================================================

    expected_row_count = (
        approved_join
        .diagnostics
        .predicted_output_row_count
    )

    actual_row_count = int(
        len(
            output
        )
    )

    if (
        actual_row_count
        ==
        expected_row_count
        ==
        execution_step
        .predicted_output_row_count
        ==
        execution_step
        .actual_output_row_count
    ):
        checks.append(
            _passed_check(
                check_id=(
                    f"{request_id}:row-count"
                ),

                request_id=
                    request_id,

                message=(
                    "Actual join grain exactly "
                    "matches the planner prediction."
                ),

                details={
                    "rows":
                        actual_row_count,
                },
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=(
                    f"{request_id}:row-count"
                ),

                request_id=
                    request_id,

                message=(
                    "Join output row count does not "
                    "match approved or reported grain."
                ),

                details={
                    "approved":
                        expected_row_count,

                    "reported_predicted": (
                        execution_step
                        .predicted_output_row_count
                    ),

                    "reported_actual": (
                        execution_step
                        .actual_output_row_count
                    ),

                    "observed":
                        actual_row_count,
                },
            )
        )

    # ========================================================
    # SOURCE FINGERPRINTS
    # ========================================================

    left = (
        _require_dataset(
            datasets=
                datasets,

            dataset_id=
                approved_join.left_dataset_id,
        )
    )

    right = (
        _require_dataset(
            datasets=
                datasets,

            dataset_id=
                approved_join.right_dataset_id,
        )
    )

    left_fingerprint = (
        _dataframe_fingerprint(
            left
        )
    )

    right_fingerprint = (
        _dataframe_fingerprint(
            right
        )
    )

    if (
        left_fingerprint
        ==
        execution_step.left_fingerprint
    ):
        checks.append(
            _passed_check(
                check_id=(
                    f"{request_id}:left-fingerprint"
                ),

                request_id=
                    request_id,

                message=(
                    "Left source still matches "
                    "the executor input fingerprint."
                ),
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=(
                    f"{request_id}:left-fingerprint"
                ),

                request_id=
                    request_id,

                message=(
                    "Left source differs from "
                    "the dataset used by executor."
                ),
            )
        )

    if (
        right_fingerprint
        ==
        execution_step.right_fingerprint
    ):
        checks.append(
            _passed_check(
                check_id=(
                    f"{request_id}:right-fingerprint"
                ),

                request_id=
                    request_id,

                message=(
                    "Right source still matches "
                    "the executor input fingerprint."
                ),
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=(
                    f"{request_id}:right-fingerprint"
                ),

                request_id=
                    request_id,

                message=(
                    "Right source differs from "
                    "the dataset used by executor."
                ),
            )
        )

    # ========================================================
    # OUTPUT FINGERPRINT
    # ========================================================

    observed_output_fingerprint = (
        _dataframe_fingerprint(
            output
        )
    )

    if (
        execution_step.output_fingerprint
        is not None
        and
        observed_output_fingerprint
        ==
        execution_step.output_fingerprint
    ):
        checks.append(
            _passed_check(
                check_id=(
                    f"{request_id}:output-fingerprint"
                ),

                request_id=
                    request_id,

                message=(
                    "Joined dataset fingerprint "
                    "matches the executor report."
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
                    "Joined dataset fingerprint "
                    "does not match executor report."
                ),
            )
        )

    # ========================================================
    # INDEPENDENT RECOMPUTATION
    # ========================================================

    try:
        expected_output = (
            _recompute_join_output(
                datasets=
                    datasets,

                join=
                    approved_join,
            )
        )

    except Exception as exc:
        checks.append(
            _failed_check(
                check_id=(
                    f"{request_id}:independent-recompute"
                ),

                request_id=
                    request_id,

                message=(
                    "Independent join recomputation "
                    "failed."
                ),

                details={
                    "error":
                        str(
                            exc
                        ),
                },
            )
        )

    else:
        if (
            _dataframes_equal(
                output,
                expected_output,
            )
        ):
            checks.append(
                _passed_check(
                    check_id=(
                        f"{request_id}:independent-recompute"
                    ),

                    request_id=
                        request_id,

                    message=(
                        "Joined dataset exactly matches "
                        "an independent recomputation "
                        "of the approved join."
                    ),
                )
            )

        else:
            checks.append(
                _failed_check(
                    check_id=(
                        f"{request_id}:independent-recompute"
                    ),

                    request_id=
                        request_id,

                    message=(
                        "Joined dataset differs from "
                        "independent recomputation."
                    ),
                )
            )

    # ========================================================
    # DUPLICATE COLUMNS
    # ========================================================

    duplicate_columns = (
        output.columns[
            output.columns
            .duplicated()
        ]
        .tolist()
    )

    if not (
        duplicate_columns
    ):
        checks.append(
            _passed_check(
                check_id=(
                    f"{request_id}:duplicate-columns"
                ),

                request_id=
                    request_id,

                message=(
                    "Joined dataset contains no "
                    "duplicate column names."
                ),
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=(
                    f"{request_id}:duplicate-columns"
                ),

                request_id=
                    request_id,

                message=(
                    "Joined dataset contains duplicate "
                    "column names."
                ),

                details={
                    "columns":
                        duplicate_columns,
                },
            )
        )

    return (
        _build_step_validation(
            join=
                approved_join,

            checks=
                checks,
        )
    )


# ============================================================
# PUBLIC API
# ============================================================


def validate_join_execution(
    *,
    source_datasets: Dict[
        str,
        pd.DataFrame,
    ],
    approved_plan: ApprovedJoinPlan,
    execution_result: JoinExecutionResult,
) -> PostJoinValidationReport:
    """
    Independently validate Join Executor output.

    This function NEVER mutates data.

    It verifies:

    - approval readiness;
    - execution report step count;
    - output dataset count;
    - approved ↔ execution reconciliation;
    - source fingerprints;
    - detected cardinality;
    - predicted vs actual grain;
    - output fingerprints;
    - no output for rejected joins;
    - no duplicate output column names;
    - full independent recomputation of each executable join.

    Any FAILED check makes valid_for_downstream=False.
    """

    checks: List[
        JoinValidationCheck
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
                    "Approved join plan was "
                    "ready for execution."
                ),
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=
                    "plan-ready",

                message=(
                    "Approved join plan was not "
                    "ready for execution."
                ),
            )
        )

    # ========================================================
    # STEP COUNT
    # ========================================================

    if (
        len(
            execution_result
            .report
            .steps
        )
        ==
        len(
            approved_plan.joins
        )
    ):
        checks.append(
            _passed_check(
                check_id=
                    "execution-step-count",

                message=(
                    "Execution report contains one "
                    "record for every approved join."
                ),
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=
                    "execution-step-count",

                message=(
                    "Execution report join count does "
                    "not match the approved plan."
                ),
            )
        )

    # ========================================================
    # EXPECTED OUTPUT DATASETS
    # ========================================================

    expected_output_ids = {
        join.output_dataset_id

        for join
        in approved_plan.joins

        if join.executable
    }

    observed_output_ids = set(
        execution_result
        .joined_datasets
        .keys()
    )

    if (
        expected_output_ids
        ==
        observed_output_ids
        and
        len(
            observed_output_ids
        )
        ==
        execution_result
        .report
        .output_dataset_count
    ):
        checks.append(
            _passed_check(
                check_id=
                    "output-dataset-set",

                message=(
                    "Join output datasets exactly "
                    "match the executable approved plan."
                ),
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=
                    "output-dataset-set",

                message=(
                    "Join output dataset set does not "
                    "match the executable approved plan."
                ),

                details={
                    "expected":
                        sorted(
                            expected_output_ids
                        ),

                    "observed":
                        sorted(
                            observed_output_ids
                        ),
                },
            )
        )

    # ========================================================
    # EXECUTION INDEX
    # ========================================================

    execution_index = (
        _execution_step_index(
            execution_result
        )
    )

    join_validations: List[
        JoinStepValidation
    ] = []

    # ========================================================
    # APPROVED JOIN RECONCILIATION
    # ========================================================

    for join in (
        approved_plan.joins
    ):
        execution_step = (
            execution_index.get(
                join.request_id
            )
        )

        if (
            execution_step is None
        ):
            join_validations.append(
                _build_step_validation(
                    join=
                        join,

                    checks=[
                        _failed_check(
                            check_id=(
                                f"{join.request_id}:"
                                "missing-execution-step"
                            ),

                            request_id=
                                join.request_id,

                            message=(
                                "Approved join is missing "
                                "from execution report."
                            ),
                        )
                    ],
                )
            )

            continue

        join_validations.append(
            _validate_join(
                datasets=
                    source_datasets,

                approved_join=
                    join,

                execution_result=
                    execution_result,

                execution_step=
                    execution_step,
            )
        )

    # ========================================================
    # EXTRA EXECUTION STEPS
    # ========================================================

    approved_request_ids = {
        join.request_id

        for join
        in approved_plan.joins
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
                    "No unapproved joins appear "
                    "in the execution report."
                ),
            )
        )

    else:
        checks.append(
            _failed_check(
                check_id=
                    "no-extra-execution-steps",

                message=(
                    "Execution contains joins absent "
                    "from approved plan."
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

    for validation in (
        join_validations
    ):
        all_checks.extend(
            validation.checks
        )

    passed_count = sum(
        check.status
        ==
        JoinValidationStatus.PASSED

        for check
        in all_checks
    )

    failed_count = sum(
        check.status
        ==
        JoinValidationStatus.FAILED

        for check
        in all_checks
    )

    warning_count = sum(
        check.status
        ==
        JoinValidationStatus.WARNING

        for check
        in all_checks
    )

    valid_for_downstream = (
        failed_count
        ==
        0
    )

    return (
        PostJoinValidationReport(
            status=(
                "passed"
                if valid_for_downstream
                else
                "failed"
            ),

            valid_for_downstream=
                valid_for_downstream,

            total_join_count=
                len(
                    approved_plan.joins
                ),

            validated_join_count=
                len(
                    join_validations
                ),

            passed_check_count=
                passed_count,

            failed_check_count=
                failed_count,

            warning_count=
                warning_count,

            output_dataset_count=
                len(
                    execution_result
                    .joined_datasets
                ),

            join_validations=
                join_validations,

            checks=
                checks,

            notes=[
                (
                    "Post-join Validation v0.1 "
                    "never mutates datasets."
                ),

                (
                    "Every executable join is "
                    "independently recomputed from "
                    "the original source datasets."
                ),

                (
                    "Missing join keys retain "
                    "DataLens null-safe semantics "
                    "and never match each other."
                ),

                (
                    "Actual output grain must match "
                    "the planner prediction exactly."
                ),

                (
                    "Source and output fingerprints "
                    "are reconciled with the executor "
                    "audit report."
                ),

                (
                    "Rejected joins must produce "
                    "no output dataset."
                ),

                (
                    "Any failed check sets "
                    "valid_for_downstream=False."
                ),
            ],

            rule_version=
                POST_JOIN_VALIDATION_RULE_VERSION,
        )
    )