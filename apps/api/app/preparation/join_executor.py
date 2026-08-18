from __future__ import annotations

from dataclasses import dataclass

from enum import Enum

import hashlib
import json

from typing import (
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
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
    JoinDiagnostics,
    JoinIntent,
    JoinKeyPair,
    JoinPlanningStatus,
    JoinType,
)

from app.preparation.join_planner import (
    plan_joins,
)


# ============================================================
# VERSION
# ============================================================


JOIN_EXECUTOR_RULE_VERSION = (
    "join_executor_v0.1"
)


# ============================================================
# ERROR
# ============================================================


class JoinExecutionError(
    RuntimeError,
):
    """
    Raised when an approved join cannot be executed safely.

    Execution is transactional:

    - source DataFrames are never mutated;
    - partial joined datasets are never returned after failure.
    """


# ============================================================
# EXECUTION STATUS
# ============================================================


class JoinExecutionStatus(
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
# PREFLIGHT
# ============================================================


class JoinPreflightResult(
    BaseModel,
):
    request_id: str

    passed: bool

    detected_cardinality: JoinCardinality

    predicted_output_row_count: int

    approved_predicted_output_row_count: int

    diagnostics: JoinDiagnostics

    mismatches: List[
        str
    ] = Field(
        default_factory=list
    )


# ============================================================
# STEP EXECUTION
# ============================================================


class JoinStepExecution(
    BaseModel,
):
    order: int

    request_id: str

    authorization_status: JoinAuthorizationStatus

    status: JoinExecutionStatus

    left_dataset_id: str

    right_dataset_id: str

    output_dataset_id: str

    join_type: JoinType

    detected_cardinality: JoinCardinality

    predicted_output_row_count: int

    actual_output_row_count: int

    left_row_count: int

    right_row_count: int

    left_fingerprint: str

    right_fingerprint: str

    output_fingerprint: Optional[
        str
    ] = None

    preflight_passed: bool

    rationale: str


# ============================================================
# REPORT
# ============================================================


class JoinExecutionReport(
    BaseModel,
):
    status: Literal[
        "success"
    ] = (
        "success"
    )

    total_join_count: int

    executable_join_count: int

    applied_count: int

    skipped_count: int

    output_dataset_count: int

    output_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    steps: List[
        JoinStepExecution
    ] = Field(
        default_factory=list
    )

    notes: List[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        JOIN_EXECUTOR_RULE_VERSION
    )


# ============================================================
# RESULT
# ============================================================


@dataclass(
    frozen=True
)
class JoinExecutionResult:
    joined_datasets: Dict[
        str,
        pd.DataFrame,
    ]

    report: JoinExecutionReport


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
# DATASET
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
        raise JoinExecutionError(
            (
                "Join execution dataset does "
                "not exist: "
                f"{dataset_id}"
            )
        )

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise JoinExecutionError(
            (
                "Join execution source is not "
                "a pandas DataFrame: "
                f"{dataset_id}"
            )
        )

    if dataframe.empty:
        raise JoinExecutionError(
            (
                "Join execution source cannot "
                "be empty: "
                f"{dataset_id}"
            )
        )

    return dataframe


# ============================================================
# PREFLIGHT INTENT
# ============================================================


def _rebuild_join_intent(
    join: ApprovedJoin,
) -> JoinIntent:
    """
    Rebuild exactly the structured JoinIntent represented by
    the approved join.

    No parameter is changed by the executor.
    """

    return (
        JoinIntent(
            request_id=
                join.request_id,

            left_dataset_id=
                join.left_dataset_id,

            left_dataset_filename=
                join.left_dataset_filename,

            right_dataset_id=
                join.right_dataset_id,

            right_dataset_filename=
                join.right_dataset_filename,

            join_type=
                join.join_type,

            keys=[
                JoinKeyPair(
                    left_column=
                        key.left_column,

                    right_column=
                        key.right_column,
                )

                for key
                in join.keys
            ],

            expected_cardinality=
                join.expected_cardinality,

            output_dataset_id=
                join.output_dataset_id,

            output_dataset_filename=
                join.output_dataset_filename,

            left_suffix=
                join.left_suffix,

            right_suffix=
                join.right_suffix,
        )
    )


# ============================================================
# PREFLIGHT COMPARISON
# ============================================================


def _run_preflight(
    *,
    datasets: Dict[
        str,
        pd.DataFrame,
    ],
    join: ApprovedJoin,
) -> JoinPreflightResult:
    left = (
        _require_dataset(
            datasets=
                datasets,

            dataset_id=
                join.left_dataset_id,
        )
    )

    right = (
        _require_dataset(
            datasets=
                datasets,

            dataset_id=
                join.right_dataset_id,
        )
    )

    current_plan = (
        plan_joins(
            datasets={
                join.left_dataset_id:
                    left,

                join.right_dataset_id:
                    right,
            },

            intents=[
                _rebuild_join_intent(
                    join
                )
            ],
        )
    )

    current = (
        current_plan.joins[
            0
        ]
    )

    mismatches: List[
        str
    ] = []

    # ========================================================
    # BLOCKED STATE
    # ========================================================

    if (
        current.status
        ==
        JoinPlanningStatus.BLOCKED
    ):
        mismatches.append(
            (
                "Current join state is BLOCKED "
                "by Join Planner."
            )
        )

    # ========================================================
    # CARDINALITY
    # ========================================================

    if (
        current.detected_cardinality
        !=
        join.detected_cardinality
    ):
        mismatches.append(
            (
                "Detected cardinality changed: "
                f"approved="
                f"{join.detected_cardinality.value}, "
                f"current="
                f"{current.detected_cardinality.value}."
            )
        )

    # ========================================================
    # EXPECTATION
    # ========================================================

    if (
        current
        .cardinality_matches_expectation
        !=
        join
        .cardinality_matches_expectation
    ):
        mismatches.append(
            (
                "Cardinality expectation result "
                "changed since planning."
            )
        )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    approved_diagnostics = (
        join
        .diagnostics
        .model_dump()
    )

    current_diagnostics = (
        current
        .diagnostics
        .model_dump()
    )

    for field_name in (
        approved_diagnostics
    ):
        approved_value = (
            approved_diagnostics[
                field_name
            ]
        )

        current_value = (
            current_diagnostics[
                field_name
            ]
        )

        if (
            approved_value
            !=
            current_value
        ):
            mismatches.append(
                (
                    "Join diagnostic changed "
                    f"for '{field_name}': "
                    f"approved={approved_value}, "
                    f"current={current_value}."
                )
            )

    # ========================================================
    # RISK
    # ========================================================

    if (
        current.risk
        !=
        join.risk
    ):
        mismatches.append(
            (
                "Join risk changed: "
                f"approved={join.risk.value}, "
                f"current={current.risk.value}."
            )
        )

    return (
        JoinPreflightResult(
            request_id=
                join.request_id,

            passed=(
                len(
                    mismatches
                )
                ==
                0
            ),

            detected_cardinality=
                current.detected_cardinality,

            predicted_output_row_count=(
                current
                .diagnostics
                .predicted_output_row_count
            ),

            approved_predicted_output_row_count=(
                join
                .diagnostics
                .predicted_output_row_count
            ),

            diagnostics=
                current
                .diagnostics
                .model_copy(
                    deep=True
                ),

            mismatches=
                mismatches,
        )
    )


# ============================================================
# NULL-SAFE COMPOSITE JOIN KEY
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
    pandas.merge normally allows NaN/NaN join-key matches.

    DataLens Planner v0.1 explicitly treats missing join keys
    as unmatched.

    Therefore executor uses a temporary composite key:

        valid row:
            ("__DATALENS_VALUE__", (<key values>))

        missing-key row:
            ("__DATALENS_NULL__", side, row_number)

    Missing left/right keys can therefore never match.
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


# ============================================================
# INTERNAL COLUMN NAME
# ============================================================


def _temporary_join_column(
    *,
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> str:
    base = (
        "__datalens_internal_join_key__"
    )

    candidate = (
        base
    )

    counter = 1

    all_columns = (
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
        all_columns
    ):
        candidate = (
            f"{base}{counter}"
        )

        counter += 1

    return candidate


# ============================================================
# CARDINALITY → PANDAS VALIDATE
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

    raise JoinExecutionError(
        (
            "MANY_TO_MANY joins are not "
            "executable in Join Executor v0.1."
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
    """
    Because the executor joins through an internal composite
    key, pandas initially treats same-name original join keys
    as overlapping ordinary columns.

    Example:

        customer_id_left
        customer_id_right

    DataLens restores the intuitive output:

        customer_id

    using left value first, right value second.
    """

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

        # ----------------------------------------------------
        # With our temporary merge key both versions should
        # exist when suffixes are valid.
        # ----------------------------------------------------

        if (
            left_name
            not in
            result.columns
            or
            right_name
            not in
            result.columns
        ):
            raise JoinExecutionError(
                (
                    "Executor could not reconcile "
                    "same-name join key columns: "
                    f"{base_name}"
                )
            )

        left_position = (
            result.columns
            .get_loc(
                left_name
            )
        )

        right_position = (
            result.columns
            .get_loc(
                right_name
            )
        )

        insert_position = min(
            int(
                left_position
            ),
            int(
                right_position
            ),
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

    return result


# ============================================================
# EXECUTE SINGLE JOIN
# ============================================================


def _execute_join(
    *,
    datasets: Dict[
        str,
        pd.DataFrame,
    ],
    join: ApprovedJoin,
    preflight: JoinPreflightResult,
) -> pd.DataFrame:
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

    # ========================================================
    # NEVER MUTATE SOURCES
    # ========================================================

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

    # ========================================================
    # SUFFIX SAFETY
    # ========================================================

    if (
        join.left_suffix
        ==
        join.right_suffix
    ):
        raise JoinExecutionError(
            (
                "Join suffixes must be distinct "
                "at execution time."
            )
        )

    # ========================================================
    # INTERNAL KEY
    # ========================================================

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

    # ========================================================
    # MERGE
    # ========================================================

    try:
        output = (
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

                sort=
                    False,

                copy=
                    True,

                validate=
                    _pandas_validate_mode(
                        preflight
                        .detected_cardinality
                    ),
            )
        )

    except Exception as exc:
        raise JoinExecutionError(
            (
                "Approved join execution failed: "
                f"{exc}"
            )
        ) from exc

    # ========================================================
    # REMOVE INTERNAL KEY
    # ========================================================

    if (
        temporary_key
        not in
        output.columns
    ):
        raise JoinExecutionError(
            (
                "Internal join key disappeared "
                "unexpectedly during merge."
            )
        )

    output = (
        output.drop(
            columns=[
                temporary_key
            ]
        )
    )

    # ========================================================
    # RESTORE SAME-NAME KEYS
    # ========================================================

    output = (
        _coalesce_same_named_keys(
            dataframe=
                output,

            join=
                join,
        )
    )

    # ========================================================
    # NO DUPLICATE OUTPUT COLUMN NAMES
    # ========================================================

    duplicate_columns = (
        output.columns[
            output.columns
            .duplicated()
        ]
        .tolist()
    )

    if (
        duplicate_columns
    ):
        raise JoinExecutionError(
            (
                "Join produced duplicate output "
                "column names: "
                f"{duplicate_columns}"
            )
        )

    # ========================================================
    # EXACT GRAIN CHECK
    # ========================================================

    actual_row_count = int(
        len(
            output
        )
    )

    expected_row_count = (
        preflight
        .predicted_output_row_count
    )

    if (
        actual_row_count
        !=
        expected_row_count
    ):
        raise JoinExecutionError(
            (
                "Join output row count does not "
                "match preflight prediction. "
                f"Expected {expected_row_count}, "
                f"observed {actual_row_count}."
            )
        )

    return (
        output
    )


# ============================================================
# PUBLIC API
# ============================================================


def execute_join_plan(
    *,
    datasets: Dict[
        str,
        pd.DataFrame,
    ],
    approved_plan: ApprovedJoinPlan,
) -> JoinExecutionResult:
    """
    Execute an ApprovedJoinPlan transactionally.

    Safety guarantees:

    - only ApprovedJoinPlan is accepted;
    - plan must be ready_for_execution=True;
    - PENDING / DEFERRED plans are rejected;
    - only APPROVED joins may execute;
    - REJECTED joins are skipped;
    - source DataFrames are never mutated;
    - Join Planner diagnostics are recomputed immediately
      before execution;
    - cardinality changes refuse execution;
    - match-rate changes refuse execution;
    - orphan-count changes refuse execution;
    - predicted row-count changes refuse execution;
    - MANY_TO_MANY is never executed in v0.1;
    - pandas merge cardinality validation is enabled;
    - missing keys never match other missing keys;
    - actual output grain must equal predicted output grain;
    - output dataset IDs may never overwrite input datasets;
    - if any executable join fails, no partial output result
      is returned.
    """

    if not isinstance(
        datasets,
        dict,
    ):
        raise TypeError(
            (
                "Join Executor requires a "
                "dataset mapping."
            )
        )

    if not (
        datasets
    ):
        raise ValueError(
            (
                "Join Executor requires at "
                "least one dataset."
            )
        )

    if not (
        approved_plan
        .ready_for_execution
    ):
        raise JoinExecutionError(
            (
                "ApprovedJoinPlan is not "
                "ready for execution."
            )
        )

    executable_joins = [
        join

        for join
        in approved_plan.joins

        if join.executable
    ]

    if (
        len(
            executable_joins
        )
        !=
        approved_plan
        .executable_join_count
    ):
        raise JoinExecutionError(
            (
                "ApprovedJoinPlan contains an "
                "inconsistent executable_join_count."
            )
        )

    # ========================================================
    # SOURCE SNAPSHOTS
    # ========================================================

    source_fingerprints = {
        dataset_id:
            _dataframe_fingerprint(
                dataframe
            )

        for (
            dataset_id,
            dataframe,
        )
        in datasets.items()
    }

    # ========================================================
    # TRANSACTIONAL OUTPUT
    # ========================================================

    joined_datasets: Dict[
        str,
        pd.DataFrame,
    ] = {}

    step_reports: List[
        JoinStepExecution
    ] = []

    applied_count = 0

    skipped_count = 0

    # ========================================================
    # PROCESS APPROVED PLAN
    # ========================================================

    for order, join in enumerate(
        approved_plan.joins,
        start=1,
    ):
        left = (
            _require_dataset(
                datasets=
                    datasets,

                dataset_id=
                    join.left_dataset_id,
            )
        )

        right = (
            _require_dataset(
                datasets=
                    datasets,

                dataset_id=
                    join.right_dataset_id,
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

        # ====================================================
        # RESOLVED NON-EXECUTABLE JOIN
        # ====================================================

        if not (
            join.executable
        ):
            if not (
                join.resolved
            ):
                raise JoinExecutionError(
                    (
                        "Unresolved join reached "
                        "Join Executor: "
                        f"{join.request_id}"
                    )
                )

            if (
                join.authorization_status
                !=
                JoinAuthorizationStatus.REJECTED
            ):
                raise JoinExecutionError(
                    (
                        "Resolved non-executable join "
                        "must have REJECTED status: "
                        f"{join.request_id}"
                    )
                )

            skipped_count += 1

            step_reports.append(
                JoinStepExecution(
                    order=
                        order,

                    request_id=
                        join.request_id,

                    authorization_status=
                        join.authorization_status,

                    status=
                        JoinExecutionStatus.SKIPPED,

                    left_dataset_id=
                        join.left_dataset_id,

                    right_dataset_id=
                        join.right_dataset_id,

                    output_dataset_id=
                        join.output_dataset_id,

                    join_type=
                        join.join_type,

                    detected_cardinality=
                        join.detected_cardinality,

                    predicted_output_row_count=(
                        join
                        .diagnostics
                        .predicted_output_row_count
                    ),

                    actual_output_row_count=
                        0,

                    left_row_count=
                        int(
                            len(
                                left
                            )
                        ),

                    right_row_count=
                        int(
                            len(
                                right
                            )
                        ),

                    left_fingerprint=
                        left_fingerprint,

                    right_fingerprint=
                        right_fingerprint,

                    output_fingerprint=None,

                    preflight_passed=
                        False,

                    rationale=(
                        "Join was explicitly rejected "
                        "and was therefore skipped."
                    ),
                )
            )

            continue

        # ====================================================
        # AUTHORIZATION
        # ====================================================

        if (
            join.authorization_status
            !=
            JoinAuthorizationStatus.APPROVED
        ):
            raise JoinExecutionError(
                (
                    "Executable join does not have "
                    "APPROVED authorization: "
                    f"{join.request_id}"
                )
            )

        # ====================================================
        # OUTPUT ID COLLISION
        # ====================================================

        if (
            join.output_dataset_id
            in
            datasets
        ):
            raise JoinExecutionError(
                (
                    "Join output dataset_id would "
                    "overwrite an input dataset: "
                    f"{join.output_dataset_id}"
                )
            )

        if (
            join.output_dataset_id
            in
            joined_datasets
        ):
            raise JoinExecutionError(
                (
                    "Duplicate join output dataset_id "
                    "during execution: "
                    f"{join.output_dataset_id}"
                )
            )

        # ====================================================
        # PREFLIGHT
        # ====================================================

        preflight = (
            _run_preflight(
                datasets=
                    datasets,

                join=
                    join,
            )
        )

        if not (
            preflight.passed
        ):
            raise JoinExecutionError(
                (
                    "Join preflight no longer matches "
                    "the approved plan for "
                    f"'{join.request_id}'. "
                    "Mismatches: "
                    f"{preflight.mismatches}"
                )
            )

        # ====================================================
        # MANY TO MANY SECONDARY GUARD
        # ====================================================

        if (
            preflight.detected_cardinality
            ==
            JoinCardinality.MANY_TO_MANY
        ):
            raise JoinExecutionError(
                (
                    "MANY_TO_MANY joins are blocked "
                    "in Join Executor v0.1."
                )
            )

        # ====================================================
        # EXECUTE
        # ====================================================

        output = (
            _execute_join(
                datasets=
                    datasets,

                join=
                    join,

                preflight=
                    preflight,
            )
        )

        output_fingerprint = (
            _dataframe_fingerprint(
                output
            )
        )

        joined_datasets[
            join.output_dataset_id
        ] = (
            output.copy(
                deep=True
            )
        )

        applied_count += 1

        step_reports.append(
            JoinStepExecution(
                order=
                    order,

                request_id=
                    join.request_id,

                authorization_status=
                    join.authorization_status,

                status=
                    JoinExecutionStatus.APPLIED,

                left_dataset_id=
                    join.left_dataset_id,

                right_dataset_id=
                    join.right_dataset_id,

                output_dataset_id=
                    join.output_dataset_id,

                join_type=
                    join.join_type,

                detected_cardinality=
                    preflight
                    .detected_cardinality,

                predicted_output_row_count=(
                    preflight
                    .predicted_output_row_count
                ),

                actual_output_row_count=
                    int(
                        len(
                            output
                        )
                    ),

                left_row_count=
                    int(
                        len(
                            left
                        )
                    ),

                right_row_count=
                    int(
                        len(
                            right
                        )
                    ),

                left_fingerprint=
                    left_fingerprint,

                right_fingerprint=
                    right_fingerprint,

                output_fingerprint=
                    output_fingerprint,

                preflight_passed=
                    True,

                rationale=(
                    "Approved join passed deterministic "
                    "preflight and produced exactly the "
                    "predicted output grain."
                ),
            )
        )

    # ========================================================
    # SOURCE IMMUTABILITY
    # ========================================================

    for (
        dataset_id,
        dataframe,
    ) in datasets.items():
        current_fingerprint = (
            _dataframe_fingerprint(
                dataframe
            )
        )

        if (
            current_fingerprint
            !=
            source_fingerprints[
                dataset_id
            ]
        ):
            raise JoinExecutionError(
                (
                    "Join Executor unexpectedly "
                    "mutated source dataset: "
                    f"{dataset_id}"
                )
            )

    # ========================================================
    # REPORT
    # ========================================================

    report = (
        JoinExecutionReport(
            total_join_count=
                len(
                    approved_plan.joins
                ),

            executable_join_count=
                approved_plan
                .executable_join_count,

            applied_count=
                applied_count,

            skipped_count=
                skipped_count,

            output_dataset_count=
                len(
                    joined_datasets
                ),

            output_dataset_ids=
                list(
                    joined_datasets.keys()
                ),

            steps=
                step_reports,

            notes=[
                (
                    "Join Executor v0.1 never "
                    "mutates source datasets."
                ),

                (
                    "Every executable join is "
                    "replanned immediately before "
                    "execution as a deterministic "
                    "preflight."
                ),

                (
                    "Join-relevant changes in "
                    "cardinality, match rates, "
                    "orphan counts, schema collisions "
                    "or predicted output grain refuse "
                    "execution."
                ),

                (
                    "MANY_TO_MANY joins remain "
                    "non-executable in v0.1."
                ),

                (
                    "pandas merge cardinality validation "
                    "is enabled as a second guardrail."
                ),

                (
                    "Missing join keys are explicitly "
                    "prevented from matching other "
                    "missing join keys."
                ),

                (
                    "Actual output row count must equal "
                    "the planner preflight prediction."
                ),

                (
                    "Execution is transactional. "
                    "A failing join returns no partial "
                    "joined result."
                ),
            ],
        )
    )

    return (
        JoinExecutionResult(
            joined_datasets={
                dataset_id:
                    dataframe.copy(
                        deep=True
                    )

                for (
                    dataset_id,
                    dataframe,
                )
                in joined_datasets.items()
            },

            report=
                report,
        )
    )