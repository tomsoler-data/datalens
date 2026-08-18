from __future__ import annotations

from collections import Counter

from typing import (
    Dict,
    Iterable,
    List,
    Set,
    Tuple,
)

import pandas as pd

from app.preparation.join_contracts import (
    JoinCardinality,
    JoinDiagnostics,
    JoinIntent,
    JoinPlan,
    JoinPlanningStatus,
    JoinRisk,
    JoinType,
    PlannedJoin,
)


# ============================================================
# VERSION
# ============================================================


JOIN_PLANNER_RULE_VERSION = (
    "join_planner_v0.1"
)


# ============================================================
# BASIC HELPERS
# ============================================================


def _safe_ratio(
    numerator: int,
    denominator: int,
) -> float:
    if (
        denominator
        <=
        0
    ):
        return 0.0

    return float(
        numerator
        /
        denominator
    )


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
                "Unknown join dataset_id: "
                f"{dataset_id}"
            )
        )

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            (
                "Join dataset must be a "
                "pandas DataFrame: "
                f"{dataset_id}"
            )
        )

    if dataframe.empty:
        raise ValueError(
            (
                "Join dataset cannot be empty: "
                f"{dataset_id}"
            )
        )

    return dataframe


def _require_column(
    *,
    dataframe: pd.DataFrame,
    column: str,
    dataset_id: str,
) -> None:
    if (
        column
        not in
        dataframe.columns
    ):
        raise ValueError(
            (
                "Unknown join key column "
                f"'{column}' in dataset "
                f"'{dataset_id}'."
            )
        )


# ============================================================
# KEY EXTRACTION
# ============================================================


def _build_key_tuples(
    *,
    dataframe: pd.DataFrame,
    columns: List[
        str
    ],
) -> Tuple[
    List[
        Tuple
    ],
    int,
]:
    """
    Build tuples only for rows where every key component
    is non-missing.

    Missing join keys are explicitly treated as unusable for
    matching in DataLens v0.1.
    """

    keys: List[
        Tuple
    ] = []

    null_count = 0

    for values in (
        dataframe[
            columns
        ]
        .itertuples(
            index=False,
            name=None,
        )
    ):
        if any(
            pd.isna(
                value
            )
            for value
            in values
        ):
            null_count += 1

            continue

        keys.append(
            tuple(
                values
            )
        )

    return (
        keys,
        null_count,
    )


# ============================================================
# CARDINALITY
# ============================================================


def _detect_cardinality(
    *,
    left_counts: Counter,
    right_counts: Counter,
) -> JoinCardinality:
    left_unique = all(
        count
        ==
        1

        for count
        in left_counts.values()
    )

    right_unique = all(
        count
        ==
        1

        for count
        in right_counts.values()
    )

    if (
        left_unique
        and
        right_unique
    ):
        return (
            JoinCardinality
            .ONE_TO_ONE
        )

    if (
        left_unique
        and
        not right_unique
    ):
        return (
            JoinCardinality
            .ONE_TO_MANY
        )

    if (
        not left_unique
        and
        right_unique
    ):
        return (
            JoinCardinality
            .MANY_TO_ONE
        )

    return (
        JoinCardinality
        .MANY_TO_MANY
    )


# ============================================================
# ROW DIAGNOSTICS
# ============================================================


def _duplicated_key_row_count(
    counts: Counter,
) -> int:
    return sum(
        count

        for count
        in counts.values()

        if (
            count
            >
            1
        )
    )


def _matching_statistics(
    *,
    left_counts: Counter,
    right_counts: Counter,
) -> Dict[
    str,
    int,
]:
    left_keys = set(
        left_counts.keys()
    )

    right_keys = set(
        right_counts.keys()
    )

    matched_keys = (
        left_keys
        &
        right_keys
    )

    left_matched_rows = sum(
        left_counts[
            key
        ]

        for key
        in matched_keys
    )

    right_matched_rows = sum(
        right_counts[
            key
        ]

        for key
        in matched_keys
    )

    left_unmatched_rows = sum(
        left_counts[
            key
        ]

        for key
        in (
            left_keys
            -
            right_keys
        )
    )

    right_unmatched_rows = sum(
        right_counts[
            key
        ]

        for key
        in (
            right_keys
            -
            left_keys
        )
    )

    matched_output_rows = sum(
        left_counts[
            key
        ]
        *
        right_counts[
            key
        ]

        for key
        in matched_keys
    )

    return {
        "matched_distinct_key_count":
            len(
                matched_keys
            ),

        "left_matched_rows":
            left_matched_rows,

        "right_matched_rows":
            right_matched_rows,

        "left_unmatched_rows":
            left_unmatched_rows,

        "right_unmatched_rows":
            right_unmatched_rows,

        "matched_output_rows":
            matched_output_rows,
    }


# ============================================================
# PREDICT OUTPUT SIZE
# ============================================================


def _predict_output_rows(
    *,
    join_type: JoinType,
    matched_output_rows: int,
    left_unmatched_rows: int,
    right_unmatched_rows: int,
    left_null_key_rows: int,
    right_null_key_rows: int,
) -> int:
    if (
        join_type
        ==
        JoinType.INNER
    ):
        return (
            matched_output_rows
        )

    if (
        join_type
        ==
        JoinType.LEFT
    ):
        return (
            matched_output_rows
            +
            left_unmatched_rows
            +
            left_null_key_rows
        )

    if (
        join_type
        ==
        JoinType.RIGHT
    ):
        return (
            matched_output_rows
            +
            right_unmatched_rows
            +
            right_null_key_rows
        )

    if (
        join_type
        ==
        JoinType.OUTER
    ):
        return (
            matched_output_rows
            +
            left_unmatched_rows
            +
            right_unmatched_rows
            +
            left_null_key_rows
            +
            right_null_key_rows
        )

    raise ValueError(
        (
            "Unsupported join type: "
            f"{join_type}"
        )
    )


# ============================================================
# COLUMN COLLISIONS
# ============================================================


def _non_key_overlaps(
    *,
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_keys: List[
        str
    ],
    right_keys: List[
        str
    ],
) -> List[
    str
]:
    left_non_keys = (
        set(
            str(
                column
            )
            for column
            in left.columns
        )
        -
        set(
            left_keys
        )
    )

    right_non_keys = (
        set(
            str(
                column
            )
            for column
            in right.columns
        )
        -
        set(
            right_keys
        )
    )

    return sorted(
        left_non_keys
        &
        right_non_keys
    )


# ============================================================
# RISK
# ============================================================


def _determine_risk(
    *,
    cardinality: JoinCardinality,
    diagnostics: JoinDiagnostics,
    expected_cardinality: JoinCardinality,
    expected_supplied: bool,
) -> JoinRisk:
    if (
        cardinality
        ==
        JoinCardinality
        .MANY_TO_MANY
    ):
        return (
            JoinRisk.CRITICAL
        )

    if (
        expected_supplied
        and
        cardinality
        !=
        expected_cardinality
    ):
        return (
            JoinRisk.CRITICAL
        )

    if (
        diagnostics
        .matched_distinct_key_count
        ==
        0
    ):
        return (
            JoinRisk.HIGH
        )

    if (
        diagnostics
        .predicted_row_multiplier_vs_left
        >
        3.0
    ):
        return (
            JoinRisk.HIGH
        )

    if (
        diagnostics.left_match_ratio
        <
        0.50
        or
        diagnostics.right_match_ratio
        <
        0.50
    ):
        return (
            JoinRisk.HIGH
        )

    if (
        diagnostics.left_null_key_rows
        >
        0
        or
        diagnostics.right_null_key_rows
        >
        0
    ):
        return (
            JoinRisk.MEDIUM
        )

    if (
        diagnostics
        .overlapping_non_key_columns
    ):
        return (
            JoinRisk.MEDIUM
        )

    if (
        cardinality
        in {
            JoinCardinality.ONE_TO_MANY,
            JoinCardinality.MANY_TO_ONE,
        }
    ):
        return (
            JoinRisk.MEDIUM
        )

    return (
        JoinRisk.LOW
    )


# ============================================================
# SINGLE JOIN
# ============================================================


def _plan_join(
    *,
    datasets: Dict[
        str,
        pd.DataFrame,
    ],
    intent: JoinIntent,
) -> PlannedJoin:
    if not (
        intent.request_id
        .strip()
    ):
        raise ValueError(
            "Join request_id cannot be empty."
        )

    if (
        intent.left_dataset_id
        ==
        intent.right_dataset_id
    ):
        raise ValueError(
            (
                "Self-joins are not supported "
                "in Join Planner v0.1."
            )
        )

    if not (
        intent.keys
    ):
        raise ValueError(
            (
                "Join requires at least "
                "one key pair."
            )
        )

    if not (
        intent.output_dataset_id
        .strip()
    ):
        raise ValueError(
            (
                "Join output_dataset_id "
                "cannot be empty."
            )
        )

    if (
        intent.output_dataset_id
        in {
            intent.left_dataset_id,
            intent.right_dataset_id,
        }
    ):
        raise ValueError(
            (
                "Join output must use a new "
                "dataset_id."
            )
        )

    if not (
        intent.output_dataset_filename
        .strip()
    ):
        raise ValueError(
            (
                "Join output filename "
                "cannot be empty."
            )
        )

    left = (
        _require_dataset(
            datasets=
                datasets,

            dataset_id=
                intent.left_dataset_id,
        )
    )

    right = (
        _require_dataset(
            datasets=
                datasets,

            dataset_id=
                intent.right_dataset_id,
        )
    )

    left_keys: List[
        str
    ] = []

    right_keys: List[
        str
    ] = []

    seen_pairs: Set[
        Tuple[
            str,
            str,
        ]
    ] = set()

    for key_pair in (
        intent.keys
    ):
        left_column = (
            key_pair.left_column
            .strip()
        )

        right_column = (
            key_pair.right_column
            .strip()
        )

        if not (
            left_column
            and
            right_column
        ):
            raise ValueError(
                (
                    "Join key columns cannot "
                    "be empty."
                )
            )

        pair = (
            left_column,
            right_column,
        )

        if (
            pair
            in
            seen_pairs
        ):
            raise ValueError(
                (
                    "Duplicate join key pair: "
                    f"{pair}"
                )
            )

        seen_pairs.add(
            pair
        )

        _require_column(
            dataframe=
                left,

            column=
                left_column,

            dataset_id=
                intent.left_dataset_id,
        )

        _require_column(
            dataframe=
                right,

            column=
                right_column,

            dataset_id=
                intent.right_dataset_id,
        )

        left_keys.append(
            left_column
        )

        right_keys.append(
            right_column
        )

    # ========================================================
    # KEY DATA
    # ========================================================

    (
        left_key_values,
        left_null_key_rows,
    ) = (
        _build_key_tuples(
            dataframe=
                left,

            columns=
                left_keys,
        )
    )

    (
        right_key_values,
        right_null_key_rows,
    ) = (
        _build_key_tuples(
            dataframe=
                right,

            columns=
                right_keys,
        )
    )

    if not (
        left_key_values
    ):
        raise ValueError(
            (
                "Left dataset contains no "
                "usable non-missing join keys."
            )
        )

    if not (
        right_key_values
    ):
        raise ValueError(
            (
                "Right dataset contains no "
                "usable non-missing join keys."
            )
        )

    left_counts = Counter(
        left_key_values
    )

    right_counts = Counter(
        right_key_values
    )

    cardinality = (
        _detect_cardinality(
            left_counts=
                left_counts,

            right_counts=
                right_counts,
        )
    )

    stats = (
        _matching_statistics(
            left_counts=
                left_counts,

            right_counts=
                right_counts,
        )
    )

    predicted_output_rows = (
        _predict_output_rows(
            join_type=
                intent.join_type,

            matched_output_rows=
                stats[
                    "matched_output_rows"
                ],

            left_unmatched_rows=
                stats[
                    "left_unmatched_rows"
                ],

            right_unmatched_rows=
                stats[
                    "right_unmatched_rows"
                ],

            left_null_key_rows=
                left_null_key_rows,

            right_null_key_rows=
                right_null_key_rows,
        )
    )

    overlaps = (
        _non_key_overlaps(
            left=
                left,

            right=
                right,

            left_keys=
                left_keys,

            right_keys=
                right_keys,
        )
    )

    if (
        overlaps
        and
        (
            not intent.left_suffix
            or
            not intent.right_suffix
            or
            intent.left_suffix
            ==
            intent.right_suffix
        )
    ):
        raise ValueError(
            (
                "Overlapping non-key columns "
                "require two distinct suffixes. "
                f"Columns: {overlaps}"
            )
        )

    diagnostics = (
        JoinDiagnostics(
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

            left_non_null_key_rows=
                len(
                    left_key_values
                ),

            right_non_null_key_rows=
                len(
                    right_key_values
                ),

            left_null_key_rows=
                left_null_key_rows,

            right_null_key_rows=
                right_null_key_rows,

            left_distinct_key_count=
                len(
                    left_counts
                ),

            right_distinct_key_count=
                len(
                    right_counts
                ),

            left_duplicate_key_rows=
                _duplicated_key_row_count(
                    left_counts
                ),

            right_duplicate_key_rows=
                _duplicated_key_row_count(
                    right_counts
                ),

            matched_distinct_key_count=
                stats[
                    "matched_distinct_key_count"
                ],

            left_matched_row_count=
                stats[
                    "left_matched_rows"
                ],

            right_matched_row_count=
                stats[
                    "right_matched_rows"
                ],

            left_unmatched_row_count=(
                stats[
                    "left_unmatched_rows"
                ]
                +
                left_null_key_rows
            ),

            right_unmatched_row_count=(
                stats[
                    "right_unmatched_rows"
                ]
                +
                right_null_key_rows
            ),

            left_match_ratio=
                _safe_ratio(
                    stats[
                        "left_matched_rows"
                    ],
                    len(
                        left
                    ),
                ),

            right_match_ratio=
                _safe_ratio(
                    stats[
                        "right_matched_rows"
                    ],
                    len(
                        right
                    ),
                ),

            predicted_output_row_count=
                predicted_output_rows,

            predicted_row_multiplier_vs_left=
                _safe_ratio(
                    predicted_output_rows,
                    len(
                        left
                    ),
                ),

            overlapping_non_key_columns=
                overlaps,
        )
    )

    # ========================================================
    # EXPECTED CARDINALITY
    # ========================================================

    expected_supplied = (
        intent.expected_cardinality
        is not None
    )

    if (
        expected_supplied
    ):
        cardinality_matches = (
            cardinality
            ==
            intent.expected_cardinality
        )

    else:
        cardinality_matches = None

    risk = (
        _determine_risk(
            cardinality=
                cardinality,

            diagnostics=
                diagnostics,

            expected_cardinality=(
                intent.expected_cardinality
                or
                cardinality
            ),

            expected_supplied=
                expected_supplied,
        )
    )

    warnings: List[
        str
    ] = []

    # ========================================================
    # BLOCKING RULES
    # ========================================================

    blocked = False

    if (
        cardinality
        ==
        JoinCardinality.MANY_TO_MANY
    ):
        blocked = True

        warnings.append(
            (
                "Many-to-many cardinality detected. "
                "Join Planner v0.1 blocks this join "
                "because it can multiply rows."
            )
        )

    if (
        expected_supplied
        and
        not cardinality_matches
    ):
        blocked = True

        warnings.append(
            (
                "Detected cardinality does not match "
                "the expected cardinality supplied "
                "by the transformation intent."
            )
        )

    if (
        diagnostics
        .matched_distinct_key_count
        ==
        0
    ):
        blocked = True

        warnings.append(
            (
                "No join keys match between the "
                "two datasets."
            )
        )

    # ========================================================
    # WARNINGS
    # ========================================================

    if (
        left_null_key_rows
        >
        0
    ):
        warnings.append(
            (
                f"{left_null_key_rows} left rows "
                "contain missing join keys."
            )
        )

    if (
        right_null_key_rows
        >
        0
    ):
        warnings.append(
            (
                f"{right_null_key_rows} right rows "
                "contain missing join keys."
            )
        )

    if (
        diagnostics.left_unmatched_row_count
        >
        0
    ):
        warnings.append(
            (
                f"{diagnostics.left_unmatched_row_count} "
                "left rows have no matching right key."
            )
        )

    if (
        diagnostics.right_unmatched_row_count
        >
        0
    ):
        warnings.append(
            (
                f"{diagnostics.right_unmatched_row_count} "
                "right rows have no matching left key."
            )
        )

    if overlaps:
        warnings.append(
            (
                "Non-key column collisions detected: "
                f"{overlaps}. Approved suffixes will "
                "be required during execution."
            )
        )

    if (
        diagnostics
        .predicted_row_multiplier_vs_left
        >
        2.0
    ):
        warnings.append(
            (
                "Predicted join output contains more "
                "than twice the number of left rows."
            )
        )

    if blocked:
        status = (
            JoinPlanningStatus
            .BLOCKED
        )

        rationale = (
            "The proposed join violates a deterministic "
            "join-safety rule and cannot proceed to "
            "execution in Join Planner v0.1."
        )

    else:
        status = (
            JoinPlanningStatus
            .REVIEW_REQUIRED
        )

        rationale = (
            "The join is structurally valid. "
            "Cardinality, matching rates, orphan rows "
            "and predicted output size were measured. "
            "Explicit human approval is required because "
            "joining datasets can alter analytical grain."
        )

    return (
        PlannedJoin(
            request_id=
                intent.request_id,

            left_dataset_id=
                intent.left_dataset_id,

            left_dataset_filename=
                intent.left_dataset_filename,

            right_dataset_id=
                intent.right_dataset_id,

            right_dataset_filename=
                intent.right_dataset_filename,

            join_type=
                intent.join_type,

            keys=
                list(
                    intent.keys
                ),

            detected_cardinality=
                cardinality,

            expected_cardinality=
                intent.expected_cardinality,

            cardinality_matches_expectation=
                cardinality_matches,

            output_dataset_id=
                intent.output_dataset_id,

            output_dataset_filename=
                intent.output_dataset_filename,

            left_suffix=
                intent.left_suffix,

            right_suffix=
                intent.right_suffix,

            status=
                status,

            risk=
                risk,

            diagnostics=
                diagnostics,

            requires_human_approval=
                True,

            executable=
                False,

            rationale=
                rationale,

            warnings=
                warnings,
        )
    )


# ============================================================
# PUBLIC API
# ============================================================


def plan_joins(
    *,
    datasets: Dict[
        str,
        pd.DataFrame,
    ],
    intents: List[
        JoinIntent
    ],
) -> JoinPlan:
    """
    Deterministically inspect proposed joins.

    Safety guarantees:

    - never mutates datasets;
    - never executes merge/join;
    - validates every key against the actual DataFrames;
    - measures missing join keys;
    - detects actual cardinality;
    - measures match rates;
    - identifies orphan rows;
    - predicts output row count;
    - identifies non-key column collisions;
    - blocks many-to-many joins by default;
    - blocks expected-cardinality mismatches;
    - blocks joins with zero matching keys;
    - all non-blocked joins still require human approval.
    """

    if not (
        datasets
    ):
        raise ValueError(
            (
                "Join Planner requires "
                "at least one dataset."
            )
        )

    if not (
        intents
    ):
        raise ValueError(
            (
                "Join Planner requires at "
                "least one JoinIntent."
            )
        )

    seen_request_ids: Set[
        str
    ] = set()

    seen_output_dataset_ids: Set[
        str
    ] = set(
        datasets.keys()
    )

    joins: List[
        PlannedJoin
    ] = []

    for intent in intents:
        if (
            intent.request_id
            in
            seen_request_ids
        ):
            raise ValueError(
                (
                    "Duplicate join request_id: "
                    f"{intent.request_id}"
                )
            )

        seen_request_ids.add(
            intent.request_id
        )

        if (
            intent.output_dataset_id
            in
            seen_output_dataset_ids
        ):
            raise ValueError(
                (
                    "Join output dataset_id "
                    "already exists or is reused: "
                    f"{intent.output_dataset_id}"
                )
            )

        planned = (
            _plan_join(
                datasets=
                    datasets,

                intent=
                    intent,
            )
        )

        joins.append(
            planned
        )

        seen_output_dataset_ids.add(
            intent.output_dataset_id
        )

    review_required_count = sum(
        1

        for join
        in joins

        if (
            join.status
            ==
            JoinPlanningStatus
            .REVIEW_REQUIRED
        )
    )

    blocked_count = sum(
        1

        for join
        in joins

        if (
            join.status
            ==
            JoinPlanningStatus
            .BLOCKED
        )
    )

    return (
        JoinPlan(
            request_count=
                len(
                    intents
                ),

            join_count=
                len(
                    joins
                ),

            review_required_count=
                review_required_count,

            blocked_count=
                blocked_count,

            ready_for_approval=(
                blocked_count
                ==
                0
            ),

            joins=
                joins,

            notes=[
                (
                    "Join Planner v0.1 never executes "
                    "or mutates datasets."
                ),

                (
                    "Missing join keys are treated as "
                    "unmatched and are never assumed "
                    "to match each other."
                ),

                (
                    "Cardinality is measured from "
                    "actual non-missing key values."
                ),

                (
                    "Many-to-many joins are blocked "
                    "by default in v0.1."
                ),

                (
                    "A mismatch between expected and "
                    "detected cardinality is blocking."
                ),

                (
                    "All accepted joins still require "
                    "explicit human approval because "
                    "joins can alter analytical grain."
                ),
            ],
        )
    )