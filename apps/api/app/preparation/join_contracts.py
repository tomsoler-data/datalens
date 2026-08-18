from __future__ import annotations

from enum import Enum

from typing import (
    List,
    Optional,
)

from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# VERSION
# ============================================================


JOIN_CONTRACT_VERSION = (
    "join_contracts_v0.1"
)


# ============================================================
# ENUMS
# ============================================================


class JoinType(
    str,
    Enum,
):
    INNER = (
        "inner"
    )

    LEFT = (
        "left"
    )

    RIGHT = (
        "right"
    )

    OUTER = (
        "outer"
    )


class JoinCardinality(
    str,
    Enum,
):
    ONE_TO_ONE = (
        "one_to_one"
    )

    ONE_TO_MANY = (
        "one_to_many"
    )

    MANY_TO_ONE = (
        "many_to_one"
    )

    MANY_TO_MANY = (
        "many_to_many"
    )


class JoinPlanningStatus(
    str,
    Enum,
):
    REVIEW_REQUIRED = (
        "review_required"
    )

    BLOCKED = (
        "blocked"
    )


class JoinRisk(
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

    CRITICAL = (
        "critical"
    )


# ============================================================
# KEY PAIR
# ============================================================


class JoinKeyPair(
    BaseModel,
):
    """
    Explicit mapping between one left-side join key
    and one right-side join key.

    Example:

        customers.customer_id
            ↔
        orders.customer_id
    """

    left_column: str

    right_column: str


# ============================================================
# JOIN INTENT
# ============================================================


class JoinIntent(
    BaseModel,
):
    """
    Structured proposal for combining two datasets.

    This contract does NOT execute the join.

    It only describes:

    - source datasets;
    - join type;
    - explicit key mapping;
    - optional expected cardinality;
    - destination dataset;
    - suffix strategy.
    """

    request_id: str

    left_dataset_id: str

    left_dataset_filename: str

    right_dataset_id: str

    right_dataset_filename: str

    join_type: JoinType

    keys: List[
        JoinKeyPair
    ]

    expected_cardinality: Optional[
        JoinCardinality
    ] = None

    output_dataset_id: str

    output_dataset_filename: str

    left_suffix: str = (
        "_left"
    )

    right_suffix: str = (
        "_right"
    )


# ============================================================
# JOIN DIAGNOSTICS
# ============================================================


class JoinDiagnostics(
    BaseModel,
):
    """
    Deterministic diagnostics computed before a join.

    No DataFrame is modified.
    """

    # --------------------------------------------------------
    # SOURCE SIZE
    # --------------------------------------------------------

    left_row_count: int

    right_row_count: int

    # --------------------------------------------------------
    # KEY COMPLETENESS
    # --------------------------------------------------------

    left_non_null_key_rows: int

    right_non_null_key_rows: int

    left_null_key_rows: int

    right_null_key_rows: int

    # --------------------------------------------------------
    # KEY UNIQUENESS
    # --------------------------------------------------------

    left_distinct_key_count: int

    right_distinct_key_count: int

    left_duplicate_key_rows: int

    right_duplicate_key_rows: int

    # --------------------------------------------------------
    # MATCHING
    # --------------------------------------------------------

    matched_distinct_key_count: int

    left_matched_row_count: int

    right_matched_row_count: int

    left_unmatched_row_count: int

    right_unmatched_row_count: int

    left_match_ratio: float

    right_match_ratio: float

    # --------------------------------------------------------
    # PREDICTED GRAIN EFFECT
    # --------------------------------------------------------

    predicted_output_row_count: int

    predicted_row_multiplier_vs_left: float

    # --------------------------------------------------------
    # SCHEMA COLLISIONS
    # --------------------------------------------------------

    overlapping_non_key_columns: List[
        str
    ] = Field(
        default_factory=list
    )


# ============================================================
# PLANNED JOIN
# ============================================================


class PlannedJoin(
    BaseModel,
):
    """
    Deterministically inspected join proposal.

    executable MUST remain False at the planning stage.
    """

    request_id: str

    # --------------------------------------------------------
    # INPUT DATASETS
    # --------------------------------------------------------

    left_dataset_id: str

    left_dataset_filename: str

    right_dataset_id: str

    right_dataset_filename: str

    # --------------------------------------------------------
    # JOIN SPECIFICATION
    # --------------------------------------------------------

    join_type: JoinType

    keys: List[
        JoinKeyPair
    ]

    # --------------------------------------------------------
    # CARDINALITY
    # --------------------------------------------------------

    detected_cardinality: JoinCardinality

    expected_cardinality: Optional[
        JoinCardinality
    ] = None

    cardinality_matches_expectation: Optional[
        bool
    ] = None

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_dataset_id: str

    output_dataset_filename: str

    left_suffix: str

    right_suffix: str

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    status: JoinPlanningStatus

    risk: JoinRisk

    diagnostics: JoinDiagnostics

    requires_human_approval: bool

    executable: bool = False

    rationale: str

    warnings: List[
        str
    ] = Field(
        default_factory=list
    )


# ============================================================
# JOIN PLAN
# ============================================================


class JoinPlan(
    BaseModel,
):
    """
    Collection of inspected joins.

    ready_for_approval=True means:

    - no deterministic blocking rule was triggered;
    - joins may move to human approval.

    It does NOT mean that joins may execute.
    """

    request_count: int

    join_count: int

    review_required_count: int

    blocked_count: int

    ready_for_approval: bool

    joins: List[
        PlannedJoin
    ] = Field(
        default_factory=list
    )

    notes: List[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        JOIN_CONTRACT_VERSION
    )