from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)

from enum import Enum

from typing import (
    Dict,
    List,
    Optional,
)

from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
)

from app.preparation.join_contracts import (
    JoinCardinality,
    JoinDiagnostics,
    JoinKeyPair,
    JoinPlan,
    JoinPlanningStatus,
    JoinRisk,
    JoinType,
    PlannedJoin,
)


# ============================================================
# VERSION
# ============================================================


JOIN_APPROVAL_RULE_VERSION = (
    "join_approval_v0.1"
)


# ============================================================
# USER DECISION
# ============================================================


class JoinApprovalDecision(
    str,
    Enum,
):
    APPROVE = (
        "approve"
    )

    REJECT = (
        "reject"
    )

    DEFER = (
        "defer"
    )


# ============================================================
# AUTHORIZATION STATUS
# ============================================================


class JoinAuthorizationStatus(
    str,
    Enum,
):
    APPROVED = (
        "approved"
    )

    REJECTED = (
        "rejected"
    )

    DEFERRED = (
        "deferred"
    )

    PENDING = (
        "pending"
    )


# ============================================================
# APPROVAL COMMAND
# ============================================================


class JoinApprovalCommand(
    BaseModel,
):
    request_id: str

    decision: JoinApprovalDecision

    actor: str = (
        "user"
    )

    comment: Optional[
        str
    ] = None

    decided_at: Optional[
        datetime
    ] = None


# ============================================================
# APPROVED JOIN
# ============================================================


class ApprovedJoin(
    BaseModel,
):
    approval_id: Optional[
        str
    ] = None

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
    ] = Field(
        default_factory=list
    )

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
    # PLANNER EVIDENCE
    # --------------------------------------------------------

    planner_status: JoinPlanningStatus

    risk: JoinRisk

    diagnostics: JoinDiagnostics

    warnings: List[
        str
    ] = Field(
        default_factory=list
    )

    # --------------------------------------------------------
    # AUTHORIZATION
    # --------------------------------------------------------

    requires_human_approval: bool

    authorization_status: JoinAuthorizationStatus

    user_decision: Optional[
        JoinApprovalDecision
    ] = None

    actor: Optional[
        str
    ] = None

    decided_at: Optional[
        datetime
    ] = None

    comment: Optional[
        str
    ] = None

    resolved: bool

    executable: bool

    rationale: str


# ============================================================
# APPROVED PLAN
# ============================================================


class ApprovedJoinPlan(
    BaseModel,
):
    total_join_count: int

    approved_count: int

    rejected_count: int

    deferred_count: int

    pending_count: int

    executable_join_count: int

    ready_for_execution: bool

    joins: List[
        ApprovedJoin
    ] = Field(
        default_factory=list
    )

    notes: List[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        JOIN_APPROVAL_RULE_VERSION
    )


# ============================================================
# COMMAND INDEX
# ============================================================


def _build_command_index(
    commands: List[
        JoinApprovalCommand
    ],
) -> Dict[
    str,
    JoinApprovalCommand,
]:
    output: Dict[
        str,
        JoinApprovalCommand,
    ] = {}

    for command in commands:
        request_id = (
            command.request_id
            .strip()
        )

        if not request_id:
            raise ValueError(
                (
                    "Join approval request_id "
                    "cannot be empty."
                )
            )

        if (
            request_id
            in output
        ):
            raise ValueError(
                (
                    "Duplicate join approval "
                    "command: "
                    f"{request_id}"
                )
            )

        output[
            request_id
        ] = command

    return output


# ============================================================
# JOIN INDEX
# ============================================================


def _build_join_index(
    plan: JoinPlan,
) -> Dict[
    str,
    PlannedJoin,
]:
    output: Dict[
        str,
        PlannedJoin,
    ] = {}

    for join in (
        plan.joins
    ):
        request_id = (
            join.request_id
        )

        if (
            request_id
            in output
        ):
            raise ValueError(
                (
                    "JoinPlan contains duplicate "
                    "request_id: "
                    f"{request_id}"
                )
            )

        output[
            request_id
        ] = join

    return output


# ============================================================
# BASE APPROVED JOIN
# ============================================================


def _approved_join_from_planned(
    *,
    join: PlannedJoin,
    authorization_status: JoinAuthorizationStatus,
    user_decision: Optional[
        JoinApprovalDecision
    ],
    actor: Optional[
        str
    ],
    decided_at: Optional[
        datetime
    ],
    comment: Optional[
        str
    ],
    resolved: bool,
    executable: bool,
    rationale: str,
    approval_id: Optional[
        str
    ],
) -> ApprovedJoin:
    """
    Copy the planned join exactly.

    Approval must not silently modify:

    - datasets;
    - keys;
    - join type;
    - cardinality;
    - suffixes;
    - output dataset;
    - diagnostics.

    A modified join must return to Join Planner.
    """

    return (
        ApprovedJoin(
            approval_id=
                approval_id,

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
                key.model_copy(
                    deep=True
                )

                for key
                in join.keys
            ],

            detected_cardinality=
                join.detected_cardinality,

            expected_cardinality=
                join.expected_cardinality,

            cardinality_matches_expectation=
                join
                .cardinality_matches_expectation,

            output_dataset_id=
                join.output_dataset_id,

            output_dataset_filename=
                join.output_dataset_filename,

            left_suffix=
                join.left_suffix,

            right_suffix=
                join.right_suffix,

            planner_status=
                join.status,

            risk=
                join.risk,

            diagnostics=
                join.diagnostics
                .model_copy(
                    deep=True
                ),

            warnings=
                list(
                    join.warnings
                ),

            requires_human_approval=
                join.requires_human_approval,

            authorization_status=
                authorization_status,

            user_decision=
                user_decision,

            actor=
                actor,

            decided_at=
                decided_at,

            comment=
                comment,

            resolved=
                resolved,

            executable=
                executable,

            rationale=
                rationale,
        )
    )


# ============================================================
# APPROVE
# ============================================================


def _approve_join(
    *,
    join: PlannedJoin,
    command: JoinApprovalCommand,
) -> ApprovedJoin:
    if (
        join.status
        ==
        JoinPlanningStatus.BLOCKED
    ):
        raise ValueError(
            (
                "A BLOCKED join cannot be "
                "approved. It must return to "
                "Join Planner."
            )
        )

    decided_at = (
        command.decided_at
        or
        datetime.now(
            timezone.utc
        )
    )

    return (
        _approved_join_from_planned(
            join=
                join,

            authorization_status=
                JoinAuthorizationStatus
                .APPROVED,

            user_decision=
                JoinApprovalDecision
                .APPROVE,

            actor=
                command.actor,

            decided_at=
                decided_at,

            comment=
                command.comment,

            resolved=
                True,

            executable=
                True,

            rationale=(
                "The analyst explicitly approved "
                "the join after reviewing its "
                "cardinality, match rates, orphan "
                "rows and predicted output grain."
            ),

            approval_id=(
                "join-approval:"
                +
                uuid4().hex
            ),
        )
    )


# ============================================================
# REJECT
# ============================================================


def _reject_join(
    *,
    join: PlannedJoin,
    command: JoinApprovalCommand,
) -> ApprovedJoin:
    decided_at = (
        command.decided_at
        or
        datetime.now(
            timezone.utc
        )
    )

    return (
        _approved_join_from_planned(
            join=
                join,

            authorization_status=
                JoinAuthorizationStatus
                .REJECTED,

            user_decision=
                JoinApprovalDecision
                .REJECT,

            actor=
                command.actor,

            decided_at=
                decided_at,

            comment=
                command.comment,

            resolved=
                True,

            executable=
                False,

            rationale=(
                "The analyst explicitly rejected "
                "the proposed join. No dataset "
                "combination is authorized."
            ),

            approval_id=(
                "join-approval:"
                +
                uuid4().hex
            ),
        )
    )


# ============================================================
# DEFER
# ============================================================


def _defer_join(
    *,
    join: PlannedJoin,
    command: JoinApprovalCommand,
) -> ApprovedJoin:
    decided_at = (
        command.decided_at
        or
        datetime.now(
            timezone.utc
        )
    )

    return (
        _approved_join_from_planned(
            join=
                join,

            authorization_status=
                JoinAuthorizationStatus
                .DEFERRED,

            user_decision=
                JoinApprovalDecision
                .DEFER,

            actor=
                command.actor,

            decided_at=
                decided_at,

            comment=
                command.comment,

            resolved=
                False,

            executable=
                False,

            rationale=(
                "The analyst deferred the join "
                "decision. Execution remains blocked "
                "until the decision is resolved."
            ),

            approval_id=(
                "join-approval:"
                +
                uuid4().hex
            ),
        )
    )


# ============================================================
# PENDING
# ============================================================


def _pending_join(
    *,
    join: PlannedJoin,
) -> ApprovedJoin:
    return (
        _approved_join_from_planned(
            join=
                join,

            authorization_status=
                JoinAuthorizationStatus
                .PENDING,

            user_decision=None,

            actor=None,

            decided_at=None,

            comment=None,

            resolved=
                False,

            executable=
                False,

            rationale=(
                "The join requires explicit human "
                "approval and no decision has yet "
                "been recorded."
            ),

            approval_id=None,
        )
    )


# ============================================================
# PUBLIC API
# ============================================================


def apply_join_approvals(
    *,
    plan: JoinPlan,
    commands: List[
        JoinApprovalCommand
    ],
) -> ApprovedJoinPlan:
    """
    Convert a JoinPlan into an explicit authorization contract.

    Safety guarantees:

    - never executes a join;
    - never mutates input datasets;
    - never modifies join parameters;
    - BLOCKED joins cannot enter approval;
    - all executable joins require explicit APPROVE;
    - REJECT is resolved but non-executable;
    - DEFER and PENDING remain unresolved;
    - MODIFY is intentionally unsupported;
    - changed join parameters must return to Join Planner.
    """

    # ========================================================
    # PLAN MUST BE APPROVABLE
    # ========================================================

    if not (
        plan.ready_for_approval
    ):
        raise ValueError(
            (
                "JoinPlan is not ready for approval. "
                "At least one join is blocked."
            )
        )

    if (
        plan.blocked_count
        >
        0
    ):
        raise ValueError(
            (
                "JoinPlan contains blocked joins "
                "and cannot enter approval."
            )
        )

    for join in (
        plan.joins
    ):
        if (
            join.status
            ==
            JoinPlanningStatus.BLOCKED
        ):
            raise ValueError(
                (
                    "BLOCKED join found inside an "
                    "approvable JoinPlan: "
                    f"{join.request_id}"
                )
            )

    # ========================================================
    # INDEXES
    # ========================================================

    join_index = (
        _build_join_index(
            plan
        )
    )

    command_index = (
        _build_command_index(
            commands
        )
    )

    # ========================================================
    # UNKNOWN COMMANDS
    # ========================================================

    for request_id in (
        command_index
    ):
        if (
            request_id
            not in
            join_index
        ):
            raise ValueError(
                (
                    "Join approval references "
                    "unknown request_id: "
                    f"{request_id}"
                )
            )

    # ========================================================
    # AUTHORIZE
    # ========================================================

    approved_joins: List[
        ApprovedJoin
    ] = []

    for join in (
        plan.joins
    ):
        command = (
            command_index.get(
                join.request_id
            )
        )

        if (
            command is None
        ):
            approved_joins.append(
                _pending_join(
                    join=
                        join
                )
            )

            continue

        if (
            command.decision
            ==
            JoinApprovalDecision.APPROVE
        ):
            approved_joins.append(
                _approve_join(
                    join=
                        join,

                    command=
                        command,
                )
            )

            continue

        if (
            command.decision
            ==
            JoinApprovalDecision.REJECT
        ):
            approved_joins.append(
                _reject_join(
                    join=
                        join,

                    command=
                        command,
                )
            )

            continue

        if (
            command.decision
            ==
            JoinApprovalDecision.DEFER
        ):
            approved_joins.append(
                _defer_join(
                    join=
                        join,

                    command=
                        command,
                )
            )

            continue

        raise ValueError(
            (
                "Unsupported join approval "
                f"decision: {command.decision}"
            )
        )

    # ========================================================
    # COUNTS
    # ========================================================

    approved_count = sum(
        1

        for join
        in approved_joins

        if (
            join.authorization_status
            ==
            JoinAuthorizationStatus.APPROVED
        )
    )

    rejected_count = sum(
        1

        for join
        in approved_joins

        if (
            join.authorization_status
            ==
            JoinAuthorizationStatus.REJECTED
        )
    )

    deferred_count = sum(
        1

        for join
        in approved_joins

        if (
            join.authorization_status
            ==
            JoinAuthorizationStatus.DEFERRED
        )
    )

    pending_count = sum(
        1

        for join
        in approved_joins

        if (
            join.authorization_status
            ==
            JoinAuthorizationStatus.PENDING
        )
    )

    executable_join_count = sum(
        1

        for join
        in approved_joins

        if join.executable
    )

    ready_for_execution = all(
        join.resolved

        for join
        in approved_joins
    )

    # ========================================================
    # INVARIANTS
    # ========================================================

    for join in (
        approved_joins
    ):
        if (
            join.executable
            and
            join.authorization_status
            !=
            JoinAuthorizationStatus.APPROVED
        ):
            raise ValueError(
                (
                    "Executable join does not have "
                    "APPROVED authorization: "
                    f"{join.request_id}"
                )
            )

        if (
            join.executable
            and
            not join.resolved
        ):
            raise ValueError(
                (
                    "Executable join cannot remain "
                    "unresolved: "
                    f"{join.request_id}"
                )
            )

    # ========================================================
    # RESULT
    # ========================================================

    return (
        ApprovedJoinPlan(
            total_join_count=
                len(
                    approved_joins
                ),

            approved_count=
                approved_count,

            rejected_count=
                rejected_count,

            deferred_count=
                deferred_count,

            pending_count=
                pending_count,

            executable_join_count=
                executable_join_count,

            ready_for_execution=
                ready_for_execution,

            joins=
                approved_joins,

            notes=[
                (
                    "Join Approval v0.1 never "
                    "executes or mutates datasets."
                ),

                (
                    "Every executable join requires "
                    "an explicit APPROVE decision."
                ),

                (
                    "BLOCKED joins cannot be overridden "
                    "by the approval layer."
                ),

                (
                    "REJECT is a resolved decision "
                    "and produces no join output."
                ),

                (
                    "DEFER and PENDING remain "
                    "unresolved and prevent execution."
                ),

                (
                    "MODIFY is intentionally unsupported. "
                    "Any change to keys, join type, "
                    "cardinality assumptions, suffixes "
                    "or output dataset must return "
                    "through Join Planner."
                ),

                (
                    "Planner diagnostics are copied "
                    "unchanged into the approved join "
                    "for future execution validation."
                ),
            ],
        )
    )