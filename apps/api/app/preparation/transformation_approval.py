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
    Set,
)

from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
)

from app.preparation.transformation_contracts import (
    TransformationOperation,
    TransformationPlan,
    TransformationRisk,
    TransformationStatus,
    TransformationStep,
)


# ============================================================
# VERSION
# ============================================================


TRANSFORMATION_APPROVAL_RULE_VERSION = (
    "transformation_approval_v0.1"
)


# ============================================================
# USER DECISION
# ============================================================


class TransformationApprovalDecision(
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


class TransformationAuthorizationStatus(
    str,
    Enum,
):
    AUTOMATIC = (
        "automatic"
    )

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

    BLOCKED_DEPENDENCY = (
        "blocked_dependency"
    )


# ============================================================
# INPUT
# ============================================================


class TransformationApprovalCommand(
    BaseModel,
):
    request_id: str

    decision: TransformationApprovalDecision

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
# OUTPUT STEP
# ============================================================


class ApprovedTransformationStep(
    BaseModel,
):
    approval_id: Optional[
        str
    ] = None

    step_id: str

    request_id: str

    dataset_id: str

    dataset_filename: str

    operation: TransformationOperation

    planner_status: TransformationStatus

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

    parameters: Dict = Field(
        default_factory=dict
    )

    requires_human_approval: bool

    authorization_status: (
        TransformationAuthorizationStatus
    )

    user_decision: Optional[
        TransformationApprovalDecision
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

    depends_on_request_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    resolved: bool

    executable: bool

    rationale: str


# ============================================================
# OUTPUT PLAN
# ============================================================


class ApprovedTransformationPlan(
    BaseModel,
):
    dataset_id: str

    dataset_filename: str

    total_step_count: int

    automatic_count: int

    approved_count: int

    rejected_count: int

    deferred_count: int

    pending_count: int

    blocked_dependency_count: int

    executable_step_count: int

    ready_for_execution: bool

    steps: List[
        ApprovedTransformationStep
    ] = Field(
        default_factory=list
    )

    notes: List[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        TRANSFORMATION_APPROVAL_RULE_VERSION
    )


# ============================================================
# COMMAND INDEX
# ============================================================


def _command_index(
    commands: List[
        TransformationApprovalCommand
    ],
) -> Dict[
    str,
    TransformationApprovalCommand,
]:
    output: Dict[
        str,
        TransformationApprovalCommand,
    ] = {}

    for command in commands:
        request_id = (
            command.request_id.strip()
        )

        if not request_id:
            raise ValueError(
                (
                    "Transformation approval "
                    "request_id cannot be empty."
                )
            )

        if request_id in output:
            raise ValueError(
                (
                    "Duplicate transformation "
                    "approval command: "
                    f"{request_id}"
                )
            )

        output[
            request_id
        ] = command

    return output


# ============================================================
# STEP INDEX
# ============================================================


def _step_index(
    plan: TransformationPlan,
) -> Dict[
    str,
    TransformationStep,
]:
    output: Dict[
        str,
        TransformationStep,
    ] = {}

    for step in plan.steps:
        if (
            step.request_id
            in output
        ):
            raise ValueError(
                (
                    "TransformationPlan contains "
                    "duplicate request_id: "
                    f"{step.request_id}"
                )
            )

        output[
            step.request_id
        ] = step

    return output


# ============================================================
# DEPENDENCIES
# ============================================================


def _build_dependencies(
    plan: TransformationPlan,
) -> Dict[
    str,
    List[
        str
    ],
]:
    """
    Determine dependencies between transformation steps.

    Example:

        derive-revenue
            produces revenue

        aggregate-segment-revenue
            consumes revenue

    therefore:

        aggregate-segment-revenue
            depends on derive-revenue
    """

    producer_by_column: Dict[
        str,
        str,
    ] = {}

    dependencies: Dict[
        str,
        List[
            str
        ],
    ] = {}

    for step in plan.steps:
        step_dependencies: List[
            str
        ] = []

        for column in (
            step.input_columns
        ):
            producer = (
                producer_by_column.get(
                    column
                )
            )

            if (
                producer is not None
                and
                producer
                !=
                step.request_id
            ):
                if (
                    producer
                    not in
                    step_dependencies
                ):
                    step_dependencies.append(
                        producer
                    )

        dependencies[
            step.request_id
        ] = step_dependencies

        if (
            step.output_column
            is not None
        ):
            if (
                step.output_column
                in producer_by_column
            ):
                raise ValueError(
                    (
                        "Multiple transformation "
                        "steps produce the same "
                        "virtual output column: "
                        f"{step.output_column}"
                    )
                )

            producer_by_column[
                step.output_column
            ] = step.request_id

    return dependencies


# ============================================================
# AUTOMATIC AUTHORIZATION
# ============================================================


def _automatic_step(
    *,
    step: TransformationStep,
    dependencies: List[
        str
    ],
) -> ApprovedTransformationStep:
    if (
        step.status
        !=
        TransformationStatus.VALIDATED
    ):
        raise ValueError(
            (
                "Automatic transformation "
                "authorization requires "
                "VALIDATED planner status."
            )
        )

    if (
        step.requires_human_approval
    ):
        raise ValueError(
            (
                "A transformation requiring human "
                "approval cannot be automatically "
                "authorized."
            )
        )

    return (
        ApprovedTransformationStep(
            approval_id=None,

            step_id=
                step.step_id,

            request_id=
                step.request_id,

            dataset_id=
                step.dataset_id,

            dataset_filename=
                step.dataset_filename,

            operation=
                step.operation,

            planner_status=
                step.status,

            risk=
                step.risk,

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

            parameters=
                dict(
                    step.parameters
                ),

            requires_human_approval=
                step.requires_human_approval,

            authorization_status=(
                TransformationAuthorizationStatus
                .AUTOMATIC
            ),

            user_decision=None,

            actor=None,

            decided_at=None,

            comment=None,

            depends_on_request_ids=
                list(
                    dependencies
                ),

            resolved=True,

            executable=True,

            rationale=(
                "The Transformation Planner "
                "validated this low-risk step "
                "and did not require explicit "
                "human approval."
            ),
        )
    )


# ============================================================
# HUMAN DECISION
# ============================================================


def _human_step(
    *,
    step: TransformationStep,
    command: TransformationApprovalCommand,
    dependencies: List[
        str
    ],
) -> ApprovedTransformationStep:
    decided_at = (
        command.decided_at
        or
        datetime.now(
            timezone.utc
        )
    )

    approval_id = (
        "transform-approval:"
        +
        uuid4().hex
    )

    # ========================================================
    # APPROVE
    # ========================================================

    if (
        command.decision
        ==
        TransformationApprovalDecision
        .APPROVE
    ):
        return (
            ApprovedTransformationStep(
                approval_id=
                    approval_id,

                step_id=
                    step.step_id,

                request_id=
                    step.request_id,

                dataset_id=
                    step.dataset_id,

                dataset_filename=
                    step.dataset_filename,

                operation=
                    step.operation,

                planner_status=
                    step.status,

                risk=
                    step.risk,

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

                parameters=
                    dict(
                        step.parameters
                    ),

                requires_human_approval=
                    step.requires_human_approval,

                authorization_status=(
                    TransformationAuthorizationStatus
                    .APPROVED
                ),

                user_decision=
                    command.decision,

                actor=
                    command.actor,

                decided_at=
                    decided_at,

                comment=
                    command.comment,

                depends_on_request_ids=
                    list(
                        dependencies
                    ),

                resolved=True,

                executable=True,

                rationale=(
                    "The transformation was "
                    "explicitly approved."
                ),
            )
        )

    # ========================================================
    # REJECT
    # ========================================================

    if (
        command.decision
        ==
        TransformationApprovalDecision
        .REJECT
    ):
        return (
            ApprovedTransformationStep(
                approval_id=
                    approval_id,

                step_id=
                    step.step_id,

                request_id=
                    step.request_id,

                dataset_id=
                    step.dataset_id,

                dataset_filename=
                    step.dataset_filename,

                operation=
                    step.operation,

                planner_status=
                    step.status,

                risk=
                    step.risk,

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

                parameters=
                    dict(
                        step.parameters
                    ),

                requires_human_approval=
                    step.requires_human_approval,

                authorization_status=(
                    TransformationAuthorizationStatus
                    .REJECTED
                ),

                user_decision=
                    command.decision,

                actor=
                    command.actor,

                decided_at=
                    decided_at,

                comment=
                    command.comment,

                depends_on_request_ids=
                    list(
                        dependencies
                    ),

                resolved=True,

                executable=False,

                rationale=(
                    "The transformation was "
                    "explicitly rejected and "
                    "will not be executed."
                ),
            )
        )

    # ========================================================
    # DEFER
    # ========================================================

    if (
        command.decision
        ==
        TransformationApprovalDecision
        .DEFER
    ):
        return (
            ApprovedTransformationStep(
                approval_id=
                    approval_id,

                step_id=
                    step.step_id,

                request_id=
                    step.request_id,

                dataset_id=
                    step.dataset_id,

                dataset_filename=
                    step.dataset_filename,

                operation=
                    step.operation,

                planner_status=
                    step.status,

                risk=
                    step.risk,

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

                parameters=
                    dict(
                        step.parameters
                    ),

                requires_human_approval=
                    step.requires_human_approval,

                authorization_status=(
                    TransformationAuthorizationStatus
                    .DEFERRED
                ),

                user_decision=
                    command.decision,

                actor=
                    command.actor,

                decided_at=
                    decided_at,

                comment=
                    command.comment,

                depends_on_request_ids=
                    list(
                        dependencies
                    ),

                resolved=False,

                executable=False,

                rationale=(
                    "The transformation decision "
                    "was deferred."
                ),
            )
        )

    raise ValueError(
        (
            "Unsupported transformation "
            "approval decision."
        )
    )


# ============================================================
# PENDING
# ============================================================


def _pending_step(
    *,
    step: TransformationStep,
    dependencies: List[
        str
    ],
) -> ApprovedTransformationStep:
    return (
        ApprovedTransformationStep(
            approval_id=None,

            step_id=
                step.step_id,

            request_id=
                step.request_id,

            dataset_id=
                step.dataset_id,

            dataset_filename=
                step.dataset_filename,

            operation=
                step.operation,

            planner_status=
                step.status,

            risk=
                step.risk,

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

            parameters=
                dict(
                    step.parameters
                ),

            requires_human_approval=
                step.requires_human_approval,

            authorization_status=(
                TransformationAuthorizationStatus
                .PENDING
            ),

            user_decision=None,

            actor=None,

            decided_at=None,

            comment=None,

            depends_on_request_ids=
                list(
                    dependencies
                ),

            resolved=False,

            executable=False,

            rationale=(
                "This transformation requires "
                "explicit human approval and "
                "no decision has been recorded."
            ),
        )
    )


# ============================================================
# DEPENDENCY GUARDRAIL
# ============================================================


def _apply_dependency_guardrail(
    steps: List[
        ApprovedTransformationStep
    ],
) -> List[
    ApprovedTransformationStep
]:
    """
    An executable transformation cannot depend on the output
    of a transformation that will not execute.

    Example:

        derive revenue
            REJECTED

        aggregate SUM(revenue)
            APPROVED

    The aggregate must become BLOCKED_DEPENDENCY.
    """

    result = list(
        steps
    )

    # Multiple passes make chained dependencies safe.
    changed = True

    while changed:
        changed = False

        step_by_request = {
            step.request_id:
                step

            for step
            in result
        }

        updated_steps: List[
            ApprovedTransformationStep
        ] = []

        for step in result:
            if not (
                step.executable
            ):
                updated_steps.append(
                    step
                )

                continue

            unavailable_dependencies: List[
                str
            ] = []

            for dependency_id in (
                step.depends_on_request_ids
            ):
                dependency = (
                    step_by_request.get(
                        dependency_id
                    )
                )

                if (
                    dependency is None
                    or
                    not dependency.executable
                ):
                    unavailable_dependencies.append(
                        dependency_id
                    )

            if not (
                unavailable_dependencies
            ):
                updated_steps.append(
                    step
                )

                continue

            updated_steps.append(
                step.model_copy(
                    update={
                        "authorization_status": (
                            TransformationAuthorizationStatus
                            .BLOCKED_DEPENDENCY
                        ),

                        "resolved":
                            False,

                        "executable":
                            False,

                        "rationale": (
                            "The transformation cannot "
                            "execute because one or more "
                            "required derived inputs are "
                            "not authorized for execution: "
                            f"{unavailable_dependencies}"
                        ),
                    }
                )
            )

            changed = True

        result = (
            updated_steps
        )

    return result


# ============================================================
# PUBLIC API
# ============================================================


def apply_transformation_approvals(
    *,
    plan: TransformationPlan,
    commands: List[
        TransformationApprovalCommand
    ],
) -> ApprovedTransformationPlan:
    """
    Convert a validated TransformationPlan into an explicit
    execution authorization contract.

    Safety guarantees:

    - never executes transformations;
    - never modifies transformation parameters;
    - no MODIFY operation is supported;
    - any changed transformation must return to the planner;
    - low-risk VALIDATED steps may be automatically authorized;
    - REVIEW_REQUIRED steps require a user decision;
    - rejected/deferred dependencies block downstream steps;
    - only executable=True steps may enter the future executor.
    """

    if not (
        plan.ready_for_approval
    ):
        raise ValueError(
            (
                "TransformationPlan is not "
                "ready for approval."
            )
        )

    step_index = (
        _step_index(
            plan
        )
    )

    command_index = (
        _command_index(
            commands
        )
    )

    dependencies = (
        _build_dependencies(
            plan
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
            step_index
        ):
            raise ValueError(
                (
                    "Transformation approval "
                    "references unknown request_id: "
                    f"{request_id}"
                )
            )

    # ========================================================
    # INITIAL AUTHORIZATION
    # ========================================================

    approved_steps: List[
        ApprovedTransformationStep
    ] = []

    for step in plan.steps:
        command = (
            command_index.get(
                step.request_id
            )
        )

        step_dependencies = (
            dependencies.get(
                step.request_id,
                [],
            )
        )

        # ----------------------------------------------------
        # Explicit user decision always wins.
        #
        # This means a user may also reject/defer an otherwise
        # low-risk automatically validated transformation.
        # ----------------------------------------------------

        if (
            command is not None
        ):
            approved_steps.append(
                _human_step(
                    step=
                        step,

                    command=
                        command,

                    dependencies=
                        step_dependencies,
                )
            )

            continue

        # ----------------------------------------------------
        # Automatic low-risk validated step.
        # ----------------------------------------------------

        if (
            step.status
            ==
            TransformationStatus.VALIDATED
            and
            not step.requires_human_approval
        ):
            approved_steps.append(
                _automatic_step(
                    step=
                        step,

                    dependencies=
                        step_dependencies,
                )
            )

            continue

        # ----------------------------------------------------
        # Otherwise human approval is still missing.
        # ----------------------------------------------------

        approved_steps.append(
            _pending_step(
                step=
                    step,

                dependencies=
                    step_dependencies,
            )
        )

    # ========================================================
    # DEPENDENCY VALIDATION
    # ========================================================

    approved_steps = (
        _apply_dependency_guardrail(
            approved_steps
        )
    )

    # ========================================================
    # COUNTS
    # ========================================================

    automatic_count = sum(
        1

        for step
        in approved_steps

        if (
            step.authorization_status
            ==
            TransformationAuthorizationStatus
            .AUTOMATIC
        )
    )

    approved_count = sum(
        1

        for step
        in approved_steps

        if (
            step.authorization_status
            ==
            TransformationAuthorizationStatus
            .APPROVED
        )
    )

    rejected_count = sum(
        1

        for step
        in approved_steps

        if (
            step.authorization_status
            ==
            TransformationAuthorizationStatus
            .REJECTED
        )
    )

    deferred_count = sum(
        1

        for step
        in approved_steps

        if (
            step.authorization_status
            ==
            TransformationAuthorizationStatus
            .DEFERRED
        )
    )

    pending_count = sum(
        1

        for step
        in approved_steps

        if (
            step.authorization_status
            ==
            TransformationAuthorizationStatus
            .PENDING
        )
    )

    blocked_dependency_count = sum(
        1

        for step
        in approved_steps

        if (
            step.authorization_status
            ==
            TransformationAuthorizationStatus
            .BLOCKED_DEPENDENCY
        )
    )

    executable_step_count = sum(
        1

        for step
        in approved_steps

        if step.executable
    )

    ready_for_execution = all(
        step.resolved

        for step
        in approved_steps
    )

    return (
        ApprovedTransformationPlan(
            dataset_id=
                plan.dataset_id,

            dataset_filename=
                plan.dataset_filename,

            total_step_count=
                len(
                    approved_steps
                ),

            automatic_count=
                automatic_count,

            approved_count=
                approved_count,

            rejected_count=
                rejected_count,

            deferred_count=
                deferred_count,

            pending_count=
                pending_count,

            blocked_dependency_count=
                blocked_dependency_count,

            executable_step_count=
                executable_step_count,

            ready_for_execution=
                ready_for_execution,

            steps=
                approved_steps,

            notes=[
                (
                    "Transformation Approval v0.1 "
                    "never executes data transformations."
                ),

                (
                    "Low-risk VALIDATED transformations "
                    "may be automatically authorized."
                ),

                (
                    "REVIEW_REQUIRED transformations "
                    "require explicit user approval."
                ),

                (
                    "Users may still reject or defer "
                    "an automatically validated step."
                ),

                (
                    "MODIFY is intentionally unsupported. "
                    "Any parameter change must create a new "
                    "TransformationIntent and return through "
                    "Transformation Planner."
                ),

                (
                    "Downstream transformations are blocked "
                    "if a required derived transformation "
                    "will not execute."
                ),

                (
                    "Only executable=True steps may enter "
                    "the future Transformation Executor."
                ),
            ],
        )
    )