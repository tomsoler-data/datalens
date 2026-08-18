from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)

from enum import Enum

from typing import (
    Any,
    Literal,
)

from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.preparation.context_resolver import (
    ContextualPreparationProposal,
    PreparationContextResolutionReport,
    ResolutionEvidenceReference,
)

from app.preparation.contracts import (
    DecisionRisk,
    DecisionStatus,
    PreparationAction,
    PreparationDecision,
    PreparationPlan,
)


# ============================================================
# VERSION
# ============================================================


PREPARATION_APPROVAL_RULE_VERSION = (
    "preparation_approval_v0.1"
)


# ============================================================
# USER DECISIONS
# ============================================================


class ApprovalDecision(
    str,
    Enum,
):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    DEFER = "defer"


class ApprovalStepStatus(
    str,
    Enum,
):
    AUTOMATIC = "automatic"

    APPROVED = "approved"

    MODIFIED = "modified"

    REJECTED = "rejected"

    DEFERRED = "deferred"

    PENDING = "pending"

    MANUAL_FOLLOWUP = "manual_followup"


# ============================================================
# ACTION CLASSIFICATION
# ============================================================


MUTATING_ACTIONS = {
    PreparationAction.TRIM_WHITESPACE,

    PreparationAction.NORMALIZE_EMPTY_TO_MISSING,

    PreparationAction.NORMALIZE_MISSING_MARKERS,

    PreparationAction.NORMALIZE_CASE,

    PreparationAction.MERGE_CATEGORY_VALUES,

    PreparationAction.CONVERT_TO_NUMERIC,

    PreparationAction.DROP_ROWS_WITH_MISSING,

    PreparationAction.DROP_COLUMN,

    PreparationAction.IMPUTE_MEAN,

    PreparationAction.IMPUTE_MEDIAN,

    PreparationAction.IMPUTE_MODE,

    PreparationAction.CREATE_MISSING_CATEGORY,

    PreparationAction.DOMAIN_SPECIFIC_VALUE,

    PreparationAction.REMOVE_DUPLICATE_ROWS,

    PreparationAction.CAP_OUTLIERS,

    PreparationAction.REMOVE_OUTLIER_ROWS,

    PreparationAction.RENAME_DUPLICATE_COLUMNS,
}


NO_CHANGE_ACTIONS = {
    PreparationAction.KEEP_AS_IS,

    PreparationAction.KEEP_MISSING,

    PreparationAction.KEEP_DUPLICATE_ROWS,

    PreparationAction.KEEP_CATEGORIES_SEPARATE,
}


MANUAL_FOLLOWUP_ACTIONS = {
    PreparationAction.REVIEW_DUPLICATES,

    PreparationAction.INVESTIGATE_OUTLIERS,

    PreparationAction.REVIEW_INVALID_VALUES,

    PreparationAction.REVIEW_INVALID_DATES,

    PreparationAction.REVIEW_SEMANTIC_CONTEXT,

    PreparationAction.REIMPORT_OR_FIX_SOURCE,

    PreparationAction.CONFIRM_IDENTIFIER,
}


# ============================================================
# INPUT
# ============================================================


class PreparationApprovalCommand(
    BaseModel
):
    """
    Décision explicite prise par l'utilisateur.

    MODIFY nécessite une action et, lorsque nécessaire,
    des paramètres explicites.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    decision_id: str

    decision: ApprovalDecision

    modified_action: (
        PreparationAction
        | None
    ) = None

    modified_parameters: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    comment: (
        str
        | None
    ) = None

    actor: str = "user"

    decided_at: (
        datetime
        | None
    ) = None


# ============================================================
# OUTPUT
# ============================================================


class ApprovedPreparationStep(
    BaseModel
):
    """
    État auditable d'une décision après la couche
    d'approbation.

    approved_action représente ce qui pourra être transmis
    au futur executor lorsque executor_eligible=True.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    approval_id: (
        str
        | None
    ) = None

    decision_id: str

    source_issue_id: str

    source_issue_kind: str

    dataset_id: str

    dataset_filename: str

    column: (
        str
        | None
    ) = None

    original_decision_status: DecisionStatus

    proposal_status: (
        str
        | None
    ) = None

    proposal_confidence: (
        str
        | None
    ) = None

    risk: DecisionRisk

    recommended_action: (
        PreparationAction
        | None
    ) = None

    recommended_parameters: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    user_decision: (
        ApprovalDecision
        | None
    ) = None

    approved_action: (
        PreparationAction
        | None
    ) = None

    approved_parameters: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    approved_operation: (
        str
        | None
    ) = None

    status: ApprovalStepStatus

    actor: (
        str
        | None
    ) = None

    decided_at: (
        datetime
        | None
    ) = None

    comment: (
        str
        | None
    ) = None

    rationale: str

    evidence: list[
        ResolutionEvidenceReference
    ] = Field(
        default_factory=list
    )

    human_validation_performed: bool = False

    resolved: bool = False

    mutates_data: bool = False

    executor_eligible: bool = False

    requires_manual_followup: bool = False


class ApprovedPreparationPlan(
    BaseModel
):
    """
    Seul ce contrat devra être accepté par le futur
    Cleaning Executor.

    ready_for_execution=True signifie :

    - aucune décision en attente ;
    - aucune décision différée ;
    - aucune investigation manuelle encore ouverte ;
    - toutes les décisions sont résolues.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    total_step_count: int = Field(
        ge=0
    )

    auto_approved_count: int = Field(
        ge=0
    )

    approved_count: int = Field(
        ge=0
    )

    modified_count: int = Field(
        ge=0
    )

    rejected_count: int = Field(
        ge=0
    )

    deferred_count: int = Field(
        ge=0
    )

    pending_count: int = Field(
        ge=0
    )

    manual_followup_count: int = Field(
        ge=0
    )

    execution_step_count: int = Field(
        ge=0
    )

    no_change_count: int = Field(
        ge=0
    )

    ready_for_execution: bool

    steps: list[
        ApprovedPreparationStep
    ] = Field(
        default_factory=list
    )

    notes: list[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        PREPARATION_APPROVAL_RULE_VERSION
    )


# ============================================================
# INDEXES
# ============================================================


def _decision_index(
    plan: PreparationPlan,
) -> dict[
    str,
    PreparationDecision,
]:
    output: dict[
        str,
        PreparationDecision,
    ] = {}

    for decision in plan.decisions:
        if decision.decision_id in output:
            raise ValueError(
                (
                    "PreparationPlan contient plusieurs "
                    "décisions avec decision_id="
                    f"{decision.decision_id}."
                )
            )

        output[
            decision.decision_id
        ] = decision

    return output


def _proposal_index(
    report: PreparationContextResolutionReport,
) -> dict[
    str,
    ContextualPreparationProposal,
]:
    output: dict[
        str,
        ContextualPreparationProposal,
    ] = {}

    for proposal in report.proposals:
        if proposal.decision_id in output:
            raise ValueError(
                (
                    "PreparationContextResolutionReport "
                    "contient plusieurs propositions pour "
                    f"{proposal.decision_id}."
                )
            )

        output[
            proposal.decision_id
        ] = proposal

    return output


def _command_index(
    commands: list[
        PreparationApprovalCommand
    ],
) -> dict[
    str,
    PreparationApprovalCommand,
]:
    output: dict[
        str,
        PreparationApprovalCommand,
    ] = {}

    for command in commands:
        if command.decision_id in output:
            raise ValueError(
                (
                    "Plusieurs décisions utilisateur ont "
                    "été fournies pour "
                    f"{command.decision_id}."
                )
            )

        output[
            command.decision_id
        ] = command

    return output


# ============================================================
# ACTION CHARACTERISTICS
# ============================================================


def _action_characteristics(
    action: PreparationAction,
) -> tuple[
    bool,
    bool,
    bool,
]:
    """
    Retourne :

        mutates_data
        executor_eligible
        requires_manual_followup
    """

    if action in MUTATING_ACTIONS:
        return (
            True,
            True,
            False,
        )

    if action in NO_CHANGE_ACTIONS:
        return (
            False,
            False,
            False,
        )

    if action in MANUAL_FOLLOWUP_ACTIONS:
        return (
            False,
            False,
            True,
        )

    # Safe fallback:
    # une nouvelle action inconnue du classifieur
    # ne devient jamais exécutable automatiquement.
    return (
        False,
        False,
        True,
    )


# ============================================================
# PARAMETER VALIDATION
# ============================================================


def _require_parameter(
    *,
    parameters: dict[
        str,
        Any,
    ],
    name: str,
    action: PreparationAction,
) -> Any:
    if name not in parameters:
        raise ValueError(
            (
                f"L'action {action.value} nécessite "
                f"le paramètre '{name}'."
            )
        )

    return parameters[
        name
    ]


def _validate_action_parameters(
    *,
    action: PreparationAction,
    parameters: dict[
        str,
        Any,
    ],
) -> None:
    """
    Validation déterministe minimale des paramètres.

    Cette validation ne dépend pas encore du DataFrame.
    Les validations de type de colonne seront ajoutées
    avant ou dans le Cleaning Executor.
    """

    if (
        action
        ==
        PreparationAction.DOMAIN_SPECIFIC_VALUE
    ):
        _require_parameter(
            parameters=
                parameters,

            name=
                "value",

            action=
                action,
        )

        return

    if (
        action
        ==
        PreparationAction.CREATE_MISSING_CATEGORY
    ):
        value = _require_parameter(
            parameters=
                parameters,

            name=
                "value",

            action=
                action,
        )

        if (
            not isinstance(
                value,
                str,
            )
            or
            not value.strip()
        ):
            raise ValueError(
                (
                    "CREATE_MISSING_CATEGORY nécessite "
                    "une valeur textuelle non vide."
                )
            )

        return

    if (
        action
        ==
        PreparationAction.NORMALIZE_CASE
    ):
        case = _require_parameter(
            parameters=
                parameters,

            name=
                "case",

            action=
                action,
        )

        allowed = {
            "lower",
            "upper",
            "title",
            "casefold",
        }

        if case not in allowed:
            raise ValueError(
                (
                    "NORMALIZE_CASE nécessite case parmi "
                    f"{sorted(allowed)}."
                )
            )

        return

    if (
        action
        ==
        PreparationAction.MERGE_CATEGORY_VALUES
    ):
        source_values = _require_parameter(
            parameters=
                parameters,

            name=
                "source_values",

            action=
                action,
        )

        canonical_value = _require_parameter(
            parameters=
                parameters,

            name=
                "canonical_value",

            action=
                action,
        )

        if (
            not isinstance(
                source_values,
                list,
            )
            or
            not source_values
        ):
            raise ValueError(
                (
                    "MERGE_CATEGORY_VALUES nécessite "
                    "source_values sous forme de liste "
                    "non vide."
                )
            )

        if canonical_value is None:
            raise ValueError(
                (
                    "MERGE_CATEGORY_VALUES nécessite "
                    "canonical_value."
                )
            )

        return

    if (
        action
        ==
        PreparationAction.CAP_OUTLIERS
    ):
        if (
            "lower_bound"
            not in parameters
            and
            "upper_bound"
            not in parameters
        ):
            raise ValueError(
                (
                    "CAP_OUTLIERS nécessite au moins "
                    "lower_bound ou upper_bound."
                )
            )

        return

    if (
        action
        ==
        PreparationAction.REMOVE_OUTLIER_ROWS
    ):
        row_indices = parameters.get(
            "row_indices"
        )

        comparator = parameters.get(
            "documented_comparator"
        )

        threshold = parameters.get(
            "documented_threshold"
        )

        has_rows = (
            isinstance(
                row_indices,
                list,
            )
            and
            bool(
                row_indices
            )
        )

        has_rule = (
            comparator
            in {
                ">",
                ">=",
                "<",
                "<=",
            }
            and
            threshold
            is not None
        )

        if (
            not has_rows
            and
            not has_rule
        ):
            raise ValueError(
                (
                    "REMOVE_OUTLIER_ROWS nécessite soit "
                    "row_indices, soit documented_comparator "
                    "+ documented_threshold."
                )
            )

        return

    if (
        action
        ==
        PreparationAction.RENAME_DUPLICATE_COLUMNS
    ):
        mapping = _require_parameter(
            parameters=
                parameters,

            name=
                "mapping",

            action=
                action,
        )

        if (
            not isinstance(
                mapping,
                dict,
            )
            or
            not mapping
        ):
            raise ValueError(
                (
                    "RENAME_DUPLICATE_COLUMNS nécessite "
                    "un mapping non vide."
                )
            )

        return

    # Les autres actions n'exigent aucun paramètre
    # obligatoire dans Approval v0.1.


# ============================================================
# CANDIDATE ACTION GUARDRAIL
# ============================================================


def _validate_action_allowed(
    *,
    source_decision: PreparationDecision,
    action: PreparationAction,
) -> None:
    """
    Si le planner expose déjà des candidate_actions,
    une modification utilisateur doit rester dans cet
    espace d'actions autorisées.

    Une future version pourra prévoir un mécanisme
    explicite d'override manuel hors contrat.
    """

    if not source_decision.candidate_actions:
        return

    if (
        action
        not in
        source_decision.candidate_actions
    ):
        allowed = [
            item.value
            for item in (
                source_decision
                .candidate_actions
            )
        ]

        raise ValueError(
            (
                f"L'action '{action.value}' n'est pas "
                "autorisée pour "
                f"{source_decision.decision_id}. "
                f"Actions autorisées : {allowed}"
            )
        )


# ============================================================
# AUTOMATIC STEP
# ============================================================


def _automatic_step(
    decision: PreparationDecision,
) -> ApprovedPreparationStep:
    """
    Les décisions AUTO_APPROVABLE avaient déjà été
    validées comme faible risque par le planner.

    Elles sont intégrées dans l'ApprovedPreparationPlan
    sans prétendre qu'une validation humaine a eu lieu.
    """

    action = (
        decision.selected_action
    )

    operation = (
        decision.selected_operation
    )

    if (
        action is None
        and
        operation is None
    ):
        raise ValueError(
            (
                "Une décision AUTO_APPROVABLE doit "
                "contenir selected_action ou "
                "selected_operation."
            )
        )

    if action is not None:
        (
            mutates_data,
            executor_eligible,
            requires_manual_followup,
        ) = (
            _action_characteristics(
                action
            )
        )

    else:
        # Une opération déterministe déjà validée par
        # le planner est destinée au futur executor.
        mutates_data = True
        executor_eligible = True
        requires_manual_followup = False

    return (
        ApprovedPreparationStep(
            approval_id=None,

            decision_id=
                decision.decision_id,

            source_issue_id=
                decision.source_issue_id,

            source_issue_kind=
                decision.source_issue_kind,

            dataset_id=
                decision.dataset_id,

            dataset_filename=
                decision.dataset_filename,

            column=
                decision.column,

            original_decision_status=
                decision.status,

            proposal_status=None,

            proposal_confidence=None,

            risk=
                decision.risk,

            recommended_action=
                decision.recommended_action,

            recommended_parameters=
                dict(
                    decision
                    .source_operation_parameters
                ),

            user_decision=None,

            approved_action=
                action,

            approved_parameters=
                dict(
                    decision
                    .source_operation_parameters
                ),

            approved_operation=
                operation,

            status=
                ApprovalStepStatus
                .AUTOMATIC,

            actor=None,

            decided_at=None,

            comment=None,

            rationale=(
                "Décision déterministe classée "
                "AUTO_APPROVABLE par le "
                "Preparation Planner."
            ),

            evidence=[],

            human_validation_performed=False,

            resolved=(
                not requires_manual_followup
            ),

            mutates_data=
                mutates_data,

            executor_eligible=
                executor_eligible,

            requires_manual_followup=
                requires_manual_followup,
        )
    )


# ============================================================
# PENDING STEP
# ============================================================


def _pending_step(
    *,
    decision: PreparationDecision,
    proposal: (
        ContextualPreparationProposal
        | None
    ),
    rationale: str,
) -> ApprovedPreparationStep:
    return (
        ApprovedPreparationStep(
            approval_id=None,

            decision_id=
                decision.decision_id,

            source_issue_id=
                decision.source_issue_id,

            source_issue_kind=
                decision.source_issue_kind,

            dataset_id=
                decision.dataset_id,

            dataset_filename=
                decision.dataset_filename,

            column=
                decision.column,

            original_decision_status=
                decision.status,

            proposal_status=(
                proposal.status
                if proposal
                is not None
                else None
            ),

            proposal_confidence=(
                proposal.confidence
                if proposal
                is not None
                else None
            ),

            risk=(
                proposal.risk
                if proposal
                is not None
                else decision.risk
            ),

            recommended_action=(
                proposal.recommended_action
                if proposal
                is not None
                else None
            ),

            recommended_parameters=(
                dict(
                    proposal
                    .recommended_parameters
                )
                if proposal
                is not None
                else {}
            ),

            user_decision=None,

            approved_action=None,

            approved_parameters={},

            approved_operation=None,

            status=
                ApprovalStepStatus
                .PENDING,

            actor=None,

            decided_at=None,

            comment=None,

            rationale=
                rationale,

            evidence=(
                list(
                    proposal.evidence
                )
                if proposal
                is not None
                else []
            ),

            human_validation_performed=False,

            resolved=False,

            mutates_data=False,

            executor_eligible=False,

            requires_manual_followup=False,
        )
    )


# ============================================================
# HUMAN DECISION
# ============================================================


def _apply_human_command(
    *,
    source_decision: PreparationDecision,
    proposal: ContextualPreparationProposal,
    command: PreparationApprovalCommand,
) -> ApprovedPreparationStep:
    decided_at = (
        command.decided_at
        or
        datetime.now(
            timezone.utc
        )
    )

    approval_id = (
        "approval:"
        +
        uuid4().hex
    )

    # ========================================================
    # DEFER
    # ========================================================

    if (
        command.decision
        ==
        ApprovalDecision.DEFER
    ):
        return (
            ApprovedPreparationStep(
                approval_id=
                    approval_id,

                decision_id=
                    source_decision
                    .decision_id,

                source_issue_id=
                    source_decision
                    .source_issue_id,

                source_issue_kind=
                    source_decision
                    .source_issue_kind,

                dataset_id=
                    source_decision
                    .dataset_id,

                dataset_filename=
                    source_decision
                    .dataset_filename,

                column=
                    source_decision.column,

                original_decision_status=
                    source_decision.status,

                proposal_status=
                    proposal.status,

                proposal_confidence=
                    proposal.confidence,

                risk=
                    proposal.risk,

                recommended_action=
                    proposal
                    .recommended_action,

                recommended_parameters=
                    dict(
                        proposal
                        .recommended_parameters
                    ),

                user_decision=
                    command.decision,

                approved_action=None,

                approved_parameters={},

                approved_operation=None,

                status=
                    ApprovalStepStatus
                    .DEFERRED,

                actor=
                    command.actor,

                decided_at=
                    decided_at,

                comment=
                    command.comment,

                rationale=(
                    "L'utilisateur a différé "
                    "la décision."
                ),

                evidence=
                    list(
                        proposal.evidence
                    ),

                human_validation_performed=True,

                resolved=False,

                mutates_data=False,

                executor_eligible=False,

                requires_manual_followup=False,
            )
        )

    # ========================================================
    # REJECT
    #
    # Reject means:
    # keep the source data unchanged.
    # ========================================================

    if (
        command.decision
        ==
        ApprovalDecision.REJECT
    ):
        return (
            ApprovedPreparationStep(
                approval_id=
                    approval_id,

                decision_id=
                    source_decision
                    .decision_id,

                source_issue_id=
                    source_decision
                    .source_issue_id,

                source_issue_kind=
                    source_decision
                    .source_issue_kind,

                dataset_id=
                    source_decision
                    .dataset_id,

                dataset_filename=
                    source_decision
                    .dataset_filename,

                column=
                    source_decision.column,

                original_decision_status=
                    source_decision.status,

                proposal_status=
                    proposal.status,

                proposal_confidence=
                    proposal.confidence,

                risk=
                    proposal.risk,

                recommended_action=
                    proposal
                    .recommended_action,

                recommended_parameters=
                    dict(
                        proposal
                        .recommended_parameters
                    ),

                user_decision=
                    command.decision,

                approved_action=
                    PreparationAction
                    .KEEP_AS_IS,

                approved_parameters={},

                approved_operation=None,

                status=
                    ApprovalStepStatus
                    .REJECTED,

                actor=
                    command.actor,

                decided_at=
                    decided_at,

                comment=
                    command.comment,

                rationale=(
                    "La proposition a été rejetée. "
                    "Les données concernées doivent "
                    "rester inchangées."
                ),

                evidence=
                    list(
                        proposal.evidence
                    ),

                human_validation_performed=True,

                resolved=True,

                mutates_data=False,

                executor_eligible=False,

                requires_manual_followup=False,
            )
        )

    # ========================================================
    # ABSTENTION GUARDRAIL
    # ========================================================

    if (
        proposal.status
        ==
        "abstained"
    ):
        raise ValueError(
            (
                "Une proposition au statut ABSTAINED "
                "ne peut pas être APPROVE ou MODIFY. "
                "Utiliser REJECT, DEFER ou fournir "
                "davantage de contexte."
            )
        )

    # ========================================================
    # APPROVE
    # ========================================================

    if (
        command.decision
        ==
        ApprovalDecision.APPROVE
    ):
        action = (
            proposal
            .recommended_action
        )

        if action is None:
            raise ValueError(
                (
                    "La proposition ne contient aucune "
                    "recommended_action à approuver."
                )
            )

        _validate_action_allowed(
            source_decision=
                source_decision,

            action=
                action,
        )

        parameters = dict(
            proposal
            .recommended_parameters
        )

        _validate_action_parameters(
            action=
                action,

            parameters=
                parameters,
        )

        (
            mutates_data,
            executor_eligible,
            requires_manual_followup,
        ) = (
            _action_characteristics(
                action
            )
        )

        status = (
            ApprovalStepStatus
            .MANUAL_FOLLOWUP
            if requires_manual_followup
            else
            ApprovalStepStatus
            .APPROVED
        )

        return (
            ApprovedPreparationStep(
                approval_id=
                    approval_id,

                decision_id=
                    source_decision
                    .decision_id,

                source_issue_id=
                    source_decision
                    .source_issue_id,

                source_issue_kind=
                    source_decision
                    .source_issue_kind,

                dataset_id=
                    source_decision
                    .dataset_id,

                dataset_filename=
                    source_decision
                    .dataset_filename,

                column=
                    source_decision.column,

                original_decision_status=
                    source_decision.status,

                proposal_status=
                    proposal.status,

                proposal_confidence=
                    proposal.confidence,

                risk=
                    proposal.risk,

                recommended_action=
                    action,

                recommended_parameters=
                    dict(
                        parameters
                    ),

                user_decision=
                    command.decision,

                approved_action=
                    action,

                approved_parameters=
                    dict(
                        parameters
                    ),

                approved_operation=None,

                status=
                    status,

                actor=
                    command.actor,

                decided_at=
                    decided_at,

                comment=
                    command.comment,

                rationale=(
                    (
                        "La recommandation DataLens "
                        "a été approuvée."
                    )
                    if not requires_manual_followup
                    else
                    (
                        "La recommandation a été "
                        "approuvée mais correspond à "
                        "une investigation ou revue "
                        "manuelle. La décision de "
                        "préparation reste donc ouverte."
                    )
                ),

                evidence=
                    list(
                        proposal.evidence
                    ),

                human_validation_performed=True,

                resolved=(
                    not requires_manual_followup
                ),

                mutates_data=
                    mutates_data,

                executor_eligible=
                    executor_eligible,

                requires_manual_followup=
                    requires_manual_followup,
            )
        )

    # ========================================================
    # MODIFY
    # ========================================================

    if (
        command.decision
        ==
        ApprovalDecision.MODIFY
    ):
        action = (
            command.modified_action
        )

        if action is None:
            raise ValueError(
                (
                    "MODIFY nécessite "
                    "modified_action."
                )
            )

        if (
            command.comment is None
            or
            not command.comment.strip()
        ):
            raise ValueError(
                (
                    "MODIFY nécessite un commentaire "
                    "expliquant la modification."
                )
            )

        _validate_action_allowed(
            source_decision=
                source_decision,

            action=
                action,
        )

        parameters = dict(
            command
            .modified_parameters
        )

        _validate_action_parameters(
            action=
                action,

            parameters=
                parameters,
        )

        (
            mutates_data,
            executor_eligible,
            requires_manual_followup,
        ) = (
            _action_characteristics(
                action
            )
        )

        status = (
            ApprovalStepStatus
            .MANUAL_FOLLOWUP
            if requires_manual_followup
            else
            ApprovalStepStatus
            .MODIFIED
        )

        return (
            ApprovedPreparationStep(
                approval_id=
                    approval_id,

                decision_id=
                    source_decision
                    .decision_id,

                source_issue_id=
                    source_decision
                    .source_issue_id,

                source_issue_kind=
                    source_decision
                    .source_issue_kind,

                dataset_id=
                    source_decision
                    .dataset_id,

                dataset_filename=
                    source_decision
                    .dataset_filename,

                column=
                    source_decision.column,

                original_decision_status=
                    source_decision.status,

                proposal_status=
                    proposal.status,

                proposal_confidence=
                    proposal.confidence,

                risk=
                    proposal.risk,

                recommended_action=
                    proposal
                    .recommended_action,

                recommended_parameters=
                    dict(
                        proposal
                        .recommended_parameters
                    ),

                user_decision=
                    command.decision,

                approved_action=
                    action,

                approved_parameters=
                    parameters,

                approved_operation=None,

                status=
                    status,

                actor=
                    command.actor,

                decided_at=
                    decided_at,

                comment=
                    command.comment,

                rationale=(
                    (
                        "L'utilisateur a remplacé la "
                        "recommandation DataLens par "
                        "une action explicitement "
                        "validée."
                    )
                    if not requires_manual_followup
                    else
                    (
                        "L'utilisateur a choisi une "
                        "action nécessitant encore une "
                        "investigation ou revue manuelle."
                    )
                ),

                evidence=
                    list(
                        proposal.evidence
                    ),

                human_validation_performed=True,

                resolved=(
                    not requires_manual_followup
                ),

                mutates_data=
                    mutates_data,

                executor_eligible=
                    executor_eligible,

                requires_manual_followup=
                    requires_manual_followup,
            )
        )

    raise ValueError(
        (
            "Décision d'approbation "
            "non supportée."
        )
    )


# ============================================================
# PUBLIC API
# ============================================================


def apply_preparation_approvals(
    *,
    plan: PreparationPlan,
    resolution_report: PreparationContextResolutionReport,
    commands: list[
        PreparationApprovalCommand
    ],
) -> ApprovedPreparationPlan:
    """
    Applique les décisions humaines au plan de préparation.

    IMPORTANT :

    Cette fonction :

    - ne modifie pas le DataFrame ;
    - ne modifie pas PreparationPlan ;
    - ne modifie pas le Context Resolver report ;
    - ne transforme aucune donnée ;
    - ne rend pas une investigation manuelle exécutable ;
    - conserve les décisions non traitées au statut PENDING.

    Le résultat ApprovedPreparationPlan sera le seul type
    que le futur Cleaning Executor devra accepter.
    """

    decisions = (
        _decision_index(
            plan
        )
    )

    proposals = (
        _proposal_index(
            resolution_report
        )
    )

    command_index = (
        _command_index(
            commands
        )
    )

    # ========================================================
    # VALIDATE PROPOSALS
    # ========================================================

    for decision_id in proposals:
        if decision_id not in decisions:
            raise ValueError(
                (
                    "Le Context Resolver référence une "
                    "décision absente du PreparationPlan : "
                    f"{decision_id}"
                )
            )

    # ========================================================
    # VALIDATE COMMAND TARGETS
    # ========================================================

    for decision_id in command_index:
        if decision_id not in decisions:
            raise ValueError(
                (
                    "Commande d'approbation pour une "
                    "décision inconnue : "
                    f"{decision_id}"
                )
            )

        source_decision = (
            decisions[
                decision_id
            ]
        )

        if (
            source_decision.status
            ==
            DecisionStatus.AUTO_APPROVABLE
        ):
            raise ValueError(
                (
                    "Les décisions AUTO_APPROVABLE "
                    "ne passent pas par l'approbation "
                    "humaine dans v0.1 : "
                    f"{decision_id}"
                )
            )

    # ========================================================
    # BUILD STEPS
    # ========================================================

    steps: list[
        ApprovedPreparationStep
    ] = []

    for decision in plan.decisions:
        # ----------------------------------------------------
        # Automatic deterministic decision.
        # ----------------------------------------------------

        if (
            decision.status
            ==
            DecisionStatus.AUTO_APPROVABLE
        ):
            steps.append(
                _automatic_step(
                    decision
                )
            )

            continue

        proposal = proposals.get(
            decision.decision_id
        )

        command = command_index.get(
            decision.decision_id
        )

        # ----------------------------------------------------
        # No resolver proposal.
        # ----------------------------------------------------

        if proposal is None:
            steps.append(
                _pending_step(
                    decision=
                        decision,

                    proposal=None,

                    rationale=(
                        "Aucune proposition du "
                        "Context Resolver n'est "
                        "disponible."
                    ),
                )
            )

            continue

        # ----------------------------------------------------
        # Proposal exists but user has not decided yet.
        # ----------------------------------------------------

        if command is None:
            steps.append(
                _pending_step(
                    decision=
                        decision,

                    proposal=
                        proposal,

                    rationale=(
                        "Une proposition existe mais "
                        "aucune décision utilisateur "
                        "n'a encore été enregistrée."
                    ),
                )
            )

            continue

        steps.append(
            _apply_human_command(
                source_decision=
                    decision,

                proposal=
                    proposal,

                command=
                    command,
            )
        )

    # ========================================================
    # COUNTS
    # ========================================================

    auto_approved_count = sum(
        1
        for step in steps
        if (
            step.status
            ==
            ApprovalStepStatus.AUTOMATIC
        )
    )

    approved_count = sum(
        1
        for step in steps
        if (
            step.user_decision
            ==
            ApprovalDecision.APPROVE
        )
    )

    modified_count = sum(
        1
        for step in steps
        if (
            step.user_decision
            ==
            ApprovalDecision.MODIFY
        )
    )

    rejected_count = sum(
        1
        for step in steps
        if (
            step.status
            ==
            ApprovalStepStatus.REJECTED
        )
    )

    deferred_count = sum(
        1
        for step in steps
        if (
            step.status
            ==
            ApprovalStepStatus.DEFERRED
        )
    )

    pending_count = sum(
        1
        for step in steps
        if (
            step.status
            ==
            ApprovalStepStatus.PENDING
        )
    )

    manual_followup_count = sum(
        1
        for step in steps
        if step.requires_manual_followup
    )

    execution_step_count = sum(
        1
        for step in steps
        if step.executor_eligible
    )

    no_change_count = sum(
        1
        for step in steps
        if (
            step.resolved
            and
            not step.mutates_data
            and
            not step.requires_manual_followup
        )
    )

    # ========================================================
    # EXECUTION READINESS
    # ========================================================

    ready_for_execution = (
        bool(
            steps
        )
        and
        all(
            step.resolved
            and
            not step.requires_manual_followup

            for step in steps
        )
        and
        pending_count == 0
        and
        deferred_count == 0
    )

    return (
        ApprovedPreparationPlan(
            total_step_count=
                len(
                    steps
                ),

            auto_approved_count=
                auto_approved_count,

            approved_count=
                approved_count,

            modified_count=
                modified_count,

            rejected_count=
                rejected_count,

            deferred_count=
                deferred_count,

            pending_count=
                pending_count,

            manual_followup_count=
                manual_followup_count,

            execution_step_count=
                execution_step_count,

            no_change_count=
                no_change_count,

            ready_for_execution=
                ready_for_execution,

            steps=
                steps,

            notes=[
                (
                    "Une recommandation DataLens "
                    "et une décision utilisateur "
                    "sont conservées séparément."
                ),
                (
                    "REJECT conserve les données "
                    "concernées sans modification."
                ),
                (
                    "DEFER laisse explicitement "
                    "la décision non résolue."
                ),
                (
                    "MODIFY est contrôlé par des "
                    "guardrails Python avant "
                    "d'être accepté."
                ),
                (
                    "Les actions d'investigation ou "
                    "de revue manuelle empêchent "
                    "ready_for_execution=True."
                ),
                (
                    "Une proposition ABSTAINED ne "
                    "peut pas être APPROVE ou MODIFY."
                ),
                (
                    "ApprovedPreparationPlan ne "
                    "modifie toujours aucune donnée."
                ),
            ],

            rule_version=
                PREPARATION_APPROVAL_RULE_VERSION,
        )
    )