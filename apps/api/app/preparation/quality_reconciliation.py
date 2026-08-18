from __future__ import annotations

from enum import Enum

from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.preparation.approval import (
    ApprovedPreparationPlan,
    ApprovedPreparationStep,
)

from app.preparation.contracts import (
    PreparationAction,
)

from app.preparation.data_quality import (
    DataQualityReport,
)


# ============================================================
# VERSION
# ============================================================


QUALITY_RECONCILIATION_RULE_VERSION = (
    "quality_reconciliation_v0.1"
)


# ============================================================
# STATES
# ============================================================


class QualityReconciliationState(
    str,
    Enum,
):
    RESOLVED = "resolved"

    EXPECTED_TO_REMAIN = (
        "expected_to_remain"
    )

    PERSISTED = "persisted"

    CHANGED = "changed"

    NEW = "new"


# ============================================================
# ACTION GROUPS
# ============================================================


NO_CHANGE_ACTIONS = {
    PreparationAction.KEEP_AS_IS,

    PreparationAction.KEEP_MISSING,

    PreparationAction.KEEP_DUPLICATE_ROWS,

    PreparationAction
    .KEEP_CATEGORIES_SEPARATE,
}


# ============================================================
# MODELS
# ============================================================


class QualityIssueSnapshot(
    BaseModel
):
    """
    Vue minimale et stable d'une issue qualité.

    On évite volontairement de recopier tout le contrat
    DataQualityIssue dans la couche de réconciliation.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    issue_id: str

    dataset_id: str

    dataset_filename: str

    column: (
        str
        | None
    ) = None

    kind: str

    severity: str

    title: str

    observed_count: int = Field(
        ge=0
    )

    affected_ratio: float = Field(
        ge=0.0
    )


class QualityIssueReconciliation(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    reconciliation_id: str

    dataset_id: str

    dataset_filename: str

    column: (
        str
        | None
    ) = None

    kind: str

    state: QualityReconciliationState

    before: (
        QualityIssueSnapshot
        | None
    ) = None

    after: (
        QualityIssueSnapshot
        | None
    ) = None

    approved_decision_id: (
        str
        | None
    ) = None

    approved_action: (
        PreparationAction
        | None
    ) = None

    direct_approval_found: bool = False

    expected_to_remain: bool = False

    issue_metrics_changed: bool = False

    aligned_with_approved_plan: bool

    requires_attention: bool

    rationale: str


class QualityReconciliationReport(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    status: Literal[
        "ready"
    ] = "ready"

    dataset_id: str

    dataset_filename: str

    before_issue_count: int = Field(
        ge=0
    )

    after_issue_count: int = Field(
        ge=0
    )

    resolved_count: int = Field(
        ge=0
    )

    expected_to_remain_count: int = Field(
        ge=0
    )

    persisted_count: int = Field(
        ge=0
    )

    changed_count: int = Field(
        ge=0
    )

    new_count: int = Field(
        ge=0
    )

    aligned_count: int = Field(
        ge=0
    )

    attention_required_count: int = Field(
        ge=0
    )

    unresolved_approved_action_count: int = Field(
        ge=0
    )

    unaddressed_persisted_count: int = Field(
        ge=0
    )

    ready_for_analysis: bool

    reconciliations: list[
        QualityIssueReconciliation
    ] = Field(
        default_factory=list
    )

    notes: list[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        QUALITY_RECONCILIATION_RULE_VERSION
    )


# ============================================================
# GENERIC VALUE HELPERS
# ============================================================


def _enum_value(
    value: Any,
) -> str:
    """
    Supporte à la fois les enums Python et les strings.
    """

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


def _normalize_optional_column(
    column: (
        str
        | None
    ),
) -> str:
    if column is None:
        return "__dataset__"

    return column.strip()


# ============================================================
# ISSUE SNAPSHOT
# ============================================================


def _issue_snapshot(
    issue,
) -> QualityIssueSnapshot:
    evidence = issue.evidence

    observed_count = int(
        getattr(
            evidence,
            "observed_count",
            0,
        )
    )

    affected_ratio = float(
        getattr(
            evidence,
            "affected_ratio",
            0.0,
        )
    )

    return (
        QualityIssueSnapshot(
            issue_id=
                issue.issue_id,

            dataset_id=
                issue.dataset_id,

            dataset_filename=
                issue.dataset_filename,

            column=
                issue.column,

            kind=
                _enum_value(
                    issue.kind
                ),

            severity=
                _enum_value(
                    issue.severity
                ),

            title=
                issue.title,

            observed_count=
                observed_count,

            affected_ratio=
                affected_ratio,
        )
    )


# ============================================================
# SEMANTIC SIGNATURE
# ============================================================


def _issue_signature(
    *,
    dataset_id: str,
    column: (
        str
        | None
    ),
    kind: str,
) -> tuple[
    str,
    str,
    str,
]:
    """
    issue_id est utile pour la provenance mais la
    réconciliation ne doit pas supposer qu'il restera
    identique après modification des données.

    La correspondance BEFORE/AFTER repose donc sur :

        dataset_id
        +
        column
        +
        kind
    """

    return (
        dataset_id,
        _normalize_optional_column(
            column
        ),
        kind.strip().lower(),
    )


def _snapshot_signature(
    snapshot: QualityIssueSnapshot,
) -> tuple[
    str,
    str,
    str,
]:
    return (
        _issue_signature(
            dataset_id=
                snapshot.dataset_id,

            column=
                snapshot.column,

            kind=
                snapshot.kind,
        )
    )


def _signature_text(
    signature: tuple[
        str,
        str,
        str,
    ],
) -> str:
    (
        dataset_id,
        column,
        kind,
    ) = signature

    return (
        f"{dataset_id}"
        f"|{column}"
        f"|{kind}"
    )


# ============================================================
# ISSUE INDEX
# ============================================================


def _build_issue_index(
    report: DataQualityReport,
) -> dict[
    tuple[
        str,
        str,
        str,
    ],
    QualityIssueSnapshot,
]:
    output: dict[
        tuple[
            str,
            str,
            str,
        ],
        QualityIssueSnapshot,
    ] = {}

    for issue in report.issues:
        snapshot = (
            _issue_snapshot(
                issue
            )
        )

        signature = (
            _snapshot_signature(
                snapshot
            )
        )

        if signature in output:
            raise ValueError(
                (
                    "Quality Reconciliation v0.1 "
                    "a trouvé plusieurs issues avec "
                    "la même signature : "
                    f"{_signature_text(signature)}"
                )
            )

        output[
            signature
        ] = snapshot

    return output


# ============================================================
# APPROVAL INDEX
# ============================================================


def _approved_step_by_issue_id(
    approved_plan: ApprovedPreparationPlan,
) -> dict[
    str,
    ApprovedPreparationStep,
]:
    output: dict[
        str,
        ApprovedPreparationStep,
    ] = {}

    for step in approved_plan.steps:
        issue_id = (
            step.source_issue_id
        )

        if issue_id in output:
            raise ValueError(
                (
                    "ApprovedPreparationPlan contient "
                    "plusieurs décisions pour "
                    f"source_issue_id={issue_id}."
                )
            )

        output[
            issue_id
        ] = step

    return output


def _approved_step_by_signature(
    approved_plan: ApprovedPreparationPlan,
) -> dict[
    tuple[
        str,
        str,
        str,
    ],
    ApprovedPreparationStep,
]:
    """
    Fallback lorsque le lien exact par issue_id
    n'est pas disponible.
    """

    output: dict[
        tuple[
            str,
            str,
            str,
        ],
        ApprovedPreparationStep,
    ] = {}

    for step in approved_plan.steps:
        signature = (
            _issue_signature(
                dataset_id=
                    step.dataset_id,

                column=
                    step.column,

                kind=
                    step.source_issue_kind,
            )
        )

        if signature in output:
            # issue_id reste le lien prioritaire.
            # On ne remplace pas silencieusement.
            continue

        output[
            signature
        ] = step

    return output


def _find_approved_step(
    *,
    before_issue: QualityIssueSnapshot,
    exact_index: dict[
        str,
        ApprovedPreparationStep,
    ],
    signature_index: dict[
        tuple[
            str,
            str,
            str,
        ],
        ApprovedPreparationStep,
    ],
) -> (
    ApprovedPreparationStep
    | None
):
    exact = exact_index.get(
        before_issue.issue_id
    )

    if exact is not None:
        return exact

    return signature_index.get(
        _snapshot_signature(
            before_issue
        )
    )


# ============================================================
# APPROVAL INTERPRETATION
# ============================================================


def _step_action(
    step: (
        ApprovedPreparationStep
        | None
    ),
) -> (
    PreparationAction
    | None
):
    if step is None:
        return None

    return step.approved_action


def _is_expected_to_remain(
    step: (
        ApprovedPreparationStep
        | None
    ),
) -> bool:
    if step is None:
        return False

    action = (
        _step_action(
            step
        )
    )

    if action is None:
        return False

    return (
        action
        in
        NO_CHANGE_ACTIONS
    )


def _approved_step_expected_to_mutate_issue(
    step: (
        ApprovedPreparationStep
        | None
    ),
) -> bool:
    if step is None:
        return False

    return bool(
        step.mutates_data
        and
        step.executor_eligible
    )


# ============================================================
# CHANGE DETECTION
# ============================================================


def _issue_metrics_changed(
    *,
    before: QualityIssueSnapshot,
    after: QualityIssueSnapshot,
) -> bool:
    if (
        before.observed_count
        !=
        after.observed_count
    ):
        return True

    if (
        abs(
            before.affected_ratio
            -
            after.affected_ratio
        )
        >
        1e-12
    ):
        return True

    if (
        before.severity
        !=
        after.severity
    ):
        return True

    return False


# ============================================================
# SINGLE RECONCILIATION
# ============================================================


def _reconcile_existing_issue(
    *,
    before: QualityIssueSnapshot,
    after: (
        QualityIssueSnapshot
        | None
    ),
    approved_step: (
        ApprovedPreparationStep
        | None
    ),
) -> QualityIssueReconciliation:
    signature = (
        _snapshot_signature(
            before
        )
    )

    reconciliation_id = (
        "quality-reconciliation:"
        +
        _signature_text(
            signature
        )
    )

    approved_action = (
        _step_action(
            approved_step
        )
    )

    expected_to_remain = (
        _is_expected_to_remain(
            approved_step
        )
    )

    mutation_expected = (
        _approved_step_expected_to_mutate_issue(
            approved_step
        )
    )

    # ========================================================
    # ISSUE DISAPPEARED
    # ========================================================

    if after is None:
        return (
            QualityIssueReconciliation(
                reconciliation_id=
                    reconciliation_id,

                dataset_id=
                    before.dataset_id,

                dataset_filename=
                    before.dataset_filename,

                column=
                    before.column,

                kind=
                    before.kind,

                state=
                    QualityReconciliationState
                    .RESOLVED,

                before=
                    before,

                after=None,

                approved_decision_id=(
                    approved_step.decision_id
                    if approved_step
                    is not None
                    else None
                ),

                approved_action=
                    approved_action,

                direct_approval_found=(
                    approved_step
                    is not None
                ),

                expected_to_remain=
                    expected_to_remain,

                issue_metrics_changed=False,

                # A resolved issue is considered safe
                # because only approved executor actions
                # were allowed to modify the dataset.
                aligned_with_approved_plan=True,

                requires_attention=False,

                rationale=(
                    (
                        "Le problème qualité n'est plus "
                        "détecté après la préparation "
                        "approuvée."
                    )
                    if not expected_to_remain
                    else
                    (
                        "Le problème n'est plus détecté. "
                        "Il avait été accepté comme pouvant "
                        "rester, mais une autre transformation "
                        "approuvée l'a également résolu."
                    )
                ),
            )
        )

    changed = (
        _issue_metrics_changed(
            before=
                before,

            after=
                after,
        )
    )

    # ========================================================
    # EXPLICIT KEEP / NO CHANGE
    # ========================================================

    if expected_to_remain:
        return (
            QualityIssueReconciliation(
                reconciliation_id=
                    reconciliation_id,

                dataset_id=
                    before.dataset_id,

                dataset_filename=
                    before.dataset_filename,

                column=
                    before.column,

                kind=
                    before.kind,

                state=(
                    QualityReconciliationState
                    .EXPECTED_TO_REMAIN
                ),

                before=
                    before,

                after=
                    after,

                approved_decision_id=(
                    approved_step.decision_id
                    if approved_step
                    is not None
                    else None
                ),

                approved_action=
                    approved_action,

                direct_approval_found=True,

                expected_to_remain=True,

                issue_metrics_changed=
                    changed,

                aligned_with_approved_plan=True,

                requires_attention=False,

                rationale=(
                    "Le problème est toujours détecté, "
                    "mais sa conservation correspond "
                    "explicitement à la décision approuvée."
                ),
            )
        )

    # ========================================================
    # MUTATING ACTION SHOULD HAVE RESOLVED IT
    # ========================================================

    if mutation_expected:
        state = (
            QualityReconciliationState
            .CHANGED
            if changed
            else
            QualityReconciliationState
            .PERSISTED
        )

        return (
            QualityIssueReconciliation(
                reconciliation_id=
                    reconciliation_id,

                dataset_id=
                    before.dataset_id,

                dataset_filename=
                    before.dataset_filename,

                column=
                    before.column,

                kind=
                    before.kind,

                state=
                    state,

                before=
                    before,

                after=
                    after,

                approved_decision_id=
                    approved_step.decision_id,

                approved_action=
                    approved_action,

                direct_approval_found=True,

                expected_to_remain=False,

                issue_metrics_changed=
                    changed,

                aligned_with_approved_plan=False,

                requires_attention=True,

                rationale=(
                    "Une transformation approuvée "
                    "devait traiter ce problème, mais "
                    "le moteur qualité le détecte encore "
                    "après exécution."
                ),
            )
        )

    # ========================================================
    # NO APPROVED DECISION
    # ========================================================

    state = (
        QualityReconciliationState
        .CHANGED
        if changed
        else
        QualityReconciliationState
        .PERSISTED
    )

    return (
        QualityIssueReconciliation(
            reconciliation_id=
                reconciliation_id,

            dataset_id=
                before.dataset_id,

            dataset_filename=
                before.dataset_filename,

            column=
                before.column,

            kind=
                before.kind,

            state=
                state,

            before=
                before,

            after=
                after,

            approved_decision_id=None,

            approved_action=None,

            direct_approval_found=False,

            expected_to_remain=False,

            issue_metrics_changed=
                changed,

            aligned_with_approved_plan=False,

            requires_attention=True,

            rationale=(
                "Le problème qualité reste présent "
                "mais aucune décision approuvée ne "
                "justifie explicitement sa conservation."
            ),
        )
    )


def _reconcile_new_issue(
    after: QualityIssueSnapshot,
) -> QualityIssueReconciliation:
    signature = (
        _snapshot_signature(
            after
        )
    )

    return (
        QualityIssueReconciliation(
            reconciliation_id=(
                "quality-reconciliation:"
                +
                _signature_text(
                    signature
                )
            ),

            dataset_id=
                after.dataset_id,

            dataset_filename=
                after.dataset_filename,

            column=
                after.column,

            kind=
                after.kind,

            state=
                QualityReconciliationState
                .NEW,

            before=None,

            after=
                after,

            approved_decision_id=None,

            approved_action=None,

            direct_approval_found=False,

            expected_to_remain=False,

            issue_metrics_changed=False,

            aligned_with_approved_plan=False,

            requires_attention=True,

            rationale=(
                "Ce problème qualité n'était pas "
                "détecté avant la préparation et "
                "apparaît dans le dataset final."
            ),
        )
    )


# ============================================================
# PUBLIC API
# ============================================================


def reconcile_data_quality(
    *,
    before_report: DataQualityReport,
    after_report: DataQualityReport,
    approved_plan: ApprovedPreparationPlan,
    dataset_id: str,
    dataset_filename: str,
) -> QualityReconciliationReport:
    """
    Compare la qualité déterministe avant/après nettoyage.

    IMPORTANT :

    cette fonction ne relance pas elle-même le moteur qualité.
    Elle consomme deux DataQualityReport produits par le même
    data_quality.py.

    Le caller fait donc :

        before_report = build_data_quality_report(...)
        cleaning
        after_report = build_data_quality_report(...)
        reconcile_data_quality(...)

    La réconciliation ne modifie aucune donnée.
    """

    before_index = (
        _build_issue_index(
            before_report
        )
    )

    after_index = (
        _build_issue_index(
            after_report
        )
    )

    exact_approval_index = (
        _approved_step_by_issue_id(
            approved_plan
        )
    )

    signature_approval_index = (
        _approved_step_by_signature(
            approved_plan
        )
    )

    reconciliations: list[
        QualityIssueReconciliation
    ] = []

    # ========================================================
    # BEFORE ISSUES
    # ========================================================

    for signature in sorted(
        before_index
    ):
        before_issue = (
            before_index[
                signature
            ]
        )

        if (
            before_issue.dataset_id
            !=
            dataset_id
        ):
            continue

        approved_step = (
            _find_approved_step(
                before_issue=
                    before_issue,

                exact_index=
                    exact_approval_index,

                signature_index=
                    signature_approval_index,
            )
        )

        after_issue = (
            after_index.get(
                signature
            )
        )

        reconciliations.append(
            _reconcile_existing_issue(
                before=
                    before_issue,

                after=
                    after_issue,

                approved_step=
                    approved_step,
            )
        )

    # ========================================================
    # NEW AFTER ISSUES
    # ========================================================

    for signature in sorted(
        after_index
    ):
        if signature in before_index:
            continue

        after_issue = (
            after_index[
                signature
            ]
        )

        if (
            after_issue.dataset_id
            !=
            dataset_id
        ):
            continue

        reconciliations.append(
            _reconcile_new_issue(
                after_issue
            )
        )

    # ========================================================
    # COUNTS
    # ========================================================

    def count_state(
        state: QualityReconciliationState,
    ) -> int:
        return sum(
            1
            for item in reconciliations
            if item.state == state
        )

    resolved_count = (
        count_state(
            QualityReconciliationState
            .RESOLVED
        )
    )

    expected_count = (
        count_state(
            QualityReconciliationState
            .EXPECTED_TO_REMAIN
        )
    )

    persisted_count = (
        count_state(
            QualityReconciliationState
            .PERSISTED
        )
    )

    changed_count = (
        count_state(
            QualityReconciliationState
            .CHANGED
        )
    )

    new_count = (
        count_state(
            QualityReconciliationState
            .NEW
        )
    )

    aligned_count = sum(
        1
        for item in reconciliations
        if item.aligned_with_approved_plan
    )

    attention_required_count = sum(
        1
        for item in reconciliations
        if item.requires_attention
    )

    unresolved_approved_action_count = sum(
        1
        for item in reconciliations
        if (
            item.requires_attention
            and
            item.direct_approval_found
        )
    )

    unaddressed_persisted_count = sum(
        1
        for item in reconciliations
        if (
            item.requires_attention
            and
            not item.direct_approval_found
            and
            item.state
            in {
                QualityReconciliationState
                .PERSISTED,

                QualityReconciliationState
                .CHANGED,
            }
        )
    )

    ready_for_analysis = (
        attention_required_count
        ==
        0
    )

    return (
        QualityReconciliationReport(
            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            before_issue_count=sum(
                1
                for issue in (
                    before_report.issues
                )
                if issue.dataset_id
                ==
                dataset_id
            ),

            after_issue_count=sum(
                1
                for issue in (
                    after_report.issues
                )
                if issue.dataset_id
                ==
                dataset_id
            ),

            resolved_count=
                resolved_count,

            expected_to_remain_count=
                expected_count,

            persisted_count=
                persisted_count,

            changed_count=
                changed_count,

            new_count=
                new_count,

            aligned_count=
                aligned_count,

            attention_required_count=
                attention_required_count,

            unresolved_approved_action_count=
                unresolved_approved_action_count,

            unaddressed_persisted_count=
                unaddressed_persisted_count,

            ready_for_analysis=
                ready_for_analysis,

            reconciliations=
                reconciliations,

            notes=[
                (
                    "Les rapports BEFORE et AFTER "
                    "proviennent du moteur déterministe "
                    "data_quality.py."
                ),
                (
                    "La correspondance des issues utilise "
                    "dataset_id + column + kind plutôt que "
                    "de supposer un issue_id stable."
                ),
                (
                    "Une anomalie explicitement conservée "
                    "par une décision KEEP est classée "
                    "EXPECTED_TO_REMAIN."
                ),
                (
                    "Une anomalie persistante sans décision "
                    "approuvée requiert une attention."
                ),
                (
                    "Une anomalie qui reste après une "
                    "transformation censée la traiter "
                    "requiert une attention."
                ),
                (
                    "Toute nouvelle anomalie détectée "
                    "après nettoyage requiert une attention."
                ),
                (
                    "ready_for_analysis=True exige "
                    "qu'aucune réconciliation ne nécessite "
                    "d'attention."
                ),
                (
                    "Cette couche ne modifie jamais "
                    "le DataFrame."
                ),
            ],

            rule_version=
                QUALITY_RECONCILIATION_RULE_VERSION,
        )
    )