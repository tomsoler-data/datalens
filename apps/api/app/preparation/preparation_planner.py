from __future__ import annotations

import re
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from app.preparation.contracts import (
    DecisionRisk,
    DecisionStatus,
    PreparationAction,
    PreparationDecision,
    PreparationPlan,
)

from app.preparation.data_quality import (
    DataQualityReport,
)

from app.preparation.semantic_review import (
    SemanticReviewReport,
    SemanticVerdict,
    ValidatedSemanticDecision,
)


# ============================================================
# VERSION
# ============================================================


PREPARATION_PLANNER_RULE_VERSION = (
    "preparation_planner_v0.2"
)


# ============================================================
# GENERIC HELPERS
# ============================================================


def _enum_value(
    value: Any,
) -> str:
    """
    Retourne proprement la valeur d'un Enum
    ou la représentation texte de la valeur.
    """

    if value is None:
        return ""

    enum_value = getattr(
        value,
        "value",
        None,
    )

    if enum_value is not None:
        return str(
            enum_value
        )

    return str(
        value
    )


def _model_to_dict(
    value: Any,
) -> Dict[str, Any]:
    """
    Convertit un modèle Pydantic ou un dictionnaire
    en dictionnaire JSON-compatible.
    """

    if value is None:
        return {}

    if isinstance(
        value,
        dict,
    ):
        return dict(
            value
        )

    model_dump = getattr(
        value,
        "model_dump",
        None,
    )

    if callable(
        model_dump
    ):
        dumped = model_dump(
            mode="json"
        )

        if isinstance(
            dumped,
            dict,
        ):
            return dumped

    return {
        "value":
            str(
                value
            )
    }


def _safe_identifier_part(
    value: str,
) -> str:
    normalized = (
        value
        .strip()
        .lower()
    )

    normalized = re.sub(
        r"[^a-z0-9_-]+",
        "_",
        normalized,
    )

    normalized = re.sub(
        r"_+",
        "_",
        normalized,
    )

    normalized = normalized.strip(
        "_"
    )

    return (
        normalized
        or
        "issue"
    )


def _build_decision_id(
    index: int,
    issue_id: str,
) -> str:
    return (
        f"prep:{index:04d}:"
        f"{_safe_identifier_part(issue_id)}"
    )


def _deduplicate_actions(
    actions: List[
        PreparationAction
    ],
) -> List[
    PreparationAction
]:
    output: List[
        PreparationAction
    ] = []

    for action in actions:
        if action in output:
            continue

        output.append(
            action
        )

    return output


# ============================================================
# RISK
# ============================================================


_RISK_ORDER = {
    DecisionRisk.LOW:
        1,

    DecisionRisk.MEDIUM:
        2,

    DecisionRisk.HIGH:
        3,
}


def _max_risk(
    left: DecisionRisk,
    right: DecisionRisk,
) -> DecisionRisk:
    if (
        _RISK_ORDER[
            right
        ]
        >
        _RISK_ORDER[
            left
        ]
    ):
        return right

    return left


def _risk_from_severity(
    severity: str,
) -> DecisionRisk:
    normalized = (
        severity
        .strip()
        .lower()
    )

    if normalized in {
        "important",
        "error",
        "critical",
        "high",
    }:
        return DecisionRisk.HIGH

    if normalized in {
        "moderate",
        "warning",
        "medium",
    }:
        return DecisionRisk.MEDIUM

    return DecisionRisk.LOW


def _risk_from_kind(
    kind: str,
) -> DecisionRisk:
    if kind in {
        "missing_values",
        "missing_identifier",
        "numeric_outliers",
        "invalid_numeric_values",
        "invalid_dates",
    }:
        return DecisionRisk.HIGH

    if kind in {
        "possible_semantic_aliases",
        "duplicate_rows",
    }:
        return DecisionRisk.MEDIUM

    return DecisionRisk.LOW


# ============================================================
# OPERATION NORMALIZATION
# ============================================================


def _action_from_operation(
    operation: Optional[str],
) -> Optional[
    PreparationAction
]:
    if operation is None:
        return None

    normalized = (
        operation
        .strip()
        .lower()
    )

    mapping = {
        "trim_whitespace":
            PreparationAction
            .TRIM_WHITESPACE,

        "trim_string_values":
            PreparationAction
            .TRIM_WHITESPACE,

        "normalize_empty_to_missing":
            PreparationAction
            .NORMALIZE_EMPTY_TO_MISSING,

        "normalize_missing_values":
            PreparationAction
            .NORMALIZE_MISSING_MARKERS,

        "normalize_missing_markers":
            PreparationAction
            .NORMALIZE_MISSING_MARKERS,

        "normalize_case":
            PreparationAction
            .NORMALIZE_CASE,

        "convert_to_numeric":
            PreparationAction
            .CONVERT_TO_NUMERIC,

        "remove_exact_duplicates":
            PreparationAction
            .REMOVE_DUPLICATE_ROWS,

        "remove_duplicate_rows":
            PreparationAction
            .REMOVE_DUPLICATE_ROWS,

        "merge_values":
            PreparationAction
            .MERGE_CATEGORY_VALUES,

        "keep_separate":
            PreparationAction
            .KEEP_CATEGORIES_SEPARATE,

        "keep_as_is":
            PreparationAction
            .KEEP_AS_IS,
    }

    return mapping.get(
        normalized
    )


# ============================================================
# CONTEXT REQUIREMENTS
# ============================================================


def _context_required_for_kind(
    kind: str,
) -> List[str]:
    if kind == "missing_values":
        return [
            (
                "Signification métier "
                "de la colonne"
            ),
            (
                "Signification possible "
                "de l'absence de valeur"
            ),
            (
                "Origine ou processus "
                "de collecte"
            ),
            (
                "Rôle de la variable "
                "dans l'analyse"
            ),
            (
                "Caractère éventuellement "
                "informatif des valeurs manquantes"
            ),
            (
                "Documentation métier "
                "disponible"
            ),
        ]

    if kind == "missing_identifier":
        return [
            (
                "Rôle exact de "
                "l'identifiant"
            ),
            (
                "Unicité attendue "
                "de la clé"
            ),
            (
                "Cause des identifiants "
                "manquants"
            ),
            (
                "Impact sur les jointures "
                "et la granularité"
            ),
        ]

    if kind == "numeric_outliers":
        return [
            (
                "Signification métier "
                "de la variable"
            ),
            "Unité de mesure",
            (
                "Bornes physiquement ou "
                "métier plausibles"
            ),
            "Origine de la donnée",
            "Distribution observée",
            (
                "Importance analytique "
                "des cas rares"
            ),
        ]

    if kind == "invalid_numeric_values":
        return [
            (
                "Format numérique "
                "attendu"
            ),
            (
                "Convention décimale "
                "ou monétaire"
            ),
            (
                "Signification des valeurs "
                "non convertibles"
            ),
            (
                "Documentation du système "
                "source"
            ),
        ]

    if kind == "invalid_dates":
        return [
            (
                "Format de date "
                "attendu"
            ),
            (
                "Convention de fuseau "
                "horaire éventuelle"
            ),
            (
                "Signification métier "
                "des dates invalides"
            ),
            (
                "Documentation du système "
                "source"
            ),
        ]

    if kind == "possible_semantic_aliases":
        return [
            (
                "Vérifier si les valeurs "
                "désignent réellement "
                "la même catégorie"
            ),
            (
                "Convention métier "
                "de nommage"
            ),
        ]

    if kind == "duplicate_rows":
        return [
            (
                "Granularité attendue "
                "du dataset"
            ),
            (
                "Présence éventuelle "
                "d'une clé primaire"
            ),
            (
                "Signification métier "
                "d'une ligne"
            ),
        ]

    return []


# ============================================================
# CANDIDATE ACTIONS
# ============================================================


def _candidate_actions_for_kind(
    kind: str,
) -> List[
    PreparationAction
]:
    if kind == "missing_values":
        return [
            PreparationAction
            .KEEP_MISSING,

            PreparationAction
            .DROP_ROWS_WITH_MISSING,

            PreparationAction
            .DROP_COLUMN,

            PreparationAction
            .IMPUTE_MEAN,

            PreparationAction
            .IMPUTE_MEDIAN,

            PreparationAction
            .IMPUTE_MODE,

            PreparationAction
            .CREATE_MISSING_CATEGORY,

            PreparationAction
            .DOMAIN_SPECIFIC_VALUE,
        ]

    if kind == "missing_identifier":
        return [
            PreparationAction
            .KEEP_MISSING,

            PreparationAction
            .DROP_ROWS_WITH_MISSING,

            PreparationAction
            .CONFIRM_IDENTIFIER,
        ]

    if kind == "numeric_outliers":
        return [
            PreparationAction
            .KEEP_AS_IS,

            PreparationAction
            .INVESTIGATE_OUTLIERS,

            PreparationAction
            .CAP_OUTLIERS,

            PreparationAction
            .REMOVE_OUTLIER_ROWS,
        ]

    if kind == "invalid_numeric_values":
        return [
            PreparationAction
            .KEEP_AS_IS,

            PreparationAction
            .REVIEW_INVALID_VALUES,

            PreparationAction
            .CONVERT_TO_NUMERIC,
        ]

    if kind == "invalid_dates":
        return [
            PreparationAction
            .KEEP_AS_IS,

            PreparationAction
            .REVIEW_INVALID_DATES,
        ]

    if kind == "possible_semantic_aliases":
        return [
            PreparationAction
            .KEEP_CATEGORIES_SEPARATE,

            PreparationAction
            .MERGE_CATEGORY_VALUES,

            PreparationAction
            .REVIEW_SEMANTIC_CONTEXT,
        ]

    if kind == "duplicate_rows":
        return [
            PreparationAction
            .REVIEW_DUPLICATES,

            PreparationAction
            .KEEP_DUPLICATE_ROWS,

            PreparationAction
            .REMOVE_DUPLICATE_ROWS,
        ]

    return [
        PreparationAction
        .KEEP_AS_IS,
    ]


# ============================================================
# SEMANTIC INDEX
# ============================================================


def _semantic_decision_index(
    *,
    semantic_report: Optional[
        SemanticReviewReport
    ],
    known_issue_ids: set[str],
) -> Dict[
    str,
    ValidatedSemanticDecision,
]:
    if semantic_report is None:
        return {}

    index: Dict[
        str,
        ValidatedSemanticDecision,
    ] = {}

    for decision in (
        semantic_report.decisions
    ):
        issue_id = (
            decision.issue_id
        )

        if issue_id in index:
            raise ValueError(
                (
                    "Le SemanticReviewReport "
                    "contient plusieurs décisions "
                    "pour le même issue_id : "
                    f"{issue_id}"
                )
            )

        if (
            issue_id
            not in known_issue_ids
        ):
            raise ValueError(
                (
                    "Le SemanticReviewReport "
                    "référence un issue_id inconnu : "
                    f"{issue_id}"
                )
            )

        index[
            issue_id
        ] = decision

    return index


# ============================================================
# QUALITY-ONLY DECISION
# ============================================================


def _quality_only_status(
    *,
    kind: str,
    semantic_review_recommended: bool,
    automatic_safe: bool,
    requires_user_confirmation: bool,
    source_operation: Optional[str],
) -> DecisionStatus:
    if semantic_review_recommended:
        return (
            DecisionStatus
            .NEEDS_CONTEXT
        )

    if (
        automatic_safe
        and
        not requires_user_confirmation
        and
        source_operation
    ):
        return (
            DecisionStatus
            .AUTO_APPROVABLE
        )

    if automatic_safe:
        return (
            DecisionStatus
            .REVIEW_REQUIRED
        )

    if kind in {
        "missing_values",
        "missing_identifier",
        "numeric_outliers",
        "invalid_numeric_values",
        "invalid_dates",
        "possible_semantic_aliases",
    }:
        return (
            DecisionStatus
            .NEEDS_CONTEXT
        )

    return (
        DecisionStatus
        .REVIEW_REQUIRED
    )


# ============================================================
# SEMANTIC INTERPRETATION
# ============================================================


def _apply_semantic_decision(
    *,
    decision: ValidatedSemanticDecision,
    candidate_actions: List[
        PreparationAction
    ],
    current_risk: DecisionRisk,
) -> tuple[
    DecisionStatus,
    DecisionRisk,
    Optional[PreparationAction],
    Optional[str],
    List[PreparationAction],
]:
    """
    Traduit une décision sémantique déjà validée par Python
    en décision de préparation.

    Important :
    semantic_review v0.3 reste non exécutable.
    Aucune action sémantique n'est sélectionnée ici.
    """

    verdict = (
        decision.verdict
    )

    if (
        verdict
        ==
        SemanticVerdict.MERGE_VALUES
    ):
        candidate_actions.append(
            PreparationAction
            .MERGE_CATEGORY_VALUES
        )

        return (
            DecisionStatus
            .REVIEW_REQUIRED,

            _max_risk(
                current_risk,
                DecisionRisk.MEDIUM,
            ),

            PreparationAction
            .MERGE_CATEGORY_VALUES,

            "merge_values",

            _deduplicate_actions(
                candidate_actions
            ),
        )

    if (
        verdict
        ==
        SemanticVerdict.KEEP_SEPARATE
    ):
        candidate_actions.append(
            PreparationAction
            .KEEP_CATEGORIES_SEPARATE
        )

        return (
            DecisionStatus
            .REVIEW_REQUIRED,

            current_risk,

            PreparationAction
            .KEEP_CATEGORIES_SEPARATE,

            "keep_separate",

            _deduplicate_actions(
                candidate_actions
            ),
        )

    if (
        verdict
        ==
        SemanticVerdict.FLAG_FOR_REVIEW
    ):
        candidate_actions.append(
            PreparationAction
            .REVIEW_SEMANTIC_CONTEXT
        )

        return (
            DecisionStatus
            .REVIEW_REQUIRED,

            _max_risk(
                current_risk,
                DecisionRisk.MEDIUM,
            ),

            PreparationAction
            .REVIEW_SEMANTIC_CONTEXT,

            None,

            _deduplicate_actions(
                candidate_actions
            ),
        )

    if (
        verdict
        ==
        SemanticVerdict.CONTEXTUALIZE
    ):
        candidate_actions.append(
            PreparationAction
            .REVIEW_SEMANTIC_CONTEXT
        )

        return (
            DecisionStatus
            .NEEDS_CONTEXT,

            current_risk,

            PreparationAction
            .REVIEW_SEMANTIC_CONTEXT,

            None,

            _deduplicate_actions(
                candidate_actions
            ),
        )

    if (
        verdict
        ==
        SemanticVerdict.NO_CHANGE
    ):
        candidate_actions.append(
            PreparationAction
            .KEEP_AS_IS
        )

        return (
            DecisionStatus
            .REVIEW_REQUIRED,

            current_risk,

            PreparationAction
            .KEEP_AS_IS,

            "keep_as_is",

            _deduplicate_actions(
                candidate_actions
            ),
        )

    # ABSTAIN
    candidate_actions.append(
        PreparationAction
        .REVIEW_SEMANTIC_CONTEXT
    )

    return (
        DecisionStatus
        .NEEDS_CONTEXT,

        _max_risk(
            current_risk,
            DecisionRisk.MEDIUM,
        ),

        None,

        None,

        _deduplicate_actions(
            candidate_actions
        ),
    )


# ============================================================
# SINGLE ISSUE
# ============================================================


def _build_decision(
    *,
    index: int,
    issue: Any,
    semantic_decision: Optional[
        ValidatedSemanticDecision
    ],
) -> PreparationDecision:
    issue_id = str(
        issue.issue_id
    )

    kind = _enum_value(
        issue.kind
    )

    severity = _enum_value(
        issue.severity
    )

    proposal = (
        issue.proposal
    )

    evidence = _model_to_dict(
        issue.evidence
    )

    source_operation = getattr(
        proposal,
        "operation",
        None,
    )

    if source_operation is not None:
        source_operation = str(
            source_operation
        )

    source_description = getattr(
        proposal,
        "description",
        None,
    )

    if source_description is not None:
        source_description = str(
            source_description
        )

    automatic_safe = bool(
        getattr(
            proposal,
            "automatic_safe",
            False,
        )
    )

    requires_confirmation = bool(
        getattr(
            proposal,
            "requires_user_confirmation",
            True,
        )
    )

    source_parameters = _model_to_dict(
        getattr(
            proposal,
            "parameters",
            {},
        )
    )

    semantic_review_recommended = bool(
        getattr(
            issue,
            "semantic_review_recommended",
            False,
        )
    )

    risk = _max_risk(
        _risk_from_severity(
            severity
        ),
        _risk_from_kind(
            kind
        ),
    )

    candidate_actions = (
        _candidate_actions_for_kind(
            kind
        )
    )

    source_action = (
        _action_from_operation(
            source_operation
        )
    )

    if (
        source_action
        is not None
    ):
        candidate_actions.append(
            source_action
        )

    candidate_actions = (
        _deduplicate_actions(
            candidate_actions
        )
    )

    context_required = (
        _context_required_for_kind(
            kind
        )
    )

    # ========================================================
    # SEMANTICALLY ENRICHED
    # ========================================================

    if semantic_decision is not None:
        (
            status,
            risk,
            recommended_action,
            recommended_operation,
            candidate_actions,
        ) = _apply_semantic_decision(
            decision=
                semantic_decision,

            candidate_actions=
                candidate_actions,

            current_risk=
                risk,
        )

        semantic_verdict = _enum_value(
            semantic_decision.verdict
        )

        rationale_parts = [
            str(
                issue.explanation
            ),
            (
                "La revue sémantique locale "
                "a été validée contre les "
                "preuves déterministes."
            ),
            semantic_decision.rationale,
        ]

        if (
            status
            ==
            DecisionStatus.NEEDS_CONTEXT
            and
            not context_required
        ):
            context_required = [
                (
                    "Contexte métier ou analytique "
                    "supplémentaire"
                )
            ]

        return PreparationDecision(
            decision_id=
                _build_decision_id(
                    index,
                    issue_id,
                ),

            source_issue_id=
                issue_id,

            source_issue_kind=
                kind,

            dataset_id=
                str(
                    issue.dataset_id
                ),

            dataset_filename=
                str(
                    issue.dataset_filename
                ),

            column=
                issue.column,

            severity=
                severity,

            title=
                str(
                    issue.title
                ),

            status=
                status,

            risk=
                risk,

            rationale=
                " ".join(
                    part.strip()
                    for part in rationale_parts
                    if (
                        isinstance(
                            part,
                            str,
                        )
                        and
                        part.strip()
                    )
                ),

            evidence=
                evidence,

            source_operation=
                source_operation,

            source_operation_description=
                source_description,

            source_operation_parameters=
                source_parameters,

            source_automatic_safe=
                automatic_safe,

            source_requires_user_confirmation=
                requires_confirmation,

            semantic_verdict=
                semantic_verdict,

            semantic_confidence=
                semantic_decision
                .confidence,

            semantic_rationale=
                semantic_decision
                .rationale,

            semantic_user_message=
                semantic_decision
                .user_message,

            semantic_source_values=
                list(
                    semantic_decision
                    .source_values
                ),

            semantic_canonical_value=
                semantic_decision
                .canonical_value,

            semantic_python_validated=
                semantic_decision
                .python_validated,

            semantic_executable=
                semantic_decision
                .executable,

            semantic_validation_notes=
                list(
                    semantic_decision
                    .validation_notes
                ),

            context_required=
                context_required,

            candidate_actions=
                candidate_actions,

            recommended_action=
                recommended_action,

            selected_action=None,

            recommended_operation=
                recommended_operation,

            selected_operation=None,

            requires_human_validation=True,
        )

    # ========================================================
    # QUALITY ONLY
    # ========================================================

    status = (
        _quality_only_status(
            kind=
                kind,

            semantic_review_recommended=
                semantic_review_recommended,

            automatic_safe=
                automatic_safe,

            requires_user_confirmation=
                requires_confirmation,

            source_operation=
                source_operation,
        )
    )

    selected_operation: Optional[
        str
    ] = None

    selected_action: Optional[
        PreparationAction
    ] = None

    recommended_operation = (
        source_operation
    )

    recommended_action = (
        source_action
    )

    requires_human_validation = (
        status
        !=
        DecisionStatus.AUTO_APPROVABLE
    )

    if (
        status
        ==
        DecisionStatus.AUTO_APPROVABLE
    ):
        selected_operation = (
            source_operation
        )

        selected_action = (
            source_action
        )

    if (
        status
        ==
        DecisionStatus.NEEDS_CONTEXT
        and
        not context_required
    ):
        context_required = [
            (
                "Contexte métier ou analytique "
                "supplémentaire"
            )
        ]

    rationale_parts = [
        str(
            issue.explanation
        )
    ]

    if semantic_review_recommended:
        rationale_parts.append(
            (
                "Le moteur déterministe recommande "
                "une revue sémantique avant toute "
                "décision de préparation."
            )
        )

    elif (
        status
        ==
        DecisionStatus.AUTO_APPROVABLE
    ):
        rationale_parts.append(
            (
                "Le moteur déterministe classe "
                "cette opération comme sûre et "
                "sans confirmation obligatoire."
            )
        )

    else:
        rationale_parts.append(
            (
                "La proposition déterministe "
                "nécessite une validation ou du "
                "contexte supplémentaire avant "
                "toute exécution."
            )
        )

    return PreparationDecision(
        decision_id=
            _build_decision_id(
                index,
                issue_id,
            ),

        source_issue_id=
            issue_id,

        source_issue_kind=
            kind,

        dataset_id=
            str(
                issue.dataset_id
            ),

        dataset_filename=
            str(
                issue.dataset_filename
            ),

        column=
            issue.column,

        severity=
            severity,

        title=
            str(
                issue.title
            ),

        status=
            status,

        risk=
            risk,

        rationale=
            " ".join(
                part.strip()
                for part in rationale_parts
                if part.strip()
            ),

        evidence=
            evidence,

        source_operation=
            source_operation,

        source_operation_description=
            source_description,

        source_operation_parameters=
            source_parameters,

        source_automatic_safe=
            automatic_safe,

        source_requires_user_confirmation=
            requires_confirmation,

        semantic_verdict=None,

        semantic_confidence=None,

        semantic_rationale=None,

        semantic_user_message=None,

        semantic_source_values=[],

        semantic_canonical_value=None,

        semantic_python_validated=False,

        semantic_executable=False,

        semantic_validation_notes=[],

        context_required=
            context_required,

        candidate_actions=
            candidate_actions,

        recommended_action=
            recommended_action,

        selected_action=
            selected_action,

        recommended_operation=
            recommended_operation,

        selected_operation=
            selected_operation,

        requires_human_validation=
            requires_human_validation,
    )


# ============================================================
# PUBLIC API
# ============================================================


def build_preparation_plan(
    *,
    quality_report: DataQualityReport,
    semantic_report: Optional[
        SemanticReviewReport
    ] = None,
) -> PreparationPlan:
    """
    Construire un plan unifié de préparation à partir de :

    1. DataQualityReport :
       faits déterministes + propositions techniques ;

    2. SemanticReviewReport :
       interprétation locale validée par Python
       pour les problèmes ambigus.

    Le planner :

    - ne modifie jamais un DataFrame ;
    - ne transforme jamais une proposition sémantique
      non exécutable en action exécutée ;
    - relie qualité et sémantique uniquement par issue_id ;
    - peut explicitement conclure NEEDS_CONTEXT.
    """

    issues = list(
        quality_report.issues
    )

    known_issue_ids = {
        str(
            issue.issue_id
        )
        for issue in issues
    }

    if (
        len(
            known_issue_ids
        )
        !=
        len(
            issues
        )
    ):
        raise ValueError(
            (
                "Le DataQualityReport contient "
                "des issue_id dupliqués."
            )
        )

    semantic_index = (
        _semantic_decision_index(
            semantic_report=
                semantic_report,

            known_issue_ids=
                known_issue_ids,
        )
    )

    decisions: List[
        PreparationDecision
    ] = []

    for (
        index,
        issue,
    ) in enumerate(
        issues,
        start=1,
    ):
        semantic_decision = (
            semantic_index.get(
                str(
                    issue.issue_id
                )
            )
        )

        decisions.append(
            _build_decision(
                index=
                    index,

                issue=
                    issue,

                semantic_decision=
                    semantic_decision,
            )
        )

    auto_approvable_count = sum(
        1
        for decision in decisions
        if (
            decision.status
            ==
            DecisionStatus.AUTO_APPROVABLE
        )
    )

    review_required_count = sum(
        1
        for decision in decisions
        if (
            decision.status
            ==
            DecisionStatus.REVIEW_REQUIRED
        )
    )

    needs_context_count = sum(
        1
        for decision in decisions
        if (
            decision.status
            ==
            DecisionStatus.NEEDS_CONTEXT
        )
    )

    unresolved_count = sum(
        1
        for decision in decisions
        if (
            decision.status
            !=
            DecisionStatus.AUTO_APPROVABLE
            or
            decision.selected_operation
            is None
        )
    )

    ready_for_execution = (
        unresolved_count
        ==
        0
    )

    notes = [
        (
            "Le DataQualityReport reste la source "
            "déterministe des problèmes de qualité."
        ),
        (
            "Les décisions sémantiques sont raccordées "
            "aux problèmes uniquement par issue_id."
        ),
        (
            "Une proposition sémantique de semantic_review "
            "reste non exécutable dans Preparation Planner v0.2."
        ),
        (
            "NEEDS_CONTEXT est un résultat valide lorsque "
            "les données disponibles ne permettent pas "
            "une décision défendable."
        ),
    ]

    return PreparationPlan(
        quality_issue_count=
            len(
                issues
            ),

        semantic_decision_count=(
            len(
                semantic_report.decisions
            )
            if semantic_report
            is not None
            else 0
        ),

        semantic_enriched_count=
            len(
                semantic_index
            ),

        total_decisions=
            len(
                decisions
            ),

        auto_approvable_count=
            auto_approvable_count,

        review_required_count=
            review_required_count,

        needs_context_count=
            needs_context_count,

        unresolved_count=
            unresolved_count,

        ready_for_execution=
            ready_for_execution,

        decisions=
            decisions,

        notes=
            notes,

        rule_version=
            PREPARATION_PLANNER_RULE_VERSION,
    )