from __future__ import annotations

import re
import unicodedata

from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.preparation.context_enrichment import (
    PreparationContextAssessment,
    PreparationContextReport,
)

from app.preparation.contracts import (
    DecisionRisk,
    PreparationAction,
    PreparationDecision,
    PreparationPlan,
)

from app.preparation.preparation_rag_context import (
    PreparationRagContext,
    PreparationRagContextReport,
    PreparationRagEvidence,
)


# ============================================================
# VERSION
# ============================================================


PREPARATION_CONTEXT_RESOLVER_RULE_VERSION = (
    "preparation_context_resolver_v0.1"
)


# ============================================================
# TYPES
# ============================================================


ResolutionStatus = Literal[
    "proposal_ready",
    "abstained",
]


ResolutionConfidence = Literal[
    "low",
    "medium",
    "high",
]


# ============================================================
# MODELS
# ============================================================


class ResolutionEvidenceReference(
    BaseModel
):
    """
    Référence vers une preuve documentaire réellement
    utilisée par le resolver.

    Le resolver ne crée pas de nouvelle preuve :
    il référence uniquement celles validées par
    Preparation RAG Context.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    chunk_id: str

    document_id: str

    filename: str

    source_locator: str

    evidence_type: str

    evidence_text: str

    retrieval_score: float

    final_score: int


class ContextualPreparationProposal(
    BaseModel
):
    """
    Proposition contextualisée de préparation.

    Important :

    - recommended_action = ce que DataLens propose ;
    - selected_action = décision réellement approuvée.

    En v0.1 selected_action reste toujours None.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    decision_id: str

    source_issue_id: str

    source_issue_kind: str

    dataset_id: str

    dataset_filename: str

    column: (
        str
        | None
    ) = None

    status: ResolutionStatus

    confidence: ResolutionConfidence

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

    selected_action: (
        PreparationAction
        | None
    ) = None

    selected_parameters: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    rationale: str

    evidence: list[
        ResolutionEvidenceReference
    ] = Field(
        default_factory=list
    )

    analytical_context_found: bool = False

    analytical_roles: list[
        str
    ] = Field(
        default_factory=list
    )

    high_analytical_impact: bool = False

    business_context_found: bool = False

    human_validation_required: bool = True

    executable: bool = False

    abstention_reason: (
        str
        | None
    ) = None


class PreparationContextResolutionReport(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    status: Literal[
        "ready"
    ] = "ready"

    decision_count: int = Field(
        ge=0
    )

    proposal_count: int = Field(
        ge=0
    )

    abstention_count: int = Field(
        ge=0
    )

    documented_business_rule_count: int = Field(
        ge=0
    )

    high_analytical_impact_count: int = Field(
        ge=0
    )

    proposals: list[
        ContextualPreparationProposal
    ] = Field(
        default_factory=list
    )

    notes: list[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        PREPARATION_CONTEXT_RESOLVER_RULE_VERSION
    )


# ============================================================
# NORMALIZATION
# ============================================================


def normalize_text(
    value: str,
) -> str:
    normalized = unicodedata.normalize(
        "NFKD",
        value,
    )

    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(
            character
        )
    )

    normalized = (
        normalized
        .lower()
        .replace(
            "_",
            " ",
        )
        .replace(
            "-",
            " ",
        )
    )

    normalized = re.sub(
        r"[^a-z0-9%.,<>]+",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


# ============================================================
# INDEXES
# ============================================================


def _analytical_context_index(
    report: (
        PreparationContextReport
        | None
    ),
) -> dict[
    str,
    PreparationContextAssessment,
]:
    if report is None:
        return {}

    output: dict[
        str,
        PreparationContextAssessment,
    ] = {}

    for assessment in report.assessments:
        if (
            assessment.decision_id
            in output
        ):
            raise ValueError(
                (
                    "PreparationContextReport contient "
                    "plusieurs assessments pour "
                    f"{assessment.decision_id}."
                )
            )

        output[
            assessment.decision_id
        ] = assessment

    return output


def _rag_context_index(
    report: PreparationRagContextReport,
) -> dict[
    str,
    PreparationRagContext,
]:
    output: dict[
        str,
        PreparationRagContext,
    ] = {}

    for context in report.contexts:
        if (
            context.decision_id
            in output
        ):
            raise ValueError(
                (
                    "PreparationRagContextReport contient "
                    "plusieurs contextes pour "
                    f"{context.decision_id}."
                )
            )

        output[
            context.decision_id
        ] = context

    return output


# ============================================================
# EVIDENCE
# ============================================================


def _evidence_reference(
    evidence: PreparationRagEvidence,
) -> ResolutionEvidenceReference:
    return (
        ResolutionEvidenceReference(
            chunk_id=
                evidence.chunk_id,

            document_id=
                evidence.document_id,

            filename=
                evidence.filename,

            source_locator=
                evidence.source_locator,

            evidence_type=
                evidence.evidence_type,

            evidence_text=
                evidence.evidence_text,

            retrieval_score=
                evidence.retrieval_score,

            final_score=
                evidence.final_score,
        )
    )


def _direct_rules(
    rag_context: PreparationRagContext,
) -> list[
    PreparationRagEvidence
]:
    return [
        evidence
        for evidence in rag_context.evidence
        if (
            evidence.evidence_type
            ==
            "direct_rule"
        )
    ]


def _guardrails(
    rag_context: PreparationRagContext,
) -> list[
    PreparationRagEvidence
]:
    return [
        evidence
        for evidence in rag_context.evidence
        if (
            evidence.evidence_type
            ==
            "guardrail"
        )
    ]


def _supporting_context(
    rag_context: PreparationRagContext,
) -> list[
    PreparationRagEvidence
]:
    return [
        evidence
        for evidence in rag_context.evidence
        if (
            evidence.evidence_type
            ==
            "supporting_context"
        )
    ]


# ============================================================
# NUMERIC VALUE EXTRACTION
# ============================================================


def _parse_number(
    raw_value: str,
) -> float:
    cleaned = (
        raw_value
        .strip()
        .replace(
            ",",
            ".",
        )
    )

    return float(
        cleaned
    )


def _clean_number(
    value: float,
) -> (
    int
    | float
):
    if value.is_integer():
        return int(
            value
        )

    return value


def extract_documented_missing_numeric_value(
    evidence_text: str,
) -> tuple[
    (
        int
        | float
        | None
    ),
    (
        str
        | None
    ),
]:
    """
    Extrait uniquement les valeurs numériques explicitement
    présentées comme interprétation métier d'une valeur absente.

    Exemple :

        "Une valeur vide ... doit être interprétée
         comme une remise de 0 %."

        -> value = 0
        -> unit = percent

    Cette fonction n'invente jamais de valeur.
    """

    normalized = normalize_text(
        evidence_text
    )

    interpretation_markers = [
        "interpretee comme",
        "interprete comme",
        "equivaut a",
        "correspond a",
    ]

    marker_position = None

    for marker in interpretation_markers:
        position = normalized.find(
            normalize_text(
                marker
            )
        )

        if position < 0:
            continue

        if (
            marker_position is None
            or
            position
            <
            marker_position
        ):
            marker_position = (
                position
            )

    if marker_position is None:
        return (
            None,
            None,
        )

    relevant_text = normalized[
        marker_position:
    ]

    match = re.search(
        r"(-?\d+(?:[.,]\d+)?)\s*(%)?",
        relevant_text,
    )

    if match is None:
        return (
            None,
            None,
        )

    value = _clean_number(
        _parse_number(
            match.group(
                1
            )
        )
    )

    unit = (
        "percent"
        if (
            match.group(
                2
            )
            ==
            "%"
        )
        else None
    )

    return (
        value,
        unit,
    )


# ============================================================
# THRESHOLD EXTRACTION
# ============================================================


def extract_documented_threshold(
    evidence_text: str,
) -> tuple[
    (
        str
        | None
    ),
    (
        int
        | float
        | None
    ),
]:
    """
    Extrait des règles simples et explicitement documentées.

    Exemples :

        "supérieures à 120"
            -> ">", 120

        "inférieures à 0"
            -> "<", 0

    Aucun seuil n'est déduit statistiquement ici.
    """

    normalized = normalize_text(
        evidence_text
    )

    patterns = [
        (
            r"superieures?\s+a\s+"
            r"(-?\d+(?:[.,]\d+)?)",
            ">",
        ),
        (
            r"au dessus de\s+"
            r"(-?\d+(?:[.,]\d+)?)",
            ">",
        ),
        (
            r"inferieures?\s+a\s+"
            r"(-?\d+(?:[.,]\d+)?)",
            "<",
        ),
        (
            r"en dessous de\s+"
            r"(-?\d+(?:[.,]\d+)?)",
            "<",
        ),
    ]

    for (
        pattern,
        comparator,
    ) in patterns:
        match = re.search(
            pattern,
            normalized,
        )

        if match is None:
            continue

        value = _clean_number(
            _parse_number(
                match.group(
                    1
                )
            )
        )

        return (
            comparator,
            value,
        )

    return (
        None,
        None,
    )


# ============================================================
# ANALYTICAL IMPACT
# ============================================================


def _analytical_values(
    analytical_context: (
        PreparationContextAssessment
        | None
    ),
) -> tuple[
    bool,
    list[str],
    bool,
]:
    if analytical_context is None:
        return (
            False,
            [],
            False,
        )

    return (
        analytical_context
        .used_in_validated_analysis,

        list(
            analytical_context
            .usage_roles
        ),

        analytical_context
        .high_analytical_impact,
    )


# ============================================================
# RISK
# ============================================================


def _resolved_risk(
    *,
    decision: PreparationDecision,
    high_analytical_impact: bool,
) -> DecisionRisk:
    """
    Le resolver ne réduit jamais le risque initial.

    Une forte importance analytique peut au contraire
    empêcher de banaliser une transformation documentée.
    """

    if high_analytical_impact:
        if (
            decision.risk
            ==
            DecisionRisk.LOW
        ):
            return (
                DecisionRisk.MEDIUM
            )

    return decision.risk


# ============================================================
# ABSTENTION
# ============================================================


def _abstain(
    *,
    decision: PreparationDecision,
    analytical_context: (
        PreparationContextAssessment
        | None
    ),
    reason: str,
) -> ContextualPreparationProposal:
    (
        analytical_found,
        roles,
        high_impact,
    ) = _analytical_values(
        analytical_context
    )

    return (
        ContextualPreparationProposal(
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

            status=
                "abstained",

            confidence=
                "low",

            risk=
                _resolved_risk(
                    decision=
                        decision,

                    high_analytical_impact=
                        high_impact,
                ),

            recommended_action=None,

            recommended_parameters={},

            selected_action=None,

            selected_parameters={},

            rationale=(
                "DataLens ne dispose pas de preuves "
                "suffisantes pour produire une "
                "proposition contextualisée défendable."
            ),

            evidence=[],

            analytical_context_found=
                analytical_found,

            analytical_roles=
                roles,

            high_analytical_impact=
                high_impact,

            business_context_found=False,

            human_validation_required=True,

            executable=False,

            abstention_reason=
                reason,
        )
    )


# ============================================================
# MISSING VALUES
# ============================================================


def _resolve_missing_values(
    *,
    decision: PreparationDecision,
    rag_context: PreparationRagContext,
    analytical_context: (
        PreparationContextAssessment
        | None
    ),
) -> ContextualPreparationProposal:
    direct_rules = (
        _direct_rules(
            rag_context
        )
    )

    guardrails = (
        _guardrails(
            rag_context
        )
    )

    supporting = (
        _supporting_context(
            rag_context
        )
    )

    if not direct_rules:
        return (
            _abstain(
                decision=
                    decision,

                analytical_context=
                    analytical_context,

                reason=(
                    "Aucune règle métier directe "
                    "n'explique la signification "
                    "de la valeur manquante."
                ),
            )
        )

    (
        analytical_found,
        roles,
        high_impact,
    ) = _analytical_values(
        analytical_context
    )

    # ========================================================
    # CASE 1
    #
    # Explicit domain-specific numeric value.
    #
    # Example:
    # missing discount_rate = 0 %
    # ========================================================

    for rule in direct_rules:
        (
            documented_value,
            documented_unit,
        ) = (
            extract_documented_missing_numeric_value(
                rule.evidence_text
            )
        )

        if documented_value is None:
            continue

        parameters: dict[
            str,
            Any,
        ] = {
            "value":
                documented_value,
        }

        if documented_unit is not None:
            parameters[
                "unit"
            ] = documented_unit

        used_evidence = [
            rule,
        ]

        used_evidence.extend(
            supporting[
                :1
            ]
        )

        used_evidence.extend(
            guardrails[
                :1
            ]
        )

        return (
            ContextualPreparationProposal(
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

                status=
                    "proposal_ready",

                confidence=
                    "high",

                risk=
                    _resolved_risk(
                        decision=
                            decision,

                        high_analytical_impact=
                            high_impact,
                    ),

                recommended_action=
                    PreparationAction
                    .DOMAIN_SPECIFIC_VALUE,

                recommended_parameters=
                    parameters,

                selected_action=None,

                selected_parameters={},

                rationale=(
                    "Une règle métier directe documente "
                    "explicitement la valeur correspondant "
                    "à l'absence de donnée. DataLens peut "
                    "donc proposer une valeur spécifique au "
                    "domaine, sans l'appliquer automatiquement."
                ),

                evidence=[
                    _evidence_reference(
                        evidence
                    )

                    for evidence in (
                        used_evidence
                    )
                ],

                analytical_context_found=
                    analytical_found,

                analytical_roles=
                    roles,

                high_analytical_impact=
                    high_impact,

                business_context_found=True,

                human_validation_required=True,

                executable=False,

                abstention_reason=None,
            )
        )

    # ========================================================
    # CASE 2
    #
    # Missingness has documented business meaning,
    # but no explicit replacement value exists.
    #
    # Safest v0.1 recommendation:
    # preserve missingness.
    # ========================================================

    used_evidence = (
        direct_rules[
            :1
        ]
        +
        guardrails[
            :1
        ]
    )

    return (
        ContextualPreparationProposal(
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

            status=
                "proposal_ready",

            confidence=
                "high",

            risk=
                _resolved_risk(
                    decision=
                        decision,

                    high_analytical_impact=
                        high_impact,
                ),

            recommended_action=
                PreparationAction
                .KEEP_MISSING,

            recommended_parameters={
                "missingness_is_informative":
                    True,
            },

            selected_action=None,

            selected_parameters={},

            rationale=(
                "La documentation attribue une "
                "signification métier explicite à "
                "l'absence de valeur mais ne fournit "
                "aucune valeur de remplacement. "
                "La recommandation la plus prudente "
                "est donc de préserver cette absence "
                "comme information."
            ),

            evidence=[
                _evidence_reference(
                    evidence
                )

                for evidence in (
                    used_evidence
                )
            ],

            analytical_context_found=
                analytical_found,

            analytical_roles=
                roles,

            high_analytical_impact=
                high_impact,

            business_context_found=True,

            human_validation_required=True,

            executable=False,

            abstention_reason=None,
        )
    )


# ============================================================
# NUMERIC OUTLIERS
# ============================================================


def _resolve_numeric_outliers(
    *,
    decision: PreparationDecision,
    rag_context: PreparationRagContext,
    analytical_context: (
        PreparationContextAssessment
        | None
    ),
) -> ContextualPreparationProposal:
    direct_rules = (
        _direct_rules(
            rag_context
        )
    )

    if not direct_rules:
        return (
            _abstain(
                decision=
                    decision,

                analytical_context=
                    analytical_context,

                reason=(
                    "Aucune règle métier directe "
                    "ne documente les bornes ou "
                    "valeurs atypiques."
                ),
            )
        )

    (
        analytical_found,
        roles,
        high_impact,
    ) = _analytical_values(
        analytical_context
    )

    for rule in direct_rules:
        (
            comparator,
            threshold,
        ) = (
            extract_documented_threshold(
                rule.evidence_text
            )
        )

        if (
            comparator is None
            or
            threshold is None
        ):
            continue

        return (
            ContextualPreparationProposal(
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

                status=
                    "proposal_ready",

                confidence=
                    "high",

                risk=
                    DecisionRisk.HIGH,

                recommended_action=
                    PreparationAction
                    .INVESTIGATE_OUTLIERS,

                recommended_parameters={
                    "documented_comparator":
                        comparator,

                    "documented_threshold":
                        threshold,
                },

                selected_action=None,

                selected_parameters={},

                rationale=(
                    "La documentation fournit une "
                    "borne métier explicite pour la "
                    "variable. DataLens recommande "
                    "d'investiguer les observations "
                    "qui dépassent cette borne mais "
                    "ne recommande ni suppression ni "
                    "plafonnement automatique."
                ),

                evidence=[
                    _evidence_reference(
                        rule
                    )
                ],

                analytical_context_found=
                    analytical_found,

                analytical_roles=
                    roles,

                high_analytical_impact=
                    high_impact,

                business_context_found=True,

                human_validation_required=True,

                executable=False,

                abstention_reason=None,
            )
        )

    return (
        _abstain(
            decision=
                decision,

            analytical_context=
                analytical_context,

            reason=(
                "La documentation parle des valeurs "
                "atypiques mais aucune borne métier "
                "simple et explicitement exploitable "
                "n'a pu être extraite."
            ),
        )
    )


# ============================================================
# INVALID DATES
# ============================================================


def _resolve_invalid_dates(
    *,
    decision: PreparationDecision,
    rag_context: PreparationRagContext,
    analytical_context: (
        PreparationContextAssessment
        | None
    ),
) -> ContextualPreparationProposal:
    direct_rules = (
        _direct_rules(
            rag_context
        )
    )

    if not direct_rules:
        return (
            _abstain(
                decision=
                    decision,

                analytical_context=
                    analytical_context,

                reason=(
                    "Aucune règle métier directe "
                    "sur les dates invalides."
                ),
            )
        )

    (
        analytical_found,
        roles,
        high_impact,
    ) = _analytical_values(
        analytical_context
    )

    rule = direct_rules[
        0
    ]

    return (
        ContextualPreparationProposal(
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

            status=
                "proposal_ready",

            confidence=
                "medium",

            risk=
                DecisionRisk.HIGH,

            recommended_action=
                PreparationAction
                .REVIEW_INVALID_DATES,

            recommended_parameters={},

            selected_action=None,

            selected_parameters={},

            rationale=(
                "Une règle documentaire confirme "
                "que les dates invalides nécessitent "
                "une vérification de la source. "
                "Aucune date de remplacement n'est "
                "inventée."
            ),

            evidence=[
                _evidence_reference(
                    rule
                )
            ],

            analytical_context_found=
                analytical_found,

            analytical_roles=
                roles,

            high_analytical_impact=
                high_impact,

            business_context_found=True,

            human_validation_required=True,

            executable=False,

            abstention_reason=None,
        )
    )


# ============================================================
# INVALID NUMERIC VALUES
# ============================================================


def _resolve_invalid_numeric_values(
    *,
    decision: PreparationDecision,
    rag_context: PreparationRagContext,
    analytical_context: (
        PreparationContextAssessment
        | None
    ),
) -> ContextualPreparationProposal:
    direct_rules = (
        _direct_rules(
            rag_context
        )
    )

    if not direct_rules:
        return (
            _abstain(
                decision=
                    decision,

                analytical_context=
                    analytical_context,

                reason=(
                    "Aucune règle métier directe "
                    "sur les valeurs numériques "
                    "invalides."
                ),
            )
        )

    (
        analytical_found,
        roles,
        high_impact,
    ) = _analytical_values(
        analytical_context
    )

    rule = direct_rules[
        0
    ]

    return (
        ContextualPreparationProposal(
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

            status=
                "proposal_ready",

            confidence=
                "medium",

            risk=
                DecisionRisk.HIGH,

            recommended_action=
                PreparationAction
                .REVIEW_INVALID_VALUES,

            recommended_parameters={},

            selected_action=None,

            selected_parameters={},

            rationale=(
                "Une règle documentaire existe "
                "mais elle ne justifie pas une "
                "conversion ou correction automatique. "
                "Les valeurs doivent être examinées."
            ),

            evidence=[
                _evidence_reference(
                    rule
                )
            ],

            analytical_context_found=
                analytical_found,

            analytical_roles=
                roles,

            high_analytical_impact=
                high_impact,

            business_context_found=True,

            human_validation_required=True,

            executable=False,

            abstention_reason=None,
        )
    )


# ============================================================
# SEMANTIC ALIASES
# ============================================================


def _resolve_semantic_aliases(
    *,
    decision: PreparationDecision,
    rag_context: PreparationRagContext,
    analytical_context: (
        PreparationContextAssessment
        | None
    ),
) -> ContextualPreparationProposal:
    (
        analytical_found,
        roles,
        high_impact,
    ) = _analytical_values(
        analytical_context
    )

    if not rag_context.evidence:
        return (
            _abstain(
                decision=
                    decision,

                analytical_context=
                    analytical_context,

                reason=(
                    "Aucune preuve documentaire "
                    "n'est disponible pour les "
                    "catégories concernées."
                ),
            )
        )

    return (
        ContextualPreparationProposal(
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

            status=
                "proposal_ready",

            confidence=
                "medium",

            risk=
                DecisionRisk.MEDIUM,

            recommended_action=
                PreparationAction
                .REVIEW_SEMANTIC_CONTEXT,

            recommended_parameters={},

            selected_action=None,

            selected_parameters={},

            rationale=(
                "La documentation apporte du "
                "contexte sur les catégories, mais "
                "Preparation Context Resolver v0.1 "
                "ne transforme pas encore ce contexte "
                "en fusion automatique."
            ),

            evidence=[
                _evidence_reference(
                    evidence
                )

                for evidence in (
                    rag_context.evidence[
                        :2
                    ]
                )
            ],

            analytical_context_found=
                analytical_found,

            analytical_roles=
                roles,

            high_analytical_impact=
                high_impact,

            business_context_found=True,

            human_validation_required=True,

            executable=False,

            abstention_reason=None,
        )
    )


# ============================================================
# SINGLE DECISION
# ============================================================


def _resolve_decision(
    *,
    decision: PreparationDecision,
    rag_context: (
        PreparationRagContext
        | None
    ),
    analytical_context: (
        PreparationContextAssessment
        | None
    ),
) -> ContextualPreparationProposal:
    if rag_context is None:
        return (
            _abstain(
                decision=
                    decision,

                analytical_context=
                    analytical_context,

                reason=(
                    "Aucun contexte RAG n'est "
                    "disponible pour cette décision."
                ),
            )
        )

    if (
        rag_context.status
        ==
        "abstained"
    ):
        return (
            _abstain(
                decision=
                    decision,

                analytical_context=
                    analytical_context,

                reason=(
                    rag_context.abstention_reason
                    or
                    (
                        "Preparation RAG Context "
                        "s'est abstenu."
                    )
                ),
            )
        )

    issue_kind = (
        decision
        .source_issue_kind
        .strip()
        .lower()
    )

    if (
        issue_kind
        ==
        "missing_values"
    ):
        return (
            _resolve_missing_values(
                decision=
                    decision,

                rag_context=
                    rag_context,

                analytical_context=
                    analytical_context,
            )
        )

    if (
        issue_kind
        ==
        "numeric_outliers"
    ):
        return (
            _resolve_numeric_outliers(
                decision=
                    decision,

                rag_context=
                    rag_context,

                analytical_context=
                    analytical_context,
            )
        )

    if (
        issue_kind
        ==
        "invalid_dates"
    ):
        return (
            _resolve_invalid_dates(
                decision=
                    decision,

                rag_context=
                    rag_context,

                analytical_context=
                    analytical_context,
            )
        )

    if (
        issue_kind
        ==
        "invalid_numeric_values"
    ):
        return (
            _resolve_invalid_numeric_values(
                decision=
                    decision,

                rag_context=
                    rag_context,

                analytical_context=
                    analytical_context,
            )
        )

    if (
        issue_kind
        ==
        "possible_semantic_aliases"
    ):
        return (
            _resolve_semantic_aliases(
                decision=
                    decision,

                rag_context=
                    rag_context,

                analytical_context=
                    analytical_context,
            )
        )

    return (
        _abstain(
            decision=
                decision,

            analytical_context=
                analytical_context,

            reason=(
                "Preparation Context Resolver v0.1 "
                "ne possède pas encore de stratégie "
                "pour ce type de problème."
            ),
        )
    )


# ============================================================
# PUBLIC API
# ============================================================


def resolve_preparation_context(
    *,
    plan: PreparationPlan,
    rag_context: PreparationRagContextReport,
    analytical_context: (
        PreparationContextReport
        | None
    ) = None,
) -> PreparationContextResolutionReport:
    """
    Fusionne :

    - le plan de préparation ;
    - le contexte analytique validé ;
    - les preuves documentaires RAG.

    Le resolver produit des propositions contextualisées.

    Il ne :

    - modifie jamais le DataFrame ;
    - modifie jamais le PreparationPlan ;
    - sélectionne jamais une action ;
    - exécute jamais une transformation ;
    - invente jamais une valeur métier ;
    - invente jamais un seuil métier.

    Une valeur ou un seuil ne peut apparaître dans les
    paramètres que s'il est explicitement extrait d'une
    preuve documentaire.
    """

    analytical_index = (
        _analytical_context_index(
            analytical_context
        )
    )

    rag_index = (
        _rag_context_index(
            rag_context
        )
    )

    proposals: list[
        ContextualPreparationProposal
    ] = []

    for decision in plan.decisions:
        # ----------------------------------------------------
        # AUTO_APPROVABLE operations do not need contextual
        # resolution in v0.1.
        # ----------------------------------------------------

        if (
            decision.status.value
            ==
            "auto_approvable"
        ):
            continue

        proposal = (
            _resolve_decision(
                decision=
                    decision,

                rag_context=
                    rag_index.get(
                        decision.decision_id
                    ),

                analytical_context=
                    analytical_index.get(
                        decision.decision_id
                    ),
            )
        )

        proposals.append(
            proposal
        )

    proposal_count = sum(
        1
        for proposal in proposals
        if (
            proposal.status
            ==
            "proposal_ready"
        )
    )

    abstention_count = sum(
        1
        for proposal in proposals
        if (
            proposal.status
            ==
            "abstained"
        )
    )

    documented_business_rule_count = sum(
        1
        for proposal in proposals
        if proposal.business_context_found
    )

    high_analytical_impact_count = sum(
        1
        for proposal in proposals
        if proposal.high_analytical_impact
    )

    return (
        PreparationContextResolutionReport(
            decision_count=
                len(
                    proposals
                ),

            proposal_count=
                proposal_count,

            abstention_count=
                abstention_count,

            documented_business_rule_count=
                documented_business_rule_count,

            high_analytical_impact_count=
                high_analytical_impact_count,

            proposals=
                proposals,

            notes=[
                (
                    "Les propositions reposent "
                    "uniquement sur des preuves "
                    "documentaires déjà validées "
                    "par Preparation RAG Context."
                ),
                (
                    "Une règle métier documentée "
                    "peut produire une recommandation "
                    "mais jamais une action sélectionnée."
                ),
                (
                    "Le contexte analytique est utilisé "
                    "pour caractériser l'impact potentiel "
                    "d'une transformation."
                ),
                (
                    "Aucune valeur métier ou borne "
                    "n'est inventée."
                ),
                (
                    "Toutes les propositions restent "
                    "executable=False en v0.1."
                ),
                (
                    "Une absence de preuve suffisamment "
                    "forte produit une abstention."
                ),
            ],

            rule_version=
                PREPARATION_CONTEXT_RESOLVER_RULE_VERSION,
        )
    )