from __future__ import annotations

import re
import unicodedata

from typing import (
    Any,
    Iterable,
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.planning.analytical_contract import (
    AnalyticalContract,
)

from app.preparation.contracts import (
    DecisionStatus,
    PreparationDecision,
    PreparationPlan,
)


# ============================================================
# VERSION
# ============================================================


PREPARATION_CONTEXT_RULE_VERSION = (
    "preparation_context_v0.1"
)


# ============================================================
# TYPES
# ============================================================


ContextRelevance = Literal[
    "none",
    "objective_only",
    "analytical_usage",
    "objective_and_analytical_usage",
]


ContextSignalSource = Literal[
    "objective",
    "analytical_contract",
]


# ============================================================
# MODELS
# ============================================================


class PreparationContextSignal(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    source: ContextSignalSource

    signal_type: str

    message: str

    contract_id: (
        str
        | None
    ) = None

    family: (
        str
        | None
    ) = None

    role: (
        str
        | None
    ) = None

    column: (
        str
        | None
    ) = None


class AnalyticalColumnUsage(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    contract_id: str

    contract_title: str

    family: str

    dataset_id: str

    column: str

    role: str

    analysis_kind: (
        str
        | None
    ) = None

    semantic_concept: (
        str
        | None
    ) = None


class PreparationContextAssessment(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    decision_id: str

    source_issue_id: str

    source_issue_kind: str

    dataset_id: str

    column: (
        str
        | None
    ) = None

    original_status: DecisionStatus

    context_relevance: ContextRelevance

    objective_mentioned: bool = False

    used_in_validated_analysis: bool = False

    analytical_usage_count: int = Field(
        ge=0
    )

    usage_roles: list[
        str
    ] = Field(
        default_factory=list
    )

    contract_ids: list[
        str
    ] = Field(
        default_factory=list
    )

    analytical_usages: list[
        AnalyticalColumnUsage
    ] = Field(
        default_factory=list
    )

    signals: list[
        PreparationContextSignal
    ] = Field(
        default_factory=list
    )

    satisfied_context: list[
        str
    ] = Field(
        default_factory=list
    )

    remaining_context: list[
        str
    ] = Field(
        default_factory=list
    )

    context_reduced: bool = False

    high_analytical_impact: bool = False

    still_needs_context: bool = False

    guidance: list[
        str
    ] = Field(
        default_factory=list
    )


class PreparationContextReport(
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

    objective_mentioned_count: int = Field(
        ge=0
    )

    analytically_used_count: int = Field(
        ge=0
    )

    context_reduced_count: int = Field(
        ge=0
    )

    still_needs_context_count: int = Field(
        ge=0
    )

    assessments: list[
        PreparationContextAssessment
    ] = Field(
        default_factory=list
    )

    notes: list[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        PREPARATION_CONTEXT_RULE_VERSION
    )


# ============================================================
# GENERIC HELPERS
# ============================================================


def _enum_value(
    value: Any,
) -> str:
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


def _normalize_text(
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
        r"[^a-z0-9]+",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


def _unique_strings(
    values: Iterable[
        str
    ],
) -> list[
    str
]:
    output: list[
        str
    ] = []

    seen: set[
        str
    ] = set()

    for value in values:
        cleaned = (
            str(
                value
            )
            .strip()
        )

        if not cleaned:
            continue

        if cleaned in seen:
            continue

        seen.add(
            cleaned
        )

        output.append(
            cleaned
        )

    return output


# ============================================================
# OBJECTIVE CONTEXT
# ============================================================


def _objective_mentions_column(
    *,
    objective: (
        str
        | None
    ),
    column: (
        str
        | None
    ),
) -> bool:
    if (
        objective is None
        or
        column is None
    ):
        return False

    normalized_objective = (
        _normalize_text(
            objective
        )
    )

    normalized_column = (
        _normalize_text(
            column
        )
    )

    if not normalized_column:
        return False

    padded_objective = (
        f" {normalized_objective} "
    )

    padded_column = (
        f" {normalized_column} "
    )

    return (
        padded_column
        in
        padded_objective
    )


# ============================================================
# ANALYTICAL CONTRACT VALIDATION
# ============================================================


def _is_validated_contract(
    contract: Any,
) -> bool:
    status = _enum_value(
        getattr(
            contract,
            "status",
            None,
        )
    )

    return (
        status
        .strip()
        .lower()
        ==
        "validated"
    )


def _binding_dataset_id(
    binding: Any,
) -> str:
    value = getattr(
        binding,
        "dataset_id",
        None,
    )

    if value is None:
        return ""

    return str(
        value
    )


def _binding_column(
    binding: Any,
) -> str:
    value = getattr(
        binding,
        "column",
        None,
    )

    if value is None:
        return ""

    return str(
        value
    )


def _binding_role(
    binding: Any,
) -> str:
    return _enum_value(
        getattr(
            binding,
            "role",
            None,
        )
    )


# ============================================================
# ANALYTICAL USAGE INDEX
# ============================================================


def _build_usage_index(
    analytical_contracts: Iterable[
        AnalyticalContract
    ],
) -> dict[
    tuple[
        str,
        str,
    ],
    list[
        AnalyticalColumnUsage
    ],
]:
    """
    Construit un index strict :

        (dataset_id, column)
            -> usages analytiques validés

    Aucun fuzzy matching n'est utilisé.

    Une colonne n'est considérée comme utilisée que si :

    - le contrat est `validated` ;
    - le dataset_id correspond exactement ;
    - le nom de colonne correspond exactement.
    """

    index: dict[
        tuple[
            str,
            str,
        ],
        list[
            AnalyticalColumnUsage
        ],
    ] = {}

    for contract in (
        analytical_contracts
    ):
        if not _is_validated_contract(
            contract
        ):
            continue

        contract_id = str(
            getattr(
                contract,
                "contract_id",
                "",
            )
        )

        contract_title = str(
            getattr(
                contract,
                "title",
                "",
            )
        )

        family = _enum_value(
            getattr(
                contract,
                "family",
                None,
            )
        )

        bindings = (
            getattr(
                contract,
                "bindings",
                [],
            )
            or
            []
        )

        for binding in bindings:
            dataset_id = (
                _binding_dataset_id(
                    binding
                )
            )

            column = (
                _binding_column(
                    binding
                )
            )

            role = (
                _binding_role(
                    binding
                )
            )

            if (
                not dataset_id
                or
                not column
                or
                not role
            ):
                continue

            analysis_kind_raw = (
                getattr(
                    binding,
                    "analysis_kind",
                    None,
                )
            )

            semantic_concept_raw = (
                getattr(
                    binding,
                    "semantic_concept",
                    None,
                )
            )

            usage = (
                AnalyticalColumnUsage(
                    contract_id=
                        contract_id,

                    contract_title=
                        contract_title,

                    family=
                        family,

                    dataset_id=
                        dataset_id,

                    column=
                        column,

                    role=
                        role,

                    analysis_kind=(
                        str(
                            analysis_kind_raw
                        )
                        if (
                            analysis_kind_raw
                            is not None
                        )
                        else None
                    ),

                    semantic_concept=(
                        str(
                            semantic_concept_raw
                        )
                        if (
                            semantic_concept_raw
                            is not None
                        )
                        else None
                    ),
                )
            )

            key = (
                dataset_id,
                column,
            )

            index.setdefault(
                key,
                [],
            ).append(
                usage
            )

    return index


# ============================================================
# CONTEXT SATISFACTION
# ============================================================


def _context_item_is_satisfied(
    *,
    context_item: str,
    usages: list[
        AnalyticalColumnUsage
    ],
) -> bool:
    if not usages:
        return False

    normalized = (
        _normalize_text(
            context_item
        )
    )

    # --------------------------------------------------------
    # We can deterministically know the analytical role
    # because it comes from validated AnalyticalContract
    # bindings.
    # --------------------------------------------------------

    if (
        "role de la variable"
        in normalized
        and
        "analyse"
        in normalized
    ):
        return True

    return False


# ============================================================
# IMPACT
# ============================================================


_HIGH_IMPACT_ISSUE_KINDS = {
    "missing_values",
    "missing_identifier",
    "numeric_outliers",
    "invalid_numeric_values",
    "invalid_dates",
    "possible_semantic_aliases",
}


_HIGH_IMPACT_ROLES = {
    "x",
    "y",
    "value",
    "measure",
    "group",
    "dimension",
    "time",
    "entity",
}


def _is_high_analytical_impact(
    *,
    issue_kind: str,
    usages: list[
        AnalyticalColumnUsage
    ],
) -> bool:
    if (
        issue_kind
        not in
        _HIGH_IMPACT_ISSUE_KINDS
    ):
        return False

    roles = {
        usage.role
        .strip()
        .lower()

        for usage in usages
    }

    return bool(
        roles
        &
        _HIGH_IMPACT_ROLES
    )


# ============================================================
# GUIDANCE
# ============================================================


def _guidance_for_decision(
    *,
    decision: PreparationDecision,
    usages: list[
        AnalyticalColumnUsage
    ],
) -> list[
    str
]:
    if not usages:
        return []

    issue_kind = (
        decision
        .source_issue_kind
        .strip()
        .lower()
    )

    roles = {
        usage.role
        .strip()
        .lower()

        for usage in usages
    }

    guidance: list[
        str
    ] = []

    if issue_kind == "missing_values":
        guidance.append(
            (
                "Cette colonne est utilisée dans au moins "
                "une analyse validée. Une suppression ou "
                "une imputation peut donc modifier le "
                "résultat analytique."
            )
        )

        if roles & {
            "x",
            "y",
            "value",
            "measure",
        }:
            guidance.append(
                (
                    "La colonne joue un rôle quantitatif "
                    "ou analytique direct. Une imputation "
                    "peut modifier la distribution, les "
                    "agrégats ou les associations observées."
                )
            )

        if roles & {
            "group",
            "dimension",
        }:
            guidance.append(
                (
                    "La colonne sert à définir des groupes "
                    "ou des dimensions. Le traitement des "
                    "valeurs manquantes peut modifier la "
                    "composition des groupes comparés."
                )
            )

        if "time" in roles:
            guidance.append(
                (
                    "La colonne intervient dans une analyse "
                    "temporelle. Le traitement des valeurs "
                    "manquantes peut modifier l'ordre ou la "
                    "couverture temporelle."
                )
            )

    elif issue_kind == "numeric_outliers":
        guidance.append(
            (
                "Les valeurs atypiques concernent une "
                "variable utilisée dans une analyse validée. "
                "Elles doivent être investiguées avant "
                "suppression ou plafonnement."
            )
        )

        if roles & {
            "x",
            "y",
            "value",
            "measure",
        }:
            guidance.append(
                (
                    "Les valeurs atypiques peuvent influencer "
                    "les agrégats, corrélations, régressions "
                    "ou comparaisons quantitatives."
                )
            )

    elif (
        issue_kind
        ==
        "possible_semantic_aliases"
    ):
        if roles & {
            "group",
            "dimension",
            "entity",
        }:
            guidance.append(
                (
                    "Ces catégories sont utilisées comme "
                    "groupes, dimensions ou entités dans une "
                    "analyse validée. Une fusion modifierait "
                    "directement les effectifs et résultats "
                    "par catégorie."
                )
            )

    elif (
        issue_kind
        ==
        "invalid_numeric_values"
    ):
        if roles & {
            "x",
            "y",
            "value",
            "measure",
        }:
            guidance.append(
                (
                    "La colonne est attendue dans un rôle "
                    "quantitatif. Les valeurs non numériques "
                    "doivent être comprises avant conversion "
                    "ou exclusion."
                )
            )

    elif issue_kind == "invalid_dates":
        if "time" in roles:
            guidance.append(
                (
                    "La colonne est utilisée comme dimension "
                    "temporelle. Les dates invalides doivent "
                    "être résolues avant une analyse de série "
                    "temporelle."
                )
            )

    elif (
        issue_kind
        ==
        "missing_identifier"
    ):
        if "entity" in roles:
            guidance.append(
                (
                    "L'identifiant manquant correspond à une "
                    "entité utilisée dans l'analyse. Sa "
                    "suppression peut modifier la population "
                    "analysée."
                )
            )

    return _unique_strings(
        guidance
    )


# ============================================================
# SIGNALS
# ============================================================


def _build_signals(
    *,
    decision: PreparationDecision,
    objective_mentioned: bool,
    usages: list[
        AnalyticalColumnUsage
    ],
) -> list[
    PreparationContextSignal
]:
    signals: list[
        PreparationContextSignal
    ] = []

    if (
        objective_mentioned
        and
        decision.column
        is not None
    ):
        signals.append(
            PreparationContextSignal(
                source=
                    "objective",

                signal_type=
                    "column_explicitly_mentioned",

                message=(
                    "La colonne est explicitement "
                    "mentionnée dans l'objectif "
                    "analytique fourni."
                ),

                column=
                    decision.column,
            )
        )

    for usage in usages:
        signals.append(
            PreparationContextSignal(
                source=
                    "analytical_contract",

                signal_type=
                    "validated_analytical_usage",

                message=(
                    f"La colonne `{usage.column}` "
                    f"est utilisée avec le rôle "
                    f"`{usage.role}` dans l'analyse "
                    f"validée `{usage.contract_title}`."
                ),

                contract_id=
                    usage.contract_id,

                family=
                    usage.family,

                role=
                    usage.role,

                column=
                    usage.column,
            )
        )

    return signals


# ============================================================
# RELEVANCE
# ============================================================


def _context_relevance(
    *,
    objective_mentioned: bool,
    usages: list[
        AnalyticalColumnUsage
    ],
) -> ContextRelevance:
    has_usage = bool(
        usages
    )

    if (
        objective_mentioned
        and
        has_usage
    ):
        return (
            "objective_and_analytical_usage"
        )

    if has_usage:
        return (
            "analytical_usage"
        )

    if objective_mentioned:
        return (
            "objective_only"
        )

    return "none"


# ============================================================
# SINGLE DECISION
# ============================================================


def _assess_decision(
    *,
    decision: PreparationDecision,
    objective: (
        str
        | None
    ),
    usage_index: dict[
        tuple[
            str,
            str,
        ],
        list[
            AnalyticalColumnUsage
        ],
    ],
) -> PreparationContextAssessment:
    objective_mentioned = (
        _objective_mentions_column(
            objective=
                objective,

            column=
                decision.column,
        )
    )

    if decision.column is None:
        usages: list[
            AnalyticalColumnUsage
        ] = []

    else:
        usages = list(
            usage_index.get(
                (
                    decision.dataset_id,
                    decision.column,
                ),
                [],
            )
        )

    satisfied_context: list[
        str
    ] = []

    remaining_context: list[
        str
    ] = []

    for context_item in (
        decision.context_required
    ):
        if _context_item_is_satisfied(
            context_item=
                context_item,

            usages=
                usages,
        ):
            satisfied_context.append(
                context_item
            )

        else:
            remaining_context.append(
                context_item
            )

    usage_roles = (
        _unique_strings(
            usage.role
            for usage in usages
        )
    )

    contract_ids = (
        _unique_strings(
            usage.contract_id
            for usage in usages
        )
    )

    high_analytical_impact = (
        _is_high_analytical_impact(
            issue_kind=
                decision.source_issue_kind,

            usages=
                usages,
        )
    )

    guidance = (
        _guidance_for_decision(
            decision=
                decision,

            usages=
                usages,
        )
    )

    signals = (
        _build_signals(
            decision=
                decision,

            objective_mentioned=
                objective_mentioned,

            usages=
                usages,
        )
    )

    context_reduced = bool(
        satisfied_context
    )

    # ========================================================
    # IMPORTANT SAFETY RULE
    #
    # Analytical context can reduce uncertainty about HOW
    # the column will be used.
    #
    # It cannot determine WHY a value is missing, whether
    # an outlier is legitimate, or what a business code means.
    #
    # Therefore v0.1 never automatically resolves a
    # NEEDS_CONTEXT decision.
    # ========================================================

    still_needs_context = (
        decision.status
        ==
        DecisionStatus.NEEDS_CONTEXT
    )

    return PreparationContextAssessment(
        decision_id=
            decision.decision_id,

        source_issue_id=
            decision.source_issue_id,

        source_issue_kind=
            decision.source_issue_kind,

        dataset_id=
            decision.dataset_id,

        column=
            decision.column,

        original_status=
            decision.status,

        context_relevance=
            _context_relevance(
                objective_mentioned=
                    objective_mentioned,

                usages=
                    usages,
            ),

        objective_mentioned=
            objective_mentioned,

        used_in_validated_analysis=
            bool(
                usages
            ),

        analytical_usage_count=
            len(
                usages
            ),

        usage_roles=
            usage_roles,

        contract_ids=
            contract_ids,

        analytical_usages=
            usages,

        signals=
            signals,

        satisfied_context=
            satisfied_context,

        remaining_context=
            remaining_context,

        context_reduced=
            context_reduced,

        high_analytical_impact=
            high_analytical_impact,

        still_needs_context=
            still_needs_context,

        guidance=
            guidance,
    )


# ============================================================
# PUBLIC API
# ============================================================


def enrich_preparation_context(
    *,
    plan: PreparationPlan,
    objective: (
        str
        | None
    ) = None,
    analytical_contracts: Iterable[
        AnalyticalContract
    ] = (),
) -> PreparationContextReport:
    """
    Enrichit un PreparationPlan avec le contexte analytique
    déjà validé par DataLens.

    Sources autorisées en v0.1 :

    - objectif utilisateur ;
    - AnalyticalContract au statut `validated`.

    Cette fonction :

    - ne modifie pas le PreparationPlan ;
    - ne modifie pas les données ;
    - ne sélectionne aucune action ;
    - n'invente aucun contexte métier ;
    - n'utilise aucun fuzzy matching de colonne ;
    - ignore les contrats non validés.

    Le RAG métier sera ajouté dans une version suivante.
    """

    contracts = list(
        analytical_contracts
    )

    usage_index = (
        _build_usage_index(
            contracts
        )
    )

    assessments = [
        _assess_decision(
            decision=
                decision,

            objective=
                objective,

            usage_index=
                usage_index,
        )

        for decision in (
            plan.decisions
        )
    ]

    objective_mentioned_count = sum(
        1
        for assessment in assessments
        if assessment.objective_mentioned
    )

    analytically_used_count = sum(
        1
        for assessment in assessments
        if assessment.used_in_validated_analysis
    )

    context_reduced_count = sum(
        1
        for assessment in assessments
        if assessment.context_reduced
    )

    still_needs_context_count = sum(
        1
        for assessment in assessments
        if assessment.still_needs_context
    )

    return PreparationContextReport(
        decision_count=
            len(
                assessments
            ),

        objective_mentioned_count=
            objective_mentioned_count,

        analytically_used_count=
            analytically_used_count,

        context_reduced_count=
            context_reduced_count,

        still_needs_context_count=
            still_needs_context_count,

        assessments=
            assessments,

        notes=[
            (
                "Seuls les AnalyticalContract au statut "
                "`validated` sont utilisés."
            ),
            (
                "Le rapprochement d'une colonne utilise "
                "strictement `(dataset_id, column)`."
            ),
            (
                "Le contexte analytique peut expliquer "
                "comment une variable sera utilisée mais "
                "ne peut pas inventer la signification "
                "métier d'une valeur manquante."
            ),
            (
                "Preparation Context v0.1 ne sélectionne "
                "et n'exécute aucune action de nettoyage."
            ),
            (
                "Le contexte métier documentaire sera "
                "ajouté dans une version ultérieure via RAG."
            ),
        ],

        rule_version=
            PREPARATION_CONTEXT_RULE_VERSION,
    )