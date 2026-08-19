from __future__ import annotations

from collections import Counter
from typing import (
    Any,
    Literal,
)

from pydantic import BaseModel, ConfigDict, Field

from app.discovery.schemas import (
    AnalysisDiscoveryReport,
    DiscoveredAnalysis,
    DiscoveredVariable,
)


ANALYSIS_PRIORITIZATION_RULE_VERSION = (
    "analysis_prioritization_v0.1"
)

ANALYTICAL_VALUE_GUARD_RULE_VERSION = (
    "analytical_value_guard_v0.1"
)

ANALYSIS_PRIORITIZATION_AUDIT_RULE_VERSION = (
    "analysis_prioritization_audit_v0.1"
)

MAX_SELECTED_ANALYSES = 36
MIN_PRIORITY_SCORE = 40.0
MAX_VARIABLE_OCCURRENCES = 8

FAMILY_CAPS: dict[str, int] = {
    "data_quality": 4,
    "time_series": 5,
    "quantitative_association": 12,
    "group_comparison": 8,
    "categorical_association": 6,
    "distribution": 6,
    "derived_gap": 5,
    "entity_ranking": 5,
    "ranking": 5,
    "inequality": 4,
    "geographic_comparison": 5,
}

DEFAULT_FAMILY_CAP = 4


PrioritizationDecision = Literal[
    "selected",
    "deferred",
    "rejected",
]

PrioritizationReasonCode = Literal[
    "selected_by_priority",
    "quality_guard",
    "not_executable_now",
    "identifier_misuse",
    "record_label_dimension",
    "fragmented_group_dimension",
    "sparse_categorical_structure",
    "priority_below_threshold",
    "family_budget_exhausted",
    "variable_budget_exhausted",
    "global_budget_exhausted",
]


class AnalysisPrioritizationDecision(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    analysis_id: str
    family: str
    title: str
    original_priority_score: float
    execution_priority_score: float
    decision: PrioritizationDecision
    reason_code: PrioritizationReasonCode

    reasons: list[str] = Field(
        default_factory=list
    )

    variable_keys: list[str] = Field(
        default_factory=list
    )


class AnalysisPrioritizationReport(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    discovered_count: int
    selected_count: int
    deferred_count: int
    rejected_count: int

    selected_analysis_ids: list[str]
    deferred_analysis_ids: list[str]
    rejected_analysis_ids: list[str]

    decisions: list[
        AnalysisPrioritizationDecision
    ]

    selected_candidates: list[
        DiscoveredAnalysis
    ]

    family_selected_counts: dict[str, int]
    notes: list[str]

    rule_version: str = (
        ANALYSIS_PRIORITIZATION_RULE_VERSION
    )


class AnalysisPrioritizationAudit(BaseModel):
    """
    Compact public audit view of exploratory prioritization.

    Unlike AnalysisPrioritizationReport, this contract never
    embeds DiscoveredAnalysis candidates. It exposes only the
    deterministic decisions and aggregate counters required to
    explain why Discovery is broader than automatic execution.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    discovered_count: int
    selected_for_execution_count: int
    deferred_count: int
    rejected_count: int

    decision_counts: dict[str, int]
    reason_counts: dict[str, int]
    non_execution_reason_counts: dict[str, int]
    family_selected_counts: dict[str, int]

    decisions: list[
        AnalysisPrioritizationDecision
    ]

    prioritization_rule_version: str
    analytical_value_guard_rule_version: str

    audit_rule_version: str = (
        ANALYSIS_PRIORITIZATION_AUDIT_RULE_VERSION
    )


IDENTIFIER_ALLOWED_FAMILIES = {
    "entity_ranking",
    "ranking",
    "inequality",
}

IDENTIFIER_ALLOWED_ROLES = {
    "entity",
    "dimension",
}


def _variable_key(
    variable: DiscoveredVariable,
) -> str:
    return (
        f"{variable.dataset_id}:"
        f"{variable.column}"
    )


def _candidate_variable_keys(
    candidate: DiscoveredAnalysis,
) -> list[str]:
    return list(
        dict.fromkeys(
            _variable_key(
                variable
            )
            for variable
            in candidate.variables
        )
    )


def _identifier_misuse_variables(
    candidate: DiscoveredAnalysis,
) -> list[DiscoveredVariable]:
    output: list[
        DiscoveredVariable
    ] = []

    for variable in candidate.variables:
        if (
            variable.semantic_role
            !=
            "identifier"
        ):
            continue

        allowed = (
            candidate.family
            in
            IDENTIFIER_ALLOWED_FAMILIES
            and
            variable.role
            in
            IDENTIFIER_ALLOWED_ROLES
        )

        if not allowed:
            output.append(
                variable
            )

    return output



# ============================================================
# ANALYTICAL VALUE GUARD
# ============================================================


RECORD_LABEL_SIGNALS = {
    "name",
    "firstname",
    "first",
    "lastname",
    "last",
    "surname",
    "nom",
    "prenom",
    "email",
    "mail",
    "phone",
    "telephone",
    "mobile",
    "address",
    "adresse",
    "street",
    "rue",
    "username",
    "user",
    "label",
}


MIN_FRAGMENTED_GROUP_COUNT = 8

MAX_FRAGMENTED_MEDIAN_GROUP_SIZE = 2.0

MIN_STRONGLY_FRAGMENTED_GROUP_COUNT = 12

MAX_STRONGLY_FRAGMENTED_MEAN_GROUP_SIZE = 3.0

MIN_RECORD_LABEL_GROUP_COUNT = 6

MIN_RECORD_LABEL_UNIQUE_RATIO = 0.25

MIN_CATEGORICAL_CELL_DENSITY = 2.0


def _normalize_column_tokens(
    value: str,
) -> set[str]:
    normalized = []

    current = []


    for character in str(
        value
    ).lower():
        if character.isalnum():
            current.append(
                character
            )
        else:
            if current:
                normalized.append(
                    "".join(
                        current
                    )
                )

                current = []


    if current:
        normalized.append(
            "".join(
                current
            )
        )


    expanded: set[str] = set(
        normalized
    )


    for token in list(
        normalized
    ):
        if token.endswith(
            "name"
        ) and token != "name":
            expanded.add(
                "name"
            )


    return expanded


def _looks_like_record_label(
    variable: DiscoveredVariable,
) -> bool:
    tokens = (
        _normalize_column_tokens(
            variable.column
        )
    )


    return bool(
        tokens
        &
        RECORD_LABEL_SIGNALS
    )


def _dataset_dataframe_map(
    datasets: (
        list[
            dict[
                str,
                Any,
            ]
        ]
        | None
    ),
) -> dict[
    str,
    Any,
]:
    if not datasets:
        return {}


    output: dict[
        str,
        Any,
    ] = {}


    for record in datasets:
        dataset_id = (
            record.get(
                "dataset_id"
            )
        )


        dataframe = (
            record.get(
                "dataframe"
            )
        )


        if (
            dataset_id is None
            or
            dataframe is None
        ):
            continue


        output[
            str(
                dataset_id
            )
        ] = dataframe


    return output


def _group_variable(
    candidate: DiscoveredAnalysis,
) -> (
    DiscoveredVariable
    | None
):
    preferred_roles = {
        "group",
        "dimension",
    }


    for variable in (
        candidate.variables
    ):
        if (
            variable.role
            in
            preferred_roles
        ):
            return variable


    return None


def _series_for_variable(
    *,
    variable: DiscoveredVariable,
    dataframe_map: dict[
        str,
        Any,
    ],
):
    dataframe = (
        dataframe_map.get(
            variable.dataset_id
        )
    )


    if dataframe is None:
        return None


    try:
        if (
            variable.column
            not in dataframe.columns
        ):
            return None


        return dataframe[
            variable.column
        ]


    except Exception:
        return None


def _group_dimension_guard(
    *,
    candidate: DiscoveredAnalysis,
    dataframe_map: dict[
        str,
        Any,
    ],
) -> tuple[
    PrioritizationReasonCode,
    list[str],
] | None:
    if (
        candidate.family
        !=
        "group_comparison"
    ):
        return None


    variable = (
        _group_variable(
            candidate
        )
    )


    if variable is None:
        return None


    series = (
        _series_for_variable(
            variable=
                variable,

            dataframe_map=
                dataframe_map,
        )
    )


    if series is None:
        return None


    try:
        valid = (
            series
            .dropna()
        )


        valid_count = int(
            len(
                valid
            )
        )


        if (
            valid_count
            <=
            0
        ):
            return None


        group_sizes = (
            valid
            .value_counts(
                dropna=True
            )
        )


        group_count = int(
            len(
                group_sizes
            )
        )


        if (
            group_count
            <
            2
        ):
            return None


        unique_ratio = (
            group_count
            /
            valid_count
        )


        mean_group_size = (
            valid_count
            /
            group_count
        )


        median_group_size = float(
            group_sizes.median()
        )


    except Exception:
        return None


    # --------------------------------------------------------
    # Record-label / quasi-identifier dimension
    # --------------------------------------------------------
    #
    # The name signal alone is never sufficient.
    #
    # It is combined with observed cardinality so a low-
    # cardinality business field such as "product_name" with
    # only three products is not automatically rejected.
    # --------------------------------------------------------

    if (
        _looks_like_record_label(
            variable
        )
        and
        group_count
        >=
        MIN_RECORD_LABEL_GROUP_COUNT
        and
        unique_ratio
        >=
        MIN_RECORD_LABEL_UNIQUE_RATIO
    ):
        return (
            "record_label_dimension",

            [
                (
                    f"`{variable.column}` ressemble à un libellé "
                    "d'enregistrement ou d'entité et possède une "
                    "cardinalité trop élevée pour une comparaison "
                    "automatique de groupes."
                ),

                (
                    f"{group_count} groupe(s) pour "
                    f"{valid_count} observation(s), soit un ratio "
                    f"de cardinalité de {unique_ratio:.2f}."
                ),

                (
                    "Une analyse par entité peut rester pertinente "
                    "lorsqu'elle est explicitement demandée, mais "
                    "elle n'est pas prioritaire en exploration "
                    "automatique."
                ),
            ],
        )


    # --------------------------------------------------------
    # Fragmented dimension
    # --------------------------------------------------------

    fragmented = (
        group_count
        >=
        MIN_FRAGMENTED_GROUP_COUNT
        and
        median_group_size
        <
        MAX_FRAGMENTED_MEDIAN_GROUP_SIZE
    )


    strongly_fragmented = (
        group_count
        >=
        MIN_STRONGLY_FRAGMENTED_GROUP_COUNT
        and
        mean_group_size
        <
        MAX_STRONGLY_FRAGMENTED_MEAN_GROUP_SIZE
    )


    if (
        fragmented
        or
        strongly_fragmented
    ):
        return (
            "fragmented_group_dimension",

            [
                (
                    f"`{variable.column}` fragmente les données en "
                    f"{group_count} groupes pour seulement "
                    f"{valid_count} observations."
                ),

                (
                    f"Taille moyenne d'un groupe : "
                    f"{mean_group_size:.2f}; taille médiane : "
                    f"{median_group_size:.2f}."
                ),

                (
                    "Une comparaison automatique de distributions "
                    "serait peu informative avec autant de groupes "
                    "faiblement alimentés."
                ),
            ],
        )


    return None


def _categorical_structure_guard(
    *,
    candidate: DiscoveredAnalysis,
) -> tuple[
    PrioritizationReasonCode,
    list[str],
] | None:
    if (
        candidate.family
        !=
        "categorical_association"
    ):
        return None


    signals = (
        candidate.observed_signals
    )


    try:
        valid_observations = int(
            signals.get(
                "valid_observations",
                0,
            )
        )


        left_levels = int(
            signals.get(
                "left_levels",
                0,
            )
        )


        right_levels = int(
            signals.get(
                "right_levels",
                0,
            )
        )


    except (
        TypeError,
        ValueError,
    ):
        return None


    cell_count = (
        left_levels
        *
        right_levels
    )


    if (
        valid_observations
        <=
        0
        or
        cell_count
        <=
        0
    ):
        return None


    density = (
        valid_observations
        /
        cell_count
    )


    if (
        density
        <
        MIN_CATEGORICAL_CELL_DENSITY
    ):
        return (
            "sparse_categorical_structure",

            [
                (
                    "La table de contingence potentielle est trop "
                    "fragmentée pour être prioritaire en exploration "
                    "automatique."
                ),

                (
                    f"{valid_observations} observation(s) pour "
                    f"{left_levels} × {right_levels} modalités, "
                    f"soit {density:.2f} observation(s) moyenne(s) "
                    "par cellule potentielle."
                ),

                (
                    "Le moteur d'exécution conserve ses propres "
                    "contrôles statistiques ; ce garde-fou intervient "
                    "plus tôt pour éviter une analyse exploratoire "
                    "de faible valeur."
                ),
            ],
        )


    return None


def _analytical_value_guard(
    *,
    candidate: DiscoveredAnalysis,
    dataframe_map: dict[
        str,
        Any,
    ],
) -> tuple[
    PrioritizationReasonCode,
    list[str],
] | None:
    group_guard = (
        _group_dimension_guard(
            candidate=
                candidate,

            dataframe_map=
                dataframe_map,
        )
    )


    if (
        group_guard
        is not None
    ):
        return group_guard


    categorical_guard = (
        _categorical_structure_guard(
            candidate=
                candidate
        )
    )


    if (
        categorical_guard
        is not None
    ):
        return categorical_guard


    return None


def _execution_priority_score(
    candidate: DiscoveredAnalysis,
) -> float:
    """
    The discovery score remains the main signal.

    V0.1 adds only small deterministic tie-breaking bonuses.

    The user objective has already influenced Discovery's
    priority_score upstream, so this layer does not reinterpret
    natural language itself.
    """

    score = float(
        candidate.priority_score
    )

    if (
        candidate.family
        ==
        "data_quality"
    ):
        score += 12.0

    if (
        candidate.scope
        ==
        "cross_dataset"
        and
        candidate.relationship_status
        ==
        "validated"
    ):
        score += 2.0

    return round(
        min(
            score,
            100.0,
        ),
        4,
    )


def _sorted_candidates(
    discovery: AnalysisDiscoveryReport,
) -> list[DiscoveredAnalysis]:
    return sorted(
        discovery.candidates,
        key=lambda candidate: (
            _execution_priority_score(
                candidate
            ),
            float(
                candidate.priority_score
            ),
            1
            if (
                candidate.scope
                ==
                "cross_dataset"
            )
            else 0,
            candidate.analysis_id,
        ),
        reverse=True,
    )


def _decision(
    candidate: DiscoveredAnalysis,
    *,
    decision: PrioritizationDecision,
    reason_code: PrioritizationReasonCode,
    reasons: list[str],
) -> AnalysisPrioritizationDecision:
    return AnalysisPrioritizationDecision(
        analysis_id=
            candidate.analysis_id,

        family=
            candidate.family,

        title=
            candidate.title,

        original_priority_score=
            float(
                candidate.priority_score
            ),

        execution_priority_score=
            _execution_priority_score(
                candidate
            ),

        decision=
            decision,

        reason_code=
            reason_code,

        reasons=
            reasons,

        variable_keys=
            _candidate_variable_keys(
                candidate
            ),
    )


def prioritize_analysis_discovery(
    discovery: AnalysisDiscoveryReport,
    *,
    datasets: (
        list[
            dict[
                str,
                Any,
            ]
        ]
        | None
    ) = None,
) -> AnalysisPrioritizationReport:
    """
    Convert broad discovery into a bounded execution shortlist.

    The original discovery object is not mutated.
    """

    selected: list[
        DiscoveredAnalysis
    ] = []

    decisions: list[
        AnalysisPrioritizationDecision
    ] = []

    family_counts: Counter[str] = Counter()
    variable_counts: Counter[str] = Counter()

    dataframe_map = (
        _dataset_dataframe_map(
            datasets
        )
    )

    for candidate in _sorted_candidates(
        discovery
    ):
        identifier_misuse = (
            _identifier_misuse_variables(
                candidate
            )
        )

        if identifier_misuse:
            columns = [
                variable.column
                for variable
                in identifier_misuse
            ]

            decisions.append(
                _decision(
                    candidate,
                    decision=
                        "rejected",
                    reason_code=
                        "identifier_misuse",
                    reasons=[
                        (
                            "Une ou plusieurs colonnes identifiantes "
                            "sont utilisées comme variables analytiques "
                            "dans un rôle non autorisé."
                        ),
                        (
                            "Colonnes concernées : "
                            +
                            ", ".join(
                                columns
                            )
                            +
                            "."
                        ),
                    ],
                )
            )

            continue

        # ====================================================
        # ANALYTICAL VALUE GUARD
        # ====================================================

        analytical_value_issue = (
            _analytical_value_guard(
                candidate=
                    candidate,

                dataframe_map=
                    dataframe_map,
            )
        )


        if (
            analytical_value_issue
            is not None
        ):
            (
                reason_code,
                reasons,
            ) = analytical_value_issue


            decisions.append(
                _decision(
                    candidate,

                    decision=
                        "deferred",

                    reason_code=
                        reason_code,

                    reasons=
                        reasons,
                )
            )

            continue

        if (
            candidate.readiness
            !=
            "executable_now"
        ):
            decisions.append(
                _decision(
                    candidate,
                    decision=
                        "deferred",
                    reason_code=
                        "not_executable_now",
                    reasons=[
                        (
                            "Le moteur de découverte ne considère "
                            "pas encore cette analyse comme "
                            "exécutable immédiatement."
                        )
                    ],
                )
            )

            continue

        if (
            candidate.family
            ==
            "data_quality"
        ):
            family_cap = FAMILY_CAPS[
                "data_quality"
            ]

            if (
                family_counts[
                    candidate.family
                ]
                <
                family_cap
                and
                len(
                    selected
                )
                <
                MAX_SELECTED_ANALYSES
            ):
                selected.append(
                    candidate
                )

                family_counts[
                    candidate.family
                ] += 1

                decisions.append(
                    _decision(
                        candidate,
                        decision=
                            "selected",
                        reason_code=
                            "quality_guard",
                        reasons=[
                            (
                                "Le contrôle qualité est conservé "
                                "comme preuve méthodologique avant "
                                "les analyses statistiques."
                            )
                        ],
                    )
                )

                continue

        execution_score = (
            _execution_priority_score(
                candidate
            )
        )

        if (
            execution_score
            <
            MIN_PRIORITY_SCORE
        ):
            decisions.append(
                _decision(
                    candidate,
                    decision=
                        "deferred",
                    reason_code=
                        "priority_below_threshold",
                    reasons=[
                        (
                            "Le score de découverte est trop faible "
                            "pour consommer le budget d'exécution "
                            "automatique de la V0.1."
                        )
                    ],
                )
            )

            continue

        family_cap = (
            FAMILY_CAPS.get(
                candidate.family,
                DEFAULT_FAMILY_CAP,
            )
        )

        if (
            family_counts[
                candidate.family
            ]
            >=
            family_cap
        ):
            decisions.append(
                _decision(
                    candidate,
                    decision=
                        "deferred",
                    reason_code=
                        "family_budget_exhausted",
                    reasons=[
                        (
                            "Le budget de diversité de cette famille "
                            "d'analyse est déjà atteint."
                        ),
                        (
                            f"Limite V0.1 pour `{candidate.family}` : "
                            f"{family_cap}."
                        ),
                    ],
                )
            )

            continue

        variable_keys = (
            _candidate_variable_keys(
                candidate
            )
        )

        saturated_variables = [
            variable_key
            for variable_key
            in variable_keys
            if (
                variable_counts[
                    variable_key
                ]
                >=
                MAX_VARIABLE_OCCURRENCES
            )
        ]

        if saturated_variables:
            decisions.append(
                _decision(
                    candidate,
                    decision=
                        "deferred",
                    reason_code=
                        "variable_budget_exhausted",
                    reasons=[
                        (
                            "Une variable est déjà suffisamment "
                            "représentée dans le sous-ensemble "
                            "d'exécution."
                        ),
                        (
                            "Variables saturées : "
                            +
                            ", ".join(
                                saturated_variables
                            )
                            +
                            "."
                        ),
                    ],
                )
            )

            continue

        if (
            len(
                selected
            )
            >=
            MAX_SELECTED_ANALYSES
        ):
            decisions.append(
                _decision(
                    candidate,
                    decision=
                        "deferred",
                    reason_code=
                        "global_budget_exhausted",
                    reasons=[
                        (
                            "Le budget global d'analyses "
                            "automatiques est atteint."
                        ),
                        (
                            f"Limite V0.1 : "
                            f"{MAX_SELECTED_ANALYSES}."
                        ),
                    ],
                )
            )

            continue

        selected.append(
            candidate
        )

        family_counts[
            candidate.family
        ] += 1

        for variable_key in variable_keys:
            variable_counts[
                variable_key
            ] += 1

        decisions.append(
            _decision(
                candidate,
                decision=
                    "selected",
                reason_code=
                    "selected_by_priority",
                reasons=[
                    (
                        "Le candidat respecte les garde-fous "
                        "déterministes et reste dans les budgets "
                        "de diversité et de calcul."
                    )
                ],
            )
        )

    selected_ids = [
        decision.analysis_id
        for decision
        in decisions
        if (
            decision.decision
            ==
            "selected"
        )
    ]

    deferred_ids = [
        decision.analysis_id
        for decision
        in decisions
        if (
            decision.decision
            ==
            "deferred"
        )
    ]

    rejected_ids = [
        decision.analysis_id
        for decision
        in decisions
        if (
            decision.decision
            ==
            "rejected"
        )
    ]

    notes = [
        (
            "La découverte reste exhaustive : cette couche "
            "ne supprime aucun candidat du rapport source."
        ),
        (
            "La V0.1 limite uniquement le sous-ensemble "
            "destiné à l'exécution statistique automatique."
        ),
        (
            "Le priority_score de Discovery reste le signal "
            "principal ; il contient déjà le bonus lié à "
            "l'objectif utilisateur lorsqu'un objectif est fourni."
        ),
        (
            "Les colonnes identifiantes peuvent servir de "
            "dimension d'entité dans certaines familles dédiées, "
            "mais ne sont pas autorisées comme mesure ou variable "
            "d'association ordinaire."
        ),
        (
            "Le garde-fou de valeur analytique mesure la "
            "fragmentation réelle des dimensions de groupe et la "
            "densité des structures catégorielles lorsque les "
            "DataFrames préparés sont disponibles."
        ),
        (
            "Un libellé d'entité n'est pas interdit par son nom "
            "seul : le signal lexical doit être corroboré par une "
            "cardinalité élevée observée dans les données."
        ),
        (
            "Les candidats différés restent auditables et pourront "
            "être exécutés ultérieurement sur demande explicite."
        ),
        (
            "Analytical Value Guard version: "
            f"{ANALYTICAL_VALUE_GUARD_RULE_VERSION}."
        ),
    ]

    return AnalysisPrioritizationReport(
        discovered_count=
            len(
                discovery.candidates
            ),

        selected_count=
            len(
                selected_ids
            ),

        deferred_count=
            len(
                deferred_ids
            ),

        rejected_count=
            len(
                rejected_ids
            ),

        selected_analysis_ids=
            selected_ids,

        deferred_analysis_ids=
            deferred_ids,

        rejected_analysis_ids=
            rejected_ids,

        decisions=
            decisions,

        selected_candidates=
            selected,

        family_selected_counts=
            dict(
                family_counts
            ),

        notes=
            notes,

        rule_version=
            ANALYSIS_PRIORITIZATION_RULE_VERSION,
    )


def _sorted_counter_dict(
    counter: Counter[str],
) -> dict[str, int]:
    """Return deterministic count mappings for API/UI use."""

    return {
        key:
            int(
                count
            )

        for (
            key,
            count,
        )
        in sorted(
            counter.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )
    }


def build_analysis_prioritization_audit(
    prioritization: AnalysisPrioritizationReport,
) -> AnalysisPrioritizationAudit:
    """
    Build the public, compact explainability contract for the
    deterministic exploratory-prioritization layer.

    The audit is derived exclusively from Python decisions that
    already governed the execution shortlist. No LLM decision is
    introduced here.
    """

    decision_counts: Counter[str] = Counter(
        decision.decision
        for decision
        in prioritization.decisions
    )

    reason_counts: Counter[str] = Counter(
        decision.reason_code
        for decision
        in prioritization.decisions
    )

    non_execution_reason_counts: Counter[str] = Counter(
        decision.reason_code
        for decision
        in prioritization.decisions
        if (
            decision.decision
            !=
            "selected"
        )
    )

    return AnalysisPrioritizationAudit(
        discovered_count=
            prioritization.discovered_count,

        selected_for_execution_count=
            prioritization.selected_count,

        deferred_count=
            prioritization.deferred_count,

        rejected_count=
            prioritization.rejected_count,

        decision_counts=
            _sorted_counter_dict(
                decision_counts
            ),

        reason_counts=
            _sorted_counter_dict(
                reason_counts
            ),

        non_execution_reason_counts=
            _sorted_counter_dict(
                non_execution_reason_counts
            ),

        family_selected_counts=
            dict(
                sorted(
                    prioritization
                    .family_selected_counts
                    .items()
                )
            ),

        decisions=[
            decision.model_copy(
                deep=True
            )
            for decision
            in prioritization.decisions
        ],

        prioritization_rule_version=
            prioritization.rule_version,

        analytical_value_guard_rule_version=
            ANALYTICAL_VALUE_GUARD_RULE_VERSION,

        audit_rule_version=
            ANALYSIS_PRIORITIZATION_AUDIT_RULE_VERSION,
    )



def build_prioritized_execution_discovery(
    *,
    source_discovery: AnalysisDiscoveryReport,
    prioritization: AnalysisPrioritizationReport,
) -> AnalysisDiscoveryReport:
    """
    Build an execution-only discovery view.

    The source discovery keeps the complete candidate inventory.

    Important:
    execution scope is taken from ``selected_candidates`` rather
    than reconstructed from ``selected_analysis_ids``.

    Older Discovery candidates are not guaranteed to have a
    globally unique ``analysis_id``. In particular, legacy
    time-series IDs can collide when the same value variable is
    paired with several temporal columns.

    Reconstructing the shortlist from a set of public IDs could
    therefore admit a deferred/rejected candidate sharing an ID
    with a selected candidate.

    The prioritization report already stores the exact selected
    candidate objects, so those objects are the authoritative
    execution scope.
    """

    selected_candidates = [
        candidate.model_copy(
            deep=True
        )

        for candidate
        in prioritization.selected_candidates
    ]

    single_count = sum(
        1
        for candidate
        in selected_candidates
        if (
            candidate.scope
            ==
            "single_dataset"
        )
    )

    cross_count = sum(
        1
        for candidate
        in selected_candidates
        if (
            candidate.scope
            ==
            "cross_dataset"
        )
    )

    return source_discovery.model_copy(
        deep=True,
        update={
            "candidate_count":
                len(
                    selected_candidates
                ),

            "single_dataset_candidate_count":
                single_count,

            "cross_dataset_candidate_count":
                cross_count,

            "candidates":
                selected_candidates,

            "discovery_notes": [
                *source_discovery.discovery_notes,
                (
                    "Execution shortlist generated by "
                    f"{ANALYSIS_PRIORITIZATION_RULE_VERSION}: "
                    f"{len(selected_candidates)} / "
                    f"{len(source_discovery.candidates)} "
                    "candidate(s) selected."
                ),
                (
                    "Execution scope reused the exact selected "
                    "candidate objects rather than reconstructing "
                    "them from analysis_id, preventing legacy "
                    "duplicate-ID collisions from widening the "
                    "shortlist."
                ),
            ],
        },
    )
