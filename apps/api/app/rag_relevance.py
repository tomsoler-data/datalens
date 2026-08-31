from __future__ import annotations


import re
import unicodedata


from typing import (
    Literal,
    TypeAlias,
)


from pydantic import (
    BaseModel,
)


from app.ai.provider import (
    client,
)

from app.security.llm_payload import (
    LLMPayloadClass,
    classified_llm_chat,
)


# ============================================================
# VERSION
# ============================================================

RELEVANCE_RULE_VERSION = (
    "rag_relevance_v0.8"
)


# ============================================================
# MODEL
# ============================================================

DEFAULT_RELEVANCE_MODEL = (
    "gemma3:4b"
)


# ============================================================
# TYPES
# ============================================================

RelevanceRelationType: TypeAlias = Literal[
    "explicit_request",
    "business_rule",
    "business_definition",
    "objective_support",
    "interpretation_context",
    "methodological_context",
    "not_relevant",
]


RelevanceStrength: TypeAlias = Literal[
    "direct",
    "supporting",
    "none",
]


# ============================================================
# ANALYTICAL SIGNATURE
# ============================================================

class AnalyticalSignature(
    BaseModel
):
    family: (
        str
        | None
    ) = None

    measure_column: (
        str
        | None
    ) = None

    group_column: (
        str
        | None
    ) = None

    x_column: (
        str
        | None
    ) = None

    y_column: (
        str
        | None
    ) = None

    time_column: (
        str
        | None
    ) = None


# ============================================================
# RAW STRUCTURED OUTPUT
# ============================================================

class RawRelevanceDecision(
    BaseModel
):
    verdict: Literal[
        "relevant",
        "not_relevant",
    ]

    relation_type: RelevanceRelationType

    strength: RelevanceStrength

    evidence_unit_id: int

    reason: str


# ============================================================
# VERIFIED DECISION
# ============================================================

class RelevanceDecision(
    BaseModel
):
    verdict: Literal[
        "relevant",
        "not_relevant",
    ]

    relation_type: RelevanceRelationType

    strength: RelevanceStrength

    evidence_unit_id: (
        int
        | None
    ) = None

    evidence_quote: (
        str
        | None
    ) = None

    reason: str


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
Tu es le composant de validation documentaire d'un système
RAG d'analyse de données appelé DataLens.

Python a déjà :

1. construit un contrat analytique déterministe ;
2. découpé le document en unités locales ;
3. éliminé les unités qui ne peuvent pas satisfaire les
   variables et la structure du contrat.

Tu dois maintenant déterminer si UNE SEULE des unités
candidates restantes constitue réellement un contexte
documentaire pertinent pour le contrat analytique.

============================================================
RÈGLE DE SORTIE OBLIGATOIRE
============================================================

Le champ evidence_unit_id est toujours obligatoire.

Si :

verdict = "not_relevant"

alors :

evidence_unit_id = 0
relation_type = "not_relevant"
strength = "none"


Si :

verdict = "relevant"

alors :

evidence_unit_id doit correspondre exactement au numéro
d'une unité candidate fournie.

relation_type ne doit pas être "not_relevant".

strength doit être "direct" ou "supporting".

============================================================
UNE SEULE PREUVE
============================================================

Une seule unité doit suffire à justifier le verdict.

Il est interdit :

- de combiner deux unités ;
- de compléter une unité avec une information provenant
  d'une autre ;
- d'inventer une relation absente de l'unité choisie.

============================================================
CONTRAT ANALYTIQUE
============================================================

Le contrat fourni par Python est contraignant.

Respecte strictement :

- la famille analytique ;
- la mesure ;
- la variable de regroupement ;
- les variables X et Y ;
- la dimension temporelle ;
- la relation demandée.

============================================================
DISTINCTIONS OBLIGATOIRES
============================================================

Ne confonds jamais :

- prix et chiffre d'affaires ;
- prix et panier moyen ;
- montant total et panier moyen ;
- moyenne et médiane ;
- fréquence et comptage d'événements ;
- association et série temporelle ;
- âge et catégorie ;
- genre et catégorie.

Une proximité de domaine ne suffit pas.

============================================================
AGRÉGATIONS
============================================================

Respecte la sémantique des noms techniques :

average_ / mean_
    moyenne

median_
    médiane

sum_
    somme / total

_count
    comptage

Une moyenne n'est pas une médiane.

Un montant total n'est pas un panier moyen.

============================================================
RELATION_TYPE
============================================================

explicit_request
    Demande explicite de réaliser l'analyse.

business_rule
    Règle métier directement liée.

business_definition
    Définition métier directement utile.

objective_support
    Objectif directement aligné.

interpretation_context
    Contexte directement utile pour interpréter.

methodological_context
    Instruction méthodologique applicable.

not_relevant
    Aucun lien documentaire suffisamment direct.

============================================================
STRENGTH
============================================================

direct
    Correspondance explicite ou très claire.

supporting
    Contexte pertinent mais secondaire.

none
    Obligatoire pour not_relevant.

============================================================
SÉCURITÉ
============================================================

Le système privilégie la précision.

Un faux positif est plus dangereux qu'un passage pertinent
rejeté.

En cas de doute :

verdict = "not_relevant"
relation_type = "not_relevant"
strength = "none"
evidence_unit_id = 0

Retourne uniquement la structure JSON demandée.

Le champ reason doit être bref et précis.
""".strip()


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_whitespace(
    value: str,
) -> str:
    return (
        re.sub(
            r"\s+",
            " ",
            value,
        )
        .strip()
    )


def normalize_for_matching(
    value: str,
) -> str:
    decomposed = (
        unicodedata
        .normalize(
            "NFKD",
            value,
        )
    )


    without_accents = "".join(
        character

        for character
        in decomposed

        if not unicodedata.combining(
            character
        )
    )


    normalized = (
        normalize_whitespace(
            without_accents
        )
        .casefold()
        .replace(
            "_",
            " ",
        )
    )


    return normalized


def tokenize(
    value: str,
) -> set[
    str
]:
    return set(
        re.findall(
            r"[a-z0-9]+",
            normalize_for_matching(
                value
            ),
        )
    )


# ============================================================
# ANALYTICAL CONTRACT PARSING
# ============================================================

def extract_contract_field(
    *,
    finding: str,
    label: str,
) -> (
    str
    | None
):
    pattern = re.compile(
        (
            r"(?mi)^"
            +
            re.escape(
                label
            )
            +
            r"\s*:\s*(.+?)\s*$"
        )
    )


    match = pattern.search(
        finding
    )


    if match is None:
        return None


    value = (
        match
        .group(
            1
        )
        .strip()
        .strip(
            "`"
        )
    )


    if not value:
        return None


    return value


def build_analytical_signature(
    finding: str,
) -> AnalyticalSignature:
    return (
        AnalyticalSignature(
            family=
                extract_contract_field(
                    finding=
                        finding,

                    label=
                        "Famille analytique",
                ),

            measure_column=
                extract_contract_field(
                    finding=
                        finding,

                    label=
                        "Mesure",
                ),

            group_column=
                extract_contract_field(
                    finding=
                        finding,

                    label=
                        "Variable de regroupement",
                ),

            x_column=
                extract_contract_field(
                    finding=
                        finding,

                    label=
                        "Variable X",
                ),

            y_column=
                extract_contract_field(
                    finding=
                        finding,

                    label=
                        "Variable Y",
                ),

            time_column=
                extract_contract_field(
                    finding=
                        finding,

                    label=
                        "Dimension temporelle",
                ),
        )
    )


# ============================================================
# EVIDENCE UNIT EXTRACTION
# ============================================================

INLINE_BULLET_PATTERN = re.compile(
    r"\s*[•●▪◦‣]\s*"
)


NUMBERED_ITEM_PATTERN = re.compile(
    r"(?m)^\s*\d+[.)]\s+"
)


SENTENCE_BOUNDARY_PATTERN = re.compile(
    (
        r"(?<=[.!?…])\s+"
        r"(?=[A-ZÀ-ÖØ-Þ0-9«\"'])"
    )
)


def split_sentences(
    value: str,
) -> list[
    str
]:
    normalized = (
        normalize_whitespace(
            value
        )
    )


    if not normalized:
        return []


    candidates = (
        SENTENCE_BOUNDARY_PATTERN
        .split(
            normalized
        )
    )


    return [
        normalize_whitespace(
            candidate
        )

        for candidate
        in candidates

        if normalize_whitespace(
            candidate
        )
    ]


def build_evidence_units(
    passage: str,
) -> list[
    str
]:
    normalized_passage = (
        passage
        .replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
        .strip()
    )


    if not normalized_passage:
        return []


    normalized_passage = re.sub(
        r"(?m)^\s*[-*]\s+",
        "● ",
        normalized_passage,
    )


    normalized_passage = (
        NUMBERED_ITEM_PATTERN
        .sub(
            "● ",
            normalized_passage,
        )
    )


    raw_paragraphs = re.split(
        r"\n\s*\n+",
        normalized_passage,
    )


    units: list[
        str
    ] = []


    for raw_paragraph in raw_paragraphs:
        if not raw_paragraph.strip():
            continue


        structural_parts = (
            INLINE_BULLET_PATTERN
            .split(
                raw_paragraph
            )
        )


        for structural_part in structural_parts:
            normalized_part = (
                normalize_whitespace(
                    structural_part
                )
            )


            if not normalized_part:
                continue


            sentence_units = (
                split_sentences(
                    normalized_part
                )
            )


            if sentence_units:
                units.extend(
                    sentence_units
                )


            else:
                units.append(
                    normalized_part
                )


    if not units:
        compact = (
            normalize_whitespace(
                normalized_passage
            )
        )


        if compact:
            units.append(
                compact
            )


    return units


# ============================================================
# LEXICAL HELPERS
# ============================================================

def contains_any_phrase(
    value: str,
    phrases: set[
        str
    ],
) -> bool:
    normalized_value = (
        normalize_for_matching(
            value
        )
    )


    return any(
        normalize_for_matching(
            phrase
        )
        in
        normalized_value

        for phrase
        in phrases
    )


def contains_any_token(
    value: str,
    terms: set[
        str
    ],
) -> bool:
    value_tokens = (
        tokenize(
            value
        )
    )


    normalized_terms = {
        normalize_for_matching(
            term
        )

        for term
        in terms
    }


    return bool(
        value_tokens
        &
        normalized_terms
    )


# ============================================================
# SEMANTIC CUES
# ============================================================

AVERAGE_CUES = {
    "average",
    "mean",
    "moyen",
    "moyenne",
}


MEDIAN_CUES = {
    "median",
    "mediane",
}


SUM_CUES = {
    "sum",
    "somme",
    "total",
    "totale",
}


COUNT_CUES = {
    "count",
    "nombre",
    "comptage",
}


BASKET_CUES = {
    "basket",
    "panier",
}


PRICE_CUES = {
    "price",
    "prix",
    "montant",
}


CATEGORY_CUES = {
    "categ",
    "category",
    "categories",
    "categorie",
}


AGE_CUES = {
    "age",
}


EVENT_CUES = {
    "event",
    "events",
    "evenement",
    "evenements",
}


TEMPORAL_CUES = {
    "time",
    "temporal",
    "temporel",
    "temporelle",
    "date",
    "dates",
    "month",
    "months",
    "mois",
    "year",
    "years",
    "annee",
    "annees",
    "day",
    "days",
    "jour",
    "jours",
    "week",
    "weeks",
    "semaine",
    "semaines",
    "quarter",
    "quarters",
    "trimestre",
    "trimestres",
    "period",
    "periods",
    "periode",
    "periodes",
    "mensuel",
    "mensuelle",
    "monthly",
    "annuel",
    "annuelle",
    "yearly",
    "evolution",
    "evoluer",
    "evolue",
    "tendance",
    "trend",
}


# ============================================================
# VARIABLE SUPPORT
# ============================================================

def supports_base_concept(
    *,
    evidence: str,
    variable: str,
) -> bool:
    normalized_variable = (
        normalize_for_matching(
            variable
        )
    )


    if (
        normalized_variable
        in
        normalize_for_matching(
            evidence
        )
    ):
        return True


    tokens = (
        normalized_variable
        .split()
    )


    if (
        "basket"
        in
        tokens
    ):
        return (
            contains_any_token(
                evidence,
                BASKET_CUES,
            )
        )


    if (
        "price"
        in
        tokens
    ):
        return (
            contains_any_token(
                evidence,
                PRICE_CUES,
            )
        )


    if (
        "categ"
        in
        tokens
        or
        "category"
        in
        tokens
    ):
        return (
            contains_any_token(
                evidence,
                CATEGORY_CUES,
            )
        )


    if (
        "age"
        in
        tokens
    ):
        return (
            contains_any_token(
                evidence,
                AGE_CUES,
            )
        )


    if (
        "event"
        in
        tokens
        or
        "events"
        in
        tokens
    ):
        return (
            contains_any_token(
                evidence,
                EVENT_CUES,
            )
        )


    ignored_tokens = {
        "at",
        "first",
        "of",
        "the",
        "by",
        "per",
    }


    meaningful_tokens = [
        token

        for token
        in tokens

        if (
            token
            not in
            ignored_tokens
            and
            len(
                token
            )
            >=
            3
        )
    ]


    if not meaningful_tokens:
        return False


    evidence_tokens = (
        tokenize(
            evidence
        )
    )


    return all(
        token
        in
        evidence_tokens

        for token
        in meaningful_tokens
    )


def evidence_supports_variable(
    *,
    evidence: str,
    variable: (
        str
        | None
    ),
) -> bool:
    if variable is None:
        return True


    normalized_variable = (
        normalize_for_matching(
            variable
        )
    )


    if (
        normalized_variable
        in
        normalize_for_matching(
            evidence
        )
    ):
        return True


    if (
        normalized_variable
        .startswith(
            "average "
        )
        or
        normalized_variable
        .startswith(
            "mean "
        )
    ):
        base_variable = (
            normalized_variable
            .split(
                " ",
                1,
            )[
                1
            ]
        )


        return (
            contains_any_token(
                evidence,
                AVERAGE_CUES,
            )
            and
            supports_base_concept(
                evidence=
                    evidence,

                variable=
                    base_variable,
            )
        )


    if (
        normalized_variable
        .startswith(
            "median "
        )
    ):
        base_variable = (
            normalized_variable
            .split(
                " ",
                1,
            )[
                1
            ]
        )


        return (
            contains_any_token(
                evidence,
                MEDIAN_CUES,
            )
            and
            supports_base_concept(
                evidence=
                    evidence,

                variable=
                    base_variable,
            )
        )


    if (
        normalized_variable
        .startswith(
            "sum "
        )
    ):
        base_variable = (
            normalized_variable
            .split(
                " ",
                1,
            )[
                1
            ]
        )


        return (
            contains_any_token(
                evidence,
                SUM_CUES,
            )
            and
            supports_base_concept(
                evidence=
                    evidence,

                variable=
                    base_variable,
            )
        )


    if (
        normalized_variable
        .endswith(
            " count"
        )
    ):
        base_variable = (
            normalized_variable[
                :
                -
                len(
                    " count"
                )
            ]
            .strip()
        )


        return (
            contains_any_token(
                evidence,
                COUNT_CUES,
            )
            and
            supports_base_concept(
                evidence=
                    evidence,

                variable=
                    base_variable,
            )
        )


    if (
        normalized_variable
        .startswith(
            "count "
        )
    ):
        base_variable = (
            normalized_variable
            .split(
                " ",
                1,
            )[
                1
            ]
        )


        return (
            contains_any_token(
                evidence,
                COUNT_CUES,
            )
            and
            supports_base_concept(
                evidence=
                    evidence,

                variable=
                    base_variable,
            )
        )


    if (
        normalized_variable
        .startswith(
            "age "
        )
    ):
        return (
            contains_any_token(
                evidence,
                AGE_CUES,
            )
        )


    return (
        supports_base_concept(
            evidence=
                evidence,

            variable=
                variable,
        )
    )


# ============================================================
# TEMPORAL SUPPORT
# ============================================================

def evidence_has_temporal_signal(
    evidence: str,
) -> bool:
    if (
        contains_any_token(
            evidence,
            TEMPORAL_CUES,
        )
    ):
        return True


    temporal_phrases = {
        "au fil du temps",
        "au fil des mois",
        "dans le temps",
        "over time",
    }


    return (
        contains_any_phrase(
            evidence,
            temporal_phrases,
        )
    )


# ============================================================
# HEADING DETECTION
# ============================================================

def is_heading_like_unit(
    evidence: str,
) -> bool:
    normalized = (
        evidence
        .strip()
    )


    if not normalized:
        return True


    if (
        normalized
        .endswith(
            ":"
        )
    ):
        return True


    return False


# ============================================================
# DETERMINISTIC CONTRACT MATCHING
# ============================================================

def evidence_unit_matches_contract(
    *,
    finding: str,
    evidence: str,
) -> bool:
    if (
        is_heading_like_unit(
            evidence
        )
    ):
        return False


    signature = (
        build_analytical_signature(
            finding
        )
    )


    family = (
        signature.family
        or ""
    )


    if (
        family
        ==
        "aggregate_breakdown"
    ):
        return (
            evidence_supports_variable(
                evidence=
                    evidence,

                variable=
                    signature.measure_column,
            )
            and
            evidence_supports_variable(
                evidence=
                    evidence,

                variable=
                    signature.group_column,
            )
        )


    if (
        family
        ==
        "group_comparison"
    ):
        return (
            evidence_supports_variable(
                evidence=
                    evidence,

                variable=
                    signature.measure_column,
            )
            and
            evidence_supports_variable(
                evidence=
                    evidence,

                variable=
                    signature.group_column,
            )
        )


    if (
        family
        ==
        "quantitative_association"
    ):
        return (
            evidence_supports_variable(
                evidence=
                    evidence,

                variable=
                    signature.x_column,
            )
            and
            evidence_supports_variable(
                evidence=
                    evidence,

                variable=
                    signature.y_column,
            )
        )


    if (
        family
        ==
        "time_series"
    ):
        return (
            evidence_supports_variable(
                evidence=
                    evidence,

                variable=
                    signature.measure_column,
            )
            and
            evidence_has_temporal_signal(
                evidence
            )
        )


    return True


# ============================================================
# CANDIDATE FILTER
# ============================================================

def filter_candidate_evidence_units(
    *,
    finding: str,
    evidence_units: list[
        str
    ],
) -> list[
    tuple[
        int,
        str,
    ]
]:
    candidates: list[
        tuple[
            int,
            str,
        ]
    ] = []


    for (
        index,
        evidence,
    ) in enumerate(
        evidence_units,
        start=1,
    ):
        if (
            evidence_unit_matches_contract(
                finding=
                    finding,

                evidence=
                    evidence,
            )
        ):
            candidates.append(
                (
                    index,
                    evidence,
                )
            )


    return candidates


# ============================================================
# USER PROMPT
# ============================================================

def build_user_prompt(
    *,
    finding: str,
    candidate_units: list[
        tuple[
            int,
            str,
        ]
    ],
) -> str:
    normalized_finding = (
        finding
        .strip()
    )


    if not normalized_finding:
        raise ValueError(
            "Le finding ne peut pas être vide."
        )


    if not candidate_units:
        raise ValueError(
            (
                "Aucune unité candidate "
                "n'est disponible."
            )
        )


    rendered_units = "\n\n".join(
        (
            f"[UNITÉ {unit_id}]\n"
            f"{unit}"
        )

        for (
            unit_id,
            unit,
        )
        in candidate_units
    )


    return (
        "CONTRAT ANALYTIQUE:\n"
        f"{normalized_finding}\n\n"
        "UNITÉS CANDIDATES APRÈS FILTRAGE PYTHON:\n"
        f"{rendered_units}\n\n"
        "Décide si UNE SEULE unité candidate constitue "
        "réellement un contexte documentaire pertinent.\n\n"
        "Utilise exclusivement les identifiants affichés.\n"
        "Ne combine jamais plusieurs unités."
    )


# ============================================================
# NEGATIVE DECISION
# ============================================================

def build_negative_decision(
    *,
    reason: str,
) -> RelevanceDecision:
    normalized_reason = (
        reason
        .strip()
    )


    if not normalized_reason:
        normalized_reason = (
            "Aucune preuve documentaire locale ne couvre "
            "le contrat analytique complet."
        )


    return (
        RelevanceDecision(
            verdict=
                "not_relevant",

            relation_type=
                "not_relevant",

            strength=
                "none",

            evidence_unit_id=
                None,

            evidence_quote=
                None,

            reason=
                normalized_reason,
        )
    )


# ============================================================
# EXPLICIT REQUEST RELATION GUARD
# ============================================================

EXPLICIT_REQUEST_RELATION_PHRASES = {
    "analyse à réaliser",
    "analyse a realiser",
    "analyse à effectuer",
    "analyse a effectuer",
    "j'aimerais que",
    "je souhaite",
    "nous souhaitons",
    "veuillez",
}


EXPLICIT_REQUEST_RELATION_ACTIONS = {
    "calcule",
    "calculer",
    "calculez",
    "compte",
    "compter",
    "comptez",
    "combien",
    "analyser",
    "analysez",
    "étudier",
    "etudier",
    "étudiez",
    "etudiez",
    "comparer",
    "comparez",
}


def evidence_is_exact_explicit_request(
    *,
    finding: str,
    evidence: str,
) -> bool:
    """
    Promote a selected evidence unit to explicit_request only
    when:

    1. it is exactly the finding title;
    2. the title itself contains an explicit analytical
       request/action cue.

    Exact equality prevents definitions, business rules and
    interpretation sentences from being promoted only because
    they share vocabulary with the analytical request.
    """

    title = extract_contract_field(
        finding=
            finding,

        label=
            "Titre du finding",
    )


    if title is None:
        return False


    if (
        normalize_for_matching(
            title
        )
        !=
        normalize_for_matching(
            evidence
        )
    ):
        return False


    if contains_any_phrase(
        title,
        EXPLICIT_REQUEST_RELATION_PHRASES,
    ):
        return True


    if contains_any_token(
        title,
        EXPLICIT_REQUEST_RELATION_ACTIONS,
    ):
        return True


    return False


# ============================================================
# DETERMINISTIC DECISION VERIFICATION
# ============================================================

def verify_decision(
    *,
    finding: str,
    decision: RawRelevanceDecision,
    candidate_units: list[
        tuple[
            int,
            str,
        ]
    ],
) -> RelevanceDecision:
    reason = (
        decision
        .reason
        .strip()
    )


    if (
        decision.verdict
        ==
        "not_relevant"
    ):
        return (
            build_negative_decision(
                reason=
                    reason,
            )
        )


    if (
        decision.relation_type
        ==
        "not_relevant"
        or
        decision.strength
        ==
        "none"
    ):
        return (
            build_negative_decision(
                reason=(
                    "Décision positive incohérente avec "
                    "la relation ou la force documentaire."
                ),
            )
        )


    if (
        decision.evidence_unit_id
        <=
        0
    ):
        return (
            build_negative_decision(
                reason=(
                    "Le modèle a déclaré le passage pertinent "
                    "sans sélectionner de preuve locale."
                ),
            )
        )


    candidate_lookup = {
        unit_id:
            unit

        for (
            unit_id,
            unit,
        )
        in candidate_units
    }


    evidence_quote = (
        candidate_lookup.get(
            decision.evidence_unit_id
        )
    )


    if evidence_quote is None:
        return (
            build_negative_decision(
                reason=(
                    "Le modèle a sélectionné une unité qui "
                    "n'appartient pas aux candidats validés "
                    "par Python."
                ),
            )
        )


    if not (
        evidence_unit_matches_contract(
            finding=
                finding,

            evidence=
                evidence_quote,
        )
    ):
        return (
            build_negative_decision(
                reason=(
                    "La preuve sélectionnée ne satisfait "
                    "plus le contrat lors de la validation "
                    "déterministe finale."
                ),
            )
        )


    verified_relation_type = (
        decision.relation_type
    )

    verified_strength = (
        decision.strength
    )

    verified_reason = (
        reason
        or
        (
            "Une unité documentaire validée par "
            "Python couvre le contrat analytique."
        )
    )


    if evidence_is_exact_explicit_request(
        finding=
            finding,

        evidence=
            evidence_quote,
    ):
        verified_relation_type = (
            "explicit_request"
        )

        verified_strength = (
            "direct"
        )

        verified_reason = (
            "La preuve sélectionnée correspond "
            "exactement au titre d'une demande "
            "analytique explicite."
        )


    return (
        RelevanceDecision(
            verdict=
                "relevant",

            relation_type=
                verified_relation_type,

            strength=
                verified_strength,

            evidence_unit_id=
                decision
                .evidence_unit_id,

            evidence_quote=
                evidence_quote,

            reason=
                verified_reason,
        )
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_relevance(
    *,
    finding: str,
    passage: str,
    model: str = (
        DEFAULT_RELEVANCE_MODEL
    ),
) -> RelevanceDecision:
    normalized_finding = (
        finding
        .strip()
    )


    normalized_passage = (
        passage
        .strip()
    )


    if not normalized_finding:
        raise ValueError(
            "Le finding ne peut pas être vide."
        )


    if not normalized_passage:
        raise ValueError(
            (
                "Le passage documentaire "
                "ne peut pas être vide."
            )
        )


    evidence_units = (
        build_evidence_units(
            normalized_passage
        )
    )


    if not evidence_units:
        return (
            build_negative_decision(
                reason=(
                    "Aucune unité documentaire exploitable "
                    "n'a pu être extraite du passage."
                ),
            )
        )


    candidate_units = (
        filter_candidate_evidence_units(
            finding=
                normalized_finding,

            evidence_units=
                evidence_units,
        )
    )


    if not candidate_units:
        return (
            build_negative_decision(
                reason=(
                    "Aucune unité documentaire ne satisfait "
                    "les variables et la structure du contrat "
                    "analytique."
                ),
            )
        )


    user_prompt = (
        build_user_prompt(
            finding=
                normalized_finding,

            candidate_units=
                candidate_units,
        )
    )


    try:
        response = classified_llm_chat(
            client,
            payload_class=(
                LLMPayloadClass
                .DOCUMENT_CONTENT
            ),
            model=
                model,

            messages=[
                {
                    "role":
                        "system",

                    "content":
                        SYSTEM_PROMPT,
                },

                {
                    "role":
                        "user",

                    "content":
                        user_prompt,
                },
            ],

            format=(
                RawRelevanceDecision
                .model_json_schema()
            ),

            options={
                "temperature":
                    0,
            },
        )


    except Exception as error:
        raise RuntimeError(
            (
                "La validation de pertinence "
                "par Ollama a échoué."
            )
        ) from error


    content = (
        response
        .message
        .content
    )


    try:
        raw_decision = (
            RawRelevanceDecision
            .model_validate_json(
                content
            )
        )


    except Exception as error:
        raise RuntimeError(
            (
                "Gemma a retourné une réponse "
                "qui ne respecte pas le schéma "
                "de pertinence attendu."
            )
        ) from error


    return (
        verify_decision(
            finding=
                normalized_finding,

            decision=
                raw_decision,

            candidate_units=
                candidate_units,
        )
    )