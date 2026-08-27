from __future__ import annotations

import re
import unicodedata

from collections import defaultdict

from typing import (
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
)

from app.ai.provider import (
    client,
)

from app.security.llm_payload import (
    LLMPayloadClass,
    classified_llm_chat,
)

from app.rag import (
    DocumentChunk,
    DocumentIngestionReport,
)

from app.rag_relevance import (
    build_evidence_units,
)


# ============================================================
# VERSION
# ============================================================

DOCUMENT_SUMMARY_RULE_VERSION = (
    "document_summary_v0.5"
)


DEFAULT_DOCUMENT_SUMMARY_MODEL = (
    "gemma3:4b"
)


# ============================================================
# LIMITS
# ============================================================

MAX_RAW_CLAIMS_PER_BATCH = 20

MAX_SUMMARY_POINTS_PER_DOCUMENT = 12

MAX_ANALYTICAL_REQUESTS_PER_DOCUMENT = 40

MAX_GLOBAL_SUMMARY_POINTS = 20

MAX_GLOBAL_ANALYTICAL_REQUESTS = 60

MAX_STATEMENT_LENGTH = 420

MAX_EVIDENCE_QUOTE_LENGTH = 900

MAX_CONTEXT_QUOTE_LENGTH = 900

MAX_BATCH_UNITS = 70

MAX_BATCH_CHARACTERS = 14_000


# ============================================================
# CLAIM TYPES
# ============================================================

DocumentClaimCategory = Literal[
    "business_context",
    "business_objective",
    "analytical_request",
    "definition",
    "business_rule",
    "constraint",
    "methodology",
]


# ============================================================
# RAW LLM OUTPUT
# ============================================================

class RawDocumentClaim(
    BaseModel
):
    category: DocumentClaimCategory

    statement: str

    source_chunk_id: str

    evidence_unit_id: int


class RawDocumentBatchExtraction(
    BaseModel
):
    claims: list[
        RawDocumentClaim
    ] = Field(
        default_factory=list,
        max_length=
            MAX_RAW_CLAIMS_PER_BATCH,
    )


# ============================================================
# VERIFIED OUTPUT
# ============================================================

class DocumentSummaryCitation(
    BaseModel
):
    chunk_id: str

    document_id: str

    filename: str

    source_locator: str

    page_number: (
        int
        | None
    ) = None


class VerifiedDocumentClaim(
    BaseModel
):
    category: DocumentClaimCategory

    statement: str

    evidence_quote: str

    evidence_unit_id: int

    context_quote: (
        str
        | None
    ) = None

    context_evidence_unit_id: (
        int
        | None
    ) = None

    citation: DocumentSummaryCitation


class PerDocumentSummary(
    BaseModel
):
    document_id: str

    filename: str

    summary_points: list[
        VerifiedDocumentClaim
    ]

    analytical_requests: list[
        VerifiedDocumentClaim
    ]

    verified_claim_count: int

    source_chunk_count: int


class DocumentSummaryReport(
    BaseModel
):
    status: Literal[
        "ready",
        "abstained",
    ]

    document_count: int

    chunk_count: int

    verified_claim_count: int

    summary_point_count: int

    analytical_request_count: int

    summary_points: list[
        VerifiedDocumentClaim
    ]

    analytical_requests: list[
        VerifiedDocumentClaim
    ]

    documents: list[
        PerDocumentSummary
    ]

    warnings: list[
        str
    ]

    abstention_reason: (
        str
        | None
    ) = None

    model: str

    summary_rule_version: str = (
        DOCUMENT_SUMMARY_RULE_VERSION
    )


# ============================================================
# INTERNAL EVIDENCE UNIT
# ============================================================

class DocumentEvidenceUnit(
    BaseModel
):
    document_id: str

    filename: str

    chunk_id: str

    chunk_index: int

    page_number: (
        int
        | None
    )

    source_locator: str

    evidence_unit_id: int

    text: str

    parent_context: (
        str
        | None
    ) = None

    parent_evidence_unit_id: (
        int
        | None
    ) = None


# ============================================================
# PROMPT
# ============================================================

DOCUMENT_SUMMARY_SYSTEM_PROMPT = """
Tu extrais uniquement des informations explicitement présentes
dans des documents métier.

Ton rôle n'est PAS d'interpréter les données statistiques.

Ton rôle est d'identifier des éléments documentaires utiles :
- contexte métier ;
- objectifs métier ;
- demandes analytiques explicites ;
- définitions ;
- règles métier ;
- contraintes ;
- instructions méthodologiques.

RÈGLES STRICTES

1. Chaque claim doit pointer vers une unité de preuve fournie.
2. source_chunk_id doit correspondre exactement à un chunk fourni.
3. evidence_unit_id doit correspondre exactement à une unité fournie.
4. N'invente aucune information absente du document.
5. Ne fusionne JAMAIS deux puces ou deux unités différentes.
6. Une unité de preuve doit produire au maximum une information
   atomique correspondant à son propre contenu.
7. Ne transforme pas une simple description en demande analytique.
8. Utilise analytical_request uniquement lorsqu'une analyse,
   mesure, comparaison, relation, indicateur ou représentation
   est explicitement demandée.
9. Une phrase générale comme "mieux comprendre nos clients"
   est un business_objective, pas une analytical_request.
10. Un titre de document ou de section n'est jamais une
    analytical_request.
11. Ne remplace pas moyenne par médiane, somme par moyenne,
    fréquence par montant, prix par chiffre d'affaires,
    ou une métrique par une métrique supposée équivalente.
12. Ne déduis aucune causalité.
13. Ne crée aucune recommandation.
14. Si rien d'utile n'est explicitement présent, retourne claims=[].
15. L'objectif utilisateur éventuellement fourni sert seulement
    à prioriser la lecture. Il ne constitue jamais une preuve.
16. Préfère les faux négatifs aux faux positifs.

LISTES

Python fournit parfois VERIFIED_PARENT_CONTEXT.

Ce contexte est fourni uniquement pour expliquer la fonction
d'une puce dans une liste.

Tu ne dois jamais recopier dans le statement une information
qui appartient à une autre puce.

Exemple :

VERIFIED_PARENT_CONTEXT:
"Nous souhaitons élaborer différents graphiques..."

TEXT:
"nombre de transactions"

Le claim peut indiquer qu'il s'agit d'une demande analytique,
mais il ne doit parler QUE du nombre de transactions.

CATÉGORIES

business_context
    Information décrivant le contexte de l'entreprise,
    du projet ou du problème métier.

business_objective
    Objectif métier général explicitement formulé.

analytical_request
    Analyse, métrique, indicateur, comparaison, relation
    ou graphique concret explicitement demandé.

definition
    Définition explicite d'un concept, indicateur ou terme.

business_rule
    Règle métier explicitement formulée.

constraint
    Contrainte, limite ou exigence explicitement formulée.

methodology
    Instruction méthodologique explicitement formulée.

Retourne uniquement la structure JSON demandée.
""".strip()


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_whitespace(
    value: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def normalize_for_matching(
    value: str,
) -> str:
    decomposed = (
        unicodedata.normalize(
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
        without_accents
        .casefold()
        .replace(
            "_",
            " ",
        )
    )

    normalized = re.sub(
        r"[^a-z0-9%]+",
        " ",
        normalized,
    )

    return normalize_whitespace(
        normalized
    )


def clean_evidence_statement(
    value: str,
) -> str:
    value = (
        normalize_whitespace(
            value
        )
    )

    value = re.sub(
        r"^[●•\-*]\s*",
        "",
        value,
    )

    value = value.strip(
        " ,;:"
    )

    return value


# ============================================================
# NUMERIC VALIDATION
# ============================================================

NUMERIC_TOKEN_PATTERN = re.compile(
    r"""
    (?<![A-Za-zÀ-ÿ0-9_])
    [-+]?
    (?:
        \d+(?:[.,]\d+)?
        |
        [.,]\d+
    )
    %?
    (?![A-Za-zÀ-ÿ0-9_])
    """,
    re.VERBOSE,
)


def normalize_numeric_token(
    token: str,
) -> str:
    return (
        token
        .strip()
        .replace(
            ",",
            ".",
        )
    )


def extract_numeric_tokens(
    value: str,
) -> set[
    str
]:
    return {
        normalize_numeric_token(
            token
        )
        for token
        in NUMERIC_TOKEN_PATTERN.findall(
            value
        )
    }


# ============================================================
# TOKEN VALIDATION
# ============================================================

STOPWORDS = {
    "a",
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "cette",
    "de",
    "des",
    "du",
    "dans",
    "en",
    "et",
    "est",
    "etre",
    "il",
    "la",
    "le",
    "les",
    "leur",
    "leurs",
    "l",
    "nos",
    "notre",
    "ou",
    "par",
    "pour",
    "que",
    "qui",
    "se",
    "sur",
    "un",
    "une",
    "vos",
    "votre",
    "the",
    "of",
    "and",
    "to",
    "in",
    "for",
    "with",
    "is",
    "are",
}


SAFE_PARAPHRASE_TOKENS = {
    "analyse",
    "analyser",
    "demande",
    "demander",
    "document",
    "indique",
    "indiquer",
    "objectif",
    "souhaite",
    "souhaiter",
    "etudier",
    "comprendre",
}


def content_tokens(
    value: str,
) -> set[
    str
]:
    normalized = (
        normalize_for_matching(
            value
        )
    )

    return {
        token
        for token
        in normalized.split()
        if (
            len(
                token
            )
            >=
            3
            and
            token
            not in STOPWORDS
        )
    }


# ============================================================
# REQUEST / OBJECTIVE DETECTION
# ============================================================

EXPLICIT_RELATION_CUES = {
    "association",
    "comparer",
    "comparaison",
    "correlation",
    "etudier",
    "lien",
    "mesurer",
    "relation",
    "regarder",
}


GENERAL_OBJECTIVE_CUES = {
    "comprendre",
    "objectif",
    "souhaitons",
    "souhaite",
    "aimerais",
    "aimerions",
}


LIST_PARENT_CUES = {
    "graphiques",
    "graphique",
    "souhaitons",
    "souhaite",
    "aimerais",
    "aimerions",
    "interessant",
    "informations",
    "zoom",
    "analyse",
    "analyser",
}


def looks_like_explicit_relation_request(
    value: str,
) -> bool:
    tokens = (
        content_tokens(
            value
        )
    )

    return bool(
        tokens
        &
        EXPLICIT_RELATION_CUES
    )


def looks_like_general_objective(
    value: str,
) -> bool:
    tokens = (
        content_tokens(
            value
        )
    )


    if not (
        tokens
        &
        GENERAL_OBJECTIVE_CUES
    ):
        return False


    if (
        tokens
        &
        {
            "lien",
            "relation",
            "correlation",
            "association",
            "comparer",
            "comparaison",
            "mesurer",
            "etudier",
            "regarder",
        }
    ):
        return False


    return True


def looks_like_list_parent(
    value: str,
) -> bool:
    normalized = (
        normalize_whitespace(
            value
        )
    )


    if not normalized:
        return False


    if not normalized.endswith(
        ":"
    ):
        return False


    tokens = (
        content_tokens(
            normalized
        )
    )


    return bool(
        tokens
        &
        LIST_PARENT_CUES
    )


def is_list_terminator(
    value: str,
) -> bool:
    normalized = (
        normalize_for_matching(
            value
        )
    )


    return normalized in {
        "etc",
        "etcetera",
    }


# ============================================================
# EVIDENCE EXTRACTION
# ============================================================

# ============================================================
# COUNTED LIST SCOPE
# ============================================================

COUNTED_LIST_NOUNS = {
    "analyse",
    "analyses",
    "correlation",
    "correlations",
    "graphique",
    "graphiques",
    "indicateur",
    "indicateurs",
    "relation",
    "relations",
    "comparaison",
    "comparaisons",
    "test",
    "tests",
    "mesure",
    "mesures",
    "point",
    "points",
    "question",
    "questions",
    "demande",
    "demandes",
    "variable",
    "variables",
    "element",
    "elements",
}


def declared_list_item_count(
    value: str,
) -> (
    int
    | None
):
    """
    Return an explicitly declared number of list items.

    Examples:

        "5 corr?lations :" -> 5
        "3 analyses :"     -> 3

    The count is accepted only when:

    - the text is already structurally recognized as a
      list parent;
    - the integer is associated with a known analytical /
      list noun;
    - the value remains within a conservative range.

    This prevents unrelated numbers such as years from
    accidentally defining list scope.
    """

    if not looks_like_list_parent(
        value
    ):
        return None


    normalized = (
        normalize_for_matching(
            value
        )
    )


    matches = re.finditer(
        r"\b(\d{1,3})\s+([a-z]+)\b",
        normalized,
    )


    for match in matches:
        count = int(
            match.group(
                1
            )
        )


        noun = (
            match.group(
                2
            )
        )


        if (
            noun
            not in
            COUNTED_LIST_NOUNS
        ):
            continue


        if (
            count
            <
            1
            or
            count
            >
            100
        ):
            continue


        return count


    return None


# ============================================================
# EVIDENCE EXTRACTION
# ============================================================

def evidence_units_for_chunk(
    chunk: DocumentChunk,
) -> list[
    DocumentEvidenceUnit
]:
    raw_units = (
        build_evidence_units(
            chunk.text
        )
    )


    prepared_units: list[
        tuple[
            int,
            str,
        ]
    ] = []


    for (
        index,
        raw_unit,
    ) in enumerate(
        raw_units,
        start=1,
    ):
        normalized_unit = (
            normalize_whitespace(
                raw_unit
            )
        )


        if not normalized_unit:
            continue


        prepared_units.append(
            (
                index,
                normalized_unit,
            )
        )


    output: list[
        DocumentEvidenceUnit
    ] = []


    active_parent_text: (
        str
        | None
    ) = None


    active_parent_id: (
        int
        | None
    ) = None


    active_parent_expected_count: (
        int
        | None
    ) = None


    active_parent_consumed_count = (
        0
    )


    for (
        evidence_unit_id,
        normalized_unit,
    ) in prepared_units:
        current_is_parent = (
            looks_like_list_parent(
                normalized_unit
            )
        )


        # ----------------------------------------------------
        # NEW LIST PARENT
        # ----------------------------------------------------

        if current_is_parent:
            active_parent_text = (
                normalized_unit
            )

            active_parent_id = (
                evidence_unit_id
            )

            active_parent_expected_count = (
                declared_list_item_count(
                    normalized_unit
                )
            )

            active_parent_consumed_count = (
                0
            )


            output.append(
                DocumentEvidenceUnit(
                    document_id=
                        chunk.document_id,

                    filename=
                        chunk.filename,

                    chunk_id=
                        chunk.chunk_id,

                    chunk_index=
                        chunk.chunk_index,

                    page_number=
                        chunk.page_number,

                    source_locator=
                        chunk.source_locator,

                    evidence_unit_id=
                        evidence_unit_id,

                    text=
                        normalized_unit,

                    parent_context=
                        None,

                    parent_evidence_unit_id=
                        None,
                )
            )


            continue


        # ----------------------------------------------------
        # NORMAL UNIT
        # ----------------------------------------------------

        output.append(
            DocumentEvidenceUnit(
                document_id=
                    chunk.document_id,

                filename=
                    chunk.filename,

                chunk_id=
                    chunk.chunk_id,

                chunk_index=
                    chunk.chunk_index,

                page_number=
                    chunk.page_number,

                source_locator=
                    chunk.source_locator,

                evidence_unit_id=
                    evidence_unit_id,

                text=
                    normalized_unit,

                parent_context=
                    active_parent_text,

                parent_evidence_unit_id=
                    active_parent_id,
            )
        )


        if (
            active_parent_text
            is None
        ):
            continue


        # ----------------------------------------------------
        # EXPLICIT TERMINATOR
        #
        # Existing behaviour:
        #
        #   ? ...
        #   ? ...
        #   ? etc.
        #
        # ----------------------------------------------------

        if is_list_terminator(
            normalized_unit
        ):
            active_parent_text = (
                None
            )

            active_parent_id = (
                None
            )

            active_parent_expected_count = (
                None
            )

            active_parent_consumed_count = (
                0
            )


            continue


        # ----------------------------------------------------
        # DECLARED ITEM COUNT
        #
        # Example:
        #
        #   "5 corr?lations :"
        #
        # After exactly five child units, the parent context
        # expires before the next evidence unit.
        #
        # ----------------------------------------------------

        if (
            active_parent_expected_count
            is not None
        ):
            active_parent_consumed_count += (
                1
            )


            if (
                active_parent_consumed_count
                >=
                active_parent_expected_count
            ):
                active_parent_text = (
                    None
                )

                active_parent_id = (
                    None
                )

                active_parent_expected_count = (
                    None
                )

                active_parent_consumed_count = (
                    0
                )


    return output


def build_document_evidence_catalog(
    ingestion:
        DocumentIngestionReport,
) -> dict[
    str,
    list[
        DocumentEvidenceUnit
    ],
]:
    catalog: dict[
        str,
        list[
            DocumentEvidenceUnit
        ],
    ] = defaultdict(
        list
    )


    ordered_chunks = sorted(
        ingestion.chunks,
        key=lambda chunk: (
            chunk.document_id,
            chunk.chunk_index,
            chunk.chunk_id,
        ),
    )


    for chunk in ordered_chunks:
        catalog[
            chunk.document_id
        ].extend(
            evidence_units_for_chunk(
                chunk
            )
        )


    return dict(
        catalog
    )


# ============================================================
# CITATION
# ============================================================

def citation_for_unit(
    unit:
        DocumentEvidenceUnit,
) -> DocumentSummaryCitation:
    return (
        DocumentSummaryCitation(
            chunk_id=
                unit.chunk_id,

            document_id=
                unit.document_id,

            filename=
                unit.filename,

            source_locator=
                unit.source_locator,

            page_number=
                unit.page_number,
        )
    )


# ============================================================
# STRUCTURAL REQUEST CHECK
# ============================================================

def unit_is_verified_list_request(
    unit:
        DocumentEvidenceUnit,
) -> bool:
    if not unit.parent_context:
        return False


    if (
        unit.parent_evidence_unit_id
        is None
    ):
        return False


    if not looks_like_list_parent(
        unit.parent_context
    ):
        return False


    if is_list_terminator(
        unit.text
    ):
        return False


    return True


def unit_is_verified_analytical_request(
    unit:
        DocumentEvidenceUnit,
) -> bool:
    if (
        unit_is_verified_list_request(
            unit
        )
    ):
        return True


    if (
        looks_like_explicit_relation_request(
            unit.text
        )
    ):
        return True


    return False


# ============================================================
# DETERMINISTIC CLAIMS
# ============================================================

def deterministic_claim_for_unit(
    unit:
        DocumentEvidenceUnit,
) -> (
    VerifiedDocumentClaim
    | None
):
    evidence_quote = (
        normalize_whitespace(
            unit.text
        )
    )


    if not evidence_quote:
        return None


    # --------------------------------------------------------
    # LIST INTRODUCTION
    # --------------------------------------------------------

    if (
        unit.parent_context
        is None
        and
        looks_like_list_parent(
            evidence_quote
        )
    ):
        return (
            VerifiedDocumentClaim(
                category=
                    "business_objective",

                statement=
                    clean_evidence_statement(
                        evidence_quote
                    ),

                evidence_quote=
                    evidence_quote,

                evidence_unit_id=
                    unit.evidence_unit_id,

                context_quote=
                    None,

                context_evidence_unit_id=
                    None,

                citation=
                    citation_for_unit(
                        unit
                    ),
            )
        )


    # --------------------------------------------------------
    # VERIFIED LIST ITEM
    # --------------------------------------------------------

    if (
        unit_is_verified_list_request(
            unit
        )
    ):
        return (
            VerifiedDocumentClaim(
                category=
                    "analytical_request",

                statement=
                    clean_evidence_statement(
                        evidence_quote
                    ),

                evidence_quote=
                    evidence_quote,

                evidence_unit_id=
                    unit.evidence_unit_id,

                context_quote=
                    normalize_whitespace(
                        unit.parent_context
                        or ""
                    ),

                context_evidence_unit_id=
                    unit.parent_evidence_unit_id,

                citation=
                    citation_for_unit(
                        unit
                    ),
            )
        )


    # --------------------------------------------------------
    # EXPLICIT STANDALONE RELATION
    # --------------------------------------------------------

    if (
        looks_like_explicit_relation_request(
            evidence_quote
        )
    ):
        return (
            VerifiedDocumentClaim(
                category=
                    "analytical_request",

                statement=
                    clean_evidence_statement(
                        evidence_quote
                    ),

                evidence_quote=
                    evidence_quote,

                evidence_unit_id=
                    unit.evidence_unit_id,

                context_quote=
                    None,

                context_evidence_unit_id=
                    None,

                citation=
                    citation_for_unit(
                        unit
                    ),
            )
        )


    # --------------------------------------------------------
    # GENERAL OBJECTIVE
    # --------------------------------------------------------

    if (
        looks_like_general_objective(
            evidence_quote
        )
    ):
        return (
            VerifiedDocumentClaim(
                category=
                    "business_objective",

                statement=
                    clean_evidence_statement(
                        evidence_quote
                    ),

                evidence_quote=
                    evidence_quote,

                evidence_unit_id=
                    unit.evidence_unit_id,

                context_quote=
                    None,

                context_evidence_unit_id=
                    None,

                citation=
                    citation_for_unit(
                        unit
                    ),
            )
        )


    return None


def build_deterministic_claims(
    units: list[
        DocumentEvidenceUnit
    ],
) -> list[
    VerifiedDocumentClaim
]:
    output: list[
        VerifiedDocumentClaim
    ] = []


    for unit in units:
        claim = (
            deterministic_claim_for_unit(
                unit
            )
        )


        if claim is None:
            continue


        output.append(
            claim
        )


    return output


# ============================================================
# BATCHING
# ============================================================

def batch_evidence_units(
    units: list[
        DocumentEvidenceUnit
    ],
) -> list[
    list[
        DocumentEvidenceUnit
    ]
]:
    batches: list[
        list[
            DocumentEvidenceUnit
        ]
    ] = []

    current_batch: list[
        DocumentEvidenceUnit
    ] = []

    current_characters = 0


    for unit in units:
        estimated_characters = (
            len(
                unit.text
            )
            +
            len(
                unit.parent_context
                or ""
            )
            +
            len(
                unit.chunk_id
            )
            +
            100
        )


        would_exceed_units = (
            len(
                current_batch
            )
            >=
            MAX_BATCH_UNITS
        )


        would_exceed_characters = (
            bool(
                current_batch
            )
            and
            (
                current_characters
                +
                estimated_characters
                >
                MAX_BATCH_CHARACTERS
            )
        )


        if (
            would_exceed_units
            or
            would_exceed_characters
        ):
            batches.append(
                current_batch
            )

            current_batch = []

            current_characters = 0


        current_batch.append(
            unit
        )


        current_characters += (
            estimated_characters
        )


    if current_batch:
        batches.append(
            current_batch
        )


    return batches


# ============================================================
# PROMPT BUILDING
# ============================================================

def build_batch_user_prompt(
    *,
    filename: str,

    units: list[
        DocumentEvidenceUnit
    ],

    objective: (
        str
        | None
    ),
) -> str:
    lines = [
        (
            "Analyse les unités documentaires "
            "ci-dessous."
        ),
        "",
        f"DOCUMENT: {filename}",
    ]


    if objective:
        lines.extend(
            [
                "",
                (
                    "OBJECTIF UTILISATEUR "
                    "(priorisation uniquement, "
                    "ce texte n'est PAS une preuve):"
                ),
                objective,
            ]
        )


    lines.extend(
        [
            "",
            "UNITÉS DE PREUVE:",
            "",
        ]
    )


    current_chunk_id: (
        str
        | None
    ) = None


    for unit in units:
        if (
            unit.chunk_id
            !=
            current_chunk_id
        ):
            current_chunk_id = (
                unit.chunk_id
            )


            lines.extend(
                [
                    (
                        "SOURCE_CHUNK_ID: "
                        f"{unit.chunk_id}"
                    ),
                    (
                        "SOURCE: "
                        f"{unit.filename} "
                        f"· {unit.source_locator}"
                    ),
                    "",
                ]
            )


        lines.append(
            (
                f"[EVIDENCE "
                f"{unit.evidence_unit_id}]"
            )
        )


        if (
            unit.parent_context
            and
            unit.parent_evidence_unit_id
            is not None
        ):
            lines.append(
                (
                    "VERIFIED_PARENT_CONTEXT "
                    f"[EVIDENCE "
                    f"{unit.parent_evidence_unit_id}]: "
                    f"{unit.parent_context}"
                )
            )


        lines.append(
            (
                "TEXT: "
                f"{unit.text}"
            )
        )

        lines.append(
            ""
        )


    lines.extend(
        [
            (
                "Retourne uniquement les claims "
                "explicitement soutenus par "
                "les unités ci-dessus."
            ),
        ]
    )


    return "\n".join(
        lines
    )


# ============================================================
# RAW GENERATION
# ============================================================

def generate_raw_batch_extraction(
    *,
    filename: str,

    units: list[
        DocumentEvidenceUnit
    ],

    objective: (
        str
        | None
    ),

    model: str,
) -> RawDocumentBatchExtraction:
    response = (
        classified_llm_chat(
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
                        DOCUMENT_SUMMARY_SYSTEM_PROMPT,
                },
                {
                    "role":
                        "user",

                    "content":
                        build_batch_user_prompt(
                            filename=
                                filename,

                            units=
                                units,

                            objective=
                                objective,
                        ),
                },
            ],

            format=
                RawDocumentBatchExtraction
                .model_json_schema(),

            options={
                "temperature":
                    0,
            },
        )
    )


    message = (
        response.get(
            "message",
            {}
        )
    )


    content = (
        message.get(
            "content",
            ""
        )
    )


    if not isinstance(
        content,
        str,
    ):
        raise RuntimeError(
            (
                "Le modèle de résumé documentaire "
                "n'a pas retourné de contenu texte."
            )
        )


    content = (
        content.strip()
    )


    if not content:
        raise RuntimeError(
            (
                "Le modèle de résumé documentaire "
                "a retourné une réponse vide."
            )
        )


    return (
        RawDocumentBatchExtraction
        .model_validate_json(
            content
        )
    )


# ============================================================
# LLM CLAIM VALIDATION
# ============================================================

def build_verified_support_text(
    *,
    evidence_quote: str,

    context_quote: (
        str
        | None
    ),
) -> str:
    if context_quote:
        return (
            f"{context_quote} "
            f"{evidence_quote}"
        )


    return evidence_quote


def validate_llm_statement(
    *,
    statement: str,

    evidence_quote: str,

    context_quote: (
        str
        | None
    ),
) -> bool:
    normalized_statement = (
        normalize_whitespace(
            statement
        )
    )


    if not normalized_statement:
        return False


    if (
        len(
            normalized_statement
        )
        >
        MAX_STATEMENT_LENGTH
    ):
        return False


    lowered_statement = (
        normalized_statement
        .casefold()
    )


    if (
        "source_chunk_id"
        in lowered_statement
        or
        "evidence_unit_id"
        in lowered_statement
        or
        "verified_parent_context"
        in lowered_statement
    ):
        return False


    support_text = (
        build_verified_support_text(
            evidence_quote=
                evidence_quote,

            context_quote=
                context_quote,
        )
    )


    statement_numbers = (
        extract_numeric_tokens(
            normalized_statement
        )
    )


    support_numbers = (
        extract_numeric_tokens(
            support_text
        )
    )


    if not statement_numbers.issubset(
        support_numbers
    ):
        return False


    statement_tokens = (
        content_tokens(
            normalized_statement
        )
    )


    support_tokens = (
        content_tokens(
            support_text
        )
    )


    unsupported_tokens = (
        statement_tokens
        -
        support_tokens
        -
        SAFE_PARAPHRASE_TOKENS
    )


    if (
        len(
            unsupported_tokens
        )
        >
        1
    ):
        return False


    if (
        statement_tokens
        and
        support_tokens
        and
        not (
            statement_tokens
            &
            support_tokens
        )
    ):
        return False


    return True


def verify_raw_claim(
    *,
    raw_claim:
        RawDocumentClaim,

    batch_units: list[
        DocumentEvidenceUnit
    ],
) -> (
    VerifiedDocumentClaim
    | None
):
    lookup = {
        (
            unit.chunk_id,
            unit.evidence_unit_id,
        ):
            unit

        for unit
        in batch_units
    }


    key = (
        raw_claim.source_chunk_id,
        raw_claim.evidence_unit_id,
    )


    evidence_unit = (
        lookup.get(
            key
        )
    )


    if evidence_unit is None:
        return None


    evidence_quote = (
        normalize_whitespace(
            evidence_unit.text
        )
    )


    if not evidence_quote:
        return None


    if (
        len(
            evidence_quote
        )
        >
        MAX_EVIDENCE_QUOTE_LENGTH
    ):
        return None


    context_quote = (
        normalize_whitespace(
            evidence_unit.parent_context
        )
        if evidence_unit.parent_context
        else None
    )


    if (
        context_quote
        and
        len(
            context_quote
        )
        >
        MAX_CONTEXT_QUOTE_LENGTH
    ):
        return None


    category: DocumentClaimCategory = (
        raw_claim.category
    )


    verified_list_request = (
        unit_is_verified_list_request(
            evidence_unit
        )
    )


    verified_explicit_request = (
        looks_like_explicit_relation_request(
            evidence_quote
        )
    )


    # --------------------------------------------------------
    # PYTHON OVERRIDES STRUCTURAL CLASSIFICATION
    # --------------------------------------------------------

    if verified_list_request:
        category = (
            "analytical_request"
        )


    elif (
        looks_like_list_parent(
            evidence_quote
        )
    ):
        category = (
            "business_objective"
        )


    elif verified_explicit_request:
        category = (
            "analytical_request"
        )


    elif (
        looks_like_general_objective(
            evidence_quote
        )
    ):
        category = (
            "business_objective"
        )


    # --------------------------------------------------------
    # FINAL PYTHON GATE FOR ANALYTICAL REQUESTS
    # --------------------------------------------------------

    if (
        category
        ==
        "analytical_request"
        and
        not (
            verified_list_request
            or
            verified_explicit_request
        )
    ):
        return None


    # --------------------------------------------------------
    # ANALYTICAL REQUESTS NEVER USE FREE LLM WORDING
    # --------------------------------------------------------

    if (
        category
        ==
        "analytical_request"
    ):
        statement = (
            clean_evidence_statement(
                evidence_quote
            )
        )


    else:
        statement = (
            normalize_whitespace(
                raw_claim.statement
            )
        )


        if not validate_llm_statement(
            statement=
                statement,

            evidence_quote=
                evidence_quote,

            context_quote=
                context_quote,
        ):
            return None


    return (
        VerifiedDocumentClaim(
            category=
                category,

            statement=
                statement,

            evidence_quote=
                evidence_quote,

            evidence_unit_id=
                evidence_unit
                .evidence_unit_id,

            context_quote=
                context_quote,

            context_evidence_unit_id=
                evidence_unit
                .parent_evidence_unit_id,

            citation=
                citation_for_unit(
                    evidence_unit
                ),
        )
    )


# ============================================================
# DEDUPLICATION
# ============================================================

def verified_claim_key(
    claim:
        VerifiedDocumentClaim,
) -> tuple[
    str,
    str,
    int,
]:
    return (
        claim.category,
        claim.citation.chunk_id,
        claim.evidence_unit_id,
    )


def deduplicate_verified_claims(
    claims: list[
        VerifiedDocumentClaim
    ],
) -> list[
    VerifiedDocumentClaim
]:
    seen: set[
        tuple[
            str,
            str,
            int,
        ]
    ] = set()


    output: list[
        VerifiedDocumentClaim
    ] = []


    for claim in claims:
        key = (
            verified_claim_key(
                claim
            )
        )


        if key in seen:
            continue


        seen.add(
            key
        )


        output.append(
            claim
        )


    return output


# ============================================================
# SUMMARY PRIORITY
# ============================================================

CATEGORY_PRIORITY = {
    "business_context":
        70,

    "business_objective":
        80,

    "definition":
        55,

    "business_rule":
        65,

    "constraint":
        60,

    "methodology":
        50,

    "analytical_request":
        90,
}


def claim_sort_key(
    claim:
        VerifiedDocumentClaim,
) -> tuple[
    int,
    str,
    str,
    int,
]:
    return (
        -CATEGORY_PRIORITY[
            claim.category
        ],

        claim
        .citation
        .filename
        .casefold(),

        claim.citation.chunk_id,

        claim.evidence_unit_id,
    )


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_verified_claims_for_document(
    *,
    filename: str,

    units: list[
        DocumentEvidenceUnit
    ],

    objective: (
        str
        | None
    ),

    model: str,
) -> list[
    VerifiedDocumentClaim
]:
    verified_claims: list[
        VerifiedDocumentClaim
    ] = []


    # --------------------------------------------------------
    # PYTHON-FIRST STRUCTURAL EXTRACTION
    # --------------------------------------------------------

    verified_claims.extend(
        build_deterministic_claims(
            units
        )
    )


    # --------------------------------------------------------
    # LLM SUPPLEMENT
    # --------------------------------------------------------

    batches = (
        batch_evidence_units(
            units
        )
    )


    for batch in batches:
        raw_extraction = (
            generate_raw_batch_extraction(
                filename=
                    filename,

                units=
                    batch,

                objective=
                    objective,

                model=
                    model,
            )
        )


        for raw_claim in (
            raw_extraction.claims
        ):
            verified_claim = (
                verify_raw_claim(
                    raw_claim=
                        raw_claim,

                    batch_units=
                        batch,
                )
            )


            if verified_claim is None:
                continue


            verified_claims.append(
                verified_claim
            )


    verified_claims = (
        deduplicate_verified_claims(
            verified_claims
        )
    )


    verified_claims.sort(
        key=
            claim_sort_key
    )


    return verified_claims


# ============================================================
# DOCUMENT SUMMARY BUILDING
# ============================================================

def build_per_document_summary(
    *,
    document_id: str,

    filename: str,

    source_chunk_count: int,

    claims: list[
        VerifiedDocumentClaim
    ],
) -> PerDocumentSummary:
    analytical_requests = [
        claim
        for claim
        in claims
        if (
            claim.category
            ==
            "analytical_request"
        )
    ]


    summary_points = [
        claim
        for claim
        in claims
        if (
            claim.category
            !=
            "analytical_request"
        )
    ]


    return (
        PerDocumentSummary(
            document_id=
                document_id,

            filename=
                filename,

            summary_points=
                summary_points[
                    :
                    MAX_SUMMARY_POINTS_PER_DOCUMENT
                ],

            analytical_requests=
                analytical_requests[
                    :
                    MAX_ANALYTICAL_REQUESTS_PER_DOCUMENT
                ],

            verified_claim_count=
                len(
                    claims
                ),

            source_chunk_count=
                source_chunk_count,
        )
    )


# ============================================================
# GLOBAL SUMMARY
# ============================================================

def summarize_document_ingestion(
    *,
    ingestion:
        DocumentIngestionReport,

    objective: (
        str
        | None
    ) = None,

    model: str = (
        DEFAULT_DOCUMENT_SUMMARY_MODEL
    ),
) -> DocumentSummaryReport:
    if not ingestion.chunks:
        return (
            DocumentSummaryReport(
                status=
                    "abstained",

                document_count=
                    ingestion.document_count,

                chunk_count=
                    ingestion.chunk_count,

                verified_claim_count=
                    0,

                summary_point_count=
                    0,

                analytical_request_count=
                    0,

                summary_points=[],

                analytical_requests=[],

                documents=[],

                warnings=
                    list(
                        ingestion.warnings
                    ),

                abstention_reason=(
                    "Aucun passage documentaire "
                    "n'est disponible."
                ),

                model=
                    model,
            )
        )


    normalized_objective = (
        normalize_whitespace(
            objective
        )
        if objective
        else None
    )


    evidence_catalog = (
        build_document_evidence_catalog(
            ingestion
        )
    )


    valid_document_ids = {
        document.document_id
        for document
        in ingestion.documents
    }


    chunk_counts: dict[
        str,
        int
    ] = defaultdict(
        int
    )


    for chunk in ingestion.chunks:
        chunk_counts[
            chunk.document_id
        ] += 1


    all_verified_claims: list[
        VerifiedDocumentClaim
    ] = []


    document_summaries: list[
        PerDocumentSummary
    ] = []


    warnings = list(
        ingestion.warnings
    )


    for document in ingestion.documents:
        units = (
            evidence_catalog.get(
                document.document_id,
                [],
            )
        )


        if not units:
            warnings.append(
                (
                    "Aucune unité documentaire "
                    f"exploitable pour "
                    f"{document.filename}."
                )
            )


            document_summaries.append(
                PerDocumentSummary(
                    document_id=
                        document.document_id,

                    filename=
                        document.filename,

                    summary_points=[],

                    analytical_requests=[],

                    verified_claim_count=
                        0,

                    source_chunk_count=
                        chunk_counts.get(
                            document.document_id,
                            0,
                        ),
                )
            )


            continue


        claims = (
            extract_verified_claims_for_document(
                filename=
                    document.filename,

                units=
                    units,

                objective=
                    normalized_objective,

                model=
                    model,
            )
        )


        claims = [
            claim
            for claim
            in claims
            if (
                claim
                .citation
                .document_id
                in
                valid_document_ids
            )
        ]


        all_verified_claims.extend(
            claims
        )


        document_summaries.append(
            build_per_document_summary(
                document_id=
                    document.document_id,

                filename=
                    document.filename,

                source_chunk_count=
                    chunk_counts.get(
                        document.document_id,
                        0,
                    ),

                claims=
                    claims,
            )
        )


    all_verified_claims = (
        deduplicate_verified_claims(
            all_verified_claims
        )
    )


    all_verified_claims.sort(
        key=
            claim_sort_key
    )


    analytical_requests = [
        claim
        for claim
        in all_verified_claims
        if (
            claim.category
            ==
            "analytical_request"
        )
    ]


    summary_points = [
        claim
        for claim
        in all_verified_claims
        if (
            claim.category
            !=
            "analytical_request"
        )
    ]


    analytical_requests = (
        analytical_requests[
            :
            MAX_GLOBAL_ANALYTICAL_REQUESTS
        ]
    )


    summary_points = (
        summary_points[
            :
            MAX_GLOBAL_SUMMARY_POINTS
        ]
    )


    if not all_verified_claims:
        return (
            DocumentSummaryReport(
                status=
                    "abstained",

                document_count=
                    ingestion.document_count,

                chunk_count=
                    ingestion.chunk_count,

                verified_claim_count=
                    0,

                summary_point_count=
                    0,

                analytical_request_count=
                    0,

                summary_points=[],

                analytical_requests=[],

                documents=
                    document_summaries,

                warnings=
                    warnings,

                abstention_reason=(
                    "Aucune information documentaire "
                    "suffisamment explicite n'a pu "
                    "être vérifiée."
                ),

                model=
                    model,
            )
        )


    return (
        DocumentSummaryReport(
            status=
                "ready",

            document_count=
                ingestion.document_count,

            chunk_count=
                ingestion.chunk_count,

            verified_claim_count=
                len(
                    all_verified_claims
                ),

            summary_point_count=
                len(
                    summary_points
                ),

            analytical_request_count=
                len(
                    analytical_requests
                ),

            summary_points=
                summary_points,

            analytical_requests=
                analytical_requests,

            documents=
                document_summaries,

            warnings=
                warnings,

            abstention_reason=
                None,

            model=
                model,
        )
    )