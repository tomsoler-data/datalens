from __future__ import annotations


import re


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

from app.rag_relevance import (
    build_evidence_units,
    filter_candidate_evidence_units,
)

from app.rag_retrieval import (
    RagSearchHit,
)


# ============================================================
# VERSION
# ============================================================

RAG_EXPLANATION_RULE_VERSION = (
    "rag_explanation_v0.3"
)


# ============================================================
# MODEL
# ============================================================

DEFAULT_EXPLANATION_MODEL = (
    "gemma3:4b"
)


# ============================================================
# LIMITS
# ============================================================

MAX_CLAIMS = 3

MAX_STATEMENT_LENGTH = 420

MAX_EVIDENCE_QUOTE_LENGTH = 420


# ============================================================
# RAW LLM OUTPUT
# ============================================================

class RawGroundedClaim(
    BaseModel
):
    statement: str

    source_chunk_id: str

    evidence_unit_id: int


class RawRagExplanation(
    BaseModel
):
    claims: list[
        RawGroundedClaim
    ] = Field(
        min_length=1,
        max_length=MAX_CLAIMS,
    )


# ============================================================
# VERIFIED OUTPUT
# ============================================================

class RagCitation(
    BaseModel
):
    chunk_id: str

    filename: str

    source_locator: str

    page_number: (
        int
        | None
    ) = None


class VerifiedGroundedClaim(
    BaseModel
):
    statement: str

    evidence_quote: str

    citation: RagCitation


class VerifiedRagExplanation(
    BaseModel
):
    status: Literal[
        "ready",
        "abstained",
    ]

    explanation: str

    claims: list[
        VerifiedGroundedClaim
    ]

    abstention_reason: (
        str
        | None
    ) = None

    explanation_rule_version: str = (
        RAG_EXPLANATION_RULE_VERSION
    )

    model: str


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
Tu es le composant d'explication documentaire de DataLens.

Les passages documentaires ont déjà passé le relevance gate.

Python a ensuite extrait uniquement les UNITÉS DE PREUVE
compatibles avec le contrat analytique.

Ta tâche est limitée à produire entre un et trois claims
documentaires fidèles à ces unités.

============================================================
PRINCIPE FONDAMENTAL
============================================================

Tu ne crées JAMAIS la preuve documentaire.

Pour chaque claim :

- source_chunk_id doit être exactement un identifiant
  SOURCE_CHUNK_ID fourni ;

- evidence_unit_id doit être exactement le numéro d'une
  UNITÉ fournie sous ce SOURCE_CHUNK_ID ;

- statement doit être une reformulation concise et fidèle
  de CETTE unité.

Python reconstruira lui-même la citation exacte.

============================================================
UNE SEULE UNITÉ PAR CLAIM
============================================================

Chaque claim doit être soutenu par UNE SEULE unité.

Il est interdit :

- de fusionner plusieurs unités ;
- de prendre une variable dans une unité et une autre
  variable dans une autre ;
- de compléter l'unité sélectionnée avec des connaissances
  générales ;
- d'utiliser une information absente de l'unité choisie.

============================================================
RÔLE DU STATEMENT
============================================================

Le statement explique seulement ce que la documentation
apporte au finding.

Il ne doit PAS :

- recalculer le résultat statistique ;
- modifier le résultat Python ;
- inventer une causalité ;
- inventer une recommandation ;
- inventer une définition ;
- transformer une variable en une autre.

============================================================
DISTINCTIONS IMPORTANTES
============================================================

Respecte notamment les distinctions suivantes :

- prix != chiffre d'affaires ;
- prix != panier moyen ;
- montant total != panier moyen ;
- moyenne != médiane ;
- fréquence != comptage d'événements ;
- association != évolution temporelle.

============================================================
NOMBRES
============================================================

Tu peux utiliser une valeur numérique dans statement
UNIQUEMENT si cette même valeur apparaît dans l'unité
documentaire sélectionnée.

N'invente jamais de nombre.

============================================================
IDENTIFIANTS TECHNIQUES
============================================================

Ne mets jamais dans statement :

- SOURCE_CHUNK_ID ;
- evidence_unit_id ;
- un identifiant interne DataLens.

============================================================
STYLE
============================================================

Le statement doit :

- être court ;
- être factuel ;
- être compréhensible ;
- rester strictement documentaire ;
- ne pas répéter inutilement le finding.

============================================================
EXEMPLE
============================================================

SOURCE_CHUNK_ID:
document_abc:chunk_001

[UNITÉ 17]
le lien entre l'âge des clients et la taille du panier moyen

Réponse possible :

{
  "claims": [
    {
      "statement":
        "La documentation demande d'étudier le lien entre l'âge des clients et leur panier moyen.",
      "source_chunk_id":
        "document_abc:chunk_001",
      "evidence_unit_id":
        17
    }
  ]
}

============================================================
SÉCURITÉ
============================================================

Si plusieurs unités sont disponibles, utilise uniquement
celles qui apportent réellement un contexte utile.

Ne crée jamais un claim dans le seul but d'atteindre le
maximum de trois claims.

Retourne uniquement la structure JSON demandée.
""".strip()


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_whitespace(
    value: str,
) -> str:
    return " ".join(
        value
        .strip()
        .split()
    )


# ============================================================
# NUMERIC TOKENS
# ============================================================

NUMERIC_TOKEN_PATTERN = re.compile(
    (
        r"(?<![A-Za-zÀ-ÖØ-öø-ÿ0-9_])"
        r"[+-]?"
        r"\d+"
        r"(?:[.,]\d+)?"
        r"%?"
        r"(?![A-Za-zÀ-ÖØ-öø-ÿ0-9_])"
    )
)


def normalize_numeric_token(
    value: str,
) -> str:
    return (
        value
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
            match
        )

        for match
        in NUMERIC_TOKEN_PATTERN.findall(
            value
        )
    }


# ============================================================
# CONTENT TOKEN SUPPORT
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
    "en",
    "et",
    "est",
    "la",
    "le",
    "les",
    "leur",
    "leurs",
    "lien",
    "pour",
    "que",
    "qui",
    "sur",
    "un",
    "une",
    "the",
    "of",
    "and",
    "to",
    "in",
    "for",
    "with",
}


def content_tokens(
    value: str,
) -> set[
    str
]:
    normalized = (
        value
        .casefold()
    )


    tokens = set(
        re.findall(
            r"[a-zà-öø-ÿ0-9_]+",
            normalized,
        )
    )


    return {
        token

        for token
        in tokens

        if (
            token
            not in
            STOPWORDS
            and
            len(
                token
            )
            >=
            3
        )
    }


# ============================================================
# EVIDENCE CANDIDATES
# ============================================================

def build_candidate_evidence_map(
    *,
    finding_text: str,
    accepted_hits: list[
        RagSearchHit
    ],
) -> dict[
    str,
    dict[
        int,
        str,
    ],
]:
    result: dict[
        str,
        dict[
            int,
            str,
        ],
    ] = {}


    for hit in accepted_hits:
        units = (
            build_evidence_units(
                hit.text
            )
        )


        candidates = (
            filter_candidate_evidence_units(
                finding=
                    finding_text,

                evidence_units=
                    units,
            )
        )


        if not candidates:
            continue


        result[
            hit.chunk_id
        ] = {
            unit_id:
                unit

            for (
                unit_id,
                unit,
            )
            in candidates
        }


    return result


# ============================================================
# PROMPT CONSTRUCTION
# ============================================================

def build_user_prompt(
    *,
    finding_text: str,
    accepted_hits: list[
        RagSearchHit
    ],
    candidate_evidence_map: dict[
        str,
        dict[
            int,
            str,
        ],
    ],
) -> str:
    finding_text = (
        finding_text
        .strip()
    )


    if not finding_text:
        raise ValueError(
            "Le finding ne peut pas être vide."
        )


    if not accepted_hits:
        raise ValueError(
            (
                "Au moins un passage documentaire "
                "accepté doit être fourni."
            )
        )


    if not candidate_evidence_map:
        raise ValueError(
            (
                "Aucune unité documentaire compatible "
                "avec le contrat analytique n'est "
                "disponible."
            )
        )


    hit_lookup = {
        hit.chunk_id:
            hit

        for hit
        in accepted_hits
    }


    source_blocks: list[
        str
    ] = []


    for (
        chunk_id,
        units,
    ) in candidate_evidence_map.items():
        hit = (
            hit_lookup.get(
                chunk_id
            )
        )


        if hit is None:
            continue


        rendered_units = "\n\n".join(
            (
                f"[UNITÉ {unit_id}]\n"
                f"{unit}"
            )

            for (
                unit_id,
                unit,
            )
            in units.items()
        )


        source_blocks.append(
            (
                "SOURCE_CHUNK_ID: "
                f"{chunk_id}\n\n"
                f"{rendered_units}"
            )
        )


    sources_text = (
        "\n\n"
        "========================================\n\n"
        .join(
            source_blocks
        )
    )


    if not sources_text.strip():
        raise ValueError(
            (
                "Aucune source documentaire "
                "candidate n'a pu être préparée."
            )
        )


    return (
        "CONTRAT / FINDING ANALYTIQUE:\n"
        f"{finding_text}\n\n"
        "PREUVES DOCUMENTAIRES VALIDÉES PAR PYTHON:\n\n"
        f"{sources_text}\n\n"
        "Produis uniquement des claims directement soutenus "
        "par UNE unité affichée.\n\n"
        "Pour chaque claim, retourne le SOURCE_CHUNK_ID "
        "exact et le numéro exact de l'UNITÉ."
    )


# ============================================================
# RAW GENERATION
# ============================================================

def generate_raw_explanation(
    *,
    finding_text: str,
    accepted_hits: list[
        RagSearchHit
    ],
    candidate_evidence_map: dict[
        str,
        dict[
            int,
            str,
        ],
    ],
    model: str = (
        DEFAULT_EXPLANATION_MODEL
    ),
) -> RawRagExplanation:
    prompt = (
        build_user_prompt(
            finding_text=
                finding_text,

            accepted_hits=
                accepted_hits,

            candidate_evidence_map=
                candidate_evidence_map,
        )
    )


    try:
        response = client.chat(
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
                        prompt,
                },
            ],

            format=(
                RawRagExplanation
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
                "La génération des claims "
                "documentaires a échoué."
            )
        ) from error


    content = (
        response
        .message
        .content
    )


    try:
        return (
            RawRagExplanation
            .model_validate_json(
                content
            )
        )


    except Exception as error:
        raise RuntimeError(
            (
                "Gemma a retourné une réponse "
                "qui ne respecte pas le schéma "
                "d'explication attendu."
            )
        ) from error


# ============================================================
# STATEMENT VALIDATION
# ============================================================

def validate_claim_statement(
    *,
    statement: str,
    evidence_quote: str,
) -> (
    str
    | None
):
    normalized_statement = (
        normalize_whitespace(
            statement
        )
    )


    if not normalized_statement:
        return None


    if (
        len(
            normalized_statement
        )
        >
        MAX_STATEMENT_LENGTH
    ):
        return None


    if (
        "source_chunk_id"
        in
        normalized_statement
        .casefold()
    ):
        return None


    if (
        "evidence_unit_id"
        in
        normalized_statement
        .casefold()
    ):
        return None


    statement_numbers = (
        extract_numeric_tokens(
            normalized_statement
        )
    )


    evidence_numbers = (
        extract_numeric_tokens(
            evidence_quote
        )
    )


    if not (
        statement_numbers
        <=
        evidence_numbers
    ):
        return None


    statement_tokens = (
        content_tokens(
            normalized_statement
        )
    )


    evidence_tokens = (
        content_tokens(
            evidence_quote
        )
    )


    if (
        statement_tokens
        and
        evidence_tokens
        and
        not (
            statement_tokens
            &
            evidence_tokens
        )
    ):
        return None


    return normalized_statement


# ============================================================
# VERIFIED EXPLANATION CONSTRUCTION
# ============================================================

def build_verified_explanation(
    *,
    claims: list[
        VerifiedGroundedClaim
    ],
) -> str:
    if not claims:
        raise ValueError(
            (
                "Aucun claim documentaire "
                "vérifié n'est disponible."
            )
        )


    return " ".join(
        claim.statement

        for claim
        in claims
    )


# ============================================================
# ABSTENTION BUILDER
# ============================================================

def build_abstained_explanation(
    *,
    model: str,
    reason: str,
) -> VerifiedRagExplanation:
    return (
        VerifiedRagExplanation(
            status=
                "abstained",

            explanation=
                "",

            claims=[],

            abstention_reason=
                reason,

            model=
                model,
        )
    )


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def validate_and_build_explanation(
    *,
    raw: RawRagExplanation,
    accepted_hits: list[
        RagSearchHit
    ],
    candidate_evidence_map: dict[
        str,
        dict[
            int,
            str,
        ],
    ],
    model: str,
) -> VerifiedRagExplanation:
    hit_lookup = {
        hit.chunk_id:
            hit

        for hit
        in accepted_hits
    }


    verified_claims: list[
        VerifiedGroundedClaim
    ] = []


    seen_evidence: set[
        tuple[
            str,
            int,
        ]
    ] = set()


    for claim in raw.claims:
        source_chunk_id = (
            claim
            .source_chunk_id
            .strip()
        )


        if not source_chunk_id:
            continue


        source_hit = (
            hit_lookup.get(
                source_chunk_id
            )
        )


        if source_hit is None:
            continue


        chunk_candidates = (
            candidate_evidence_map.get(
                source_chunk_id
            )
        )


        if not chunk_candidates:
            continue


        evidence_quote = (
            chunk_candidates.get(
                claim.evidence_unit_id
            )
        )


        if evidence_quote is None:
            continue


        evidence_quote = (
            normalize_whitespace(
                evidence_quote
            )
        )


        if not evidence_quote:
            continue


        if (
            len(
                evidence_quote
            )
            >
            MAX_EVIDENCE_QUOTE_LENGTH
        ):
            continue


        evidence_key = (
            source_chunk_id,
            claim.evidence_unit_id,
        )


        if (
            evidence_key
            in
            seen_evidence
        ):
            continue


        statement = (
            validate_claim_statement(
                statement=
                    claim.statement,

                evidence_quote=
                    evidence_quote,
            )
        )


        if statement is None:
            continue


        citation = (
            RagCitation(
                chunk_id=
                    source_hit.chunk_id,

                filename=
                    source_hit.filename,

                source_locator=
                    source_hit.source_locator,

                page_number=
                    source_hit.page_number,
            )
        )


        verified_claims.append(
            VerifiedGroundedClaim(
                statement=
                    statement,

                evidence_quote=
                    evidence_quote,

                citation=
                    citation,
            )
        )


        seen_evidence.add(
            evidence_key
        )


        if (
            len(
                verified_claims
            )
            >=
            MAX_CLAIMS
        ):
            break


    if not verified_claims:
        return (
            build_abstained_explanation(
                model=
                    model,

                reason=(
                    "Aucun claim généré n'a pu être "
                    "validé contre une preuve documentaire "
                    "locale."
                ),
            )
        )


    explanation = (
        build_verified_explanation(
            claims=
                verified_claims,
        )
    )


    return (
        VerifiedRagExplanation(
            status=
                "ready",

            explanation=
                explanation,

            claims=
                verified_claims,

            abstention_reason=
                None,

            model=
                model,
        )
    )


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def generate_grounded_explanation(
    *,
    finding_text: str,
    accepted_hits: list[
        RagSearchHit
    ],
    model: str = (
        DEFAULT_EXPLANATION_MODEL
    ),
) -> VerifiedRagExplanation:
    finding_text = (
        finding_text
        .strip()
    )


    if not finding_text:
        raise ValueError(
            "Le finding ne peut pas être vide."
        )


    if not accepted_hits:
        return (
            build_abstained_explanation(
                model=
                    model,

                reason=(
                    "Aucune source documentaire validée "
                    "n'est disponible pour cette analyse."
                ),
            )
        )


    candidate_evidence_map = (
        build_candidate_evidence_map(
            finding_text=
                finding_text,

            accepted_hits=
                accepted_hits,
        )
    )


    if not candidate_evidence_map:
        return (
            build_abstained_explanation(
                model=
                    model,

                reason=(
                    "Aucune unité documentaire compatible "
                    "avec le contrat analytique n'est "
                    "disponible pour l'explication."
                ),
            )
        )


    raw = (
        generate_raw_explanation(
            finding_text=
                finding_text,

            accepted_hits=
                accepted_hits,

            candidate_evidence_map=
                candidate_evidence_map,

            model=
                model,
        )
    )


    return (
        validate_and_build_explanation(
            raw=
                raw,

            accepted_hits=
                accepted_hits,

            candidate_evidence_map=
                candidate_evidence_map,

            model=
                model,
        )
    )