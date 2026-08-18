from __future__ import annotations

import re
import unicodedata

from typing import (
    Iterable,
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.preparation.contracts import (
    DecisionStatus,
    PreparationDecision,
    PreparationPlan,
)

from app.rag import (
    DocumentIngestionReport,
)

from app.rag_relevance import (
    build_evidence_units,
)

from app.rag_retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    RagSearchHit,
    search_document_chunks,
)


# ============================================================
# VERSION
# ============================================================


PREPARATION_RAG_CONTEXT_RULE_VERSION = (
    "preparation_rag_context_v0.2"
)


# ============================================================
# TYPES
# ============================================================


PreparationRagStatus = Literal[
    "evidence_found",
    "abstained",
]


EvidenceType = Literal[
    "direct_rule",
    "supporting_context",
    "guardrail",
]


# ============================================================
# MODELS
# ============================================================


class PreparationRagEvidence(
    BaseModel
):
    """
    Preuve documentaire validée par Python.

    Le texte provient directement de la documentation
    ingérée par le RAG.

    Aucun contenu n'est généré.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    chunk_id: str

    document_id: str

    filename: str

    extension: str

    chunk_index: int

    page_number: (
        int
        | None
    ) = None

    source_locator: str

    retrieval_rank: int

    retrieval_score: float

    evidence_text: str

    matched_column: str

    matched_keywords: list[
        str
    ] = Field(
        default_factory=list
    )

    evidence_type: EvidenceType

    deterministic_score: int = Field(
        ge=0
    )

    final_score: int = Field(
        ge=0
    )


class PreparationRagContext(
    BaseModel
):
    """
    Contexte documentaire associé à une décision
    de préparation.

    Aucune action de nettoyage n'est exécutée ici.
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

    original_status: DecisionStatus

    query: str

    status: PreparationRagStatus

    retrieved_hit_count: int = Field(
        ge=0
    )

    accepted_evidence_count: int = Field(
        ge=0
    )

    evidence: list[
        PreparationRagEvidence
    ] = Field(
        default_factory=list
    )

    abstention_reason: (
        str
        | None
    ) = None

    needs_context_after_rag: bool = True

    human_validation_required: bool = True


class PreparationRagContextReport(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    status: Literal[
        "ready"
    ] = "ready"

    document_count: int = Field(
        ge=0
    )

    chunk_count: int = Field(
        ge=0
    )

    eligible_decision_count: int = Field(
        ge=0
    )

    evidence_found_count: int = Field(
        ge=0
    )

    abstained_count: int = Field(
        ge=0
    )

    total_accepted_evidence_count: int = Field(
        ge=0
    )

    direct_rule_count: int = Field(
        ge=0
    )

    supporting_context_count: int = Field(
        ge=0
    )

    guardrail_count: int = Field(
        ge=0
    )

    contexts: list[
        PreparationRagContext
    ] = Field(
        default_factory=list
    )

    notes: list[
        str
    ] = Field(
        default_factory=list
    )

    embedding_model: str

    rule_version: str = (
        PREPARATION_RAG_CONTEXT_RULE_VERSION
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
        r"[^a-z0-9%]+",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


def normalized_column_tokens(
    column: str,
) -> list[str]:
    normalized = normalize_text(
        column
    )

    return [
        token
        for token in normalized.split()
        if token
    ]


# ============================================================
# COLUMN MATCHING
# ============================================================


def evidence_mentions_column(
    *,
    evidence_text: str,
    column: str,
) -> bool:
    """
    Une preuve doit mentionner explicitement
    la colonne concernée.

    Exemples équivalents après normalisation :

        discount_rate
        `discount_rate`
        discount rate
    """

    normalized_evidence = normalize_text(
        evidence_text
    )

    tokens = normalized_column_tokens(
        column
    )

    if not tokens:
        return False

    normalized_column = " ".join(
        tokens
    )

    padded_evidence = (
        f" {normalized_evidence} "
    )

    padded_column = (
        f" {normalized_column} "
    )

    return (
        padded_column
        in
        padded_evidence
    )


# ============================================================
# EXPLICIT NON-EVIDENCE / DISCLAIMERS
# ============================================================


NON_EVIDENCE_PATTERNS = [
    "ne donne aucune information",
    "ne fournit aucune information",
    "aucune information concernant",
    "aucune information sur",
    "sans rapport avec",
    "ne concerne pas",
    "n est pas applicable a",
    "n est pas applicable au",
    "n est pas applicable aux",
]


def is_explicit_non_evidence(
    evidence_text: str,
) -> bool:
    """
    Rejette les phrases indiquant explicitement
    qu'elles ne fournissent pas d'information.

    Important :
    on ne rejette PAS des règles négatives utiles
    telles que :

        "ne doit pas être remplacée"
        "ne signifie pas que..."
    """

    normalized = normalize_text(
        evidence_text
    )

    return any(
        normalize_text(
            pattern
        )
        in
        normalized

        for pattern in (
            NON_EVIDENCE_PATTERNS
        )
    )


# ============================================================
# ISSUE-SPECIFIC KEYWORDS
# ============================================================


ISSUE_KEYWORDS: dict[
    str,
    list[str],
] = {
    "missing_values": [
        "vide",
        "vides",
        "manquante",
        "manquantes",
        "absence",
        "absente",
        "absentes",
        "non disponible",
        "non renseigne",
        "aucune",
        "aucun",
        "laisse vide",
        "laisse volontairement vide",
    ],

    "missing_identifier": [
        "identifiant",
        "unique",
        "vide",
        "manquante",
        "probleme",
        "generation",
        "import",
    ],

    "numeric_outliers": [
        "superieure",
        "superieures",
        "inferieure",
        "inferieures",
        "borne",
        "bornes",
        "plausible",
        "plausibles",
        "invalide",
        "invalides",
        "verifiee",
        "verifier",
        "annees",
        "maximum",
        "minimum",
    ],

    "invalid_numeric_values": [
        "numerique",
        "format",
        "convertible",
        "conversion",
        "invalide",
        "invalides",
        "verifier",
    ],

    "invalid_dates": [
        "date",
        "format",
        "yyyy",
        "convertible",
        "invalide",
        "invalides",
        "source",
        "verifier",
    ],

    "possible_semantic_aliases": [
        "categorie",
        "categories",
        "fusion",
        "fusionner",
        "normalisation",
        "signification",
        "distincte",
        "distinctes",
    ],

    "duplicate_rows": [
        "doublon",
        "doublons",
        "unique",
        "commande",
        "evenement",
        "ligne",
        "lignes",
    ],
}


# ============================================================
# QUERY
# ============================================================


def issue_context_instruction(
    issue_kind: str,
) -> str:
    mapping = {
        "missing_values":
            (
                "Chercher la définition métier de la colonne "
                "et surtout la signification documentée "
                "d'une valeur vide, absente ou manquante."
            ),

        "missing_identifier":
            (
                "Chercher la définition de l'identifiant, "
                "son caractère obligatoire et la signification "
                "d'un identifiant absent."
            ),

        "numeric_outliers":
            (
                "Chercher la définition métier, l'unité, "
                "les bornes plausibles et toute règle "
                "concernant les valeurs atypiques."
            ),

        "invalid_numeric_values":
            (
                "Chercher le format numérique attendu "
                "et les règles concernant les valeurs "
                "non convertibles."
            ),

        "invalid_dates":
            (
                "Chercher le format de date attendu "
                "et les règles concernant les dates "
                "invalides ou non convertibles."
            ),

        "possible_semantic_aliases":
            (
                "Chercher les définitions des catégories "
                "et les règles indiquant si des valeurs "
                "peuvent ou non être fusionnées."
            ),

        "duplicate_rows":
            (
                "Chercher la granularité métier d'une ligne, "
                "les identifiants et les règles concernant "
                "les doublons."
            ),
    }

    return mapping.get(
        issue_kind,
        (
            "Chercher une définition, une règle métier "
            "ou une convention directement applicable "
            "à cette colonne et à ce problème de qualité."
        ),
    )


def build_preparation_retrieval_query(
    *,
    decision: PreparationDecision,
) -> str:
    if decision.column is None:
        raise ValueError(
            (
                "Une décision sans colonne ne peut pas "
                "être interrogée par Preparation RAG."
            )
        )

    parts = [
        (
            "Contexte métier pour la préparation "
            "des données."
        ),
        (
            f"Dataset : "
            f"{decision.dataset_filename}"
        ),
        (
            f"Colonne exacte : "
            f"{decision.column}"
        ),
        (
            f"Problème de qualité : "
            f"{decision.source_issue_kind}"
        ),
        issue_context_instruction(
            decision.source_issue_kind
        ),
    ]

    if decision.context_required:
        parts.append(
            (
                "Informations encore nécessaires : "
                +
                "; ".join(
                    decision.context_required
                )
            )
        )

    return "\n".join(
        parts
    ).strip()


# ============================================================
# KEYWORD MATCHING
# ============================================================


def matched_issue_keywords(
    *,
    evidence_text: str,
    issue_kind: str,
) -> list[str]:
    normalized_evidence = normalize_text(
        evidence_text
    )

    keywords = ISSUE_KEYWORDS.get(
        issue_kind,
        [],
    )

    matches: list[str] = []

    for keyword in keywords:
        normalized_keyword = normalize_text(
            keyword
        )

        if not normalized_keyword:
            continue

        if (
            normalized_keyword
            in
            normalized_evidence
        ):
            matches.append(
                keyword
            )

    return matches


# ============================================================
# EVIDENCE TYPE
# ============================================================


GUARDRAIL_MARKERS = [
    "ne doit pas",
    "ne doivent pas",
    "ne signifie pas",
    "ne signifie donc pas",
    "uniquement",
    "ne peut pas",
    "ne peuvent pas",
    "doit preserver",
    "doivent preserver",
]


DIRECT_RULE_MARKERS = [
    "doit donc etre interpretee",
    "doit donc etre interprete",
    "doit etre interpretee",
    "doit etre interprete",
    "signifie que",
    "signifie",
    "sont considerees comme",
    "sont consideres comme",
    "est consideree comme",
    "est considere comme",
    "doit etre verifiee",
    "doit etre verifie",
    "doivent etre verifiees",
    "doivent etre verifies",
    "format attendu",
    "doit posseder",
    "n est pas une valeur metier valide",
]


SUPPORTING_CONTEXT_MARKERS = [
    "represente",
    "correspond a",
    "lorsqu",
    "volontairement laisse vide",
    "volontairement laissee vide",
    "est exprime",
    "est exprimee",
    "les exemples",
]


def classify_evidence_type(
    evidence_text: str,
) -> EvidenceType:
    """
    Classe la nature de la preuve documentaire.

    Ordre important :

    1. guardrail ;
    2. règle directe ;
    3. contexte complémentaire.

    Cela permet à :

        "ne signifie pas que..."

    d'être classé comme garde-fou plutôt que
    comme règle positive à cause du mot "signifie".
    """

    normalized = normalize_text(
        evidence_text
    )

    if any(
        normalize_text(
            marker
        )
        in
        normalized

        for marker in (
            GUARDRAIL_MARKERS
        )
    ):
        return "guardrail"

    if any(
        normalize_text(
            marker
        )
        in
        normalized

        for marker in (
            DIRECT_RULE_MARKERS
        )
    ):
        return "direct_rule"

    if any(
        normalize_text(
            marker
        )
        in
        normalized

        for marker in (
            SUPPORTING_CONTEXT_MARKERS
        )
    ):
        return "supporting_context"

    return "supporting_context"


# ============================================================
# EVIDENCE SCORING
# ============================================================


EVIDENCE_TYPE_BONUS: dict[
    EvidenceType,
    int,
] = {
    "direct_rule":
        30,

    "supporting_context":
        20,

    "guardrail":
        10,
}


def deterministic_evidence_score(
    *,
    evidence_text: str,
    column: str,
    issue_kind: str,
) -> tuple[
    int,
    list[str],
]:
    """
    Score lexical déterministe de base.

    Conditions :

    - mention explicite de la colonne ;
    - vocabulaire relié au problème de qualité ;
    - absence de disclaimer explicite.
    """

    if is_explicit_non_evidence(
        evidence_text
    ):
        return (
            0,
            [],
        )

    if not evidence_mentions_column(
        evidence_text=
            evidence_text,

        column=
            column,
    ):
        return (
            0,
            [],
        )

    matched_keywords = (
        matched_issue_keywords(
            evidence_text=
                evidence_text,

            issue_kind=
                issue_kind,
        )
    )

    issue_has_keywords = bool(
        ISSUE_KEYWORDS.get(
            issue_kind
        )
    )

    if (
        issue_has_keywords
        and
        not matched_keywords
    ):
        return (
            0,
            [],
        )

    # Exact column match.
    score = 10

    # More issue-specific vocabulary increases confidence
    # that the sentence actually discusses the detected issue.
    score += min(
        len(
            matched_keywords
        ),
        10,
    )

    normalized = normalize_text(
        evidence_text
    )

    interpretation_markers = [
        "signifie",
        "represente",
        "doit",
        "interpretee",
        "interprete",
        "consideree",
        "considere",
        "verifiee",
        "verifie",
    ]

    for marker in interpretation_markers:
        if (
            normalize_text(
                marker
            )
            in
            normalized
        ):
            score += 1

    return (
        score,
        matched_keywords,
    )


def final_evidence_score(
    *,
    deterministic_score: int,
    evidence_type: EvidenceType,
) -> int:
    return (
        deterministic_score
        +
        EVIDENCE_TYPE_BONUS[
            evidence_type
        ]
    )


# ============================================================
# EVIDENCE EXTRACTION
# ============================================================


def build_evidence_candidates(
    *,
    hits: Iterable[
        RagSearchHit
    ],
    decision: PreparationDecision,
) -> list[
    PreparationRagEvidence
]:
    if decision.column is None:
        return []

    candidates: list[
        PreparationRagEvidence
    ] = []

    for hit in hits:
        units = build_evidence_units(
            hit.text
        )

        for unit in units:
            cleaned_unit = (
                unit.strip()
            )

            if not cleaned_unit:
                continue

            if is_explicit_non_evidence(
                cleaned_unit
            ):
                continue

            (
                deterministic_score,
                matched_keywords,
            ) = (
                deterministic_evidence_score(
                    evidence_text=
                        cleaned_unit,

                    column=
                        decision.column,

                    issue_kind=
                        decision.source_issue_kind,
                )
            )

            if deterministic_score <= 0:
                continue

            evidence_type = (
                classify_evidence_type(
                    cleaned_unit
                )
            )

            final_score = (
                final_evidence_score(
                    deterministic_score=
                        deterministic_score,

                    evidence_type=
                        evidence_type,
                )
            )

            candidates.append(
                PreparationRagEvidence(
                    chunk_id=
                        hit.chunk_id,

                    document_id=
                        hit.document_id,

                    filename=
                        hit.filename,

                    extension=
                        hit.extension,

                    chunk_index=
                        hit.chunk_index,

                    page_number=
                        hit.page_number,

                    source_locator=
                        hit.source_locator,

                    retrieval_rank=
                        hit.rank,

                    retrieval_score=
                        float(
                            hit.score
                        ),

                    evidence_text=
                        cleaned_unit,

                    matched_column=
                        decision.column,

                    matched_keywords=
                        matched_keywords,

                    evidence_type=
                        evidence_type,

                    deterministic_score=
                        deterministic_score,

                    final_score=
                        final_score,
                )
            )

    return candidates


# ============================================================
# GLOBAL EVIDENCE DEDUPLICATION
# ============================================================


def evidence_quality_key(
    evidence: PreparationRagEvidence,
) -> tuple[
    int,
    int,
    float,
    int,
]:
    """
    Plus grand = meilleure occurrence.
    """

    return (
        evidence.final_score,
        evidence.deterministic_score,
        evidence.retrieval_score,
        -evidence.retrieval_rank,
    )


def deduplicate_evidence(
    evidence: list[
        PreparationRagEvidence
    ],
) -> list[
    PreparationRagEvidence
]:
    """
    Déduplique globalement par texte de preuve.

    Dans la v0.1 la clé contenait aussi chunk_id,
    ce qui permettait à la même phrase d'apparaître
    plusieurs fois lorsque les chunks se chevauchaient.

    Ici :

        même phrase
        +
        chunk différent
        =
        UNE seule preuve

    On conserve la meilleure occurrence.
    """

    best_by_text: dict[
        str,
        PreparationRagEvidence,
    ] = {}

    for item in evidence:
        normalized_text = (
            normalize_text(
                item.evidence_text
            )
        )

        if not normalized_text:
            continue

        current = best_by_text.get(
            normalized_text
        )

        if current is None:
            best_by_text[
                normalized_text
            ] = item

            continue

        if (
            evidence_quality_key(
                item
            )
            >
            evidence_quality_key(
                current
            )
        ):
            best_by_text[
                normalized_text
            ] = item

    output = list(
        best_by_text.values()
    )

    output.sort(
        key=lambda item: (
            -item.final_score,
            -item.deterministic_score,
            -item.retrieval_score,
            item.retrieval_rank,
            item.filename,
            item.chunk_index,
            item.evidence_text,
        )
    )

    return output


# ============================================================
# ELIGIBILITY
# ============================================================


def decision_is_rag_eligible(
    decision: PreparationDecision,
) -> bool:
    if decision.column is None:
        return False

    if (
        decision.status
        ==
        DecisionStatus.AUTO_APPROVABLE
    ):
        return False

    if not decision.context_required:
        return False

    return True


# ============================================================
# SINGLE DECISION RETRIEVAL
# ============================================================


def retrieve_context_for_decision(
    *,
    decision: PreparationDecision,
    ingestion: DocumentIngestionReport,
    top_k: int = DEFAULT_TOP_K,
    model: str = DEFAULT_EMBEDDING_MODEL,
    max_evidence: int = 3,
) -> PreparationRagContext:
    if decision.column is None:
        raise ValueError(
            (
                "Preparation RAG Context "
                "nécessite une colonne."
            )
        )

    if max_evidence < 1:
        raise ValueError(
            (
                "max_evidence doit être supérieur "
                "ou égal à 1."
            )
        )

    query = (
        build_preparation_retrieval_query(
            decision=
                decision
        )
    )

    search = (
        search_document_chunks(
            ingestion=
                ingestion,

            query=
                query,

            top_k=
                top_k,

            model=
                model,
        )
    )

    candidates = (
        build_evidence_candidates(
            hits=
                search.hits,

            decision=
                decision,
        )
    )

    candidates = (
        deduplicate_evidence(
            candidates
        )
    )

    accepted = candidates[
        :max_evidence
    ]

    if not accepted:
        return (
            PreparationRagContext(
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

                original_status=
                    decision.status,

                query=
                    query,

                status=
                    "abstained",

                retrieved_hit_count=
                    len(
                        search.hits
                    ),

                accepted_evidence_count=
                    0,

                evidence=[],

                abstention_reason=(
                    "Aucune preuve documentaire valide "
                    "n'a été trouvée pour cette colonne "
                    "et ce problème de qualité."
                ),

                needs_context_after_rag=
                    True,

                human_validation_required=
                    True,
            )
        )

    return (
        PreparationRagContext(
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

            original_status=
                decision.status,

            query=
                query,

            status=
                "evidence_found",

            retrieved_hit_count=
                len(
                    search.hits
                ),

            accepted_evidence_count=
                len(
                    accepted
                ),

            evidence=
                accepted,

            abstention_reason=None,

            # ------------------------------------------------
            # Finding documentary evidence is NOT equivalent
            # to resolving the cleaning decision.
            # ------------------------------------------------

            needs_context_after_rag=
                True,

            human_validation_required=
                True,
        )
    )


# ============================================================
# REPORT HELPERS
# ============================================================


def count_evidence_type(
    *,
    contexts: Iterable[
        PreparationRagContext
    ],
    evidence_type: EvidenceType,
) -> int:
    return sum(
        1
        for context in contexts
        for evidence in context.evidence
        if (
            evidence.evidence_type
            ==
            evidence_type
        )
    )


# ============================================================
# PUBLIC API
# ============================================================


def retrieve_preparation_rag_context(
    *,
    plan: PreparationPlan,
    ingestion: DocumentIngestionReport,
    top_k: int = DEFAULT_TOP_K,
    model: str = DEFAULT_EMBEDDING_MODEL,
    max_evidence_per_decision: int = 3,
) -> PreparationRagContextReport:
    """
    Récupère du contexte métier documentaire pour les
    décisions de préparation.

    Pipeline v0.2 :

        PreparationDecision
                ↓
        query orientée préparation
                ↓
        vector retrieval
                ↓
        evidence units
                ↓
        exact column matching
                ↓
        issue-specific vocabulary
                ↓
        explicit non-evidence rejection
                ↓
        evidence classification
                ↓
        global textual deduplication
                ↓
        ranking
                ↓
        evidence OR abstention

    Cette fonction :

    - ne modifie pas le PreparationPlan ;
    - ne modifie pas les données ;
    - ne choisit aucune action ;
    - ne remplit aucune valeur manquante ;
    - ne supprime aucun outlier ;
    - ne transforme aucune preuve documentaire
      directement en règle d'exécution.
    """

    if not ingestion.chunks:
        raise ValueError(
            (
                "Aucun passage documentaire "
                "n'est disponible."
            )
        )

    if max_evidence_per_decision < 1:
        raise ValueError(
            (
                "max_evidence_per_decision doit être "
                "supérieur ou égal à 1."
            )
        )

    eligible_decisions = [
        decision
        for decision in plan.decisions
        if decision_is_rag_eligible(
            decision
        )
    ]

    contexts: list[
        PreparationRagContext
    ] = []

    for decision in eligible_decisions:
        contexts.append(
            retrieve_context_for_decision(
                decision=
                    decision,

                ingestion=
                    ingestion,

                top_k=
                    top_k,

                model=
                    model,

                max_evidence=
                    max_evidence_per_decision,
            )
        )

    evidence_found_count = sum(
        1
        for context in contexts
        if (
            context.status
            ==
            "evidence_found"
        )
    )

    abstained_count = sum(
        1
        for context in contexts
        if (
            context.status
            ==
            "abstained"
        )
    )

    total_accepted_evidence_count = sum(
        context.accepted_evidence_count
        for context in contexts
    )

    return (
        PreparationRagContextReport(
            document_count=
                ingestion.document_count,

            chunk_count=
                ingestion.chunk_count,

            eligible_decision_count=
                len(
                    eligible_decisions
                ),

            evidence_found_count=
                evidence_found_count,

            abstained_count=
                abstained_count,

            total_accepted_evidence_count=
                total_accepted_evidence_count,

            direct_rule_count=
                count_evidence_type(
                    contexts=
                        contexts,

                    evidence_type=
                        "direct_rule",
                ),

            supporting_context_count=
                count_evidence_type(
                    contexts=
                        contexts,

                    evidence_type=
                        "supporting_context",
                ),

            guardrail_count=
                count_evidence_type(
                    contexts=
                        contexts,

                    evidence_type=
                        "guardrail",
                ),

            contexts=
                contexts,

            notes=[
                (
                    "Le retrieval vectoriel réutilise "
                    "l'infrastructure RAG existante."
                ),
                (
                    "Une preuve doit mentionner explicitement "
                    "la colonne concernée."
                ),
                (
                    "Les phrases indiquant explicitement "
                    "qu'elles ne fournissent aucune information "
                    "sont rejetées."
                ),
                (
                    "Les preuves sont classées comme "
                    "direct_rule, supporting_context "
                    "ou guardrail."
                ),
                (
                    "Les preuves identiques présentes dans "
                    "plusieurs chunks ne sont conservées "
                    "qu'une seule fois."
                ),
                (
                    "Une preuve documentaire trouvée ne devient "
                    "jamais automatiquement une action "
                    "de nettoyage."
                ),
                (
                    "L'absence de preuve valide produit "
                    "une abstention explicite."
                ),
            ],

            embedding_model=
                model,

            rule_version=
                PREPARATION_RAG_CONTEXT_RULE_VERSION,
        )
    )