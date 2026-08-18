from __future__ import annotations


from typing import (
    Any,
    Literal,
)


from pydantic import (
    BaseModel,
)


from app.rag import (
    DocumentIngestionReport,
)

from app.rag_explanation import (
    DEFAULT_EXPLANATION_MODEL,
    RAG_EXPLANATION_RULE_VERSION,
    RagCitation,
    VerifiedRagExplanation,
    generate_grounded_explanation,
)

from app.rag_relevance import (
    DEFAULT_RELEVANCE_MODEL,
    RELEVANCE_RULE_VERSION,
    RelevanceRelationType,
    RelevanceStrength,
    classify_relevance,
)

from app.rag_retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    RAG_RETRIEVAL_RULE_VERSION,
    RagSearchHit,
    build_search_hit,
    cosine_similarity,
    embed_text_batch,
    prepare_document_for_embedding,
    prepare_query_for_embedding,
)

from app.reporting.unified_schemas import (
    UnifiedAnalysisReport,
)


# ============================================================
# VERSION
# ============================================================

RAG_CONTEXT_RULE_VERSION = (
    "rag_contextualization_v0.6"
)


# ============================================================
# SCHEMAS
# ============================================================

class AnalyticalContract(
    BaseModel
):
    """
    Deterministic semantic description of an analytical
    finding.

    This contract deliberately excludes calculated values.

    Its role is to describe WHAT was analysed, not WHAT
    result Python obtained.
    """

    family: str

    title: str

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

    measure_semantics: (
        str
        | None
    ) = None

    analytical_relationship: str


class RagHitRelevanceDecision(
    BaseModel
):
    rank: int

    chunk_id: str

    filename: str

    source_locator: str

    score: float

    verdict: Literal[
        "relevant",
        "not_relevant",
    ]

    relation_type: RelevanceRelationType

    strength: RelevanceStrength

    reason: str


class DeterministicDocumentContext(
    BaseModel
):
    status: Literal[
        "available",
        "abstained",
    ]

    relation_type: (
        RelevanceRelationType
        | None
    ) = None

    strength: (
        RelevanceStrength
        | None
    ) = None

    message: str

    citation: (
        RagCitation
        | None
    ) = None


class FindingRagContext(
    BaseModel
):
    analysis_id: str

    title: str

    family: str

    analytical_contract: AnalyticalContract

    query: str

    relevance_finding_text: str

    hits: list[
        RagSearchHit
    ]

    relevance_decisions: list[
        RagHitRelevanceDecision
    ]

    accepted_hits: list[
        RagSearchHit
    ]

    documentary_context: DeterministicDocumentContext

    abstained: bool

    abstention_reason: (
        str
        | None
    ) = None

    explanation: VerifiedRagExplanation

    explanation_error: (
        str
        | None
    ) = None


class RagContextReport(
    BaseModel
):
    status: Literal[
        "ready"
    ] = "ready"

    objective: (
        str
        | None
    )

    document_count: int

    chunk_count: int

    finding_count: int

    top_k: int

    model: str

    relevance_model: str

    explanation_model: str

    validated_candidate_count: int

    accepted_hit_count: int

    accepted_finding_count: int

    abstained_finding_count: int

    documentary_context_available_count: int

    explanation_ready_count: int

    explanation_abstained_count: int

    explanation_error_count: int

    contexts: list[
        FindingRagContext
    ]

    retrieval_rule_version: str = (
        RAG_RETRIEVAL_RULE_VERSION
    )

    relevance_rule_version: str = (
        RELEVANCE_RULE_VERSION
    )

    explanation_rule_version: str = (
        RAG_EXPLANATION_RULE_VERSION
    )

    context_rule_version: str = (
        RAG_CONTEXT_RULE_VERSION
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_optional_text(
    value: (
        str
        | None
    ),
) -> (
    str
    | None
):
    if value is None:
        return None


    normalized = (
        value
        .strip()
    )


    if not normalized:
        return None


    return normalized


def normalize_metric_name(
    value: Any,
) -> (
    str
    | None
):
    if value is None:
        return None


    normalized = str(
        value
    ).strip()


    if not normalized:
        return None


    return normalized


def get_finding_metrics(
    *,
    finding: Any,
) -> dict[
    str,
    Any,
]:
    """
    Return finding.metrics as a normal dictionary.

    Unified report findings currently expose metrics as a
    dict, but this helper also tolerates Pydantic models.
    """

    metrics = getattr(
        finding,
        "metrics",
        {},
    )


    if metrics is None:
        return {}


    if isinstance(
        metrics,
        BaseModel,
    ):
        return metrics.model_dump()


    if isinstance(
        metrics,
        dict,
    ):
        return metrics


    try:
        return dict(
            metrics
        )


    except (
        TypeError,
        ValueError,
    ):
        return {}


# ============================================================
# MEASURE SEMANTICS
# ============================================================

def build_measure_semantics(
    *,
    measure_column: (
        str
        | None
    ),
) -> (
    str
    | None
):
    """
    Add only semantics that can be derived safely from the
    technical variable name itself.

    No business-domain synonym is invented here.
    """

    if measure_column is None:
        return None


    if (
        measure_column
        ==
        "event_count"
    ):
        return (
            "La mesure technique `event_count` représente "
            "un comptage d'événements. Elle ne doit pas être "
            "assimilée automatiquement à un chiffre "
            "d'affaires, un prix, un montant d'achat ou une "
            "autre mesure."
        )


    if (
        measure_column.startswith(
            "sum_"
        )
        and
        len(
            measure_column
        )
        >
        len(
            "sum_"
        )
    ):
        source_column = (
            measure_column[
                len(
                    "sum_"
                ):
            ]
        )


        return (
            f"La mesure `{measure_column}` représente une "
            f"agrégation par somme de la variable technique "
            f"`{source_column}`."
        )


    return (
        f"La mesure analysée est exactement "
        f"`{measure_column}`. Elle ne doit pas être remplacée "
        f"par une autre mesure simplement parce qu'elle est "
        f"proche dans le domaine métier."
    )


# ============================================================
# ANALYTICAL CONTRACT
# ============================================================

def build_analytical_contract(
    *,
    finding: Any,
) -> AnalyticalContract:
    """
    Build a deterministic contract from the finding family
    and its structured metrics.

    Calculated values such as coefficients, totals, shares,
    p-values and changes are deliberately ignored.
    """

    title = str(
        getattr(
            finding,
            "title",
            "",
        )
        or ""
    ).strip()


    family = str(
        getattr(
            finding,
            "family",
            "",
        )
        or "unknown"
    ).strip()


    metrics = (
        get_finding_metrics(
            finding=
                finding,
        )
    )


    measure_column = (
        normalize_metric_name(
            metrics.get(
                "measure_column"
            )
        )
    )


    group_column = (
        normalize_metric_name(
            metrics.get(
                "group_column"
            )
        )
    )


    x_column = (
        normalize_metric_name(
            metrics.get(
                "x_column"
            )
        )
    )


    y_column = (
        normalize_metric_name(
            metrics.get(
                "y_column"
            )
        )
    )


    time_column = (
        normalize_metric_name(
            metrics.get(
                "time_column"
            )
        )
    )


    measure_semantics = (
        build_measure_semantics(
            measure_column=
                measure_column,
        )
    )


    # --------------------------------------------------------
    # Aggregate breakdown
    # --------------------------------------------------------

    if (
        family
        ==
        "aggregate_breakdown"
        and
        measure_column
        and
        group_column
    ):
        analytical_relationship = (
            f"Étudier la répartition de la mesure "
            f"`{measure_column}` entre les groupes définis "
            f"par `{group_column}`."
        )


    # --------------------------------------------------------
    # Quantitative association
    # --------------------------------------------------------

    elif (
        family
        ==
        "quantitative_association"
        and
        x_column
        and
        y_column
    ):
        analytical_relationship = (
            f"Étudier l'association entre la variable "
            f"`{x_column}` et la variable `{y_column}`."
        )


    # --------------------------------------------------------
    # Time series
    # --------------------------------------------------------

    elif (
        family
        ==
        "time_series"
        and
        measure_column
        and
        time_column
    ):
        analytical_relationship = (
            f"Étudier l'évolution temporelle de la mesure "
            f"`{measure_column}` selon la dimension "
            f"temporelle `{time_column}`."
        )


    # --------------------------------------------------------
    # Group comparison
    # --------------------------------------------------------

    elif (
        family
        ==
        "group_comparison"
        and
        measure_column
        and
        group_column
    ):
        analytical_relationship = (
            f"Comparer la mesure `{measure_column}` entre "
            f"les groupes définis par `{group_column}`."
        )


    # --------------------------------------------------------
    # Generic structured association fallback
    # --------------------------------------------------------

    elif (
        x_column
        and
        y_column
    ):
        analytical_relationship = (
            f"Étudier la relation analytique entre "
            f"`{x_column}` et `{y_column}`."
        )


    # --------------------------------------------------------
    # Generic measure/group fallback
    # --------------------------------------------------------

    elif (
        measure_column
        and
        group_column
    ):
        analytical_relationship = (
            f"Étudier la mesure `{measure_column}` en "
            f"fonction des groupes définis par "
            f"`{group_column}`."
        )


    # --------------------------------------------------------
    # Generic measure/time fallback
    # --------------------------------------------------------

    elif (
        measure_column
        and
        time_column
    ):
        analytical_relationship = (
            f"Étudier la mesure `{measure_column}` selon "
            f"la dimension temporelle `{time_column}`."
        )


    # --------------------------------------------------------
    # Safe fallback
    # --------------------------------------------------------

    elif title:
        analytical_relationship = (
            "Le contrat analytique est défini par le titre "
            f"du finding : `{title}`. Aucune équivalence de "
            "variable absente du contrat ne doit être "
            "inventée."
        )


    else:
        raise ValueError(
            (
                "Impossible de construire un contrat "
                "analytique pour ce finding."
            )
        )


    return (
        AnalyticalContract(
            family=
                family,

            title=
                title,

            measure_column=
                measure_column,

            group_column=
                group_column,

            x_column=
                x_column,

            y_column=
                y_column,

            time_column=
                time_column,

            measure_semantics=
                measure_semantics,

            analytical_relationship=
                analytical_relationship,
        )
    )


# ============================================================
# ANALYTICAL CONTRACT TEXT
# ============================================================

def build_analytical_contract_text(
    *,
    contract: AnalyticalContract,
) -> str:
    parts: list[
        str
    ] = [
        "CONTRAT ANALYTIQUE DÉTERMINISTE",
        (
            "Famille analytique : "
            f"{contract.family}"
        ),
    ]


    if contract.measure_column:
        parts.append(
            (
                "Mesure : "
                f"{contract.measure_column}"
            )
        )


    if contract.group_column:
        parts.append(
            (
                "Variable de regroupement : "
                f"{contract.group_column}"
            )
        )


    if contract.x_column:
        parts.append(
            (
                "Variable X : "
                f"{contract.x_column}"
            )
        )


    if contract.y_column:
        parts.append(
            (
                "Variable Y : "
                f"{contract.y_column}"
            )
        )


    if contract.time_column:
        parts.append(
            (
                "Dimension temporelle : "
                f"{contract.time_column}"
            )
        )


    if contract.measure_semantics:
        parts.append(
            (
                "Sémantique déterministe de la mesure : "
                f"{contract.measure_semantics}"
            )
        )


    parts.extend(
        [
            (
                "Relation analytique requise : "
                f"{contract.analytical_relationship}"
            ),

            (
                "Règle stricte : les variables et la relation "
                "ci-dessus sont contraignantes. Un document "
                "qui traite seulement du même domaine général "
                "n'est pas pertinent."
            ),

            (
                "Règle stricte : ne jamais remplacer une "
                "mesure, une variable, une dimension ou une "
                "relation par une autre simplement parce "
                "qu'elles semblent proches."
            ),

            (
                "Une formulation métier équivalente peut être "
                "acceptée uniquement si son équivalence avec "
                "le contrat est directement justifiable. "
                "En cas de doute, le passage doit être rejeté."
            ),
        ]
    )


    return "\n".join(
        parts
    )


# ============================================================
# FINDING RETRIEVAL QUERY
# ============================================================

def build_finding_retrieval_query(
    *,
    finding: Any,
) -> str:
    """
    Build a concept-oriented retrieval query.

    Numerical results are deliberately excluded.

    The retrieval layer is looking for documentary context
    about the analytical question, not for a document that
    reproduces values calculated by Python.
    """

    contract = (
        build_analytical_contract(
            finding=
                finding,
        )
    )


    parts: list[
        str
    ] = [
        build_analytical_contract_text(
            contract=
                contract,
        )
    ]


    datasets = (
        getattr(
            finding,
            "datasets",
            [],
        )
        or []
    )


    dataset_names = [
        str(
            dataset
        ).strip()

        for dataset
        in datasets

        if str(
            dataset
        ).strip()
    ]


    if dataset_names:
        parts.append(
            (
                "Données concernées : "
                +
                ", ".join(
                    dataset_names
                )
            )
        )


    parts.append(
        (
            "Chercher un passage documentaire directement "
            "pertinent pour ce contrat analytique : demande "
            "explicite, objectif métier, définition, règle, "
            "contrainte ou contexte d'interprétation."
        )
    )


    query = "\n".join(
        parts
    ).strip()


    if not query:
        raise ValueError(
            (
                "Impossible de construire "
                "une requête documentaire "
                "pour cette analyse."
            )
        )


    return query


# ============================================================
# FINDING TEXT FOR RELEVANCE / EXPLANATION
# ============================================================

def build_finding_relevance_text(
    *,
    finding: Any,
) -> str:
    """
    Build the deterministic conceptual representation sent
    to relevance classification and grounded explanation.

    Calculated results are intentionally excluded.
    """

    contract = (
        build_analytical_contract(
            finding=
                finding,
        )
    )


    parts: list[
        str
    ] = []


    if contract.title:
        parts.append(
            (
                "Titre du finding : "
                f"{contract.title}"
            )
        )


    parts.append(
        build_analytical_contract_text(
            contract=
                contract,
        )
    )


    finding_text = "\n\n".join(
        parts
    ).strip()


    if not finding_text:
        raise ValueError(
            (
                "Impossible de construire "
                "le finding utilisé pour "
                "la validation documentaire."
            )
        )


    return finding_text


# ============================================================
# SAFE EXPLANATION FALLBACK
# ============================================================

def build_safe_explanation_fallback(
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
# RELATION PRIORITY
# ============================================================

RELATION_PRIORITY: dict[
    RelevanceRelationType,
    int,
] = {
    "explicit_request":
        60,

    "business_rule":
        50,

    "business_definition":
        40,

    "objective_support":
        30,

    "methodological_context":
        20,

    "interpretation_context":
        10,

    "not_relevant":
        0,
}


STRENGTH_PRIORITY: dict[
    RelevanceStrength,
    int,
] = {
    "direct":
        20,

    "supporting":
        10,

    "none":
        0,
}


# ============================================================
# DETERMINISTIC CONTEXT MESSAGE
# ============================================================

def build_context_message(
    *,
    relation_type: RelevanceRelationType,
) -> str:
    messages: dict[
        RelevanceRelationType,
        str,
    ] = {
        "explicit_request":
            (
                "La documentation contient une demande "
                "explicitement liée à cette analyse."
            ),

        "business_rule":
            (
                "La documentation contient une règle métier "
                "directement liée à cette analyse."
            ),

        "business_definition":
            (
                "La documentation contient une définition "
                "métier directement utile à cette analyse."
            ),

        "objective_support":
            (
                "Cette analyse répond directement à un "
                "objectif mentionné dans la documentation."
            ),

        "interpretation_context":
            (
                "La documentation apporte un contexte "
                "directement utile à l'interprétation "
                "de cette analyse."
            ),

        "methodological_context":
            (
                "La documentation contient une instruction "
                "méthodologique directement applicable "
                "à cette analyse."
            ),

        "not_relevant":
            (
                "Aucun contexte documentaire suffisamment "
                "direct n'est disponible."
            ),
    }


    return messages[
        relation_type
    ]


# ============================================================
# DETERMINISTIC DOCUMENT CONTEXT
# ============================================================

def build_deterministic_document_context(
    *,
    accepted_hits: list[
        RagSearchHit
    ],
    relevance_decisions: list[
        RagHitRelevanceDecision
    ],
) -> DeterministicDocumentContext:
    if not accepted_hits:
        return (
            DeterministicDocumentContext(
                status=
                    "abstained",

                relation_type=
                    None,

                strength=
                    None,

                message=(
                    "Aucun passage documentaire suffisamment "
                    "pertinent n'a été validé pour cette "
                    "analyse."
                ),

                citation=
                    None,
            )
        )


    accepted_hit_lookup = {
        hit.chunk_id:
            hit

        for hit
        in accepted_hits
    }


    positive_decisions = [
        decision

        for decision
        in relevance_decisions

        if (
            decision.verdict
            ==
            "relevant"
            and
            decision.chunk_id
            in
            accepted_hit_lookup
        )
    ]


    if not positive_decisions:
        return (
            DeterministicDocumentContext(
                status=
                    "abstained",

                relation_type=
                    None,

                strength=
                    None,

                message=(
                    "Aucun contexte documentaire validé "
                    "n'est disponible."
                ),

                citation=
                    None,
            )
        )


    positive_decisions.sort(
        key=lambda decision: (
            -STRENGTH_PRIORITY[
                decision.strength
            ],
            -RELATION_PRIORITY[
                decision.relation_type
            ],
            -decision.score,
            decision.rank,
        )
    )


    best_decision = (
        positive_decisions[
            0
        ]
    )


    best_hit = (
        accepted_hit_lookup[
            best_decision.chunk_id
        ]
    )


    citation = (
        RagCitation(
            chunk_id=
                best_hit.chunk_id,

            filename=
                best_hit.filename,

            source_locator=
                best_hit.source_locator,

            page_number=
                best_hit.page_number,
        )
    )


    return (
        DeterministicDocumentContext(
            status=
                "available",

            relation_type=
                best_decision
                .relation_type,

            strength=
                best_decision
                .strength,

            message=
                build_context_message(
                    relation_type=
                        best_decision
                        .relation_type,
                ),

            citation=
                citation,
        )
    )


# ============================================================
# CONTEXT RETRIEVAL + RELEVANCE + EXPLANATION
# ============================================================

def retrieve_context_for_report(
    *,
    report: UnifiedAnalysisReport,
    ingestion: DocumentIngestionReport,
    objective: (
        str
        | None
    ) = None,
    top_k: int = (
        DEFAULT_TOP_K
    ),
    model: str = (
        DEFAULT_EMBEDDING_MODEL
    ),
    relevance_model: str = (
        DEFAULT_RELEVANCE_MODEL
    ),
    explanation_model: str = (
        DEFAULT_EXPLANATION_MODEL
    ),
) -> RagContextReport:
    if not ingestion.chunks:
        raise ValueError(
            (
                "Aucun passage documentaire "
                "n'est disponible."
            )
        )


    if (
        top_k
        <
        1
    ):
        raise ValueError(
            (
                "top_k doit être supérieur "
                "ou égal à 1."
            )
        )


    findings = list(
        report.main_findings
    )


    effective_top_k = min(
        top_k,
        MAX_TOP_K,
        len(
            ingestion.chunks
        ),
    )


    normalized_objective = (
        normalize_optional_text(
            objective
        )
    )


    if not findings:
        return (
            RagContextReport(
                objective=
                    normalized_objective,

                document_count=
                    ingestion.document_count,

                chunk_count=
                    ingestion.chunk_count,

                finding_count=
                    0,

                top_k=
                    effective_top_k,

                model=
                    model,

                relevance_model=
                    relevance_model,

                explanation_model=
                    explanation_model,

                validated_candidate_count=
                    0,

                accepted_hit_count=
                    0,

                accepted_finding_count=
                    0,

                abstained_finding_count=
                    0,

                documentary_context_available_count=
                    0,

                explanation_ready_count=
                    0,

                explanation_abstained_count=
                    0,

                explanation_error_count=
                    0,

                contexts=[],
            )
        )


    analytical_contracts = [
        build_analytical_contract(
            finding=
                finding,
        )

        for finding
        in findings
    ]


    finding_queries = [
        build_finding_retrieval_query(
            finding=
                finding,
        )

        for finding
        in findings
    ]


    relevance_finding_texts = [
        build_finding_relevance_text(
            finding=
                finding,
        )

        for finding
        in findings
    ]


    prepared_queries = [
        prepare_query_for_embedding(
            query
        )

        for query
        in finding_queries
    ]


    prepared_documents = [
        prepare_document_for_embedding(
            chunk
        )

        for chunk
        in ingestion.chunks
    ]


    embedding_inputs = [
        *prepared_queries,
        *prepared_documents,
    ]


    embeddings = (
        embed_text_batch(
            embedding_inputs,
            model=
                model,
        )
    )


    query_count = len(
        prepared_queries
    )


    query_embeddings = (
        embeddings[
            :query_count
        ]
    )


    document_embeddings = (
        embeddings[
            query_count:
        ]
    )


    if (
        len(
            document_embeddings
        )
        !=
        len(
            ingestion.chunks
        )
    ):
        raise RuntimeError(
            (
                "Le nombre d'embeddings "
                "documentaires ne correspond "
                "pas au nombre de chunks."
            )
        )


    contexts: list[
        FindingRagContext
    ] = []


    validated_candidate_count = 0

    accepted_hit_count = 0

    accepted_finding_count = 0

    abstained_finding_count = 0

    documentary_context_available_count = 0

    explanation_ready_count = 0

    explanation_abstained_count = 0

    explanation_error_count = 0


    for (
        finding_index,
        (
            finding,
            analytical_contract,
            query,
            relevance_finding_text,
            query_embedding,
        ),
    ) in enumerate(
        zip(
            findings,
            analytical_contracts,
            finding_queries,
            relevance_finding_texts,
            query_embeddings,
            strict=True,
        ),
        start=1,
    ):
        scored_chunks: list[
            tuple[
                float,
                Any,
            ]
        ] = []


        for (
            chunk,
            document_embedding,
        ) in zip(
            ingestion.chunks,
            document_embeddings,
            strict=True,
        ):
            score = (
                cosine_similarity(
                    query_embedding,
                    document_embedding,
                )
            )


            scored_chunks.append(
                (
                    score,
                    chunk,
                )
            )


        scored_chunks.sort(
            key=lambda item: (
                -item[
                    0
                ],
                item[
                    1
                ].filename,
                item[
                    1
                ].chunk_index,
            )
        )


        selected = (
            scored_chunks[
                :effective_top_k
            ]
        )


        hits = [
            build_search_hit(
                rank=
                    rank,

                score=
                    score,

                chunk=
                    chunk,
            )

            for (
                rank,
                (
                    score,
                    chunk,
                ),
            ) in enumerate(
                selected,
                start=1,
            )
        ]


        relevance_decisions: list[
            RagHitRelevanceDecision
        ] = []


        accepted_hits: list[
            RagSearchHit
        ] = []


        for hit in hits:
            decision = (
                classify_relevance(
                    finding=
                        relevance_finding_text,

                    passage=
                        hit.text,

                    model=
                        relevance_model,
                )
            )


            validated_candidate_count += 1


            relevance_decisions.append(
                RagHitRelevanceDecision(
                    rank=
                        hit.rank,

                    chunk_id=
                        hit.chunk_id,

                    filename=
                        hit.filename,

                    source_locator=
                        hit.source_locator,

                    score=
                        hit.score,

                    verdict=
                        decision.verdict,

                    relation_type=
                        decision.relation_type,

                    strength=
                        decision.strength,

                    reason=
                        decision.reason,
                )
            )


            if (
                decision.verdict
                ==
                "relevant"
            ):
                accepted_hits.append(
                    hit
                )


                accepted_hit_count += 1


        abstained = (
            len(
                accepted_hits
            )
            ==
            0
        )


        if abstained:
            abstained_finding_count += 1


            abstention_reason = (
                "Aucun passage documentaire récupéré n'a "
                "été jugé suffisamment pertinent pour "
                "contextualiser cette analyse."
            )


        else:
            accepted_finding_count += 1


            abstention_reason = None


        documentary_context = (
            build_deterministic_document_context(
                accepted_hits=
                    accepted_hits,

                relevance_decisions=
                    relevance_decisions,
            )
        )


        if (
            documentary_context.status
            ==
            "available"
        ):
            documentary_context_available_count += 1


        explanation_error: (
            str
            | None
        ) = None


        try:
            explanation = (
                generate_grounded_explanation(
                    finding_text=
                        relevance_finding_text,

                    accepted_hits=
                        accepted_hits,

                    model=
                        explanation_model,
                )
            )


        except RuntimeError as error:
            explanation_error_count += 1


            explanation_error = str(
                error
            )


            explanation = (
                build_safe_explanation_fallback(
                    model=
                        explanation_model,

                    reason=(
                        "L'explication enrichie n'a pas pu "
                        "être validée. Le contexte "
                        "documentaire déterministe reste "
                        "disponible lorsqu'une source a "
                        "été validée."
                    ),
                )
            )


        if (
            explanation.status
            ==
            "ready"
        ):
            explanation_ready_count += 1


        else:
            explanation_abstained_count += 1


        raw_analysis_id = getattr(
            finding,
            "analysis_id",
            None,
        )


        analysis_id = (
            str(
                raw_analysis_id
            ).strip()

            if raw_analysis_id
            else (
                "finding:"
                f"{finding_index:04d}"
            )
        )


        title = str(
            getattr(
                finding,
                "title",
                "",
            )
            or analysis_id
        )


        family = str(
            getattr(
                finding,
                "family",
                "",
            )
            or "unknown"
        )


        contexts.append(
            FindingRagContext(
                analysis_id=
                    analysis_id,

                title=
                    title,

                family=
                    family,

                analytical_contract=
                    analytical_contract,

                query=
                    query,

                relevance_finding_text=
                    relevance_finding_text,

                hits=
                    hits,

                relevance_decisions=
                    relevance_decisions,

                accepted_hits=
                    accepted_hits,

                documentary_context=
                    documentary_context,

                abstained=
                    abstained,

                abstention_reason=
                    abstention_reason,

                explanation=
                    explanation,

                explanation_error=
                    explanation_error,
            )
        )


    return (
        RagContextReport(
            objective=
                normalized_objective,

            document_count=
                ingestion.document_count,

            chunk_count=
                ingestion.chunk_count,

            finding_count=
                len(
                    contexts
                ),

            top_k=
                effective_top_k,

            model=
                model,

            relevance_model=
                relevance_model,

            explanation_model=
                explanation_model,

            validated_candidate_count=
                validated_candidate_count,

            accepted_hit_count=
                accepted_hit_count,

            accepted_finding_count=
                accepted_finding_count,

            abstained_finding_count=
                abstained_finding_count,

            documentary_context_available_count=
                documentary_context_available_count,

            explanation_ready_count=
                explanation_ready_count,

            explanation_abstained_count=
                explanation_abstained_count,

            explanation_error_count=
                explanation_error_count,

            contexts=
                contexts,
        )
    )