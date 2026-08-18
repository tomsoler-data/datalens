from __future__ import annotations


from math import sqrt

from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
)


from app.ai.provider import (
    client,
)

from app.rag import (
    DocumentChunk,
    DocumentIngestionReport,
)


# ============================================================
# VERSION
# ============================================================

RAG_RETRIEVAL_RULE_VERSION = (
    "rag_retrieval_v0.2"
)


# ============================================================
# MODEL
# ============================================================

DEFAULT_EMBEDDING_MODEL = (
    "embeddinggemma"
)


# ============================================================
# EMBEDDINGGEMMA RETRIEVAL PROMPTS
# ============================================================

QUERY_RETRIEVAL_PREFIX = (
    "task: search result | query: "
)


DOCUMENT_RETRIEVAL_PREFIX = (
    "title: none | text: "
)


# ============================================================
# LIMITS
# ============================================================

DEFAULT_TOP_K = 5

MAX_TOP_K = 20


# ============================================================
# SCHEMAS
# ============================================================

class RagSearchHit(
    BaseModel
):
    rank: int

    score: float

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

    text: str

    character_count: int


class RagSearchResponse(
    BaseModel
):
    status: Literal[
        "ready"
    ] = "ready"

    query: str

    model: str

    document_count: int

    chunk_count: int

    top_k: int

    hits: list[
        RagSearchHit
    ]

    retrieval_rule_version: str = (
        RAG_RETRIEVAL_RULE_VERSION
    )


# ============================================================
# VECTOR HELPERS
# ============================================================

def vector_dot_product(
    left: list[
        float
    ],
    right: list[
        float
    ],
) -> float:
    if (
        len(
            left
        )
        !=
        len(
            right
        )
    ):
        raise ValueError(
            (
                "Les vecteurs comparés "
                "n'ont pas la même dimension."
            )
        )


    return sum(
        left_value
        *
        right_value

        for (
            left_value,
            right_value,
        ) in zip(
            left,
            right,
            strict=True,
        )
    )


def vector_norm(
    vector: list[
        float
    ],
) -> float:
    return sqrt(
        sum(
            value
            *
            value

            for value
            in vector
        )
    )


def cosine_similarity(
    left: list[
        float
    ],
    right: list[
        float
    ],
) -> float:
    left_norm = (
        vector_norm(
            left
        )
    )


    right_norm = (
        vector_norm(
            right
        )
    )


    if (
        left_norm
        ==
        0.0
        or
        right_norm
        ==
        0.0
    ):
        return 0.0


    return (
        vector_dot_product(
            left,
            right,
        )
        /
        (
            left_norm
            *
            right_norm
        )
    )


# ============================================================
# OLLAMA RESPONSE
# ============================================================

def extract_embeddings(
    response: object,
) -> list[
    list[
        float
    ]
]:
    raw_embeddings = getattr(
        response,
        "embeddings",
        None,
    )


    if (
        raw_embeddings
        is None
        and
        isinstance(
            response,
            dict,
        )
    ):
        raw_embeddings = (
            response.get(
                "embeddings"
            )
        )


    if not raw_embeddings:
        raise RuntimeError(
            (
                "Ollama n'a retourné "
                "aucun embedding."
            )
        )


    embeddings: list[
        list[
            float
        ]
    ] = []


    for raw_vector in (
        raw_embeddings
    ):
        vector = [
            float(
                value
            )

            for value
            in raw_vector
        ]


        if not vector:
            raise RuntimeError(
                (
                    "Ollama a retourné "
                    "un vecteur vide."
                )
            )


        embeddings.append(
            vector
        )


    return embeddings


# ============================================================
# EMBEDDING
# ============================================================

def embed_text_batch(
    texts: list[
        str
    ],
    *,
    model: str = (
        DEFAULT_EMBEDDING_MODEL
    ),
) -> list[
    list[
        float
    ]
]:
    if not texts:
        raise ValueError(
            (
                "Au moins un texte doit "
                "être fourni pour produire "
                "des embeddings."
            )
        )


    cleaned_texts = [
        text.strip()

        for text
        in texts
    ]


    if any(
        not text

        for text
        in cleaned_texts
    ):
        raise ValueError(
            (
                "Un texte vide ne peut "
                "pas être encodé."
            )
        )


    try:
        response = client.embed(
            model=
                model,

            input=
                cleaned_texts,
        )


    except Exception as error:
        raise RuntimeError(
            (
                "La génération des embeddings "
                "Ollama a échoué. Vérifiez que "
                "Ollama fonctionne et que le "
                f"modèle '{model}' est disponible."
            )
        ) from error


    embeddings = (
        extract_embeddings(
            response
        )
    )


    if (
        len(
            embeddings
        )
        !=
        len(
            cleaned_texts
        )
    ):
        raise RuntimeError(
            (
                "Le nombre d'embeddings "
                "retournés par Ollama ne "
                "correspond pas au nombre "
                "de textes envoyés."
            )
        )


    vector_dimensions = {
        len(
            vector
        )

        for vector
        in embeddings
    }


    if (
        len(
            vector_dimensions
        )
        !=
        1
    ):
        raise RuntimeError(
            (
                "Les embeddings retournés "
                "n'ont pas tous la même "
                "dimension."
            )
        )


    return embeddings


# ============================================================
# RETRIEVAL INPUT PREPARATION
# ============================================================

def prepare_query_for_embedding(
    query: str,
) -> str:
    query = (
        query
        .strip()
    )


    if not query:
        raise ValueError(
            (
                "La requête de recherche "
                "ne peut pas être vide."
            )
        )


    return (
        QUERY_RETRIEVAL_PREFIX
        +
        query
    )


def prepare_document_for_embedding(
    chunk: DocumentChunk,
) -> str:
    text = (
        chunk.text
        .strip()
    )


    if not text:
        raise ValueError(
            (
                "Un passage documentaire "
                "vide ne peut pas être "
                "encodé."
            )
        )


    return (
        DOCUMENT_RETRIEVAL_PREFIX
        +
        text
    )


# ============================================================
# HIT BUILDER
# ============================================================

def build_search_hit(
    *,
    rank: int,
    score: float,
    chunk: DocumentChunk,
) -> RagSearchHit:
    return RagSearchHit(
        rank=
            rank,

        score=
            score,

        chunk_id=
            chunk.chunk_id,

        document_id=
            chunk.document_id,

        filename=
            chunk.filename,

        extension=
            chunk.extension,

        chunk_index=
            chunk.chunk_index,

        page_number=
            chunk.page_number,

        source_locator=
            chunk.source_locator,

        text=
            chunk.text,

        character_count=
            chunk.character_count,
    )


# ============================================================
# RETRIEVAL
# ============================================================

def search_document_chunks(
    *,
    ingestion:
        DocumentIngestionReport,

    query:
        str,

    top_k: int = (
        DEFAULT_TOP_K
    ),

    model: str = (
        DEFAULT_EMBEDDING_MODEL
    ),
) -> RagSearchResponse:
    query = (
        query
        .strip()
    )


    if not query:
        raise ValueError(
            (
                "La requête de recherche "
                "ne peut pas être vide."
            )
        )


    if not ingestion.chunks:
        raise ValueError(
            (
                "Aucun passage documentaire "
                "n'est disponible pour la "
                "recherche."
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


    effective_top_k = min(
        top_k,
        MAX_TOP_K,
        len(
            ingestion.chunks
        ),
    )


    prepared_query = (
        prepare_query_for_embedding(
            query
        )
    )


    prepared_documents = [
        prepare_document_for_embedding(
            chunk
        )

        for chunk
        in ingestion.chunks
    ]


    # One Ollama batch call:
    #
    # [retrieval query,
    #  retrieval document 1,
    #  retrieval document 2,
    #  ...]
    #
    # The query and documents use different
    # EmbeddingGemma retrieval instructions,
    # but still share the same embedding model.

    embedding_inputs = [
        prepared_query,
        *prepared_documents,
    ]


    embeddings = (
        embed_text_batch(
            embedding_inputs,
            model=
                model,
        )
    )


    query_embedding = (
        embeddings[
            0
        ]
    )


    chunk_embeddings = (
        embeddings[
            1:
        ]
    )


    scored_chunks: list[
        tuple[
            float,
            DocumentChunk,
        ]
    ] = []


    for (
        chunk,
        chunk_embedding,
    ) in zip(
        ingestion.chunks,
        chunk_embeddings,
        strict=True,
    ):
        score = (
            cosine_similarity(
                query_embedding,
                chunk_embedding,
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
                index,

            score=
                score,

            chunk=
                chunk,
        )

        for (
            index,
            (
                score,
                chunk,
            ),
        ) in enumerate(
            selected,
            start=1,
        )
    ]


    return RagSearchResponse(
        query=
            query,

        model=
            model,

        document_count=
            ingestion.document_count,

        chunk_count=
            ingestion.chunk_count,

        top_k=
            effective_top_k,

        hits=
            hits,
    )