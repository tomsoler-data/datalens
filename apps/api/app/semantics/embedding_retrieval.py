from __future__ import annotations

from typing import (
    Iterable,
    Mapping,
)

import numpy as np

from app.ai.provider import (
    client,
)

from app.semantics.embedding_retrieval_schemas import (
    EMBEDDING_CANDIDATE_RETRIEVER_VERSION,
    SemanticEmbeddingCandidatePair,
    SemanticEmbeddingCandidateResult,
    SemanticEmbeddingColumnRetrieval,
    SemanticEmbeddingNeighbour,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_EMBEDDING_MODEL = (
    "embeddinggemma"
)


DEFAULT_EMBEDDING_TOP_K = (
    3
)


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def _normalize_columns(
    columns: Iterable[
        str
    ],
) -> list[
    str
]:
    normalized = []

    seen = set()


    for column in (
        columns
    ):
        value = str(
            column
        ).strip()


        if not value:
            continue


        if (
            value
            in
            seen
        ):
            continue


        seen.add(
            value
        )

        normalized.append(
            value
        )


    return normalized


# ============================================================
# PAIR COUNT
# ============================================================

def _all_possible_pair_count(
    column_count: int,
) -> int:
    if (
        column_count
        <
        2
    ):
        return 0


    return (
        column_count
        *
        (
            column_count
            -
            1
        )
        //
        2
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def _cosine_similarity(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    denominator = (
        np.linalg.norm(
            left
        )
        *
        np.linalg.norm(
            right
        )
    )


    if (
        denominator
        ==
        0
    ):
        raise ValueError(
            "Cannot compute cosine similarity for a "
            "zero-length embedding vector."
        )


    return float(
        np.dot(
            left,
            right,
        )
        /
        denominator
    )


# ============================================================
# EMBEDDING GENERATION
# ============================================================

def embed_column_names(
    *,
    columns: Iterable[
        str
    ],
    model: str = (
        DEFAULT_EMBEDDING_MODEL
    ),
) -> dict[
    str,
    np.ndarray,
]:
    normalized_columns = (
        _normalize_columns(
            columns
        )
    )


    if not normalized_columns:
        return {}


    response = (
        client.embed(
            model=
                model,

            input=
                normalized_columns,
        )
    )


    vectors = (
        response[
            "embeddings"
        ]
    )


    if (
        len(
            vectors
        )
        !=
        len(
            normalized_columns
        )
    ):
        raise RuntimeError(
            "Embedding response count does not match "
            "the number of requested columns."
        )


    embedding_index = {
        column:
            np.asarray(
                vector,
                dtype=float,
            )

        for column, vector
        in zip(
            normalized_columns,
            vectors,
            strict=True,
        )
    }


    dimensions = {
        vector.shape[
            0
        ]

        for vector
        in embedding_index.values()
    }


    if (
        len(
            dimensions
        )
        !=
        1
    ):
        raise RuntimeError(
            "Embedding vectors have inconsistent "
            "dimensions."
        )


    return (
        embedding_index
    )


# ============================================================
# INTERNAL DIRECTED RETRIEVAL
#
# Important:
#
# This helper receives PRECOMPUTED embeddings.
#
# It therefore performs no network/model call.
# ============================================================

def _rank_embedding_neighbours_from_index(
    *,
    columns: list[
        str
    ],
    embedding_index: Mapping[
        str,
        np.ndarray,
    ],
    top_k: int,
) -> list[
    SemanticEmbeddingColumnRetrieval
]:
    if (
        top_k
        <
        1
    ):
        raise ValueError(
            "top_k must be at least 1."
        )


    retrievals = []


    for query_column in (
        columns
    ):
        candidates = []


        for candidate_column in (
            columns
        ):
            if (
                candidate_column
                ==
                query_column
            ):
                continue


            score = (
                _cosine_similarity(
                    embedding_index[
                        query_column
                    ],

                    embedding_index[
                        candidate_column
                    ],
                )
            )


            candidates.append(
                (
                    candidate_column,
                    score,
                )
            )


        # ----------------------------------------------------
        # Deterministic ordering:
        #
        # 1. cosine similarity descending
        # 2. column name ascending for exact ties
        # ----------------------------------------------------

        candidates.sort(
            key=lambda item: (
                -item[
                    1
                ],
                item[
                    0
                ],
            )
        )


        effective_k = min(
            top_k,
            len(
                candidates
            ),
        )


        neighbours = [
            SemanticEmbeddingNeighbour(
                column=
                    candidate_column,

                rank=
                    rank,

                cosine_similarity=
                    score,
            )

            for rank, (
                candidate_column,
                score,
            )
            in enumerate(
                candidates[
                    :effective_k
                ],
                start=1,
            )
        ]


        retrievals.append(
            SemanticEmbeddingColumnRetrieval(
                column=
                    query_column,

                neighbours=
                    neighbours,
            )
        )


    return (
        retrievals
    )


# ============================================================
# PUBLIC DIRECTED RETRIEVAL
#
# One embedding batch.
# ============================================================

def rank_embedding_neighbours(
    *,
    columns: Iterable[
        str
    ],
    model: str = (
        DEFAULT_EMBEDDING_MODEL
    ),
    top_k: int = (
        DEFAULT_EMBEDDING_TOP_K
    ),
) -> list[
    SemanticEmbeddingColumnRetrieval
]:
    if (
        top_k
        <
        1
    ):
        raise ValueError(
            "top_k must be at least 1."
        )


    normalized_columns = (
        _normalize_columns(
            columns
        )
    )


    if not normalized_columns:
        return []


    embedding_index = (
        embed_column_names(
            columns=
                normalized_columns,

            model=
                model,
        )
    )


    return (
        _rank_embedding_neighbours_from_index(
            columns=
                normalized_columns,

            embedding_index=
                embedding_index,

            top_k=
                top_k,
        )
    )


# ============================================================
# UNDIRECTED CANDIDATE GRAPH
#
# Candidate inclusion rule:
#
# retain {A, B} when:
#
#     B in top-k(A)
#
# OR:
#
#     A in top-k(B)
#
# This is intentionally recall-oriented.
#
# IMPORTANT SAFETY PROPERTY:
#
# Inclusion in this graph does NOT authorize any semantic
# relation or analytical operation.
#
# The graph only says:
#
#     "this pair deserves further examination"
#
# One embedding batch is used for the complete dataset.
# ============================================================

def build_embedding_candidate_graph(
    *,
    dataset_id: str,
    columns: Iterable[
        str
    ],
    model: str = (
        DEFAULT_EMBEDDING_MODEL
    ),
    top_k: int = (
        DEFAULT_EMBEDDING_TOP_K
    ),
) -> SemanticEmbeddingCandidateResult:
    if (
        top_k
        <
        1
    ):
        raise ValueError(
            "top_k must be at least 1."
        )


    normalized_columns = (
        _normalize_columns(
            columns
        )
    )


    column_count = (
        len(
            normalized_columns
        )
    )


    all_pair_count = (
        _all_possible_pair_count(
            column_count
        )
    )


    # --------------------------------------------------------
    # Empty / single-column dataset
    # --------------------------------------------------------

    if (
        column_count
        <
        2
    ):
        return (
            SemanticEmbeddingCandidateResult(
                dataset_id=
                    dataset_id,

                embedding_model=
                    model,

                top_k=
                    top_k,

                column_count=
                    column_count,

                all_possible_pair_count=
                    all_pair_count,

                candidate_pair_count=
                    0,

                reduction_ratio=
                    0.0,

                retrievals=[],

                candidate_pairs=[],

                retriever_version=
                    EMBEDDING_CANDIDATE_RETRIEVER_VERSION,
            )
        )


    # --------------------------------------------------------
    # ONE embedding call for the whole dataset.
    # --------------------------------------------------------

    embedding_index = (
        embed_column_names(
            columns=
                normalized_columns,

            model=
                model,
        )
    )


    # --------------------------------------------------------
    # Directed top-k neighbours reuse the precomputed vectors.
    # --------------------------------------------------------

    retrievals = (
        _rank_embedding_neighbours_from_index(
            columns=
                normalized_columns,

            embedding_index=
                embedding_index,

            top_k=
                top_k,
        )
    )


    # --------------------------------------------------------
    # Directed retrieval evidence.
    # --------------------------------------------------------

    directed_rank = {}


    for retrieval in (
        retrievals
    ):
        for neighbour in (
            retrieval.neighbours
        ):
            directed_rank[
                (
                    retrieval.column,
                    neighbour.column,
                )
            ] = (
                neighbour.rank
            )


    # --------------------------------------------------------
    # Convert directed retrieval edges into an undirected
    # candidate graph.
    # --------------------------------------------------------

    candidate_pair_keys = set()


    for (
        left_column,
        right_column,
    ) in directed_rank:
        candidate_pair_keys.add(
            frozenset(
                {
                    left_column,
                    right_column,
                }
            )
        )


    candidate_pairs = []


    for pair_key in sorted(
        candidate_pair_keys,

        key=lambda pair:
            tuple(
                sorted(
                    pair
                )
            ),
    ):
        left_column, right_column = (
            sorted(
                pair_key
            )
        )


        left_to_right_rank = (
            directed_rank.get(
                (
                    left_column,
                    right_column,
                )
            )
        )


        right_to_left_rank = (
            directed_rank.get(
                (
                    right_column,
                    left_column,
                )
            )
        )


        retrieved_from_left = (
            left_to_right_rank
            is not None
        )


        retrieved_from_right = (
            right_to_left_rank
            is not None
        )


        # ----------------------------------------------------
        # Reuse the SAME vectors.
        #
        # No second call to embeddinggemma.
        # ----------------------------------------------------

        similarity = (
            _cosine_similarity(
                embedding_index[
                    left_column
                ],

                embedding_index[
                    right_column
                ],
            )
        )


        candidate_pairs.append(
            SemanticEmbeddingCandidatePair(
                left_column=
                    left_column,

                right_column=
                    right_column,

                cosine_similarity=
                    similarity,

                left_to_right_rank=
                    left_to_right_rank,

                right_to_left_rank=
                    right_to_left_rank,

                retrieved_from_left=
                    retrieved_from_left,

                retrieved_from_right=
                    retrieved_from_right,

                mutual_retrieval=(
                    retrieved_from_left
                    and
                    retrieved_from_right
                ),
            )
        )


    candidate_pair_count = (
        len(
            candidate_pairs
        )
    )


    reduction_ratio = (
        1.0
        -
        (
            candidate_pair_count
            /
            all_pair_count
        )

        if (
            all_pair_count
            >
            0
        )

        else
        0.0
    )


    return (
        SemanticEmbeddingCandidateResult(
            dataset_id=
                dataset_id,

            embedding_model=
                model,

            top_k=
                top_k,

            column_count=
                column_count,

            all_possible_pair_count=
                all_pair_count,

            candidate_pair_count=
                candidate_pair_count,

            reduction_ratio=
                reduction_ratio,

            retrievals=
                retrievals,

            candidate_pairs=
                candidate_pairs,

            retriever_version=
                EMBEDDING_CANDIDATE_RETRIEVER_VERSION,
        )
    )
