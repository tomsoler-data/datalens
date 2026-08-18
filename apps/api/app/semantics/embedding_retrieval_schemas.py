from __future__ import annotations

from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# VERSION
# ============================================================

EMBEDDING_CANDIDATE_RETRIEVER_VERSION = (
    "embedding_candidate_retriever_v0.1"
)


# ============================================================
# NEIGHBOUR
# ============================================================

class SemanticEmbeddingNeighbour(
    BaseModel
):
    column: str

    rank: int = Field(
        ge=1,
    )

    cosine_similarity: float


# ============================================================
# COLUMN RETRIEVAL
# ============================================================

class SemanticEmbeddingColumnRetrieval(
    BaseModel
):
    column: str

    neighbours: list[
        SemanticEmbeddingNeighbour
    ] = Field(
        default_factory=list,
    )


# ============================================================
# UNDIRECTED CANDIDATE PAIR
# ============================================================

class SemanticEmbeddingCandidatePair(
    BaseModel
):
    left_column: str

    right_column: str

    cosine_similarity: float

    left_to_right_rank: int | None = None

    right_to_left_rank: int | None = None

    retrieved_from_left: bool

    retrieved_from_right: bool

    mutual_retrieval: bool


# ============================================================
# DATASET RESULT
# ============================================================

class SemanticEmbeddingCandidateResult(
    BaseModel
):
    dataset_id: str

    embedding_model: str

    top_k: int = Field(
        ge=1,
    )

    column_count: int = Field(
        ge=0,
    )

    all_possible_pair_count: int = Field(
        ge=0,
    )

    candidate_pair_count: int = Field(
        ge=0,
    )

    reduction_ratio: float

    retrievals: list[
        SemanticEmbeddingColumnRetrieval
    ] = Field(
        default_factory=list,
    )

    candidate_pairs: list[
        SemanticEmbeddingCandidatePair
    ] = Field(
        default_factory=list,
    )

    safety_role: str = (
        "none_candidate_generation_only"
    )

    retriever_version: str = (
        EMBEDDING_CANDIDATE_RETRIEVER_VERSION
    )
