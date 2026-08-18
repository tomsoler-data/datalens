from __future__ import annotations

from collections import (
    Counter,
)

from typing import (
    Any,
)

from app.semantics.embedding_retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_TOP_K,
    build_embedding_candidate_graph,
)

from app.semantics.embedding_shadow_schemas import (
    EMBEDDING_DISCOVERY_SHADOW_AUDIT_VERSION,
    EmbeddingDiscoveryShadowAuditResult,
    EmbeddingShadowCandidateAudit,
    EmbeddingShadowDatasetSummary,
    EmbeddingShadowStageSummary,
)


# ============================================================
# SHADOW POLICY v0.1
#
# IMPORTANT
#
# These families are NOT equivalent.
#
# semantic_pair_families:
#     Semantic relatedness is relevant to whether the pair
#     deserves further examination.
#
# generic_pair_observation_families:
#     We measure embedding coverage only diagnostically.
#     Missing these pairs MUST NOT prune Discovery.
# ============================================================

SEMANTIC_PAIR_FAMILIES = {
    "derived_gap",
}


GENERIC_PAIR_OBSERVATION_FAMILIES = {
    "quantitative_association",
}


TECHNICAL_PAIR_FAMILIES = {
    "time_series",
}


UNIVARIATE_FAMILIES = {
    "distribution",
}


DATASET_LEVEL_FAMILIES = {
    "data_quality",
}


# ============================================================
# DATASET COLUMN INDEX
# ============================================================

def _build_dataset_column_index(
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    list[
        str
    ],
]:
    index = {}


    for dataset in (
        datasets
    ):
        dataset_id = str(
            dataset[
                "dataset_id"
            ]
        )


        dataframe = (
            dataset[
                "dataframe"
            ]
        )


        columns = [
            str(
                column
            )

            for column
            in dataframe.columns
        ]


        index[
            dataset_id
        ] = columns


    return (
        index
    )


# ============================================================
# PAIR EXTRACTION
#
# This function intentionally has a narrow contract.
#
# A semantic/generic pair must:
#
# - belong to a single dataset;
# - contain exactly two variables;
# - refer to two distinct columns;
# - use the same dataset_id.
# ============================================================

def _extract_single_dataset_pair(
    candidate,
) -> tuple[
    str,
    str,
    str,
] | None:
    if (
        candidate.scope
        !=
        "single_dataset"
    ):
        return None


    variables = list(
        candidate.variables
    )


    if (
        len(
            variables
        )
        !=
        2
    ):
        return None


    left = (
        variables[
            0
        ]
    )


    right = (
        variables[
            1
        ]
    )


    if (
        left.dataset_id
        !=
        right.dataset_id
    ):
        return None


    if (
        left.column
        ==
        right.column
    ):
        return None


    return (
        left.dataset_id,
        left.column,
        right.column,
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def _classify_candidate(
    candidate,
) -> tuple[
    str,
    tuple[
        str,
        str,
        str,
    ] | None,
]:
    if (
        candidate.scope
        ==
        "cross_dataset"
    ):
        return (
            "out_of_scope_cross_dataset",
            None,
        )


    family = (
        candidate.family
    )


    if (
        family
        in
        SEMANTIC_PAIR_FAMILIES
    ):
        pair = (
            _extract_single_dataset_pair(
                candidate
            )
        )


        if (
            pair
            is None
        ):
            return (
                "out_of_scope_contract_mismatch",
                None,
            )


        return (
            "semantic_pair_in_scope",
            pair,
        )


    if (
        family
        in
        GENERIC_PAIR_OBSERVATION_FAMILIES
    ):
        pair = (
            _extract_single_dataset_pair(
                candidate
            )
        )


        if (
            pair
            is None
        ):
            return (
                "out_of_scope_contract_mismatch",
                None,
            )


        return (
            "generic_pair_observation_only",
            pair,
        )


    if (
        family
        in
        TECHNICAL_PAIR_FAMILIES
    ):
        return (
            "out_of_scope_technical_pair",
            None,
        )


    if (
        family
        in
        UNIVARIATE_FAMILIES
    ):
        return (
            "out_of_scope_univariate",
            None,
        )


    if (
        family
        in
        DATASET_LEVEL_FAMILIES
    ):
        return (
            "out_of_scope_dataset_level",
            None,
        )


    return (
        "out_of_scope_unclassified_family",
        None,
    )


# ============================================================
# DETERMINE WHICH DATASETS NEED EMBEDDINGS
#
# We build a candidate graph only for datasets that contain
# at least one semantic-pair or generic-pair observation.
# ============================================================

def _collect_pair_dataset_ids(
    *,
    unvalidated_report,
    validated_report,
) -> set[
    str
]:
    dataset_ids = set()


    for report in (
        unvalidated_report,
        validated_report,
    ):
        for candidate in (
            report.candidates
        ):
            (
                classification,
                pair,
            ) = (
                _classify_candidate(
                    candidate
                )
            )


            if (
                classification
                not in {
                    "semantic_pair_in_scope",
                    "generic_pair_observation_only",
                }
            ):
                continue


            if (
                pair
                is None
            ):
                continue


            dataset_id, _, _ = (
                pair
            )


            dataset_ids.add(
                dataset_id
            )


    return (
        dataset_ids
    )


# ============================================================
# BUILD EMBEDDING GRAPHS
# ============================================================

def _build_embedding_graphs(
    *,
    dataset_columns: dict[
        str,
        list[
            str
        ],
    ],
    required_dataset_ids: set[
        str
    ],
    model: str,
    top_k: int,
):
    graph_results = {}


    pair_indexes = {}


    for dataset_id in sorted(
        required_dataset_ids
    ):
        columns = (
            dataset_columns.get(
                dataset_id
            )
        )


        if (
            columns
            is None
        ):
            raise KeyError(
                "Discovery candidate references unknown "
                f"dataset_id: {dataset_id}"
            )


        result = (
            build_embedding_candidate_graph(
                dataset_id=
                    dataset_id,

                columns=
                    columns,

                model=
                    model,

                top_k=
                    top_k,
            )
        )


        graph_results[
            dataset_id
        ] = (
            result
        )


        pair_indexes[
            dataset_id
        ] = {
            frozenset(
                {
                    pair.left_column,
                    pair.right_column,
                }
            ):
                pair

            for pair
            in result.candidate_pairs
        }


    return (
        graph_results,
        pair_indexes,
    )


# ============================================================
# AUDIT ONE STAGE
# ============================================================

def _audit_stage(
    *,
    stage: str,
    report,
    pair_indexes,
) -> tuple[
    EmbeddingShadowStageSummary,
    list[
        EmbeddingShadowCandidateAudit
    ],
]:
    records = []


    classification_counts = (
        Counter()
    )


    semantic_total = 0
    semantic_covered = 0


    generic_total = 0
    generic_covered = 0


    out_of_scope = 0


    for candidate in (
        report.candidates
    ):
        (
            classification,
            pair,
        ) = (
            _classify_candidate(
                candidate
            )
        )


        classification_counts[
            classification
        ] += 1


        dataset_id = None
        left_column = None
        right_column = None


        pair_in_graph = None
        similarity = None

        left_to_right_rank = None
        right_to_left_rank = None

        mutual_retrieval = None


        if (
            classification
            in {
                "semantic_pair_in_scope",
                "generic_pair_observation_only",
            }
            and
            pair
            is not None
        ):
            (
                dataset_id,
                left_column,
                right_column,
            ) = (
                pair
            )


            pair_index = (
                pair_indexes.get(
                    dataset_id,
                    {},
                )
            )


            evidence = (
                pair_index.get(
                    frozenset(
                        {
                            left_column,
                            right_column,
                        }
                    )
                )
            )


            pair_in_graph = (
                evidence
                is not None
            )


            if (
                evidence
                is not None
            ):
                similarity = (
                    evidence
                    .cosine_similarity
                )


                # ------------------------------------------------
                # The candidate variable order may differ from
                # the canonical alphabetical order stored in the
                # undirected embedding pair.
                # ------------------------------------------------

                if (
                    evidence.left_column
                    ==
                    left_column
                    and
                    evidence.right_column
                    ==
                    right_column
                ):
                    left_to_right_rank = (
                        evidence
                        .left_to_right_rank
                    )


                    right_to_left_rank = (
                        evidence
                        .right_to_left_rank
                    )

                else:
                    left_to_right_rank = (
                        evidence
                        .right_to_left_rank
                    )


                    right_to_left_rank = (
                        evidence
                        .left_to_right_rank
                    )


                mutual_retrieval = (
                    evidence
                    .mutual_retrieval
                )


            if (
                classification
                ==
                "semantic_pair_in_scope"
            ):
                semantic_total += 1


                if (
                    pair_in_graph
                ):
                    semantic_covered += 1


            else:
                generic_total += 1


                if (
                    pair_in_graph
                ):
                    generic_covered += 1


        else:
            out_of_scope += 1


        records.append(
            EmbeddingShadowCandidateAudit(
                stage=
                    stage,

                analysis_id=
                    candidate.analysis_id,

                family=
                    candidate.family,

                scope=
                    candidate.scope,

                classification=
                    classification,

                dataset_id=
                    dataset_id,

                left_column=
                    left_column,

                right_column=
                    right_column,

                pair_in_embedding_graph=
                    pair_in_graph,

                cosine_similarity=
                    similarity,

                left_to_right_rank=
                    left_to_right_rank,

                right_to_left_rank=
                    right_to_left_rank,

                mutual_retrieval=
                    mutual_retrieval,
            )
        )


    semantic_coverage = (
        semantic_covered
        /
        semantic_total

        if (
            semantic_total
            >
            0
        )

        else None
    )


    generic_coverage = (
        generic_covered
        /
        generic_total

        if (
            generic_total
            >
            0
        )

        else None
    )


    summary = (
        EmbeddingShadowStageSummary(
            stage=
                stage,

            candidate_count=
                len(
                    report.candidates
                ),

            semantic_pair_in_scope_count=
                semantic_total,

            semantic_pair_covered_count=
                semantic_covered,

            semantic_pair_coverage=
                semantic_coverage,

            generic_pair_observation_count=
                generic_total,

            generic_pair_covered_count=
                generic_covered,

            generic_pair_coverage=
                generic_coverage,

            out_of_scope_count=
                out_of_scope,

            classification_counts=
                dict(
                    classification_counts
                ),
        )
    )


    return (
        summary,
        records,
    )


# ============================================================
# PUBLIC SHADOW AUDIT
#
# CRITICAL CONTRACT:
#
# This function:
#
# - does NOT mutate Discovery reports;
# - does NOT filter Discovery candidates;
# - does NOT alter priority scores;
# - does NOT authorize any analysis;
# - does NOT modify validation outcomes.
#
# It only observes whether selected candidate pairs appear in
# the accepted embedding candidate graph.
# ============================================================

def audit_embedding_discovery_shadow(
    *,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
    unvalidated_report,
    validated_report,
    model: str = (
        DEFAULT_EMBEDDING_MODEL
    ),
    top_k: int = (
        DEFAULT_EMBEDDING_TOP_K
    ),
) -> EmbeddingDiscoveryShadowAuditResult:
    if (
        top_k
        <
        1
    ):
        raise ValueError(
            "top_k must be at least 1."
        )


    dataset_columns = (
        _build_dataset_column_index(
            datasets
        )
    )


    required_dataset_ids = (
        _collect_pair_dataset_ids(
            unvalidated_report=
                unvalidated_report,

            validated_report=
                validated_report,
        )
    )


    (
        graph_results,
        pair_indexes,
    ) = (
        _build_embedding_graphs(
            dataset_columns=
                dataset_columns,

            required_dataset_ids=
                required_dataset_ids,

            model=
                model,

            top_k=
                top_k,
        )
    )


    (
        unvalidated_summary,
        unvalidated_records,
    ) = (
        _audit_stage(
            stage=
                "unvalidated",

            report=
                unvalidated_report,

            pair_indexes=
                pair_indexes,
        )
    )


    (
        validated_summary,
        validated_records,
    ) = (
        _audit_stage(
            stage=
                "validated",

            report=
                validated_report,

            pair_indexes=
                pair_indexes,
        )
    )


    dataset_summaries = [
        EmbeddingShadowDatasetSummary(
            dataset_id=
                dataset_id,

            column_count=
                result.column_count,

            all_possible_pair_count=
                result.all_possible_pair_count,

            embedding_candidate_pair_count=
                result.candidate_pair_count,

            pair_reduction_ratio=
                result.reduction_ratio,

            top_k=
                result.top_k,
        )

        for dataset_id, result
        in sorted(
            graph_results.items()
        )
    ]


    return (
        EmbeddingDiscoveryShadowAuditResult(
            embedding_model=
                model,

            top_k=
                top_k,

            dataset_summaries=
                dataset_summaries,

            unvalidated_summary=
                unvalidated_summary,

            validated_summary=
                validated_summary,

            candidate_audits=(
                unvalidated_records
                +
                validated_records
            ),

            semantic_pair_families=
                sorted(
                    SEMANTIC_PAIR_FAMILIES
                ),

            generic_pair_observation_families=
                sorted(
                    GENERIC_PAIR_OBSERVATION_FAMILIES
                ),

            audit_version=
                EMBEDDING_DISCOVERY_SHADOW_AUDIT_VERSION,
        )
    )
