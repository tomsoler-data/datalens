from __future__ import annotations

from typing import (
    Any,
)

import numpy as np
import pandas as pd

from app.security.llm_payload import (
    LLMPayloadClass,
    classified_llm_chat,
)

from app.ai.provider import (
    DEFAULT_MODEL,
    client,
)

from app.semantics.graph_schemas import (
    SemanticGraphAbstention,
    SemanticGraphReport,
    SemanticPairRelationDraft,
    SemanticRelationEdge,
)

from app.semantics.schemas import (
    DatasetSemanticProfile,
)


# ============================================================
# VERSION
# ============================================================

SEMANTIC_GRAPH_RULE_VERSION = (
    "semantic_graph_v0.1"
)


SEMANTIC_PAIR_ADJUDICATOR_RULE_VERSION = (
    "semantic_pair_adjudicator_v0.1"
)


MAX_PAIR_ATTEMPTS = 2


# ============================================================
# SYSTEM PROMPT
#
# The model evaluates RELATIONS ONLY.
#
# It is deliberately forbidden from rewriting node semantic
# attributes or declaring arithmetic operations safe.
# ============================================================

PAIR_SYSTEM_PROMPT = """
You are DataLens' semantic relationship adjudicator.

You receive TWO columns from ONE dataset.

Your responsibility is strictly relational.

You must determine only:

1. same_domain_family

   True when both columns belong to the same broad business
   or operational system.

   Subdomains may differ.

   For example, workforce, billing, operations and service
   quality may belong to one broader business domain.

2. same_quantity_family

   True only when both columns represent different states,
   stages or observations of the same underlying measurable
   quantity.

   Ask:

   "Ignoring state and unit representation, are the two
   variables measuring the same underlying thing?"

   Examples of the TYPE of reasoning required:

   - submitted requests vs resolved tickets can represent
     two states of operational case volume;

   - scheduled employees vs employees currently available
     can represent two states of workforce count;

   - quoted amount vs final invoiced amount can represent
     two states of a commercial monetary amount.

   In contrast:

   - response duration vs SLA percentage are related but are
     NOT the same measurable quantity;

   - invoice amount vs staff count are NOT the same measurable
     quantity.

STRICT SAFETY RULES:

- Do NOT decide whether units are compatible.
- Do NOT decide whether subtraction is valid.
- Do NOT decide whether a derived gap is safe.
- Do NOT change measure_kind.
- Do NOT infer state.
- Do NOT rewrite either column profile.

Those decisions belong to other deterministic DataLens
components.

Be conservative.

If the two quantities are merely associated or causally
related, same_quantity_family must be false.

Return only the structured response.
""".strip()


# ============================================================
# VALUE PROFILE
# ============================================================

def build_value_profile(
    series: pd.Series,
) -> dict[
    str,
    Any,
]:
    non_null = (
        series
        .dropna()
    )


    result: dict[
        str,
        Any,
    ] = {
        "row_count":
            int(
                len(
                    series
                )
            ),

        "non_null_count":
            int(
                len(
                    non_null
                )
            ),

        "unique_count":
            int(
                non_null.nunique()
            ),
    }


    numeric = (
        pd.to_numeric(
            non_null,
            errors=
                "coerce",
        )
        .dropna()
    )


    if (
        len(
            numeric
        )
        ==
        0
    ):
        result.update(
            {
                "numeric":
                    False,

                "min":
                    None,

                "max":
                    None,

                "mean":
                    None,

                "integer_like":
                    False,
            }
        )


        return result


    values = (
        numeric
        .astype(
            float
        )
        .to_numpy()
    )


    result.update(
        {
            "numeric":
                True,

            "min":
                round(
                    float(
                        np.min(
                            values
                        )
                    ),
                    6,
                ),

            "max":
                round(
                    float(
                        np.max(
                            values
                        )
                    ),
                    6,
                ),

            "mean":
                round(
                    float(
                        np.mean(
                            values
                        )
                    ),
                    6,
                ),

            "integer_like":
                bool(
                    np.allclose(
                        values,
                        np.round(
                            values
                        ),
                        atol=
                            1e-9,
                    )
                ),
        }
    )


    return result


# ============================================================
# PROFILE PAYLOAD
# ============================================================

def build_graph_profile_payload(
    *,
    profile,
    dataframe: pd.DataFrame,
) -> dict[
    str,
    Any,
]:
    return {
        "column":
            profile.column,

        "concept":
            profile.concept,

        "domain":
            profile.domain,

        "semantic_group":
            profile.semantic_group,

        "variant":
            profile.variant,

        "measure_kind":
            profile.measure_kind,

        "unit_kind":
            profile.unit_kind,

        "quantity_dimension":
            profile.quantity_dimension,

        "quantity_unit":
            profile.quantity_unit,

        "entity_role":
            profile.entity_role,

        "confidence":
            profile.confidence,

        "value_profile":
            build_value_profile(
                dataframe[
                    profile.column
                ]
            ),
    }


# ============================================================
# COLUMN INDEX
# ============================================================

def build_column_profile_index(
    profile: DatasetSemanticProfile,
) -> dict[
    str,
    Any,
]:
    return {
        column.column:
            column

        for column
        in profile.columns
    }


# ============================================================
# VALIDATE CANDIDATE PAIRS
# ============================================================

def normalize_candidate_pairs(
    *,
    dataset_profile: DatasetSemanticProfile,
    candidate_pairs: list[
        tuple[
            str,
            str,
        ]
    ],
) -> list[
    tuple[
        str,
        str,
    ]
]:
    available_columns = {
        profile.column

        for profile
        in dataset_profile.columns
    }


    resolved: list[
        tuple[
            str,
            str,
        ]
    ] = []


    seen: set[
        tuple[
            str,
            str,
        ]
    ] = set()


    for (
        left_column,
        right_column,
    ) in candidate_pairs:
        if (
            left_column
            not in available_columns
        ):
            raise ValueError(
                "Unknown semantic graph column: "
                f"{left_column}"
            )


        if (
            right_column
            not in available_columns
        ):
            raise ValueError(
                "Unknown semantic graph column: "
                f"{right_column}"
            )


        if (
            left_column
            ==
            right_column
        ):
            raise ValueError(
                "Semantic graph self-pairs are not allowed: "
                f"{left_column}"
            )


        canonical_pair = tuple(
            sorted(
                (
                    left_column,
                    right_column,
                )
            )
        )


        if (
            canonical_pair
            in seen
        ):
            continue


        seen.add(
            canonical_pair
        )


        resolved.append(
            (
                left_column,
                right_column,
            )
        )


    return resolved


# ============================================================
# SINGLE PAIR ADJUDICATION
# ============================================================

def adjudicate_semantic_pair(
    *,
    dataset_profile: DatasetSemanticProfile,
    dataframe: pd.DataFrame,
    left_column: str,
    right_column: str,
    model: str = DEFAULT_MODEL,
) -> SemanticRelationEdge:
    index = (
        build_column_profile_index(
            dataset_profile
        )
    )


    if (
        left_column
        not in index
        or
        right_column
        not in index
    ):
        raise ValueError(
            "Pair columns must exist in the dataset "
            "semantic profile."
        )


    left_profile = (
        index[
            left_column
        ]
    )


    right_profile = (
        index[
            right_column
        ]
    )


    payload = {
        "dataset_id":
            dataset_profile.dataset_id,

        "filename":
            dataset_profile.filename,

        "left":
            build_graph_profile_payload(
                profile=
                    left_profile,

                dataframe=
                    dataframe,
            ),

        "right":
            build_graph_profile_payload(
                profile=
                    right_profile,

                dataframe=
                    dataframe,
            ),
    }


    user_prompt = (
        "Evaluate the semantic relationship between these "
        "two columns.\n\n"
        f"{payload}\n\n"
        "Return only the structured relationship decision."
    )


    last_error: Exception | None = None


    for attempt in range(
        MAX_PAIR_ATTEMPTS
    ):
        try:
            response = (
                classified_llm_chat(
            client,
            payload_class=(
                LLMPayloadClass
                .DETERMINISTIC_EVIDENCE
            ),
                    model=
                        model,

                    messages=[
                        {
                            "role":
                                "system",

                            "content":
                                PAIR_SYSTEM_PROMPT,
                        },
                        {
                            "role":
                                "user",

                            "content":
                                user_prompt,
                        },
                    ],

                    format=(
                        SemanticPairRelationDraft
                        .model_json_schema()
                    ),

                    options={
                        "temperature":
                            0.0,
                    },
                )
            )


            draft = (
                SemanticPairRelationDraft
                .model_validate_json(
                    response.message.content
                )
            )


            return SemanticRelationEdge(
                dataset_id=
                    dataset_profile.dataset_id,

                filename=
                    dataset_profile.filename,

                left_column=
                    left_column,

                right_column=
                    right_column,

                same_domain_family=
                    draft.same_domain_family,

                same_quantity_family=
                    draft.same_quantity_family,

                confidence=
                    draft.confidence,

                reason=
                    draft.reason,

                source=
                    "llm",

                relation_rule_version=
                    SEMANTIC_PAIR_ADJUDICATOR_RULE_VERSION,
            )


        except Exception as error:
            last_error = error


    raise RuntimeError(
        "Semantic pair adjudication failed after "
        f"{MAX_PAIR_ATTEMPTS} attempts. "
        f"Last error: {last_error}"
    )


# ============================================================
# GRAPH BUILDER
#
# IMPORTANT:
#
# The graph does NOT modify ColumnSemanticProfile.
#
# It is a parallel evidence structure.
# ============================================================

def build_semantic_graph(
    *,
    dataset_profile: DatasetSemanticProfile,
    dataframe: pd.DataFrame,
    candidate_pairs: list[
        tuple[
            str,
            str,
        ]
    ],
    model: str = DEFAULT_MODEL,
) -> SemanticGraphReport:
    pairs = (
        normalize_candidate_pairs(
            dataset_profile=
                dataset_profile,

            candidate_pairs=
                candidate_pairs,
        )
    )


    edges: list[
        SemanticRelationEdge
    ] = []


    abstentions: list[
        SemanticGraphAbstention
    ] = []


    for (
        left_column,
        right_column,
    ) in pairs:
        try:
            edge = (
                adjudicate_semantic_pair(
                    dataset_profile=
                        dataset_profile,

                    dataframe=
                        dataframe,

                    left_column=
                        left_column,

                    right_column=
                        right_column,

                    model=
                        model,
                )
            )


            edges.append(
                edge
            )


        except Exception as error:
            # ------------------------------------------------
            # Fail closed.
            #
            # No relationship is invented when the semantic
            # model fails.
            # ------------------------------------------------

            abstentions.append(
                SemanticGraphAbstention(
                    dataset_id=
                        dataset_profile.dataset_id,

                    filename=
                        dataset_profile.filename,

                    left_column=
                        left_column,

                    right_column=
                        right_column,

                    reason=
                        str(
                            error
                        ),
                )
            )


    return SemanticGraphReport(
        dataset_id=
            dataset_profile.dataset_id,

        filename=
            dataset_profile.filename,

        candidate_pair_count=
            len(
                pairs
            ),

        adjudicated_pair_count=
            len(
                edges
            ),

        abstention_count=
            len(
                abstentions
            ),

        same_domain_edge_count=
            sum(
                edge.same_domain_family

                for edge
                in edges
            ),

        same_quantity_edge_count=
            sum(
                edge.same_quantity_family

                for edge
                in edges
            ),

        edges=
            edges,

        abstentions=
            abstentions,

        graph_rule_version=
            SEMANTIC_GRAPH_RULE_VERSION,
    )


# ============================================================
# LOOKUP HELPERS
# ============================================================

def find_semantic_edge(
    *,
    graph: SemanticGraphReport,
    left_column: str,
    right_column: str,
) -> SemanticRelationEdge | None:
    target = {
        left_column,
        right_column,
    }


    for edge in (
        graph.edges
    ):
        if (
            {
                edge.left_column,
                edge.right_column,
            }
            ==
            target
        ):
            return edge


    return None
