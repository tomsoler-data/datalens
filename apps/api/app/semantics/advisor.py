from __future__ import annotations

from typing import (
    Any,
)

from app.semantics.advisor_schemas import (
    SemanticCandidateAdvice,
    SemanticDiscoveryAdviceReport,
)

from app.semantics.comparator import (
    compare_semantic_profiles,
)

from app.semantics.schemas import (
    ColumnSemanticProfile,
    DatasetSemanticProfile,
)


# ============================================================
# CONFIGURATION
# ============================================================

PAIRWISE_FAMILIES = {
    "quantitative_association",
    "categorical_association",
    "derived_gap",
}


MEASURE_PAIR_FAMILIES = {
    "quantitative_association",
    "derived_gap",
}


# ============================================================
# GENERIC OBJECT SERIALIZATION
# ============================================================

def object_to_payload(
    value: Any,
) -> Any:
    if hasattr(
        value,
        "model_dump",
    ):
        return value.model_dump(
            mode="python"
        )


    if isinstance(
        value,
        dict,
    ):
        return value


    if hasattr(
        value,
        "__dict__",
    ):
        return vars(
            value
        )


    return value


# ============================================================
# RECURSIVE STRING EXTRACTION
# ============================================================

def collect_strings(
    value: Any,
) -> list[
    str
]:
    result: list[
        str
    ] = []


    if value is None:
        return result


    if isinstance(
        value,
        str,
    ):
        return [
            value
        ]


    if isinstance(
        value,
        dict,
    ):
        for key, child in (
            value.items()
        ):
            result.extend(
                collect_strings(
                    key
                )
            )

            result.extend(
                collect_strings(
                    child
                )
            )


        return result


    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        for child in value:
            result.extend(
                collect_strings(
                    child
                )
            )


        return result


    if hasattr(
        value,
        "model_dump",
    ):
        return collect_strings(
            value.model_dump(
                mode="python"
            )
        )


    return result


# ============================================================
# PROFILE FLATTENING
# ============================================================

def flatten_semantic_profiles(
    semantic_profiles: list[
        DatasetSemanticProfile
    ],
) -> list[
    ColumnSemanticProfile
]:
    return [
        column_profile
        for dataset_profile
        in semantic_profiles
        for column_profile
        in dataset_profile.columns
    ]


# ============================================================
# CANDIDATE METADATA
# ============================================================

def candidate_analysis_id(
    candidate: Any,
) -> str:
    return str(
        getattr(
            candidate,
            "analysis_id",
            "",
        )
        or
        ""
    )


def candidate_family(
    candidate: Any,
) -> str:
    return str(
        getattr(
            candidate,
            "family",
            "",
        )
        or
        ""
    )


def candidate_title(
    candidate: Any,
) -> str:
    return str(
        getattr(
            candidate,
            "title",
            "",
        )
        or
        ""
    )


# ============================================================
# DATASET RESOLUTION
# ============================================================

def resolve_candidate_dataset_ids(
    *,
    strings: list[
        str
    ],
    profiles: list[
        ColumnSemanticProfile
    ],
) -> set[
    str
]:
    dataset_ids: set[
        str
    ] = set()


    string_set = set(
        strings
    )


    for profile in profiles:
        if (
            profile.dataset_id
            in string_set
            or
            profile.filename
            in string_set
        ):
            dataset_ids.add(
                profile.dataset_id
            )


    return dataset_ids


# ============================================================
# COLUMN RESOLUTION
# ============================================================

def profile_allowed_for_family(
    *,
    profile: ColumnSemanticProfile,
    family: str,
) -> bool:
    if (
        family
        in MEASURE_PAIR_FAMILIES
    ):
        return (
            profile.entity_role
            ==
            "measure"
        )


    if (
        family
        ==
        "categorical_association"
    ):
        return (
            profile.entity_role
            not in {
                "time",
                "identifier",
            }
        )


    return True


def resolve_candidate_profiles(
    *,
    candidate: Any,
    profiles: list[
        ColumnSemanticProfile
    ],
) -> list[
    ColumnSemanticProfile
]:
    payload = object_to_payload(
        candidate
    )


    strings = collect_strings(
        payload
    )


    string_set = set(
        strings
    )


    family = candidate_family(
        candidate
    )


    title = candidate_title(
        candidate
    )


    title_lower = title.lower()


    dataset_ids = (
        resolve_candidate_dataset_ids(
            strings=
                strings,

            profiles=
                profiles,
        )
    )


    eligible_profiles = [
        profile
        for profile
        in profiles
        if (
            (
                not dataset_ids
                or
                profile.dataset_id
                in dataset_ids
            )
            and
            profile_allowed_for_family(
                profile=
                    profile,

                family=
                    family,
            )
        )
    ]


    resolved: list[
        ColumnSemanticProfile
    ] = []


    seen: set[
        tuple[
            str,
            str,
        ]
    ] = set()


    # ========================================================
    # PASS 1
    #
    # Exact values contained in the candidate payload.
    # ========================================================

    for profile in eligible_profiles:
        key = (
            profile.dataset_id,
            profile.column,
        )


        if (
            profile.column
            in string_set
            and
            key
            not in seen
        ):
            seen.add(
                key
            )

            resolved.append(
                profile
            )


    # ========================================================
    # PASS 2
    #
    # Title fallback.
    # ========================================================

    if (
        len(
            resolved
        )
        <
        2
    ):
        for profile in eligible_profiles:
            key = (
                profile.dataset_id,
                profile.column,
            )


            if (
                key
                in seen
            ):
                continue


            column_lower = (
                profile.column.lower()
            )


            if (
                column_lower
                and
                column_lower
                in title_lower
            ):
                seen.add(
                    key
                )

                resolved.append(
                    profile
                )


    return resolved


# ============================================================
# SEMANTIC RULES
# ============================================================

def association_advice(
    comparison,
) -> tuple[
    float,
    str,
    list[
        str
    ],
]:
    reasons = list(
        comparison.reasons
    )


    if (
        comparison.association_novelty
        ==
        "high"
        and
        comparison.redundancy_risk
        ==
        "low"
    ):
        reasons.append(
            (
                "Le candidat reçoit un bonus "
                "sémantique car il rapproche des "
                "concepts distincts sans forte "
                "redondance détectée."
            )
        )


        return (
            8.0,
            "boost",
            reasons,
        )


    if (
        comparison.association_novelty
        ==
        "low"
        and
        comparison.redundancy_risk
        ==
        "high"
    ):
        reasons.append(
            (
                "Le candidat reçoit une pénalité "
                "sémantique car l'association relie "
                "des mesures conceptuellement très "
                "proches."
            )
        )


        return (
            -10.0,
            "penalize",
            reasons,
        )


    if (
        comparison.association_novelty
        ==
        "medium"
        and
        comparison.redundancy_risk
        !=
        "high"
    ):
        reasons.append(
            (
                "Le candidat apporte une nouveauté "
                "sémantique modérée."
            )
        )


        return (
            3.0,
            "boost",
            reasons,
        )


    return (
        0.0,
        "neutral",
        reasons,
    )


def gap_advice(
    comparison,
) -> tuple[
    float,
    str,
    list[
        str
    ],
]:
    reasons = list(
        comparison.reasons
    )


    if (
        comparison.derived_gap_compatible
    ):
        reasons.append(
            (
                "Le candidat de type écart est "
                "sémantiquement défendable : les "
                "mesures sont liées, distinctes et "
                "leurs unités sont compatibles."
            )
        )


        return (
            8.0,
            "boost",
            reasons,
        )


    reasons.append(
        (
            "Le candidat de type écart devrait être "
            "revu avant toute utilisation : la "
            "compatibilité sémantique ou d'unité "
            "n'est pas suffisamment établie."
        )
    )


    return (
        -20.0,
        "review",
        reasons,
    )


# ============================================================
# SINGLE CANDIDATE
# ============================================================

def advise_candidate(
    *,
    candidate: Any,
    profiles: list[
        ColumnSemanticProfile
    ],
) -> SemanticCandidateAdvice:
    analysis_id = (
        candidate_analysis_id(
            candidate
        )
    )


    family = candidate_family(
        candidate
    )


    title = candidate_title(
        candidate
    )


    # ========================================================
    # V0.1 SCOPE
    # ========================================================

    if (
        family
        not in PAIRWISE_FAMILIES
    ):
        return (
            SemanticCandidateAdvice(
                analysis_id=
                    analysis_id,

                title=
                    title,

                family=
                    family,

                status=
                    "not_applicable",

                semantic_score_delta=
                    0.0,

                decision=
                    "neutral",

                reasons=[
                    (
                        "Semantic Discovery Advisor "
                        "v0.1 n'applique pas encore de "
                        "règle à cette famille "
                        "d'analyse."
                    )
                ],
            )
        )


    resolved = (
        resolve_candidate_profiles(
            candidate=
                candidate,

            profiles=
                profiles,
        )
    )


    referenced_dataset_ids = list(
        dict.fromkeys(
            profile.dataset_id
            for profile
            in resolved
        )
    )


    referenced_columns = list(
        dict.fromkeys(
            profile.column
            for profile
            in resolved
        )
    )


    if (
        len(
            resolved
        )
        !=
        2
    ):
        return (
            SemanticCandidateAdvice(
                analysis_id=
                    analysis_id,

                title=
                    title,

                family=
                    family,

                status=
                    "insufficient_semantic_context",

                referenced_dataset_ids=
                    referenced_dataset_ids,

                referenced_columns=
                    referenced_columns,

                semantic_score_delta=
                    0.0,

                decision=
                    "neutral",

                reasons=[
                    (
                        "Impossible de résoudre "
                        "exactement deux colonnes "
                        "sémantiques pour ce candidat. "
                        "Aucune décision n'est prise."
                    )
                ],
            )
        )


    left = resolved[
        0
    ]


    right = resolved[
        1
    ]


    comparison = (
        compare_semantic_profiles(
            left,
            right,
        )
    )


    if (
        family
        ==
        "derived_gap"
    ):
        (
            score_delta,
            decision,
            reasons,
        ) = gap_advice(
            comparison
        )


    else:
        (
            score_delta,
            decision,
            reasons,
        ) = association_advice(
            comparison
        )


    return (
        SemanticCandidateAdvice(
            analysis_id=
                analysis_id,

            title=
                title,

            family=
                family,

            status=
                "annotated",

            referenced_dataset_ids=
                referenced_dataset_ids,

            referenced_columns=
                referenced_columns,

            comparison=
                comparison,

            semantic_score_delta=
                score_delta,

            decision=
                decision,

            reasons=
                reasons,
        )
    )


# ============================================================
# PUBLIC DISCOVERY ADVISOR
# ============================================================

def advise_discovery_semantics(
    *,
    discovery: Any,
    semantic_profiles: list[
        DatasetSemanticProfile
    ],
) -> SemanticDiscoveryAdviceReport:
    candidates = list(
        getattr(
            discovery,
            "candidates",
            [],
        )
    )


    profiles = (
        flatten_semantic_profiles(
            semantic_profiles
        )
    )


    advice = [
        advise_candidate(
            candidate=
                candidate,

            profiles=
                profiles,
        )
        for candidate
        in candidates
    ]


    annotated_count = sum(
        item.status
        ==
        "annotated"
        for item
        in advice
    )


    boosted_count = sum(
        item.decision
        ==
        "boost"
        for item
        in advice
    )


    penalized_count = sum(
        item.decision
        ==
        "penalize"
        for item
        in advice
    )


    review_count = sum(
        item.decision
        ==
        "review"
        for item
        in advice
    )


    neutral_count = sum(
        item.decision
        ==
        "neutral"
        for item
        in advice
    )


    not_applicable_count = sum(
        item.status
        ==
        "not_applicable"
        for item
        in advice
    )


    insufficient_context_count = sum(
        item.status
        ==
        "insufficient_semantic_context"
        for item
        in advice
    )


    return (
        SemanticDiscoveryAdviceReport(
            candidate_count=
                len(
                    candidates
                ),

            annotated_count=
                annotated_count,

            boosted_count=
                boosted_count,

            penalized_count=
                penalized_count,

            review_count=
                review_count,

            neutral_count=
                neutral_count,

            not_applicable_count=
                not_applicable_count,

            insufficient_context_count=
                insufficient_context_count,

            advice=
                advice,
        )
    )


# ============================================================
# LOOKUP
# ============================================================

def advice_by_analysis_id(
    report: SemanticDiscoveryAdviceReport,
) -> dict[
    str,
    SemanticCandidateAdvice,
]:
    return {
        item.analysis_id:
            item
        for item
        in report.advice
    }
