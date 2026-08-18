from __future__ import annotations

import math

from typing import (
    Any,
)

from app.execution.cross_schemas import (
    CrossDatasetExecutedAnalysis,
    CrossDatasetExecutionReport,
)

from app.ranking.schemas import (
    AnalysisRankingReport,
    RankedAnalysis,
)


# ============================================================
# THRESHOLDS
# ============================================================

KEY_FINDING_THRESHOLD = 78.0

SUPPORTING_FINDING_THRESHOLD = 60.0


BLOCKING_STATUSES = {
    "requires_alignment",
    "failed",
    "skipped",
}


# ============================================================
# NUMERIC HELPERS
# ============================================================

def safe_float(
    value: Any,
) -> float | None:
    if value is None:
        return None


    try:
        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None


    if not math.isfinite(
        result
    ):
        return None


    return result


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> float:
    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


# ============================================================
# EXECUTION STATE
# ============================================================

def is_blocked_result(
    result: CrossDatasetExecutedAnalysis,
) -> bool:
    return (
        result.execution_status
        in BLOCKING_STATUSES
    )


# ============================================================
# ASSOCIATION VALUE
# ============================================================

def get_primary_association(
    result: CrossDatasetExecutedAnalysis,
) -> float | None:
    if is_blocked_result(
        result
    ):
        return None


    metrics = (
        result.metrics
    )


    value = safe_float(
        metrics.get(
            "median_period_spearman"
        )
    )


    if value is not None:
        return value


    value = safe_float(
        metrics.get(
            "spearman"
        )
    )


    if value is not None:
        return value


    return safe_float(
        metrics.get(
            "overall_preliminary_spearman"
        )
    )


# ============================================================
# DIRECTION
# ============================================================

def determine_direction(
    result: CrossDatasetExecutedAnalysis,
    coefficient: float | None,
) -> str:
    if is_blocked_result(
        result
    ):
        return "unknown"


    metrics = (
        result.metrics
    )


    positive_periods = int(
        metrics.get(
            "positive_periods",
            0,
        )
        or
        0
    )

    negative_periods = int(
        metrics.get(
            "negative_periods",
            0,
        )
        or
        0
    )


    if (
        positive_periods
        >
        0
        and
        negative_periods
        >
        0
    ):
        return "mixed"


    if (
        positive_periods
        >
        0
        and
        negative_periods
        ==
        0
    ):
        return "positive"


    if (
        negative_periods
        >
        0
        and
        positive_periods
        ==
        0
    ):
        return "negative"


    if coefficient is None:
        return "unknown"


    if coefficient > 0:
        return "positive"


    if coefficient < 0:
        return "negative"


    return "unknown"


# ============================================================
# ASSOCIATION STRENGTH
# ============================================================

def determine_strength(
    coefficient: float | None,
) -> str:
    if coefficient is None:
        return "unknown"


    magnitude = abs(
        coefficient
    )


    if magnitude < 0.10:
        return "negligible"


    if magnitude < 0.30:
        return "weak"


    if magnitude < 0.50:
        return "moderate"


    if magnitude < 0.70:
        return "moderately_strong"


    return "strong"


# ============================================================
# EFFECT
# ============================================================

def calculate_effect_score(
    coefficient: float | None,
) -> float:
    if coefficient is None:
        return 0.0


    return clamp(
        abs(
            coefficient
        )
        *
        100.0
    )


# ============================================================
# COVERAGE
# ============================================================

def calculate_coverage_score(
    result: CrossDatasetExecutedAnalysis,
) -> float:
    if is_blocked_result(
        result
    ):
        return 0.0


    left = clamp(
        float(
            result.left_key_coverage
        )
        *
        100.0
    )


    right = clamp(
        float(
            result.right_key_coverage
        )
        *
        100.0
    )


    if (
        left
        <=
        0.0
        or
        right
        <=
        0.0
    ):
        return 0.0


    return math.sqrt(
        left
        *
        right
    )


# ============================================================
# CONSISTENCY
# ============================================================

def calculate_consistency(
    result: CrossDatasetExecutedAnalysis,
) -> tuple[
    float,
    float | None,
]:
    if is_blocked_result(
        result
    ):
        return (
            0.0,
            None,
        )


    metrics = (
        result.metrics
    )


    period_count = int(
        metrics.get(
            "period_count_analysed",
            0,
        )
        or
        0
    )


    positive = int(
        metrics.get(
            "positive_periods",
            0,
        )
        or
        0
    )

    negative = int(
        metrics.get(
            "negative_periods",
            0,
        )
        or
        0
    )


    if period_count > 0:
        dominant = max(
            positive,
            negative,
        )


        ratio = (
            dominant
            /
            period_count
        )


        return (
            clamp(
                ratio
                *
                100.0
            ),
            ratio,
        )


    if (
        result.execution_status
        ==
        "complete"
    ):
        return (
            50.0,
            None,
        )


    return (
        0.0,
        None,
    )


# ============================================================
# SAMPLE SIZE
# ============================================================

def calculate_sample_score(
    result: CrossDatasetExecutedAnalysis,
) -> float:
    if is_blocked_result(
        result
    ):
        return 0.0


    n = max(
        0,
        int(
            result.joined_rows
        ),
    )


    if n == 0:
        return 0.0


    denominator = math.log10(
        3001
    )


    score = (
        math.log10(
            n
            +
            1
        )
        /
        denominator
        *
        100.0
    )


    return clamp(
        score
    )


# ============================================================
# RELATIONSHIP CONFIDENCE
# ============================================================

def calculate_relationship_component(
    result: CrossDatasetExecutedAnalysis,
) -> float:
    score = safe_float(
        result.relationship_score
    )


    if score is None:
        return 0.0


    return clamp(
        score
    )


# ============================================================
# ROBUSTNESS
# ============================================================

def calculate_robustness_score(
    result: CrossDatasetExecutedAnalysis,
    *,
    coverage_score: float,
    consistency_score: float,
    sample_score: float,
    relationship_score: float,
) -> float:
    """
    Robustness describes an executed result.

    A blocked candidate can have a promising
    relationship score, but it does not yet have
    empirical analytical robustness.
    """

    if is_blocked_result(
        result
    ):
        return 0.0


    score = (
        coverage_score
        *
        0.40
        +
        consistency_score
        *
        0.30
        +
        sample_score
        *
        0.15
        +
        relationship_score
        *
        0.15
    )


    return clamp(
        score
    )


# ============================================================
# SCORE ADJUSTMENTS
# ============================================================

def calculate_adjustment(
    result: CrossDatasetExecutedAnalysis,
    *,
    coverage_score: float,
) -> float:
    if is_blocked_result(
        result
    ):
        return 0.0


    adjustment = 0.0


    if (
        result.relationship_status
        ==
        "validated"
    ):
        adjustment += 5.0


    if (
        result.relationship_status
        ==
        "partial"
    ):
        adjustment -= 3.0


    if coverage_score < 50.0:
        adjustment -= 10.0

    elif coverage_score < 65.0:
        adjustment -= 6.0

    elif coverage_score < 75.0:
        adjustment -= 3.0


    return adjustment


# ============================================================
# INTERESTINGNESS
# ============================================================

def calculate_interestingness_score(
    result: CrossDatasetExecutedAnalysis,
) -> tuple[
    float,
    dict[
        str,
        float,
    ],
]:
    relationship_score = (
        calculate_relationship_component(
            result
        )
    )


    if is_blocked_result(
        result
    ):
        return (
            0.0,

            {
                "effect_score":
                    0.0,

                "coverage_score":
                    0.0,

                "consistency_score":
                    0.0,

                "sample_score":
                    0.0,

                "relationship_score":
                    round(
                        relationship_score,
                        2,
                    ),

                "adjustment":
                    0.0,
            },
        )


    coefficient = (
        get_primary_association(
            result
        )
    )


    effect_score = (
        calculate_effect_score(
            coefficient
        )
    )


    coverage_score = (
        calculate_coverage_score(
            result
        )
    )


    (
        consistency_score,
        _,
    ) = calculate_consistency(
        result
    )


    sample_score = (
        calculate_sample_score(
            result
        )
    )


    score = (
        effect_score
        *
        0.35
        +
        consistency_score
        *
        0.20
        +
        coverage_score
        *
        0.20
        +
        sample_score
        *
        0.10
        +
        relationship_score
        *
        0.10
    )


    adjustment = (
        calculate_adjustment(
            result,

            coverage_score=
                coverage_score,
        )
    )


    score += adjustment


    return (
        round(
            clamp(
                score
            ),
            2,
        ),

        {
            "effect_score":
                round(
                    effect_score,
                    2,
                ),

            "coverage_score":
                round(
                    coverage_score,
                    2,
                ),

            "consistency_score":
                round(
                    consistency_score,
                    2,
                ),

            "sample_score":
                round(
                    sample_score,
                    2,
                ),

            "relationship_score":
                round(
                    relationship_score,
                    2,
                ),

            "adjustment":
                round(
                    adjustment,
                    2,
                ),
        },
    )


# ============================================================
# TIER
# ============================================================

def determine_tier(
    result: CrossDatasetExecutedAnalysis,
    score: float,
) -> str:
    if is_blocked_result(
        result
    ):
        return "blocked"


    if (
        score
        >=
        KEY_FINDING_THRESHOLD
    ):
        return "key_finding"


    if (
        score
        >=
        SUPPORTING_FINDING_THRESHOLD
    ):
        return "supporting_finding"


    return "supplementary"


# ============================================================
# IMPORTANCE REASONS
# ============================================================

def build_importance_reasons(
    result: CrossDatasetExecutedAnalysis,
    *,
    coefficient: float | None,
    coverage_score: float,
    consistency_score: float,
    period_count: int,
) -> list[
    str
]:
    if is_blocked_result(
        result
    ):
        return []


    reasons: list[
        str
    ] = []


    if coefficient is not None:
        magnitude = abs(
            coefficient
        )


        if magnitude >= 0.70:
            reasons.append(
                (
                    "L'association observée est "
                    "forte en valeur absolue."
                )
            )

        elif magnitude >= 0.50:
            reasons.append(
                (
                    "L'association observée est "
                    "d'une amplitude notable."
                )
            )

        elif magnitude >= 0.30:
            reasons.append(
                (
                    "Une association modérée est "
                    "visible dans les données."
                )
            )


    if (
        period_count
        >=
        3
        and
        consistency_score
        >=
        90.0
    ):
        reasons.append(
            (
                "Le sens de l'association est "
                "très cohérent entre les périodes "
                "analysées."
            )
        )


    if coverage_score >= 85.0:
        reasons.append(
            (
                "La couverture des clés appariées "
                "est élevée dans les deux datasets."
            )
        )

    elif coverage_score < 65.0:
        reasons.append(
            (
                "L'association observée doit être "
                "interprétée avec prudence en "
                "raison d'une couverture limitée."
            )
        )


    if (
        result.joined_rows
        >=
        1000
    ):
        reasons.append(
            (
                "L'analyse repose sur un nombre "
                "important d'observations "
                "appariées."
            )
        )


    return reasons


# ============================================================
# BLOCKED REASONS
# ============================================================

def build_blocked_reasons(
    result: CrossDatasetExecutedAnalysis,
) -> list[
    str
]:
    reasons: list[
        str
    ] = []


    if (
        result.execution_status
        ==
        "requires_alignment"
    ):
        reasons.append(
            (
                "L'analyse a été identifiée comme "
                "potentiellement pertinente, mais "
                "elle n'a pas été exécutée car le "
                "grain des datasets n'est pas "
                "encore suffisamment compatible."
            )
        )


    elif (
        result.execution_status
        ==
        "skipped"
    ):
        reasons.append(
            (
                "L'analyse a été examinée mais "
                "n'a pas fourni suffisamment "
                "d'observations exploitables pour "
                "être exécutée."
            )
        )


    elif (
        result.execution_status
        ==
        "failed"
    ):
        reasons.append(
            (
                "L'analyse n'a pas pu être "
                "exécutée correctement."
            )
        )


    return reasons


# ============================================================
# CAVEATS
# ============================================================

def build_caveats(
    result: CrossDatasetExecutedAnalysis,
    *,
    coverage_score: float,
) -> list[
    str
]:
    caveats: list[
        str
    ] = []


    if is_blocked_result(
        result
    ):
        if (
            result.execution_status
            ==
            "requires_alignment"
        ):
            caveats.append(
                (
                    "Le grain des datasets n'est "
                    "pas encore suffisamment "
                    "compatible pour exécuter cette "
                    "analyse de manière sûre."
                )
            )


        caveats.extend(
            result.limitations
        )


        caveats.extend(
            result.warnings
        )


        return list(
            dict.fromkeys(
                caveats
            )
        )


    if (
        result.execution_status
        ==
        "descriptive_only"
    ):
        caveats.append(
            (
                "Le résultat est exploratoire et "
                "descriptif ; il ne constitue pas "
                "un test causal ou confirmatoire."
            )
        )


    if (
        result.metrics.get(
            "panel_structure"
        )
    ):
        caveats.append(
            (
                "Les mêmes entités sont observées "
                "à plusieurs périodes ; les lignes "
                "ne doivent pas être considérées "
                "comme indépendantes."
            )
        )


    if (
        result.relationship_status
        ==
        "partial"
    ):
        caveats.append(
            (
                "La relation repose sur un "
                "sous-ensemble apparié et ne "
                "couvre pas toutes les clés "
                "disponibles."
            )
        )


    if coverage_score < 65.0:
        caveats.append(
            (
                "La couverture limitée réduit la "
                "généralisabilité de ce constat."
            )
        )


    caveats.extend(
        result.limitations
    )


    return list(
        dict.fromkeys(
            caveats
        )
    )


# ============================================================
# RANK ONE RESULT
# ============================================================

def rank_cross_dataset_result(
    result: CrossDatasetExecutedAnalysis,
) -> RankedAnalysis:
    coefficient = (
        get_primary_association(
            result
        )
    )


    (
        score,
        components,
    ) = calculate_interestingness_score(
        result
    )


    (
        consistency_score,
        consistency_ratio,
    ) = calculate_consistency(
        result
    )


    coverage_score = (
        components[
            "coverage_score"
        ]
    )


    sample_score = (
        components[
            "sample_score"
        ]
    )


    relationship_component = (
        components[
            "relationship_score"
        ]
    )


    robustness_score = (
        calculate_robustness_score(
            result,

            coverage_score=
                coverage_score,

            consistency_score=
                consistency_score,

            sample_score=
                sample_score,

            relationship_score=
                relationship_component,
        )
    )


    period_count = int(
        result.metrics.get(
            "period_count_analysed",
            0,
        )
        or
        0
    )


    if is_blocked_result(
        result
    ):
        reasons = (
            build_blocked_reasons(
                result
            )
        )

    else:
        reasons = (
            build_importance_reasons(
                result,

                coefficient=
                    coefficient,

                coverage_score=
                    coverage_score,

                consistency_score=
                    consistency_score,

                period_count=
                    period_count,
            )
        )


    return RankedAnalysis(
        rank=0,

        analysis_id=
            result.analysis_id,

        title=
            result.title,

        scope=
            "cross_dataset",

        family=
            result.family,

        execution_status=
            result.execution_status,

        interestingness_score=
            score,

        tier=
            determine_tier(
                result,
                score,
            ),

        direction=
            determine_direction(
                result,
                coefficient,
            ),

        association_strength=
            determine_strength(
                coefficient
            ),

        effect_score=
            round(
                components[
                    "effect_score"
                ],
                2,
            ),

        coverage_score=
            round(
                coverage_score,
                2,
            ),

        consistency_score=
            round(
                consistency_score,
                2,
            ),

        robustness_score=
            round(
                robustness_score,
                2,
            ),

        relationship_score=
            result.relationship_score,

        sample_size=(
            int(
                result.joined_rows
            )
            if (
                not is_blocked_result(
                    result
                )
            )
            else 0
        ),

        period_count=(
            period_count
            if (
                not is_blocked_result(
                    result
                )
            )
            else 0
        ),

        consistency_ratio=(
            round(
                consistency_ratio,
                4,
            )
            if (
                consistency_ratio
                is not None
                and
                not is_blocked_result(
                    result
                )
            )
            else None
        ),

        importance_reasons=
            reasons,

        caveats=(
            build_caveats(
                result,

                coverage_score=
                    coverage_score,
            )
        ),

        metrics={
            "association":
                coefficient,

            "score_components":
                components,

            "joined_rows":
                result.joined_rows,

            "left_key_coverage":
                result.left_key_coverage,

            "right_key_coverage":
                result.right_key_coverage,

            "join_safety":
                result.join_safety,

            "alignment_actions":
                result.alignment_actions,

            "candidate_relationship_score":
                result.relationship_score,
        },
    )


# ============================================================
# SORTING
# ============================================================

def ranking_sort_key(
    item: RankedAnalysis,
) -> tuple[
    int,
    float,
    float,
]:
    if (
        item.tier
        ==
        "blocked"
    ):
        return (
            0,

            float(
                item.relationship_score
                or
                0.0
            ),

            0.0,
        )


    return (
        1,

        item.interestingness_score,

        float(
            item.relationship_score
            or
            0.0
        ),
    )


# ============================================================
# COMPLETE RANKING
# ============================================================

def rank_cross_dataset_execution(
    execution: CrossDatasetExecutionReport,
) -> AnalysisRankingReport:
    ranked = [
        rank_cross_dataset_result(
            result
        )
        for result
        in execution.results
    ]


    ranked.sort(
        key=
            ranking_sort_key,
        reverse=True,
    )


    for (
        index,
        item,
    ) in enumerate(
        ranked,
        start=1,
    ):
        item.rank = index


    return AnalysisRankingReport(
        ranked_count=
            len(
                ranked
            ),

        key_finding_count=
            sum(
                1
                for item
                in ranked
                if (
                    item.tier
                    ==
                    "key_finding"
                )
            ),

        supporting_finding_count=
            sum(
                1
                for item
                in ranked
                if (
                    item.tier
                    ==
                    "supporting_finding"
                )
            ),

        supplementary_count=
            sum(
                1
                for item
                in ranked
                if (
                    item.tier
                    ==
                    "supplementary"
                )
            ),

        blocked_count=
            sum(
                1
                for item
                in ranked
                if (
                    item.tier
                    ==
                    "blocked"
                )
            ),

        findings=
            ranked,

        ranking_notes=[
            (
                "Le score d'intérêt n'est ni une "
                "p-value, ni une mesure de causalité, "
                "ni une mesure automatique "
                "d'importance métier."
            ),

            (
                "Pour les analyses exécutées, le "
                "score combine l'amplitude observée, "
                "la cohérence temporelle, la "
                "couverture des données appariées, "
                "la taille de l'échantillon et la "
                "confiance dans la relation entre "
                "datasets."
            ),

            (
                "Les analyses non exécutables "
                "reçoivent un score d'intérêt et "
                "un score de robustesse nuls."
            ),

            (
                "Le score de relation des analyses "
                "bloquées est conservé uniquement "
                "pour prioriser de futures tentatives "
                "d'alignement du grain."
            ),

            (
                "Pour les données de panel, "
                "l'association médiane calculée "
                "séparément par période est "
                "préférée à l'association globale "
                "agrégée."
            ),
        ],

        ranking_rule_version=(
            "interestingness_ranker_v0.2"
        ),
    )