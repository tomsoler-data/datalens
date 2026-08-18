from __future__ import annotations

from typing import (
    Any,
)

from app.ranking.semantic_overlay_schemas import (
    SemanticOverlayFinding,
    SemanticRankingOverlayReport,
)


# ============================================================
# CONFIGURATION
# ============================================================

BLOCKED_EXECUTION_STATUSES = {
    "requires_alignment",
    "failed",
    "skipped",
}


# ============================================================
# SAFE ATTRIBUTE ACCESS
# ============================================================

def safe_string(
    value: Any,
    default: str = "",
) -> str:
    if value is None:
        return default


    return str(
        value
    )


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return default


def safe_optional_int(
    value: Any,
) -> int | None:
    if value is None:
        return None


    try:
        return int(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None


# ============================================================
# SCORE
# ============================================================

def clamp_score(
    value: float,
) -> float:
    return round(
        max(
            0.0,
            min(
                100.0,
                value,
            ),
        ),
        2,
    )


# ============================================================
# BLOCKED SAFETY
# ============================================================

def finding_is_blocked(
    finding: Any,
) -> bool:
    tier = safe_string(
        getattr(
            finding,
            "tier",
            "",
        )
    )


    execution_status = (
        safe_string(
            getattr(
                finding,
                "execution_status",
                "",
            )
        )
    )


    return (
        tier
        ==
        "blocked"
        or
        execution_status
        in BLOCKED_EXECUTION_STATUSES
    )


# ============================================================
# ADVICE LOOKUP
# ============================================================

def build_advice_map(
    semantic_advice: Any,
) -> dict[
    str,
    Any,
]:
    advice_items = list(
        getattr(
            semantic_advice,
            "advice",
            [],
        )
    )


    return {
        safe_string(
            getattr(
                item,
                "analysis_id",
                "",
            )
        ):
            item

        for item
        in advice_items

        if safe_string(
            getattr(
                item,
                "analysis_id",
                "",
            )
        )
    }


# ============================================================
# SINGLE OVERLAY FINDING
# ============================================================

def build_overlay_finding(
    *,
    finding: Any,
    advice: Any | None,
) -> SemanticOverlayFinding:
    analysis_id = safe_string(
        getattr(
            finding,
            "analysis_id",
            "",
        )
    )


    title = safe_string(
        getattr(
            finding,
            "title",
            "",
        )
    )


    family = safe_string(
        getattr(
            finding,
            "family",
            "",
        )
    )


    execution_status = (
        safe_string(
            getattr(
                finding,
                "execution_status",
                "",
            )
        )
    )


    base_tier = safe_string(
        getattr(
            finding,
            "tier",
            "",
        )
    )


    base_rank = safe_optional_int(
        getattr(
            finding,
            "rank",
            None,
        )
    )


    base_score = clamp_score(
        safe_float(
            getattr(
                finding,
                "interestingness_score",
                0.0,
            )
        )
    )


    blocked = finding_is_blocked(
        finding
    )


    # ========================================================
    # NO SEMANTIC ADVICE
    # ========================================================

    if advice is None:
        semantic_score = (
            0.0
            if blocked
            else base_score
        )


        return SemanticOverlayFinding(
            analysis_id=
                analysis_id,

            title=
                title,

            family=
                family,

            execution_status=
                execution_status,

            base_tier=
                base_tier,

            base_rank=
                base_rank,

            base_score=
                base_score,

            semantic_decision=
                "none",

            semantic_advice_status=
                None,

            raw_delta=
                0.0,

            applied_delta=
                0.0,

            semantic_score=
                semantic_score,

            blocked=
                blocked,

            application_status=(
                "blocked"
                if blocked
                else "no_advice"
            ),

            reasons=
                [],
        )


    # ========================================================
    # SEMANTIC ADVICE
    # ========================================================

    raw_delta = safe_float(
        getattr(
            advice,
            "semantic_score_delta",
            0.0,
        )
    )


    semantic_decision = (
        safe_string(
            getattr(
                advice,
                "decision",
                "neutral",
            )
        )
    )


    semantic_advice_status = (
        safe_string(
            getattr(
                advice,
                "status",
                "",
            )
        )
    )


    reasons = list(
        getattr(
            advice,
            "reasons",
            [],
        )
        or
        []
    )


    # ========================================================
    # HARD SAFETY RULE
    #
    # Semantic information can NEVER make a blocked
    # analytical result executable or rankable.
    # ========================================================

    if blocked:
        return SemanticOverlayFinding(
            analysis_id=
                analysis_id,

            title=
                title,

            family=
                family,

            execution_status=
                execution_status,

            base_tier=
                base_tier,

            base_rank=
                base_rank,

            base_score=
                base_score,

            semantic_decision=
                semantic_decision,

            semantic_advice_status=
                semantic_advice_status,

            raw_delta=
                raw_delta,

            applied_delta=
                0.0,

            semantic_score=
                0.0,

            blocked=
                True,

            application_status=
                "blocked",

            reasons=
                reasons,
        )


    # ========================================================
    # APPLY DELTA
    # ========================================================

    applied_delta = raw_delta


    semantic_score = clamp_score(
        base_score
        +
        applied_delta
    )


    if (
        applied_delta
        ==
        0.0
    ):
        application_status = (
            "neutral"
        )

    else:
        application_status = (
            "applied"
        )


    return SemanticOverlayFinding(
        analysis_id=
            analysis_id,

        title=
            title,

        family=
            family,

        execution_status=
            execution_status,

        base_tier=
            base_tier,

        base_rank=
            base_rank,

        base_score=
            base_score,

        semantic_decision=
            semantic_decision,

        semantic_advice_status=
            semantic_advice_status,

        raw_delta=
            raw_delta,

        applied_delta=
            applied_delta,

        semantic_score=
            semantic_score,

        blocked=
            False,

        application_status=
            application_status,

        reasons=
            reasons,
    )


# ============================================================
# SORTING
# ============================================================

def semantic_sort_key(
    finding: SemanticOverlayFinding,
) -> tuple:
    return (
        finding.blocked,
        -finding.semantic_score,
        -finding.base_score,
        finding.analysis_id,
    )


# ============================================================
# PUBLIC OVERLAY
# ============================================================

def apply_semantic_ranking_overlay(
    *,
    ranking: Any,
    semantic_advice: Any,
) -> SemanticRankingOverlayReport:
    source_findings = list(
        getattr(
            ranking,
            "findings",
            [],
        )
    )


    advice_map = build_advice_map(
        semantic_advice
    )


    overlay_findings = [
        build_overlay_finding(
            finding=
                finding,

            advice=
                advice_map.get(
                    safe_string(
                        getattr(
                            finding,
                            "analysis_id",
                            "",
                        )
                    )
                ),
        )
        for finding
        in source_findings
    ]


    overlay_findings.sort(
        key=
            semantic_sort_key
    )


    ranked_findings: list[
        SemanticOverlayFinding
    ] = []


    for rank, finding in enumerate(
        overlay_findings,
        start=1,
    ):
        ranked_findings.append(
            finding.model_copy(
                update={
                    "semantic_rank":
                        rank,
                }
            )
        )


    blocked_count = sum(
        finding.blocked
        for finding
        in ranked_findings
    )


    non_blocked_count = (
        len(
            ranked_findings
        )
        -
        blocked_count
    )


    changed_count = sum(
        finding.applied_delta
        !=
        0.0
        for finding
        in ranked_findings
    )


    boosted_applied_count = sum(
        (
            finding.applied_delta
            >
            0.0
        )
        for finding
        in ranked_findings
    )


    penalized_applied_count = sum(
        (
            finding.applied_delta
            <
            0.0
            and
            finding.semantic_decision
            ==
            "penalize"
        )
        for finding
        in ranked_findings
    )


    review_applied_count = sum(
        (
            finding.applied_delta
            <
            0.0
            and
            finding.semantic_decision
            ==
            "review"
        )
        for finding
        in ranked_findings
    )


    source_ranking_version = (
        safe_string(
            getattr(
                ranking,
                "ranking_rule_version",
                "",
            )
        )
        or
        safe_string(
            getattr(
                ranking,
                "rule_version",
                "",
            )
        )
        or
        "unknown"
    )


    semantic_advisor_version = (
        safe_string(
            getattr(
                semantic_advice,
                "semantic_rule_version",
                "",
            )
        )
        or
        "unknown"
    )


    return SemanticRankingOverlayReport(
        source_ranking_version=
            source_ranking_version,

        semantic_advisor_version=
            semantic_advisor_version,

        finding_count=
            len(
                ranked_findings
            ),

        non_blocked_count=
            non_blocked_count,

        blocked_count=
            blocked_count,

        changed_count=
            changed_count,

        boosted_applied_count=
            boosted_applied_count,

        penalized_applied_count=
            penalized_applied_count,

        review_applied_count=
            review_applied_count,

        findings=
            ranked_findings,
    )


# ============================================================
# LOOKUP
# ============================================================

def semantic_overlay_by_analysis_id(
    report: SemanticRankingOverlayReport,
) -> dict[
    str,
    SemanticOverlayFinding,
]:
    return {
        finding.analysis_id:
            finding

        for finding
        in report.findings
    }
