from __future__ import annotations


import math
import re


from pathlib import Path


from app.analysis.entity_outliers import (
    EntityOutlierCandidate,
    EntityOutlierEvidence,
    EntityOutlierReport,
    EntityOutlierViewResult,
)


from app.analysis.entity_outlier_profiles import (
    DEFAULT_PRIORITY_MIN_MAX_DISTANCE_IQR,
    DEFAULT_PRIORITY_MIN_SIGNAL_COUNT,
    DEFAULT_STRONG_MIN_MAX_DISTANCE_IQR,
    _severity,
    build_entity_outlier_profiles,
)


EXPECTED_PRIORITY_MIN_SIGNAL_COUNT = 2
EXPECTED_FAR_OUT_DISTANCE_IQR = 1.5


def make_evidence(
    *,
    metric: str,
    distance_iqr: float,
) -> EntityOutlierEvidence:

    lower_bound = -1.5
    upper_bound = 2.5
    iqr = 1.0

    value = (
        upper_bound
        +
        distance_iqr
        *
        iqr
    )


    return EntityOutlierEvidence(
        metric=metric,
        value=value,
        direction="high",
        q1=0.0,
        q3=1.0,
        iqr=iqr,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        distance_iqr=distance_iqr,
        score=1.0 + distance_iqr,
    )


def make_candidate(
    *,
    entity: str,
    distances: list[float],
) -> EntityOutlierCandidate:

    evidence = [
        make_evidence(
            metric=f"metric_{index}",
            distance_iqr=distance,
        )

        for index, distance
        in enumerate(
            distances,
            start=1,
        )
    ]


    return EntityOutlierCandidate(
        entity=entity,
        anomaly_score=sum(
            item.score
            for item in evidence
        ),
        outlier_metric_count=len(
            evidence
        ),
        evidence=evidence,
    )


def make_report(
) -> EntityOutlierReport:

    candidates = [
        make_candidate(
            entity="entity_alpha",
            distances=[3.0, 2.5],
        ),
        make_candidate(
            entity="entity_beta",
            distances=[2.5, 2.0],
        ),
        make_candidate(
            entity="entity_gamma",
            distances=[1.5],
        ),
        make_candidate(
            entity="entity_delta",
            distances=[0.5],
        ),
    ]


    result = EntityOutlierViewResult(
        dataset_id="derived:synthetic:entity",
        dataset_filename="synthetic_entity.csv",
        derivation_type="entity_additive_measure",
        operation="synthetic_entity_materialization",
        entity_column="entity_id",
        entity_count=100,
        evaluated_metrics=[
            "metric_1",
            "metric_2",
        ],
        primary_metric="metric_1",
        thresholds=[],
        flagged_entity_count=len(
            candidates
        ),
        top_entities=candidates,
        notes=[],
    )


    return EntityOutlierReport(
        candidate_view_count=1,
        evaluated_view_count=1,
        total_flagged_entity_count=len(
            candidates
        ),
        results=[
            result
        ],
        notes=[],
    )


def observed_severity(
    *,
    signal_count: int,
    max_distance_iqr: float,
) -> str:

    return _severity(
        max_distance_iqr=max_distance_iqr,
        signal_count=signal_count,
        priority_min_signal_count=(
            DEFAULT_PRIORITY_MIN_SIGNAL_COUNT
        ),
        priority_min_max_distance_iqr=(
            DEFAULT_PRIORITY_MIN_MAX_DISTANCE_IQR
        ),
        strong_min_max_distance_iqr=(
            DEFAULT_STRONG_MIN_MAX_DISTANCE_IQR
        ),
    )


def main() -> None:

    gaps: list[str] = []


    print(
        "=== DATALENS NEUTRAL TUKEY ENTITY PROFILE POLICY v0.1 ==="
    )


    # ========================================================
    # 1. DEFAULT POLICY CONSTANTS
    # ========================================================

    if (
        DEFAULT_PRIORITY_MIN_SIGNAL_COUNT
        !=
        EXPECTED_PRIORITY_MIN_SIGNAL_COUNT
    ):

        gaps.append(
            (
                "RED_GAP neutral_priority_signal_count_default "
                f"expected={EXPECTED_PRIORITY_MIN_SIGNAL_COUNT} "
                f"observed={DEFAULT_PRIORITY_MIN_SIGNAL_COUNT}"
            )
        )


    if not math.isclose(
        float(
            DEFAULT_PRIORITY_MIN_MAX_DISTANCE_IQR
        ),
        EXPECTED_FAR_OUT_DISTANCE_IQR,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):

        gaps.append(
            (
                "RED_GAP neutral_priority_distance_default "
                f"expected={EXPECTED_FAR_OUT_DISTANCE_IQR} "
                f"observed={DEFAULT_PRIORITY_MIN_MAX_DISTANCE_IQR}"
            )
        )


    if not math.isclose(
        float(
            DEFAULT_STRONG_MIN_MAX_DISTANCE_IQR
        ),
        EXPECTED_FAR_OUT_DISTANCE_IQR,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):

        gaps.append(
            (
                "RED_GAP neutral_strong_distance_default "
                f"expected={EXPECTED_FAR_OUT_DISTANCE_IQR} "
                f"observed={DEFAULT_STRONG_MIN_MAX_DISTANCE_IQR}"
            )
        )


    # ========================================================
    # 2. PRODUCTION POLICY MUST NOT CONTAIN A DATASET-SPECIFIC
    #    CALIBRATION RATIONALE
    # ========================================================

    api_root = (
        Path(__file__)
        .resolve()
        .parents[
            2
        ]
    )


    profile_path = (
        api_root
        /
        "app"
        /
        "analysis"
        /
        "entity_outlier_profiles.py"
    )


    source = profile_path.read_text(
        encoding="utf-8-sig"
    )


    threshold_start = source.find(
        "# THRESHOLDS"
    )

    metric_start = source.find(
        "# METRIC FAMILIES"
    )


    if (
        threshold_start
        <
        0
        or
        metric_start
        <
        0
        or
        metric_start
        <=
        threshold_start
    ):

        gaps.append(
            "RED_GAP threshold_policy_section_unreadable"
        )

        threshold_section = ""


    else:

        threshold_section = source[
            threshold_start:
            metric_start
        ]


    dataset_name_reference = bool(
        "lapage"
        in
        threshold_section.casefold()
    )


    dataset_entity_reference = bool(
        re.search(
            r"\bc_\d+\b",
            threshold_section,
            flags=re.IGNORECASE,
        )
    )


    ranked_dataset_reference = bool(
        re.search(
            r"\bfifth\s+ranked\s+client\b",
            threshold_section,
            flags=re.IGNORECASE,
        )
    )


    if (
        dataset_name_reference
        or
        dataset_entity_reference
        or
        ranked_dataset_reference
    ):

        gaps.append(
            "RED_GAP benchmark_specific_threshold_rationale"
        )


    # ========================================================
    # 3. DEFAULT SYNTHETIC SEVERITY MATRIX
    # ========================================================

    severity_cases = [
        (
            "single_inner_outlier",
            1,
            0.25,
            "moderate",
        ),
        (
            "single_far_out",
            1,
            1.50,
            "strong",
        ),
        (
            "multi_inner_outlier",
            2,
            0.75,
            "moderate",
        ),
        (
            "multi_far_out",
            2,
            1.50,
            "extreme",
        ),
        (
            "three_inner_signals",
            3,
            0.75,
            "strong",
        ),
    ]


    for (
        name,
        signal_count,
        distance,
        expected,
    ) in severity_cases:

        observed = observed_severity(
            signal_count=signal_count,
            max_distance_iqr=distance,
        )


        if observed != expected:

            gaps.append(
                (
                    f"RED_GAP severity_{name} "
                    f"expected={expected} "
                    f"observed={observed}"
                )
            )


        else:

            print(
                f"[PASS] {name} -> {observed}"
            )


    # ========================================================
    # 4. PUBLIC API MUST ALLOW THE SAME FAR-OUT BOUNDARY FOR
    #    STRONG AND PRIORITY.
    #
    # Signal count resolves:
    #
    #   one far-out signal  -> strong
    #   >=2 + far-out       -> extreme / priority
    # ========================================================

    neutral_public_api_accepted = True


    try:

        build_entity_outlier_profiles(
            make_report(),
            priority_min_signal_count=2,
            priority_min_max_distance_iqr=1.5,
            strong_min_max_distance_iqr=1.5,
            top_limit=10,
        )


    except ValueError as exc:

        neutral_public_api_accepted = False

        gaps.append(
            (
                "RED_GAP public_api_rejects_equal_far_out_thresholds "
                f"error={exc}"
            )
        )


    if neutral_public_api_accepted:

        print(
            "[PASS] equal neutral far-out thresholds accepted"
        )


    # ========================================================
    # 5. EXISTING TOP-LIMIT CONTRACT
    #
    # Use explicit non-equal thresholds here so this control
    # remains independent of the RED above.
    # ========================================================

    top_limit_report = build_entity_outlier_profiles(
        make_report(),
        priority_min_signal_count=2,
        priority_min_max_distance_iqr=2.0,
        strong_min_max_distance_iqr=1.0,
        top_limit=1,
    )


    if len(
        top_limit_report.results
    ) != 1:

        gaps.append(
            "CONTROL_GAP top_limit_result_count"
        )


    else:

        result = (
            top_limit_report.results[
                0
            ]
        )


        top_limit_contract = bool(
            result.priority_profile_count
            ==
            2
            and
            result.behavioral_signal_count
            ==
            2
            and
            len(
                result.priority_profiles
            )
            ==
            1
            and
            len(
                result.behavioral_signals
            )
            ==
            1
            and
            result.classified_entity_count
            ==
            4
            and
            result.source_flagged_entity_count
            ==
            4
        )


        if not top_limit_contract:

            gaps.append(
                (
                    "CONTROL_GAP top_limit_contract "
                    f"priority_count={result.priority_profile_count} "
                    f"priority_exposed={len(result.priority_profiles)} "
                    f"behavioral_count={result.behavioral_signal_count} "
                    f"behavioral_exposed={len(result.behavioral_signals)} "
                    f"classified={result.classified_entity_count}"
                )
            )


        else:

            print(
                (
                    "[PASS] top_limit preserves global counts "
                    "and truncates exposed lists"
                )
            )


    # ========================================================
    # FINAL
    # ========================================================

    print()

    print(
        "DEFAULT_PRIORITY_MIN_SIGNAL_COUNT       "
        f"{DEFAULT_PRIORITY_MIN_SIGNAL_COUNT}"
    )

    print(
        "DEFAULT_PRIORITY_MIN_MAX_DISTANCE_IQR   "
        f"{DEFAULT_PRIORITY_MIN_MAX_DISTANCE_IQR}"
    )

    print(
        "DEFAULT_STRONG_MIN_MAX_DISTANCE_IQR     "
        f"{DEFAULT_STRONG_MIN_MAX_DISTANCE_IQR}"
    )

    print(
        f"Observed gaps                           {gaps}"
    )


    if gaps:

        print()

        for gap in gaps:
            print(
                gap
            )

        print()

        raise AssertionError(
            "Neutral Tukey entity-profile policy is not GREEN."
        )


    print()

    print(
        "PASS - neutral Tukey entity-profile policy v0.1"
    )


if __name__ == "__main__":

    main()