from __future__ import annotations

import math

from typing import (
    Any,
)

import numpy as np

import pandas as pd

from scipy.stats import (
    spearmanr,
)

from app.discovery.schemas import (
    AnalysisDiscoveryReport,
    DiscoveredAnalysis,
)

from app.discovery.validator import (
    infer_measure_unit,
)

from app.execution.cross_schemas import (
    CrossDatasetExecutedAnalysis,
    CrossDatasetExecutionReport,
)

from app.execution.single_schemas import (
    SingleDatasetExecutedAnalysis,
    SingleDatasetExecutionReport,
)

from app.ranking.unified_schemas import (
    UnifiedRankedAnalysis,
    UnifiedRankingReport,
)


# ============================================================
# THRESHOLDS
# ============================================================

KEY_FINDING_THRESHOLD = 78.0

SUPPORTING_FINDING_THRESHOLD = 60.0


BLOCKED_STATUSES = {
    "requires_alignment",
    "needs_specialized_method",
    "skipped",
    "failed",
}


# ============================================================
# NUMERIC HELPERS
# ============================================================

def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> float:
    return max(
        minimum,
        min(
            maximum,
            float(
                value
            ),
        ),
    )


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


# ============================================================
# DATASET ROW COUNTS
# ============================================================

def build_dataset_row_counts(
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    int,
]:
    counts: dict[
        str,
        int,
    ] = {}


    for dataset in datasets:
        dataset_id = str(
            dataset[
                "dataset_id"
            ]
        )


        dataframe = dataset.get(
            "dataframe"
        )


        if isinstance(
            dataframe,
            pd.DataFrame,
        ):
            counts[
                dataset_id
            ] = int(
                len(
                    dataframe
                )
            )


    return counts


# ============================================================
# DISCOVERY LOOKUP
# ============================================================

def build_candidate_map(
    discovery: AnalysisDiscoveryReport,
) -> dict[
    str,
    DiscoveredAnalysis,
]:
    return {
        candidate.analysis_id:
            candidate
        for candidate
        in discovery.candidates
    }


# ============================================================
# BLOCKED STATE
# ============================================================

def is_blocked_status(
    status: str,
) -> bool:
    return (
        status
        in BLOCKED_STATUSES
    )


# ============================================================
# SIGNAL TYPE
# ============================================================

def get_signal_type(
    family: str,
) -> str:
    mapping = {
        "quantitative_association":
            "association",

        "categorical_association":
            "categorical_association",

        "time_series":
            "trend",

        "derived_gap":
            "gap",

        "distribution":
            "distribution_anomaly",

        "data_quality":
            "data_quality",

        "group_comparison":
            "group_difference",

        "geographic_comparison":
            "geographic_ranking",
    }


    return mapping.get(
        family,
        "unknown",
    )


# ============================================================
# PRIMARY ASSOCIATION
# ============================================================

def get_primary_association(
    metrics: dict[
        str,
        Any,
    ],
) -> float | None:
    for key in (
        "median_period_spearman",
        "spearman",
        "preliminary_spearman",
        "overall_preliminary_spearman",
    ):
        value = safe_float(
            metrics.get(
                key
            )
        )


        if value is not None:
            return value


    return None


# ============================================================
# ASSOCIATION DIRECTION
# ============================================================

def association_direction(
    metrics: dict[
        str,
        Any,
    ],
    coefficient: float | None,
) -> str:
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


    if (
        positive
        >
        0
        and
        negative
        >
        0
    ):
        return "mixed"


    if (
        positive
        >
        0
        and
        negative
        ==
        0
    ):
        return "positive"


    if (
        negative
        >
        0
        and
        positive
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


    return "neutral"


# ============================================================
# ASSOCIATION STRENGTH
# ============================================================

def association_strength(
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
# CANDIDATE MEASURE UNIT
# ============================================================

def candidate_measure_unit(
    candidate: DiscoveredAnalysis,
) -> str:
    for variable in (
        candidate.variables
    ):
        if (
            variable.analysis_kind
            ==
            "quantitative"
            or
            variable.semantic_role
            in {
                "measure",
                "percentage",
            }
        ):
            return infer_measure_unit(
                variable
            )


    return "unknown"


# ============================================================
# SEMANTIC PENALTY
# ============================================================

def calculate_semantic_penalty(
    candidate: DiscoveredAnalysis,
) -> float:
    if (
        candidate.family
        !=
        "quantitative_association"
    ):
        return 0.0


    quantitative = [
        variable
        for variable
        in candidate.variables
        if (
            variable.analysis_kind
            ==
            "quantitative"
            or
            variable.semantic_role
            in {
                "measure",
                "percentage",
            }
        )
    ]


    if (
        len(
            quantitative
        )
        <
        2
    ):
        return 0.0


    left = (
        quantitative[
            0
        ]
    )

    right = (
        quantitative[
            1
        ]
    )


    left_concepts = set(
        left.concepts
    )

    right_concepts = set(
        right.concepts
    )


    overlap = (
        left_concepts
        &
        right_concepts
    )


    union = (
        left_concepts
        |
        right_concepts
    )


    overlap_ratio = (
        len(
            overlap
        )
        /
        len(
            union
        )
        if union
        else 0.0
    )


    left_unit = (
        infer_measure_unit(
            left
        )
    )

    right_unit = (
        infer_measure_unit(
            right
        )
    )


    if (
        overlap
        and
        {
            left_unit,
            right_unit,
        }
        ==
        {
            "rate",
            "count",
        }
    ):
        return 25.0


    if (
        overlap_ratio
        >=
        0.50
    ):
        return 15.0


    return 0.0


# ============================================================
# PERIOD CONSISTENCY
# ============================================================

def period_consistency(
    metrics: dict[
        str,
        Any,
    ],
) -> float:
    period_count = int(
        metrics.get(
            "period_count_analysed",
            0,
        )
        or
        0
    )


    # No temporal consistency measurement exists.
    # 50 is deliberately neutral rather than
    # artificially rewarding the analysis.
    if (
        period_count
        <=
        0
    ):
        return 50.0


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


    dominant = max(
        positive,
        negative,
    )


    if dominant <= 0:
        return 50.0


    return clamp(
        dominant
        /
        period_count
        *
        100.0
    )


# ============================================================
# ASSOCIATION SIGNAL
# ============================================================

def score_association(
    metrics: dict[
        str,
        Any,
    ],
) -> tuple[
    float,
    float,
    str,
    str,
]:
    coefficient = (
        get_primary_association(
            metrics
        )
    )


    signal = (
        clamp(
            abs(
                coefficient
            )
            *
            100.0
        )
        if coefficient
        is not None
        else 0.0
    )


    consistency = (
        period_consistency(
            metrics
        )
    )


    return (
        signal,
        consistency,
        association_direction(
            metrics,
            coefficient,
        ),
        association_strength(
            coefficient
        ),
    )


# ============================================================
# TIME-SERIES SIGNAL
# ============================================================

def extract_time_values(
    chart_data: list[
        dict[
            str,
            Any,
        ]
    ],
) -> list[
    float
]:
    values: list[
        float
    ] = []


    for item in chart_data:
        value = (
            item.get(
                "median"
            )
            if (
                item.get(
                    "median"
                )
                is not None
            )
            else
            item.get(
                "value"
            )
        )


        numeric = safe_float(
            value
        )


        if numeric is not None:
            values.append(
                numeric
            )


    return values


def score_time_series(
    candidate: DiscoveredAnalysis,
    result: SingleDatasetExecutedAnalysis,
) -> tuple[
    float,
    float,
    str,
    str,
]:
    values = (
        extract_time_values(
            result.chart_data
        )
    )


    if (
        len(
            values
        )
        <
        2
    ):
        return (
            0.0,
            0.0,
            "unknown",
            "unknown",
        )


    first = float(
        values[
            0
        ]
    )

    last = float(
        values[
            -1
        ]
    )


    change = (
        last
        -
        first
    )


    unit = (
        candidate_measure_unit(
            candidate
        )
    )


    if (
        unit
        ==
        "percentage"
    ):
        # Around 20 percentage points is treated
        # as a very large temporal movement.
        change_score = clamp(
            abs(
                change
            )
            /
            20.0
            *
            100.0
        )

    else:
        array = np.asarray(
            values,
            dtype=float,
        )


        q10 = float(
            np.quantile(
                array,
                0.10,
            )
        )

        q90 = float(
            np.quantile(
                array,
                0.90,
            )
        )


        robust_range = abs(
            q90
            -
            q10
        )


        if (
            robust_range
            >
            1e-12
        ):
            change_score = clamp(
                abs(
                    change
                )
                /
                robust_range
                *
                80.0
            )

        else:
            change_score = 0.0


    if (
        len(
            values
        )
        >=
        3
    ):
        trend_result = spearmanr(
            np.arange(
                len(
                    values
                )
            ),
            values,
        )


        trend_coefficient = safe_float(
            trend_result.statistic
        )


        monotonicity = (
            clamp(
                abs(
                    trend_coefficient
                )
                *
                100.0
            )
            if trend_coefficient
            is not None
            else 0.0
        )

    else:
        monotonicity = 50.0


    signal = clamp(
        change_score
        *
        0.65
        +
        monotonicity
        *
        0.35
    )


    direction = (
        "positive"
        if change
        >
        0
        else
        "negative"
        if change
        <
        0
        else
        "neutral"
    )


    return (
        signal,
        monotonicity,
        direction,
        "trend",
    )


# ============================================================
# DERIVED GAP SIGNAL
# ============================================================

def score_derived_gap(
    candidate: DiscoveredAnalysis,
    result: SingleDatasetExecutedAnalysis,
) -> tuple[
    float,
    float,
    str,
    str,
]:
    metrics = (
        result.metrics
    )


    median_gap = safe_float(
        metrics.get(
            "median_gap"
        )
    )


    if median_gap is None:
        return (
            0.0,
            0.0,
            "unknown",
            "unknown",
        )


    unit = (
        candidate_measure_unit(
            candidate
        )
    )


    if (
        unit
        ==
        "percentage"
    ):
        signal = clamp(
            abs(
                median_gap
            )
            /
            25.0
            *
            100.0
        )

    else:
        signal = clamp(
            50.0
            +
            min(
                abs(
                    median_gap
                ),
                50.0,
            )
        )


    positive = int(
        metrics.get(
            "positive_count",
            0,
        )
        or
        0
    )

    negative = int(
        metrics.get(
            "negative_count",
            0,
        )
        or
        0
    )

    zero = int(
        metrics.get(
            "zero_count",
            0,
        )
        or
        0
    )


    total = (
        positive
        +
        negative
        +
        zero
    )


    consistency = (
        clamp(
            max(
                positive,
                negative,
                zero,
            )
            /
            total
            *
            100.0
        )
        if total
        >
        0
        else 50.0
    )


    direction = (
        "positive"
        if median_gap
        >
        0
        else
        "negative"
        if median_gap
        <
        0
        else
        "neutral"
    )


    return (
        signal,
        consistency,
        direction,
        "gap",
    )


# ============================================================
# DISTRIBUTION SIGNAL
# ============================================================

def score_distribution(
    result: SingleDatasetExecutedAnalysis,
) -> tuple[
    float,
    float,
    str,
    str,
]:
    metrics = (
        result.metrics
    )


    skewness = abs(
        safe_float(
            metrics.get(
                "skewness"
            )
        )
        or
        0.0
    )


    outlier_ratio = max(
        0.0,
        safe_float(
            metrics.get(
                "outlier_ratio"
            )
        )
        or
        0.0,
    )


    n = int(
        metrics.get(
            "n",
            0,
        )
        or
        0
    )


    missing = int(
        metrics.get(
            "missing_count",
            0,
        )
        or
        0
    )


    total = (
        n
        +
        missing
    )


    missing_ratio = (
        missing
        /
        total
        if total
        >
        0
        else 0.0
    )


    skew_score = clamp(
        skewness
        /
        2.0
        *
        45.0
    )


    outlier_score = clamp(
        outlier_ratio
        /
        0.10
        *
        35.0
    )


    missing_score = clamp(
        missing_ratio
        /
        0.30
        *
        20.0
    )


    signal = clamp(
        skew_score
        +
        outlier_score
        +
        missing_score
    )


    return (
        signal,
        50.0,
        "unknown",
        "anomaly",
    )


# ============================================================
# DATA QUALITY SIGNAL
# ============================================================

def score_data_quality(
    result: SingleDatasetExecutedAnalysis,
) -> tuple[
    float,
    float,
    str,
    str,
]:
    metrics = (
        result.metrics
    )


    missing_ratio = max(
        0.0,
        safe_float(
            metrics.get(
                "missing_ratio"
            )
        )
        or
        0.0,
    )


    duplicate_ratio = max(
        0.0,
        safe_float(
            metrics.get(
                "duplicate_ratio"
            )
        )
        or
        0.0,
    )


    completely_missing = len(
        metrics.get(
            "completely_missing_columns",
            [],
        )
        or
        []
    )


    constant_columns = len(
        metrics.get(
            "constant_columns",
            [],
        )
        or
        []
    )


    missing_score = clamp(
        missing_ratio
        /
        0.30
        *
        55.0
    )


    duplicate_score = clamp(
        duplicate_ratio
        /
        0.10
        *
        25.0
    )


    structural_score = min(
        20.0,

        completely_missing
        *
        10.0
        +
        constant_columns
        *
        3.0,
    )


    signal = clamp(
        missing_score
        +
        duplicate_score
        +
        structural_score
    )


    # There is no concept of temporal consistency
    # for a data-quality audit. A neutral value is
    # preferable to 100.
    return (
        signal,
        50.0,
        "unknown",
        "quality",
    )


# ============================================================
# GROUP DIFFERENCE SIGNAL
# ============================================================

def score_group_comparison(
    candidate: DiscoveredAnalysis,
    result: SingleDatasetExecutedAnalysis,
) -> tuple[
    float,
    float,
    str,
    str,
]:
    medians = [
        safe_float(
            item.get(
                "median"
            )
        )
        for item
        in result.chart_data
    ]


    medians = [
        value
        for value
        in medians
        if value
        is not None
    ]


    if (
        len(
            medians
        )
        <
        2
    ):
        return (
            0.0,
            0.0,
            "unknown",
            "unknown",
        )


    spread = (
        max(
            medians
        )
        -
        min(
            medians
        )
    )


    unit = (
        candidate_measure_unit(
            candidate
        )
    )


    if (
        unit
        ==
        "percentage"
    ):
        signal = clamp(
            abs(
                spread
            )
            /
            30.0
            *
            100.0
        )

    else:
        central = float(
            np.median(
                np.abs(
                    medians
                )
            )
        )


        if central > 1e-12:
            signal = clamp(
                abs(
                    spread
                )
                /
                central
                *
                70.0
            )

        else:
            signal = clamp(
                abs(
                    spread
                )
            )


    return (
        signal,
        50.0,
        "mixed",
        "group_difference",
    )


# ============================================================
# CATEGORICAL ASSOCIATION SIGNAL
# ============================================================

def score_categorical_association(
    result: SingleDatasetExecutedAnalysis,
) -> tuple[
    float,
    float,
    str,
    str,
]:
    coefficient = safe_float(
        result.metrics.get(
            "cramers_v"
        )
    )


    signal = (
        clamp(
            coefficient
            *
            100.0
        )
        if coefficient
        is not None
        else 0.0
    )


    return (
        signal,
        50.0,
        "unknown",
        association_strength(
            coefficient
        ),
    )


# ============================================================
# GEOGRAPHIC SIGNAL
# ============================================================

def score_geographic_comparison(
    result: SingleDatasetExecutedAnalysis,
) -> tuple[
    float,
    float,
    str,
    str,
]:
    entity_count = int(
        result.metrics.get(
            "entity_count",
            0,
        )
        or
        0
    )


    # A ranking is useful context, but simply
    # having many countries is not itself a
    # strong analytical finding.
    signal = clamp(
        30.0
        +
        min(
            20.0,

            entity_count
            /
            200.0
            *
            20.0,
        ),

        maximum=
            50.0,
    )


    return (
        signal,
        50.0,
        "unknown",
        "ranking",
    )


# ============================================================
# SINGLE-DATASET COVERAGE
# ============================================================

def get_structure_row_count(
    metrics: dict[
        str,
        Any,
    ],
) -> int | None:
    for key in (
        "observation_structure_after",
        "observation_structure",
    ):
        structure = metrics.get(
            key
        )


        if isinstance(
            structure,
            dict,
        ):
            row_count = (
                structure.get(
                    "row_count"
                )
            )


            if row_count is not None:
                try:
                    return int(
                        row_count
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    pass


    return None


def calculate_single_coverage(
    result: SingleDatasetExecutedAnalysis,
    *,
    source_row_count: int | None,
) -> float:
    if is_blocked_status(
        result.execution_status
    ):
        return 0.0


    metrics = (
        result.metrics
    )


    if (
        result.family
        ==
        "data_quality"
    ):
        return 100.0


    if (
        result.family
        ==
        "distribution"
    ):
        n = int(
            metrics.get(
                "n",
                0,
            )
            or
            0
        )


        missing = int(
            metrics.get(
                "missing_count",
                0,
            )
            or
            0
        )


        denominator = (
            n
            +
            missing
        )


        if denominator > 0:
            return clamp(
                n
                /
                denominator
                *
                100.0
            )


    structure_rows = (
        get_structure_row_count(
            metrics
        )
    )


    denominator = (
        structure_rows
        or
        source_row_count
    )


    if (
        denominator is None
        or
        denominator
        <=
        0
    ):
        return 70.0


    return clamp(
        result.valid_observations
        /
        denominator
        *
        100.0
    )


# ============================================================
# CROSS-DATASET COVERAGE
# ============================================================

def calculate_cross_coverage(
    result: CrossDatasetExecutedAnalysis,
) -> float:
    if is_blocked_status(
        result.execution_status
    ):
        return 0.0


    left = clamp(
        result.left_key_coverage
        *
        100.0
    )


    right = clamp(
        result.right_key_coverage
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
# EXECUTION CONFIDENCE
# ============================================================

def execution_confidence(
    status: str,
) -> float:
    if (
        status
        ==
        "complete"
    ):
        return 95.0


    if (
        status
        ==
        "descriptive_only"
    ):
        return 85.0


    return 0.0


# ============================================================
# SINGLE SIGNAL ROUTER
# ============================================================

def score_single_signal(
    candidate: DiscoveredAnalysis,
    result: SingleDatasetExecutedAnalysis,
) -> tuple[
    float,
    float,
    str,
    str,
]:
    family = (
        result.family
    )


    if (
        family
        ==
        "quantitative_association"
    ):
        return score_association(
            result.metrics
        )


    if (
        family
        ==
        "categorical_association"
    ):
        return score_categorical_association(
            result
        )


    if (
        family
        ==
        "time_series"
    ):
        return score_time_series(
            candidate,
            result,
        )


    if (
        family
        ==
        "derived_gap"
    ):
        return score_derived_gap(
            candidate,
            result,
        )


    if (
        family
        ==
        "distribution"
    ):
        return score_distribution(
            result
        )


    if (
        family
        ==
        "data_quality"
    ):
        return score_data_quality(
            result
        )


    if (
        family
        ==
        "group_comparison"
    ):
        return score_group_comparison(
            candidate,
            result,
        )


    if (
        family
        ==
        "geographic_comparison"
    ):
        return score_geographic_comparison(
            result
        )


    return (
        30.0,
        50.0,
        "unknown",
        "unknown",
    )


# ============================================================
# CROSS SIGNAL ROUTER
# ============================================================

def score_cross_signal(
    result: CrossDatasetExecutedAnalysis,
) -> tuple[
    float,
    float,
    str,
    str,
]:
    if (
        result.family
        ==
        "quantitative_association"
    ):
        return score_association(
            result.metrics
        )


    return (
        30.0,
        50.0,
        "unknown",
        "unknown",
    )


# ============================================================
# FINAL INTERESTINGNESS SCORE
# ============================================================

def calculate_final_score(
    *,
    signal_score: float,
    coverage_score: float,
    consistency_score: float,
    discovery_priority_score: float,
    execution_confidence_score: float,
    semantic_penalty: float,
    relationship_penalty: float = 0.0,
) -> float:
    """
    Interestingness is primarily driven by
    the observed analytical signal.

    Coverage, consistency, discovery priority
    and execution confidence can strengthen or
    weaken a real signal, but cannot create an
    interesting finding on their own.
    """

    score = (
        signal_score
        *
        0.65
        +
        coverage_score
        *
        0.10
        +
        consistency_score
        *
        0.10
        +
        discovery_priority_score
        *
        0.075
        +
        execution_confidence_score
        *
        0.075
        -
        semantic_penalty
        -
        relationship_penalty
    )


    return round(
        clamp(
            score
        ),
        2,
    )


# ============================================================
# TIER
# ============================================================

def determine_tier(
    *,
    execution_status: str,
    score: float,
    signal_score: float,
) -> str:
    if is_blocked_status(
        execution_status
    ):
        return "blocked"


    # A weak analytical signal cannot become a
    # main finding purely because the execution
    # itself was reliable.
    if (
        score
        >=
        KEY_FINDING_THRESHOLD
        and
        signal_score
        >=
        60.0
    ):
        return "key_finding"


    if (
        score
        >=
        SUPPORTING_FINDING_THRESHOLD
        and
        signal_score
        >=
        40.0
    ):
        return "supporting_finding"


    return "supplementary"


# ============================================================
# REASONS
# ============================================================

def build_reasons(
    *,
    family: str,
    signal_score: float,
    coverage_score: float,
    consistency_score: float,
    semantic_penalty: float,
    direction: str,
) -> list[
    str
]:
    reasons: list[
        str
    ] = []


    if (
        family
        ==
        "quantitative_association"
    ):
        if (
            signal_score
            >=
            70.0
        ):
            reasons.append(
                (
                    "Une association de forte "
                    "amplitude est observée."
                )
            )

        elif (
            signal_score
            >=
            50.0
        ):
            reasons.append(
                (
                    "Une association d'amplitude "
                    "notable est observée."
                )
            )


        if (
            consistency_score
            >=
            90.0
        ):
            reasons.append(
                (
                    "Le sens de l'association est "
                    "très cohérent entre les "
                    "périodes analysées."
                )
            )


    elif (
        family
        ==
        "time_series"
    ):
        if (
            signal_score
            >=
            60.0
        ):
            reasons.append(
                (
                    "L'évolution temporelle est "
                    "suffisamment marquée pour "
                    "mériter une attention "
                    "particulière."
                )
            )


        if (
            consistency_score
            >=
            80.0
        ):
            reasons.append(
                (
                    "La trajectoire présente une "
                    "direction relativement "
                    "cohérente dans le temps."
                )
            )


    elif (
        family
        ==
        "derived_gap"
    ):
        if (
            signal_score
            >=
            60.0
        ):
            reasons.append(
                (
                    "L'écart entre les deux "
                    "mesures est important par "
                    "rapport à leur unité."
                )
            )


        if (
            consistency_score
            >=
            90.0
        ):
            reasons.append(
                (
                    "Le sens de l'écart est très "
                    "stable entre les observations."
                )
            )


    elif (
        family
        ==
        "distribution"
    ):
        if (
            signal_score
            >=
            70.0
        ):
            reasons.append(
                (
                    "La distribution présente une "
                    "asymétrie, des valeurs "
                    "atypiques ou des données "
                    "manquantes suffisamment "
                    "marquées pour être signalées."
                )
            )


    elif (
        family
        ==
        "data_quality"
    ):
        if (
            signal_score
            >=
            50.0
        ):
            reasons.append(
                (
                    "Des problèmes de qualité "
                    "suffisamment importants ont "
                    "été détectés dans ce dataset."
                )
            )

        elif (
            signal_score
            <=
            15.0
        ):
            reasons.append(
                (
                    "Aucun problème majeur de "
                    "qualité n'a été détecté."
                )
            )


    elif (
        family
        ==
        "group_comparison"
    ):
        if (
            signal_score
            >=
            60.0
        ):
            reasons.append(
                (
                    "Les groupes présentent des "
                    "différences descriptives "
                    "importantes."
                )
            )


    if (
        signal_score
        >=
        40.0
    ):
        if (
            coverage_score
            >=
            85.0
        ):
            reasons.append(
                (
                    "La couverture des observations "
                    "utilisées est élevée."
                )
            )

        elif (
            coverage_score
            <
            60.0
        ):
            reasons.append(
                (
                    "Le signal doit être interprété "
                    "avec prudence en raison d'une "
                    "couverture partielle."
                )
            )


    if (
        semantic_penalty
        >
        0
    ):
        reasons.append(
            (
                "Le score a été réduit car les "
                "variables sont sémantiquement "
                "proches et leur association "
                "peut être en partie structurelle."
            )
        )


    if (
        not reasons
        and
        direction
        !=
        "unknown"
    ):
        reasons.append(
            (
                "Un signal descriptif exploitable "
                "a été détecté dans les données."
            )
        )


    return reasons


# ============================================================
# CAVEATS
# ============================================================

def build_caveats(
    *,
    execution_status: str,
    warnings: list[
        str
    ],
    limitations: list[
        str
    ],
    semantic_penalty: float,
) -> list[
    str
]:
    caveats: list[
        str
    ] = []


    if (
        execution_status
        ==
        "descriptive_only"
    ):
        caveats.append(
            (
                "Le résultat est descriptif et "
                "ne constitue pas une conclusion "
                "causale ou confirmatoire."
            )
        )


    if is_blocked_status(
        execution_status
    ):
        caveats.append(
            (
                "Cette analyse n'a pas produit "
                "un résultat suffisamment sûr "
                "pour être présentée comme un "
                "constat."
            )
        )


    if (
        semantic_penalty
        >
        0
    ):
        caveats.append(
            (
                "La proximité conceptuelle entre "
                "les variables peut contribuer "
                "à l'association observée."
            )
        )


    caveats.extend(
        warnings
    )


    caveats.extend(
        limitations
    )


    return list(
        dict.fromkeys(
            caveats
        )
    )


# ============================================================
# SINGLE RESULT RANKING
# ============================================================

def rank_single_result(
    *,
    result: SingleDatasetExecutedAnalysis,
    candidate: DiscoveredAnalysis,
    source_row_count: int | None,
) -> UnifiedRankedAnalysis:
    blocked = (
        is_blocked_status(
            result.execution_status
        )
    )


    if blocked:
        signal_score = 0.0
        consistency_score = 0.0
        direction = "unknown"
        strength = "unknown"

    else:
        (
            signal_score,
            consistency_score,
            direction,
            strength,
        ) = score_single_signal(
            candidate,
            result,
        )


    coverage_score = (
        calculate_single_coverage(
            result,

            source_row_count=
                source_row_count,
        )
    )


    discovery_priority = clamp(
        candidate.priority_score
    )


    confidence = (
        execution_confidence(
            result.execution_status
        )
    )


    semantic_penalty = (
        0.0
        if blocked
        else
        calculate_semantic_penalty(
            candidate
        )
    )


    score = (
        0.0
        if blocked
        else
        calculate_final_score(
            signal_score=
                signal_score,

            coverage_score=
                coverage_score,

            consistency_score=
                consistency_score,

            discovery_priority_score=
                discovery_priority,

            execution_confidence_score=
                confidence,

            semantic_penalty=
                semantic_penalty,
        )
    )


    period_count = int(
        result.metrics.get(
            "period_count_analysed",
            result.metrics.get(
                "period_count",
                0,
            ),
        )
        or
        0
    )


    return UnifiedRankedAnalysis(
        rank=0,

        analysis_id=
            result.analysis_id,

        title=
            result.title,

        scope=
            "single_dataset",

        family=
            result.family,

        execution_status=
            result.execution_status,

        tier=
            determine_tier(
                execution_status=
                    result.execution_status,

                score=
                    score,

                signal_score=
                    signal_score,
            ),

        signal_type=
            get_signal_type(
                result.family
            ),

        interestingness_score=
            score,

        signal_score=
            round(
                signal_score,
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

        discovery_priority_score=
            round(
                discovery_priority,
                2,
            ),

        execution_confidence_score=
            round(
                confidence,
                2,
            ),

        semantic_penalty=
            round(
                semantic_penalty,
                2,
            ),

        direction=
            direction,

        strength=
            strength,

        sample_size=(
            int(
                result.valid_observations
            )
            if not blocked
            else 0
        ),

        period_count=(
            period_count
            if not blocked
            else 0
        ),

        dataset_ids=[
            result.dataset_id
        ],

        datasets=[
            result.dataset
        ],

        reasons=(
            build_reasons(
                family=
                    result.family,

                signal_score=
                    signal_score,

                coverage_score=
                    coverage_score,

                consistency_score=
                    consistency_score,

                semantic_penalty=
                    semantic_penalty,

                direction=
                    direction,
            )
            if not blocked
            else [
                (
                    "L'analyse a été identifiée "
                    "mais n'a pas pu produire un "
                    "résultat suffisamment sûr."
                )
            ]
        ),

        caveats=(
            build_caveats(
                execution_status=
                    result.execution_status,

                warnings=
                    result.warnings,

                limitations=
                    result.limitations,

                semantic_penalty=
                    semantic_penalty,
            )
        ),

        metrics={
            "execution_metrics":
                result.metrics,

            "chart_type":
                result.chart_type,

            "discovery_priority":
                candidate.priority_score,
        },
    )


# ============================================================
# CROSS RESULT RANKING
# ============================================================

def rank_cross_result(
    *,
    result: CrossDatasetExecutedAnalysis,
    candidate: DiscoveredAnalysis,
) -> UnifiedRankedAnalysis:
    blocked = (
        is_blocked_status(
            result.execution_status
        )
    )


    if blocked:
        signal_score = 0.0
        consistency_score = 0.0
        direction = "unknown"
        strength = "unknown"

    else:
        (
            signal_score,
            consistency_score,
            direction,
            strength,
        ) = score_cross_signal(
            result
        )


    coverage_score = (
        calculate_cross_coverage(
            result
        )
    )


    discovery_priority = clamp(
        candidate.priority_score
    )


    confidence = (
        execution_confidence(
            result.execution_status
        )
    )


    semantic_penalty = (
        0.0
        if blocked
        else
        calculate_semantic_penalty(
            candidate
        )
    )


    relationship_penalty = (
        4.0
        if (
            result.relationship_status
            ==
            "partial"
            and
            not blocked
        )
        else 0.0
    )


    score = (
        0.0
        if blocked
        else
        calculate_final_score(
            signal_score=
                signal_score,

            coverage_score=
                coverage_score,

            consistency_score=
                consistency_score,

            discovery_priority_score=
                discovery_priority,

            execution_confidence_score=
                confidence,

            semantic_penalty=
                semantic_penalty,

            relationship_penalty=
                relationship_penalty,
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


    return UnifiedRankedAnalysis(
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

        tier=
            determine_tier(
                execution_status=
                    result.execution_status,

                score=
                    score,

                signal_score=
                    signal_score,
            ),

        signal_type=
            get_signal_type(
                result.family
            ),

        interestingness_score=
            score,

        signal_score=
            round(
                signal_score,
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

        discovery_priority_score=
            round(
                discovery_priority,
                2,
            ),

        execution_confidence_score=
            round(
                confidence,
                2,
            ),

        semantic_penalty=
            round(
                semantic_penalty,
                2,
            ),

        direction=
            direction,

        strength=
            strength,

        sample_size=(
            int(
                result.joined_rows
            )
            if not blocked
            else 0
        ),

        period_count=(
            period_count
            if not blocked
            else 0
        ),

        dataset_ids=list(
            result.dataset_ids
        ),

        datasets=list(
            result.datasets
        ),

        reasons=(
            build_reasons(
                family=
                    result.family,

                signal_score=
                    signal_score,

                coverage_score=
                    coverage_score,

                consistency_score=
                    consistency_score,

                semantic_penalty=
                    semantic_penalty,

                direction=
                    direction,
            )
            if not blocked
            else [
                (
                    "L'analyse reste pertinente "
                    "à explorer, mais le grain "
                    "des datasets ne permet pas "
                    "encore une exécution sûre."
                )
            ]
        ),

        caveats=(
            build_caveats(
                execution_status=
                    result.execution_status,

                warnings=
                    result.warnings,

                limitations=
                    result.limitations,

                semantic_penalty=
                    semantic_penalty,
            )
        ),

        metrics={
            "execution_metrics":
                result.metrics,

            "relationship_score":
                result.relationship_score,

            "relationship_status":
                result.relationship_status,

            "join_safety":
                result.join_safety,

            "relationship_penalty":
                relationship_penalty,

            "discovery_priority":
                candidate.priority_score,
        },
    )


# ============================================================
# SORT KEY
# ============================================================

def unified_sort_key(
    finding: UnifiedRankedAnalysis,
) -> tuple[
    int,
    float,
    float,
]:
    if (
        finding.tier
        ==
        "blocked"
    ):
        return (
            0,

            finding.discovery_priority_score,

            finding.signal_score,
        )


    return (
        1,

        finding.interestingness_score,

        finding.signal_score,
    )


# ============================================================
# COMPLETE UNIFIED RANKING
# ============================================================

def rank_unified_analysis(
    *,
    discovery: AnalysisDiscoveryReport,
    single_execution: SingleDatasetExecutionReport,
    cross_execution: CrossDatasetExecutionReport,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> UnifiedRankingReport:
    candidate_map = (
        build_candidate_map(
            discovery
        )
    )


    row_counts = (
        build_dataset_row_counts(
            datasets
        )
    )


    findings: list[
        UnifiedRankedAnalysis
    ] = []


    # ========================================================
    # SINGLE DATASET
    # ========================================================

    for result in (
        single_execution.results
    ):
        candidate = (
            candidate_map.get(
                result.analysis_id
            )
        )


        if candidate is None:
            continue


        findings.append(
            rank_single_result(
                result=
                    result,

                candidate=
                    candidate,

                source_row_count=
                    row_counts.get(
                        result.dataset_id
                    ),
            )
        )


    # ========================================================
    # CROSS DATASET
    # ========================================================

    for result in (
        cross_execution.results
    ):
        candidate = (
            candidate_map.get(
                result.analysis_id
            )
        )


        if candidate is None:
            continue


        findings.append(
            rank_cross_result(
                result=
                    result,

                candidate=
                    candidate,
            )
        )


    findings.sort(
        key=
            unified_sort_key,

        reverse=True,
    )


    for (
        index,
        finding,
    ) in enumerate(
        findings,
        start=1,
    ):
        finding.rank = (
            index
        )


    return UnifiedRankingReport(
        ranked_count=
            len(
                findings
            ),

        key_finding_count=
            sum(
                1
                for finding
                in findings
                if (
                    finding.tier
                    ==
                    "key_finding"
                )
            ),

        supporting_finding_count=
            sum(
                1
                for finding
                in findings
                if (
                    finding.tier
                    ==
                    "supporting_finding"
                )
            ),

        supplementary_count=
            sum(
                1
                for finding
                in findings
                if (
                    finding.tier
                    ==
                    "supplementary"
                )
            ),

        blocked_count=
            sum(
                1
                for finding
                in findings
                if (
                    finding.tier
                    ==
                    "blocked"
                )
            ),

        findings=
            findings,

        ranking_notes=[
            (
                "Le signal analytique constitue "
                "désormais le principal moteur "
                "du score d'intérêt."
            ),

            (
                "La couverture, la cohérence, la "
                "priorité Discovery et la confiance "
                "d'exécution renforcent ou réduisent "
                "un signal existant mais ne peuvent "
                "plus créer artificiellement un "
                "constat important."
            ),

            (
                "Une analyse avec un signal faible "
                "reste supplémentaire même lorsque "
                "sa qualité d'exécution est "
                "excellente."
            ),

            (
                "Un constat principal nécessite "
                "à la fois un score global élevé "
                "et un signal analytique d'au "
                "moins 60."
            ),

            (
                "Un constat secondaire nécessite "
                "un signal analytique d'au moins "
                "40."
            ),

            (
                "Les associations entre variables "
                "conceptuellement proches restent "
                "pénalisées."
            ),

            (
                "Les analyses bloquées restent "
                "traçables avec un score nul."
            ),

            (
                "La qualité des données restera "
                "toujours visible dans sa section "
                "du rapport, même lorsqu'elle "
                "n'est pas un constat important."
            ),
        ],

        ranking_rule_version=(
            "unified_ranker_v0.2"
        ),
    )