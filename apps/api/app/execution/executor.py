from __future__ import annotations


import math
import re

from typing import (
    Any,
)


import numpy as np

import pandas as pd


from scipy.stats import (
    chi2_contingency,
)


from app.execution.schemas import (
    AnalysisExecutionReport,
    ExecutedAnalysis,
)

from app.planning.schemas import (
    AnalysisCandidate,
    AnalysisPlanReport,
)

from app.statistics import (
    decide_correlation_test,
    execute_correlation_decision,
)

from app.statistics.executor import (
    calculate_pearson_statistic,
    calculate_spearman_statistic,
)

from app.visualization import (
    decide_correlation_visualization,
)


# ============================================================
# CONSTANTS
# ============================================================

MAX_CHART_POINTS = 2000

MAX_GROUPS_FOR_TABLE = 30

MAX_GROUPS_FOR_LINE_CHART = 12

MAX_CATEGORICAL_LEVELS = 40


GROUP_CHART_CONSISTENCY_RULE_VERSION = (
    "group_chart_consistency_v0.1"
)


ENTITY_NAME_SIGNALS = {
    "country",
    "countries",
    "pays",
    "customer",
    "client",
    "user",
    "patient",
    "employee",
    "store",
    "shop",
    "account",
    "product",
    "entity",
}


TEMPORAL_NAME_SIGNALS = {
    "year",
    "annee",
    "date",
    "datetime",
    "timestamp",
    "month",
    "mois",
    "time",
}


# ============================================================
# GENERIC HELPERS
# ============================================================

def normalize_name(
    value: str,
) -> str:
    normalized = (
        str(
            value
        )
        .strip()
        .lower()
    )

    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        normalized,
    )

    return normalized.strip(
        "_"
    )


def name_tokens(
    value: str,
) -> set[
    str
]:
    return {
        token
        for token
        in normalize_name(
            value
        ).split(
            "_"
        )
        if token
    }


def to_native(
    value: Any,
) -> Any:
    """
    Convert NumPy / pandas values into
    JSON-safe native Python objects.
    """

    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                key
            ):
                to_native(
                    item
                )
            for (
                key,
                item,
            )
            in value.items()
        }


    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            to_native(
                item
            )
            for item
            in value
        ]


    if isinstance(
        value,
        np.generic,
    ):
        value = value.item()


    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()


    if isinstance(
        value,
        float,
    ):
        if (
            math.isnan(
                value
            )
            or
            math.isinf(
                value
            )
        ):
            return None


    try:
        if pd.isna(
            value
        ):
            return None

    except (
        TypeError,
        ValueError,
    ):
        pass


    return value


def numeric_series(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def get_analysis_variable(
    analysis: AnalysisCandidate,
    preferred_roles: tuple[
        str,
        ...
    ],
    *,
    fallback_index: (
        int
        | None
    ) = None,
) -> str | None:
    for variable in (
        analysis.variables
    ):
        if (
            variable.role
            in preferred_roles
        ):
            return variable.column


    if (
        fallback_index
        is not None
        and
        len(
            analysis.variables
        )
        >
        fallback_index
    ):
        return (
            analysis
            .variables[
                fallback_index
            ]
            .column
        )


    return None


def validate_columns(
    dataframe: pd.DataFrame,
    columns: list[
        str | None
    ],
) -> list[
    str
]:
    missing: list[
        str
    ] = []


    for column in columns:
        if column is None:
            continue


        if (
            column
            not in dataframe.columns
        ):
            missing.append(
                column
            )


    return missing


# ============================================================
# GROUP / CHART CONSISTENCY
# ============================================================

def validate_group_chart_consistency(
    *,
    expected_group_count: int,
    chart_data: list[
        dict[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    Any,
]:
    """
    Verify that a group-comparison result exposes exactly one
    chart row per computed group.

    This is a fail-closed integrity guard. A report must never
    announce N groups while exposing fewer (or duplicated)
    graphical groups downstream.
    """

    chart_group_keys: list[
        tuple[
            str,
            str,
        ]
    ] = []


    for item in chart_data:
        if not isinstance(
            item,
            dict,
        ):
            continue


        group_value = (
            item.get(
                "group"
            )
        )


        if group_value is None:
            continue


        native_value = (
            to_native(
                group_value
            )
        )


        chart_group_keys.append(
            (
                type(
                    native_value
                ).__name__,
                repr(
                    native_value
                ),
            )
        )


    chart_group_count = len(
        chart_group_keys
    )


    distinct_chart_group_count = len(
        set(
            chart_group_keys
        )
    )


    consistent = (
        expected_group_count
        ==
        chart_group_count
        ==
        distinct_chart_group_count
    )


    return {
        "consistent":
            consistent,

        "expected_group_count":
            int(
                expected_group_count
            ),

        "chart_group_count":
            int(
                chart_group_count
            ),

        "distinct_chart_group_count":
            int(
                distinct_chart_group_count
            ),

        "rule_version":
            GROUP_CHART_CONSISTENCY_RULE_VERSION,
    }


# ============================================================
# STRUCTURE DETECTION
# ============================================================

def detect_temporal_columns(
    dataframe: pd.DataFrame,
) -> list[
    str
]:
    result: list[
        str
    ] = []


    for column in (
        dataframe.columns
    ):
        column_name = str(
            column
        )

        tokens = (
            name_tokens(
                column_name
            )
        )


        if (
            tokens
            &
            TEMPORAL_NAME_SIGNALS
        ):
            result.append(
                column_name
            )

            continue


        if pd.api.types.is_datetime64_any_dtype(
            dataframe[
                column
            ]
        ):
            result.append(
                column_name
            )


    return result


def detect_entity_columns(
    dataframe: pd.DataFrame,
) -> list[
    str
]:
    result: list[
        str
    ] = []


    for column in (
        dataframe.columns
    ):
        column_name = str(
            column
        )

        tokens = (
            name_tokens(
                column_name
            )
        )


        if (
            tokens
            &
            ENTITY_NAME_SIGNALS
        ):
            result.append(
                column_name
            )


    return result


def detect_repeated_measure_structure(
    dataframe: pd.DataFrame,
) -> dict[
    str,
    Any,
] | None:
    """
    Detect a likely panel / longitudinal structure.

    Example:

        Country + Year

    where the same country appears at several
    moments in time.

    This is intentionally conservative.
    """

    temporal_columns = (
        detect_temporal_columns(
            dataframe
        )
    )

    entity_columns = (
        detect_entity_columns(
            dataframe
        )
    )


    if (
        not temporal_columns
        or
        not entity_columns
    ):
        return None


    best_result: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None


    for entity_column in (
        entity_columns
    ):
        for temporal_column in (
            temporal_columns
        ):
            subset = (
                dataframe[
                    [
                        entity_column,
                        temporal_column,
                    ]
                ]
                .dropna()
                .drop_duplicates()
            )


            if subset.empty:
                continue


            period_counts = (
                subset
                .groupby(
                    entity_column,
                    dropna=True,
                )[
                    temporal_column
                ]
                .nunique()
            )


            if period_counts.empty:
                continue


            repeated_entities = int(
                (
                    period_counts
                    >
                    1
                ).sum()
            )

            entity_count = int(
                len(
                    period_counts
                )
            )


            repeated_ratio = (
                repeated_entities
                /
                entity_count
                if entity_count
                else
                0.0
            )


            if (
                repeated_entities
                <
                2
                or
                repeated_ratio
                <
                0.05
            ):
                continue


            candidate = {
                "entity_column":
                    entity_column,

                "temporal_column":
                    temporal_column,

                "entity_count":
                    entity_count,

                "repeated_entity_count":
                    repeated_entities,

                "repeated_entity_ratio":
                    round(
                        repeated_ratio,
                        4,
                    ),
            }


            if (
                best_result
                is None
                or
                repeated_ratio
                >
                best_result[
                    "repeated_entity_ratio"
                ]
            ):
                best_result = (
                    candidate
                )


    return best_result


def detect_additional_grain_dimensions(
    dataframe: pd.DataFrame,
    base_columns: list[
        str
    ],
) -> list[
    str
]:
    """
    Find small categorical columns that help
    explain duplicate observations at a proposed
    analytical grain.

    Example:

        Country + Year

    may still contain duplicates because
    Granularity = Total / Rural / Urban.
    """

    if not base_columns:
        return []


    valid_base = [
        column
        for column
        in base_columns
        if (
            column
            in dataframe.columns
        )
    ]


    if (
        len(
            valid_base
        )
        !=
        len(
            base_columns
        )
    ):
        return []


    base_frame = (
        dataframe[
            valid_base
        ]
        .dropna()
    )


    if base_frame.empty:
        return []


    baseline_duplicates = int(
        base_frame
        .duplicated()
        .sum()
    )


    if baseline_duplicates == 0:
        return []


    candidates: list[
        tuple[
            str,
            int,
        ]
    ] = []


    for column in (
        dataframe.columns
    ):
        column_name = str(
            column
        )


        if (
            column_name
            in valid_base
        ):
            continue


        unique_count = int(
            dataframe[
                column
            ]
            .nunique(
                dropna=True
            )
        )


        if (
            unique_count
            <
            2
            or
            unique_count
            >
            20
        ):
            continue


        test_columns = [
            *valid_base,
            column_name,
        ]


        test_frame = (
            dataframe[
                test_columns
            ]
            .dropna()
        )


        duplicate_count = int(
            test_frame
            .duplicated()
            .sum()
        )


        if (
            duplicate_count
            <
            baseline_duplicates
        ):
            candidates.append(
                (
                    column_name,
                    duplicate_count,
                )
            )


    candidates.sort(
        key=lambda item:
            item[
                1
            ]
    )


    if not candidates:
        return []


    return [
        candidates[
            0
        ][
            0
        ]
    ]


# ============================================================
# RESULT BUILDER
# ============================================================

def build_result(
    analysis: AnalysisCandidate,
    *,
    execution_status: str,
    summary: list[
        str
    ] | None = None,
    metrics: dict[
        str,
        Any,
    ] | None = None,
    chart_data: list[
        dict[
            str,
            Any,
        ]
    ] | None = None,
    statistical_decision: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None,
    statistical_result: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None,
    visualization: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None,
    warnings: list[
        str
    ] | None = None,
    limitations: list[
        str
    ] | None = None,
) -> ExecutedAnalysis:
    return ExecutedAnalysis(
        analysis_id=
            analysis.analysis_id,

        dataset_id=
            analysis.dataset_id,

        dataset_filename=
            analysis.dataset_filename,

        title=
            analysis.title,

        family=
            analysis.family,

        planned_readiness=
            analysis.readiness,

        execution_status=
            execution_status,

        chart_type=
            analysis.chart_type,

        summary=
            summary
            or [],

        metrics=
            to_native(
                metrics
                or {}
            ),

        chart_data=
            to_native(
                chart_data
                or []
            ),

        statistical_decision=(
            to_native(
                statistical_decision
            )
            if statistical_decision
            is not None
            else None
        ),

        statistical_result=(
            to_native(
                statistical_result
            )
            if statistical_result
            is not None
            else None
        ),

        visualization=(
            to_native(
                visualization
            )
            if visualization
            is not None
            else None
        ),

        warnings=
            warnings
            or [],

        limitations=[
            *analysis.limitations,
            *(
                limitations
                or []
            ),
        ],
    )


# ============================================================
# DISTRIBUTION
# ============================================================

def execute_distribution(
    analysis: AnalysisCandidate,
    dataframe: pd.DataFrame,
) -> ExecutedAnalysis:
    value_column = (
        get_analysis_variable(
            analysis,
            (
                "value",
                "x",
            ),
            fallback_index=0,
        )
    )


    if value_column is None:
        return build_result(
            analysis,
            execution_status=
                "failed",
            warnings=[
                (
                    "No quantitative variable "
                    "was available for the "
                    "distribution analysis."
                )
            ],
        )


    missing = validate_columns(
        dataframe,
        [
            value_column
        ],
    )


    if missing:
        return build_result(
            analysis,
            execution_status=
                "failed",
            warnings=[
                (
                    "The planned variable is "
                    "not present in the dataset."
                )
            ],
        )


    values = numeric_series(
        dataframe[
            value_column
        ]
    )


    clean = (
        values
        .dropna()
        .astype(
            float
        )
    )


    if clean.empty:
        return build_result(
            analysis,
            execution_status=
                "skipped",
            warnings=[
                (
                    "No valid numeric values "
                    "were available."
                )
            ],
        )


    count = int(
        clean.count()
    )

    missing_count = int(
        values.isna().sum()
    )


    q1 = float(
        clean.quantile(
            0.25
        )
    )

    median = float(
        clean.median()
    )

    q3 = float(
        clean.quantile(
            0.75
        )
    )

    iqr = (
        q3
        -
        q1
    )


    lower_bound = (
        q1
        -
        1.5
        *
        iqr
    )

    upper_bound = (
        q3
        +
        1.5
        *
        iqr
    )


    outlier_count = int(
        (
            (
                clean
                <
                lower_bound
            )
            |
            (
                clean
                >
                upper_bound
            )
        ).sum()
    )


    bin_count = min(
        40,
        max(
            5,
            int(
                math.sqrt(
                    count
                )
            ),
        ),
    )


    histogram_counts, edges = (
        np.histogram(
            clean.to_numpy(),
            bins=
                bin_count,
        )
    )


    chart_data: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for index in range(
        len(
            histogram_counts
        )
    ):
        chart_data.append(
            {
                "bin_start":
                    float(
                        edges[
                            index
                        ]
                    ),

                "bin_end":
                    float(
                        edges[
                            index
                            +
                            1
                        ]
                    ),

                "count":
                    int(
                        histogram_counts[
                            index
                        ]
                    ),
            }
        )


    metrics = {
        "column":
            value_column,

        "count":
            count,

        "missing_count":
            missing_count,

        "mean":
            float(
                clean.mean()
            ),

        "median":
            median,

        "std":
            (
                float(
                    clean.std(
                        ddof=1
                    )
                )
                if count
                >
                1
                else None
            ),

        "min":
            float(
                clean.min()
            ),

        "q1":
            q1,

        "q3":
            q3,

        "max":
            float(
                clean.max()
            ),

        "iqr":
            iqr,

        "outlier_count_iqr":
            outlier_count,

        "outlier_ratio_iqr":
            (
                outlier_count
                /
                count
                if count
                else 0.0
            ),

        "skewness":
            (
                float(
                    clean.skew()
                )
                if count
                >
                2
                else None
            ),
    }


    summary = [
        (
            f"{count} valeur(s) valides "
            f"ont été analysées pour "
            f"{value_column}."
        ),

        (
            "La médiane est "
            f"{median:.4g} et l'intervalle "
            "interquartile s'étend de "
            f"{q1:.4g} à {q3:.4g}."
        ),

        (
            f"{outlier_count} observation(s) "
            "se situent en dehors des bornes "
            "IQR 1,5."
        ),
    ]


    return build_result(
        analysis,
        execution_status=
            "complete",

        summary=
            summary,

        metrics=
            metrics,

        chart_data=
            chart_data,
    )


# ============================================================
# GROUP COMPARISON
# ============================================================

def execute_group_comparison(
    analysis: AnalysisCandidate,
    dataframe: pd.DataFrame,
) -> ExecutedAnalysis:
    group_column = (
        get_analysis_variable(
            analysis,
            (
                "group",
                "category",
                "x",
            ),
            fallback_index=0,
        )
    )

    value_column = (
        get_analysis_variable(
            analysis,
            (
                "value",
                "y",
            ),
            fallback_index=1,
        )
    )


    if (
        group_column is None
        or
        value_column is None
    ):
        return build_result(
            analysis,
            execution_status=
                "failed",
            warnings=[
                (
                    "The group comparison "
                    "does not contain both a "
                    "group and a quantitative "
                    "variable."
                )
            ],
        )


    missing = validate_columns(
        dataframe,
        [
            group_column,
            value_column,
        ],
    )


    if missing:
        return build_result(
            analysis,
            execution_status=
                "failed",
            warnings=[
                (
                    "One or more planned "
                    "variables are missing "
                    "from the dataset."
                )
            ],
        )


    working = pd.DataFrame(
        {
            "group":
                dataframe[
                    group_column
                ],

            "value":
                numeric_series(
                    dataframe[
                        value_column
                    ]
                ),
        }
    ).dropna()


    if working.empty:
        return build_result(
            analysis,
            execution_status=
                "skipped",
            warnings=[
                (
                    "No complete group/value "
                    "observations were available."
                )
            ],
        )


    group_count = int(
        working[
            "group"
        ]
        .nunique()
    )


    if (
        group_count
        >
        MAX_GROUPS_FOR_TABLE
    ):
        return build_result(
            analysis,
            execution_status=
                "skipped",

            metrics={
                "group_count":
                    group_count,
            },

            warnings=[
                (
                    f"{group_count} groups were "
                    "detected. The current group "
                    "comparison executor limits "
                    "automatic summary tables to "
                    f"{MAX_GROUPS_FOR_TABLE} groups."
                )
            ],
        )


    chart_data: list[
        dict[
            str,
            Any,
        ]
    ] = []


    grouped = (
        working
        .groupby(
            "group",
            dropna=True,
        )[
            "value"
        ]
    )


    for (
        group,
        values,
    ) in grouped:
        clean = (
            values
            .dropna()
            .astype(
                float
            )
        )


        if clean.empty:
            continue


        chart_data.append(
            {
                "group":
                    to_native(
                        group
                    ),

                "count":
                    int(
                        clean.count()
                    ),

                "mean":
                    float(
                        clean.mean()
                    ),

                "median":
                    float(
                        clean.median()
                    ),

                "std":
                    (
                        float(
                            clean.std(
                                ddof=1
                            )
                        )
                        if len(
                            clean
                        )
                        >
                        1
                        else None
                    ),

                "min":
                    float(
                        clean.min()
                    ),

                "q1":
                    float(
                        clean.quantile(
                            0.25
                        )
                    ),

                "q3":
                    float(
                        clean.quantile(
                            0.75
                        )
                    ),

                "max":
                    float(
                        clean.max()
                    ),
            }
        )


    group_chart_consistency = (
        validate_group_chart_consistency(
            expected_group_count=
                group_count,

            chart_data=
                chart_data,
        )
    )


    if not (
        group_chart_consistency[
            "consistent"
        ]
    ):
        return build_result(
            analysis,
            execution_status=
                "failed",

            summary=[
                (
                    "Une incohérence interne a été détectée "
                    "entre les groupes calculés et les données "
                    "destinées à la visualisation."
                )
            ],

            metrics={
                "group_column":
                    group_column,

                "value_column":
                    value_column,

                "group_count":
                    group_count,

                "chart_group_count":
                    group_chart_consistency[
                        "chart_group_count"
                    ],

                "distinct_chart_group_count":
                    group_chart_consistency[
                        "distinct_chart_group_count"
                    ],

                "valid_observations":
                    int(
                        len(
                            working
                        )
                    ),

                "group_chart_consistency":
                    False,

                "group_chart_consistency_rule_version":
                    GROUP_CHART_CONSISTENCY_RULE_VERSION,
            },

            chart_data=[],

            warnings=[
                (
                    "DataLens a bloqué ce résultat : "
                    f"{group_count} groupe(s) ont été calculés, "
                    "mais la représentation graphique n'expose "
                    f"que {group_chart_consistency['chart_group_count']} "
                    "groupe(s) et "
                    f"{group_chart_consistency['distinct_chart_group_count']} "
                    "libellé(s) distinct(s)."
                ),

                (
                    "Le résultat n'est pas transmis au rapport "
                    "tant que cette incohérence n'est pas résolue."
                ),
            ],
        )


    panel_structure = (
        detect_repeated_measure_structure(
            dataframe
        )
    )


    warnings = [
        (
            "This execution compares group "
            "distributions descriptively. "
            "No inferential group-comparison "
            "test has been applied yet."
        )
    ]


    if panel_structure:
        warnings.append(
            (
                "Repeated observations over "
                "time were detected. A simple "
                "independent-groups inferential "
                "test would therefore require "
                "additional design checks."
            )
        )


    summary = [
        (
            f"{group_count} groupe(s) ont été "
            f"comparés pour {value_column}."
        ),

        (
            "DataLens calculated the sample "
            "size, mean, median, dispersion "
            "and quartiles for each group."
        ),
    ]


    return build_result(
        analysis,
        execution_status=
            "descriptive_only",

        summary=
            summary,

        metrics={
            "group_column":
                group_column,

            "value_column":
                value_column,

            "group_count":
                group_count,

            "chart_group_count":
                group_chart_consistency[
                    "chart_group_count"
                ],

            "group_chart_consistency":
                True,

            "group_chart_consistency_rule_version":
                GROUP_CHART_CONSISTENCY_RULE_VERSION,

            "valid_observations":
                int(
                    len(
                        working
                    )
                ),

            "repeated_measure_structure":
                panel_structure,
        },

        chart_data=
            chart_data,

        warnings=
            warnings,
    )


# ============================================================
# TIME SERIES
# ============================================================

def execute_time_series(
    analysis: AnalysisCandidate,
    dataframe: pd.DataFrame,
) -> ExecutedAnalysis:
    time_column = (
        get_analysis_variable(
            analysis,
            (
                "time",
                "date",
            ),
            fallback_index=0,
        )
    )

    value_column = (
        get_analysis_variable(
            analysis,
            (
                "value",
                "y",
            ),
            fallback_index=1,
        )
    )

    group_column = (
        get_analysis_variable(
            analysis,
            (
                "group",
            ),
        )
    )


    if (
        time_column is None
        or
        value_column is None
    ):
        return build_result(
            analysis,
            execution_status=
                "failed",
            warnings=[
                (
                    "The time-series analysis "
                    "does not contain both a "
                    "time variable and a value."
                )
            ],
        )


    missing = validate_columns(
        dataframe,
        [
            time_column,
            value_column,
            group_column,
        ],
    )


    if missing:
        return build_result(
            analysis,
            execution_status=
                "failed",
            warnings=[
                (
                    "One or more time-series "
                    "variables are missing."
                )
            ],
        )


    working_data: dict[
        str,
        pd.Series,
    ] = {
        "time":
            dataframe[
                time_column
            ],

        "value":
            numeric_series(
                dataframe[
                    value_column
                ]
            ),
    }


    if group_column:
        working_data[
            "group"
        ] = dataframe[
            group_column
        ]


    working = (
        pd.DataFrame(
            working_data
        )
        .dropna(
            subset=[
                "time",
                "value",
            ]
        )
    )


    if working.empty:
        return build_result(
            analysis,
            execution_status=
                "skipped",
            warnings=[
                (
                    "No valid time/value "
                    "observations were available."
                )
            ],
        )


    period_count = int(
        working[
            "time"
        ]
        .nunique()
    )


    group_count = (
        int(
            working[
                "group"
            ]
            .nunique()
        )
        if (
            group_column
            and
            "group"
            in working
        )
        else 1
    )


    base_grain = []


    if group_column:
        base_grain.append(
            group_column
        )


    base_grain.append(
        time_column
    )


    grain_dimensions = (
        detect_additional_grain_dimensions(
            dataframe,
            base_grain,
        )
    )


    warnings: list[
        str
    ] = []


    if grain_dimensions:
        warnings.append(
            (
                "Additional grain dimensions "
                "were detected: "
                +
                ", ".join(
                    grain_dimensions
                )
                +
                ". DataLens will not collapse "
                "these observations into a "
                "single time series automatically."
            )
        )


    if (
        group_count
        >
        MAX_GROUPS_FOR_LINE_CHART
    ):
        warnings.append(
            (
                f"{group_count} series/groups "
                "were detected. A single line "
                "chart would be too dense, so "
                "the report should use filtering, "
                "faceting or a selected subset."
            )
        )


    chart_data: list[
        dict[
            str,
            Any,
        ]
    ] = []


    can_emit_chart_data = (
        not grain_dimensions
        and
        group_count
        <=
        MAX_GROUPS_FOR_LINE_CHART
    )


    if can_emit_chart_data:
        sorted_working = (
            working
            .sort_values(
                "time"
            )
        )


        if (
            len(
                sorted_working
            )
            >
            MAX_CHART_POINTS
        ):
            indexes = np.linspace(
                0,
                len(
                    sorted_working
                )
                -
                1,
                MAX_CHART_POINTS,
                dtype=int,
            )

            sorted_working = (
                sorted_working.iloc[
                    indexes
                ]
            )


        for _, row in (
            sorted_working
            .iterrows()
        ):
            item = {
                "time":
                    to_native(
                        row[
                            "time"
                        ]
                    ),

                "value":
                    to_native(
                        row[
                            "value"
                        ]
                    ),
            }


            if (
                group_column
                and
                "group"
                in row.index
            ):
                item[
                    "group"
                ] = to_native(
                    row[
                        "group"
                    ]
                )


            chart_data.append(
                item
            )


    time_values = (
        working[
            "time"
        ]
        .dropna()
    )


    try:
        time_start = (
            time_values.min()
        )

        time_end = (
            time_values.max()
        )

    except TypeError:
        sorted_times = sorted(
            {
                str(
                    value
                )
                for value
                in time_values
            }
        )

        time_start = (
            sorted_times[
                0
            ]
            if sorted_times
            else None
        )

        time_end = (
            sorted_times[
                -1
            ]
            if sorted_times
            else None
        )


    panel_structure = (
        detect_repeated_measure_structure(
            dataframe
        )
    )


    summary = [
        (
            f"{period_count} période(s) "
            f"distincte(s) ont été détectées "
            f"pour {value_column}."
        )
    ]


    if group_column:
        summary.append(
            (
                f"{group_count} groupe(s) "
                f"distinct(s) ont été détectés "
                f"via {group_column}."
            )
        )


    if grain_dimensions:
        summary.append(
            (
                "The dataset contains a finer "
                "observational grain than the "
                "initial Country/Time structure."
            )
        )


    execution_status = (
        "descriptive_only"
        if (
            grain_dimensions
            or
            group_count
            >
            MAX_GROUPS_FOR_LINE_CHART
        )
        else
        "complete"
    )


    return build_result(
        analysis,
        execution_status=
            execution_status,

        summary=
            summary,

        metrics={
            "time_column":
                time_column,

            "value_column":
                value_column,

            "group_column":
                group_column,

            "period_count":
                period_count,

            "group_count":
                group_count,

            "valid_observations":
                int(
                    len(
                        working
                    )
                ),

            "time_start":
                to_native(
                    time_start
                ),

            "time_end":
                to_native(
                    time_end
                ),

            "value_min":
                float(
                    working[
                        "value"
                    ].min()
                ),

            "value_median":
                float(
                    working[
                        "value"
                    ].median()
                ),

            "value_max":
                float(
                    working[
                        "value"
                    ].max()
                ),

            "additional_grain_dimensions":
                grain_dimensions,

            "repeated_measure_structure":
                panel_structure,
        },

        chart_data=
            chart_data,

        warnings=
            warnings,
    )


# ============================================================
# QUANTITATIVE ASSOCIATION
# ============================================================

def execute_quantitative_association(
    analysis: AnalysisCandidate,
    dataframe: pd.DataFrame,
) -> ExecutedAnalysis:
    x_column = (
        get_analysis_variable(
            analysis,
            (
                "x",
            ),
            fallback_index=0,
        )
    )

    y_column = (
        get_analysis_variable(
            analysis,
            (
                "y",
            ),
            fallback_index=1,
        )
    )


    if (
        x_column is None
        or
        y_column is None
    ):
        return build_result(
            analysis,
            execution_status=
                "failed",
            warnings=[
                (
                    "The quantitative "
                    "association does not "
                    "contain two variables."
                )
            ],
        )


    missing = validate_columns(
        dataframe,
        [
            x_column,
            y_column,
        ],
    )


    if missing:
        return build_result(
            analysis,
            execution_status=
                "failed",
            warnings=[
                (
                    "One or more association "
                    "variables are missing."
                )
            ],
        )


    pair_frame = pd.DataFrame(
        {
            x_column:
                numeric_series(
                    dataframe[
                        x_column
                    ]
                ),

            y_column:
                numeric_series(
                    dataframe[
                        y_column
                    ]
                ),
        }
    ).dropna()


    if (
        len(
            pair_frame
        )
        <
        3
    ):
        return build_result(
            analysis,
            execution_status=
                "skipped",
            warnings=[
                (
                    "Fewer than three complete "
                    "paired observations are "
                    "available."
                )
            ],
        )


    panel_structure = (
        detect_repeated_measure_structure(
            dataframe
        )
    )


    if panel_structure:
        # ----------------------------------------------------
        # Descriptive visualization remains valid even when
        # classical inferential correlation is not.
        #
        # Important invariant:
        # - keep the repeated-measures guard;
        # - do not execute Pearson/Spearman automatically;
        # - still expose the observed x/y pairs so the user can
        #   inspect the relationship descriptively.
        # ----------------------------------------------------

        chart_source = (
            pair_frame
            .copy()
        )


        if (
            len(
                chart_source
            )
            >
            MAX_CHART_POINTS
        ):
            indexes = np.linspace(
                0,
                len(
                    chart_source
                )
                -
                1,
                MAX_CHART_POINTS,
                dtype=int,
            )

            chart_source = (
                chart_source.iloc[
                    indexes
                ]
            )


        chart_data = [
            {
                "x":
                    float(
                        row[
                            x_column
                        ]
                    ),

                "y":
                    float(
                        row[
                            y_column
                        ]
                    ),
            }

            for _, row
            in chart_source.iterrows()
        ]


        return build_result(
            analysis,
            execution_status=
                "needs_specialized_method",

            summary=[
                (
                    "Cette relation est "
                    "potentiellement intéressante, "
                    "mais les observations ne "
                    "peuvent pas être considérées "
                    "comme simplement indépendantes."
                ),

                (
                    "DataLens a détecté une "
                    "structure longitudinale ou "
                    "de panel dans le dataset."
                ),

                (
                    "Le nuage de points reste "
                    "affiché à titre descriptif ; "
                    "aucune corrélation inférentielle "
                    "classique n'est interprétée "
                    "automatiquement."
                ),
            ],

            metrics={
                "x_column":
                    x_column,

                "y_column":
                    y_column,

                "valid_pairs":
                    int(
                        len(
                            pair_frame
                        )
                    ),

                "chart_point_count":
                    int(
                        len(
                            chart_data
                        )
                    ),

                "inference_performed":
                    False,

                "interpretation_scope":
                    "descriptive_only",

                "repeated_measure_structure":
                    panel_structure,
            },

            chart_data=
                chart_data,

            warnings=[
                (
                    "A simple Pearson or Spearman "
                    "test over all rows could "
                    "underestimate dependence "
                    "between observations."
                ),

                (
                    "A panel-aware, repeated-"
                    "measures or appropriately "
                    "aggregated analysis should "
                    "be considered before "
                    "inferential execution."
                ),
            ],

            limitations=[
                (
                    "No inferential correlation "
                    "was executed automatically."
                ),

                (
                    "The scatter plot is descriptive "
                    "only and must not be interpreted "
                    "as evidence from an independent-"
                    "observations correlation test."
                ),
            ],
        )


    decision = (
        decide_correlation_test(
            dataframe=
                pair_frame,

            x_column=
                x_column,

            y_column=
                y_column,

            analysis_goal=
                "general_association",

            analysis_mode=
                "exploratory",

            x_kind=
                "continuous",

            y_kind=
                "continuous",

            observations_independent=
                True,
        )
    )


    visualization = (
        decide_correlation_visualization(
            decision
        )
    )


    decision_dump = (
        decision.model_dump(
            mode="json"
        )
    )

    visualization_dump = (
        visualization.model_dump(
            mode="json"
        )
    )


    if (
        decision.status
        !=
        "selected"
    ):
        # ====================================================
        # DESCRIPTIVE CORRELATION FALLBACK v0.1
        #
        # Statistical authority remains unchanged:
        #
        # - decision.status is still not selected;
        # - execute_correlation_decision() is NOT called;
        # - no p-value is calculated;
        # - no inferential test is promoted.
        #
        # We still quantify the observed relationship
        # descriptively and preserve the requested scatter.
        # ====================================================

        descriptive_warnings: list[
            str
        ] = []


        x_values = (
            pair_frame[
                x_column
            ]
            .to_numpy(
                dtype=float
            )
        )

        y_values = (
            pair_frame[
                y_column
            ]
            .to_numpy(
                dtype=float
            )
        )


        try:
            pearson_r = float(
                calculate_pearson_statistic(
                    x_values,
                    y_values,
                )
            )


            if not np.isfinite(
                pearson_r
            ):
                pearson_r = None


        except Exception as error:
            pearson_r = None

            descriptive_warnings.append(
                (
                    "Le coefficient descriptif "
                    "de Pearson n'a pas pu être "
                    "calculé : "
                    f"{type(error).__name__}: "
                    f"{error}"
                )
            )


        try:
            spearman_rho = float(
                calculate_spearman_statistic(
                    x_values,
                    y_values,
                )
            )


            if not np.isfinite(
                spearman_rho
            ):
                spearman_rho = None


        except Exception as error:
            spearman_rho = None

            descriptive_warnings.append(
                (
                    "Le coefficient descriptif "
                    "de Spearman n'a pas pu être "
                    "calculé : "
                    f"{type(error).__name__}: "
                    f"{error}"
                )
            )


        chart_source = (
            pair_frame
            .copy()
        )


        if (
            len(
                chart_source
            )
            >
            MAX_CHART_POINTS
        ):
            indexes = np.linspace(
                0,
                len(
                    chart_source
                )
                -
                1,
                MAX_CHART_POINTS,
                dtype=int,
            )


            chart_source = (
                chart_source.iloc[
                    indexes
                ]
            )


        chart_data = [
            {
                "x":
                    float(
                        row[
                            x_column
                        ]
                    ),

                "y":
                    float(
                        row[
                            y_column
                        ]
                    ),
            }

            for _, row
            in chart_source.iterrows()
        ]


        summary = [
            (
                "DataLens n'a pas sélectionné "
                "automatiquement de test de "
                "corrélation inférentiel pour "
                "cette relation."
            ),

            (
                "La relation reste quantifiée "
                "de manière descriptive et le "
                "nuage de points reste disponible."
            ),
        ]


        if (
            pearson_r
            is not None
        ):
            summary.append(
                (
                    "Pearson r descriptif = "
                    f"{pearson_r:.4g}."
                )
            )


        if (
            spearman_rho
            is not None
        ):
            summary.append(
                (
                    "Spearman rho descriptif = "
                    f"{spearman_rho:.4g}."
                )
            )


        return build_result(
            analysis,

            execution_status=
                "needs_information",

            summary=
                summary,

            metrics={
                "x_column":
                    x_column,

                "y_column":
                    y_column,

                "valid_pairs":
                    int(
                        len(
                            pair_frame
                        )
                    ),

                "chart_point_count":
                    int(
                        len(
                            chart_data
                        )
                    ),

                "pearson_r":
                    pearson_r,

                "spearman_rho":
                    spearman_rho,

                "inference_performed":
                    False,

                "p_value":
                    None,

                "statistically_significant":
                    None,

                "interpretation_scope":
                    "descriptive_only",

                "descriptive_fallback_rule_version":
                    "descriptive_correlation_fallback_v0.1",
            },

            chart_data=
                chart_data,

            statistical_decision=
                decision_dump,

            statistical_result=
                None,

            visualization=
                visualization_dump,

            warnings=[
                *decision.warnings,
                *descriptive_warnings,
            ],

            limitations=[
                (
                    "The current statistical "
                    "decision engine did not "
                    "select a sufficiently "
                    "defensible inferential "
                    "correlation test."
                ),

                (
                    "Pearson r and Spearman rho "
                    "are reported as descriptive "
                    "association measures only."
                ),

                (
                    "No p-value is calculated or "
                    "interpreted for this "
                    "descriptive fallback."
                ),
            ],
        )



    execution = (
        execute_correlation_decision(
            dataframe=
                pair_frame,

            decision=
                decision,
        )
    )


    execution_dump = (
        execution.model_dump(
            mode="json"
        )
    )


    statistical_result = (
        execution_dump.get(
            "result"
        )
    )


    coefficient = (
        statistical_result.get(
            "coefficient"
        )
        if statistical_result
        else None
    )

    p_value = (
        statistical_result.get(
            "p_value"
        )
        if statistical_result
        else None
    )

    test_name = (
        statistical_result.get(
            "test"
        )
        if statistical_result
        else None
    )


    chart_source = (
        pair_frame
        .copy()
    )


    if (
        len(
            chart_source
        )
        >
        MAX_CHART_POINTS
    ):
        indexes = np.linspace(
            0,
            len(
                chart_source
            )
            -
            1,
            MAX_CHART_POINTS,
            dtype=int,
        )

        chart_source = (
            chart_source.iloc[
                indexes
            ]
        )


    chart_data = [
        {
            "x":
                float(
                    row[
                        x_column
                    ]
                ),

            "y":
                float(
                    row[
                        y_column
                    ]
                ),
        }
        for _, row
        in chart_source.iterrows()
    ]


    summary = [
        (
            f"{test_name} a été sélectionné "
            f"automatiquement pour étudier "
            f"{x_column} et {y_column}."
        )
    ]


    if coefficient is not None:
        summary.append(
            (
                "Le coefficient calculé est "
                f"{float(coefficient):.4g}."
            )
        )


    if p_value is not None:
        summary.append(
            (
                "La p-value calculée est "
                f"{float(p_value):.4g}."
            )
        )


    return build_result(
        analysis,
        execution_status=
            "complete",

        summary=
            summary,

        metrics={
            "x_column":
                x_column,

            "y_column":
                y_column,

            "valid_pairs":
                int(
                    len(
                        pair_frame
                    )
                ),

            "test":
                test_name,

            "coefficient":
                coefficient,

            "p_value":
                p_value,

            "statistically_significant": (
                statistical_result.get(
                    "statistically_significant"
                )
                if statistical_result
                else None
            ),
        },

        chart_data=
            chart_data,

        statistical_decision=
            decision_dump,

        statistical_result=
            statistical_result,

        visualization=
            visualization_dump,

        warnings=[
            *execution.warnings,
        ],
    )


# ============================================================
# CATEGORICAL ASSOCIATION
# ============================================================

def execute_categorical_association(
    analysis: AnalysisCandidate,
    dataframe: pd.DataFrame,
) -> ExecutedAnalysis:
    x_column = (
        get_analysis_variable(
            analysis,
            (
                "x",
                "category_x",
                "category_a",
            ),
            fallback_index=0,
        )
    )

    y_column = (
        get_analysis_variable(
            analysis,
            (
                "y",
                "category_y",
                "category_b",
            ),
            fallback_index=1,
        )
    )


    if (
        x_column is None
        or
        y_column is None
    ):
        return build_result(
            analysis,
            execution_status=
                "failed",
            warnings=[
                (
                    "The categorical association "
                    "does not contain two "
                    "categorical variables."
                )
            ],
        )


    missing = validate_columns(
        dataframe,
        [
            x_column,
            y_column,
        ],
    )


    if missing:
        return build_result(
            analysis,
            execution_status=
                "failed",
            warnings=[
                (
                    "One or more categorical "
                    "variables are missing."
                )
            ],
        )


    working = (
        dataframe[
            [
                x_column,
                y_column,
            ]
        ]
        .dropna()
    )


    if working.empty:
        return build_result(
            analysis,
            execution_status=
                "skipped",
            warnings=[
                (
                    "No complete categorical "
                    "observations are available."
                )
            ],
        )


    x_levels = int(
        working[
            x_column
        ]
        .nunique()
    )

    y_levels = int(
        working[
            y_column
        ]
        .nunique()
    )


    if (
        x_levels
        >
        MAX_CATEGORICAL_LEVELS
        or
        y_levels
        >
        MAX_CATEGORICAL_LEVELS
    ):
        return build_result(
            analysis,
            execution_status=
                "skipped",

            metrics={
                "x_levels":
                    x_levels,

                "y_levels":
                    y_levels,
            },

            warnings=[
                (
                    "Too many categorical "
                    "levels were detected for "
                    "a reliable automatic "
                    "contingency-table analysis."
                )
            ],
        )


    contingency = pd.crosstab(
        working[
            x_column
        ],
        working[
            y_column
        ],
    )


    if (
        contingency.shape[
            0
        ]
        <
        2
        or
        contingency.shape[
            1
        ]
        <
        2
    ):
        return build_result(
            analysis,
            execution_status=
                "skipped",
            warnings=[
                (
                    "At least two levels are "
                    "required for both variables."
                )
            ],
        )


    chart_data: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for x_value in (
        contingency.index
    ):
        for y_value in (
            contingency.columns
        ):
            chart_data.append(
                {
                    "x":
                        str(
                            x_value
                        ),

                    "y":
                        str(
                            y_value
                        ),

                    "count":
                        int(
                            contingency.loc[
                                x_value,
                                y_value,
                            ]
                        ),
                }
            )


    panel_structure = (
        detect_repeated_measure_structure(
            dataframe
        )
    )


    if panel_structure:
        return build_result(
            analysis,
            execution_status=
                "descriptive_only",

            summary=[
                (
                    "A contingency table was "
                    "constructed, but DataLens "
                    "did not execute a classical "
                    "chi-square test because "
                    "repeated observations were "
                    "detected."
                )
            ],

            metrics={
                "x_column":
                    x_column,

                "y_column":
                    y_column,

                "x_levels":
                    x_levels,

                "y_levels":
                    y_levels,

                "valid_observations":
                    int(
                        len(
                            working
                        )
                    ),

                "repeated_measure_structure":
                    panel_structure,
            },

            chart_data=
                chart_data,

            warnings=[
                (
                    "The independence assumption "
                    "required by a classical "
                    "chi-square test may not hold."
                )
            ],
        )


    (
        chi2,
        p_value,
        dof,
        expected,
    ) = chi2_contingency(
        contingency
    )


    n = int(
        contingency
        .to_numpy()
        .sum()
    )


    rows, columns = (
        contingency.shape
    )


    denominator = min(
        rows
        -
        1,
        columns
        -
        1,
    )


    cramers_v = (
        math.sqrt(
            (
                chi2
                /
                n
            )
            /
            denominator
        )
        if (
            n
            >
            0
            and
            denominator
            >
            0
        )
        else None
    )


    expected_min = float(
        np.min(
            expected
        )
    )


    warnings: list[
        str
    ] = []


    if (
        expected_min
        <
        5
    ):
        warnings.append(
            (
                "At least one expected "
                "contingency-table frequency "
                "is below 5. The chi-square "
                "approximation should be "
                "interpreted cautiously."
            )
        )


    statistical_result = {
        "test":
            "chi_square_independence",

        "chi2":
            float(
                chi2
            ),

        "p_value":
            float(
                p_value
            ),

        "degrees_of_freedom":
            int(
                dof
            ),

        "cramers_v":
            (
                float(
                    cramers_v
                )
                if cramers_v
                is not None
                else None
            ),

        "n":
            n,

        "expected_min":
            expected_min,

        "alpha":
            0.05,

        "statistically_significant":
            bool(
                p_value
                <
                0.05
            ),
    }


    return build_result(
        analysis,
        execution_status=
            "complete",

        summary=[
            (
                "A chi-square independence "
                "test was executed for "
                f"{x_column} and {y_column}."
            ),

            (
                "Cramer's V was calculated "
                "to quantify the strength "
                "of categorical association."
            ),
        ],

        metrics={
            "x_column":
                x_column,

            "y_column":
                y_column,

            "x_levels":
                x_levels,

            "y_levels":
                y_levels,

            "valid_observations":
                int(
                    len(
                        working
                    )
                ),
        },

        chart_data=
            chart_data,

        statistical_result=
            statistical_result,

        warnings=
            warnings,
    )


# ============================================================
# DISPATCH
# ============================================================

def execute_analysis_candidate(
    analysis: AnalysisCandidate,
    dataframe: pd.DataFrame,
) -> ExecutedAnalysis:
    try:
        if (
            analysis.family
            ==
            "distribution"
        ):
            return execute_distribution(
                analysis,
                dataframe,
            )


        if (
            analysis.family
            ==
            "group_comparison"
        ):
            return execute_group_comparison(
                analysis,
                dataframe,
            )


        if (
            analysis.family
            ==
            "time_series"
        ):
            return execute_time_series(
                analysis,
                dataframe,
            )


        if (
            analysis.family
            ==
            "quantitative_association"
        ):
            return (
                execute_quantitative_association(
                    analysis,
                    dataframe,
                )
            )


        if (
            analysis.family
            ==
            "categorical_association"
        ):
            return (
                execute_categorical_association(
                    analysis,
                    dataframe,
                )
            )


        return build_result(
            analysis,
            execution_status=
                "skipped",

            warnings=[
                (
                    "No executor is currently "
                    "registered for this "
                    f"analysis family: "
                    f"{analysis.family}."
                )
            ],
        )


    except Exception as error:
        return build_result(
            analysis,
            execution_status=
                "failed",

            warnings=[
                (
                    "The analysis executor "
                    "encountered an unexpected "
                    "error."
                ),

                (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            ],
        )


# ============================================================
# PLAN EXECUTOR
# ============================================================

def execute_analysis_plan(
    *,
    plan: AnalysisPlanReport,
    datasets: dict[
        str,
        pd.DataFrame,
    ],
) -> AnalysisExecutionReport:
    results: list[
        ExecutedAnalysis
    ] = []


    for analysis in (
        plan.recommended_analyses
    ):
        dataframe = (
            datasets.get(
                analysis.dataset_id
            )
        )


        if dataframe is None:
            results.append(
                build_result(
                    analysis,
                    execution_status=
                        "failed",

                    warnings=[
                        (
                            "The dataframe "
                            "associated with this "
                            "dataset ID was not "
                            "provided to the "
                            "executor."
                        )
                    ],
                )
            )

            continue


        results.append(
            execute_analysis_candidate(
                analysis,
                dataframe,
            )
        )


    def count_status(
        status: str,
    ) -> int:
        return sum(
            1
            for result
            in results
            if (
                result.execution_status
                ==
                status
            )
        )


    return AnalysisExecutionReport(
        candidate_count=
            len(
                results
            ),

        complete_count=
            count_status(
                "complete"
            ),

        descriptive_only_count=
            count_status(
                "descriptive_only"
            ),

        needs_information_count=
            count_status(
                "needs_information"
            ),

        needs_specialized_method_count=
            count_status(
                "needs_specialized_method"
            ),

        skipped_count=
            count_status(
                "skipped"
            ),

        failed_count=
            count_status(
                "failed"
            ),

        results=
            results,

        executor_notes=[
            (
                "All planner candidates were "
                "evaluated by the execution "
                "layer rather than only those "
                "previously marked executable."
            ),

            (
                "Descriptive execution does "
                "not imply that an inferential "
                "statistical test was appropriate."
            ),

            (
                "DataLens checks for repeated "
                "entity/time observations before "
                "running simple independent-row "
                "association tests."
            ),

            (
                "Cross-dataset analyses are "
                "not executed by this version. "
                "They will be enabled after "
                "validated relationship and "
                "grain alignment."
            ),
        ],
    )