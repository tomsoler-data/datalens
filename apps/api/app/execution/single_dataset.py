from __future__ import annotations

import math
import re
import unicodedata

from typing import (
    Any,
)

import numpy as np

import pandas as pd

from scipy.stats import (
    chi2_contingency,
    spearmanr,
)

from app.discovery.schemas import (
    AnalysisDiscoveryReport,
    DiscoveredAnalysis,
    DiscoveredVariable,
)

from app.discovery.validator import (
    infer_measure_unit,
)

from app.execution.single_schemas import (
    SingleDatasetExecutedAnalysis,
    SingleDatasetExecutionReport,
)

from app.execution.structure import (
    detect_observation_structure,
    find_explicit_total_slice,
)


# ============================================================
# CONSTANTS
# ============================================================

MIN_VALID_NUMERIC = 5

MIN_VALID_ASSOCIATION = 20

MIN_VALID_PERIOD_ASSOCIATION = 15

MIN_VALID_CATEGORICAL = 20

MAX_CHART_POINTS = 2000

MAX_RANKING_ROWS = 20


GEOGRAPHIC_SIGNALS = {
    "country",
    "pays",
    "region",
    "continent",
    "state",
    "province",
    "city",
    "ville",
    "territory",
    "territoire",
}


# ============================================================
# GENERIC HELPERS
# ============================================================

def bounded_chart_frame(
    dataframe: pd.DataFrame,
    *,
    max_points: int = MAX_CHART_POINTS,
) -> pd.DataFrame:
    """
    Return a deterministic bounded view for chart rendering.

    Analytical calculations continue to use the complete
    dataframe. Only the visualization payload is bounded.

    The sampling is position-based and evenly distributed
    across the already ordered analytical frame, preserving
    both the first and last observations.

    This prevents high-cardinality analytical results from
    being materialized as hundreds of thousands of Python
    dictionaries solely for chart serialization.
    """

    if (
        max_points
        <=
        0
    ):
        raise ValueError(
            "max_points must be greater than zero."
        )


    if (
        len(
            dataframe
        )
        <=
        max_points
    ):
        return (
            dataframe
        )


    indexes = np.linspace(
        0,
        len(
            dataframe
        )
        -
        1,
        max_points,
        dtype=int,
    )


    return (
        dataframe.iloc[
            indexes
        ]
    )


def normalize_text(
    value: object,
) -> str:
    text = (
        unicodedata
        .normalize(
            "NFKD",
            str(
                value
            ),
        )
        .encode(
            "ascii",
            "ignore",
        )
        .decode(
            "ascii",
        )
        .lower()
        .strip()
    )


    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )


    return text.strip(
        "_"
    )


def text_tokens(
    value: object,
) -> set[
    str
]:
    return {
        token
        for token
        in normalize_text(
            value
        ).split(
            "_"
        )
        if token
    }


def is_geographic_name(
    column: str,
) -> bool:
    return bool(
        text_tokens(
            column
        )
        &
        GEOGRAPHIC_SIGNALS
    )


def to_native(
    value: Any,
) -> Any:
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
        return to_native(
            value.item()
        )


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


# ============================================================
# DATASET LOOKUP
# ============================================================

def build_dataset_map(
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    dict[
        str,
        Any,
    ],
]:
    result: dict[
        str,
        dict[
            str,
            Any,
        ]
    ] = {}


    for dataset in datasets:
        dataset_id = str(
            dataset[
                "dataset_id"
            ]
        )


        dataframe = dataset[
            "dataframe"
        ]


        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                (
                    "Each dataset record must "
                    "contain a pandas DataFrame."
                )
            )


        result[
            dataset_id
        ] = dataset


    return result


# ============================================================
# VARIABLE HELPERS
# ============================================================

def variable_columns(
    candidate: DiscoveredAnalysis,
) -> list[
    str
]:
    return list(
        dict.fromkeys(
            variable.column
            for variable
            in candidate.variables
        )
    )


def quantitative_variables(
    candidate: DiscoveredAnalysis,
) -> list[
    DiscoveredVariable
]:
    return [
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


def categorical_variables(
    candidate: DiscoveredAnalysis,
) -> list[
    DiscoveredVariable
]:
    return [
        variable
        for variable
        in candidate.variables
        if (
            variable.analysis_kind
            ==
            "categorical"
            or
            variable.semantic_role
            in {
                "category",
                "entity",
                "country",
                "region",
                "granularity",
            }
        )
    ]


def temporal_variables(
    candidate: DiscoveredAnalysis,
) -> list[
    DiscoveredVariable
]:
    return [
        variable
        for variable
        in candidate.variables
        if (
            variable.analysis_kind
            ==
            "temporal"
            or
            variable.semantic_role
            ==
            "time"
        )
    ]


# ============================================================
# RESULT BUILDER
# ============================================================

def build_result(
    candidate: DiscoveredAnalysis,
    *,
    dataset_id: str,
    dataset_name: str,
    execution_status: str,
    valid_observations: int = 0,
    summary: list[
        str
    ] | None = None,
    metrics: dict[
        str,
        Any,
    ] | None = None,
    chart_type: str | None = None,
    chart_data: list[
        dict[
            str,
            Any,
        ]
    ] | None = None,
    warnings: list[
        str
    ] | None = None,
    limitations: list[
        str
    ] | None = None,
) -> SingleDatasetExecutedAnalysis:
    return SingleDatasetExecutedAnalysis(
        analysis_id=
            candidate.analysis_id,

        title=
            candidate.title,

        family=
            candidate.family,

        dataset_id=
            dataset_id,

        dataset=
            dataset_name,

        execution_status=
            execution_status,

        variables=
            variable_columns(
                candidate
            ),

        valid_observations=
            valid_observations,

        summary=
            summary
            or [],

        metrics=
            to_native(
                metrics
                or {}
            ),

        chart_type=
            chart_type,

        chart_data=
            to_native(
                chart_data
                or []
            ),

        warnings=
            warnings
            or [],

        limitations=[
            *candidate.limitations,
            *(
                limitations
                or []
            ),
        ],
    )


# ============================================================
# SAFE SPEARMAN
# ============================================================

def safe_spearman(
    x: pd.Series,
    y: pd.Series,
    *,
    minimum: int = MIN_VALID_ASSOCIATION,
) -> float | None:
    if (
        len(
            x
        )
        <
        minimum
    ):
        return None


    if (
        x.nunique()
        <
        2
        or
        y.nunique()
        <
        2
    ):
        return None


    result = spearmanr(
        x,
        y,
        nan_policy="omit",
    )


    coefficient = float(
        result.statistic
    )


    if not math.isfinite(
        coefficient
    ):
        return None


    return coefficient


# ============================================================
# DATA QUALITY
# ============================================================

def execute_data_quality(
    candidate: DiscoveredAnalysis,
    *,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_name: str,
) -> SingleDatasetExecutedAnalysis:
    row_count = int(
        len(
            dataframe
        )
    )

    column_count = int(
        len(
            dataframe.columns
        )
    )


    duplicate_rows = int(
        dataframe
        .duplicated()
        .sum()
    )


    total_cells = (
        row_count
        *
        column_count
    )


    missing_cells = int(
        dataframe
        .isna()
        .sum()
        .sum()
    )


    duplicate_ratio = (
        duplicate_rows
        /
        row_count
        if row_count
        else 0.0
    )


    missing_ratio = (
        missing_cells
        /
        total_cells
        if total_cells
        else 0.0
    )


    missing_by_column = {
        str(
            column
        ):
            int(
                count
            )
        for (
            column,
            count,
        )
        in (
            dataframe
            .isna()
            .sum()
            .items()
        )
        if count
        >
        0
    }


    completely_missing_columns = [
        str(
            column
        )
        for column
        in dataframe.columns
        if (
            dataframe[
                column
            ]
            .isna()
            .all()
        )
    ]


    constant_columns = [
        str(
            column
        )
        for column
        in dataframe.columns
        if (
            dataframe[
                column
            ]
            .nunique(
                dropna=True
            )
            <=
            1
        )
    ]


    return build_result(
        candidate,
        dataset_id=
            dataset_id,
        dataset_name=
            dataset_name,
        execution_status=
            "complete",
        valid_observations=
            row_count,
        summary=[
            (
                f"{row_count} ligne(s) et "
                f"{column_count} colonne(s) "
                "ont été contrôlées."
            ),

            (
                f"{missing_cells} valeur(s) "
                "manquante(s) ont été détectées."
                if missing_cells
                else
                "Aucune valeur manquante détectée."
            ),

            (
                f"{duplicate_rows} ligne(s) "
                "strictement dupliquée(s) ont été "
                "détectées."
                if duplicate_rows
                else
                "Aucun doublon strict détecté."
            ),
        ],
        metrics={
            "row_count":
                row_count,

            "column_count":
                column_count,

            "duplicate_rows":
                duplicate_rows,

            "duplicate_ratio":
                duplicate_ratio,

            "missing_cells":
                missing_cells,

            "missing_ratio":
                missing_ratio,

            "missing_by_column":
                missing_by_column,

            "completely_missing_columns":
                completely_missing_columns,

            "constant_columns":
                constant_columns,
        },
    )


# ============================================================
# DISTRIBUTION
# ============================================================

def execute_distribution(
    candidate: DiscoveredAnalysis,
    *,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_name: str,
) -> SingleDatasetExecutedAnalysis:
    variables = (
        quantitative_variables(
            candidate
        )
    )


    if not variables:
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "failed",
            warnings=[
                "No quantitative variable found."
            ],
        )


    column = (
        variables[
            0
        ].column
    )


    if column not in dataframe.columns:
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "failed",
            warnings=[
                f"Missing column: {column}"
            ],
        )


    values = (
        pd.to_numeric(
            dataframe[
                column
            ],
            errors="coerce",
        )
        .dropna()
    )


    n = int(
        len(
            values
        )
    )


    if (
        n
        <
        MIN_VALID_NUMERIC
    ):
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "skipped",
            valid_observations=
                n,
            warnings=[
                "Too few valid numeric observations."
            ],
        )


    q1 = float(
        values.quantile(
            0.25
        )
    )

    median = float(
        values.median()
    )

    q3 = float(
        values.quantile(
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


    outlier_mask = (
        (
            values
            <
            lower_bound
        )
        |
        (
            values
            >
            upper_bound
        )
    )


    outlier_count = int(
        outlier_mask.sum()
    )


    histogram_counts, histogram_edges = (
        np.histogram(
            values,
            bins="auto",
        )
    )


    chart_data = [
        {
            "bin_start":
                float(
                    histogram_edges[
                        index
                    ]
                ),

            "bin_end":
                float(
                    histogram_edges[
                        index
                        +
                        1
                    ]
                ),

            "count":
                int(
                    count
                ),
        }
        for index, count
        in enumerate(
            histogram_counts
        )
    ]


    return build_result(
        candidate,
        dataset_id=
            dataset_id,
        dataset_name=
            dataset_name,
        execution_status=
            "complete",
        valid_observations=
            n,
        summary=[
            (
                f"{n} observation(s) numérique(s) "
                "valides ont été analysées."
            ),

            (
                f"La médiane est {median:.4g} "
                f"et la moyenne "
                f"{float(values.mean()):.4g}."
            ),

            (
                f"{outlier_count} observation(s) "
                "sont situées hors des bornes "
                "IQR usuelles."
            ),
        ],
        metrics={
            "column":
                column,

            "n":
                n,

            "missing_count":
                int(
                    dataframe[
                        column
                    ]
                    .isna()
                    .sum()
                ),

            "mean":
                float(
                    values.mean()
                ),

            "median":
                median,

            "std":
                float(
                    values.std(
                        ddof=1
                    )
                )
                if n
                >
                1
                else None,

            "minimum":
                float(
                    values.min()
                ),

            "q1":
                q1,

            "q3":
                q3,

            "maximum":
                float(
                    values.max()
                ),

            "iqr":
                iqr,

            "skewness":
                float(
                    values.skew()
                )
                if n
                >=
                3
                else None,

            "unique_count":
                int(
                    values.nunique()
                ),

            "outlier_count":
                outlier_count,

            "outlier_ratio":
                (
                    outlier_count
                    /
                    n
                ),

            "lower_iqr_bound":
                lower_bound,

            "upper_iqr_bound":
                upper_bound,
        },
        chart_type=
            "histogram",
        chart_data=
            chart_data,
    )


# ============================================================
# TIME SORTING
# ============================================================

def sort_time_frame(
    dataframe: pd.DataFrame,
    *,
    time_column: str,
) -> pd.DataFrame:
    result = dataframe.copy()


    numeric_time = pd.to_numeric(
        result[
            time_column
        ],
        errors="coerce",
    )


    if (
        numeric_time
        .notna()
        .mean()
        >=
        0.90
    ):
        result[
            "__datalens_time_sort"
        ] = numeric_time

    else:
        parsed = pd.to_datetime(
            result[
                time_column
            ],
            errors="coerce",
        )


        if (
            parsed
            .notna()
            .mean()
            >=
            0.70
        ):
            result[
                "__datalens_time_sort"
            ] = parsed

        else:
            result[
                "__datalens_time_sort"
            ] = (
                result[
                    time_column
                ]
                .astype(
                    str
                )
            )


    return result.sort_values(
        "__datalens_time_sort"
    )


# ============================================================
# TIME SERIES
# ============================================================

def execute_time_series(
    candidate: DiscoveredAnalysis,
    *,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_name: str,
) -> SingleDatasetExecutedAnalysis:
    quantitative = (
        quantitative_variables(
            candidate
        )
    )


    if not quantitative:
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "failed",
            warnings=[
                "No quantitative measure found."
            ],
        )


    measure = (
        quantitative[
            0
        ].column
    )


    # ========================================================
    # EXPLICIT TIME BINDING
    #
    # The analytical candidate owns the variable binding.
    #
    # Observation-structure detection may describe how the
    # dataframe is organized, but it must never silently
    # replace the temporal axis declared by Discovery.
    #
    # Example:
    #
    #     analysis_id = dataset:...:time:birth:price
    #
    # must execute with:
    #
    #     time_column = birth
    #
    # and must never be rebound to another temporal column
    # such as `date` merely because structure detection
    # considers it the dataset's primary timeline.
    # ========================================================

    explicit_time_variables = [
        variable

        for variable
        in candidate.variables

        if str(
            getattr(
                variable,
                "semantic_role",
                "",
            )
            or
            ""
        )
        .strip()
        .lower()
        in {
            "time",
            "date",
        }
    ]


    # Conservative compatibility fallback:
    #
    # if the candidate does not expose an explicit semantic
    # role but contains exactly one temporal analytical
    # variable, that variable may still be used.
    if not explicit_time_variables:
        explicit_time_variables = [
            variable

            for variable
            in candidate.variables

            if str(
                getattr(
                    variable,
                    "analysis_kind",
                    "",
                )
                or
                ""
            )
            .strip()
            .lower()
            ==
            "temporal"
        ]


    if (
        len(
            explicit_time_variables
        )
        !=
        1
    ):
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "failed",
            warnings=[
                (
                    "The time-series candidate must expose "
                    "exactly one explicit temporal variable. "
                    "DataLens will not infer or substitute "
                    "another dataframe time column."
                )
            ],
        )


    time_column = str(
        explicit_time_variables[
            0
        ].column
    )


    structure = (
        detect_observation_structure(
            dataframe
        )
    )


    if (
        time_column is None
        or
        time_column
        not in dataframe.columns
        or
        measure
        not in dataframe.columns
    ):
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "failed",
            warnings=[
                (
                    "The temporal variable or "
                    "measure is unavailable."
                )
            ],
        )


    working = (
        dataframe.copy()
    )


    alignment_note = None


    if (
        structure.multi_grain_repeated
    ):
        (
            aligned,
            total_slice,
        ) = find_explicit_total_slice(
            working,

            structure=
                structure,

            required_columns=[
                measure,
            ],
        )


        if total_slice.found:
            working = (
                aligned
            )

            alignment_note = (
                total_slice.note
            )

            structure = (
                detect_observation_structure(
                    working
                )
            )


        else:
            return build_result(
                candidate,
                dataset_id=
                    dataset_id,
                dataset_name=
                    dataset_name,
                execution_status=
                    "needs_specialized_method",
                warnings=[
                    (
                        "Plusieurs niveaux de grain "
                        "existent pour une même "
                        "entité et une même période, "
                        "et aucun niveau Total "
                        "explicite ne permet de "
                        "sélectionner un grain "
                        "national unique."
                    )
                ],
                metrics={
                    "observation_structure":
                        structure.model_dump(),
                },
            )


    working = working[
        [
            time_column,
            measure,
        ]
    ].copy()


    working[
        measure
    ] = pd.to_numeric(
        working[
            measure
        ],
        errors="coerce",
    )


    working = (
        working
        .dropna(
            subset=[
                time_column,
                measure,
            ]
        )
    )


    period_count = int(
        working[
            time_column
        ]
        .nunique()
    )


    if (
        period_count
        <
        2
    ):
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "skipped",
            valid_observations=
                int(
                    len(
                        working
                    )
                ),
            warnings=[
                "Only one valid period is available."
            ],
        )


    rows_per_period = (
        working
        .groupby(
            time_column,
            dropna=False,
        )
        .size()
    )


    one_row_per_period = bool(
        (
            rows_per_period
            <=
            1
        )
        .all()
    )


    if one_row_per_period:
        ordered = (
            sort_time_frame(
                working,
                time_column=
                    time_column,
            )
        )


        first_value = float(
            ordered.iloc[
                0
            ][
                measure
            ]
        )

        last_value = float(
            ordered.iloc[
                -1
            ][
                measure
            ]
        )


        absolute_change = (
            last_value
            -
            first_value
        )


        relative_change = (
            absolute_change
            /
            abs(
                first_value
            )
            if first_value
            !=
            0
            else None
        )


        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "complete",
            valid_observations=
                int(
                    len(
                        working
                    )
                ),
            summary=[
                (
                    f"{period_count} période(s) "
                    "ont été analysées."
                ),
                *(
                    [
                        alignment_note
                    ]
                    if alignment_note
                    else []
                ),
            ],
            metrics={
                "time_column":
                    time_column,

                "measure_column":
                    measure,

                "period_count":
                    period_count,

                "first_value":
                    first_value,

                "last_value":
                    last_value,

                "absolute_change":
                    absolute_change,

                "relative_change":
                    relative_change,

                "observation_structure":
                    structure.model_dump(),

                "alignment_note":
                    alignment_note,
            },
            chart_type=
                "line",
            chart_data=[
                {
                    "period":
                        to_native(
                            row[
                                time_column
                            ]
                        ),

                    "value":
                        float(
                            row[
                                measure
                            ]
                        ),
                }
                for _, row
                in bounded_chart_frame(
                    ordered
                ).iterrows()
            ],
        )


    grouped = (
        working
        .groupby(
            time_column,
            dropna=False,
        )[
            measure
        ]
        .agg(
            [
                "count",
                "median",
            ]
        )
        .reset_index()
    )


    q1 = (
        working
        .groupby(
            time_column,
            dropna=False,
        )[
            measure
        ]
        .quantile(
            0.25
        )
        .rename(
            "q1"
        )
        .reset_index()
    )


    q3 = (
        working
        .groupby(
            time_column,
            dropna=False,
        )[
            measure
        ]
        .quantile(
            0.75
        )
        .rename(
            "q3"
        )
        .reset_index()
    )


    grouped = (
        grouped
        .merge(
            q1,
            on=
                time_column,
        )
        .merge(
            q3,
            on=
                time_column,
        )
    )


    grouped = (
        sort_time_frame(
            grouped,
            time_column=
                time_column,
        )
    )


    first_value = float(
        grouped.iloc[
            0
        ][
            "median"
        ]
    )

    last_value = float(
        grouped.iloc[
            -1
        ][
            "median"
        ]
    )


    return build_result(
        candidate,
        dataset_id=
            dataset_id,
        dataset_name=
            dataset_name,
        execution_status=
            "descriptive_only",
        valid_observations=
            int(
                len(
                    working
                )
            ),
        summary=[
            (
                f"{period_count} période(s) "
                "ont été analysées."
            ),

            (
                "La médiane transversale et "
                "l'intervalle interquartile "
                "sont calculés pour chaque "
                "période."
            ),

            *(
                [
                    alignment_note
                ]
                if alignment_note
                else []
            ),
        ],
        metrics={
            "time_column":
                time_column,

            "measure_column":
                measure,

            "period_count":
                period_count,

            "multiple_rows_per_period":
                True,

            "period_summary":
                "median_q1_q3",

            "first_period_median":
                first_value,

            "last_period_median":
                last_value,

            "median_change":
                (
                    last_value
                    -
                    first_value
                ),

            "observation_structure":
                structure.model_dump(),

            "alignment_note":
                alignment_note,
        },
        chart_type=
            "line_band",
        chart_data=[
            {
                "period":
                    to_native(
                        row[
                            time_column
                        ]
                    ),

                "median":
                    float(
                        row[
                            "median"
                        ]
                    ),

                "q1":
                    float(
                        row[
                            "q1"
                        ]
                    ),

                "q3":
                    float(
                        row[
                            "q3"
                        ]
                    ),

                "n":
                    int(
                        row[
                            "count"
                        ]
                    ),
            }
            for _, row
            in bounded_chart_frame(
                    grouped
                ).iterrows()
        ],
        warnings=[
            (
                "La médiane par période est un "
                "résumé descriptif transversal "
                "et ne représente pas un total "
                "métier."
            )
        ],
    )


# ============================================================
# GROUP COMPARISON
# ============================================================

def execute_group_comparison(
    candidate: DiscoveredAnalysis,
    *,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_name: str,
) -> SingleDatasetExecutedAnalysis:
    quantitative = (
        quantitative_variables(
            candidate
        )
    )

    categorical = (
        categorical_variables(
            candidate
        )
    )


    if (
        not quantitative
        or
        not categorical
    ):
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "failed",
            warnings=[
                (
                    "A group comparison requires "
                    "one quantitative and one "
                    "categorical variable."
                )
            ],
        )


    measure = (
        quantitative[
            0
        ].column
    )

    group = (
        categorical[
            0
        ].column
    )


    working = dataframe[
        [
            group,
            measure,
        ]
    ].copy()


    working[
        measure
    ] = pd.to_numeric(
        working[
            measure
        ],
        errors="coerce",
    )


    working = (
        working
        .dropna()
    )


    valid_group_count = int(
        working[
            group
        ]
        .nunique()
    )


    if (
        valid_group_count
        <
        2
    ):
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "skipped",
            valid_observations=
                int(
                    len(
                        working
                    )
                ),
            warnings=[
                (
                    "Fewer than two groups contain "
                    "valid observations."
                )
            ],
        )


    stats = (
        working
        .groupby(
            group,
            dropna=False,
        )[
            measure
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
            ]
        )
        .reset_index()
    )


    q1 = (
        working
        .groupby(
            group,
            dropna=False,
        )[
            measure
        ]
        .quantile(
            0.25
        )
        .rename(
            "q1"
        )
        .reset_index()
    )


    q3 = (
        working
        .groupby(
            group,
            dropna=False,
        )[
            measure
        ]
        .quantile(
            0.75
        )
        .rename(
            "q3"
        )
        .reset_index()
    )


    stats = (
        stats
        .merge(
            q1,
            on=
                group,
        )
        .merge(
            q3,
            on=
                group,
        )
        .sort_values(
            "median",
            ascending=False,
        )
    )


    structure = (
        detect_observation_structure(
            dataframe
        )
    )


    return build_result(
        candidate,
        dataset_id=
            dataset_id,
        dataset_name=
            dataset_name,
        execution_status=
            "descriptive_only",
        valid_observations=
            int(
                len(
                    working
                )
            ),
        summary=[
            (
                f"{valid_group_count} groupe(s) "
                "disposent d'observations "
                "exploitables."
            ),

            (
                "Les groupes sont comparés "
                "descriptivement."
            ),
        ],
        metrics={
            "group_column":
                group,

            "measure_column":
                measure,

            "valid_group_count":
                valid_group_count,

            "observation_structure":
                structure.model_dump(),
        },
        chart_type=
            "grouped_summary",
        chart_data=[
            {
                "group":
                    to_native(
                        row[
                            group
                        ]
                    ),

                "n":
                    int(
                        row[
                            "count"
                        ]
                    ),

                "mean":
                    float(
                        row[
                            "mean"
                        ]
                    ),

                "median":
                    float(
                        row[
                            "median"
                        ]
                    ),

                "q1":
                    float(
                        row[
                            "q1"
                        ]
                    ),

                "q3":
                    float(
                        row[
                            "q3"
                        ]
                    ),
            }
            for _, row
            in stats.iterrows()
        ],
        warnings=[
            (
                "Aucun test inférentiel n'est "
                "interprété automatiquement sans "
                "validation du plan d'observation."
            )
        ],
    )


# ============================================================
# ASSOCIATION SEMANTIC CAUTION
# ============================================================

def build_metric_dependency_warnings(
    left: DiscoveredVariable,
    right: DiscoveredVariable,
) -> list[
    str
]:
    warnings: list[
        str
    ] = []


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
        warnings.append(
            (
                "Les deux mesures décrivent un "
                "phénomène très proche mais avec "
                "des unités différentes. Une "
                "association forte peut en partie "
                "refléter leur construction ou "
                "une variable commune sous-jacente."
            )
        )


    elif (
        overlap_ratio
        >=
        0.50
    ):
        warnings.append(
            (
                "Les deux mesures partagent une "
                "forte proximité sémantique. "
                "L'association observée peut "
                "donc être en partie structurelle "
                "et ne doit pas être interprétée "
                "comme une relation indépendante."
            )
        )


    return warnings


# ============================================================
# PERIOD-SPECIFIC ASSOCIATIONS
# ============================================================

def calculate_period_associations(
    dataframe: pd.DataFrame,
    *,
    time_column: str,
    x_column: str,
    y_column: str,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    results: list[
        dict[
            str,
            Any,
        ]
    ] = []


    periods = (
        dataframe[
            time_column
        ]
        .dropna()
        .drop_duplicates()
        .tolist()
    )


    for period in periods:
        subset = (
            dataframe.loc[
                dataframe[
                    time_column
                ]
                ==
                period,
                [
                    x_column,
                    y_column,
                ],
            ]
            .copy()
        )


        subset[
            x_column
        ] = pd.to_numeric(
            subset[
                x_column
            ],
            errors="coerce",
        )


        subset[
            y_column
        ] = pd.to_numeric(
            subset[
                y_column
            ],
            errors="coerce",
        )


        subset = (
            subset
            .dropna()
        )


        coefficient = (
            safe_spearman(
                subset[
                    x_column
                ],
                subset[
                    y_column
                ],
                minimum=
                    MIN_VALID_PERIOD_ASSOCIATION,
            )
        )


        if coefficient is None:
            continue


        results.append(
            {
                "period":
                    to_native(
                        period
                    ),

                "n":
                    int(
                        len(
                            subset
                        )
                    ),

                "spearman":
                    coefficient,
            }
        )


    return results


# ============================================================
# QUANTITATIVE ASSOCIATION
# ============================================================

def execute_quantitative_association(
    candidate: DiscoveredAnalysis,
    *,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_name: str,
) -> SingleDatasetExecutedAnalysis:
    variables = (
        quantitative_variables(
            candidate
        )
    )


    if (
        len(
            variables
        )
        <
        2
    ):
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "failed",
            warnings=[
                "Two quantitative variables required."
            ],
        )


    left_variable = (
        variables[
            0
        ]
    )

    right_variable = (
        variables[
            1
        ]
    )


    x_column = (
        left_variable.column
    )

    y_column = (
        right_variable.column
    )


    structure_before = (
        detect_observation_structure(
            dataframe
        )
    )


    working = (
        dataframe.copy()
    )


    alignment_note = None


    if (
        structure_before.multi_grain_repeated
    ):
        (
            aligned,
            total_slice,
        ) = find_explicit_total_slice(
            working,

            structure=
                structure_before,

            required_columns=[
                x_column,
                y_column,
            ],
        )


        if total_slice.found:
            working = (
                aligned
            )

            alignment_note = (
                total_slice.note
            )

        else:
            return build_result(
                candidate,
                dataset_id=
                    dataset_id,
                dataset_name=
                    dataset_name,
                execution_status=
                    "needs_specialized_method",
                valid_observations=
                    0,
                summary=[
                    (
                        "Plusieurs observations "
                        "existent pour la même clé "
                        "analytique et aucun niveau "
                        "Total explicite ne permet "
                        "de restaurer un grain "
                        "unique."
                    )
                ],
                metrics={
                    "observation_structure_before":
                        structure_before.model_dump(),
                },
                warnings=[
                    (
                        "DataLens refuse de moyenner "
                        "ou agréger arbitrairement "
                        "les mesures."
                    )
                ],
            )


    structure_after = (
        detect_observation_structure(
            working
        )
    )


    analysis_frame = working[
        [
            x_column,
            y_column,
        ]
    ].copy()


    analysis_frame[
        x_column
    ] = pd.to_numeric(
        analysis_frame[
            x_column
        ],
        errors="coerce",
    )


    analysis_frame[
        y_column
    ] = pd.to_numeric(
        analysis_frame[
            y_column
        ],
        errors="coerce",
    )


    analysis_frame = (
        analysis_frame
        .dropna()
    )


    n = int(
        len(
            analysis_frame
        )
    )


    if (
        n
        <
        MIN_VALID_ASSOCIATION
    ):
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "skipped",
            valid_observations=
                n,
            warnings=[
                (
                    "Too few complete numeric "
                    "pairs are available."
                )
            ],
        )


    coefficient = (
        safe_spearman(
            analysis_frame[
                x_column
            ],
            analysis_frame[
                y_column
            ],
        )
    )


    if coefficient is None:
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "skipped",
            valid_observations=
                n,
            warnings=[
                (
                    "The association could not "
                    "be calculated safely."
                )
            ],
        )


    dependency_warnings = (
        build_metric_dependency_warnings(
            left_variable,
            right_variable,
        )
    )


    chart_source = (
        analysis_frame
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


    # ========================================================
    # LONGITUDINAL PANEL
    # ========================================================

    if (
        structure_after.longitudinal_panel
        and
        structure_after.time_column
        is not None
    ):
        period_associations = (
            calculate_period_associations(
                working,

                time_column=
                    structure_after.time_column,

                x_column=
                    x_column,

                y_column=
                    y_column,
            )
        )


        coefficients = [
            float(
                item[
                    "spearman"
                ]
            )
            for item
            in period_associations
        ]


        median_period_spearman = (
            float(
                np.median(
                    coefficients
                )
            )
            if coefficients
            else None
        )


        positive_periods = sum(
            1
            for value
            in coefficients
            if value
            >
            0
        )


        negative_periods = sum(
            1
            for value
            in coefficients
            if value
            <
            0
        )


        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "descriptive_only",
            valid_observations=
                n,
            summary=[
                (
                    f"{n} paire(s) complètes "
                    "ont été analysées après "
                    "contrôle du grain."
                ),

                (
                    "Une structure longitudinale "
                    "entité × temps reste présente. "
                    "L'association est donc "
                    "analysée séparément par "
                    "période."
                ),

                *(
                    [
                        alignment_note
                    ]
                    if alignment_note
                    else []
                ),
            ],
            metrics={
                "x_column":
                    x_column,

                "y_column":
                    y_column,

                "overall_preliminary_spearman":
                    coefficient,

                "period_count_analysed":
                    len(
                        coefficients
                    ),

                "median_period_spearman":
                    median_period_spearman,

                "positive_periods":
                    positive_periods,

                "negative_periods":
                    negative_periods,

                "period_associations":
                    period_associations,

                "observation_structure_before":
                    structure_before.model_dump(),

                "observation_structure_after":
                    structure_after.model_dump(),

                "alignment_note":
                    alignment_note,
            },
            chart_type=
                "scatter",
            chart_data=
                chart_data,
            warnings=[
                (
                    "La corrélation globale est "
                    "conservée uniquement comme "
                    "signal exploratoire."
                ),

                (
                    "Les associations par période "
                    "sont descriptives."
                ),

                *dependency_warnings,
            ],
        )


    # ========================================================
    # UNRESOLVED REPETITION
    # ========================================================

    if (
        structure_after
        .repeated_entity_time_rows
    ):
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "needs_specialized_method",
            valid_observations=
                n,
            summary=[
                (
                    "Des observations répétées "
                    "subsistent après l'alignement."
                )
            ],
            metrics={
                "preliminary_spearman":
                    coefficient,

                "observation_structure_before":
                    structure_before.model_dump(),

                "observation_structure_after":
                    structure_after.model_dump(),

                "alignment_note":
                    alignment_note,
            },
            warnings=[
                (
                    "L'association globale n'est "
                    "pas interprétée comme un "
                    "résultat indépendant."
                ),

                *dependency_warnings,
            ],
        )


    # ========================================================
    # CROSS-SECTIONAL UNIQUE
    # ========================================================

    return build_result(
        candidate,
        dataset_id=
            dataset_id,
        dataset_name=
            dataset_name,
        execution_status=
            "complete",
        valid_observations=
            n,
        summary=[
            (
                f"{n} paire(s) complètes ont été "
                "analysées sur un grain unique."
            ),

            (
                f"L'association de rang observée "
                f"est {coefficient:.3f}."
            ),

            *(
                [
                    alignment_note
                ]
                if alignment_note
                else []
            ),
        ],
        metrics={
            "x_column":
                x_column,

            "y_column":
                y_column,

            "spearman":
                coefficient,

            "observation_structure_before":
                structure_before.model_dump(),

            "observation_structure_after":
                structure_after.model_dump(),

            "alignment_note":
                alignment_note,
        },
        chart_type=
            "scatter",
        chart_data=
            chart_data,
        warnings=[
            (
                "L'association est exploratoire "
                "et ne constitue pas une preuve "
                "de causalité."
            ),

            *dependency_warnings,
        ],
    )


# ============================================================
# DERIVED GAP
# ============================================================

def execute_derived_gap(
    candidate: DiscoveredAnalysis,
    *,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_name: str,
) -> SingleDatasetExecutedAnalysis:
    variables = (
        quantitative_variables(
            candidate
        )
    )


    if (
        len(
            variables
        )
        <
        2
    ):
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "failed",
            warnings=[
                (
                    "A derived gap requires two "
                    "quantitative variables."
                )
            ],
        )


    left_column = (
        variables[
            0
        ].column
    )

    right_column = (
        variables[
            1
        ].column
    )


    working = dataframe[
        [
            left_column,
            right_column,
        ]
    ].copy()


    working[
        left_column
    ] = pd.to_numeric(
        working[
            left_column
        ],
        errors="coerce",
    )


    working[
        right_column
    ] = pd.to_numeric(
        working[
            right_column
        ],
        errors="coerce",
    )


    working = (
        working
        .dropna()
    )


    n = int(
        len(
            working
        )
    )


    if (
        n
        <
        MIN_VALID_NUMERIC
    ):
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "skipped",
            valid_observations=
                n,
        )


    gap = (
        working[
            left_column
        ]
        -
        working[
            right_column
        ]
    )


    return build_result(
        candidate,
        dataset_id=
            dataset_id,
        dataset_name=
            dataset_name,
        execution_status=
            "complete",
        valid_observations=
            n,
        summary=[
            (
                f"{n} paire(s) ont été utilisées."
            ),

            (
                f"L'écart médian est "
                f"{float(gap.median()):.4g}."
            ),
        ],
        metrics={
            "left_column":
                left_column,

            "right_column":
                right_column,

            "gap_definition":
                (
                    f"{left_column} - "
                    f"{right_column}"
                ),

            "mean_gap":
                float(
                    gap.mean()
                ),

            "median_gap":
                float(
                    gap.median()
                ),

            "median_absolute_gap":
                float(
                    gap.abs().median()
                ),

            "minimum_gap":
                float(
                    gap.min()
                ),

            "maximum_gap":
                float(
                    gap.max()
                ),

            "positive_count":
                int(
                    (
                        gap
                        >
                        0
                    )
                    .sum()
                ),

            "negative_count":
                int(
                    (
                        gap
                        <
                        0
                    )
                    .sum()
                ),

            "zero_count":
                int(
                    (
                        gap
                        ==
                        0
                    )
                    .sum()
                ),
        },
        chart_type=
            "distribution",
    )


# ============================================================
# GEOGRAPHIC COMPARISON
# ============================================================

def execute_geographic_comparison(
    candidate: DiscoveredAnalysis,
    *,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_name: str,
) -> SingleDatasetExecutedAnalysis:
    quantitative = (
        quantitative_variables(
            candidate
        )
    )


    if not quantitative:
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "failed",
        )


    measure = (
        quantitative[
            0
        ].column
    )


    geography = next(
        (
            variable.column
            for variable
            in candidate.variables
            if is_geographic_name(
                variable.column
            )
        ),
        None,
    )


    if geography is None:
        geography = next(
            (
                str(
                    column
                )
                for column
                in dataframe.columns
                if is_geographic_name(
                    str(
                        column
                    )
                )
            ),
            None,
        )


    if geography is None:
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "failed",
            warnings=[
                "No geographic entity found."
            ],
        )


    structure = (
        detect_observation_structure(
            dataframe
        )
    )


    working = (
        dataframe.copy()
    )


    alignment_note = None


    if (
        structure.multi_grain_repeated
    ):
        (
            aligned,
            total_slice,
        ) = find_explicit_total_slice(
            working,

            structure=
                structure,

            required_columns=[
                measure,
            ],
        )


        if total_slice.found:
            working = (
                aligned
            )

            alignment_note = (
                total_slice.note
            )


    time_column = (
        structure.time_column
    )


    working[
        measure
    ] = pd.to_numeric(
        working[
            measure
        ],
        errors="coerce",
    )


    working = (
        working
        .dropna(
            subset=[
                geography,
                measure,
            ]
        )
    )


    selected_period = None


    if (
        time_column is not None
        and
        time_column
        in working.columns
    ):
        values = (
            working[
                time_column
            ]
            .dropna()
        )


        numeric = pd.to_numeric(
            values,
            errors="coerce",
        )


        if (
            numeric
            .notna()
            .mean()
            >=
            0.90
        ):
            latest = (
                numeric.max()
            )


            selected_period = (
                values.loc[
                    numeric
                    ==
                    latest
                ]
                .iloc[
                    0
                ]
            )

        else:
            selected_period = (
                sorted(
                    values
                    .astype(
                        str
                    )
                    .unique()
                )[
                    -1
                ]
            )


        working = working.loc[
            working[
                time_column
            ]
            .astype(
                str
            )
            ==
            str(
                selected_period
            )
        ].copy()


    entity_value_counts = (
        working
        .groupby(
            geography
        )[
            measure
        ]
        .nunique(
            dropna=True
        )
    )


    if (
        entity_value_counts
        >
        1
    ).any():
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "needs_specialized_method",
            valid_observations=
                int(
                    len(
                        working
                    )
                ),
            warnings=[
                (
                    "Plusieurs valeurs distinctes "
                    "subsistent pour certaines "
                    "entités. DataLens refuse de "
                    "les moyenner automatiquement."
                )
            ],
        )


    working = (
        working
        .drop_duplicates(
            subset=[
                geography,
                measure,
            ]
        )
    )


    ranking = (
        working[
            [
                geography,
                measure,
            ]
        ]
        .sort_values(
            measure,
            ascending=False,
        )
        .head(
            MAX_RANKING_ROWS
        )
    )


    return build_result(
        candidate,
        dataset_id=
            dataset_id,
        dataset_name=
            dataset_name,
        execution_status=
            "complete",
        valid_observations=
            int(
                len(
                    working
                )
            ),
        summary=[
            (
                f"{working[geography].nunique()} "
                "entité(s) disposent d'une "
                "valeur exploitable."
            ),

            *(
                [
                    f"Période sélectionnée : "
                    f"{selected_period}."
                ]
                if selected_period
                is not None
                else []
            ),

            *(
                [
                    alignment_note
                ]
                if alignment_note
                else []
            ),
        ],
        metrics={
            "geography_column":
                geography,

            "measure_column":
                measure,

            "selected_period":
                to_native(
                    selected_period
                ),

            "entity_count":
                int(
                    working[
                        geography
                    ]
                    .nunique()
                ),

            "alignment_note":
                alignment_note,
        },
        chart_type=
            "ranking_bar",
        chart_data=[
            {
                "entity":
                    to_native(
                        row[
                            geography
                        ]
                    ),

                "value":
                    float(
                        row[
                            measure
                        ]
                    ),
            }
            for _, row
            in ranking.iterrows()
        ],
    )


# ============================================================
# CATEGORICAL ASSOCIATION
# ============================================================

def execute_categorical_association(
    candidate: DiscoveredAnalysis,
    *,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_name: str,
) -> SingleDatasetExecutedAnalysis:
    variables = (
        categorical_variables(
            candidate
        )
    )


    if (
        len(
            variables
        )
        <
        2
    ):
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "failed",
        )


    left = (
        variables[
            0
        ].column
    )

    right = (
        variables[
            1
        ].column
    )


    working = (
        dataframe[
            [
                left,
                right,
            ]
        ]
        .dropna()
    )


    n = int(
        len(
            working
        )
    )


    if (
        n
        <
        MIN_VALID_CATEGORICAL
    ):
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "skipped",
            valid_observations=
                n,
        )


    contingency = pd.crosstab(
        working[
            left
        ],
        working[
            right
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
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
            execution_status=
                "skipped",
            valid_observations=
                n,
        )


    # ========================================================
    # VISUAL EVIDENCE
    # ========================================================

    chart_data: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for left_value in (
        contingency.index
    ):
        for right_value in (
            contingency.columns
        ):
            chart_data.append(
                {
                    "x":
                        str(
                            left_value
                        ),

                    "y":
                        str(
                            right_value
                        ),

                    "count":
                        int(
                            contingency.loc[
                                left_value,
                                right_value,
                            ]
                        ),
                }
            )


    # ========================================================
    # DESCRIPTIVE ASSOCIATION
    # ========================================================

    chi2, _, _, expected = (
        chi2_contingency(
            contingency,
            correction=False,
        )
    )


    denominator = min(
        contingency.shape[
            0
        ]
        -
        1,
        contingency.shape[
            1
        ]
        -
        1,
    )


    cramers_v = (
        math.sqrt(
            float(
                chi2
            )
            /
            (
                n
                *
                denominator
            )
        )
        if denominator
        >
        0
        else None
    )


    structure = (
        detect_observation_structure(
            dataframe
        )
    )


    return build_result(
        candidate,
        dataset_id=
            dataset_id,
        dataset_name=
            dataset_name,
        execution_status=
            "descriptive_only",
        valid_observations=
            n,
        summary=[
            (
                f"{n} observation(s) complètes "
                "ont été utilisées."
            ),

            (
                f"Le V de Cramér descriptif "
                f"est {cramers_v:.3f}."
                if cramers_v
                is not None
                else
                "Le V de Cramér n'a pas pu "
                "être calculé."
            ),
        ],
        metrics={
            "left_column":
                left,

            "right_column":
                right,

            "row_levels":
                int(
                    contingency.shape[
                        0
                    ]
                ),

            "column_levels":
                int(
                    contingency.shape[
                        1
                    ]
                ),

            "chi_square_statistic":
                float(
                    chi2
                ),

            "cramers_v":
                cramers_v,

            "low_expected_cell_ratio":
                float(
                    (
                        expected
                        <
                        5
                    )
                    .mean()
                ),

            "observation_structure":
                structure.model_dump(),
        },

        chart_type=
            "heatmap",

        chart_data=
            chart_data,

        warnings=[
            (
                "Le V de Cramér est utilisé "
                "comme mesure descriptive."
            ),

            (
                "Aucune p-value n'est interprétée "
                "automatiquement."
            ),
        ],
    )


# ============================================================
# SINGLE DATASET CANDIDATE
# ============================================================

def execute_single_dataset_candidate(
    candidate: DiscoveredAnalysis,
    *,
    dataset_map: dict[
        str,
        dict[
            str,
            Any,
        ]
    ],
) -> SingleDatasetExecutedAnalysis:
    if (
        candidate.scope
        !=
        "single_dataset"
    ):
        dataset_id = (
            candidate.dataset_ids[
                0
            ]
            if candidate.dataset_ids
            else "unknown"
        )


        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                "unknown",
            execution_status=
                "skipped",
        )


    if (
        len(
            candidate.dataset_ids
        )
        !=
        1
    ):
        dataset_id = (
            candidate.dataset_ids[
                0
            ]
            if candidate.dataset_ids
            else "unknown"
        )


        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                "unknown",
            execution_status=
                "failed",
        )


    dataset_id = (
        candidate.dataset_ids[
            0
        ]
    )


    record = dataset_map.get(
        dataset_id
    )


    if record is None:
        return build_result(
            candidate,
            dataset_id=
                dataset_id,
            dataset_name=
                "unknown",
            execution_status=
                "failed",
        )


    dataframe = record[
        "dataframe"
    ]


    dataset_name = str(
        record[
            "filename"
        ]
    )


    family = (
        candidate.family
    )


    if family == "data_quality":
        return execute_data_quality(
            candidate,
            dataframe=
                dataframe,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
        )


    if family == "distribution":
        return execute_distribution(
            candidate,
            dataframe=
                dataframe,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
        )


    if family == "time_series":
        return execute_time_series(
            candidate,
            dataframe=
                dataframe,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
        )


    if family == "group_comparison":
        return execute_group_comparison(
            candidate,
            dataframe=
                dataframe,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
        )


    if (
        family
        ==
        "quantitative_association"
    ):
        return execute_quantitative_association(
            candidate,
            dataframe=
                dataframe,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
        )


    if family == "derived_gap":
        return execute_derived_gap(
            candidate,
            dataframe=
                dataframe,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
        )


    if family == "geographic_comparison":
        return execute_geographic_comparison(
            candidate,
            dataframe=
                dataframe,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
        )


    if family == "categorical_association":
        return execute_categorical_association(
            candidate,
            dataframe=
                dataframe,
            dataset_id=
                dataset_id,
            dataset_name=
                dataset_name,
        )


    return build_result(
        candidate,
        dataset_id=
            dataset_id,
        dataset_name=
            dataset_name,
        execution_status=
            "skipped",
        warnings=[
            (
                "Unsupported analysis family: "
                f"{family}"
            )
        ],
    )


# ============================================================
# COMPLETE SINGLE EXECUTION
# ============================================================

def execute_single_dataset_discovery(
    *,
    discovery: AnalysisDiscoveryReport,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> SingleDatasetExecutionReport:
    dataset_map = (
        build_dataset_map(
            datasets
        )
    )


    candidates = [
        candidate
        for candidate
        in discovery.candidates
        if (
            candidate.scope
            ==
            "single_dataset"
        )
    ]


    results: list[
        SingleDatasetExecutedAnalysis
    ] = []


    for candidate in candidates:
        try:
            result = (
                execute_single_dataset_candidate(
                    candidate,
                    dataset_map=
                        dataset_map,
                )
            )

        except Exception as error:
            dataset_id = (
                candidate.dataset_ids[
                    0
                ]
                if candidate.dataset_ids
                else "unknown"
            )


            record = dataset_map.get(
                dataset_id
            )


            dataset_name = (
                str(
                    record[
                        "filename"
                    ]
                )
                if record
                is not None
                else "unknown"
            )


            result = build_result(
                candidate,
                dataset_id=
                    dataset_id,
                dataset_name=
                    dataset_name,
                execution_status=
                    "failed",
                warnings=[
                    (
                        f"{type(error).__name__}: "
                        f"{error}"
                    )
                ],
            )


        results.append(
            result
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


    return SingleDatasetExecutionReport(
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
                "La structure observationnelle "
                "est distinguée entre coupe "
                "transversale, panel longitudinal "
                "et répétition multi-grain."
            ),

            (
                "Un niveau explicite Total peut "
                "être sélectionné lorsqu'il "
                "restaure un grain unique."
            ),

            (
                "Aucune moyenne ou somme n'est "
                "inventée pour supprimer une "
                "répétition de grain."
            ),

            (
                "Les panels longitudinaux sont "
                "analysés descriptivement par "
                "période."
            ),

            (
                "Les séries temporelles ne "
                "mélangent plus automatiquement "
                "plusieurs niveaux de granularité."
            ),
        ],

        executor_rule_version=(
            "single_dataset_executor_v0.2"
        ),
    )
