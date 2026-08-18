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
    spearmanr,
)

from app.discovery.schemas import (
    AnalysisDiscoveryReport,
    DiscoveredAnalysis,
    DiscoveredVariable,
)

from app.execution.cross_schemas import (
    CrossDatasetExecutedAnalysis,
    CrossDatasetExecutionReport,
)


# ============================================================
# CONSTANTS
# ============================================================

MIN_VALID_PAIRS = 20

MIN_STRATIFIED_PAIRS = 15

MAX_CHART_POINTS = 2000


TOTAL_VALUE_SIGNALS = {
    "total",
    "overall",
    "all",
    "national",
    "national_total",
    "country_total",
    "all_population",
    "all_areas",
    "all_territories",
    "ensemble",
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


ENTITY_NAME_SIGNALS = {
    "country",
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
    "company",
    "organisation",
    "organization",
}


# ============================================================
# TEXT HELPERS
# ============================================================

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


def is_temporal_column(
    column: str,
) -> bool:
    return bool(
        text_tokens(
            column
        )
        &
        TEMPORAL_NAME_SIGNALS
    )


def is_entity_column(
    column: str,
) -> bool:
    return bool(
        text_tokens(
            column
        )
        &
        ENTITY_NAME_SIGNALS
    )


# ============================================================
# JSON-SAFE VALUES
# ============================================================

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
                    "contain a pandas DataFrame "
                    "under 'dataframe'."
                )
            )


        result[
            dataset_id
        ] = dataset


    return result


# ============================================================
# CANDIDATE VARIABLES
# ============================================================

def get_candidate_variable(
    candidate: DiscoveredAnalysis,
    *,
    dataset_id: str,
    roles: set[
        str
    ],
) -> DiscoveredVariable | None:
    for variable in (
        candidate.variables
    ):
        if (
            variable.dataset_id
            ==
            dataset_id
            and
            variable.role
            in roles
        ):
            return variable


    return None


# ============================================================
# JOIN KEY NORMALIZATION
# ============================================================

def normalize_join_value(
    value: object,
) -> object:
    if pd.isna(
        value
    ):
        return None


    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()


    if isinstance(
        value,
        np.generic,
    ):
        value = value.item()


    if isinstance(
        value,
        str,
    ):
        return normalize_text(
            value
        )


    if isinstance(
        value,
        float,
    ):
        if value.is_integer():
            return int(
                value
            )


    return value


def build_normalized_join_frame(
    dataframe: pd.DataFrame,
    key_columns: list[
        str
    ],
    *,
    prefix: str,
) -> tuple[
    pd.DataFrame,
    list[
        str
    ],
]:
    result = dataframe.copy()


    normalized_columns: list[
        str
    ] = []


    for (
        index,
        column,
    ) in enumerate(
        key_columns,
        start=1,
    ):
        normalized_column = (
            f"__datalens_{prefix}_key_"
            f"{index:02d}"
        )


        result[
            normalized_column
        ] = (
            result[
                column
            ]
            .map(
                normalize_join_value
            )
        )


        normalized_columns.append(
            normalized_column
        )


    return (
        result,
        normalized_columns,
    )


# ============================================================
# KEY AUDIT
# ============================================================

def audit_key(
    dataframe: pd.DataFrame,
    key_columns: list[
        str
    ],
) -> dict[
    str,
    Any,
]:
    valid = (
        dataframe[
            key_columns
        ]
        .dropna()
    )


    valid_row_count = int(
        len(
            valid
        )
    )


    unique_key_count = int(
        len(
            valid
            .drop_duplicates()
        )
    )


    duplicate_row_count = (
        valid_row_count
        -
        unique_key_count
    )


    return {
        "valid_key_rows":
            valid_row_count,

        "unique_key_count":
            unique_key_count,

        "duplicate_key_rows":
            duplicate_row_count,

        "is_unique":
            bool(
                duplicate_row_count
                ==
                0
            ),
    }


# ============================================================
# MEASURE-CONSTANT DEDUPLICATION
# ============================================================

def collapse_measure_constant_keys(
    dataframe: pd.DataFrame,
    *,
    key_columns: list[
        str
    ],
    measure_column: str,
) -> tuple[
    pd.DataFrame,
    bool,
]:
    """
    Collapse repeated key rows only when the
    analysed measure is identical inside every
    key.

    This does NOT average anything.
    """

    working = (
        dataframe[
            [
                *key_columns,
                measure_column,
            ]
        ]
        .dropna(
            subset=[
                *key_columns,
                measure_column,
            ]
        )
    )


    if working.empty:
        return (
            dataframe,
            False,
        )


    unique_measure_counts = (
        working
        .groupby(
            key_columns,
            dropna=False,
        )[
            measure_column
        ]
        .nunique(
            dropna=True
        )
    )


    if (
        unique_measure_counts
        >
        1
    ).any():
        return (
            dataframe,
            False,
        )


    collapsed = (
        dataframe
        .drop_duplicates(
            subset=[
                *key_columns,
                measure_column,
            ]
        )
        .copy()
    )


    changed = (
        len(
            collapsed
        )
        <
        len(
            dataframe
        )
    )


    return (
        collapsed,
        changed,
    )


# ============================================================
# EXPLICIT TOTAL SLICE
# ============================================================

def find_explicit_total_slice(
    dataframe: pd.DataFrame,
    *,
    key_columns: list[
        str
    ],
    measure_column: str,
) -> tuple[
    pd.DataFrame,
    str | None,
    object | None,
]:
    """
    Search for an explicit categorical level such
    as Total / Overall / National.

    The slice is accepted only if:

    1. it contains valid measure values;
    2. the resulting join key becomes unique.

    No arbitrary group is selected.
    """

    excluded = {
        *key_columns,
        measure_column,
    }


    for column in dataframe.columns:
        column_name = str(
            column
        )


        if column_name in excluded:
            continue


        series = dataframe[
            column
        ]


        if pd.api.types.is_numeric_dtype(
            series
        ):
            continue


        unique_values = (
            series
            .dropna()
            .drop_duplicates()
            .tolist()
        )


        if (
            len(
                unique_values
            )
            <
            2
            or
            len(
                unique_values
            )
            >
            30
        ):
            continue


        for value in unique_values:
            normalized_value = (
                normalize_text(
                    value
                )
            )


            if (
                normalized_value
                not in TOTAL_VALUE_SIGNALS
            ):
                continue


            subset = (
                dataframe.loc[
                    series
                    ==
                    value
                ]
                .copy()
            )


            subset = (
                subset
                .dropna(
                    subset=[
                        *key_columns,
                        measure_column,
                    ]
                )
            )


            if (
                len(
                    subset
                )
                <
                MIN_VALID_PAIRS
            ):
                continue


            audit = audit_key(
                subset,
                key_columns,
            )


            if audit[
                "is_unique"
            ]:
                return (
                    subset,
                    column_name,
                    value,
                )


    return (
        dataframe,
        None,
        None,
    )


# ============================================================
# SAFE SIDE ALIGNMENT
# ============================================================

def align_dataset_side(
    dataframe: pd.DataFrame,
    *,
    key_columns: list[
        str
    ],
    measure_column: str,
    dataset_name: str,
) -> tuple[
    pd.DataFrame,
    list[
        str
    ],
    bool,
]:
    """
    Try only semantically conservative alignment.

    Order:

    1. already unique;
    2. exact measure-constant collapse;
    3. explicit Total / Overall / National slice.

    No mean, median or sum is invented.
    """

    actions: list[
        str
    ] = []


    working = (
        dataframe.copy()
    )


    initial_audit = audit_key(
        working,
        key_columns,
    )


    if initial_audit[
        "is_unique"
    ]:
        return (
            working,
            actions,
            True,
        )


    (
        collapsed,
        changed,
    ) = collapse_measure_constant_keys(
        working,
        key_columns=
            key_columns,
        measure_column=
            measure_column,
    )


    if changed:
        collapsed_audit = (
            audit_key(
                collapsed,
                key_columns,
            )
        )


        if collapsed_audit[
            "is_unique"
        ]:
            actions.append(
                (
                    f"{dataset_name}: repeated "
                    "join-key rows were collapsed "
                    "because the analysed measure "
                    "was identical inside every "
                    "key. No aggregation was "
                    "performed."
                )
            )


            return (
                collapsed,
                actions,
                True,
            )


    (
        total_slice,
        total_column,
        total_value,
    ) = find_explicit_total_slice(
        working,
        key_columns=
            key_columns,
        measure_column=
            measure_column,
    )


    if (
        total_column
        is not None
    ):
        actions.append(
            (
                f"{dataset_name}: the explicit "
                f"'{total_value}' level from "
                f"{total_column} was selected "
                "because it restores one "
                "observation per join key."
            )
        )


        return (
            total_slice,
            actions,
            True,
        )


    return (
        working,
        actions,
        False,
    )


# ============================================================
# COVERAGE
# ============================================================

def normalized_key_set(
    dataframe: pd.DataFrame,
    key_columns: list[
        str
    ],
) -> set[
    tuple[
        object,
        ...
    ]
]:
    values: set[
        tuple[
            object,
            ...
        ]
    ] = set()


    for row in (
        dataframe[
            key_columns
        ]
        .dropna()
        .itertuples(
            index=False,
            name=None,
        )
    ):
        values.add(
            tuple(
                normalize_join_value(
                    value
                )
                for value
                in row
            )
        )


    return values


def calculate_key_coverage(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_keys: list[
        str
    ],
    right_keys: list[
        str
    ],
) -> tuple[
    int,
    float,
    float,
]:
    left_set = normalized_key_set(
        left,
        left_keys,
    )

    right_set = normalized_key_set(
        right,
        right_keys,
    )


    matched = (
        left_set
        &
        right_set
    )


    matched_count = len(
        matched
    )


    left_coverage = (
        matched_count
        /
        len(
            left_set
        )
        if left_set
        else
        0.0
    )


    right_coverage = (
        matched_count
        /
        len(
            right_set
        )
        if right_set
        else
        0.0
    )


    return (
        matched_count,
        left_coverage,
        right_coverage,
    )


# ============================================================
# SAFE MERGE
# ============================================================

def merge_aligned_datasets(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_keys: list[
        str
    ],
    right_keys: list[
        str
    ],
    left_measure: str,
    right_measure: str,
) -> pd.DataFrame:
    (
        left_normalized,
        left_normalized_keys,
    ) = build_normalized_join_frame(
        left,
        left_keys,
        prefix="left",
    )


    (
        right_normalized,
        right_normalized_keys,
    ) = build_normalized_join_frame(
        right,
        right_keys,
        prefix="right",
    )


    rename_map = {
        right_key:
            left_key
        for (
            left_key,
            right_key,
        ) in zip(
            left_normalized_keys,
            right_normalized_keys,
        )
    }


    right_normalized = (
        right_normalized
        .rename(
            columns=
                rename_map
        )
    )


    right_normalized_keys = [
        rename_map[
            key
        ]
        for key
        in right_normalized_keys
    ]


    left_columns = [
        *left_normalized_keys,
        left_measure,
    ]


    right_columns = [
        *right_normalized_keys,
        right_measure,
    ]


    left_analysis = (
        left_normalized[
            left_columns
        ]
        .dropna()
        .copy()
    )


    right_analysis = (
        right_normalized[
            right_columns
        ]
        .dropna()
        .copy()
    )


    left_analysis = (
        left_analysis
        .rename(
            columns={
                left_measure:
                    "__datalens_x"
            }
        )
    )


    right_analysis = (
        right_analysis
        .rename(
            columns={
                right_measure:
                    "__datalens_y"
            }
        )
    )


    merged = pd.merge(
        left_analysis,
        right_analysis,
        how="inner",
        on=
            left_normalized_keys,
        validate="one_to_one",
    )


    merged[
        "__datalens_x"
    ] = pd.to_numeric(
        merged[
            "__datalens_x"
        ],
        errors="coerce",
    )


    merged[
        "__datalens_y"
    ] = pd.to_numeric(
        merged[
            "__datalens_y"
        ],
        errors="coerce",
    )


    merged = (
        merged
        .dropna(
            subset=[
                "__datalens_x",
                "__datalens_y",
            ]
        )
    )


    return merged


# ============================================================
# PANEL DETECTION FROM JOIN KEYS
# ============================================================

def detect_panel_keys(
    *,
    left_keys: list[
        str
    ],
    right_keys: list[
        str
    ],
) -> dict[
    str,
    str,
] | None:
    entity_index: int | None = None

    temporal_index: int | None = None


    for index, column in enumerate(
        left_keys
    ):
        if (
            entity_index
            is None
            and
            is_entity_column(
                column
            )
        ):
            entity_index = index


        if (
            temporal_index
            is None
            and
            is_temporal_column(
                column
            )
        ):
            temporal_index = index


    if (
        entity_index
        is None
        or
        temporal_index
        is None
    ):
        return None


    if (
        entity_index
        >=
        len(
            right_keys
        )
        or
        temporal_index
        >=
        len(
            right_keys
        )
    ):
        return None


    return {
        "left_entity":
            left_keys[
                entity_index
            ],

        "right_entity":
            right_keys[
                entity_index
            ],

        "left_time":
            left_keys[
                temporal_index
            ],

        "right_time":
            right_keys[
                temporal_index
            ],

        "entity_key_position":
            str(
                entity_index
            ),

        "time_key_position":
            str(
                temporal_index
            ),
    }


# ============================================================
# PRELIMINARY ASSOCIATION
# ============================================================

def safe_spearman(
    x: pd.Series,
    y: pd.Series,
) -> float | None:
    if (
        len(
            x
        )
        <
        MIN_VALID_PAIRS
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
# PANEL / TEMPORAL STRATIFIED ASSOCIATION
# ============================================================

def build_time_stratified_association(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_keys: list[
        str
    ],
    right_keys: list[
        str
    ],
    left_measure: str,
    right_measure: str,
    panel_keys: dict[
        str,
        str,
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    time_position = int(
        panel_keys[
            "time_key_position"
        ]
    )


    left_time = (
        left_keys[
            time_position
        ]
    )

    right_time = (
        right_keys[
            time_position
        ]
    )


    left_periods = {
        normalize_join_value(
            value
        ):
            value
        for value
        in left[
            left_time
        ].dropna().unique()
    }


    right_periods = {
        normalize_join_value(
            value
        ):
            value
        for value
        in right[
            right_time
        ].dropna().unique()
    }


    common_periods = (
        set(
            left_periods
        )
        &
        set(
            right_periods
        )
    )


    results: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for normalized_period in sorted(
        common_periods,
        key=lambda value:
            str(
                value
            ),
    ):
        left_period_value = (
            left_periods[
                normalized_period
            ]
        )

        right_period_value = (
            right_periods[
                normalized_period
            ]
        )


        left_subset = (
            left.loc[
                left[
                    left_time
                ]
                ==
                left_period_value
            ]
            .copy()
        )


        right_subset = (
            right.loc[
                right[
                    right_time
                ]
                ==
                right_period_value
            ]
            .copy()
        )


        try:
            merged = merge_aligned_datasets(
                left_subset,
                right_subset,
                left_keys=
                    left_keys,
                right_keys=
                    right_keys,
                left_measure=
                    left_measure,
                right_measure=
                    right_measure,
            )

        except pd.errors.MergeError:
            continue


        if (
            len(
                merged
            )
            <
            MIN_STRATIFIED_PAIRS
        ):
            continue


        coefficient = safe_spearman(
            merged[
                "__datalens_x"
            ],
            merged[
                "__datalens_y"
            ],
        )


        if coefficient is None:
            continue


        results.append(
            {
                "period":
                    to_native(
                        left_period_value
                    ),

                "n":
                    int(
                        len(
                            merged
                        )
                    ),

                "spearman":
                    coefficient,
            }
        )


    return results


# ============================================================
# CHART DATA
# ============================================================

def build_scatter_data(
    merged: pd.DataFrame,
) -> list[
    dict[
        str,
        float,
    ]
]:
    source = (
        merged[
            [
                "__datalens_x",
                "__datalens_y",
            ]
        ]
        .copy()
    )


    if (
        len(
            source
        )
        >
        MAX_CHART_POINTS
    ):
        indexes = np.linspace(
            0,
            len(
                source
            )
            -
            1,
            MAX_CHART_POINTS,
            dtype=int,
        )


        source = source.iloc[
            indexes
        ]


    return [
        {
            "x":
                float(
                    row[
                        "__datalens_x"
                    ]
                ),

            "y":
                float(
                    row[
                        "__datalens_y"
                    ]
                ),
        }
        for _, row
        in source.iterrows()
    ]


# ============================================================
# RESULT BUILDER
# ============================================================

def build_cross_result(
    candidate: DiscoveredAnalysis,
    *,
    execution_status: str,
    join_safety: str,
    rows_before: dict[
        str,
        int,
    ] | None = None,
    rows_after_alignment: dict[
        str,
        int,
    ] | None = None,
    joined_rows: int = 0,
    matched_key_count: int = 0,
    left_key_coverage: float = 0.0,
    right_key_coverage: float = 0.0,
    alignment_actions: list[
        str
    ] | None = None,
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
) -> CrossDatasetExecutedAnalysis:
    return CrossDatasetExecutedAnalysis(
        analysis_id=
            candidate.analysis_id,

        title=
            candidate.title,

        family=
            candidate.family,

        dataset_ids=list(
            candidate.dataset_ids
        ),

        datasets=list(
            candidate.datasets
        ),

        execution_status=
            execution_status,

        relationship_status=
            candidate.relationship_status,

        relationship_score=
            candidate.relationship_score,

        join_safety=
            join_safety,

        join_keys=
            candidate.join_keys,

        rows_before=
            rows_before
            or {},

        rows_after_alignment=
            rows_after_alignment
            or {},

        joined_rows=
            joined_rows,

        matched_key_count=
            matched_key_count,

        left_key_coverage=
            left_key_coverage,

        right_key_coverage=
            right_key_coverage,

        alignment_actions=
            alignment_actions
            or [],

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
# SINGLE CROSS-DATASET CANDIDATE
# ============================================================

def execute_cross_dataset_candidate(
    candidate: DiscoveredAnalysis,
    *,
    dataset_map: dict[
        str,
        dict[
            str,
            Any,
        ]
    ],
) -> CrossDatasetExecutedAnalysis:
    if (
        candidate.scope
        !=
        "cross_dataset"
    ):
        return build_cross_result(
            candidate,
            execution_status=
                "skipped",

            join_safety=
                "not_cross_dataset",

            warnings=[
                (
                    "This executor only handles "
                    "cross-dataset candidates."
                )
            ],
        )


    if (
        candidate.family
        !=
        "quantitative_association"
    ):
        return build_cross_result(
            candidate,
            execution_status=
                "skipped",

            join_safety=
                "unsupported_family",

            warnings=[
                (
                    "The current cross-dataset "
                    "executor handles quantitative "
                    "associations only."
                )
            ],
        )


    if (
        len(
            candidate.dataset_ids
        )
        !=
        2
    ):
        return build_cross_result(
            candidate,
            execution_status=
                "failed",

            join_safety=
                "invalid_dataset_count",

            warnings=[
                (
                    "A cross-dataset association "
                    "must currently involve "
                    "exactly two datasets."
                )
            ],
        )


    left_id = (
        candidate.dataset_ids[
            0
        ]
    )

    right_id = (
        candidate.dataset_ids[
            1
        ]
    )


    left_record = (
        dataset_map.get(
            left_id
        )
    )

    right_record = (
        dataset_map.get(
            right_id
        )
    )


    if (
        left_record
        is None
        or
        right_record
        is None
    ):
        return build_cross_result(
            candidate,
            execution_status=
                "failed",

            join_safety=
                "dataset_missing",

            warnings=[
                (
                    "At least one dataset required "
                    "for this candidate is missing "
                    "from the execution bundle."
                )
            ],
        )


    left = (
        left_record[
            "dataframe"
        ].copy()
    )

    right = (
        right_record[
            "dataframe"
        ].copy()
    )


    left_name = str(
        left_record[
            "filename"
        ]
    )

    right_name = str(
        right_record[
            "filename"
        ]
    )


    left_keys = (
        candidate.join_keys.get(
            left_id
        )
    )

    right_keys = (
        candidate.join_keys.get(
            right_id
        )
    )


    if (
        not left_keys
        or
        not right_keys
        or
        len(
            left_keys
        )
        !=
        len(
            right_keys
        )
    ):
        return build_cross_result(
            candidate,
            execution_status=
                "failed",

            join_safety=
                "invalid_join_keys",

            warnings=[
                (
                    "The candidate does not contain "
                    "a compatible pair of join keys."
                )
            ],
        )


    left_variable = (
        get_candidate_variable(
            candidate,
            dataset_id=
                left_id,
            roles={
                "x",
                "left_measure",
            },
        )
    )


    right_variable = (
        get_candidate_variable(
            candidate,
            dataset_id=
                right_id,
            roles={
                "y",
                "right_measure",
            },
        )
    )


    if (
        left_variable is None
        or
        right_variable is None
    ):
        return build_cross_result(
            candidate,
            execution_status=
                "failed",

            join_safety=
                "variables_missing",

            warnings=[
                (
                    "The candidate does not contain "
                    "one quantitative measure from "
                    "each dataset."
                )
            ],
        )


    left_measure = (
        left_variable.column
    )

    right_measure = (
        right_variable.column
    )


    required_left_columns = [
        *left_keys,
        left_measure,
    ]

    required_right_columns = [
        *right_keys,
        right_measure,
    ]


    missing_left = [
        column
        for column
        in required_left_columns
        if (
            column
            not in left.columns
        )
    ]


    missing_right = [
        column
        for column
        in required_right_columns
        if (
            column
            not in right.columns
        )
    ]


    if (
        missing_left
        or
        missing_right
    ):
        return build_cross_result(
            candidate,
            execution_status=
                "failed",

            join_safety=
                "columns_missing",

            warnings=[
                (
                    "One or more required columns "
                    "are missing from the datasets."
                ),

                (
                    "Left missing: "
                    f"{missing_left}"
                ),

                (
                    "Right missing: "
                    f"{missing_right}"
                ),
            ],
        )


    rows_before = {
        left_id:
            int(
                len(
                    left
                )
            ),

        right_id:
            int(
                len(
                    right
                )
            ),
    }


    # ========================================================
    # RELATIONSHIP BLOCK
    # ========================================================

    if (
        candidate.relationship_status
        ==
        "requires_alignment"
        and
        not candidate
        .observed_signals
        .get(
            "automatic_execution_allowed",
            False,
        )
    ):
        return build_cross_result(
            candidate,
            execution_status=
                "requires_alignment",

            join_safety=
                "grain_alignment_required",

            rows_before=
                rows_before,

            summary=[
                (
                    "La relation entre les deux "
                    "datasets est intéressante, "
                    "mais une jointure analytique "
                    "ligne à ligne n'est pas "
                    "suffisamment sûre."
                )
            ],

            warnings=[
                (
                    "DataLens n'agrège pas "
                    "automatiquement des taux, "
                    "pourcentages ou populations "
                    "pour transformer une relation "
                    "N:N en relation artificiellement "
                    "joignable."
                )
            ],
        )


    # ========================================================
    # SAFE GRAIN ALIGNMENT
    # ========================================================

    (
        aligned_left,
        left_actions,
        left_unique,
    ) = align_dataset_side(
        left,
        key_columns=
            left_keys,
        measure_column=
            left_measure,
        dataset_name=
            left_name,
    )


    (
        aligned_right,
        right_actions,
        right_unique,
    ) = align_dataset_side(
        right,
        key_columns=
            right_keys,
        measure_column=
            right_measure,
        dataset_name=
            right_name,
    )


    alignment_actions = [
        *left_actions,
        *right_actions,
    ]


    rows_after_alignment = {
        left_id:
            int(
                len(
                    aligned_left
                )
            ),

        right_id:
            int(
                len(
                    aligned_right
                )
            ),
    }


    if (
        not left_unique
        or
        not right_unique
    ):
        return build_cross_result(
            candidate,
            execution_status=
                "requires_alignment",

            join_safety=
                "unsafe_post_alignment_grain",

            rows_before=
                rows_before,

            rows_after_alignment=
                rows_after_alignment,

            alignment_actions=
                alignment_actions,

            summary=[
                (
                    "DataLens n'a pas trouvé de "
                    "méthode conservatrice permettant "
                    "de ramener automatiquement les "
                    "deux datasets à un grain commun."
                )
            ],

            warnings=[
                (
                    "Aucune moyenne, somme ou autre "
                    "agrégation arbitraire n'a été "
                    "appliquée."
                )
            ],
        )


    # ========================================================
    # COVERAGE
    # ========================================================

    (
        matched_key_count,
        left_coverage,
        right_coverage,
    ) = calculate_key_coverage(
        aligned_left,
        aligned_right,
        left_keys=
            left_keys,
        right_keys=
            right_keys,
    )


    # ========================================================
    # MERGE
    # ========================================================

    try:
        merged = merge_aligned_datasets(
            aligned_left,
            aligned_right,
            left_keys=
                left_keys,
            right_keys=
                right_keys,
            left_measure=
                left_measure,
            right_measure=
                right_measure,
        )

    except pd.errors.MergeError as error:
        return build_cross_result(
            candidate,
            execution_status=
                "requires_alignment",

            join_safety=
                "merge_not_one_to_one",

            rows_before=
                rows_before,

            rows_after_alignment=
                rows_after_alignment,

            matched_key_count=
                matched_key_count,

            left_key_coverage=
                left_coverage,

            right_key_coverage=
                right_coverage,

            alignment_actions=
                alignment_actions,

            warnings=[
                (
                    "The post-alignment merge was "
                    "not one-to-one."
                ),

                str(
                    error
                ),
            ],
        )


    if (
        len(
            merged
        )
        <
        MIN_VALID_PAIRS
    ):
        return build_cross_result(
            candidate,
            execution_status=
                "skipped",

            join_safety=
                "insufficient_overlap",

            rows_before=
                rows_before,

            rows_after_alignment=
                rows_after_alignment,

            joined_rows=
                int(
                    len(
                        merged
                    )
                ),

            matched_key_count=
                matched_key_count,

            left_key_coverage=
                left_coverage,

            right_key_coverage=
                right_coverage,

            alignment_actions=
                alignment_actions,

            warnings=[
                (
                    "Too few matched observations "
                    "remain after safe alignment."
                )
            ],
        )


    # ========================================================
    # PANEL STRUCTURE
    # ========================================================

    panel_keys = detect_panel_keys(
        left_keys=
            left_keys,
        right_keys=
            right_keys,
    )


    overall_spearman = safe_spearman(
        merged[
            "__datalens_x"
        ],
        merged[
            "__datalens_y"
        ],
    )


    scatter_data = build_scatter_data(
        merged
    )


    # ========================================================
    # PANEL-AWARE DESCRIPTIVE RESULT
    # ========================================================

    if panel_keys:
        stratified = (
            build_time_stratified_association(
                aligned_left,
                aligned_right,
                left_keys=
                    left_keys,
                right_keys=
                    right_keys,
                left_measure=
                    left_measure,
                right_measure=
                    right_measure,
                panel_keys=
                    panel_keys,
            )
        )


        coefficients = [
            float(
                result[
                    "spearman"
                ]
            )
            for result
            in stratified
            if (
                result.get(
                    "spearman"
                )
                is not None
            )
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
            for coefficient
            in coefficients
            if coefficient
            >
            0
        )


        negative_periods = sum(
            1
            for coefficient
            in coefficients
            if coefficient
            <
            0
        )


        summary = [
            (
                f"{len(merged)} observation(s) "
                "ont été appariées après contrôle "
                "du grain."
            ),

            (
                "Une structure entité × temps a "
                "été détectée. DataLens n'utilise "
                "donc pas une corrélation globale "
                "comme test inférentiel."
            ),
        ]


        if coefficients:
            summary.append(
                (
                    f"{len(coefficients)} période(s) "
                    "disposent d'assez d'observations "
                    "pour une association de rang "
                    "descriptive séparée."
                )
            )


        return build_cross_result(
            candidate,
            execution_status=
                "descriptive_only",

            join_safety=(
                "direct_aligned"
                if candidate
                .relationship_status
                ==
                "validated"
                else
                "matched_subset_aligned"
            ),

            rows_before=
                rows_before,

            rows_after_alignment=
                rows_after_alignment,

            joined_rows=
                int(
                    len(
                        merged
                    )
                ),

            matched_key_count=
                matched_key_count,

            left_key_coverage=
                left_coverage,

            right_key_coverage=
                right_coverage,

            alignment_actions=
                alignment_actions,

            summary=
                summary,

            metrics={
                "x_column":
                    left_measure,

                "y_column":
                    right_measure,

                "overall_preliminary_spearman":
                    overall_spearman,

                "panel_structure":
                    panel_keys,

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
                    stratified,
            },

            chart_type=
                "scatter",

            chart_data=
                scatter_data,

            warnings=[
                (
                    "The global Spearman value is "
                    "stored only as an exploratory "
                    "signal and must not be treated "
                    "as an inferential result."
                ),

                (
                    "Period-specific coefficients "
                    "are descriptive and are not "
                    "used as multiple significance "
                    "tests."
                ),
            ],
        )


    # ========================================================
    # CROSS-SECTIONAL DESCRIPTIVE ASSOCIATION
    # ========================================================

    return build_cross_result(
        candidate,
        execution_status=
            "complete",

        join_safety=(
            "direct_one_to_one"
            if candidate
            .relationship_status
            ==
            "validated"
            else
            "matched_subset_one_to_one"
        ),

        rows_before=
            rows_before,

        rows_after_alignment=
            rows_after_alignment,

        joined_rows=
            int(
                len(
                    merged
                )
            ),

        matched_key_count=
            matched_key_count,

        left_key_coverage=
            left_coverage,

        right_key_coverage=
            right_coverage,

        alignment_actions=
            alignment_actions,

        summary=[
            (
                f"{len(merged)} observation(s) "
                "ont été appariées sur un grain "
                "commun et unique."
            ),

            (
                "Une association de rang "
                "exploratoire a été calculée."
            ),
        ],

        metrics={
            "x_column":
                left_measure,

            "y_column":
                right_measure,

            "spearman":
                overall_spearman,
        },

        chart_type=
            "scatter",

        chart_data=
            scatter_data,

        warnings=[
            (
                "This first cross-dataset executor "
                "reports a descriptive Spearman "
                "association. It does not infer "
                "causality."
            )
        ],
    )


# ============================================================
# COMPLETE CROSS-DATASET EXECUTION
# ============================================================

def execute_cross_dataset_discovery(
    *,
    discovery: AnalysisDiscoveryReport,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> CrossDatasetExecutionReport:
    dataset_map = (
        build_dataset_map(
            datasets
        )
    )


    cross_candidates = [
        candidate
        for candidate
        in discovery.candidates
        if (
            candidate.scope
            ==
            "cross_dataset"
        )
    ]


    results: list[
        CrossDatasetExecutedAnalysis
    ] = []


    for candidate in (
        cross_candidates
    ):
        try:
            result = (
                execute_cross_dataset_candidate(
                    candidate,
                    dataset_map=
                        dataset_map,
                )
            )

        except Exception as error:
            result = build_cross_result(
                candidate,
                execution_status=
                    "failed",

                join_safety=
                    "unexpected_error",

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


    return CrossDatasetExecutionReport(
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

        requires_alignment_count=
            count_status(
                "requires_alignment"
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
                "Only cross-dataset candidates "
                "validated by Analysis Discovery "
                "are evaluated."
            ),

            (
                "Many-to-many candidates remain "
                "blocked unless a conservative "
                "grain-alignment rule can be "
                "established."
            ),

            (
                "The executor never averages, "
                "sums or otherwise aggregates "
                "measures merely to make a join "
                "possible."
            ),

            (
                "An explicit Total, Overall or "
                "National level may be selected "
                "when it restores a unique common "
                "grain without mathematical "
                "aggregation."
            ),

            (
                "Entity-by-time datasets are "
                "treated as panel structures. "
                "Associations are inspected "
                "within periods instead of "
                "treating all repeated rows as "
                "independent observations."
            ),
        ],
    )