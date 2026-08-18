from __future__ import annotations

import re
import unicodedata

from typing import (
    Any,
    Literal,
)

import pandas as pd

from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# TYPES
# ============================================================

ObservationStructureType = Literal[
    "unstructured",
    "cross_sectional_unique",
    "cross_sectional_repeated",
    "cross_sectional_multigrain",
    "longitudinal_panel",
    "longitudinal_repeated",
    "longitudinal_multigrain",
]


# ============================================================
# CONSTANTS
# ============================================================

ENTITY_SIGNALS = {
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


TEMPORAL_SIGNALS = {
    "year",
    "annee",
    "date",
    "datetime",
    "timestamp",
    "month",
    "mois",
    "time",
}


TOTAL_VALUE_SIGNALS = {
    "total",
    "overall",
    "all",
    "national",
    "national_total",
    "country_total",
    "all_population",
    "all_areas",
    "ensemble",
}


MAX_GRAIN_LEVELS = 30


# ============================================================
# SCHEMAS
# ============================================================

class ObservationStructure(
    BaseModel
):
    structure_type: ObservationStructureType

    row_count: int

    entity_column: str | None = None

    time_column: str | None = None

    entity_count: int = 0

    period_count: int = 0

    entities_with_multiple_periods: int = 0

    longitudinal_panel: bool = False

    repeated_entity_rows: bool = False

    repeated_entity_time_rows: bool = False

    multi_grain_repeated: bool = False

    grain_columns: list[
        str
    ] = Field(
        default_factory=list,
    )

    base_key_columns: list[
        str
    ] = Field(
        default_factory=list,
    )

    notes: list[
        str
    ] = Field(
        default_factory=list,
    )


class ExplicitTotalSlice(
    BaseModel
):
    found: bool

    column: str | None = None

    value: Any | None = None

    rows_before: int

    rows_after: int

    key_columns: list[
        str
    ] = Field(
        default_factory=list,
    )

    note: str | None = None


# ============================================================
# TEXT NORMALIZATION
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


# ============================================================
# COLUMN DETECTION
# ============================================================

def is_entity_column_name(
    column: str,
) -> bool:
    return bool(
        text_tokens(
            column
        )
        &
        ENTITY_SIGNALS
    )


def is_temporal_column_name(
    column: str,
) -> bool:
    return bool(
        text_tokens(
            column
        )
        &
        TEMPORAL_SIGNALS
    )


def find_entity_column(
    dataframe: pd.DataFrame,
) -> str | None:
    for column in dataframe.columns:
        column_name = str(
            column
        )


        if is_entity_column_name(
            column_name
        ):
            return column_name


    return None


def find_temporal_column(
    dataframe: pd.DataFrame,
) -> str | None:
    for column in dataframe.columns:
        column_name = str(
            column
        )


        if is_temporal_column_name(
            column_name
        ):
            return column_name


    return None


# ============================================================
# GRAIN DETECTION
# ============================================================

def detect_grain_columns(
    dataframe: pd.DataFrame,
    *,
    base_keys: list[
        str
    ],
) -> list[
    str
]:
    """
    Detect low-cardinality categorical columns
    that vary inside duplicated base keys.

    Example:

        Country + Year
            ↓
        Granularity = Rural / Total / Urban

    No assumption is made from the column name
    alone.
    """

    if not base_keys:
        return []


    valid_keys = (
        dataframe[
            base_keys
        ]
        .dropna()
    )


    if valid_keys.empty:
        return []


    duplicated_mask = (
        dataframe
        .duplicated(
            subset=
                base_keys,
            keep=False,
        )
    )


    repeated = dataframe.loc[
        duplicated_mask
    ]


    if repeated.empty:
        return []


    candidates: list[
        str
    ] = []


    for column in dataframe.columns:
        column_name = str(
            column
        )


        if column_name in base_keys:
            continue


        series = dataframe[
            column
        ]


        if pd.api.types.is_numeric_dtype(
            series
        ):
            continue


        unique_count = int(
            series
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
            MAX_GRAIN_LEVELS
        ):
            continue


        variation_by_key = (
            repeated
            .groupby(
                base_keys,
                dropna=False,
            )[
                column
            ]
            .nunique(
                dropna=True
            )
        )


        if (
            variation_by_key
            >
            1
        ).any():
            candidates.append(
                column_name
            )


    return candidates


# ============================================================
# STRUCTURE DETECTION
# ============================================================

def detect_observation_structure(
    dataframe: pd.DataFrame,
) -> ObservationStructure:
    row_count = int(
        len(
            dataframe
        )
    )


    entity_column = (
        find_entity_column(
            dataframe
        )
    )


    time_column = (
        find_temporal_column(
            dataframe
        )
    )


    if entity_column is None:
        return ObservationStructure(
            structure_type=
                "unstructured",

            row_count=
                row_count,

            time_column=
                time_column,

            period_count=(
                int(
                    dataframe[
                        time_column
                    ]
                    .nunique(
                        dropna=True
                    )
                )
                if (
                    time_column
                    is not None
                )
                else 0
            ),

            notes=[
                (
                    "Aucun identifiant d'entité "
                    "générique n'a été détecté."
                )
            ],
        )


    entity_count = int(
        dataframe[
            entity_column
        ]
        .nunique(
            dropna=True
        )
    )


    repeated_entity_rows = bool(
        dataframe[
            entity_column
        ]
        .duplicated()
        .any()
    )


    period_count = 0

    entities_with_multiple_periods = 0

    longitudinal_panel = False


    if (
        time_column
        is not None
    ):
        period_count = int(
            dataframe[
                time_column
            ]
            .nunique(
                dropna=True
            )
        )


        entity_period_counts = (
            dataframe[
                [
                    entity_column,
                    time_column,
                ]
            ]
            .dropna()
            .groupby(
                entity_column
            )[
                time_column
            ]
            .nunique()
        )


        entities_with_multiple_periods = int(
            (
                entity_period_counts
                >
                1
            )
            .sum()
        )


        longitudinal_panel = bool(
            period_count
            >
            1
            and
            entities_with_multiple_periods
            >
            0
        )


    base_keys = [
        entity_column,
    ]


    if (
        time_column
        is not None
    ):
        base_keys.append(
            time_column
        )


    repeated_entity_time_rows = bool(
        dataframe
        .duplicated(
            subset=
                base_keys,
            keep=False,
        )
        .any()
    )


    grain_columns = (
        detect_grain_columns(
            dataframe,
            base_keys=
                base_keys,
        )
        if repeated_entity_time_rows
        else []
    )


    multi_grain_repeated = bool(
        repeated_entity_time_rows
        and
        grain_columns
    )


    notes: list[
        str
    ] = []


    if longitudinal_panel:
        notes.append(
            (
                "Les mêmes entités sont observées "
                "sur plusieurs périodes."
            )
        )


    if multi_grain_repeated:
        notes.append(
            (
                "Plusieurs observations existent "
                "pour une même clé entité/période "
                "et une ou plusieurs dimensions "
                "de grain ont été détectées."
            )
        )


    if longitudinal_panel:
        if multi_grain_repeated:
            structure_type = (
                "longitudinal_multigrain"
            )

        elif repeated_entity_time_rows:
            structure_type = (
                "longitudinal_repeated"
            )

        else:
            structure_type = (
                "longitudinal_panel"
            )


    else:
        if multi_grain_repeated:
            structure_type = (
                "cross_sectional_multigrain"
            )

        elif repeated_entity_time_rows:
            structure_type = (
                "cross_sectional_repeated"
            )

        else:
            structure_type = (
                "cross_sectional_unique"
            )


    return ObservationStructure(
        structure_type=
            structure_type,

        row_count=
            row_count,

        entity_column=
            entity_column,

        time_column=
            time_column,

        entity_count=
            entity_count,

        period_count=
            period_count,

        entities_with_multiple_periods=
            entities_with_multiple_periods,

        longitudinal_panel=
            longitudinal_panel,

        repeated_entity_rows=
            repeated_entity_rows,

        repeated_entity_time_rows=
            repeated_entity_time_rows,

        multi_grain_repeated=
            multi_grain_repeated,

        grain_columns=
            grain_columns,

        base_key_columns=
            base_keys,

        notes=
            notes,
    )


# ============================================================
# EXPLICIT TOTAL SLICE
# ============================================================

def find_explicit_total_slice(
    dataframe: pd.DataFrame,
    *,
    structure: ObservationStructure,
    required_columns: list[
        str
    ],
) -> tuple[
    pd.DataFrame,
    ExplicitTotalSlice,
]:
    """
    Select an explicit Total/Overall/National
    level only when doing so restores a unique
    base observational key.

    This is filtering, not mathematical
    aggregation.
    """

    rows_before = int(
        len(
            dataframe
        )
    )


    key_columns = list(
        structure.base_key_columns
    )


    if not key_columns:
        return (
            dataframe,

            ExplicitTotalSlice(
                found=False,

                rows_before=
                    rows_before,

                rows_after=
                    rows_before,

                key_columns=
                    key_columns,
            ),
        )


    candidate_columns = [
        *structure.grain_columns,
    ]


    for column in dataframe.columns:
        column_name = str(
            column
        )


        if (
            column_name
            in candidate_columns
            or
            column_name
            in key_columns
            or
            column_name
            in required_columns
        ):
            continue


        series = dataframe[
            column
        ]


        if pd.api.types.is_numeric_dtype(
            series
        ):
            continue


        unique_count = int(
            series
            .nunique(
                dropna=True
            )
        )


        if (
            2
            <=
            unique_count
            <=
            MAX_GRAIN_LEVELS
        ):
            candidate_columns.append(
                column_name
            )


    for column in candidate_columns:
        unique_values = (
            dataframe[
                column
            ]
            .dropna()
            .drop_duplicates()
            .tolist()
        )


        for value in unique_values:
            if (
                normalize_text(
                    value
                )
                not in
                TOTAL_VALUE_SIGNALS
            ):
                continue


            subset = (
                dataframe.loc[
                    dataframe[
                        column
                    ]
                    ==
                    value
                ]
                .copy()
            )


            valid = (
                subset
                .dropna(
                    subset=[
                        *key_columns,
                        *required_columns,
                    ]
                )
            )


            if valid.empty:
                continue


            duplicated_keys = bool(
                valid
                .duplicated(
                    subset=
                        key_columns,
                    keep=False,
                )
                .any()
            )


            if duplicated_keys:
                continue


            return (
                subset,

                ExplicitTotalSlice(
                    found=True,

                    column=
                        column,

                    value=
                        value,

                    rows_before=
                        rows_before,

                    rows_after=
                        int(
                            len(
                                subset
                            )
                        ),

                    key_columns=
                        key_columns,

                    note=(
                        f"Le niveau explicite "
                        f"'{value}' de {column} "
                        "a été sélectionné afin "
                        "de restaurer une "
                        "observation unique par "
                        "clé analytique."
                    ),
                ),
            )


    return (
        dataframe,

        ExplicitTotalSlice(
            found=False,

            rows_before=
                rows_before,

            rows_after=
                rows_before,

            key_columns=
                key_columns,

            note=(
                "Aucun niveau explicite Total, "
                "Overall ou équivalent ne permet "
                "de restaurer automatiquement "
                "un grain unique."
            ),
        ),
    )