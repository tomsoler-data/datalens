from __future__ import annotations


from dataclasses import (
    dataclass,
    field,
)

from pathlib import Path

import re
import unicodedata

from typing import (
    Any,
)


import pandas as pd


from app.profiling.types import (
    infer_analytical_type,
)

from app.relationships import (
    discover_relationships,
)


# ============================================================
# VERSION
# ============================================================

ANALYTICAL_VIEW_RULE_VERSION = (
    "analytical_view_v0.6"
)


# ============================================================
# SAFETY LIMITS
# ============================================================

MIN_JOIN_COVERAGE = 0.95

MAX_DIMENSION_PAYLOAD_COLUMNS = 50

MAX_FUNCTIONAL_ATTRIBUTES = 12

MAX_CATEGORICAL_LEVELS = 50

MIN_TEMPORAL_PARSE_RATIO = 0.90


# ============================================================
# SEMANTIC SIGNALS
# ============================================================

STRONG_ADDITIVE_MONETARY_SIGNALS = {
    "revenue",
    "revenu",
    "sales",
    "sale",
    "vente",
    "ventes",
    "turnover",
    "amount",
    "montant",
    "spend",
    "spent",
    "ca",
}


UNIT_MONETARY_SIGNALS = {
    "price",
    "prix",
    "cost",
    "cout",
}


QUANTITY_SIGNALS = {
    "quantity",
    "quantite",
    "qty",
    "units",
    "unit",
    "volume",
    "nombre",
    "count",
}


# Strict signals used only for deterministic line-amount
# derivation. These are intentionally narrower than the
# generic quantity / monetary signals above.
STRICT_QUANTITY_COLUMN_NAMES = {
    "quantity",
    "quantite",
    "qty",
}


STRICT_UNIT_PRICE_COLUMN_NAMES = {
    "unit_price",
    "unitprice",
    "price_per_unit",
    "price_each",
    "prix_unitaire",
    "prix_unite",
}


DERIVED_LINE_AMOUNT_COLUMN = (
    "gross_amount"
)


SESSION_IDENTIFIER_SIGNALS = {
    "session",
    "order",
    "commande",
    "basket",
    "panier",
    "cart",
}


CUSTOMER_IDENTIFIER_SIGNALS = {
    "customer",
    "client",
    "user",
    "account",
    "buyer",
    "acheteur",
}


GENDER_SIGNALS = {
    "gender",
    "genre",
    "sex",
    "sexe",
}


CATEGORY_SIGNALS = {
    "category",
    "categorie",
    "categ",
}


# ============================================================
# RESULT TYPES
# ============================================================

@dataclass
class JoinAudit:
    fact_dataset_id: str

    fact_filename: str

    dimension_dataset_id: str

    dimension_filename: str

    fact_keys: list[str]

    dimension_keys: list[str]

    cardinality: str

    relationship_score: float

    rows_before: int

    rows_after: int

    matched_rows: int

    unmatched_rows: int

    match_rate: float

    row_preserved: bool

    dimension_key_unique: bool

    status: str

    added_columns: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )


@dataclass
class DerivedDatasetAudit:
    dataset_id: str

    filename: str

    derivation_type: str

    source_dataset_ids: list[str]

    row_count: int

    column_count: int

    grain_columns: list[str]

    measure_columns: list[str]

    provenance: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )


@dataclass
class AnalyticalViewBuildResult:
    original_datasets: list[
        dict[
            str,
            Any,
        ]
    ]

    derived_datasets: list[
        dict[
            str,
            Any,
        ]
    ]

    join_audits: list[
        JoinAudit
    ]

    derived_audits: list[
        DerivedDatasetAudit
    ]

    notes: list[str]

    rule_version: str = (
        ANALYTICAL_VIEW_RULE_VERSION
    )

    @property
    def all_datasets(
        self,
    ) -> list[
        dict[
            str,
            Any,
        ]
    ]:
        return [
            *self.original_datasets,
            *self.derived_datasets,
        ]


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
) -> set[str]:
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


def dataset_stem(
    filename: str,
) -> str:
    return normalize_text(
        Path(
            filename
        ).stem
    )


def has_signal(
    column: str,
    signals: set[str],
) -> bool:
    tokens = (
        text_tokens(
            column
        )
    )


    normalized = (
        normalize_text(
            column
        )
    )


    if (
        tokens
        &
        signals
    ):
        return True


    for signal in signals:
        if (
            len(
                signal
            )
            >=
            4
            and
            signal
            in normalized
        ):
            return True


    return False


# ============================================================
# DATASET HELPERS
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


        dataframe = dataset.get(
            "dataframe"
        )


        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                (
                    "Each analytical-view input "
                    "dataset must contain a pandas "
                    "DataFrame under 'dataframe'."
                )
            )


        result[
            dataset_id
        ] = dataset


    return result


def make_derived_dataset_record(
    *,
    dataset_id: str,
    filename: str,
    dataframe: pd.DataFrame,
    derivation_type: str,
    source_dataset_ids: list[str],
    provenance: dict[
        str,
        Any,
    ],
    discoverable: bool = True,
) -> dict[
    str,
    Any,
]:
    return {
        "dataset_id":
            dataset_id,

        "filename":
            filename,

        "extension":
            ".derived",

        "dataframe":
            dataframe,

        "is_derived":
            True,

        "discoverable":
            discoverable,

        "derivation_depth":
            1,

        "derivation_type":
            derivation_type,

        "source_dataset_ids":
            list(
                source_dataset_ids
            ),

        "provenance":
            provenance,

        "analytical_view_rule_version":
            ANALYTICAL_VIEW_RULE_VERSION,
    }


# ============================================================
# RELATIONSHIP ORIENTATION
# ============================================================

def orient_dimension_relationship(
    relationship: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
] | None:
    best = relationship.get(
        "best_candidate"
    )


    if not best:
        return None


    if not bool(
        best.get(
            "usable_for_analysis",
            False,
        )
    ):
        return None


    if (
        str(
            best.get(
                "relationship_mode",
                "",
            )
        )
        !=
        "direct"
    ):
        return None


    cardinality = str(
        best.get(
            "cardinality",
            "",
        )
    )


    left_info = relationship.get(
        "left_dataset",
        {},
    )

    right_info = relationship.get(
        "right_dataset",
        {},
    )


    left_id = str(
        left_info.get(
            "dataset_id",
            "",
        )
    )

    right_id = str(
        right_info.get(
            "dataset_id",
            "",
        )
    )


    left_keys = list(
        best.get(
            "left_columns",
            []
        )
    )

    right_keys = list(
        best.get(
            "right_columns",
            []
        )
    )


    if (
        not left_id
        or
        not right_id
        or
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
        return None


    if cardinality == "1:N":
        return {
            "dimension_dataset_id":
                left_id,

            "fact_dataset_id":
                right_id,

            "dimension_keys":
                left_keys,

            "fact_keys":
                right_keys,

            "cardinality":
                cardinality,

            "relationship_score":
                float(
                    best.get(
                        "score",
                        0.0,
                    )
                ),
        }


    if cardinality == "N:1":
        return {
            "dimension_dataset_id":
                right_id,

            "fact_dataset_id":
                left_id,

            "dimension_keys":
                right_keys,

            "fact_keys":
                left_keys,

            "cardinality":
                cardinality,

            "relationship_score":
                float(
                    best.get(
                        "score",
                        0.0,
                    )
                ),
        }


    return None


# ============================================================
# SAFE DIMENSION ENRICHMENT
# ============================================================

def dimension_keys_are_unique(
    dataframe: pd.DataFrame,
    key_columns: list[str],
) -> bool:
    working = (
        dataframe[
            key_columns
        ]
        .dropna()
    )


    if working.empty:
        return False


    return bool(
        not working
        .duplicated(
            subset=
                key_columns,
            keep=False,
        )
        .any()
    )


def prepare_dimension_payload(
    *,
    fact: pd.DataFrame,
    dimension: pd.DataFrame,
    dimension_filename: str,
    dimension_keys: list[str],
) -> tuple[
    pd.DataFrame,
    list[str],
    dict[
        str,
        str,
    ],
]:
    payload_columns = [
        str(
            column
        )

        for column
        in dimension.columns

        if (
            str(
                column
            )
            not in dimension_keys
        )
    ]


    payload_columns = payload_columns[
        :
        MAX_DIMENSION_PAYLOAD_COLUMNS
    ]


    prefix = dataset_stem(
        dimension_filename
    )


    rename_map: dict[
        str,
        str,
    ] = {}


    used_columns = {
        str(
            column
        )

        for column
        in fact.columns
    }


    for column in payload_columns:
        target_name = column


        if target_name in used_columns:
            target_name = (
                f"{prefix}_{column}"
            )


            suffix = 2


            while (
                target_name
                in used_columns
            ):
                target_name = (
                    f"{prefix}_{column}_{suffix}"
                )

                suffix += 1


        rename_map[
            column
        ] = target_name


        used_columns.add(
            target_name
        )


    selected = dimension[
        [
            *dimension_keys,
            *payload_columns,
        ]
    ].copy()


    selected = (
        selected
        .rename(
            columns=
                rename_map
        )
    )


    added_columns = [
        rename_map[
            column
        ]

        for column
        in payload_columns
    ]


    return (
        selected,
        added_columns,
        rename_map,
    )


def safe_dimension_enrichment(
    *,
    fact: pd.DataFrame,
    fact_dataset_id: str,
    fact_filename: str,
    dimension: pd.DataFrame,
    dimension_dataset_id: str,
    dimension_filename: str,
    fact_keys: list[str],
    dimension_keys: list[str],
    cardinality: str,
    relationship_score: float,
) -> tuple[
    pd.DataFrame,
    JoinAudit,
]:
    rows_before = int(
        len(
            fact
        )
    )


    missing_fact_keys = [
        column

        for column
        in fact_keys

        if column not in fact.columns
    ]


    missing_dimension_keys = [
        column

        for column
        in dimension_keys

        if column not in dimension.columns
    ]


    if (
        missing_fact_keys
        or
        missing_dimension_keys
    ):
        audit = JoinAudit(
            fact_dataset_id=
                fact_dataset_id,

            fact_filename=
                fact_filename,

            dimension_dataset_id=
                dimension_dataset_id,

            dimension_filename=
                dimension_filename,

            fact_keys=
                fact_keys,

            dimension_keys=
                dimension_keys,

            cardinality=
                cardinality,

            relationship_score=
                relationship_score,

            rows_before=
                rows_before,

            rows_after=
                rows_before,

            matched_rows=
                0,

            unmatched_rows=
                rows_before,

            match_rate=
                0.0,

            row_preserved=
                True,

            dimension_key_unique=
                False,

            status=
                "blocked",

            warnings=[
                (
                    "One or more join-key columns "
                    "are missing."
                ),

                (
                    "Missing fact keys: "
                    f"{missing_fact_keys}"
                ),

                (
                    "Missing dimension keys: "
                    f"{missing_dimension_keys}"
                ),
            ],
        )


        return (
            fact.copy(),
            audit,
        )


    unique_dimension = (
        dimension_keys_are_unique(
            dimension,
            dimension_keys,
        )
    )


    if not unique_dimension:
        audit = JoinAudit(
            fact_dataset_id=
                fact_dataset_id,

            fact_filename=
                fact_filename,

            dimension_dataset_id=
                dimension_dataset_id,

            dimension_filename=
                dimension_filename,

            fact_keys=
                fact_keys,

            dimension_keys=
                dimension_keys,

            cardinality=
                cardinality,

            relationship_score=
                relationship_score,

            rows_before=
                rows_before,

            rows_after=
                rows_before,

            matched_rows=
                0,

            unmatched_rows=
                rows_before,

            match_rate=
                0.0,

            row_preserved=
                True,

            dimension_key_unique=
                False,

            status=
                "blocked",

            warnings=[
                (
                    "The dimension join key is "
                    "not unique. DataLens refuses "
                    "to perform a many-to-one "
                    "enrichment because it could "
                    "multiply the fact-table grain."
                )
            ],
        )


        return (
            fact.copy(),
            audit,
        )


    (
        dimension_payload,
        added_columns,
        _,
    ) = prepare_dimension_payload(
        fact=
            fact,

        dimension=
            dimension,

        dimension_filename=
            dimension_filename,

        dimension_keys=
            dimension_keys,
    )


    indicator_column = (
        "__datalens_join_status"
    )


    counter = 2


    while (
        indicator_column
        in fact.columns
        or
        indicator_column
        in dimension_payload.columns
    ):
        indicator_column = (
            "__datalens_join_status_"
            f"{counter}"
        )

        counter += 1


    try:
        merged = pd.merge(
            fact,
            dimension_payload,

            how="left",

            left_on=
                fact_keys,

            right_on=
                dimension_keys,

            validate=
                "many_to_one",

            indicator=
                indicator_column,
        )

    except Exception as error:
        audit = JoinAudit(
            fact_dataset_id=
                fact_dataset_id,

            fact_filename=
                fact_filename,

            dimension_dataset_id=
                dimension_dataset_id,

            dimension_filename=
                dimension_filename,

            fact_keys=
                fact_keys,

            dimension_keys=
                dimension_keys,

            cardinality=
                cardinality,

            relationship_score=
                relationship_score,

            rows_before=
                rows_before,

            rows_after=
                rows_before,

            matched_rows=
                0,

            unmatched_rows=
                rows_before,

            match_rate=
                0.0,

            row_preserved=
                True,

            dimension_key_unique=
                unique_dimension,

            status=
                "blocked",

            warnings=[
                (
                    f"{type(error).__name__}: "
                    f"{error}"
                )
            ],
        )


        return (
            fact.copy(),
            audit,
        )


    rows_after = int(
        len(
            merged
        )
    )


    row_preserved = (
        rows_after
        ==
        rows_before
    )


    matched_rows = int(
        (
            merged[
                indicator_column
            ]
            ==
            "both"
        )
        .sum()
    )


    unmatched_rows = (
        rows_after
        -
        matched_rows
    )


    match_rate = (
        matched_rows
        /
        rows_before
        if rows_before
        else 0.0
    )


    warnings: list[str] = []


    if not row_preserved:
        warnings.append(
            (
                "The join changed the number "
                "of fact-table rows."
            )
        )


    if (
        match_rate
        <
        MIN_JOIN_COVERAGE
    ):
        warnings.append(
            (
                "Join coverage is below the "
                f"{MIN_JOIN_COVERAGE:.0%} "
                "automatic-enrichment threshold."
            )
        )


    accepted = bool(
        row_preserved
        and
        match_rate
        >=
        MIN_JOIN_COVERAGE
    )


    if not accepted:
        audit = JoinAudit(
            fact_dataset_id=
                fact_dataset_id,

            fact_filename=
                fact_filename,

            dimension_dataset_id=
                dimension_dataset_id,

            dimension_filename=
                dimension_filename,

            fact_keys=
                fact_keys,

            dimension_keys=
                dimension_keys,

            cardinality=
                cardinality,

            relationship_score=
                relationship_score,

            rows_before=
                rows_before,

            rows_after=
                rows_after,

            matched_rows=
                matched_rows,

            unmatched_rows=
                unmatched_rows,

            match_rate=
                round(
                    match_rate,
                    6,
                ),

            row_preserved=
                row_preserved,

            dimension_key_unique=
                unique_dimension,

            status=
                "blocked",

            added_columns=[],

            warnings=
                warnings,
        )


        return (
            fact.copy(),
            audit,
        )


    merged = (
        merged
        .drop(
            columns=[
                indicator_column
            ]
        )
    )


    for (
        fact_key,
        dimension_key,
    ) in zip(
        fact_keys,
        dimension_keys,
    ):
        if (
            fact_key
            !=
            dimension_key
            and
            dimension_key
            in merged.columns
            and
            dimension_key
            not in added_columns
        ):
            merged = (
                merged
                .drop(
                    columns=[
                        dimension_key
                    ]
                )
            )


    audit = JoinAudit(
        fact_dataset_id=
            fact_dataset_id,

        fact_filename=
            fact_filename,

        dimension_dataset_id=
            dimension_dataset_id,

        dimension_filename=
            dimension_filename,

        fact_keys=
            fact_keys,

        dimension_keys=
            dimension_keys,

        cardinality=
            cardinality,

        relationship_score=
            relationship_score,

        rows_before=
            rows_before,

        rows_after=
            int(
                len(
                    merged
                )
            ),

        matched_rows=
            matched_rows,

        unmatched_rows=
            unmatched_rows,

        match_rate=
            round(
                match_rate,
                6,
            ),

        row_preserved=
            True,

        dimension_key_unique=
            True,

        status=
            "complete",

        added_columns=
            added_columns,

        warnings=
            warnings,
    )


    return (
        merged,
        audit,
    )


# ============================================================
# ANALYTICAL TYPE HELPERS
# ============================================================

def analytical_type(
    dataframe: pd.DataFrame,
    column: str,
) -> dict[
    str,
    Any,
]:
    return infer_analytical_type(
        column,
        dataframe[
            column
        ],
    )


def temporal_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    result: list[str] = []


    for column in dataframe.columns:
        column_name = str(
            column
        )


        inferred = (
            analytical_type(
                dataframe,
                column_name,
            )
        )


        if (
            inferred.get(
                "type"
            )
            ==
            "temporal"
        ):
            result.append(
                column_name
            )


    return result


def event_temporal_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    result: list[str] = []


    for column in (
        temporal_columns(
            dataframe
        )
    ):
        inferred = (
            analytical_type(
                dataframe,
                column,
            )
        )


        if (
            inferred.get(
                "subtype"
            )
            ==
            "birth_year"
        ):
            continue


        result.append(
            column
        )


    return result


def identifier_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    result: list[str] = []


    for column in dataframe.columns:
        column_name = str(
            column
        )


        inferred = (
            analytical_type(
                dataframe,
                column_name,
            )
        )


        if (
            inferred.get(
                "type"
            )
            ==
            "identifier"
        ):
            result.append(
                column_name
            )


    return result


def categorical_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    result: list[str] = []


    for column in dataframe.columns:
        column_name = str(
            column
        )


        inferred = (
            analytical_type(
                dataframe,
                column_name,
            )
        )


        if (
            inferred.get(
                "type"
            )
            ==
            "categorical"
        ):
            result.append(
                column_name
            )


    return result


def quantitative_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    result: list[str] = []


    for column in dataframe.columns:
        column_name = str(
            column
        )


        inferred = (
            analytical_type(
                dataframe,
                column_name,
            )
        )


        if (
            inferred.get(
                "type"
            )
            ==
            "quantitative"
        ):
            result.append(
                column_name
            )


    return result


def select_semantic_identifier(
    dataframe: pd.DataFrame,
    *,
    signals: set[str],
) -> str | None:
    candidates = []


    for column in (
        identifier_columns(
            dataframe
        )
    ):
        if not has_signal(
            column,
            signals,
        ):
            continue


        tokens = (
            text_tokens(
                column
            )
        )


        exact_signal_count = len(
            tokens
            &
            signals
        )


        candidates.append(
            (
                exact_signal_count,
                -len(
                    normalize_text(
                        column
                    )
                ),
                column,
            )
        )


    if not candidates:
        return None


    candidates.sort(
        reverse=True
    )


    return candidates[
        0
    ][
        2
    ]


def select_semantic_low_cardinality_column(
    dataframe: pd.DataFrame,
    *,
    signals: set[str],
    excluded_columns: set[str] | None = None,
) -> str | None:
    excluded = (
        excluded_columns
        or set()
    )


    candidates: list[
        tuple[
            int,
            int,
            str,
        ]
    ] = []


    for column in dataframe.columns:
        column_name = str(
            column
        )


        if (
            column_name
            in excluded
        ):
            continue


        if not has_signal(
            column_name,
            signals,
        ):
            continue


        values = (
            dataframe[
                column_name
            ]
            .dropna()
        )


        if values.empty:
            continue


        level_count = int(
            values.nunique()
        )


        if (
            level_count
            <
            2
            or
            level_count
            >
            MAX_CATEGORICAL_LEVELS
        ):
            continue


        exact_signal_count = len(
            text_tokens(
                column_name
            )
            &
            signals
        )


        candidates.append(
            (
                exact_signal_count,
                -level_count,
                column_name,
            )
        )


    if not candidates:
        return None


    candidates.sort(
        reverse=True
    )


    return candidates[
        0
    ][
        2
    ]


def find_birth_year_column(
    dataframe: pd.DataFrame,
) -> str | None:
    for column in dataframe.columns:
        column_name = str(
            column
        )


        inferred = (
            analytical_type(
                dataframe,
                column_name,
            )
        )


        if (
            inferred.get(
                "type"
            )
            ==
            "temporal"
            and
            inferred.get(
                "subtype"
            )
            ==
            "birth_year"
        ):
            return column_name


    return None


# ============================================================
# METRIC SEMANTICS
# ============================================================

def strict_numeric_candidates(
    dataframe: pd.DataFrame,
    *,
    allowed_names: set[str],
) -> list[str]:
    """
    Return quantitative columns whose normalized name is an
    exact member of a deliberately narrow semantic allow-list.

    This helper is intentionally stricter than has_signal().
    It is used only when DataLens would create a new monetary
    measure, where false positives are more dangerous than
    abstention.
    """

    candidates: list[str] = []


    for column in quantitative_columns(
        dataframe
    ):
        if normalize_text(
            column
        ) in allowed_names:
            candidates.append(
                column
            )


    return candidates


def has_strong_additive_monetary_column(
    dataframe: pd.DataFrame,
) -> bool:
    """
    Return True when the fact grain already exposes a strong
    additive monetary measure such as revenue or amount.

    When such a measure exists, DataLens does not manufacture a
    second competing line amount from quantity * unit price.
    """

    for column in quantitative_columns(
        dataframe
    ):
        if has_signal(
            column,
            STRONG_ADDITIVE_MONETARY_SIGNALS,
        ):
            return True


    return False


def derive_safe_line_monetary_amount(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[
        str,
        Any,
    ] | None,
]:
    """
    Derive one analytical-only line amount from an unambiguous
    quantity * unit-price pair.

    Safety rules:
    - never overwrite an existing gross_amount column;
    - do not derive a competing amount when a strong additive
      monetary measure already exists;
    - require exactly one strict quantity candidate;
    - require exactly one strict unit-price candidate;
    - require at least one row where both inputs are numeric;
    - keep the derivation internal to the Analytical View
      Builder. The validated Preparation dataframe is not
      mutated.

    Generic signals such as count, nombre, volume, price or cost
    are intentionally insufficient for this derivation.
    """

    if (
        DERIVED_LINE_AMOUNT_COLUMN
        in dataframe.columns
    ):
        return (
            dataframe,
            None,
        )


    if has_strong_additive_monetary_column(
        dataframe
    ):
        return (
            dataframe,
            None,
        )


    quantity_candidates = (
        strict_numeric_candidates(
            dataframe,
            allowed_names=
                STRICT_QUANTITY_COLUMN_NAMES,
        )
    )


    unit_price_candidates = (
        strict_numeric_candidates(
            dataframe,
            allowed_names=
                STRICT_UNIT_PRICE_COLUMN_NAMES,
        )
    )


    if (
        len(
            quantity_candidates
        )
        !=
        1
        or
        len(
            unit_price_candidates
        )
        !=
        1
    ):
        return (
            dataframe,
            None,
        )


    quantity_column = (
        quantity_candidates[
            0
        ]
    )


    unit_price_column = (
        unit_price_candidates[
            0
        ]
    )


    if (
        quantity_column
        ==
        unit_price_column
    ):
        return (
            dataframe,
            None,
        )


    quantity = pd.to_numeric(
        dataframe[
            quantity_column
        ],
        errors="coerce",
    )


    unit_price = pd.to_numeric(
        dataframe[
            unit_price_column
        ],
        errors="coerce",
    )


    valid_inputs = (
        quantity.notna()
        &
        unit_price.notna()
    )


    if int(
        valid_inputs.sum()
    ) == 0:
        return (
            dataframe,
            None,
        )


    line_amount = (
        quantity
        *
        unit_price
    )


    line_amount = (
        line_amount
        .replace(
            [
                float("inf"),
                float("-inf"),
            ],
            float("nan"),
        )
    )


    if int(
        line_amount
        .notna()
        .sum()
    ) == 0:
        return (
            dataframe,
            None,
        )


    result = (
        dataframe.copy()
    )


    result[
        DERIVED_LINE_AMOUNT_COLUMN
    ] = line_amount


    audit = {
        "operation":
            "analytical_line_amount_derivation",

        "derived_column":
            DERIVED_LINE_AMOUNT_COLUMN,

        "source_quantity_column":
            quantity_column,

        "source_unit_price_column":
            unit_price_column,

        "formula":
            (
                f"{quantity_column} * "
                f"{unit_price_column}"
            ),

        "valid_count":
            int(
                line_amount
                .notna()
                .sum()
            ),

        "missing_count":
            int(
                line_amount
                .isna()
                .sum()
            ),

        "analytical_only":
            True,

        "safety_policy":
            (
                "Derived only from exactly one strict "
                "quantity column and exactly one strict "
                "unit-price column. Generic count/volume/"
                "price signals are insufficient."
            ),
    }


    return (
        result,
        audit,
    )


def has_explicit_quantity_column(
    dataframe: pd.DataFrame,
) -> bool:
    for column in (
        quantitative_columns(
            dataframe
        )
    ):
        if has_signal(
            column,
            QUANTITY_SIGNALS,
        ):
            return True


    return False


def select_additive_measures(
    dataframe: pd.DataFrame,
    *,
    fact_original_columns: set[str],
    propagated_columns: set[str],
) -> dict[
    str,
    str,
]:
    """
    Select conservatively additive monetary
    measures.

    A unit price propagated from a validated
    dimension may be summed only when no explicit
    quantity field exists at fact grain.

    If quantity exists, DataLens does not assume
    that SUM(unit_price) represents revenue.
    """

    result: dict[
        str,
        str,
    ] = {}


    quantitative = (
        quantitative_columns(
            dataframe
        )
    )


    has_quantity = (
        has_explicit_quantity_column(
            dataframe
        )
    )


    for column in quantitative:
        if (
            column
            in fact_original_columns
            and
            has_signal(
                column,
                STRONG_ADDITIVE_MONETARY_SIGNALS,
            )
        ):
            result[
                column
            ] = (
                "The measure is present on the "
                "fact table and carries a strong "
                "additive monetary signal."
            )

            continue


        if (
            column
            in propagated_columns
            and
            has_signal(
                column,
                STRONG_ADDITIVE_MONETARY_SIGNALS,
            )
        ):
            result[
                column
            ] = (
                "The measure was propagated from "
                "a validated dimension and carries "
                "a strong additive monetary signal."
            )

            continue


        if (
            column
            in propagated_columns
            and
            has_signal(
                column,
                UNIT_MONETARY_SIGNALS,
            )
            and
            not has_quantity
        ):
            result[
                column
            ] = (
                "The unit monetary measure was "
                "propagated from a validated "
                "dimension to fact grain. No "
                "explicit quantity measure was "
                "detected, so one fact row is "
                "conservatively treated as one "
                "monetary event."
            )


    return result


# ============================================================
# FUNCTIONAL DEPENDENCIES
# ============================================================

def column_is_functionally_dependent(
    dataframe: pd.DataFrame,
    *,
    key_column: str,
    attribute_column: str,
) -> bool:
    working = (
        dataframe[
            [
                key_column,
                attribute_column,
            ]
        ]
        .dropna(
            subset=[
                key_column
            ]
        )
    )


    if working.empty:
        return False


    counts = (
        working
        .groupby(
            key_column,
            dropna=False,
        )[
            attribute_column
        ]
        .nunique(
            dropna=True
        )
    )


    if counts.empty:
        return False


    return bool(
        (
            counts
            <=
            1
        )
        .all()
    )


def functional_attributes(
    dataframe: pd.DataFrame,
    *,
    key_column: str,
    excluded_columns: set[str],
) -> list[str]:
    candidates: list[str] = []


    for collection in (
        categorical_columns(
            dataframe
        ),
        temporal_columns(
            dataframe
        ),
        identifier_columns(
            dataframe
        ),
    ):
        for column in collection:
            if column not in candidates:
                candidates.append(
                    column
                )


    selected: list[str] = []


    for column in candidates:
        if (
            column
            ==
            key_column
            or
            column
            in excluded_columns
        ):
            continue


        if column_is_functionally_dependent(
            dataframe,
            key_column=
                key_column,
            attribute_column=
                column,
        ):
            selected.append(
                column
            )


        if (
            len(
                selected
            )
            >=
            MAX_FUNCTIONAL_ATTRIBUTES
        ):
            break


    return selected


# ============================================================
# MONTHLY MATERIALIZATION
# ============================================================

def build_monthly_measure_view(
    dataframe: pd.DataFrame,
    *,
    time_column: str,
    measure_column: str,
) -> pd.DataFrame | None:
    parsed = pd.to_datetime(
        dataframe[
            time_column
        ],
        errors="coerce",
    )


    valid_ratio = float(
        parsed
        .notna()
        .mean()
    )


    if (
        valid_ratio
        <
        MIN_TEMPORAL_PARSE_RATIO
    ):
        return None


    measure = pd.to_numeric(
        dataframe[
            measure_column
        ],
        errors="coerce",
    )


    working = pd.DataFrame(
        {
            "month":
                parsed
                .dt
                .to_period(
                    "M"
                )
                .dt
                .to_timestamp(),

            measure_column:
                measure,
        }
    ).dropna()


    if working.empty:
        return None


    result = (
        working
        .groupby(
            "month",
            dropna=True,
        )
        .agg(
            **{
                f"sum_{measure_column}":
                    (
                        measure_column,
                        "sum",
                    ),

                "event_count":
                    (
                        measure_column,
                        "size",
                    ),
            }
        )
        .reset_index()
    )


    if (
        len(
            result
        )
        <
        2
    ):
        return None


    return result


# ============================================================
# CATEGORICAL MATERIALIZATION
# ============================================================

def build_categorical_measure_view(
    dataframe: pd.DataFrame,
    *,
    group_column: str,
    measure_column: str,
) -> pd.DataFrame | None:
    if (
        group_column
        not in dataframe.columns
        or
        measure_column
        not in dataframe.columns
    ):
        return None


    valid_groups = (
        dataframe[
            group_column
        ]
        .dropna()
    )


    if valid_groups.empty:
        return None


    level_count = int(
        valid_groups
        .nunique()
    )


    if (
        level_count
        <
        2
        or
        level_count
        >
        MAX_CATEGORICAL_LEVELS
    ):
        return None


    measure = pd.to_numeric(
        dataframe[
            measure_column
        ],
        errors="coerce",
    )


    working = pd.DataFrame(
        {
            group_column:
                dataframe[
                    group_column
                ],

            measure_column:
                measure,
        }
    ).dropna(
        subset=[
            group_column,
            measure_column,
        ]
    )


    if working.empty:
        return None


    result = (
        working
        .groupby(
            group_column,
            dropna=False,
        )
        .agg(
            **{
                f"sum_{measure_column}":
                    (
                        measure_column,
                        "sum",
                    ),

                "event_count":
                    (
                        measure_column,
                        "size",
                    ),
            }
        )
        .reset_index()
    )


    if (
        len(
            result
        )
        <
        2
    ):
        return None


    return result


# ============================================================
# GENERIC ENTITY MATERIALIZATION
# ============================================================

def build_entity_measure_view(
    dataframe: pd.DataFrame,
    *,
    entity_column: str,
    measure_column: str,
) -> pd.DataFrame | None:
    if (
        entity_column
        not in dataframe.columns
        or
        measure_column
        not in dataframe.columns
    ):
        return None


    valid_entities = (
        dataframe[
            entity_column
        ]
        .dropna()
    )


    if valid_entities.empty:
        return None


    unique_entities = int(
        valid_entities
        .nunique()
    )


    if (
        unique_entities
        <
        2
    ):
        return None


    unique_ratio = (
        unique_entities
        /
        len(
            valid_entities
        )
    )


    if (
        unique_ratio
        >=
        0.95
    ):
        return None


    measure = pd.to_numeric(
        dataframe[
            measure_column
        ],
        errors="coerce",
    )


    working = (
        dataframe.copy()
    )


    working[
        "__datalens_measure"
    ] = measure


    working = (
        working
        .dropna(
            subset=[
                entity_column,
                "__datalens_measure",
            ]
        )
    )


    if working.empty:
        return None


    excluded = {
        entity_column,
        measure_column,
        "__datalens_measure",
    }


    attributes = (
        functional_attributes(
            working,
            key_column=
                entity_column,
            excluded_columns=
                excluded,
        )
    )


    aggregation: dict[
        str,
        tuple[
            str,
            str,
        ],
    ] = {
        f"sum_{measure_column}":
            (
                "__datalens_measure",
                "sum",
            ),

        "event_count":
            (
                "__datalens_measure",
                "size",
            ),
    }


    for attribute in attributes:
        aggregation[
            attribute
        ] = (
            attribute,
            "first",
        )


    result = (
        working
        .groupby(
            entity_column,
            dropna=True,
        )
        .agg(
            **aggregation
        )
        .reset_index()
    )


    if (
        result[
            entity_column
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            (
                "Entity-grain materialization "
                "failed to produce one row per "
                f"{entity_column}."
            )
        )


    return result


# ============================================================
# SESSION MATERIALIZATION
# ============================================================

def build_session_behavior_view(
    dataframe: pd.DataFrame,
    *,
    session_column: str,
    customer_column: str | None,
    time_column: str | None,
    measure_column: str,
) -> pd.DataFrame | None:
    if (
        session_column
        not in dataframe.columns
        or
        measure_column
        not in dataframe.columns
    ):
        return None


    measure = pd.to_numeric(
        dataframe[
            measure_column
        ],
        errors="coerce",
    )


    working = (
        dataframe.copy()
    )


    working[
        "__datalens_measure"
    ] = measure


    working = (
        working
        .dropna(
            subset=[
                session_column,
                "__datalens_measure",
            ]
        )
    )


    if working.empty:
        return None


    session_count = int(
        working[
            session_column
        ]
        .nunique()
    )


    if (
        session_count
        <
        2
    ):
        return None


    parsed_time: pd.Series | None = None


    if (
        time_column is not None
        and
        time_column
        in working.columns
    ):
        candidate_time = pd.to_datetime(
            working[
                time_column
            ],
            errors="coerce",
        )


        if (
            float(
                candidate_time
                .notna()
                .mean()
            )
            >=
            MIN_TEMPORAL_PARSE_RATIO
        ):
            parsed_time = (
                candidate_time
            )


            working[
                "__datalens_event_time"
            ] = (
                parsed_time
            )


    excluded = {
        session_column,
        measure_column,
        "__datalens_measure",
        "__datalens_event_time",
    }


    attributes = (
        functional_attributes(
            working,
            key_column=
                session_column,
            excluded_columns=
                excluded,
        )
    )


    if (
        customer_column is not None
        and
        customer_column
        in working.columns
        and
        customer_column
        not in attributes
        and
        column_is_functionally_dependent(
            working,
            key_column=
                session_column,
            attribute_column=
                customer_column,
        )
    ):
        attributes.append(
            customer_column
        )


    aggregation: dict[
        str,
        tuple[
            str,
            str,
        ],
    ] = {
        "basket_amount":
            (
                "__datalens_measure",
                "sum",
            ),

        "item_count":
            (
                "__datalens_measure",
                "size",
            ),
    }


    if parsed_time is not None:
        aggregation[
            "first_event"
        ] = (
            "__datalens_event_time",
            "min",
        )


        aggregation[
            "last_event"
        ] = (
            "__datalens_event_time",
            "max",
        )


    for attribute in attributes:
        aggregation[
            attribute
        ] = (
            attribute,
            "first",
        )


    result = (
        working
        .groupby(
            session_column,
            dropna=True,
        )
        .agg(
            **aggregation
        )
        .reset_index()
    )


    if (
        result[
            session_column
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            (
                "Session materialization failed "
                "to produce one row per "
                f"{session_column}."
            )
        )


    return result


# ============================================================
# CUSTOMER MATERIALIZATION
# ============================================================

def add_age_at_first_purchase(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[
        str,
        Any,
    ] | None,
]:
    if (
        "first_purchase"
        not in dataframe.columns
    ):
        return (
            dataframe,
            None,
        )


    birth_column = (
        find_birth_year_column(
            dataframe
        )
    )


    if birth_column is None:
        return (
            dataframe,
            None,
        )


    first_purchase = pd.to_datetime(
        dataframe[
            "first_purchase"
        ],
        errors="coerce",
    )


    birth_year = pd.to_numeric(
        dataframe[
            birth_column
        ],
        errors="coerce",
    )


    age = (
        first_purchase
        .dt
        .year
        -
        birth_year
    )


    plausible = (
        age
        .between(
            0,
            120,
            inclusive="both",
        )
    )


    age = (
        age
        .where(
            plausible
        )
    )


    if (
        age.notna().sum()
        ==
        0
    ):
        return (
            dataframe,
            None,
        )


    result = (
        dataframe.copy()
    )


    result[
        "age_at_first_purchase"
    ] = (
        age
        .astype(
            "Float64"
        )
    )


    audit = {
        "derived_column":
            "age_at_first_purchase",

        "birth_column":
            birth_column,

        "reference_column":
            "first_purchase",

        "formula":
            (
                "year(first_purchase) "
                "- birth_year"
            ),

        "valid_count":
            int(
                age
                .notna()
                .sum()
            ),

        "missing_count":
            int(
                age
                .isna()
                .sum()
            ),

        "limitation":
            (
                "When only birth year is "
                "available, age is approximate "
                "because the exact birthday is "
                "unknown."
            ),
    }


    return (
        result,
        audit,
    )


def build_customer_behavior_view(
    session_view: pd.DataFrame,
    *,
    session_column: str,
    customer_column: str,
) -> tuple[
    pd.DataFrame | None,
    dict[
        str,
        Any,
    ] | None,
]:
    if (
        session_column
        not in session_view.columns
        or
        customer_column
        not in session_view.columns
        or
        "basket_amount"
        not in session_view.columns
        or
        "item_count"
        not in session_view.columns
    ):
        return (
            None,
            None,
        )


    working = (
        session_view
        .dropna(
            subset=[
                customer_column,
            ]
        )
        .copy()
    )


    if working.empty:
        return (
            None,
            None,
        )


    customer_count = int(
        working[
            customer_column
        ]
        .nunique()
    )


    if (
        customer_count
        <
        2
    ):
        return (
            None,
            None,
        )


    excluded = {
        session_column,
        customer_column,
        "basket_amount",
        "item_count",
        "first_event",
        "last_event",
    }


    attributes = (
        functional_attributes(
            working,
            key_column=
                customer_column,
            excluded_columns=
                excluded,
        )
    )


    aggregation: dict[
        str,
        tuple[
            str,
            str,
        ],
    ] = {
        "total_spend":
            (
                "basket_amount",
                "sum",
            ),

        "purchase_sessions":
            (
                session_column,
                "nunique",
            ),

        "average_basket":
            (
                "basket_amount",
                "mean",
            ),

        "median_basket":
            (
                "basket_amount",
                "median",
            ),

        "total_items":
            (
                "item_count",
                "sum",
            ),

        "average_items_per_basket":
            (
                "item_count",
                "mean",
            ),
    }


    if (
        "first_event"
        in working.columns
    ):
        aggregation[
            "first_purchase"
        ] = (
            "first_event",
            "min",
        )


    if (
        "last_event"
        in working.columns
    ):
        aggregation[
            "last_purchase"
        ] = (
            "last_event",
            "max",
        )


    for attribute in attributes:
        aggregation[
            attribute
        ] = (
            attribute,
            "first",
        )


    result = (
        working
        .groupby(
            customer_column,
            dropna=True,
        )
        .agg(
            **aggregation
        )
        .reset_index()
    )


    if (
        result[
            customer_column
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            (
                "Customer materialization failed "
                "to produce one row per "
                f"{customer_column}."
            )
        )


    (
        result,
        age_audit,
    ) = add_age_at_first_purchase(
        result
    )


    return (
        result,
        age_audit,
    )


# ============================================================
# REQUESTED EVENT CONTEXT
# ============================================================

def build_requested_event_context_view(
    dataframe: pd.DataFrame,
    *,
    customer_column: str | None,
    time_column: str | None,
) -> tuple[
    pd.DataFrame | None,
    dict[
        str,
        Any,
    ] | None,
]:
    """
    Build a canonical event-grain context used only
    by explicit documentary requests.

    The view preserves fact rows. It does not enter
    exploratory Discovery.

    Canonical columns:
    - customer_id
    - event_time
    - gender
    - category
    - age_at_first_purchase

    Age is derived from the customer's first
    observed purchase year and a birth-year field.
    """

    if (
        customer_column is None
        or
        time_column is None
        or
        customer_column
        not in dataframe.columns
        or
        time_column
        not in dataframe.columns
    ):
        return (
            None,
            None,
        )


    gender_column = (
        select_semantic_low_cardinality_column(
            dataframe,

            signals=
                GENDER_SIGNALS,
        )
    )


    if gender_column is None:
        return (
            None,
            None,
        )


    category_column = (
        select_semantic_low_cardinality_column(
            dataframe,

            signals=
                CATEGORY_SIGNALS,

            excluded_columns={
                gender_column,
            },
        )
    )


    if category_column is None:
        return (
            None,
            None,
        )


    birth_column = (
        find_birth_year_column(
            dataframe
        )
    )


    if birth_column is None:
        return (
            None,
            None,
        )


    event_time = pd.to_datetime(
        dataframe[
            time_column
        ],
        errors="coerce",
    )


    if (
        float(
            event_time
            .notna()
            .mean()
        )
        <
        MIN_TEMPORAL_PARSE_RATIO
    ):
        return (
            None,
            None,
        )


    birth_year = pd.to_numeric(
        dataframe[
            birth_column
        ],
        errors="coerce",
    )


    working = pd.DataFrame(
        {
            "customer_id":
                dataframe[
                    customer_column
                ],

            "event_time":
                event_time,

            "gender":
                dataframe[
                    gender_column
                ],

            "category":
                dataframe[
                    category_column
                ],

            "__birth_year":
                birth_year,
        }
    )


    valid_identity = (
        working[
            "customer_id"
        ]
        .notna()
        &
        working[
            "event_time"
        ]
        .notna()
    )


    if (
        int(
            valid_identity.sum()
        )
        <
        2
    ):
        return (
            None,
            None,
        )


    first_purchase = (
        working[
            "event_time"
        ]
        .where(
            valid_identity
        )
        .groupby(
            working[
                "customer_id"
            ]
        )
        .transform(
            "min"
        )
    )


    age = (
        first_purchase
        .dt
        .year
        -
        working[
            "__birth_year"
        ]
    )


    age = (
        age
        .where(
            age.between(
                0,
                120,
                inclusive="both",
            )
        )
        .astype(
            "Float64"
        )
    )


    result = (
        working[
            [
                "customer_id",
                "event_time",
                "gender",
                "category",
            ]
        ]
        .copy()
    )


    result[
        "age_at_first_purchase"
    ] = age


    if (
        result[
            "gender"
        ]
        .dropna()
        .nunique()
        <
        2
        or
        result[
            "category"
        ]
        .dropna()
        .nunique()
        <
        2
    ):
        return (
            None,
            None,
        )


    audit = {
        "operation":
            "requested_event_context",

        "grain":
            "event",

        "source_customer_column":
            customer_column,

        "source_time_column":
            time_column,

        "source_gender_column":
            gender_column,

        "source_category_column":
            category_column,

        "source_birth_column":
            birth_column,

        "canonical_columns": {
            "customer_id":
                customer_column,

            "event_time":
                time_column,

            "gender":
                gender_column,

            "category":
                category_column,

            "age_at_first_purchase":
                (
                    "year(first observed purchase) "
                    "- birth year"
                ),
        },

        "age_valid_count":
            int(
                result[
                    "age_at_first_purchase"
                ]
                .notna()
                .sum()
            ),

        "limitation":
            (
                "The view preserves event-grain "
                "rows. Repeated purchases by the "
                "same customer therefore remain "
                "visible so downstream statistical "
                "guards can detect dependence."
            ),
    }


    return (
        result,
        audit,
    )


# ============================================================
# RECORD CREATION HELPERS
# ============================================================

def append_derived_record(
    *,
    records: list[
        dict[
            str,
            Any,
        ]
    ],
    audits: list[
        DerivedDatasetAudit
    ],
    dataset_id: str,
    filename: str,
    dataframe: pd.DataFrame,
    derivation_type: str,
    source_dataset_ids: list[str],
    grain_columns: list[str],
    measure_columns: list[str],
    provenance: dict[
        str,
        Any,
    ],
    discoverable: bool = True,
) -> None:
    record = (
        make_derived_dataset_record(
            dataset_id=
                dataset_id,

            filename=
                filename,

            dataframe=
                dataframe,

            derivation_type=
                derivation_type,

            source_dataset_ids=
                source_dataset_ids,

            provenance=
                provenance,

            discoverable=
                discoverable,
        )
    )


    records.append(
        record
    )


    audits.append(
        DerivedDatasetAudit(
            dataset_id=
                dataset_id,

            filename=
                filename,

            derivation_type=
                derivation_type,

            source_dataset_ids=
                source_dataset_ids,

            row_count=
                int(
                    len(
                        dataframe
                    )
                ),

            column_count=
                int(
                    len(
                        dataframe.columns
                    )
                ),

            grain_columns=
                grain_columns,

            measure_columns=
                measure_columns,

            provenance=
                provenance,
        )
    )


# ============================================================
# DISCOVERABLE VIEW MATERIALIZATION
# ============================================================

def materialize_views_for_fact(
    *,
    fact_dataset_id: str,
    fact_filename: str,
    enriched: pd.DataFrame,
    source_dataset_ids: list[str],
    fact_original_columns: set[str],
    propagated_columns: set[str],
    include_requested_context: bool = False,
) -> tuple[
    list[
        dict[
            str,
            Any,
        ]
    ],
    list[
        DerivedDatasetAudit
    ],
]:
    """
    Materialize controlled downstream datasets.

    Discoverable aggregate views remain separate
    from requested-only context views.

    The joined fact table remains internal.

    v0.4 preserves the behavioural hierarchy:

        event rows
            ↓
        session / order grain
            ↓
        customer grain

    Customer behavioural metrics are therefore
    calculated from session-level metrics rather
    than directly from transaction rows.
    """

    records: list[
        dict[
            str,
            Any,
        ]
    ] = []


    audits: list[
        DerivedDatasetAudit
    ] = []


    analytical_frame = (
        enriched.copy()
    )


    (
        analytical_frame,
        line_amount_audit,
    ) = derive_safe_line_monetary_amount(
        analytical_frame
    )


    fact_slug = (
        normalize_text(
            fact_dataset_id
        )
    )


    times = (
        event_temporal_columns(
            analytical_frame
        )
    )


    categories = (
        categorical_columns(
            analytical_frame
        )
    )


    identifiers = (
        identifier_columns(
            analytical_frame
        )
    )


    session_column = (
        select_semantic_identifier(
            analytical_frame,
            signals=
                SESSION_IDENTIFIER_SIGNALS,
        )
    )


    customer_column = (
        select_semantic_identifier(
            analytical_frame,
            signals=
                CUSTOMER_IDENTIFIER_SIGNALS,
        )
    )


    primary_time_column = (
        times[
            0
        ]
        if times
        else None
    )


    # ========================================================
    # REQUESTED-ONLY EVENT CONTEXT
    # ========================================================

    if include_requested_context:
        (
            requested_context,
            requested_context_audit,
        ) = build_requested_event_context_view(
            analytical_frame,

            customer_column=
                customer_column,

            time_column=
                primary_time_column,
        )


        if (
            requested_context is not None
            and
            requested_context_audit is not None
        ):
            requested_dataset_id = (
                "derived:"
                f"{fact_slug}:"
                "requested:event_context"
            )


            requested_filename = (
                f"{Path(fact_filename).stem}"
                "__requested_event_context"
                ".derived"
            )


            append_derived_record(
                records=
                    records,

                audits=
                    audits,

                dataset_id=
                    requested_dataset_id,

                filename=
                    requested_filename,

                dataframe=
                    requested_context,

                derivation_type=
                    "requested_event_context",

                source_dataset_ids=
                    source_dataset_ids,

                grain_columns=[
                    "customer_id",
                    "event_time",
                ],

                measure_columns=[
                    "age_at_first_purchase",
                ],

                provenance=
                    requested_context_audit,

                discoverable=
                    False,
            )


    measure_semantics = (
        select_additive_measures(
            analytical_frame,

            fact_original_columns=
                fact_original_columns,

            propagated_columns=
                propagated_columns,
        )
    )


    if line_amount_audit is not None:
        derived_measure_column = str(
            line_amount_audit[
                "derived_column"
            ]
        )


        measure_semantics[
            derived_measure_column
        ] = (
            "The monetary measure was derived internally at "
            "fact-row grain from one unambiguous strict "
            "quantity × unit-price pair. It is analytical-only "
            "and does not mutate the validated Preparation "
            "output."
        )


    measures = list(
        measure_semantics.keys()
    )


    def attach_source_measure_derivation(
        provenance: dict[
            str,
            Any,
        ],
        *,
        measure_column: str,
    ) -> None:
        if line_amount_audit is None:
            return


        if (
            measure_column
            !=
            str(
                line_amount_audit.get(
                    "derived_column",
                    "",
                )
            )
        ):
            return


        provenance[
            "source_measure_derivation"
        ] = dict(
            line_amount_audit
        )


    if not measures:
        return (
            records,
            audits,
        )


    handled_identifiers: set[
        str
    ] = set()


    # ========================================================
    # MONTHLY ADDITIVE VIEWS
    # ========================================================

    for time_column in times:
        for measure_column in measures:
            monthly = (
                build_monthly_measure_view(
                    analytical_frame,

                    time_column=
                        time_column,

                    measure_column=
                        measure_column,
                )
            )


            if monthly is None:
                continue


            dataset_id = (
                "derived:"
                f"{fact_slug}:"
                "monthly:"
                f"{normalize_text(time_column)}:"
                f"{normalize_text(measure_column)}"
            )


            filename = (
                f"{Path(fact_filename).stem}"
                "__monthly_"
                f"{normalize_text(measure_column)}"
                ".derived"
            )


            provenance = {
                "fact_dataset_id":
                    fact_dataset_id,

                "operation":
                    "groupby_sum",

                "source_time_column":
                    time_column,

                "source_measure_column":
                    measure_column,

                "target_time_column":
                    "month",

                "target_measure_column":
                    f"sum_{measure_column}",

                "aggregation":
                    "sum",

                "grain":
                    "month",

                "metric_semantics":
                    measure_semantics[
                        measure_column
                    ],
            }


            attach_source_measure_derivation(
                provenance,
                measure_column=
                    measure_column,
            )


            append_derived_record(
                records=
                    records,

                audits=
                    audits,

                dataset_id=
                    dataset_id,

                filename=
                    filename,

                dataframe=
                    monthly,

                derivation_type=
                    "monthly_additive_measure",

                source_dataset_ids=
                    source_dataset_ids,

                grain_columns=[
                    "month"
                ],

                measure_columns=[
                    f"sum_{measure_column}",
                    "event_count",
                ],

                provenance=
                    provenance,
            )


    # ========================================================
    # CATEGORICAL ADDITIVE VIEWS
    # ========================================================

    for group_column in categories:
        for measure_column in measures:
            categorical_view = (
                build_categorical_measure_view(
                    analytical_frame,

                    group_column=
                        group_column,

                    measure_column=
                        measure_column,
                )
            )


            if categorical_view is None:
                continue


            dataset_id = (
                "derived:"
                f"{fact_slug}:"
                "category:"
                f"{normalize_text(group_column)}:"
                f"{normalize_text(measure_column)}"
            )


            filename = (
                f"{Path(fact_filename).stem}"
                "__by_"
                f"{normalize_text(group_column)}"
                "_"
                f"{normalize_text(measure_column)}"
                ".derived"
            )


            provenance = {
                "fact_dataset_id":
                    fact_dataset_id,

                "operation":
                    "groupby_sum",

                "group_column":
                    group_column,

                "source_measure_column":
                    measure_column,

                "target_measure_column":
                    f"sum_{measure_column}",

                "aggregation":
                    "sum",

                "grain":
                    group_column,

                "metric_semantics":
                    measure_semantics[
                        measure_column
                    ],
            }


            attach_source_measure_derivation(
                provenance,
                measure_column=
                    measure_column,
            )


            append_derived_record(
                records=
                    records,

                audits=
                    audits,

                dataset_id=
                    dataset_id,

                filename=
                    filename,

                dataframe=
                    categorical_view,

                derivation_type=
                    "categorical_additive_measure",

                source_dataset_ids=
                    source_dataset_ids,

                grain_columns=[
                    group_column
                ],

                measure_columns=[
                    f"sum_{measure_column}",
                    "event_count",
                ],

                provenance=
                    provenance,
            )


    # ========================================================
    # SESSION → CUSTOMER BEHAVIOUR
    # ========================================================

    if (
        session_column is not None
    ):
        for measure_column in measures:
            session_view = (
                build_session_behavior_view(
                    analytical_frame,

                    session_column=
                        session_column,

                    customer_column=
                        customer_column,

                    time_column=
                        primary_time_column,

                    measure_column=
                        measure_column,
                )
            )


            if session_view is None:
                continue


            handled_identifiers.add(
                session_column
            )


            session_dataset_id = (
                "derived:"
                f"{fact_slug}:"
                "session:"
                f"{normalize_text(session_column)}:"
                f"{normalize_text(measure_column)}"
            )


            session_filename = (
                f"{Path(fact_filename).stem}"
                "__sessions_"
                f"{normalize_text(measure_column)}"
                ".derived"
            )


            session_provenance = {
                "fact_dataset_id":
                    fact_dataset_id,

                "operation":
                    "session_materialization",

                "entity_column":
                    session_column,

                "parent_entity_column":
                    customer_column,

                "source_time_column":
                    primary_time_column,

                "source_measure_column":
                    measure_column,

                "target_measure_column":
                    "basket_amount",

                "aggregation":
                    "sum",

                "grain":
                    session_column,

                "metric_semantics":
                    measure_semantics[
                        measure_column
                    ],

                "item_count_semantics":
                    (
                        "Number of fact rows "
                        "contributing to the session."
                    ),
            }


            attach_source_measure_derivation(
                session_provenance,
                measure_column=
                    measure_column,
            )


            session_measures = [
                "basket_amount",
                "item_count",
            ]


            append_derived_record(
                records=
                    records,

                audits=
                    audits,

                dataset_id=
                    session_dataset_id,

                filename=
                    session_filename,

                dataframe=
                    session_view,

                derivation_type=
                    "entity_additive_measure",

                source_dataset_ids=
                    source_dataset_ids,

                grain_columns=[
                    session_column
                ],

                measure_columns=
                    session_measures,

                provenance=
                    session_provenance,
            )


            # ================================================
            # CUSTOMER BEHAVIOUR FROM SESSION GRAIN
            # ================================================

            if (
                customer_column is None
                or
                customer_column
                not in session_view.columns
            ):
                continue


            (
                customer_view,
                age_audit,
            ) = build_customer_behavior_view(
                session_view,

                session_column=
                    session_column,

                customer_column=
                    customer_column,
            )


            if customer_view is None:
                continue


            handled_identifiers.add(
                customer_column
            )


            customer_dataset_id = (
                "derived:"
                f"{fact_slug}:"
                "customer:"
                f"{normalize_text(customer_column)}:"
                f"{normalize_text(measure_column)}"
            )


            customer_filename = (
                f"{Path(fact_filename).stem}"
                "__customers_"
                f"{normalize_text(measure_column)}"
                ".derived"
            )


            customer_measure_columns = [
                "total_spend",
                "purchase_sessions",
                "average_basket",
                "median_basket",
                "total_items",
                "average_items_per_basket",
            ]


            if (
                "age_at_first_purchase"
                in customer_view.columns
            ):
                customer_measure_columns.append(
                    "age_at_first_purchase"
                )


            customer_provenance = {
                "fact_dataset_id":
                    fact_dataset_id,

                "operation":
                    "customer_behavior_materialization",

                "entity_column":
                    customer_column,

                "source_session_column":
                    session_column,

                "source_measure_column":
                    measure_column,

                "target_measure_column":
                    "total_spend",

                "grain":
                    customer_column,

                "aggregation_path": [
                    (
                        f"fact rows -> "
                        f"{session_column}"
                    ),

                    (
                        f"{session_column} -> "
                        f"{customer_column}"
                    ),
                ],

                "metric_definitions": {
                    "total_spend":
                        (
                            "Sum of basket_amount "
                            "across observed sessions."
                        ),

                    "purchase_sessions":
                        (
                            "Number of distinct "
                            "observed purchase sessions."
                        ),

                    "average_basket":
                        (
                            "Mean basket_amount "
                            "across the customer's "
                            "observed sessions."
                        ),

                    "median_basket":
                        (
                            "Median basket_amount "
                            "across the customer's "
                            "observed sessions."
                        ),

                    "total_items":
                        (
                            "Sum of fact-row counts "
                            "across the customer's "
                            "observed sessions."
                        ),

                    "average_items_per_basket":
                        (
                            "Mean fact-row count per "
                            "observed session."
                        ),
                },

                "age_derivation":
                    age_audit,

                "purchase_frequency_limitation":
                    (
                        "purchase_sessions is an "
                        "observed session count, "
                        "not an exposure-normalized "
                        "purchase rate."
                    ),
            }


            attach_source_measure_derivation(
                customer_provenance,
                measure_column=
                    measure_column,
            )


            append_derived_record(
                records=
                    records,

                audits=
                    audits,

                dataset_id=
                    customer_dataset_id,

                filename=
                    customer_filename,

                dataframe=
                    customer_view,

                derivation_type=
                    "entity_additive_measure",

                source_dataset_ids=
                    source_dataset_ids,

                grain_columns=[
                    customer_column
                ],

                measure_columns=
                    customer_measure_columns,

                provenance=
                    customer_provenance,
            )


    # ========================================================
    # GENERIC ENTITY VIEWS
    #
    # Used for identifiers not already represented by the
    # session/customer hierarchy.
    #
    # Example:
    # product_id.
    # ========================================================

    for entity_column in identifiers:
        if (
            entity_column
            in handled_identifiers
        ):
            continue


        for measure_column in measures:
            entity_view = (
                build_entity_measure_view(
                    analytical_frame,

                    entity_column=
                        entity_column,

                    measure_column=
                        measure_column,
                )
            )


            if entity_view is None:
                continue


            dataset_id = (
                "derived:"
                f"{fact_slug}:"
                "entity:"
                f"{normalize_text(entity_column)}:"
                f"{normalize_text(measure_column)}"
            )


            filename = (
                f"{Path(fact_filename).stem}"
                "__by_"
                f"{normalize_text(entity_column)}"
                "_"
                f"{normalize_text(measure_column)}"
                ".derived"
            )


            provenance = {
                "fact_dataset_id":
                    fact_dataset_id,

                "operation":
                    "groupby_sum",

                "entity_column":
                    entity_column,

                "source_measure_column":
                    measure_column,

                "target_measure_column":
                    f"sum_{measure_column}",

                "aggregation":
                    "sum",

                "grain":
                    entity_column,

                "metric_semantics":
                    measure_semantics[
                        measure_column
                    ],
            }


            attach_source_measure_derivation(
                provenance,
                measure_column=
                    measure_column,
            )


            append_derived_record(
                records=
                    records,

                audits=
                    audits,

                dataset_id=
                    dataset_id,

                filename=
                    filename,

                dataframe=
                    entity_view,

                derivation_type=
                    "entity_additive_measure",

                source_dataset_ids=
                    source_dataset_ids,

                grain_columns=[
                    entity_column
                ],

                measure_columns=[
                    f"sum_{measure_column}",
                    "event_count",
                ],

                provenance=
                    provenance,
            )


    return (
        records,
        audits,
    )


# ============================================================
# PREPARATION INPUT HELPERS
# ============================================================

def is_validated_preparation_input(
    dataset: dict[
        str,
        Any,
    ],
) -> bool:
    """
    Return True only for dataset records that carry
    server-owned Preparation handoff metadata.

    The browser does not choose this state.

    AnalysisInputHandoff attaches Preparation metadata
    to the exact validated output records crossing the
    Preparation -> Analysis trust boundary.

    This distinction is important because a validated
    Preparation output may already contain columns that
    were safely enriched during COMBINE. Requiring the
    Analytical View Builder to rediscover and repeat the
    original joins would be both unnecessary and wrong.
    """

    preparation_stage = str(
        dataset.get(
            "preparation_stage",
            "",
        )
    ).strip()


    if not preparation_stage:
        return False


    preparation_workflow_id = str(
        dataset.get(
            "preparation_workflow_id",
            "",
        )
    ).strip()


    analysis_input_rule_version = str(
        dataset.get(
            "analysis_input_rule_version",
            "",
        )
    ).strip()


    has_parent_metadata = (
        "preparation_parent_dataset_ids"
        in dataset
    )


    return bool(
        preparation_workflow_id
        or
        analysis_input_rule_version
        or
        has_parent_metadata
    )


def prepared_input_enrichment_columns(
    dataset: dict[
        str,
        Any,
    ],
    *,
    dataframe: pd.DataFrame,
) -> set[str]:
    """
    Return the columns that may safely participate in
    the existing propagated-column monetary policy.

    For a validated COMBINE output, the physical join
    has already happened inside Preparation and has
    crossed the server-owned validation boundary.

    The old Analytical View Builder knew that a unit
    price propagated from a validated dimension could
    be treated as one monetary event when no explicit
    quantity column existed.

    After Preparation COMBINE, that lineage is no
    longer represented as an Analysis-time join.
    Therefore the final prepared columns are exposed to
    the same conservative propagated-column rule.

    This does NOT make arbitrary numerics additive:
    select_additive_measures() still applies all of its
    existing semantic guards.
    """

    stage = normalize_text(
        dataset.get(
            "preparation_stage",
            "",
        )
    )


    raw_parent_ids = dataset.get(
        "preparation_parent_dataset_ids",
        [],
    )


    parent_ids = (
        list(
            raw_parent_ids
        )
        if isinstance(
            raw_parent_ids,
            (
                list,
                tuple,
                set,
            ),
        )
        else []
    )


    already_combined = bool(
        stage
        ==
        "combine"
        or
        len(
            parent_ids
        )
        >
        1
    )


    if not already_combined:
        return set()


    return {
        str(
            column
        )

        for column
        in dataframe.columns
    }


def annotate_prepared_materialization(
    *,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
    audits: list[
        DerivedDatasetAudit
    ],
    source_record: dict[
        str,
        Any,
    ],
    dataframe: pd.DataFrame,
    prepared_enrichment_columns: set[str],
) -> None:
    """
    Add explicit audit metadata showing that the view
    was materialized directly from a validated
    Preparation output rather than from a new
    Analysis-time join.
    """

    stage = str(
        source_record.get(
            "preparation_stage",
            "",
        )
    )


    raw_parent_ids = source_record.get(
        "preparation_parent_dataset_ids",
        [],
    )


    parent_ids = (
        list(
            raw_parent_ids
        )
        if isinstance(
            raw_parent_ids,
            (
                list,
                tuple,
                set,
            ),
        )
        else []
    )


    def annotate(
        provenance: object,
    ) -> None:
        if not isinstance(
            provenance,
            dict,
        ):
            return


        provenance[
            "materialization_input_mode"
        ] = (
            "validated_preparation_output"
        )


        provenance[
            "preparation_stage"
        ] = stage


        provenance[
            "preparation_parent_dataset_ids"
        ] = list(
            parent_ids
        )


        source_measure = str(
            provenance.get(
                "source_measure_column",
                "",
            )
        )


        if (
            source_measure
            and
            source_measure
            in prepared_enrichment_columns
            and
            has_signal(
                source_measure,
                UNIT_MONETARY_SIGNALS,
            )
            and
            not has_explicit_quantity_column(
                dataframe
            )
        ):
            provenance[
                "metric_semantics"
            ] = (
                "The unit monetary measure is present "
                "on a server-owned validated Preparation "
                "output produced after controlled "
                "preparation/enrichment. No explicit "
                "quantity measure was detected, so one "
                "prepared event row is conservatively "
                "treated as one monetary event."
            )


    for dataset in datasets:
        annotate(
            dataset.get(
                "provenance"
            )
        )


    for audit in audits:
        annotate(
            audit.provenance
        )


def materialize_validated_preparation_output(
    *,
    dataset: dict[
        str,
        Any,
    ],
    include_requested_context: bool,
) -> tuple[
    list[
        dict[
            str,
            Any,
        ]
    ],
    list[
        DerivedDatasetAudit
    ],
]:
    """
    Materialize analytical grains directly from one
    already-validated Preparation output.

    No new join is performed here.

    This is the missing Preparation-era path:

        validated final output
                ?
        already enriched event grain
                ?
        analytical materialization
    """

    dataset_id = str(
        dataset[
            "dataset_id"
        ]
    )


    filename = str(
        dataset[
            "filename"
        ]
    )


    dataframe = dataset.get(
        "dataframe"
    )


    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            (
                "Validated Preparation analytical "
                "input must contain a pandas "
                "DataFrame under 'dataframe'."
            )
        )


    original_columns = {
        str(
            column
        )

        for column
        in dataframe.columns
    }


    enrichment_columns = (
        prepared_input_enrichment_columns(
            dataset,
            dataframe=
                dataframe,
        )
    )


    (
        records,
        audits,
    ) = materialize_views_for_fact(
        fact_dataset_id=
            dataset_id,

        fact_filename=
            filename,

        enriched=
            dataframe.copy(),

        source_dataset_ids=[
            dataset_id
        ],

        fact_original_columns=
            original_columns,

        propagated_columns=
            enrichment_columns,

        include_requested_context=
            include_requested_context,
    )


    annotate_prepared_materialization(
        datasets=
            records,

        audits=
            audits,

        source_record=
            dataset,

        dataframe=
            dataframe,

        prepared_enrichment_columns=
            enrichment_columns,
    )


    return (
        records,
        audits,
    )


# ============================================================
# COMPLETE BUILDER
# ============================================================

def build_analytical_views(
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
    *,
    include_requested_context: bool = False,
) -> AnalyticalViewBuildResult:
    """
    Build conservative DataLens analytical views.

    v0.6 supports both analytical input modes and adds a
    deterministic analytical-only line monetary amount when an
    unambiguous strict quantity × unit-price pair is available.

    Both analytical input modes remain supported:

    MODE A ? multiple non-derived datasets
        source datasets
            ?
        validated N:1 enrichment
            ?
        internal enriched fact grain
            ?
        analytical materialization

    MODE B ? validated Preparation output
        server-owned final Preparation dataset
            ?
        already enriched validated grain
            ?
        analytical materialization directly

    Both modes then use the same downstream hierarchy:

        event rows
            ?
        month / category / generic entity
            ?
        session or order grain
            ?
        customer grain

    Safety rules:

    - derived analytical views are never recursively
      used as new source datasets;
    - direct relationship enrichment remains limited
      to validated 1:N or N:1 relationships;
    - Analysis-time enrichment uses LEFT JOIN semantics;
    - dimension keys must be unique;
    - fact-row count must remain exactly unchanged;
    - Analysis-time join coverage must be at least 95%;
    - validated Preparation outputs are consumed as-is
      and are never rejoined to stale source datasets;
    - direct prepared-output materialization is enabled
      only when server-owned Preparation metadata is
      present on the dataset record;
    - no automatic SUM of arbitrary numerics;
    - no SUM(unit_price) when an explicit quantity
      variable exists;
    - quantity × unit-price derivation is allowed only for one
      unambiguous strict semantic pair and remains analytical-only;
    - customer metrics are derived from session grain
      rather than directly from raw event rows;
    - purchase_sessions is an observed count, not an
      exposure-normalized purchase rate;
    - age at first purchase is approximate when only
      birth year is known.
    """

    # ========================================================
    # 1. NON-DERIVED INPUTS ONLY
    # ========================================================

    originals = [
        dataset

        for dataset
        in datasets

        if not bool(
            dataset.get(
                "is_derived",
                False,
            )
        )
    ]


    dataset_map = (
        build_dataset_map(
            originals
        )
    )


    # ========================================================
    # 2. RELATIONSHIP DISCOVERY
    #
    # This remains the normal path when Analysis receives
    # multiple datasets that still require enrichment.
    # ========================================================

    relationship_input = [
        {
            "dataset_id":
                str(
                    dataset[
                        "dataset_id"
                    ]
                ),

            "filename":
                str(
                    dataset[
                        "filename"
                    ]
                ),

            "dataframe":
                dataset[
                    "dataframe"
                ],
        }

        for dataset
        in originals
    ]


    relationships = (
        discover_relationships(
            relationship_input
        )
    )


    oriented: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for relationship in relationships:
        candidate = (
            orient_dimension_relationship(
                relationship
            )
        )


        if candidate is not None:
            oriented.append(
                candidate
            )


    relationships_by_fact: dict[
        str,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ] = {}


    for relationship in oriented:
        fact_id = str(
            relationship[
                "fact_dataset_id"
            ]
        )


        relationships_by_fact.setdefault(
            fact_id,
            [],
        ).append(
            relationship
        )


    # ========================================================
    # 3. RESULT CONTAINERS
    # ========================================================

    derived_datasets: list[
        dict[
            str,
            Any,
        ]
    ] = []


    join_audits: list[
        JoinAudit
    ] = []


    derived_audits: list[
        DerivedDatasetAudit
    ] = []


    materialized_fact_ids: set[
        str
    ] = set()


    preparation_direct_attempt_count = 0

    preparation_direct_materialized_count = 0


    notes: list[str] = [
        (
            "Analytical views are generated only "
            "from non-derived analytical inputs."
        ),

        (
            "Validated server-owned Preparation "
            "outputs may be materialized directly "
            "when their enrichment has already "
            "occurred before the Analysis handoff."
        ),

        (
            "Only validated direct 1:N or N:1 "
            "relationships are eligible for "
            "automatic Analysis-time dimension "
            "enrichment."
        ),

        (
            "Every Analysis-time automatic "
            "enrichment uses LEFT JOIN semantics "
            "and many-to-one validation."
        ),

        (
            "The fact-table row count must remain "
            "exactly unchanged after Analysis-time "
            "enrichment."
        ),

        (
            "Enriched fact tables remain internal "
            "to the Analytical View Builder."
        ),

        (
            "Only materialized datasets with an "
            "explicit analytical grain are exposed "
            "to downstream Discovery."
        ),

        (
            "Automatic SUM is restricted to "
            "conservatively validated additive "
            "monetary measures."
        ),

        (
            "A line monetary amount may be derived "
            "analytically only from exactly one strict "
            "quantity column and exactly one strict "
            "unit-price column; the validated Preparation "
            "output is never mutated."
        ),

        (
            "Session/order behaviour is "
            "materialized before customer "
            "behaviour so basket metrics respect "
            "their correct unit of observation."
        ),

        (
            "Customer purchase_sessions represents "
            "the observed number of sessions and "
            "is not interpreted as an "
            "exposure-normalized purchase rate."
        ),

        (
            "Age at first purchase may be derived "
            "when a birth-year attribute and "
            "purchase timeline are both available."
        ),

        (
            "Requested-only event context views "
            "are excluded from exploratory "
            "Discovery and are exposed only to "
            "explicit requested-analysis execution."
        ),
    ]


    # ========================================================
    # 4. MODE A ? ANALYSIS-TIME RELATIONSHIP ENRICHMENT
    # ========================================================

    for (
        fact_id,
        fact_relationships,
    ) in relationships_by_fact.items():
        fact_record = (
            dataset_map.get(
                fact_id
            )
        )


        if fact_record is None:
            continue


        fact_filename = str(
            fact_record[
                "filename"
            ]
        )


        fact_dataframe = (
            fact_record[
                "dataframe"
            ]
        )


        enriched = (
            fact_dataframe
            .copy()
        )


        fact_original_columns = {
            str(
                column
            )

            for column
            in fact_dataframe.columns
        }


        propagated_columns: set[
            str
        ] = set()


        accepted_source_ids = [
            fact_id
        ]


        accepted_join_count = 0


        fact_relationships = sorted(
            fact_relationships,

            key=lambda item:
                float(
                    item[
                        "relationship_score"
                    ]
                ),

            reverse=True,
        )


        for relationship in (
            fact_relationships
        ):
            dimension_id = str(
                relationship[
                    "dimension_dataset_id"
                ]
            )


            dimension_record = (
                dataset_map.get(
                    dimension_id
                )
            )


            if dimension_record is None:
                continue


            (
                candidate_enriched,
                audit,
            ) = safe_dimension_enrichment(
                fact=
                    enriched,

                fact_dataset_id=
                    fact_id,

                fact_filename=
                    fact_filename,

                dimension=
                    dimension_record[
                        "dataframe"
                    ],

                dimension_dataset_id=
                    dimension_id,

                dimension_filename=
                    str(
                        dimension_record[
                            "filename"
                        ]
                    ),

                fact_keys=list(
                    relationship[
                        "fact_keys"
                    ]
                ),

                dimension_keys=list(
                    relationship[
                        "dimension_keys"
                    ]
                ),

                cardinality=str(
                    relationship[
                        "cardinality"
                    ]
                ),

                relationship_score=float(
                    relationship[
                        "relationship_score"
                    ]
                ),
            )


            join_audits.append(
                audit
            )


            if (
                audit.status
                !=
                "complete"
            ):
                continue


            enriched = (
                candidate_enriched
            )


            propagated_columns.update(
                audit.added_columns
            )


            accepted_join_count += 1


            if (
                dimension_id
                not in accepted_source_ids
            ):
                accepted_source_ids.append(
                    dimension_id
                )


        if (
            accepted_join_count
            ==
            0
        ):
            continue


        (
            fact_views,
            fact_audits,
        ) = materialize_views_for_fact(
            fact_dataset_id=
                fact_id,

            fact_filename=
                fact_filename,

            enriched=
                enriched,

            source_dataset_ids=
                accepted_source_ids,

            fact_original_columns=
                fact_original_columns,

            propagated_columns=
                propagated_columns,

            include_requested_context=
                include_requested_context,
        )


        derived_datasets.extend(
            fact_views
        )


        derived_audits.extend(
            fact_audits
        )


        materialized_fact_ids.add(
            fact_id
        )


    # ========================================================
    # 5. MODE B ? VALIDATED PREPARATION OUTPUT
    #
    # This is the v0.5 correction.
    #
    # A final Preparation output may already be the exact
    # enriched event grain required by materialize_views_for_fact.
    #
    # We therefore do not require a second inter-dataset join.
    # We also do not reload or substitute old source datasets.
    # ========================================================

    for dataset in originals:
        dataset_id = str(
            dataset[
                "dataset_id"
            ]
        )


        if (
            dataset_id
            in materialized_fact_ids
        ):
            continue


        if not is_validated_preparation_input(
            dataset
        ):
            continue


        preparation_direct_attempt_count += 1


        (
            prepared_views,
            prepared_audits,
        ) = materialize_validated_preparation_output(
            dataset=
                dataset,

            include_requested_context=
                include_requested_context,
        )


        if prepared_views:
            preparation_direct_materialized_count += 1


        derived_datasets.extend(
            prepared_views
        )


        derived_audits.extend(
            prepared_audits
        )


        materialized_fact_ids.add(
            dataset_id
        )


    # ========================================================
    # 6. BUILD NOTES
    # ========================================================

    notes.append(
        (
            f"{len(join_audits)} relationship "
            "enrichment attempt(s) were audited."
        )
    )


    notes.append(
        (
            f"{sum(1 for audit in join_audits if audit.status == 'complete')} "
            "safe Analysis-time enrichment(s) "
            "were accepted."
        )
    )


    notes.append(
        (
            f"{preparation_direct_attempt_count} "
            "validated Preparation output(s) were "
            "evaluated for direct analytical "
            "materialization."
        )
    )


    notes.append(
        (
            f"{preparation_direct_materialized_count} "
            "validated Preparation output(s) "
            "produced at least one analytical view "
            "without an additional join."
        )
    )


    discoverable_count = sum(
        1

        for dataset
        in derived_datasets

        if bool(
            dataset.get(
                "discoverable",
                True,
            )
        )
    )


    requested_only_count = (
        len(
            derived_datasets
        )
        -
        discoverable_count
    )


    notes.append(
        (
            f"{discoverable_count} discoverable "
            "analytical dataset(s) were "
            "materialized."
        )
    )


    notes.append(
        (
            f"{requested_only_count} requested-only "
            "analytical context view(s) were "
            "materialized."
        )
    )


    return AnalyticalViewBuildResult(
        original_datasets=
            originals,

        derived_datasets=
            derived_datasets,

        join_audits=
            join_audits,

        derived_audits=
            derived_audits,

        notes=
            notes,
    )
