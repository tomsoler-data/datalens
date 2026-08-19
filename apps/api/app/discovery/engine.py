from __future__ import annotations

from dataclasses import (
    dataclass,
)

from itertools import (
    combinations,
)

import math
import re
import unicodedata

from typing import (
    Any,
)


import numpy as np

import pandas as pd


from app.discovery.schemas import (
    AnalysisDiscoveryReport,
    DiscoveredAnalysis,
    DiscoveredVariable,
    RelationshipSummary,
)

from app.relationships import (
    discover_relationships,
)

from app.profiling.types import (
    infer_analytical_type,
)


# ============================================================
# LIMITS / SAFEGUARDS
# ============================================================

MIN_NUMERIC_OBSERVATIONS = 20

MIN_PAIR_OBSERVATIONS = 30

MAX_FULL_PAIRWISE_NUMERIC_COLUMNS = 30

MAX_GROUP_LEVELS = 20

MAX_CATEGORICAL_ASSOCIATION_LEVELS = 15

MIN_ENTITY_LEVELS = 3

MAX_ENTITY_LEVELS = 1000


DISCOVERY_CANDIDATE_IDENTITY_RULE_VERSION = (
    "discovery_candidate_identity_v0.1"
)


# ============================================================
# SEMANTIC SIGNALS
# ============================================================

TEMPORAL_SIGNALS = {
    "year",
    "annee",
    "date",
    "datetime",
    "timestamp",
    "month",
    "mois",
    "day",
    "jour",
    "time",
}


COUNTRY_SIGNALS = {
    "country",
    "countries",
    "pays",
}


REGION_SIGNALS = {
    "region",
    "regions",
    "continent",
    "continents",
}


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


GRANULARITY_SIGNALS = {
    "granularity",
    "granularite",
    "scope",
}


ID_SIGNALS = {
    "id",
    "identifier",
    "identifiant",
    "code",
}


PERCENTAGE_SIGNALS = {
    "percent",
    "percentage",
    "pct",
    "ratio",
    "rate",
    "taux",
    "part",
}


CONCEPT_SIGNALS = {
    "water": {
        "water",
        "wash",
        "drinking",
        "eau",
    },

    "access": {
        "access",
        "service",
        "services",
        "coverage",
        "couverture",
        "basic",
        "managed",
    },

    "mortality": {
        "mortality",
        "death",
        "deaths",
        "fatality",
        "mortalityrate",
        "mortalite",
        "deces",
    },

    "population": {
        "population",
        "inhabitants",
        "habitants",
    },

    "political_stability": {
        "political",
        "stability",
        "politique",
        "stabilite",
    },

    "urbanization": {
        "urban",
        "urbain",
        "rural",
        "rurale",
        "density",
        "densite",
    },

    "economic": {
        "gdp",
        "pib",
        "income",
        "revenue",
        "revenu",
        "poverty",
        "pauvrete",
        "economic",
        "economique",
    },

    "sales": {
        "sales",
        "sale",
        "vente",
        "ventes",
        "revenue",
        "turnover",
        "ca",
    },

    "price": {
        "price",
        "prix",
        "cost",
        "cout",
        "amount",
        "montant",
    },

    "quantity": {
        "quantity",
        "quantite",
        "count",
        "nombre",
        "volume",
    },

    "age": {
        "age",
        "birth",
        "born",
        "naissance",
        "dob",
    },

    "duration": {
        "duration",
        "duree",
        "delay",
        "delai",
    },

    "quality": {
        "quality",
        "qualite",
        "score",
        "rating",
    },
}


# ============================================================
# INTERNAL PROFILE
# ============================================================

@dataclass
class ColumnProfile:
    name: str

    kind: str

    analytical_subtype: (
        str
        | None
    )

    semantic_role: str

    concepts: set[
        str
    ]

    valid_count: int

    missing_count: int

    missing_ratio: float

    unique_count: int

    unique_ratio: float

    numeric_variance: (
        float
        | None
    )

    numeric_skewness: (
        float
        | None
    )


@dataclass
class DatasetProfile:
    dataset_id: str

    filename: str

    dataframe: pd.DataFrame

    columns: dict[
        str,
        ColumnProfile
    ]

    temporal_columns: list[
        str
    ]

    quantitative_columns: list[
        str
    ]

    categorical_columns: list[
        str
    ]

    entity_columns: list[
        str
    ]

    geographic_columns: list[
        str
    ]

    granularity_columns: list[
        str
    ]

    repeated_measure_structure: (
        dict[
            str,
            Any,
        ]
        | None
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(
    value: str,
) -> str:
    normalized = (
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

    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        normalized,
    )

    return normalized.strip(
        "_"
    )


def text_tokens(
    value: str,
) -> set[
    str
]:
    normalized = (
        normalize_text(
            value
        )
    )

    return {
        token
        for token
        in normalized.split(
            "_"
        )
        if token
    }


# ============================================================
# COLUMN SEMANTICS
# ============================================================

def detect_concepts(
    column_name: str,
) -> set[
    str
]:
    normalized = (
        normalize_text(
            column_name
        )
    )

    tokens = (
        text_tokens(
            column_name
        )
    )

    concepts: set[
        str
    ] = set()


    for (
        concept,
        signals,
    ) in CONCEPT_SIGNALS.items():
        for signal in signals:
            # Exact token matches are preferred.
            # Substring matching is allowed only for
            # sufficiently long signals so short aliases
            # such as "ca" do not incorrectly match
            # unrelated names such as "categ".
            token_match = (
                signal
                in tokens
            )


            exact_name_match = (
                normalized
                ==
                signal
            )


            safe_compound_match = (
                len(
                    signal
                )
                >=
                4
                and
                signal
                in normalized
            )


            if (
                token_match
                or
                exact_name_match
                or
                safe_compound_match
            ):
                concepts.add(
                    concept
                )

                break


    return concepts


def detect_column_kind(
    series: pd.Series,
    column_name: str,
) -> str:
    """
    Return the central DataLens analytical type.

    Discovery no longer owns an independent dtype-based
    typing heuristic. The single source of truth is:

        app.profiling.types.infer_analytical_type()

    This prevents a numeric storage dtype from being
    mistaken for quantitative business meaning.
    """

    inferred = (
        infer_analytical_type(
            column_name,
            series,
        )
    )


    return str(
        inferred.get(
            "type",
            "unknown",
        )
    )


def detect_semantic_role(
    *,
    series: pd.Series,
    column_name: str,
    kind: str,
) -> str:
    """
    Infer the role a typed column plays in analysis.

    The analytical kind is already decided by the central
    type inference engine. This function enriches that type
    with domain roles such as country, region, percentage,
    identifier or measure.
    """

    tokens = (
        text_tokens(
            column_name
        )
    )


    # Geographic roles remain more informative than their
    # underlying categorical/identifier storage semantics.
    if (
        tokens
        &
        COUNTRY_SIGNALS
    ):
        return "country"


    if (
        tokens
        &
        REGION_SIGNALS
    ):
        return "region"


    if (
        tokens
        &
        GRANULARITY_SIGNALS
    ):
        return "granularity"


    # Central temporal typing also covers semantic names such
    # as birth/birth_year that are not generic time tokens.
    if (
        kind
        ==
        "temporal"
    ):
        return "time"


    if (
        tokens
        &
        PERCENTAGE_SIGNALS
    ):
        return "percentage"


    # Respect the central identifier/category distinction.
    # This is important for names such as category_code:
    # "code" alone must not override a categorical type.
    if (
        kind
        ==
        "identifier"
    ):
        return "identifier"


    if (
        kind
        ==
        "categorical"
    ):
        return "category"


    if (
        tokens
        &
        ENTITY_SIGNALS
    ):
        return "entity"


    # Legacy name fallback for data not yet richly typed.
    if (
        tokens
        &
        ID_SIGNALS
    ):
        return "identifier"


    if (
        kind
        ==
        "quantitative"
    ):
        return "measure"


    if (
        kind
        ==
        "text"
    ):
        return "text"


    return kind


# ============================================================
# PANEL / REPEATED MEASURE DETECTION
# ============================================================

def detect_repeated_measure_structure(
    dataframe: pd.DataFrame,
    *,
    entity_columns: list[
        str
    ],
    temporal_columns: list[
        str
    ],
) -> dict[
    str,
    Any,
] | None:
    if (
        not entity_columns
        or
        not temporal_columns
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


            counts = (
                subset
                .groupby(
                    entity_column,
                    dropna=True,
                )[
                    temporal_column
                ]
                .nunique()
            )


            if counts.empty:
                continue


            entity_count = int(
                len(
                    counts
                )
            )

            repeated_count = int(
                (
                    counts
                    >
                    1
                ).sum()
            )


            ratio = (
                repeated_count
                /
                entity_count
                if entity_count
                else
                0.0
            )


            if (
                repeated_count
                <
                2
                or
                ratio
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
                    repeated_count,

                "repeated_entity_ratio":
                    round(
                        ratio,
                        4,
                    ),
            }


            if (
                best_result
                is None
                or
                ratio
                >
                float(
                    best_result[
                        "repeated_entity_ratio"
                    ]
                )
            ):
                best_result = (
                    candidate
                )


    return best_result


# ============================================================
# PROFILE BUILDING
# ============================================================

def build_column_profile(
    dataframe: pd.DataFrame,
    column_name: str,
) -> ColumnProfile:
    series = dataframe[
        column_name
    ]


    analytical_type = (
        infer_analytical_type(
            column_name,
            series,
        )
    )


    kind = str(
        analytical_type.get(
            "type",
            "unknown",
        )
    )


    raw_subtype = (
        analytical_type.get(
            "subtype"
        )
    )


    analytical_subtype = (
        str(
            raw_subtype
        )
        if raw_subtype is not None
        else None
    )


    semantic_role = (
        detect_semantic_role(
            series=
                series,

            column_name=
                column_name,

            kind=
                kind,
        )
    )


    valid_count = int(
        series.notna().sum()
    )

    missing_count = int(
        series.isna().sum()
    )


    missing_ratio = (
        missing_count
        /
        len(
            dataframe
        )
        if len(
            dataframe
        )
        else 0.0
    )


    unique_count = int(
        series.nunique(
            dropna=True
        )
    )


    unique_ratio = (
        unique_count
        /
        valid_count
        if valid_count
        else 0.0
    )


    numeric_variance: (
        float
        | None
    ) = None

    numeric_skewness: (
        float
        | None
    ) = None


    if (
        kind ==
        "quantitative"
    ):
        numeric = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()


        if (
            len(
                numeric
            )
            >
            1
        ):
            variance = float(
                numeric.var(
                    ddof=1
                )
            )

            if math.isfinite(
                variance
            ):
                numeric_variance = (
                    variance
                )


        if (
            len(
                numeric
            )
            >
            2
        ):
            skewness = float(
                numeric.skew()
            )

            if math.isfinite(
                skewness
            ):
                numeric_skewness = (
                    skewness
                )


    return ColumnProfile(
        name=
            column_name,

        kind=
            kind,

        analytical_subtype=
            analytical_subtype,

        semantic_role=
            semantic_role,

        concepts=
            detect_concepts(
                column_name
            ),

        valid_count=
            valid_count,

        missing_count=
            missing_count,

        missing_ratio=
            missing_ratio,

        unique_count=
            unique_count,

        unique_ratio=
            unique_ratio,

        numeric_variance=
            numeric_variance,

        numeric_skewness=
            numeric_skewness,
    )


def build_dataset_profile(
    *,
    dataset_id: str,
    filename: str,
    dataframe: pd.DataFrame,
) -> DatasetProfile:
    columns = {
        str(
            column
        ):
            build_column_profile(
                dataframe,
                str(
                    column
                ),
            )
        for column
        in dataframe.columns
    }


    temporal_columns = [
        profile.name
        for profile
        in columns.values()
        if (
            profile.kind
            ==
            "temporal"
        )
    ]


    quantitative_columns = [
        profile.name
        for profile
        in columns.values()
        if (
            profile.kind
            ==
            "quantitative"
            and
            profile.valid_count
            >=
            MIN_NUMERIC_OBSERVATIONS
            and
            profile.unique_count
            >=
            2
        )
    ]


    categorical_columns = [
        profile.name
        for profile
        in columns.values()
        if (
            profile.kind
            in {
                "categorical",
                "boolean",
            }
        )
    ]


    entity_columns = [
        profile.name
        for profile
        in columns.values()
        if (
            profile.semantic_role
            in {
                "country",
                "entity",
                "identifier",
            }
            and
            profile.unique_count
            >=
            MIN_ENTITY_LEVELS
            and
            profile.unique_count
            <=
            MAX_ENTITY_LEVELS
        )
    ]


    geographic_columns = [
        profile.name
        for profile
        in columns.values()
        if (
            profile.semantic_role
            in {
                "country",
                "region",
            }
        )
    ]


    granularity_columns = [
        profile.name
        for profile
        in columns.values()
        if (
            profile.semantic_role
            ==
            "granularity"
        )
    ]


    repeated_structure = (
        detect_repeated_measure_structure(
            dataframe,

            entity_columns=
                entity_columns,

            temporal_columns=
                temporal_columns,
        )
    )


    return DatasetProfile(
        dataset_id=
            dataset_id,

        filename=
            filename,

        dataframe=
            dataframe,

        columns=
            columns,

        temporal_columns=
            temporal_columns,

        quantitative_columns=
            quantitative_columns,

        categorical_columns=
            categorical_columns,

        entity_columns=
            entity_columns,

        geographic_columns=
            geographic_columns,

        granularity_columns=
            granularity_columns,

        repeated_measure_structure=
            repeated_structure,
    )


# ============================================================
# SCORING HELPERS
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


def coverage_score(
    profile: ColumnProfile,
) -> float:
    return (
        1.0
        -
        profile.missing_ratio
    )


def objective_bonus(
    objective: str | None,
    *texts: str,
) -> float:
    if not objective:
        return 0.0


    objective_tokens = (
        text_tokens(
            objective
        )
    )


    if not objective_tokens:
        return 0.0


    candidate_tokens: set[
        str
    ] = set()


    for text in texts:
        candidate_tokens.update(
            text_tokens(
                text
            )
        )


    overlap = (
        objective_tokens
        &
        candidate_tokens
    )


    if not overlap:
        return 0.0


    ratio = (
        len(
            overlap
        )
        /
        len(
            objective_tokens
        )
    )


    return min(
        12.0,
        4.0
        +
        ratio
        *
        8.0,
    )


def semantic_complement_bonus(
    left: ColumnProfile,
    right: ColumnProfile,
) -> float:
    if (
        not left.concepts
        or
        not right.concepts
    ):
        return 2.0


    if (
        left.concepts
        ==
        right.concepts
    ):
        return 1.0


    if (
        left.concepts
        &
        right.concepts
    ):
        return 4.0


    return 8.0


def concept_similarity(
    left: ColumnProfile,
    right: ColumnProfile,
) -> float:
    left_tokens = (
        text_tokens(
            left.name
        )
    )

    right_tokens = (
        text_tokens(
            right.name
        )
    )


    union = (
        left_tokens
        |
        right_tokens
    )


    if not union:
        return 0.0


    overlap = (
        left_tokens
        &
        right_tokens
    )


    return (
        len(
            overlap
        )
        /
        len(
            union
        )
    )


# ============================================================
# VARIABLE BUILDER
# ============================================================

def discovered_variable(
    profile: DatasetProfile,
    column: str,
    role: str,
) -> DiscoveredVariable:
    column_profile = (
        profile.columns[
            column
        ]
    )


    return DiscoveredVariable(
        dataset_id=
            profile.dataset_id,

        dataset_filename=
            profile.filename,

        column=
            column,

        role=
            role,

        analysis_kind=
            column_profile.kind,

        semantic_role=
            column_profile.semantic_role,

        concepts=
            sorted(
                column_profile
                .concepts
            ),
    )


# ============================================================
# QUALITY DISCOVERY
# ============================================================

def discover_quality_analysis(
    profile: DatasetProfile,
    *,
    objective: str | None,
) -> DiscoveredAnalysis:
    dataframe = (
        profile.dataframe
    )


    duplicate_count = int(
        dataframe
        .duplicated()
        .sum()
    )


    missing_cells = int(
        dataframe
        .isna()
        .sum()
        .sum()
    )


    total_cells = (
        len(
            dataframe
        )
        *
        len(
            dataframe.columns
        )
    )


    missing_ratio = (
        missing_cells
        /
        total_cells
        if total_cells
        else 0.0
    )


    score = (
        72.0
        +
        min(
            18.0,
            missing_ratio
            *
            100
        )
        +
        (
            5.0
            if duplicate_count
            >
            0
            else 0.0
        )
        +
        objective_bonus(
            objective,
            profile.filename,
            "qualité données",
        )
    )


    reasons = [
        (
            "La qualité des données doit être "
            "évaluée avant l'interprétation "
            "statistique."
        ),

        (
            f"{missing_cells} cellule(s) "
            "manquante(s) ont été détectées."
        ),
    ]


    if duplicate_count:
        reasons.append(
            (
                f"{duplicate_count} ligne(s) "
                "strictement dupliquée(s) ont "
                "été détectées."
            )
        )


    return DiscoveredAnalysis(
        analysis_id=(
            f"{profile.dataset_id}:quality"
        ),

        scope=
            "single_dataset",

        family=
            "data_quality",

        title=(
            "Qualité des données — "
            f"{profile.filename}"
        ),

        priority_score=
            clamp_score(
                score
            ),

        readiness=
            "executable_now",

        datasets=[
            profile.filename
        ],

        dataset_ids=[
            profile.dataset_id
        ],

        variables=[],

        chart_type=
            "quality_summary",

        execution_strategy=
            "automatic_data_quality_scan",

        why_interesting=
            reasons,

        limitations=[],

        observed_signals={
            "missing_cells":
                missing_cells,

            "missing_ratio":
                missing_ratio,

            "duplicate_rows":
                duplicate_count,

            "row_count":
                len(
                    dataframe
                ),

            "column_count":
                len(
                    dataframe.columns
                ),
        },

        redundancy_key=(
            f"quality:{profile.dataset_id}"
        ),
    )


# ============================================================
# DISTRIBUTIONS
# ============================================================

def discover_distributions(
    profile: DatasetProfile,
    *,
    objective: str | None,
) -> list[
    DiscoveredAnalysis
]:
    results: list[
        DiscoveredAnalysis
    ] = []


    for column in (
        profile.quantitative_columns
    ):
        column_profile = (
            profile.columns[
                column
            ]
        )


        score = (
            44.0
            +
            coverage_score(
                column_profile
            )
            *
            10.0
        )


        if (
            column_profile
            .numeric_skewness
            is not None
        ):
            score += min(
                12.0,
                abs(
                    column_profile
                    .numeric_skewness
                )
                *
                4.0,
            )


        if (
            column_profile.concepts
        ):
            score += 4.0


        score += objective_bonus(
            objective,
            profile.filename,
            column,
        )


        reasons = [
            (
                f"{column} est une variable "
                "quantitative avec suffisamment "
                "d'observations pour étudier sa "
                "distribution."
            )
        ]


        if (
            column_profile
            .numeric_skewness
            is not None
            and
            abs(
                column_profile
                .numeric_skewness
            )
            >=
            1.0
        ):
            reasons.append(
                (
                    "La distribution présente "
                    "une asymétrie suffisamment "
                    "marquée pour mériter une "
                    "inspection particulière."
                )
            )


        results.append(
            DiscoveredAnalysis(
                analysis_id=(
                    f"{profile.dataset_id}:"
                    f"distribution:"
                    f"{normalize_text(column)}"
                ),

                scope=
                    "single_dataset",

                family=
                    "distribution",

                title=(
                    f"Distribution de {column}"
                ),

                priority_score=
                    clamp_score(
                        score
                    ),

                readiness=
                    "executable_now",

                datasets=[
                    profile.filename
                ],

                dataset_ids=[
                    profile.dataset_id
                ],

                variables=[
                    discovered_variable(
                        profile,
                        column,
                        "value",
                    )
                ],

                chart_type=
                    "histogram",

                execution_strategy=
                    "descriptive_distribution",

                why_interesting=
                    reasons,

                limitations=[],

                observed_signals={
                    "valid_count":
                        column_profile
                        .valid_count,

                    "missing_ratio":
                        column_profile
                        .missing_ratio,

                    "skewness":
                        column_profile
                        .numeric_skewness,
                },

                redundancy_key=(
                    f"distribution:"
                    f"{profile.dataset_id}:"
                    f"{column}"
                ),
            )
        )


    return results


# ============================================================
# TIME SERIES
# ============================================================

def build_time_series_analysis_id(
    *,
    dataset_id: str,
    time_column: str,
    value_column: str,
) -> str:
    """
    Build the canonical identity of a time-series candidate.

    A time-series analysis is defined by three structural parts:

        dataset
        temporal axis
        value measure

    The previous contract omitted ``time_column`` and could
    therefore generate the same public analysis_id for two
    distinct analyses such as:

        quantity by order_date
        quantity by signup_date

    The readable normalized components are deterministic and
    aligned with the existing DataLens ID conventions.
    """

    return (
        f"{dataset_id}:"
        f"time:"
        f"{normalize_text(time_column)}:"
        f"{normalize_text(value_column)}"
    )


def discover_time_series(
    profile: DatasetProfile,
    *,
    objective: str | None,
) -> list[
    DiscoveredAnalysis
]:
    results: list[
        DiscoveredAnalysis
    ] = []


    for time_column in (
        profile.temporal_columns
    ):
        period_count = int(
            profile.dataframe[
                time_column
            ]
            .nunique(
                dropna=True
            )
        )


        if (
            period_count
            <
            3
        ):
            continue


        group_column = (
            profile.entity_columns[
                0
            ]
            if profile.entity_columns
            else None
        )


        for value_column in (
            profile.quantitative_columns
        ):
            value_profile = (
                profile.columns[
                    value_column
                ]
            )


            score = (
                65.0
                +
                coverage_score(
                    value_profile
                )
                *
                10.0
                +
                min(
                    10.0,
                    period_count
                    / 2.0,
                )
            )


            if (
                value_profile.concepts
            ):
                score += 5.0


            score += objective_bonus(
                objective,
                profile.filename,
                time_column,
                value_column,
            )


            variables = [
                discovered_variable(
                    profile,
                    time_column,
                    "time",
                ),

                discovered_variable(
                    profile,
                    value_column,
                    "value",
                ),
            ]


            if group_column:
                variables.append(
                    discovered_variable(
                        profile,
                        group_column,
                        "group",
                    )
                )


            reasons = [
                (
                    f"{time_column} contient "
                    f"{period_count} valeurs "
                    "temporelles distinctes."
                ),

                (
                    f"{value_column} peut être "
                    "suivie dans le temps."
                ),
            ]


            if group_column:
                reasons.append(
                    (
                        f"{group_column} permet "
                        "d'étudier les trajectoires "
                        "par entité."
                    )
                )


            limitations: list[
                str
            ] = []


            if (
                profile
                .repeated_measure_structure
            ):
                limitations.append(
                    (
                        "Le dataset présente une "
                        "structure longitudinale ; "
                        "les comparaisons "
                        "inférentielles devront "
                        "respecter cette dépendance."
                    )
                )


            results.append(
                DiscoveredAnalysis(
                    analysis_id=(
                        build_time_series_analysis_id(
                            dataset_id=
                                profile.dataset_id,

                            time_column=
                                time_column,

                            value_column=
                                value_column,
                        )
                    ),

                    scope=
                        "single_dataset",

                    family=
                        "time_series",

                    title=(
                        f"Évolution de "
                        f"{value_column}"
                    ),

                    priority_score=
                        clamp_score(
                            score
                        ),

                    readiness=
                        "executable_now",

                    datasets=[
                        profile.filename
                    ],

                    dataset_ids=[
                        profile.dataset_id
                    ],

                    variables=
                        variables,

                    chart_type=
                        "line",

                    execution_strategy=
                        "descriptive_time_series",

                    why_interesting=
                        reasons,

                    limitations=
                        limitations,

                    observed_signals={
                        "period_count":
                            period_count,

                        "repeated_measure_structure":
                            profile
                            .repeated_measure_structure,
                    },

                    redundancy_key=(
                        f"time:"
                        f"{profile.dataset_id}:"
                        f"{time_column}:"
                        f"{value_column}"
                    ),
                )
            )


    return results


# ============================================================
# GROUP COMPARISONS
# ============================================================

def discover_group_comparisons(
    profile: DatasetProfile,
    *,
    objective: str | None,
) -> list[
    DiscoveredAnalysis
]:
    results: list[
        DiscoveredAnalysis
    ] = []


    candidate_groups = [
        column
        for column
        in profile.categorical_columns
        if (
            profile.columns[
                column
            ].unique_count
            >=
            2
            and
            profile.columns[
                column
            ].unique_count
            <=
            MAX_GROUP_LEVELS
        )
    ]


    for group_column in (
        candidate_groups
    ):
        for value_column in (
            profile.quantitative_columns
        ):
            working = pd.DataFrame(
                {
                    "group":
                        profile.dataframe[
                            group_column
                        ],

                    "value":
                        pd.to_numeric(
                            profile.dataframe[
                                value_column
                            ],
                            errors="coerce",
                        ),
                }
            ).dropna()


            if working.empty:
                continue


            valid_group_count = int(
                working[
                    "group"
                ]
                .nunique()
            )


            # Important correction:
            # one valid group is not a comparison.
            if (
                valid_group_count
                <
                2
            ):
                continue


            value_profile = (
                profile.columns[
                    value_column
                ]
            )


            score = (
                54.0
                +
                coverage_score(
                    value_profile
                )
                *
                10.0
                +
                min(
                    8.0,
                    valid_group_count
                    *
                    1.5,
                )
            )


            if (
                profile.columns[
                    group_column
                ].semantic_role
                ==
                "granularity"
            ):
                score += 7.0


            score += objective_bonus(
                objective,
                profile.filename,
                group_column,
                value_column,
            )


            limitations = []


            if (
                profile
                .repeated_measure_structure
            ):
                limitations.append(
                    (
                        "Les observations répétées "
                        "dans le temps empêchent "
                        "d'assumer automatiquement "
                        "l'indépendance entre "
                        "groupes."
                    )
                )


            results.append(
                DiscoveredAnalysis(
                    analysis_id=(
                        f"{profile.dataset_id}:"
                        f"group:"
                        f"{normalize_text(group_column)}:"
                        f"{normalize_text(value_column)}"
                    ),

                    scope=
                        "single_dataset",

                    family=
                        "group_comparison",

                    title=(
                        f"{value_column} selon "
                        f"{group_column}"
                    ),

                    priority_score=
                        clamp_score(
                            score
                        ),

                    readiness=
                        "executable_now",

                    datasets=[
                        profile.filename
                    ],

                    dataset_ids=[
                        profile.dataset_id
                    ],

                    variables=[
                        discovered_variable(
                            profile,
                            group_column,
                            "group",
                        ),

                        discovered_variable(
                            profile,
                            value_column,
                            "value",
                        ),
                    ],

                    chart_type=
                        "boxplot",

                    execution_strategy=
                        "automatic_group_comparison",

                    why_interesting=[
                        (
                            f"{valid_group_count} "
                            "groupes possèdent "
                            "réellement des valeurs "
                            f"pour {value_column}."
                        ),

                        (
                            "Comparer leurs "
                            "distributions peut faire "
                            "apparaître des écarts "
                            "structurels."
                        ),
                    ],

                    limitations=
                        limitations,

                    observed_signals={
                        "valid_group_count":
                            valid_group_count,

                        "valid_observations":
                            len(
                                working
                            ),
                    },

                    redundancy_key=(
                        f"group:"
                        f"{profile.dataset_id}:"
                        f"{group_column}:"
                        f"{value_column}"
                    ),
                )
            )


    return results


# ============================================================
# QUANTITATIVE ASSOCIATIONS
# ============================================================

def prioritized_quantitative_columns(
    profile: DatasetProfile,
) -> list[
    str
]:
    columns = list(
        profile.quantitative_columns
    )


    if (
        len(
            columns
        )
        <=
        MAX_FULL_PAIRWISE_NUMERIC_COLUMNS
    ):
        return columns


    ranked = sorted(
        columns,
        key=lambda column: (
            coverage_score(
                profile.columns[
                    column
                ]
            ),

            (
                profile.columns[
                    column
                ]
                .numeric_variance
                or
                0.0
            ),
        ),
        reverse=True,
    )


    return ranked[
        :
        MAX_FULL_PAIRWISE_NUMERIC_COLUMNS
    ]


def discover_quantitative_associations(
    profile: DatasetProfile,
    *,
    objective: str | None,
) -> list[
    DiscoveredAnalysis
]:
    results: list[
        DiscoveredAnalysis
    ] = []


    columns = (
        prioritized_quantitative_columns(
            profile
        )
    )


    for (
        left_column,
        right_column,
    ) in combinations(
        columns,
        2,
    ):
        pair = pd.DataFrame(
            {
                "left":
                    pd.to_numeric(
                        profile.dataframe[
                            left_column
                        ],
                        errors="coerce",
                    ),

                "right":
                    pd.to_numeric(
                        profile.dataframe[
                            right_column
                        ],
                        errors="coerce",
                    ),
            }
        ).dropna()


        if (
            len(
                pair
            )
            <
            MIN_PAIR_OBSERVATIONS
        ):
            continue


        if (
            pair[
                "left"
            ].nunique()
            <
            2
            or
            pair[
                "right"
            ].nunique()
            <
            2
        ):
            continue


        spearman = (
            pair[
                "left"
            ]
            .corr(
                pair[
                    "right"
                ],
                method="spearman",
            )
        )


        if (
            pd.isna(
                spearman
            )
        ):
            spearman_value = None

        else:
            spearman_value = float(
                spearman
            )


        left_profile = (
            profile.columns[
                left_column
            ]
        )

        right_profile = (
            profile.columns[
                right_column
            ]
        )


        pair_coverage = (
            len(
                pair
            )
            /
            len(
                profile.dataframe
            )
            if len(
                profile.dataframe
            )
            else 0.0
        )


        score = (
            48.0
            +
            pair_coverage
            *
            10.0
            +
            semantic_complement_bonus(
                left_profile,
                right_profile,
            )
        )


        if (
            spearman_value
            is not None
        ):
            score += (
                min(
                    20.0,
                    abs(
                        spearman_value
                    )
                    *
                    20.0,
                )
            )


        score += objective_bonus(
            objective,
            profile.filename,
            left_column,
            right_column,
        )


        limitations: list[
            str
        ] = []


        readiness = (
            "executable_now"
        )


        execution_strategy = (
            "automatic_correlation_decision_engine"
        )


        if (
            profile
            .repeated_measure_structure
        ):
            readiness = (
                "planned"
            )

            execution_strategy = (
                "panel_aware_association"
            )

            limitations.append(
                (
                    "Une structure de panel ou "
                    "de mesures répétées a été "
                    "détectée. Une corrélation "
                    "simple sur toutes les lignes "
                    "ne doit pas être interprétée "
                    "comme si les observations "
                    "étaient indépendantes."
                )
            )


        results.append(
            DiscoveredAnalysis(
                analysis_id=(
                    f"{profile.dataset_id}:"
                    f"association:"
                    f"{normalize_text(left_column)}:"
                    f"{normalize_text(right_column)}"
                ),

                scope=
                    "single_dataset",

                family=
                    "quantitative_association",

                title=(
                    f"Relation entre "
                    f"{left_column} et "
                    f"{right_column}"
                ),

                priority_score=
                    clamp_score(
                        score
                    ),

                readiness=
                    readiness,

                datasets=[
                    profile.filename
                ],

                dataset_ids=[
                    profile.dataset_id
                ],

                variables=[
                    discovered_variable(
                        profile,
                        left_column,
                        "x",
                    ),

                    discovered_variable(
                        profile,
                        right_column,
                        "y",
                    ),
                ],

                chart_type=(
                    "hexbin"
                    if len(
                        pair
                    )
                    >
                    3000
                    else
                    "scatter"
                ),

                execution_strategy=
                    execution_strategy,

                why_interesting=[
                    (
                        "Les deux variables sont "
                        "quantitatives et disposent "
                        f"de {len(pair)} paires "
                        "complètes."
                    ),

                    (
                        "Une relation préliminaire "
                        "a été inspectée uniquement "
                        "pour prioriser "
                        "l'exploration."
                    ),
                ],

                limitations=
                    limitations,

                observed_signals={
                    "valid_pairs":
                        len(
                            pair
                        ),

                    "pair_coverage":
                        pair_coverage,

                    "preliminary_spearman":
                        spearman_value,

                    "data_driven_ranking":
                        True,
                },

                redundancy_key=(
                    "quant_assoc:"
                    f"{profile.dataset_id}:"
                    +
                    ":".join(
                        sorted(
                            [
                                left_column,
                                right_column,
                            ]
                        )
                    )
                ),
            )
        )


    return results


# ============================================================
# DERIVED GAPS
# ============================================================

def discover_derived_gaps(
    profile: DatasetProfile,
    *,
    objective: str | None,
) -> list[
    DiscoveredAnalysis
]:
    results: list[
        DiscoveredAnalysis
    ] = []


    for (
        left_column,
        right_column,
    ) in combinations(
        profile.quantitative_columns,
        2,
    ):
        left_profile = (
            profile.columns[
                left_column
            ]
        )

        right_profile = (
            profile.columns[
                right_column
            ]
        )


        same_measure_family = (
            (
                left_profile.semantic_role
                ==
                "percentage"
                and
                right_profile.semantic_role
                ==
                "percentage"
            )
            or
            (
                left_profile.concepts
                &
                right_profile.concepts
            )
        )


        similarity = (
            concept_similarity(
                left_profile,
                right_profile,
            )
        )


        if (
            not same_measure_family
            or
            similarity
            <
            0.10
        ):
            continue


        pair = pd.DataFrame(
            {
                "left":
                    pd.to_numeric(
                        profile.dataframe[
                            left_column
                        ],
                        errors="coerce",
                    ),

                "right":
                    pd.to_numeric(
                        profile.dataframe[
                            right_column
                        ],
                        errors="coerce",
                    ),
            }
        ).dropna()


        if (
            len(
                pair
            )
            <
            MIN_PAIR_OBSERVATIONS
        ):
            continue


        gap = (
            pair[
                "left"
            ]
            -
            pair[
                "right"
            ]
        )


        median_abs_gap = float(
            gap.abs().median()
        )


        score = (
            68.0
            +
            min(
                12.0,
                similarity
                *
                20.0,
            )
            +
            objective_bonus(
                objective,
                left_column,
                right_column,
                "écart différence",
            )
        )


        results.append(
            DiscoveredAnalysis(
                analysis_id=(
                    f"{profile.dataset_id}:"
                    f"gap:"
                    f"{normalize_text(left_column)}:"
                    f"{normalize_text(right_column)}"
                ),

                scope=
                    "single_dataset",

                family=
                    "derived_gap",

                title=(
                    f"Écart entre "
                    f"{left_column} et "
                    f"{right_column}"
                ),

                priority_score=
                    clamp_score(
                        score
                    ),

                readiness=
                    "planned",

                datasets=[
                    profile.filename
                ],

                dataset_ids=[
                    profile.dataset_id
                ],

                variables=[
                    discovered_variable(
                        profile,
                        left_column,
                        "left_measure",
                    ),

                    discovered_variable(
                        profile,
                        right_column,
                        "right_measure",
                    ),
                ],

                chart_type=
                    "distribution_or_ranking",

                execution_strategy=
                    "derived_difference_analysis",

                why_interesting=[
                    (
                        "Les deux mesures semblent "
                        "décrire des dimensions "
                        "proches d'un même phénomène."
                    ),

                    (
                        "L'écart entre les deux peut "
                        "être plus informatif que "
                        "leur niveau pris séparément."
                    ),
                ],

                limitations=[
                    (
                        "La signification métier de "
                        "la différence doit rester "
                        "compatible avec les unités "
                        "et définitions des deux "
                        "mesures."
                    )
                ],

                observed_signals={
                    "valid_pairs":
                        len(
                            pair
                        ),

                    "median_absolute_gap":
                        median_abs_gap,

                    "name_similarity":
                        similarity,
                },

                redundancy_key=(
                    "gap:"
                    f"{profile.dataset_id}:"
                    +
                    ":".join(
                        sorted(
                            [
                                left_column,
                                right_column,
                            ]
                        )
                    )
                ),
            )
        )


    return results


# ============================================================
# RANKINGS / GEOGRAPHIC COMPARISONS
# ============================================================

def discover_entity_rankings(
    profile: DatasetProfile,
    *,
    objective: str | None,
) -> list[
    DiscoveredAnalysis
]:
    results: list[
        DiscoveredAnalysis
    ] = []


    candidate_entities = (
        profile.geographic_columns
        or
        profile.entity_columns
    )


    for entity_column in (
        candidate_entities
    ):
        entity_profile = (
            profile.columns[
                entity_column
            ]
        )


        if (
            entity_profile.unique_count
            <
            MIN_ENTITY_LEVELS
        ):
            continue


        for value_column in (
            profile.quantitative_columns
        ):
            value_profile = (
                profile.columns[
                    value_column
                ]
            )


            score = (
                58.0
                +
                coverage_score(
                    value_profile
                )
                *
                10.0
            )


            if (
                entity_profile.semantic_role
                in {
                    "country",
                    "region",
                }
            ):
                score += 8.0


            if (
                value_profile.concepts
            ):
                score += 4.0


            score += objective_bonus(
                objective,
                entity_column,
                value_column,
                "classement priorité",
            )


            chart_type = (
                "map_or_bar"
                if entity_profile
                .semantic_role
                in {
                    "country",
                    "region",
                }
                else
                "ranked_bar"
            )


            results.append(
                DiscoveredAnalysis(
                    analysis_id=(
                        f"{profile.dataset_id}:"
                        f"ranking:"
                        f"{normalize_text(entity_column)}:"
                        f"{normalize_text(value_column)}"
                    ),

                    scope=
                        "single_dataset",

                    family=(
                        "geographic_comparison"
                        if entity_profile
                        .semantic_role
                        in {
                            "country",
                            "region",
                        }
                        else
                        "ranking"
                    ),

                    title=(
                        f"Classement de "
                        f"{entity_column} selon "
                        f"{value_column}"
                    ),

                    priority_score=
                        clamp_score(
                            score
                        ),

                    readiness=
                        "planned",

                    datasets=[
                        profile.filename
                    ],

                    dataset_ids=[
                        profile.dataset_id
                    ],

                    variables=[
                        discovered_variable(
                            profile,
                            entity_column,
                            "entity",
                        ),

                        discovered_variable(
                            profile,
                            value_column,
                            "value",
                        ),
                    ],

                    chart_type=
                        chart_type,

                    execution_strategy=
                        "automatic_entity_ranking",

                    why_interesting=[
                        (
                            "Cette analyse permet de "
                            "repérer les entités aux "
                            "valeurs les plus fortes "
                            "ou les plus faibles."
                        )
                    ],

                    limitations=[
                        (
                            "Si plusieurs périodes ou "
                            "granularités existent, "
                            "elles devront être "
                            "alignées avant le "
                            "classement final."
                        )
                    ],

                    observed_signals={
                        "entity_count":
                            entity_profile
                            .unique_count,

                        "has_time_dimension":
                            bool(
                                profile
                                .temporal_columns
                            ),
                    },

                    redundancy_key=(
                        f"ranking:"
                        f"{profile.dataset_id}:"
                        f"{entity_column}:"
                        f"{value_column}"
                    ),
                )
            )


    return results


# ============================================================
# CATEGORICAL ASSOCIATIONS
# ============================================================

def discover_categorical_associations(
    profile: DatasetProfile,
    *,
    objective: str | None,
) -> list[
    DiscoveredAnalysis
]:
    results: list[
        DiscoveredAnalysis
    ] = []


    candidates = [
        column
        for column
        in profile.categorical_columns
        if (
            profile.columns[
                column
            ].unique_count
            >=
            2
            and
            profile.columns[
                column
            ].unique_count
            <=
            MAX_CATEGORICAL_ASSOCIATION_LEVELS
            and
            profile.columns[
                column
            ].semantic_role
            not in {
                "identifier",
                "entity",
                "country",
            }
        )
    ]


    for (
        left_column,
        right_column,
    ) in combinations(
        candidates,
        2,
    ):
        working = (
            profile.dataframe[
                [
                    left_column,
                    right_column,
                ]
            ]
            .dropna()
        )


        if (
            len(
                working
            )
            <
            30
        ):
            continue


        score = (
            54.0
            +
            objective_bonus(
                objective,
                left_column,
                right_column,
            )
        )


        results.append(
            DiscoveredAnalysis(
                analysis_id=(
                    f"{profile.dataset_id}:"
                    f"categorical:"
                    f"{normalize_text(left_column)}:"
                    f"{normalize_text(right_column)}"
                ),

                scope=
                    "single_dataset",

                family=
                    "categorical_association",

                title=(
                    f"Association entre "
                    f"{left_column} et "
                    f"{right_column}"
                ),

                priority_score=
                    clamp_score(
                        score
                    ),

                readiness=(
                    "planned"
                    if profile
                    .repeated_measure_structure
                    else
                    "executable_now"
                ),

                datasets=[
                    profile.filename
                ],

                dataset_ids=[
                    profile.dataset_id
                ],

                variables=[
                    discovered_variable(
                        profile,
                        left_column,
                        "x",
                    ),

                    discovered_variable(
                        profile,
                        right_column,
                        "y",
                    ),
                ],

                chart_type=
                    "heatmap",

                execution_strategy=(
                    "repeated_measure_categorical_association"
                    if profile
                    .repeated_measure_structure
                    else
                    "chi_square_decision_engine"
                ),

                why_interesting=[
                    (
                        "Les deux variables "
                        "catégorielles possèdent un "
                        "nombre de modalités "
                        "compatible avec une analyse "
                        "de contingence."
                    )
                ],

                limitations=(
                    [
                        (
                            "Une structure de mesures "
                            "répétées a été détectée."
                        )
                    ]
                    if profile
                    .repeated_measure_structure
                    else []
                ),

                observed_signals={
                    "valid_observations":
                        len(
                            working
                        ),

                    "left_levels":
                        int(
                            working[
                                left_column
                            ].nunique()
                        ),

                    "right_levels":
                        int(
                            working[
                                right_column
                            ].nunique()
                        ),
                },

                redundancy_key=(
                    "categorical:"
                    f"{profile.dataset_id}:"
                    +
                    ":".join(
                        sorted(
                            [
                                left_column,
                                right_column,
                            ]
                        )
                    )
                ),
            )
        )


    return results


# ============================================================
# CROSS-DATASET DISCOVERY
# ============================================================

def relationship_status_from_mode(
    mode: str,
) -> str:
    if (
        mode ==
        "direct"
    ):
        return "validated"


    if (
        mode ==
        "matched_subset"
    ):
        return "partial"


    return "requires_alignment"


def profile_by_id(
    profiles: list[
        DatasetProfile
    ],
) -> dict[
    str,
    DatasetProfile
]:
    return {
        profile.dataset_id:
            profile
        for profile
        in profiles
    }


def discover_cross_dataset_associations(
    *,
    profiles: list[
        DatasetProfile
    ],
    relationships: list[
        dict[
            str,
            Any,
        ]
    ],
    objective: str | None,
) -> tuple[
    list[
        DiscoveredAnalysis
    ],
    list[
        RelationshipSummary
    ],
]:
    results: list[
        DiscoveredAnalysis
    ] = []

    relationship_summaries: list[
        RelationshipSummary
    ] = []


    profiles_map = (
        profile_by_id(
            profiles
        )
    )


    relationship_index = 0


    for relationship in (
        relationships
    ):
        best = relationship.get(
            "best_candidate"
        )


        if not best:
            continue


        left_info = relationship[
            "left_dataset"
        ]

        right_info = relationship[
            "right_dataset"
        ]


        left_profile = (
            profiles_map.get(
                left_info[
                    "dataset_id"
                ]
            )
        )

        right_profile = (
            profiles_map.get(
                right_info[
                    "dataset_id"
                ]
            )
        )


        if (
            left_profile
            is None
            or
            right_profile
            is None
        ):
            continue


        mode = str(
            best[
                "relationship_mode"
            ]
        )


        relationship_summaries.append(
            RelationshipSummary(
                left_dataset=
                    left_profile.filename,

                right_dataset=
                    right_profile.filename,

                relationship_mode=
                    mode,

                cardinality=
                    str(
                        best[
                            "cardinality"
                        ]
                    ),

                score=
                    float(
                        best[
                            "score"
                        ]
                    ),

                left_match_rate=
                    float(
                        best[
                            "left_match_rate"
                        ]
                    ),

                right_match_rate=
                    float(
                        best[
                            "right_match_rate"
                        ]
                    ),

                overlap_rate=
                    float(
                        best[
                            "overlap_rate"
                        ]
                    ),

                left_columns=list(
                    best[
                        "left_columns"
                    ]
                ),

                right_columns=list(
                    best[
                        "right_columns"
                    ]
                ),

                key_roles=list(
                    best[
                        "key_roles"
                    ]
                ),

                usable_for_analysis=bool(
                    best[
                        "usable_for_analysis"
                    ]
                ),

                requires_grain_alignment=bool(
                    best[
                        "requires_grain_alignment"
                    ]
                ),

                warnings=list(
                    best[
                        "warnings"
                    ]
                ),
            )
        )


        if (
            mode ==
            "blocked"
        ):
            continue


        relationship_index += 1


        left_key_columns = set(
            best[
                "left_columns"
            ]
        )

        right_key_columns = set(
            best[
                "right_columns"
            ]
        )


        left_measures = [
            column
            for column
            in left_profile
            .quantitative_columns
            if (
                column
                not in left_key_columns
            )
        ]


        right_measures = [
            column
            for column
            in right_profile
            .quantitative_columns
            if (
                column
                not in right_key_columns
            )
        ]


        for left_column in (
            left_measures
        ):
            for right_column in (
                right_measures
            ):
                left_column_profile = (
                    left_profile.columns[
                        left_column
                    ]
                )

                right_column_profile = (
                    right_profile.columns[
                        right_column
                    ]
                )


                score = (
                    46.0
                    +
                    float(
                        best[
                            "score"
                        ]
                    )
                    *
                    0.35
                    +
                    semantic_complement_bonus(
                        left_column_profile,
                        right_column_profile,
                    )
                )


                score += objective_bonus(
                    objective,
                    left_profile.filename,
                    right_profile.filename,
                    left_column,
                    right_column,
                )


                relationship_status = (
                    relationship_status_from_mode(
                        mode
                    )
                )


                readiness = (
                    "planned"
                    if mode
                    in {
                        "direct",
                        "matched_subset",
                    }
                    else
                    "requires_alignment"
                )


                execution_strategy = (
                    "cross_dataset_quantitative_association"
                    if mode
                    in {
                        "direct",
                        "matched_subset",
                    }
                    else
                    "grain_alignment_then_association"
                )


                limitations: list[
                    str
                ] = []


                if (
                    mode ==
                    "matched_subset"
                ):
                    limitations.append(
                        (
                            "L'analyse devra signaler "
                            "explicitement les clés "
                            "exclues lors de "
                            "l'alignement des deux "
                            "datasets."
                        )
                    )


                if (
                    mode ==
                    "grain_alignment_required"
                ):
                    limitations.append(
                        (
                            "Une jointure ligne à "
                            "ligne n'est pas sûre. "
                            "Les datasets doivent "
                            "d'abord être ramenés à "
                            "un grain analytique "
                            "commun."
                        )
                    )


                results.append(
                    DiscoveredAnalysis(
                        analysis_id=(
                            "cross:"
                            f"{relationship_index}:"
                            f"{normalize_text(left_column)}:"
                            f"{normalize_text(right_column)}"
                        ),

                        scope=
                            "cross_dataset",

                        family=
                            "quantitative_association",

                        title=(
                            f"Relation entre "
                            f"{left_column} et "
                            f"{right_column}"
                        ),

                        priority_score=
                            clamp_score(
                                score
                            ),

                        readiness=
                            readiness,

                        datasets=[
                            left_profile.filename,
                            right_profile.filename,
                        ],

                        dataset_ids=[
                            left_profile.dataset_id,
                            right_profile.dataset_id,
                        ],

                        variables=[
                            discovered_variable(
                                left_profile,
                                left_column,
                                "x",
                            ),

                            discovered_variable(
                                right_profile,
                                right_column,
                                "y",
                            ),
                        ],

                        chart_type=
                            "scatter_or_hexbin",

                        execution_strategy=
                            execution_strategy,

                        why_interesting=[
                            (
                                "Les deux mesures "
                                "proviennent de "
                                "datasets reliés par "
                                "une clé analytique "
                                "identifiée."
                            ),

                            (
                                "Cette analyse peut "
                                "révéler une relation "
                                "qui n'est pas visible "
                                "en étudiant chaque "
                                "fichier séparément."
                            ),
                        ],

                        limitations=
                            limitations,

                        relationship_status=
                            relationship_status,

                        relationship_score=
                            float(
                                best[
                                    "score"
                                ]
                            ),

                        join_keys={
                            left_profile.dataset_id:
                                list(
                                    best[
                                        "left_columns"
                                    ]
                                ),

                            right_profile.dataset_id:
                                list(
                                    best[
                                        "right_columns"
                                    ]
                                ),
                        },

                        observed_signals={
                            "cardinality":
                                best[
                                    "cardinality"
                                ],

                            "left_match_rate":
                                best[
                                    "left_match_rate"
                                ],

                            "right_match_rate":
                                best[
                                    "right_match_rate"
                                ],

                            "overlap_rate":
                                best[
                                    "overlap_rate"
                                ],

                            "relationship_mode":
                                mode,
                        },

                        redundancy_key=(
                            "cross_assoc:"
                            +
                            ":".join(
                                sorted(
                                    [
                                        (
                                            f"{left_profile.dataset_id}:"
                                            f"{left_column}"
                                        ),

                                        (
                                            f"{right_profile.dataset_id}:"
                                            f"{right_column}"
                                        ),
                                    ]
                                )
                            )
                        ),
                    )
                )


    return (
        results,
        relationship_summaries,
    )


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_candidates(
    candidates: list[
        DiscoveredAnalysis
    ],
) -> list[
    DiscoveredAnalysis
]:
    selected: dict[
        str,
        DiscoveredAnalysis
    ] = {}


    for candidate in (
        candidates
    ):
        existing = (
            selected.get(
                candidate
                .redundancy_key
            )
        )


        if (
            existing
            is None
            or
            candidate
            .priority_score
            >
            existing
            .priority_score
        ):
            selected[
                candidate
                .redundancy_key
            ] = (
                candidate
            )


    return list(
        selected.values()
    )


# ============================================================
# MAIN DISCOVERY ENGINE
# ============================================================

def discover_analyses(
    *,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
    objective: str | None = None,
) -> AnalysisDiscoveryReport:
    profiles: list[
        DatasetProfile
    ] = []


    for dataset in (
        datasets
    ):
        dataframe = dataset[
            "dataframe"
        ]


        if (
            not isinstance(
                dataframe,
                pd.DataFrame,
            )
        ):
            raise TypeError(
                (
                    "Each dataset must contain "
                    "a pandas DataFrame under "
                    "the 'dataframe' key."
                )
            )


        profiles.append(
            build_dataset_profile(
                dataset_id=str(
                    dataset[
                        "dataset_id"
                    ]
                ),

                filename=str(
                    dataset[
                        "filename"
                    ]
                ),

                dataframe=
                    dataframe,
            )
        )


    candidates: list[
        DiscoveredAnalysis
    ] = []


    for profile in (
        profiles
    ):
        candidates.append(
            discover_quality_analysis(
                profile,
                objective=
                    objective,
            )
        )


        candidates.extend(
            discover_distributions(
                profile,
                objective=
                    objective,
            )
        )


        candidates.extend(
            discover_time_series(
                profile,
                objective=
                    objective,
            )
        )


        candidates.extend(
            discover_group_comparisons(
                profile,
                objective=
                    objective,
            )
        )


        candidates.extend(
            discover_quantitative_associations(
                profile,
                objective=
                    objective,
            )
        )


        candidates.extend(
            discover_derived_gaps(
                profile,
                objective=
                    objective,
            )
        )


        candidates.extend(
            discover_entity_rankings(
                profile,
                objective=
                    objective,
            )
        )


        candidates.extend(
            discover_categorical_associations(
                profile,
                objective=
                    objective,
            )
        )


    relationship_input = [
        {
            "dataset_id":
                profile.dataset_id,

            "filename":
                profile.filename,

            "dataframe":
                profile.dataframe,
        }
        for profile
        in profiles
    ]


    relationships = (
        discover_relationships(
            relationship_input
        )
    )


    (
        cross_candidates,
        relationship_summaries,
    ) = (
        discover_cross_dataset_associations(
            profiles=
                profiles,

            relationships=
                relationships,

            objective=
                objective,
        )
    )


    candidates.extend(
        cross_candidates
    )


    candidates = (
        deduplicate_candidates(
            candidates
        )
    )


    candidates.sort(
        key=lambda candidate:
            (
                candidate
                .priority_score,

                1
                if (
                    candidate.scope
                    ==
                    "cross_dataset"
                )
                else
                0,
            ),
        reverse=True,
    )


    single_count = sum(
        1
        for candidate
        in candidates
        if (
            candidate.scope
            ==
            "single_dataset"
        )
    )


    cross_count = sum(
        1
        for candidate
        in candidates
        if (
            candidate.scope
            ==
            "cross_dataset"
        )
    )


    notes = [
        (
            "Le moteur ne limite plus "
            "arbitrairement le rapport aux "
            "16 premières analyses."
        ),

        (
            "Les comparaisons de groupes sont "
            "validées sur les observations "
            "réellement disponibles pour chaque "
            "mesure ; une variable avec un seul "
            "groupe valide n'est plus proposée "
            "comme comparaison."
        ),

        (
            "Les relations quantitatives sont "
            "priorisées à l'aide d'un signal "
            "exploratoire préliminaire. Ce signal "
            "n'est pas une conclusion statistique."
        ),

        (
            "Les analyses inter-datasets "
            "utilisent les relations détectées "
            "par le Relationship Engine et "
            "conservent explicitement la "
            "cardinalité, la couverture et le "
            "besoin éventuel d'aligner le grain."
        ),

        (
            "La sélection finale du rapport "
            "devra être réévaluée après "
            "exécution à partir de l'importance "
            "réelle des résultats et de leur "
            "redondance."
        ),

        (
            "L'identité des candidats de séries temporelles "
            "inclut explicitement le dataset, la colonne "
            "temporelle et la mesure afin d'éviter les "
            "collisions entre analyses distinctes. "
            "Règle : "
            f"{DISCOVERY_CANDIDATE_IDENTITY_RULE_VERSION}."
        ),
    ]


    if (
        any(
            len(
                profile
                .quantitative_columns
            )
            >
            MAX_FULL_PAIRWISE_NUMERIC_COLUMNS
            for profile
            in profiles
        )
    ):
        notes.append(
            (
                "Au moins un dataset contient "
                "plus de 30 variables "
                "quantitatives exploitables. "
                "Un garde-fou de calcul a "
                "priorisé les variables les mieux "
                "couvertes et les plus variables "
                "pour l'exploration pairwise."
            )
        )


    return AnalysisDiscoveryReport(
        objective=
            objective,

        dataset_count=
            len(
                profiles
            ),

        candidate_count=
            len(
                candidates
            ),

        single_dataset_candidate_count=
            single_count,

        cross_dataset_candidate_count=
            cross_count,

        candidates=
            candidates,

        relationships=
            relationship_summaries,

        discovery_notes=
            notes,
    )