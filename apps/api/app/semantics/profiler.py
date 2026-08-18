from __future__ import annotations

import re
import unicodedata

from typing import (
    Any,
)

import pandas as pd

from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from app.ai.provider import (
    DEFAULT_MODEL,
)

from app.semantics.provider import (
    call_column_semantic_model,
)

from app.semantics.schemas import (
    ColumnSemanticDraft,
    ColumnSemanticProfile,
    DatasetSemanticProfile,
    SemanticEntityRole,
    SemanticMeasureKind,
    SemanticUnitKind,
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_SEMANTIC_ATTEMPTS = 2

MAX_SAMPLE_VALUES = 5


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_semantic_label(
    value: str,
) -> str:
    value = unicodedata.normalize(
        "NFKD",
        str(
            value
        ),
    )


    value = value.encode(
        "ascii",
        "ignore",
    ).decode(
        "ascii"
    )


    value = value.lower()


    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )


    value = re.sub(
        r"_+",
        "_",
        value,
    )


    return value.strip(
        "_"
    ) or "unknown"


# ============================================================
# STRUCTURAL TYPE
# ============================================================

def infer_data_type(
    *,
    column: str,
    series: pd.Series,
) -> str:
    normalized = (
        normalize_semantic_label(
            column
        )
    )


    if is_bool_dtype(
        series.dtype
    ):
        return "boolean"


    if is_datetime64_any_dtype(
        series.dtype
    ):
        return "temporal"


    if is_numeric_dtype(
        series.dtype
    ):
        if (
            normalized
            in {
                "year",
                "annee",
            }
            or
            normalized.endswith(
                "_year"
            )
        ):
            return "temporal"


        return "quantitative"


    return "categorical"


# ============================================================
# PYTHON STRUCTURAL HINTS
# ============================================================

def infer_entity_role_hint(
    *,
    column: str,
    data_type: str,
) -> SemanticEntityRole:
    normalized = (
        normalize_semantic_label(
            column
        )
    )


    if (
        normalized
        in {
            "year",
            "annee",
            "date",
            "datetime",
            "timestamp",
            "month",
            "quarter",
        }
        or
        normalized.endswith(
            "_date"
        )
        or
        normalized.endswith(
            "_year"
        )
    ):
        return "time"


    geographic_signals = [
        "country",
        "region",
        "continent",
        "city",
        "state",
        "province",
        "territory",
    ]


    if any(
        signal
        in normalized
        for signal
        in geographic_signals
    ):
        return "geography"


    if (
        normalized
        in {
            "id",
            "identifier",
            "code",
        }
        or
        normalized.endswith(
            "_id"
        )
        or
        normalized.endswith(
            "_code"
        )
    ):
        return "identifier"


    if (
        data_type
        ==
        "quantitative"
    ):
        return "measure"


    if (
        data_type
        ==
        "temporal"
    ):
        return "time"


    if (
        data_type
        ==
        "boolean"
    ):
        return "dimension"


    if (
        data_type
        ==
        "categorical"
    ):
        return "dimension"


    return "unknown"


def infer_measure_kind_hint(
    *,
    column: str,
    data_type: str,
) -> SemanticMeasureKind:
    normalized = (
        normalize_semantic_label(
            column
        )
    )


    original = str(
        column
    ).lower()


    if (
        data_type
        ==
        "temporal"
    ):
        return "datetime"


    if (
        data_type
        ==
        "boolean"
    ):
        return "boolean"


    if (
        "%"
        in original
        or
        "percentage"
        in normalized
        or
        "percent"
        in normalized
        or
        normalized.endswith(
            "_pct"
        )
    ):
        return "percentage"


    if (
        "rate"
        in normalized
        or
        "ratio"
        in normalized
    ):
        return "rate"


    count_signals = [
        "population",
        "deaths",
        "count",
        "number",
        "quantity",
        "qty",
    ]


    if (
        data_type
        ==
        "quantitative"
        and
        any(
            signal
            in normalized
            for signal
            in count_signals
        )
    ):
        return "count"


    if (
        data_type
        ==
        "quantitative"
        and
        any(
            signal
            in normalized
            for signal
            in [
                "index",
                "score",
            ]
        )
    ):
        return "index"


    if (
        data_type
        ==
        "categorical"
    ):
        return "category"


    return "unknown"


def infer_unit_kind_hint(
    *,
    column: str,
    data_type: str,
) -> SemanticUnitKind:
    normalized = (
        normalize_semantic_label(
            column
        )
    )


    original = str(
        column
    ).lower()


    if (
        data_type
        ==
        "temporal"
    ):
        if (
            "year"
            in normalized
            or
            normalized
            ==
            "annee"
        ):
            return "year"


        return "date"


    if (
        data_type
        ==
        "boolean"
    ):
        return "boolean"


    if (
        "%"
        in original
        or
        "percentage"
        in normalized
        or
        "percent"
        in normalized
        or
        normalized.endswith(
            "_pct"
        )
    ):
        return "percent"


    if (
        "rate"
        in normalized
        or
        "ratio"
        in normalized
    ):
        return "rate"


    count_signals = [
        "population",
        "deaths",
        "count",
        "number",
        "quantity",
        "qty",
    ]


    if (
        data_type
        ==
        "quantitative"
        and
        any(
            signal
            in normalized
            for signal
            in count_signals
        )
    ):
        return "count"


    if (
        data_type
        ==
        "quantitative"
        and
        any(
            signal
            in normalized
            for signal
            in [
                "index",
                "score",
            ]
        )
    ):
        return "score"


    if (
        data_type
        ==
        "categorical"
    ):
        return "category"


    return "unknown"


# ============================================================
# VARIANT HINT
# ============================================================

def infer_variant_hint(
    column: str,
) -> str:
    normalized = (
        normalize_semantic_label(
            column
        )
    )


    phrase_variants = [
        (
            (
                "safely_managed",
            ),
            "safely_managed",
        ),
        (
            (
                "at_least_basic",
            ),
            "basic",
        ),
        (
            (
                "basic",
            ),
            "basic",
        ),
        (
            (
                "urban",
            ),
            "urban",
        ),
        (
            (
                "rural",
            ),
            "rural",
        ),
        (
            (
                "female",
                "women",
            ),
            "female",
        ),
        (
            (
                "male",
                "men",
            ),
            "male",
        ),
        (
            (
                "gross",
            ),
            "gross",
        ),
        (
            (
                "net",
            ),
            "net",
        ),
        (
            (
                "actual",
                "observed",
            ),
            "actual",
        ),
        (
            (
                "target",
                "goal",
            ),
            "target",
        ),
        (
            (
                "previous",
                "prior",
            ),
            "previous",
        ),
        (
            (
                "current",
            ),
            "current",
        ),
    ]


    for signals, variant in (
        phrase_variants
    ):
        if any(
            signal
            in normalized
            for signal
            in signals
        ):
            return variant


    return "unknown"


# ============================================================
# DETERMINISTIC STRUCTURAL SEMANTICS
# ============================================================

def deterministic_structural_semantics(
    *,
    column: str,
    data_type: str,
) -> dict[
    str,
    str,
] | None:
    normalized = (
        normalize_semantic_label(
            column
        )
    )


    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    if (
        normalized
        in {
            "year",
            "annee",
        }
        or
        normalized.endswith(
            "_year"
        )
    ):
        return {
            "concept":
                "year",

            "domain":
                "time",

            "semantic_group":
                "time",

            "variant":
                "unknown",
        }


    if (
        normalized
        in {
            "date",
            "datetime",
            "timestamp",
            "month",
            "quarter",
        }
        or
        normalized.endswith(
            "_date"
        )
    ):
        return {
            "concept":
                normalized,

            "domain":
                "time",

            "semantic_group":
                "time",

            "variant":
                "unknown",
        }


    # --------------------------------------------------------
    # GEOGRAPHY
    # --------------------------------------------------------

    geography_map = {
        "country":
            "country",

        "region":
            "region",

        "continent":
            "continent",

        "city":
            "city",

        "state":
            "state",

        "province":
            "province",

        "territory":
            "territory",
    }


    for signal, concept in (
        geography_map.items()
    ):
        if (
            normalized
            ==
            signal
            or
            normalized.startswith(
                f"{signal}_"
            )
            or
            normalized.endswith(
                f"_{signal}"
            )
            or
            f"_{signal}_"
            in f"_{normalized}_"
        ):
            return {
                "concept":
                    concept,

                "domain":
                    "geography",

                "semantic_group":
                    "geography",

                "variant":
                    "unknown",
            }


    # --------------------------------------------------------
    # OBSERVATION STRUCTURE
    # --------------------------------------------------------

    if (
        normalized
        in {
            "granularity",
            "granularite",
        }
    ):
        return {
            "concept":
                "observation_granularity",

            "domain":
                "metadata",

            "semantic_group":
                "observation_structure",

            "variant":
                "unknown",
        }


    # --------------------------------------------------------
    # IDENTIFIERS
    # --------------------------------------------------------

    if (
        normalized
        in {
            "id",
            "identifier",
            "code",
        }
        or
        normalized.endswith(
            "_id"
        )
        or
        normalized.endswith(
            "_code"
        )
    ):
        return {
            "concept":
                "identifier",

            "domain":
                "metadata",

            "semantic_group":
                "identifier",

            "variant":
                "unknown",
        }


    return None


# ============================================================
# VALUES / STATISTICS
# ============================================================

def json_safe_value(
    value: Any,
) -> Any:
    if value is None:
        return None


    try:
        if pd.isna(
            value
        ):
            return None

    except Exception:
        pass


    if hasattr(
        value,
        "item",
    ):
        try:
            value = value.item()

        except Exception:
            pass


    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value


    return str(
        value
    )


def sample_values(
    series: pd.Series,
) -> list[
    Any
]:
    values = (
        series
        .dropna()
        .drop_duplicates()
        .head(
            MAX_SAMPLE_VALUES
        )
        .tolist()
    )


    return [
        json_safe_value(
            value
        )
        for value
        in values
    ]


def numeric_summary(
    series: pd.Series,
) -> dict[
    str,
    float | None,
]:
    if not is_numeric_dtype(
        series.dtype
    ):
        return {}


    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()


    if numeric.empty:
        return {}


    return {
        "minimum":
            float(
                numeric.min()
            ),

        "maximum":
            float(
                numeric.max()
            ),

        "median":
            float(
                numeric.median()
            ),
    }


# ============================================================
# CONTEXT
# ============================================================

def build_column_context(
    *,
    dataset_id: str,
    filename: str,
    column: str,
    series: pd.Series,
    peer_columns: list[
        str
    ],
) -> dict[
    str,
    Any,
]:
    data_type = infer_data_type(
        column=
            column,

        series=
            series,
    )


    entity_role_hint = (
        infer_entity_role_hint(
            column=
                column,

            data_type=
                data_type,
        )
    )


    measure_kind_hint = (
        infer_measure_kind_hint(
            column=
                column,

            data_type=
                data_type,
        )
    )


    unit_kind_hint = (
        infer_unit_kind_hint(
            column=
                column,

            data_type=
                data_type,
        )
    )


    variant_hint = (
        infer_variant_hint(
            column
        )
    )


    row_count = len(
        series
    )


    missing_count = int(
        series.isna().sum()
    )


    unique_count = int(
        series.nunique(
            dropna=True
        )
    )


    missing_ratio = (
        float(
            missing_count
            /
            row_count
        )
        if row_count
        else 0.0
    )


    return {
        "dataset_id":
            dataset_id,

        "filename":
            filename,

        "target_column":
            column,

        "python_data_type":
            data_type,

        "python_hints": {
            "entity_role":
                entity_role_hint,

            "measure_kind":
                measure_kind_hint,

            "unit_kind":
                unit_kind_hint,

            "variant":
                variant_hint,
        },

        "dtype":
            str(
                series.dtype
            ),

        "row_count":
            row_count,

        "missing_ratio":
            round(
                missing_ratio,
                6,
            ),

        "unique_count":
            unique_count,

        "sample_values":
            sample_values(
                series
            ),

        "numeric_summary":
            numeric_summary(
                series
            ),

        "peer_columns":
            [
                peer
                for peer
                in peer_columns
                if (
                    peer
                    !=
                    column
                )
            ],
    }


# ============================================================
# DETERMINISTIC STRUCTURAL PROFILE
# ============================================================

def build_deterministic_structural_profile(
    *,
    context: dict[
        str,
        Any,
    ],
) -> ColumnSemanticProfile | None:
    semantics = (
        deterministic_structural_semantics(
            column=
                context[
                    "target_column"
                ],

            data_type=
                context[
                    "python_data_type"
                ],
        )
    )


    if semantics is None:
        return None


    hints = context[
        "python_hints"
    ]


    return ColumnSemanticProfile(
        dataset_id=
            context[
                "dataset_id"
            ],

        filename=
            context[
                "filename"
            ],

        column=
            context[
                "target_column"
            ],

        data_type=
            context[
                "python_data_type"
            ],

        concept=
            semantics[
                "concept"
            ],

        domain=
            semantics[
                "domain"
            ],

        semantic_group=
            semantics[
                "semantic_group"
            ],

        variant=
            semantics[
                "variant"
            ],

        measure_kind=
            hints[
                "measure_kind"
            ],

        unit_kind=
            hints[
                "unit_kind"
            ],

        entity_role=
            hints[
                "entity_role"
            ],

        qualifiers=
            [],

        confidence=
            "high",

        source=
            "deterministic",
    )


# ============================================================
# DRAFT VALIDATION / PYTHON OVERRIDES
# ============================================================

def profile_from_draft(
    *,
    draft: ColumnSemanticDraft,
    context: dict[
        str,
        Any,
    ],
) -> ColumnSemanticProfile:
    hints = context[
        "python_hints"
    ]


    entity_role = (
        hints[
            "entity_role"
        ]
        if (
            hints[
                "entity_role"
            ]
            !=
            "unknown"
        )
        else
        draft.entity_role
    )


    measure_kind = (
        hints[
            "measure_kind"
        ]
        if (
            hints[
                "measure_kind"
            ]
            !=
            "unknown"
        )
        else
        draft.measure_kind
    )


    unit_kind = (
        hints[
            "unit_kind"
        ]
        if (
            hints[
                "unit_kind"
            ]
            !=
            "unknown"
        )
        else
        draft.unit_kind
    )


    variant = (
        hints[
            "variant"
        ]
        if (
            hints[
                "variant"
            ]
            !=
            "unknown"
        )
        else
        normalize_semantic_label(
            draft.variant
        )
    )


    concept = normalize_semantic_label(
        draft.concept
    )


    domain = normalize_semantic_label(
        draft.domain
    )


    semantic_group = (
        normalize_semantic_label(
            draft.semantic_group
        )
    )


    qualifiers = []


    seen = set()


    for qualifier in (
        draft.qualifiers
    ):
        normalized = (
            normalize_semantic_label(
                qualifier
            )
        )


        if (
            normalized
            ==
            "unknown"
            or
            normalized
            in seen
        ):
            continue


        seen.add(
            normalized
        )


        qualifiers.append(
            normalized
        )


    return ColumnSemanticProfile(
        dataset_id=
            context[
                "dataset_id"
            ],

        filename=
            context[
                "filename"
            ],

        column=
            context[
                "target_column"
            ],

        data_type=
            context[
                "python_data_type"
            ],

        concept=
            concept,

        domain=
            domain,

        semantic_group=
            semantic_group,

        variant=
            variant,

        measure_kind=
            measure_kind,

        unit_kind=
            unit_kind,

        entity_role=
            entity_role,

        qualifiers=
            qualifiers,

        confidence=
            draft.confidence,

        source=
            "llm",
    )


# ============================================================
# FALLBACK
# ============================================================

def build_fallback_profile(
    *,
    context: dict[
        str,
        Any,
    ],
) -> ColumnSemanticProfile:
    hints = context[
        "python_hints"
    ]


    normalized_column = (
        normalize_semantic_label(
            context[
                "target_column"
            ]
        )
    )


    return ColumnSemanticProfile(
        dataset_id=
            context[
                "dataset_id"
            ],

        filename=
            context[
                "filename"
            ],

        column=
            context[
                "target_column"
            ],

        data_type=
            context[
                "python_data_type"
            ],

        concept=
            normalized_column,

        domain=
            "unknown",

        semantic_group=
            normalized_column,

        variant=
            hints[
                "variant"
            ],

        measure_kind=
            hints[
                "measure_kind"
            ],

        unit_kind=
            hints[
                "unit_kind"
            ],

        entity_role=
            hints[
                "entity_role"
            ],

        qualifiers=
            [],

        confidence=
            "low",

        source=
            "deterministic_fallback",
    )


# ============================================================
# PUBLIC COLUMN PROFILER
# ============================================================

def profile_column_semantics(
    *,
    dataset_id: str,
    filename: str,
    column: str,
    series: pd.Series,
    peer_columns: list[
        str
    ],
    model: str = DEFAULT_MODEL,
) -> ColumnSemanticProfile:
    context = build_column_context(
        dataset_id=
            dataset_id,

        filename=
            filename,

        column=
            column,

        series=
            series,

        peer_columns=
            peer_columns,
    )


    deterministic_profile = (
        build_deterministic_structural_profile(
            context=
                context,
        )
    )


    if (
        deterministic_profile
        is not None
    ):
        return deterministic_profile


    for attempt in range(
        MAX_SEMANTIC_ATTEMPTS
    ):
        try:
            content = (
                call_column_semantic_model(
                    context=
                        context,

                    model=
                        model,

                    strict_retry=(
                        attempt > 0
                    ),
                )
            )


            draft = (
                ColumnSemanticDraft
                .model_validate_json(
                    content
                )
            )


            return profile_from_draft(
                draft=
                    draft,

                context=
                    context,
            )


        except Exception:
            continue


    return build_fallback_profile(
        context=
            context,
    )


# ============================================================
# PUBLIC DATASET PROFILER
# ============================================================

def profile_dataset_semantics(
    *,
    dataset_id: str,
    filename: str,
    dataframe: pd.DataFrame,
    model: str = DEFAULT_MODEL,
) -> DatasetSemanticProfile:
    columns = list(
        dataframe.columns
    )


    profiles = [
        profile_column_semantics(
            dataset_id=
                dataset_id,

            filename=
                filename,

            column=
                column,

            series=
                dataframe[
                    column
                ],

            peer_columns=
                columns,

            model=
                model,
        )
        for column
        in columns
    ]


    return DatasetSemanticProfile(
        dataset_id=
            dataset_id,

        filename=
            filename,

        columns=
            profiles,
    )


# ============================================================
# MULTI-DATASET PROFILER
# ============================================================

def profile_datasets_semantics(
    *,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
    model: str = DEFAULT_MODEL,
) -> list[
    DatasetSemanticProfile
]:
    results: list[
        DatasetSemanticProfile
    ] = []


    for dataset in datasets:
        dataframe = dataset[
            "dataframe"
        ]


        results.append(
            profile_dataset_semantics(
                dataset_id=
                    dataset[
                        "dataset_id"
                    ],

                filename=
                    dataset[
                        "filename"
                    ],

                dataframe=
                    dataframe,

                model=
                    model,
            )
        )


    return results
