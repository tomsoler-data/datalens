from __future__ import annotations

import re
import unicodedata

from app.semantics.quantity import (
    QUANTITY_RULE_VERSION,
    QUANTITY_UNIT_TOKENS,
    infer_quantity_semantics,
)

from app.semantics.schemas import (
    ColumnSemanticProfile,
    DatasetSemanticProfile,
)


# ============================================================
# CONSTANTS
# ============================================================

UNKNOWN = "unknown"


NORMALIZER_RULE_VERSION = (
    "semantic_normalizer_v0.4"
)


# ============================================================
# STRONG MEASURE / UNIT SIGNALS
# ============================================================

CURRENCY_SIGNALS = {
    "price",
    "revenue",
    "cost",
    "profit",
    "income",
    "salary",
    "wage",
    "fee",
    "budget",
    "turnover",
}


COUNT_SIGNALS = {
    "count",
    "number",
    "quantity",
    "qty",
    "unit",
    "units",
    "order",
    "orders",
    "session",
    "sessions",
    "visit",
    "visits",
    "customer",
    "customers",
    "user",
    "users",
    "transaction",
    "transactions",
    "death",
    "deaths",
    "delivery",
    "deliveries",
    "job",
    "jobs",
    "core",
    "cores",
}


DURATION_SIGNALS = {
    "duration",
    "tenure",
    "age",
    "downtime",
    "runtime",
    "uptime",
    "latency",
    "delay",
    "elapsed",
    "waiting",
    "wait",
    "processing",
    "cycle",
    "transit",
    "queue",
}


DURATION_UNIT_SIGNALS = {
    "second",
    "seconds",
    "minute",
    "minutes",
    "hour",
    "hours",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
}


PERCENT_SIGNALS = {
    "percent",
    "percentage",
    "pct",
}


# ============================================================
# EXPLICIT VARIANTS
# ============================================================

VARIANT_PHRASES = [
    (
        (
            "safely",
            "managed",
        ),
        "safely_managed",
    ),
    (
        (
            "at",
            "least",
            "basic",
        ),
        "basic",
    ),
]


VARIANT_TOKENS = {
    # Commercial
    "retail":
        "retail",

    "wholesale":
        "wholesale",

    "gross":
        "gross",

    "net":
        "net",

    # Planning / observation
    "actual":
        "actual",

    "observed":
        "actual",

    "target":
        "target",

    "goal":
        "target",

    "planned":
        "planned",

    "plan":
        "planned",

    "forecast":
        "forecast",

    "forecasted":
        "forecast",

    "expected":
        "expected",

    "requested":
        "requested",

    "allocated":
        "allocated",

    "provisioned":
        "provisioned",

    "used":
        "used",

    # Process states
    "ordered":
        "ordered",

    "shipped":
        "shipped",

    "delivered":
        "delivered",

    "completed":
        "completed",

    "late":
        "late",

    "failed":
        "failed",

    "returned":
        "returned",

    "refunded":
        "refunded",

    "rejected":
        "rejected",

    "reject":
        "rejected",

    "scrapped":
        "scrapped",

    "scrap":
        "scrapped",

    "open":
        "open",

    "closed":
        "closed",

    # Population / domain
    "basic":
        "basic",

    "urban":
        "urban",

    "rural":
        "rural",

    "male":
        "male",

    "female":
        "female",

    "men":
        "male",

    "women":
        "female",

    # Temporal comparison
    "current":
        "current",

    "previous":
        "previous",

    "prior":
        "previous",
}


# ============================================================
# VALUES THAT SHOULD NOT BE USED AS SEMANTIC VARIANTS
# ============================================================

NON_VARIANT_VALUES = {
    "",
    "unknown",
    "amount",
    "value",
    "values",
    "duration",
    "month",
    "months",
    "year",
    "years",
    "day",
    "days",
    "week",
    "weeks",
    "hour",
    "hours",
    "minute",
    "minutes",
    "second",
    "seconds",
    "count",
    "rate",
    "percentage",
    "percent",
    "score",
    "index",
    "price",
    "revenue",
    "sales",
    "currency",
}


# ============================================================
# GROUP NORMALIZATION
# ============================================================

GROUP_NOISE_TOKENS = {
    "at",
    "least",
    "using",
    "use",
    "of",
    "the",
    "a",
    "an",
    "amount",
    "value",
    "values",
    "number",
    "count",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(
    value: str,
) -> str:
    text = unicodedata.normalize(
        "NFKD",
        str(
            value
        ),
    )


    text = text.encode(
        "ascii",
        "ignore",
    ).decode(
        "ascii"
    )


    text = text.lower()


    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )


    text = re.sub(
        r"_+",
        "_",
        text,
    )


    normalized = text.strip(
        "_"
    )


    return (
        normalized
        or
        UNKNOWN
    )


def tokenize(
    value: str,
) -> list[
    str
]:
    normalized = normalize_text(
        value
    )


    if (
        normalized
        ==
        UNKNOWN
    ):
        return []


    return [
        token
        for token
        in normalized.split(
            "_"
        )
        if token
    ]


# ============================================================
# VARIANT INFERENCE
# ============================================================

def infer_supported_variant(
    column: str,
) -> str:
    tokens = tokenize(
        column
    )


    token_tuple = tuple(
        tokens
    )


    for phrase, variant in (
        VARIANT_PHRASES
    ):
        phrase_length = len(
            phrase
        )


        for index in range(
            0,
            (
                len(
                    token_tuple
                )
                -
                phrase_length
                +
                1
            ),
        ):
            if (
                token_tuple[
                    index:
                    index
                    +
                    phrase_length
                ]
                ==
                phrase
            ):
                return variant


    for token in tokens:
        if (
            token
            in VARIANT_TOKENS
        ):
            return (
                VARIANT_TOKENS[
                    token
                ]
            )


    return UNKNOWN


def validate_existing_variant(
    *,
    column: str,
    existing_variant: str,
) -> str:
    inferred = (
        infer_supported_variant(
            column
        )
    )


    if (
        inferred
        !=
        UNKNOWN
    ):
        return inferred


    normalized_existing = (
        normalize_text(
            existing_variant
        )
    )


    if (
        normalized_existing
        in NON_VARIANT_VALUES
        or
        normalized_existing
        in QUANTITY_UNIT_TOKENS
    ):
        return UNKNOWN


    column_tokens = set(
        tokenize(
            column
        )
    )


    variant_tokens = set(
        tokenize(
            normalized_existing
        )
    )


    if (
        variant_tokens
        and
        variant_tokens.issubset(
            column_tokens
        )
    ):
        return normalized_existing


    return UNKNOWN


# ============================================================
# MEASURE / LEGACY UNIT INFERENCE
# ============================================================

def infer_measure_and_unit(
    *,
    column: str,
    existing_measure_kind: str,
    existing_unit_kind: str,
) -> tuple[
    str,
    str,
]:
    tokens = set(
        tokenize(
            column
        )
    )


    normalized_measure_kind = (
        normalize_text(
            existing_measure_kind
        )
    )


    normalized_unit_kind = (
        normalize_text(
            existing_unit_kind
        )
    )


    has_duration_signal = bool(
        tokens
        &
        DURATION_SIGNALS
    )


    has_duration_unit = bool(
        tokens
        &
        DURATION_UNIT_SIGNALS
    )


    # Percentage
    if (
        "%"
        in str(
            column
        )
        or
        bool(
            tokens
            &
            PERCENT_SIGNALS
        )
    ):
        return (
            "percentage",
            "percent",
        )


    # Explicit rate
    if (
        "rate"
        in tokens
    ):
        return (
            "rate",
            "rate",
        )


    # Currency
    if (
        tokens
        &
        CURRENCY_SIGNALS
    ):
        return (
            "currency",
            "currency",
        )


    if (
        "sales"
        in tokens
        and
        (
            "amount"
            in tokens
            or
            "revenue"
            in tokens
        )
    ):
        return (
            "currency",
            "currency",
        )


    # Duration
    if (
        has_duration_unit
        and
        (
            has_duration_signal
            or
            normalized_measure_kind
            ==
            "duration"
        )
    ):
        return (
            "duration",
            "duration",
        )


    if (
        normalized_measure_kind
        ==
        "duration"
        and
        normalized_unit_kind
        ==
        "duration"
    ):
        return (
            "duration",
            "duration",
        )


    # Count
    if (
        tokens
        &
        COUNT_SIGNALS
    ):
        return (
            "count",
            "count",
        )


    measure_kind = (
        existing_measure_kind
        if (
            existing_measure_kind
            !=
            UNKNOWN
        )
        else
        UNKNOWN
    )


    unit_kind = (
        existing_unit_kind
        if (
            existing_unit_kind
            !=
            UNKNOWN
        )
        else
        UNKNOWN
    )


    return (
        measure_kind,
        unit_kind,
    )


# ============================================================
# CANONICAL GROUP
# ============================================================

def canonical_group_tokens(
    *,
    column: str,
    variant: str,
) -> list[
    str
]:
    tokens = tokenize(
        column
    )


    variant_tokens = set(
        tokenize(
            variant
        )
    )


    output: list[
        str
    ] = []


    for token in tokens:
        if (
            token
            in GROUP_NOISE_TOKENS
        ):
            continue


        if (
            token
            in variant_tokens
        ):
            continue


        if (
            token
            in DURATION_UNIT_SIGNALS
        ):
            continue


        if (
            token
            in QUANTITY_UNIT_TOKENS
        ):
            continue


        if (
            token
            in PERCENT_SIGNALS
        ):
            continue


        output.append(
            token
        )


    return output


def candidate_lexical_group(
    *,
    column: str,
    variant: str,
) -> str:
    tokens = (
        canonical_group_tokens(
            column=
                column,

            variant=
                variant,
        )
    )


    if not tokens:
        return UNKNOWN


    return "_".join(
        tokens
    )


# ============================================================
# SHARED GROUP RECONCILIATION
# ============================================================

def reconcile_shared_groups(
    profiles: list[
        ColumnSemanticProfile
    ],
) -> dict[
    str,
    str,
]:
    proposed: dict[
        str,
        str,
    ] = {}


    for profile in profiles:
        variant = (
            validate_existing_variant(
                column=
                    profile.column,

                existing_variant=
                    profile.variant,
            )
        )


        proposed[
            profile.column
        ] = (
            candidate_lexical_group(
                column=
                    profile.column,

                variant=
                    variant,
            )
        )


    group_counts: dict[
        str,
        int,
    ] = {}


    for group in (
        proposed.values()
    ):
        if (
            group
            ==
            UNKNOWN
        ):
            continue


        group_counts[
            group
        ] = (
            group_counts.get(
                group,
                0,
            )
            +
            1
        )


    resolved: dict[
        str,
        str,
    ] = {}


    for profile in profiles:
        proposed_group = (
            proposed[
                profile.column
            ]
        )


        if (
            proposed_group
            !=
            UNKNOWN
            and
            group_counts.get(
                proposed_group,
                0,
            )
            >=
            2
        ):
            resolved[
                profile.column
            ] = proposed_group

        else:
            resolved[
                profile.column
            ] = (
                profile.semantic_group
            )


    return resolved


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_column_semantics(
    *,
    profile: ColumnSemanticProfile,
    semantic_group: str,
) -> ColumnSemanticProfile:
    variant = (
        validate_existing_variant(
            column=
                profile.column,

            existing_variant=
                profile.variant,
        )
    )


    (
        measure_kind,
        unit_kind,
    ) = infer_measure_and_unit(
        column=
            profile.column,

        existing_measure_kind=
            profile.measure_kind,

        existing_unit_kind=
            profile.unit_kind,
    )


    (
        quantity_dimension,
        quantity_unit,
    ) = infer_quantity_semantics(
        column=
            profile.column,

        measure_kind=
            measure_kind,

        unit_kind=
            unit_kind,
    )


    return profile.model_copy(
        update={
            "semantic_group":
                semantic_group,

            "variant":
                variant,

            "measure_kind":
                measure_kind,

            "unit_kind":
                unit_kind,

            "quantity_dimension":
                quantity_dimension,

            "quantity_unit":
                quantity_unit,

            "quantity_rule_version":
                QUANTITY_RULE_VERSION,
        }
    )


# ============================================================
# DATASET NORMALIZATION
# ============================================================

def normalize_dataset_semantics(
    dataset_profile: DatasetSemanticProfile,
) -> DatasetSemanticProfile:
    shared_groups = (
        reconcile_shared_groups(
            dataset_profile.columns
        )
    )


    normalized_columns = [
        normalize_column_semantics(
            profile=
                profile,

            semantic_group=
                shared_groups[
                    profile.column
                ],
        )

        for profile
        in dataset_profile.columns
    ]


    return dataset_profile.model_copy(
        update={
            "columns":
                normalized_columns,
        }
    )
