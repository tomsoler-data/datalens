from __future__ import annotations

import re
import unicodedata

from app.semantics.schemas import (
    SemanticQuantityDimension,
    SemanticQuantityUnit,
)


# ============================================================
# VERSION
# ============================================================

QUANTITY_RULE_VERSION = (
    "quantity_semantics_v0.1"
)


UNKNOWN = "unknown"


# ============================================================
# CONTEXT SIGNALS
# ============================================================

DURATION_CONTEXT_SIGNALS = {
    "duration",
    "time",
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


DATA_SIZE_CONTEXT_SIGNALS = {
    "storage",
    "memory",
    "size",
    "capacity",
    "disk",
    "volume",
}


# ============================================================
# UNIT TOKENS
#
# Used by the semantic-group normalizer so that physical units
# do not become part of the concept itself.
#
# Example:
#
# provisioned_storage_gb
# used_storage_tb
#
# should both collapse to:
#
# storage
# ============================================================

QUANTITY_UNIT_TOKENS = {
    # Duration
    "second",
    "seconds",
    "sec",
    "secs",
    "minute",
    "minutes",
    "min",
    "mins",
    "hour",
    "hours",
    "hr",
    "hrs",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",

    # Data size
    "b",
    "byte",
    "bytes",
    "kb",
    "kilobyte",
    "kilobytes",
    "mb",
    "megabyte",
    "megabytes",
    "gb",
    "gigabyte",
    "gigabytes",
    "tb",
    "terabyte",
    "terabytes",

    # Mass
    "g",
    "gram",
    "grams",
    "kg",
    "kilogram",
    "kilograms",
    "tonne",
    "tonnes",

    # Distance
    "metre",
    "metres",
    "meter",
    "meters",
    "km",
    "kilometre",
    "kilometres",
    "kilometer",
    "kilometers",

    # Energy
    "wh",
    "kwh",
    "mwh",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_quantity_text(
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


    return text.strip(
        "_"
    )


def quantity_tokens(
    value: str,
) -> set[
    str
]:
    normalized = (
        normalize_quantity_text(
            value
        )
    )


    if not normalized:
        return set()


    return {
        token
        for token
        in normalized.split(
            "_"
        )
        if token
    }


# ============================================================
# EXPLICIT UNIT EXTRACTION
# ============================================================

def infer_explicit_unit(
    column: str,
) -> tuple[
    SemanticQuantityDimension,
    SemanticQuantityUnit,
]:
    raw = str(
        column
    )


    tokens = quantity_tokens(
        column
    )


    # --------------------------------------------------------
    # Data size
    #
    # Abbreviations such as GB are interpreted as data size
    # only when the column also contains a storage-like signal.
    # This keeps the rule conservative.
    # --------------------------------------------------------

    has_data_size_context = bool(
        tokens
        &
        DATA_SIZE_CONTEXT_SIGNALS
    )


    if has_data_size_context:
        if (
            re.search(
                r"\bTB\b",
                raw,
            )
            or
            "terabyte"
            in tokens
            or
            "terabytes"
            in tokens
        ):
            return (
                "data_size",
                "terabyte",
            )


        if (
            re.search(
                r"\bGB\b",
                raw,
            )
            or
            "gigabyte"
            in tokens
            or
            "gigabytes"
            in tokens
        ):
            return (
                "data_size",
                "gigabyte",
            )


        if (
            re.search(
                r"\bMB\b",
                raw,
            )
            or
            "megabyte"
            in tokens
            or
            "megabytes"
            in tokens
        ):
            return (
                "data_size",
                "megabyte",
            )


        if (
            re.search(
                r"\bKB\b",
                raw,
            )
            or
            "kilobyte"
            in tokens
            or
            "kilobytes"
            in tokens
        ):
            return (
                "data_size",
                "kilobyte",
            )


        if (
            "byte"
            in tokens
            or
            "bytes"
            in tokens
        ):
            return (
                "data_size",
                "byte",
            )


    # --------------------------------------------------------
    # Energy
    # --------------------------------------------------------

    if (
        re.search(
            r"\bMWh\b",
            raw,
        )
        or
        "mwh"
        in tokens
    ):
        return (
            "energy",
            "megawatt_hour",
        )


    if (
        re.search(
            r"\bkWh\b",
            raw,
        )
        or
        "kwh"
        in tokens
    ):
        return (
            "energy",
            "kilowatt_hour",
        )


    if (
        re.search(
            r"\bWh\b",
            raw,
        )
        or
        "wh"
        in tokens
    ):
        return (
            "energy",
            "watt_hour",
        )


    # --------------------------------------------------------
    # Mass
    # --------------------------------------------------------

    if (
        "kg"
        in tokens
        or
        "kilogram"
        in tokens
        or
        "kilograms"
        in tokens
    ):
        return (
            "mass",
            "kilogram",
        )


    if (
        "tonne"
        in tokens
        or
        "tonnes"
        in tokens
    ):
        return (
            "mass",
            "tonne",
        )


    if (
        "gram"
        in tokens
        or
        "grams"
        in tokens
    ):
        return (
            "mass",
            "gram",
        )


    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    if (
        "km"
        in tokens
        or
        "kilometre"
        in tokens
        or
        "kilometres"
        in tokens
        or
        "kilometer"
        in tokens
        or
        "kilometers"
        in tokens
    ):
        return (
            "distance",
            "kilometre",
        )


    if (
        "metre"
        in tokens
        or
        "metres"
        in tokens
        or
        "meter"
        in tokens
        or
        "meters"
        in tokens
    ):
        return (
            "distance",
            "metre",
        )


    # --------------------------------------------------------
    # Duration
    #
    # The exact unit is extracted here, but the caller still
    # decides whether the column really represents a duration.
    # --------------------------------------------------------

    if (
        "second"
        in tokens
        or
        "seconds"
        in tokens
        or
        "sec"
        in tokens
        or
        "secs"
        in tokens
    ):
        return (
            "duration",
            "second",
        )


    if (
        "minute"
        in tokens
        or
        "minutes"
        in tokens
        or
        "min"
        in tokens
        or
        "mins"
        in tokens
    ):
        return (
            "duration",
            "minute",
        )


    if (
        "hour"
        in tokens
        or
        "hours"
        in tokens
        or
        "hr"
        in tokens
        or
        "hrs"
        in tokens
    ):
        return (
            "duration",
            "hour",
        )


    if (
        "day"
        in tokens
        or
        "days"
        in tokens
    ):
        return (
            "duration",
            "day",
        )


    if (
        "week"
        in tokens
        or
        "weeks"
        in tokens
    ):
        return (
            "duration",
            "week",
        )


    if (
        "month"
        in tokens
        or
        "months"
        in tokens
    ):
        return (
            "duration",
            "month",
        )


    if (
        "year"
        in tokens
        or
        "years"
        in tokens
    ):
        return (
            "duration",
            "year",
        )


    return (
        "unknown",
        "unknown",
    )


# ============================================================
# QUANTITY INFERENCE
# ============================================================

def infer_quantity_semantics(
    *,
    column: str,
    measure_kind: str,
    unit_kind: str,
) -> tuple[
    SemanticQuantityDimension,
    SemanticQuantityUnit,
]:
    tokens = quantity_tokens(
        column
    )


    explicit_dimension, explicit_unit = (
        infer_explicit_unit(
            column
        )
    )


    normalized_measure = (
        str(
            measure_kind
        )
        .strip()
        .casefold()
    )


    normalized_unit = (
        str(
            unit_kind
        )
        .strip()
        .casefold()
    )


    # --------------------------------------------------------
    # Count
    # --------------------------------------------------------

    if (
        normalized_measure
        ==
        "count"
        or
        normalized_unit
        ==
        "count"
    ):
        return (
            "count",
            "count",
        )


    # --------------------------------------------------------
    # Proportion / percentage
    # --------------------------------------------------------

    if (
        normalized_measure
        ==
        "percentage"
        or
        normalized_unit
        ==
        "percent"
    ):
        return (
            "proportion",
            "percent",
        )


    if (
        normalized_unit
        ==
        "proportion"
    ):
        return (
            "proportion",
            "proportion",
        )


    # --------------------------------------------------------
    # Rate
    # --------------------------------------------------------

    if (
        normalized_measure
        ==
        "rate"
        or
        normalized_unit
        ==
        "rate"
    ):
        return (
            "rate",
            "rate",
        )


    # --------------------------------------------------------
    # Currency
    # --------------------------------------------------------

    if (
        normalized_measure
        ==
        "currency"
        or
        normalized_unit
        ==
        "currency"
    ):
        return (
            "currency",
            "currency",
        )


    # --------------------------------------------------------
    # Duration
    #
    # A temporal word by itself is not enough.
    #
    # "Year" can represent a time dimension rather than a
    # duration.
    # --------------------------------------------------------

    duration_context = (
        normalized_measure
        ==
        "duration"
        or
        normalized_unit
        ==
        "duration"
        or
        bool(
            tokens
            &
            DURATION_CONTEXT_SIGNALS
        )
    )


    if (
        explicit_dimension
        ==
        "duration"
        and
        duration_context
    ):
        return (
            "duration",
            explicit_unit,
        )


    if (
        normalized_measure
        ==
        "duration"
        or
        normalized_unit
        ==
        "duration"
    ):
        return (
            "duration",
            "unknown",
        )


    # --------------------------------------------------------
    # Physical / technical dimensions
    # --------------------------------------------------------

    if (
        explicit_dimension
        in {
            "data_size",
            "mass",
            "distance",
            "energy",
        }
    ):
        return (
            explicit_dimension,
            explicit_unit,
        )


    return (
        "unknown",
        "unknown",
    )


# ============================================================
# COMPATIBILITY
# ============================================================

def dimensions_are_compatible(
    left_dimension: str,
    right_dimension: str,
) -> bool:
    if (
        left_dimension
        ==
        UNKNOWN
        or
        right_dimension
        ==
        UNKNOWN
    ):
        return False


    return (
        left_dimension
        ==
        right_dimension
    )


def units_are_directly_comparable(
    *,
    left_dimension: str,
    left_unit: str,
    right_dimension: str,
    right_unit: str,
) -> bool:
    if not dimensions_are_compatible(
        left_dimension,
        right_dimension,
    ):
        return False


    if (
        left_unit
        ==
        UNKNOWN
        or
        right_unit
        ==
        UNKNOWN
    ):
        return False


    return (
        left_unit
        ==
        right_unit
    )


def is_numeric_quantity_dimension(
    dimension: str,
) -> bool:
    return (
        dimension
        in {
            "count",
            "proportion",
            "rate",
            "currency",
            "duration",
            "data_size",
            "mass",
            "distance",
            "energy",
        }
    )
