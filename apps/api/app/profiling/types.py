from __future__ import annotations


import re
import unicodedata

from typing import Any


import pandas as pd


from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)


# ============================================================
# ANALYTICAL TYPE INFERENCE — VERSION
# ============================================================

ANALYTICAL_TYPE_RULE_VERSION = (
    "analytical_type_v0.2"
)


# ============================================================
# NAME SIGNALS
# ============================================================

YEAR_NAME_TOKENS = {
    "year",
    "annee",
}


BIRTH_NAME_TOKENS = {
    "birth",
    "born",
    "naissance",
    "dob",
}


CATEGORY_NAME_TOKENS = {
    "category",
    "categorie",
    "categ",
    "class",
    "classe",
    "segment",
    "group",
    "groupe",
    "type",
    "status",
    "statut",
    "gender",
    "genre",
    "sex",
    "tier",
    "band",
    "bucket",
}


STRONG_IDENTIFIER_TOKENS = {
    "id",
    "identifier",
    "identifiant",
    "uuid",
}


CODE_NAME_TOKENS = {
    "code",
}


# ============================================================
# SAFEGUARDS
# ============================================================

PLAUSIBLE_YEAR_MIN = 1800

PLAUSIBLE_YEAR_MAX = 2200


MAX_NAMED_CATEGORY_LEVELS = 100

MAX_NAMED_CATEGORY_RATIO = 0.05


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_semantic_name(
    value: str,
) -> str:
    """
    Normalize a column name for deterministic
    semantic comparisons.

    Accents, casing and separators are normalized
    without modifying the actual dataframe column.
    """

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


def semantic_name_tokens(
    value: str,
) -> set[str]:
    normalized = (
        normalize_semantic_name(
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
# NUMERIC HELPERS
# ============================================================

def numeric_non_null_values(
    series: pd.Series,
) -> pd.Series:
    """
    Return the values that can safely be interpreted
    as numeric for semantic inspection.

    This function does not modify the source series.
    """

    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .dropna()
    )


def values_are_integer_like(
    values: pd.Series,
) -> bool:
    """
    Return True when every numeric value represents
    an integer, including values physically stored
    as floats such as 0.0, 1.0 and 2.0.
    """

    if values.empty:
        return False


    fractional_part = (
        values
        .astype(float)
        %
        1
    )


    return bool(
        fractional_part
        .abs()
        .le(
            1e-9
        )
        .all()
    )


def values_are_plausible_years(
    values: pd.Series,
) -> bool:
    """
    Check whether numeric values can reasonably
    represent calendar years.
    """

    if values.empty:
        return False


    if not values_are_integer_like(
        values
    ):
        return False


    minimum = float(
        values.min()
    )

    maximum = float(
        values.max()
    )


    return bool(
        minimum
        >=
        PLAUSIBLE_YEAR_MIN

        and

        maximum
        <=
        PLAUSIBLE_YEAR_MAX
    )


# ============================================================
# NAME-SIGNAL HELPERS
# ============================================================

def has_birth_year_signal(
    column_name: str,
) -> bool:
    tokens = (
        semantic_name_tokens(
            column_name
        )
    )


    normalized = (
        normalize_semantic_name(
            column_name
        )
    )


    if (
        tokens
        &
        BIRTH_NAME_TOKENS
    ):
        return True


    explicit_names = {
        "birthyear",
        "birth_year",
        "year_of_birth",
        "birth_date_year",
        "annee_naissance",
        "annee_de_naissance",
    }


    return (
        normalized
        in explicit_names
    )


def has_year_signal(
    column_name: str,
) -> bool:
    tokens = (
        semantic_name_tokens(
            column_name
        )
    )


    if (
        tokens
        &
        YEAR_NAME_TOKENS
    ):
        return True


    return has_birth_year_signal(
        column_name
    )


def has_category_signal(
    column_name: str,
) -> bool:
    tokens = (
        semantic_name_tokens(
            column_name
        )
    )


    normalized = (
        normalize_semantic_name(
            column_name
        )
    )


    if (
        tokens
        &
        CATEGORY_NAME_TOKENS
    ):
        return True


    explicit_names = {
        "cat",
        "categ",
        "category_code",
        "categorie_code",
        "class_code",
        "classe_code",
        "segment_code",
    }


    return (
        normalized
        in explicit_names
    )


def has_strong_identifier_signal(
    column_name: str,
) -> bool:
    tokens = (
        semantic_name_tokens(
            column_name
        )
    )


    normalized = (
        normalize_semantic_name(
            column_name
        )
    )


    if (
        tokens
        &
        STRONG_IDENTIFIER_TOKENS
    ):
        return True


    if (
        normalized
        ==
        "id"
    ):
        return True


    if (
        normalized.startswith(
            "id_"
        )
        or
        normalized.endswith(
            "_id"
        )
    ):
        return True


    return False


def has_code_identifier_signal(
    column_name: str,
) -> bool:
    """
    A generic 'code' is weaker than an explicit ID.

    category_code, for example, must not automatically
    become an identifier.
    """

    if has_category_signal(
        column_name
    ):
        return False


    tokens = (
        semantic_name_tokens(
            column_name
        )
    )


    return bool(
        tokens
        &
        CODE_NAME_TOKENS
    )


# ============================================================
# ANALYTICAL TYPE INFERENCE
# ============================================================

def infer_analytical_type(
    column_name: str,
    series: pd.Series,
) -> dict[str, Any]:
    """
    Infer the analytical meaning of a dataframe column.

    Important principle:

        pandas dtype
            !=
        statistical analytical type
            !=
        business meaning

    The inference is intentionally conservative.

    Strong deterministic evidence is handled here.
    Ambiguous semantic cases can later be reviewed by
    the DataLens semantic/AI layer.

    Returned types:
    - unknown
    - identifier
    - categorical
    - temporal
    - quantitative
    - text
    """

    non_null = (
        series
        .dropna()
    )


    non_null_count = int(
        len(
            non_null
        )
    )


    if non_null_count == 0:
        return {
            "type":
                "unknown",

            "subtype":
                None,

            "reason":
                (
                    "The column contains "
                    "no non-null values."
                ),

            "rule_version":
                ANALYTICAL_TYPE_RULE_VERSION,
        }


    unique_count = int(
        non_null
        .nunique()
    )


    unique_ratio = (
        unique_count
        /
        non_null_count
    )


    # ========================================================
    # BOOLEAN
    # ========================================================

    if is_bool_dtype(
        series
    ):
        return {
            "type":
                "categorical",

            "subtype":
                "boolean",

            "reason":
                (
                    "The pandas dtype is boolean, "
                    "so the column represents "
                    "categorical binary states "
                    "rather than a quantitative "
                    "measurement."
                ),

            "rule_version":
                ANALYTICAL_TYPE_RULE_VERSION,
        }


    # ========================================================
    # DATETIME
    # ========================================================

    if is_datetime64_any_dtype(
        series
    ):
        return {
            "type":
                "temporal",

            "subtype":
                "datetime",

            "reason":
                (
                    "The pandas dtype is datetime."
                ),

            "rule_version":
                ANALYTICAL_TYPE_RULE_VERSION,
        }


    # ========================================================
    # IDENTIFIER
    #
    # Important:
    # an identifier does NOT have to be unique.
    #
    # A foreign key such as client_id in a transaction
    # table is still an identifier even though the same
    # client appears many times.
    # ========================================================

    if has_strong_identifier_signal(
        column_name
    ):
        subtype = (
            "unique_key"
            if unique_ratio
            >=
            0.95

            else
            "reference"
        )


        return {
            "type":
                "identifier",

            "subtype":
                subtype,

            "reason":
                (
                    "The column name contains a "
                    "strong identifier signal. "
                    "Identifier semantics do not "
                    "require every value to be unique; "
                    "the column may represent a "
                    "primary key or a repeated "
                    "reference/foreign key."
                ),

            "rule_version":
                ANALYTICAL_TYPE_RULE_VERSION,
        }


    # ========================================================
    # NUMERIC SEMANTICS
    # ========================================================

    if is_numeric_dtype(
        series
    ):
        numeric_values = (
            numeric_non_null_values(
                series
            )
        )


        # ====================================================
        # YEAR / BIRTH YEAR
        # ====================================================

        if (
            has_year_signal(
                column_name
            )
            and
            values_are_plausible_years(
                numeric_values
            )
        ):
            subtype = (
                "birth_year"
                if has_birth_year_signal(
                    column_name
                )
                else
                "year"
            )


            return {
                "type":
                    "temporal",

                "subtype":
                    subtype,

                "reason":
                    (
                        "The column name indicates "
                        "calendar-year semantics and "
                        "all observed values are "
                        "integer years within a "
                        "plausible range."
                    ),

                "rule_version":
                    ANALYTICAL_TYPE_RULE_VERSION,
            }


        # ====================================================
        # NUMERIC CATEGORY CODE
        #
        # Low cardinality alone is deliberately NOT enough.
        #
        # Example:
        #   quantity = 0..5
        #
        # may still be a real quantitative variable.
        #
        # We therefore require both:
        # - a category-oriented name signal;
        # - discrete/low-cardinality observed values.
        # ====================================================

        category_cardinality_compatible = (
            unique_count
            <=
            MAX_NAMED_CATEGORY_LEVELS

            or

            unique_ratio
            <=
            MAX_NAMED_CATEGORY_RATIO
        )


        if (
            has_category_signal(
                column_name
            )
            and
            values_are_integer_like(
                numeric_values
            )
            and
            category_cardinality_compatible
        ):
            return {
                "type":
                    "categorical",

                "subtype":
                    "numeric_code",

                "reason":
                    (
                        "The column is physically "
                        "numeric, but its name "
                        "indicates categorical "
                        "semantics and the observed "
                        "values form a discrete "
                        "low-cardinality code set."
                    ),

                "rule_version":
                    ANALYTICAL_TYPE_RULE_VERSION,
            }


        # ====================================================
        # GENERIC CODE
        # ====================================================

        if (
            has_code_identifier_signal(
                column_name
            )
            and
            unique_ratio
            >=
            0.50
        ):
            return {
                "type":
                    "identifier",

                "subtype":
                    "code",

                "reason":
                    (
                        "The column name indicates "
                        "a code and the observed "
                        "values have sufficiently "
                        "high cardinality to treat "
                        "the column as an identifier "
                        "rather than a measurement."
                    ),

                "rule_version":
                    ANALYTICAL_TYPE_RULE_VERSION,
            }


        # ====================================================
        # QUANTITATIVE
        # ====================================================

        if (
            is_integer_dtype(
                series
            )
            or
            values_are_integer_like(
                numeric_values
            )
        ):
            return {
                "type":
                    "quantitative",

                "subtype":
                    "discrete",

                "reason":
                    (
                        "The column contains numeric "
                        "integer-like values and no "
                        "strong identifier, temporal "
                        "or categorical-code semantic "
                        "signal was detected."
                    ),

                "rule_version":
                    ANALYTICAL_TYPE_RULE_VERSION,
            }


        return {
            "type":
                "quantitative",

            "subtype":
                "continuous",

            "reason":
                (
                    "The column contains numeric "
                    "non-integer values and no "
                    "strong identifier, temporal "
                    "or categorical-code semantic "
                    "signal was detected."
                ),

            "rule_version":
                ANALYTICAL_TYPE_RULE_VERSION,
        }


    # ========================================================
    # NON-NUMERIC CODE
    # ========================================================

    if (
        has_code_identifier_signal(
            column_name
        )
        and
        unique_ratio
        >=
        0.50
    ):
        return {
            "type":
                "identifier",

            "subtype":
                "code",

            "reason":
                (
                    "The non-numeric column name "
                    "indicates a code and its "
                    "cardinality is consistent "
                    "with identifier-like values."
                ),

            "rule_version":
                ANALYTICAL_TYPE_RULE_VERSION,
        }


    # ========================================================
    # EXPLICIT CATEGORY NAME
    # ========================================================

    if has_category_signal(
        column_name
    ):
        return {
            "type":
                "categorical",

            "subtype":
                "nominal",

            "reason":
                (
                    "The column name contains a "
                    "strong categorical semantic "
                    "signal."
                ),

            "rule_version":
                ANALYTICAL_TYPE_RULE_VERSION,
        }


    # ========================================================
    # GENERIC LOW-CARDINALITY TEXT
    # ========================================================

    if (
        unique_count
        <=
        50
        or
        unique_ratio
        <=
        0.20
    ):
        return {
            "type":
                "categorical",

            "subtype":
                "nominal",

            "reason":
                (
                    "The non-numeric column has "
                    "relatively low cardinality "
                    "compared with the number of "
                    "observations."
                ),

            "rule_version":
                ANALYTICAL_TYPE_RULE_VERSION,
        }


    # ========================================================
    # TEXT
    # ========================================================

    return {
        "type":
            "text",

        "subtype":
            None,

        "reason":
            (
                "The column contains non-numeric "
                "values with relatively high "
                "cardinality and no stronger "
                "semantic type signal was detected."
            ),

        "rule_version":
            ANALYTICAL_TYPE_RULE_VERSION,
    }