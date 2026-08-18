from __future__ import annotations

from app.semantics.quantity import (
    dimensions_are_compatible,
    is_numeric_quantity_dimension,
    units_are_directly_comparable,
)

from app.semantics.schemas import (
    ColumnSemanticProfile,
    SemanticProfileComparison,
)


# ============================================================
# HELPERS
# ============================================================

UNKNOWN_VALUES = {
    "",
    "unknown",
}


NUMERIC_MEASURE_KINDS = {
    "count",
    "rate",
    "percentage",
    "index",
    "currency",
    "duration",
}


def meaningful_equal(
    left: str,
    right: str,
) -> bool:
    if (
        left
        in UNKNOWN_VALUES
        or
        right
        in UNKNOWN_VALUES
    ):
        return False


    return (
        left
        ==
        right
    )


def meaningful_variant(
    value: str,
) -> bool:
    return (
        value
        not in UNKNOWN_VALUES
    )


# ============================================================
# LEGACY UNIT COMPATIBILITY
# ============================================================

def legacy_units_are_compatible(
    left: str,
    right: str,
) -> bool:
    if (
        left
        in UNKNOWN_VALUES
        or
        right
        in UNKNOWN_VALUES
    ):
        return False


    if (
        left
        ==
        right
    ):
        return True


    proportional_units = {
        "percent",
        "proportion",
    }


    if (
        left
        in proportional_units
        and
        right
        in proportional_units
    ):
        return True


    return False


# ============================================================
# QUANTITY COMPATIBILITY
# ============================================================

def profiles_have_compatible_units(
    left: ColumnSemanticProfile,
    right: ColumnSemanticProfile,
) -> bool:
    left_dimension = (
        left.quantity_dimension
    )


    right_dimension = (
        right.quantity_dimension
    )


    # Prefer explicit quantity dimensions when both profiles
    # provide them.
    if (
        left_dimension
        not in UNKNOWN_VALUES
        and
        right_dimension
        not in UNKNOWN_VALUES
    ):
        return (
            dimensions_are_compatible(
                left_dimension,
                right_dimension,
            )
        )


    # Backward-compatible fallback.
    return (
        legacy_units_are_compatible(
            left.unit_kind,
            right.unit_kind,
        )
    )


def profiles_are_directly_subtractable(
    left: ColumnSemanticProfile,
    right: ColumnSemanticProfile,
) -> bool:
    left_dimension = (
        left.quantity_dimension
    )


    right_dimension = (
        right.quantity_dimension
    )


    if (
        left_dimension
        not in UNKNOWN_VALUES
        and
        right_dimension
        not in UNKNOWN_VALUES
    ):
        # Same dimension is necessary.
        if not dimensions_are_compatible(
            left_dimension,
            right_dimension,
        ):
            return False


        # Exact quantity units are directly subtractable only
        # when they already use the same unit.
        if (
            left.quantity_unit
            not in UNKNOWN_VALUES
            and
            right.quantity_unit
            not in UNKNOWN_VALUES
        ):
            return (
                units_are_directly_comparable(
                    left_dimension=
                        left_dimension,

                    left_unit=
                        left.quantity_unit,

                    right_dimension=
                        right_dimension,

                    right_unit=
                        right.quantity_unit,
                )
            )


        # If the precise unit is not available, preserve the
        # old conservative unit-kind rule.
        return (
            legacy_units_are_compatible(
                left.unit_kind,
                right.unit_kind,
            )
        )


    return (
        legacy_units_are_compatible(
            left.unit_kind,
            right.unit_kind,
        )
    )


def profile_is_numeric(
    profile: ColumnSemanticProfile,
) -> bool:
    if (
        profile.measure_kind
        in NUMERIC_MEASURE_KINDS
    ):
        return True


    return (
        is_numeric_quantity_dimension(
            profile.quantity_dimension
        )
    )


# ============================================================
# PUBLIC COMPARATOR
# ============================================================

def compare_semantic_profiles(
    left: ColumnSemanticProfile,
    right: ColumnSemanticProfile,
) -> SemanticProfileComparison:
    same_concept = meaningful_equal(
        left.concept,
        right.concept,
    )


    same_group = meaningful_equal(
        left.semantic_group,
        right.semantic_group,
    )


    same_domain = meaningful_equal(
        left.domain,
        right.domain,
    )


    same_concept_family = (
        same_concept
        or
        same_group
    )


    distinct_variants = (
        meaningful_variant(
            left.variant
        )
        and
        meaningful_variant(
            right.variant
        )
        and
        left.variant
        !=
        right.variant
    )


    compatible_units = (
        profiles_have_compatible_units(
            left,
            right,
        )
    )


    # ========================================================
    # CONCEPTUAL PROXIMITY
    # ========================================================

    if same_concept_family:
        conceptual_proximity = (
            "high"
        )

    elif same_domain:
        conceptual_proximity = (
            "medium"
        )

    else:
        conceptual_proximity = (
            "low"
        )


    # ========================================================
    # ASSOCIATION NOVELTY
    # ========================================================

    if (
        conceptual_proximity
        ==
        "high"
    ):
        association_novelty = (
            "low"
        )

    elif (
        conceptual_proximity
        ==
        "medium"
    ):
        association_novelty = (
            "medium"
        )

    else:
        association_novelty = (
            "high"
        )


    # ========================================================
    # REDUNDANCY
    # ========================================================

    if same_concept:
        redundancy_risk = (
            "high"
        )

    elif same_group:
        redundancy_risk = (
            "high"
        )

    elif same_domain:
        redundancy_risk = (
            "medium"
        )

    else:
        redundancy_risk = (
            "low"
        )


    # ========================================================
    # DERIVED GAP COMPATIBILITY
    #
    # A gap must be:
    #
    # 1. numeric,
    # 2. conceptually related,
    # 3. conceptually distinct,
    # 4. dimensionally compatible,
    # 5. already expressed in directly comparable units.
    #
    # Different but convertible units will be handled by a
    # future conversion engine.
    # ========================================================

    left_numeric = (
        profile_is_numeric(
            left
        )
    )


    right_numeric = (
        profile_is_numeric(
            right
        )
    )


    conceptually_distinct = (
        distinct_variants
        or
        (
            same_group
            and
            not same_concept
        )
    )


    directly_subtractable = (
        profiles_are_directly_subtractable(
            left,
            right,
        )
    )


    derived_gap_compatible = (
        left_numeric
        and
        right_numeric
        and
        same_concept_family
        and
        conceptually_distinct
        and
        compatible_units
        and
        directly_subtractable
    )


    # ========================================================
    # EXPLANATIONS
    # ========================================================

    reasons: list[
        str
    ] = []


    if same_concept:
        reasons.append(
            (
                "Les deux colonnes représentent "
                "le même concept sémantique."
            )
        )

    elif same_group:
        reasons.append(
            (
                "Les deux colonnes appartiennent "
                "à la même famille conceptuelle."
            )
        )


    if distinct_variants:
        reasons.append(
            (
                "Les colonnes représentent des variantes "
                "différentes du phénomène."
            )
        )


    if same_domain:
        reasons.append(
            (
                "Les deux colonnes appartiennent "
                "au même domaine analytique."
            )
        )

    else:
        reasons.append(
            (
                "Les colonnes appartiennent à des "
                "domaines analytiques distincts."
            )
        )


    if (
        left.quantity_dimension
        not in UNKNOWN_VALUES
        and
        right.quantity_dimension
        not in UNKNOWN_VALUES
        and
        dimensions_are_compatible(
            left.quantity_dimension,
            right.quantity_dimension,
        )
    ):
        reasons.append(
            (
                "Les deux mesures appartiennent à la "
                "même dimension quantitative "
                f"({left.quantity_dimension})."
            )
        )


    if (
        compatible_units
        and
        not directly_subtractable
    ):
        reasons.append(
            (
                "Les dimensions quantitatives sont "
                "compatibles, mais une conversion d'unité "
                "est requise avant une soustraction."
            )
        )


    if (
        association_novelty
        ==
        "high"
    ):
        reasons.append(
            (
                "Une association entre ces mesures "
                "peut apporter un angle conceptuel "
                "distinct."
            )
        )


    if (
        redundancy_risk
        ==
        "high"
    ):
        reasons.append(
            (
                "Une association directe peut être "
                "partiellement redondante car les "
                "mesures décrivent un phénomène proche."
            )
        )


    if derived_gap_compatible:
        reasons.append(
            (
                "Les mesures sont conceptuellement "
                "liées, distinctes et directement "
                "comparables pour examiner un écart "
                "déterministe."
            )
        )


    return SemanticProfileComparison(
        left_dataset_id=
            left.dataset_id,

        left_column=
            left.column,

        left_variant=
            left.variant,

        left_quantity_dimension=
            left.quantity_dimension,

        left_quantity_unit=
            left.quantity_unit,

        right_dataset_id=
            right.dataset_id,

        right_column=
            right.column,

        right_variant=
            right.variant,

        right_quantity_dimension=
            right.quantity_dimension,

        right_quantity_unit=
            right.quantity_unit,

        same_concept=
            same_concept,

        same_concept_family=
            same_concept_family,

        same_domain=
            same_domain,

        compatible_units=
            compatible_units,

        distinct_variants=
            distinct_variants,

        conceptual_proximity=
            conceptual_proximity,

        association_novelty=
            association_novelty,

        redundancy_risk=
            redundancy_risk,

        derived_gap_compatible=
            derived_gap_compatible,

        reasons=
            reasons,
    )
