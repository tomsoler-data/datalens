from __future__ import annotations

from app.semantics.comparator import (
    compare_semantic_profiles,
    profiles_are_directly_subtractable,
)

from app.semantics.embedding_retrieval_schemas import (
    SemanticEmbeddingCandidatePair,
)

from app.semantics.family import (
    build_state_abstracted_signature,
    profiles_have_dimension_conflict,
    profiles_have_distinct_known_states,
    quantity_family_assignment_by_column,
    reconcile_quantity_family_pair,
)

from app.semantics.family_schemas import (
    DatasetQuantityFamilyReport,
    QuantityFamilyRelationDecision,
)

from app.semantics.schemas import (
    ColumnSemanticProfile,
)

from app.semantics.relation_evidence_schemas import (
    METRIC_RELATION_EVIDENCE_VERSION,
    MetricRelationComparatorEvidence,
    MetricRelationEmbeddingEvidence,
    MetricRelationEvidence,
    MetricRelationFamilyEvidence,
    MetricRelationInterpretation,
    MetricRelationProfileIdentity,
    MetricRelationQuantityEvidence,
)


# ============================================================
# CONSTANTS
# ============================================================

UNKNOWN_VALUES = {
    "",
    "unknown",
}


# ============================================================
# PROFILE HELPERS
# ============================================================

def _known(
    value: str,
) -> bool:
    return (
        value
        not in
        UNKNOWN_VALUES
    )


def _profile_identity(
    profile: ColumnSemanticProfile,
) -> MetricRelationProfileIdentity:
    return (
        MetricRelationProfileIdentity(
            dataset_id=
                profile.dataset_id,

            column=
                profile.column,

            concept=
                profile.concept,

            semantic_group=
                profile.semantic_group,

            domain=
                profile.domain,

            variant=
                profile.variant,

            measure_kind=
                profile.measure_kind,

            unit_kind=
                profile.unit_kind,

            quantity_dimension=
                profile.quantity_dimension,

            quantity_unit=
                profile.quantity_unit,

            entity_role=
                profile.entity_role,

            profile_confidence=
                profile.confidence,

            profile_source=
                profile.source,
        )
    )


# ============================================================
# QUANTITY EVIDENCE
# ============================================================

def _build_quantity_evidence(
    *,
    left: ColumnSemanticProfile,
    right: ColumnSemanticProfile,
    compatible_units: bool,
) -> MetricRelationQuantityEvidence:
    left_dimension = (
        left.quantity_dimension
    )


    right_dimension = (
        right.quantity_dimension
    )


    same_known_dimension = (
        _known(
            left_dimension
        )

        and

        _known(
            right_dimension
        )

        and

        left_dimension
        ==
        right_dimension
    )


    compatible_dimensions = (
        same_known_dimension
    )


    if (
        not _known(
            left_dimension
        )
        or
        not _known(
            right_dimension
        )
    ):
        compatible_dimensions = (
            False
        )


    exact_same_known_unit = (
        _known(
            left.quantity_unit
        )

        and

        _known(
            right.quantity_unit
        )

        and

        left.quantity_unit
        ==
        right.quantity_unit
    )


    dimension_conflict = (
        profiles_have_dimension_conflict(
            left=
                left,

            right=
                right,
        )
    )


    directly_subtractable = (
        profiles_are_directly_subtractable(
            left,
            right,
        )
    )


    return (
        MetricRelationQuantityEvidence(
            same_known_quantity_dimension=
                same_known_dimension,

            compatible_quantity_dimensions=
                compatible_dimensions,

            exact_same_known_unit=
                exact_same_known_unit,

            compatible_units=
                compatible_units,

            directly_subtractable=
                directly_subtractable,

            dimension_conflict=
                dimension_conflict,

            quantity_field_provenance_available=
                False,
        )
    )


# ============================================================
# FAMILY EVIDENCE
# ============================================================

def _empty_family_evidence(
    *,
    left: ColumnSemanticProfile,
    right: ColumnSemanticProfile,
) -> MetricRelationFamilyEvidence:
    left_signature = (
        build_state_abstracted_signature(
            column=
                left.column,

            variant=
                left.variant,
        )
    )


    right_signature = (
        build_state_abstracted_signature(
            column=
                right.column,

            variant=
                right.variant,
        )
    )


    return (
        MetricRelationFamilyEvidence(
            available=
                False,

            same_quantity_family=
                None,

            relation_source=
                None,

            left_family=
                None,

            right_family=
                None,

            left_family_source=
                None,

            right_family_source=
                None,

            left_family_confidence=
                None,

            right_family_confidence=
                None,

            left_signature=
                left_signature,

            right_signature=
                right_signature,

            signature_same=(
                left_signature
                ==
                right_signature
            ),

            left_state=
                left.variant,

            right_state=
                right.variant,

            distinct_known_states=
                profiles_have_distinct_known_states(
                    left=
                        left,

                    right=
                        right,
                ),

            llm_same_family=
                None,

            reasons=[
                (
                    "Aucun rapport de famille quantitative "
                    "compatible n'a été fourni pour cette paire."
                )
            ],
        )
    )


def _build_family_evidence(
    *,
    left: ColumnSemanticProfile,
    right: ColumnSemanticProfile,
    family_report: DatasetQuantityFamilyReport | None,
) -> MetricRelationFamilyEvidence:
    if (
        family_report
        is None
    ):
        return (
            _empty_family_evidence(
                left=
                    left,

                right=
                    right,
            )
        )


    if (
        left.dataset_id
        !=
        right.dataset_id
        or
        family_report.dataset_id
        !=
        left.dataset_id
    ):
        return (
            _empty_family_evidence(
                left=
                    left,

                right=
                    right,
            )
        )


    decision: QuantityFamilyRelationDecision = (
        reconcile_quantity_family_pair(
            left=
                left,

            right=
                right,

            report=
                family_report,
        )
    )


    left_assignment = (
        quantity_family_assignment_by_column(
            report=
                family_report,

            column=
                left.column,
        )
    )


    right_assignment = (
        quantity_family_assignment_by_column(
            report=
                family_report,

            column=
                right.column,
        )
    )


    return (
        MetricRelationFamilyEvidence(
            available=
                True,

            same_quantity_family=
                decision.same_quantity_family,

            relation_source=
                decision.source,

            left_family=
                decision.left_family,

            right_family=
                decision.right_family,

            left_family_source=(
                left_assignment.source
                if (
                    left_assignment
                    is not None
                )
                else None
            ),

            right_family_source=(
                right_assignment.source
                if (
                    right_assignment
                    is not None
                )
                else None
            ),

            left_family_confidence=(
                left_assignment.confidence
                if (
                    left_assignment
                    is not None
                )
                else None
            ),

            right_family_confidence=(
                right_assignment.confidence
                if (
                    right_assignment
                    is not None
                )
                else None
            ),

            left_signature=
                decision.left_signature,

            right_signature=
                decision.right_signature,

            signature_same=
                decision.signature_same,

            left_state=
                decision.left_state,

            right_state=
                decision.right_state,

            distinct_known_states=
                decision.distinct_known_states,

            llm_same_family=
                decision.llm_same_family,

            reasons=
                list(
                    decision.reasons
                ),
        )
    )


# ============================================================
# EMBEDDING EVIDENCE
# ============================================================

def _build_embedding_evidence(
    *,
    left: ColumnSemanticProfile,
    right: ColumnSemanticProfile,
    embedding_pair: SemanticEmbeddingCandidatePair | None,
) -> MetricRelationEmbeddingEvidence:
    if (
        embedding_pair
        is None
    ):
        return (
            MetricRelationEmbeddingEvidence(
                candidate_retrieved=
                    False,
            )
        )


    pair_columns = {
        embedding_pair.left_column,
        embedding_pair.right_column,
    }


    expected_columns = {
        left.column,
        right.column,
    }


    if (
        pair_columns
        !=
        expected_columns
    ):
        return (
            MetricRelationEmbeddingEvidence(
                candidate_retrieved=
                    False,
            )
        )


    if (
        embedding_pair.left_column
        ==
        left.column
        and
        embedding_pair.right_column
        ==
        right.column
    ):
        left_to_right_rank = (
            embedding_pair
            .left_to_right_rank
        )


        right_to_left_rank = (
            embedding_pair
            .right_to_left_rank
        )

    else:
        left_to_right_rank = (
            embedding_pair
            .right_to_left_rank
        )


        right_to_left_rank = (
            embedding_pair
            .left_to_right_rank
        )


    return (
        MetricRelationEmbeddingEvidence(
            candidate_retrieved=
                True,

            cosine_similarity=
                embedding_pair
                .cosine_similarity,

            left_to_right_rank=
                left_to_right_rank,

            right_to_left_rank=
                right_to_left_rank,

            mutual_retrieval=
                embedding_pair
                .mutual_retrieval,
        )
    )


# ============================================================
# INTERPRETATION
#
# v0.2 remains deliberately asymmetric.
#
# Positive classification is still limited to:
#
#     same_metric_different_state
#
# The important change from v0.1:
#
# we no longer require an exact state-abstracted lexical
# signature.
#
# A positive relation instead requires:
#
# 1. no known dimension conflict;
# 2. a positive quantity-family reconciliation;
# 3. at least one independent corroborating signal:
#
#    - same known quantity dimension;
#    - same concept family;
#    - distinct known states.
#
# Same-family clustering ALONE is not sufficient.
# ============================================================

def _interpret_relation(
    *,
    left: ColumnSemanticProfile,
    right: ColumnSemanticProfile,
    comparison,
    quantity: MetricRelationQuantityEvidence,
    family: MetricRelationFamilyEvidence,
    embedding: MetricRelationEmbeddingEvidence,
) -> MetricRelationInterpretation:
    same_metric_evidence: list[
        str
    ] = []


    process_stage_evidence: list[
        str
    ] = []


    related_metric_evidence: list[
        str
    ] = []


    contradictions: list[
        str
    ] = []


    limitations: list[
        str
    ] = []


    # ========================================================
    # SAME-METRIC EVIDENCE
    # ========================================================

    if (
        comparison.same_concept
    ):
        same_metric_evidence.append(
            "Les profils partagent le même concept."
        )


    if (
        comparison.same_concept_family
    ):
        same_metric_evidence.append(
            (
                "Les profils appartiennent à la même "
                "famille conceptuelle."
            )
        )


    if (
        comparison.distinct_variants
    ):
        same_metric_evidence.append(
            (
                "Les profils portent des variantes "
                "sémantiques distinctes."
            )
        )


    if (
        family.signature_same
    ):
        same_metric_evidence.append(
            (
                "Les signatures après abstraction de "
                "l'état sont identiques."
            )
        )


    if (
        family.distinct_known_states
    ):
        same_metric_evidence.append(
            (
                "Deux états distincts et connus sont "
                "présents."
            )
        )


    if (
        family.same_quantity_family
        is True
    ):
        same_metric_evidence.append(
            (
                "Le réconciliateur de familles quantitatives "
                "a établi une même famille."
            )
        )


    if (
        quantity.same_known_quantity_dimension
    ):
        same_metric_evidence.append(
            (
                "Les deux profils ont la même dimension "
                "quantitative connue."
            )
        )


    if (
        quantity.exact_same_known_unit
    ):
        same_metric_evidence.append(
            (
                "Les deux profils ont exactement la même "
                "unité quantitative connue."
            )
        )


    # ========================================================
    # RELATED-METRIC EVIDENCE
    #
    # Diagnostic only in v0.2.
    # ========================================================

    if (
        comparison.same_domain
        and
        not comparison.same_concept_family
    ):
        related_metric_evidence.append(
            (
                "Les profils partagent un domaine analytique "
                "sans partager de famille conceptuelle connue."
            )
        )


    if (
        embedding.candidate_retrieved
    ):
        related_metric_evidence.append(
            (
                "Le retriever d'embeddings a retenu la paire "
                "pour examen."
            )
        )


    # ========================================================
    # CONTRADICTIONS
    # ========================================================

    if (
        quantity.dimension_conflict
    ):
        contradictions.append(
            (
                "Les dimensions quantitatives connues sont "
                "différentes."
            )
        )


    if (
        family.available
        and
        family.same_quantity_family
        is False
    ):
        contradictions.append(
            (
                "Le réconciliateur n'a pas établi de même "
                "famille quantitative."
            )
        )


    if (
        family.relation_source
        ==
        "dimension_veto"
    ):
        contradictions.append(
            (
                "Un veto de dimension a été appliqué par "
                "le réconciliateur."
            )
        )


    # ========================================================
    # PROVENANCE LIMITATION
    # ========================================================

    if (
        not quantity
        .quantity_field_provenance_available
    ):
        limitations.append(
            (
                "La provenance et la confiance ne sont pas "
                "encore suivies séparément pour chaque champ "
                "quantity_dimension / quantity_unit."
            )
        )


    # ========================================================
    # v0.2 CORROBORATION
    # ========================================================

    independent_corroboration = (
        quantity.same_known_quantity_dimension

        or

        comparison.same_concept_family

        or

        family.distinct_known_states
    )


    same_metric_supported = (
        not quantity.dimension_conflict

        and

        family.available

        and

        family.same_quantity_family
        is True

        and

        independent_corroboration
    )


    if (
        same_metric_supported
    ):
        proposed_relation = (
            "same_metric_different_state"
        )


        confidence = (
            "medium"
        )


        limitations.append(
            (
                "La relation proposée repose sur une "
                "famille quantitative positive accompagnée "
                "d'au moins une corroboration indépendante."
            )
        )


        limitations.append(
            (
                "Cette relation sémantique n'autorise aucune "
                "opération analytique par elle-même."
            )
        )


    else:
        proposed_relation = (
            "uncertain"
        )


        confidence = (
            "low"
        )


        if (
            family.same_quantity_family
            is True
            and
            not independent_corroboration
        ):
            limitations.append(
                (
                    "Le clustering de famille quantitative "
                    "est positif, mais aucune corroboration "
                    "indépendante suffisante n'est disponible."
                )
            )


        limitations.append(
            (
                "Les preuves disponibles sont insuffisantes "
                "pour attribuer une relation plus précise "
                "sans augmenter le risque de faux positif."
            )
        )


    return (
        MetricRelationInterpretation(
            proposed_relation=
                proposed_relation,

            confidence=
                confidence,

            evidence_for_same_metric_different_state=
                same_metric_evidence,

            evidence_for_same_process_different_stage=
                process_stage_evidence,

            evidence_for_related_distinct_metric=
                related_metric_evidence,

            contradictory_evidence=
                contradictions,

            limitations=
                limitations,
        )
    )


# ============================================================
# PUBLIC BUILDER
# ============================================================

def build_metric_relation_evidence(
    *,
    left: ColumnSemanticProfile,
    right: ColumnSemanticProfile,
    family_report: DatasetQuantityFamilyReport | None = None,
    embedding_pair: SemanticEmbeddingCandidatePair | None = None,
) -> MetricRelationEvidence:
    comparison = (
        compare_semantic_profiles(
            left,
            right,
        )
    )


    quantity = (
        _build_quantity_evidence(
            left=
                left,

            right=
                right,

            compatible_units=
                comparison.compatible_units,
        )
    )


    family = (
        _build_family_evidence(
            left=
                left,

            right=
                right,

            family_report=
                family_report,
        )
    )


    embedding = (
        _build_embedding_evidence(
            left=
                left,

            right=
                right,

            embedding_pair=
                embedding_pair,
        )
    )


    comparator = (
        MetricRelationComparatorEvidence(
            same_concept=
                comparison.same_concept,

            same_concept_family=
                comparison.same_concept_family,

            same_domain=
                comparison.same_domain,

            distinct_variants=
                comparison.distinct_variants,

            conceptual_proximity=
                comparison.conceptual_proximity,

            association_novelty=
                comparison.association_novelty,

            redundancy_risk=
                comparison.redundancy_risk,

            derived_gap_compatible=
                comparison.derived_gap_compatible,

            reasons=
                list(
                    comparison.reasons
                ),
        )
    )


    interpretation = (
        _interpret_relation(
            left=
                left,

            right=
                right,

            comparison=
                comparison,

            quantity=
                quantity,

            family=
                family,

            embedding=
                embedding,
        )
    )


    return (
        MetricRelationEvidence(
            left=
                _profile_identity(
                    left
                ),

            right=
                _profile_identity(
                    right
                ),

            quantity=
                quantity,

            comparator=
                comparator,

            family=
                family,

            embedding=
                embedding,

            interpretation=
                interpretation,

            relation_evidence_version=
                METRIC_RELATION_EVIDENCE_VERSION,
        )
    )
