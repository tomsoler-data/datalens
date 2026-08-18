from __future__ import annotations


from time import (
    perf_counter,
)


from app.planning.ai_analytical_planner import (
    AIPlannerAttemptTiming,
    AIPlannerProposal,
    AIPlannerReport,
    AIPlannerTiming,
    AIPlannerValidatedItem,
    DEFAULT_AI_PLANNER_MODEL,
    PlannerCatalog,
    build_contract_id,
    plan_analyses_with_ai,
    validate_ai_proposal,
)

from app.planning.generic_intent_resolver import (
    GENERIC_ANALYTICAL_INTENT_RULE_VERSION,
    GenericAnalyticalIntentResolution,
    GenericAnalyticalTarget,
    resolve_generic_analytical_intent,
)


# ============================================================
# VERSION
# ============================================================


INTENT_ROUTED_PLANNER_RULE_VERSION = (
    "intent_routed_planner_v0.1"
)


DETERMINISTIC_GENERIC_PLANNER_MODEL = (
    "python:"
    f"{GENERIC_ANALYTICAL_INTENT_RULE_VERSION}"
)


# ============================================================
# DETERMINISTIC OUTLIER PROPOSAL
# ============================================================


def _build_outlier_proposal(
    *,
    target: GenericAnalyticalTarget,
) -> AIPlannerProposal:
    """
    Translate a deterministic generic outlier target into the
    existing canonical planner wire format.

    Outlier detection currently reuses the distribution family.

    The statistical executor remains responsible for:
    - quartiles;
    - IQR;
    - outlier count;
    - outlier ratio;
    - visual evidence.

    No statistic is calculated by this planner.
    """

    return (
        AIPlannerProposal(
            decision=
                "propose",

            title=(
                "Détection des valeurs atypiques — "
                f"{target.column}"
            ),

            family=
                "distribution",

            dataset_id=
                target.dataset_id,

            analytical_grain=
                None,

            x_column=
                None,

            y_column=
                None,

            group_column=
                None,

            value_column=
                target.column,

            time_column=
                None,

            dimension_column=
                None,

            entity_column=
                None,

            aggregation_function=
                "none",

            ranking_order=
                "none",

            ranking_limit=
                None,

            window_operation=
                "none",

            window_size=
                None,

            blockers=
                [],

            reasons=[
                (
                    "La demande générique de détection "
                    "d'outliers a été reconnue "
                    "déterministiquement par Python."
                ),

                (
                    "La colonne cible a été sélectionnée "
                    "exclusivement depuis le catalogue "
                    "réel des données."
                ),

                (
                    "L'analyse utilise la famille "
                    "`distribution`, dont l'exécution "
                    "statistique reste déterministe."
                ),
            ],

            confidence=
                1.0,
        )
    )


# ============================================================
# DETERMINISTIC BLOCKED PROPOSAL
# ============================================================


def _build_blocked_proposal(
    *,
    resolution: GenericAnalyticalIntentResolution,
) -> AIPlannerProposal:
    blockers = [
        blocker.strip()

        for blocker
        in resolution.blockers

        if blocker.strip()
    ]


    if not blockers:
        blockers = [
            (
                "La demande générique a été reconnue, "
                "mais aucun périmètre analytique "
                "déterministe exécutable n'a pu être "
                "résolu."
            )
        ]


    reasons = [
        reason.strip()

        for reason
        in resolution.reasons

        if reason.strip()
    ]


    if not reasons:
        reasons = [
            (
                "Python a choisi l'abstention plutôt "
                "que d'inventer une variable ou "
                "un dataset."
            )
        ]


    return (
        AIPlannerProposal(
            decision=
                "blocked",

            title=
                "Détection des valeurs atypiques",

            family=
                "unresolved",

            dataset_id=
                None,

            analytical_grain=
                None,

            x_column=
                None,

            y_column=
                None,

            group_column=
                None,

            value_column=
                None,

            time_column=
                None,

            dimension_column=
                None,

            entity_column=
                None,

            aggregation_function=
                "none",

            ranking_order=
                "none",

            ranking_limit=
                None,

            window_operation=
                "none",

            window_size=
                None,

            blockers=
                blockers,

            reasons=
                reasons,

            confidence=
                1.0,
        )
    )


# ============================================================
# VALIDATION OBJECTIVE
# ============================================================


def _validation_objective(
    *,
    objective: str,

    proposal: AIPlannerProposal,
) -> str:
    """
    validate_ai_proposal() correctly protects against an LLM
    choosing an arbitrary dataset when multiple datasets are
    compatible.

    A generic deterministic expansion is different:

        objective
            ↓
        generic intent resolver
            ↓
        exact dataset + exact column from Python catalog

    The dataset has therefore already been selected by Python,
    not guessed by the LLM.

    We make that deterministic dataset scope explicit during
    validation so the existing safety validator can still be
    reused unchanged.

    The original user objective is restored on the final
    canonical contract afterwards.
    """

    if (
        proposal.dataset_id
        is None
    ):
        return (
            objective
        )


    return (
        f"{objective}\n\n"
        "PÉRIMÈTRE DATASET DÉTERMINISTE : "
        f"{proposal.dataset_id}"
    )


# ============================================================
# RESTORE USER REQUEST
# ============================================================


def _restore_original_contract_context(
    *,
    objective: str,

    item: AIPlannerValidatedItem,
) -> AIPlannerValidatedItem:
    contract = (
        item.contract
    )


    if (
        contract
        is None
    ):
        return (
            item
        )


    restored_contract_id = (
        build_contract_id(
            objective=
                objective,

            proposal=
                item.proposal,

            proposal_index=
                item.proposal_index,
        )
    )


    restored_contract = (
        contract.model_copy(
            update={
                "contract_id":
                    restored_contract_id,

                "request_text":
                    objective,
            }
        )
    )


    return (
        item.model_copy(
            update={
                "contract":
                    restored_contract,
            }
        )
    )


# ============================================================
# VALIDATE DETERMINISTIC PROPOSAL
# ============================================================


def _validate_deterministic_proposal(
    *,
    objective: str,

    proposal: AIPlannerProposal,

    proposal_index: int,

    catalog: PlannerCatalog,
) -> AIPlannerValidatedItem:
    validation_objective = (
        _validation_objective(
            objective=
                objective,

            proposal=
                proposal,
        )
    )


    item = (
        validate_ai_proposal(
            objective=
                validation_objective,

            proposal=
                proposal,

            proposal_index=
                proposal_index,

            catalog=
                catalog,
        )
    )


    return (
        _restore_original_contract_context(
            objective=
                objective,

            item=
                item,
        )
    )


# ============================================================
# BUILD DETERMINISTIC REPORT
# ============================================================


def _build_deterministic_report(
    *,
    objective: str,

    resolution: GenericAnalyticalIntentResolution,

    catalog: PlannerCatalog,

    started_at: float,
) -> AIPlannerReport:
    # --------------------------------------------------------
    # TRANSLATE RESOLUTION INTO PROPOSALS
    # --------------------------------------------------------

    if (
        resolution.status
        ==
        "blocked"
    ):
        proposals = [
            _build_blocked_proposal(
                resolution=
                    resolution
            )
        ]


    elif (
        resolution.intent
        ==
        "outlier_detection"
    ):
        proposals = [
            _build_outlier_proposal(
                target=
                    target
            )

            for target
            in resolution.targets
        ]


    else:
        fallback_resolution = (
            GenericAnalyticalIntentResolution(
                status=
                    "blocked",

                matched=
                    True,

                intent=
                    resolution.intent,

                objective=
                    objective,

                target_count=
                    0,

                targets=
                    [],

                reasons=[
                    (
                        "L'intention générique a été "
                        "reconnue mais ne possède pas "
                        "encore de traduction analytique "
                        "déterministe."
                    )
                ],

                blockers=[
                    (
                        "Cette intention générique "
                        "n'est pas encore supportée "
                        "par le routeur analytique."
                    )
                ],
            )
        )


        proposals = [
            _build_blocked_proposal(
                resolution=
                    fallback_resolution
            )
        ]


    # --------------------------------------------------------
    # PYTHON VALIDATION
    # --------------------------------------------------------

    validation_started_at = (
        perf_counter()
    )


    items = [
        _validate_deterministic_proposal(
            objective=
                objective,

            proposal=
                proposal,

            proposal_index=
                proposal_index,

            catalog=
                catalog,
        )

        for (
            proposal_index,
            proposal,
        )
        in enumerate(
            proposals,
            start=1,
        )
    ]


    python_validation_ms = (
        (
            perf_counter()
            -
            validation_started_at
        )
        *
        1000.0
    )


    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    validated_count = sum(
        1

        for item
        in items

        if (
            item.validation_status
            ==
            "validated"
        )
    )


    blocked_count = sum(
        1

        for item
        in items

        if (
            item.validation_status
            ==
            "blocked"
        )
    )


    ambiguous_count = sum(
        1

        for item
        in items

        if (
            item.validation_status
            ==
            "ambiguous"
        )
    )


    rejected_count = sum(
        1

        for item
        in items

        if (
            item.validation_status
            ==
            "rejected"
        )
    )


    normalization_count = sum(
        len(
            item.normalizations
        )

        for item
        in items
    )


    # --------------------------------------------------------
    # TIMING
    # --------------------------------------------------------

    total_ms = (
        (
            perf_counter()
            -
            started_at
        )
        *
        1000.0
    )


    attempt = (
        AIPlannerAttemptTiming(
            attempt_index=
                1,

            prompt_construction_ms=
                0.0,

            model_inference_ms=
                0.0,

            structured_parse_ms=
                0.0,

            python_validation_ms=
                python_validation_ms,

            total_ms=
                total_ms,
        )
    )


    timing = (
        AIPlannerTiming(
            prompt_construction_ms=
                0.0,

            model_inference_ms=
                0.0,

            structured_parse_ms=
                0.0,

            python_validation_ms=
                python_validation_ms,

            retry_feedback_ms=
                0.0,

            total_ms=
                total_ms,

            attempts=[
                attempt
            ],
        )
    )


    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    return (
        AIPlannerReport(
            objective=
                objective,

            model=
                DETERMINISTIC_GENERIC_PLANNER_MODEL,

            proposal_count=
                len(
                    items
                ),

            validated_count=
                validated_count,

            blocked_count=
                blocked_count,

            ambiguous_count=
                ambiguous_count,

            rejected_count=
                rejected_count,

            items=
                items,

            attempt_count=
                1,

            retry_count=
                0,

            retry_triggered=
                False,

            retry_feedback=
                [],

            normalization_count=
                normalization_count,

            normalization_applied=(
                normalization_count
                >
                0
            ),

            timing=
                timing,

            planner_rule_version=
                INTENT_ROUTED_PLANNER_RULE_VERSION,
        )
    )


# ============================================================
# PUBLIC ROUTED PLANNER
# ============================================================


def plan_analyses_with_intent_routing(
    *,
    objective: str,

    catalog: PlannerCatalog,

    model: str = (
        DEFAULT_AI_PLANNER_MODEL
    ),
) -> AIPlannerReport:
    """
    Route an analytical request between:

    1. deterministic generic intent expansion;
    2. existing local AI planner.

    Current deterministic generic support:

        outlier_detection

    Generic outlier requests such as:

        "Détecte les outliers."

    are expanded against the real Python-owned catalog.

    Precise requests such as:

        "Détecte les outliers de annual_salary."

    are intentionally left to the existing semantic planner,
    whose proposal is still validated by Python.

    Unsupported requests also continue through the existing
    Gemma -> Python validation path.
    """

    started_at = (
        perf_counter()
    )


    normalized_objective = (
        objective
        .strip()
    )


    if not (
        normalized_objective
    ):
        raise ValueError(
            (
                "L'objectif utilisateur "
                "ne peut pas être vide."
            )
        )


    resolution = (
        resolve_generic_analytical_intent(
            objective=
                normalized_objective,

            catalog=
                catalog,
        )
    )


    # ========================================================
    # DETERMINISTIC GENERIC PATH
    # ========================================================

    if (
        resolution.matched
    ):
        return (
            _build_deterministic_report(
                objective=
                    normalized_objective,

                resolution=
                    resolution,

                catalog=
                    catalog,

                started_at=
                    started_at,
            )
        )


    # ========================================================
    # AI FALLBACK
    # ========================================================

    return (
        plan_analyses_with_ai(
            objective=
                normalized_objective,

            catalog=
                catalog,

            model=
                model,
        )
    )