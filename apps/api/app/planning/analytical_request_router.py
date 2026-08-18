from __future__ import annotations


from typing import (
    Any,
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
)


from app.analysis.entity_outlier_requests import (
    EntityOutlierRequestReport,
    resolve_entity_outlier_intent,
    run_entity_outlier_request,
)

from app.planning.ai_analytical_planner import (
    AIPlannerReport,
    DEFAULT_AI_PLANNER_MODEL,
    PlannerCatalog,
)

from app.planning.intent_routed_planner import (
    plan_analyses_with_intent_routing,
)


# ============================================================
# VERSION
# ============================================================


ANALYTICAL_REQUEST_ROUTER_RULE_VERSION = (
    "analytical_request_router_v0.1"
)


# ============================================================
# ROUTE TYPES
# ============================================================


AnalyticalRequestRouteKind = Literal[
    "entity_outlier",
    "analytical_plan",
]


# ============================================================
# ROUTING REPORT
# ============================================================


class AnalyticalRequestRoutingReport(
    BaseModel
):
    """
    Canonical top-level routing result.

    Exactly one execution branch must be populated:

        route_kind == "entity_outlier"
            -> entity_outlier_report

        route_kind == "analytical_plan"
            -> planner_report

    This model deliberately keeps the two result contracts
    separate.

    Entity-level analyses are not forced into AIPlannerReport,
    because some entity requests already produce a complete
    deterministic analytical result rather than merely a plan.
    """

    model_config = ConfigDict(
        extra="forbid"
    )


    objective: str

    route_kind: (
        AnalyticalRequestRouteKind
    )


    entity_outlier_report: (
        EntityOutlierRequestReport
        | None
    ) = None


    planner_report: (
        AIPlannerReport
        | None
    ) = None


    router_rule_version: str = (
        ANALYTICAL_REQUEST_ROUTER_RULE_VERSION
    )


# ============================================================
# VALIDATION
# ============================================================


def _validate_objective(
    objective: str,
) -> str:
    normalized = str(
        objective
        or
        ""
    ).strip()


    if not normalized:
        raise ValueError(
            (
                "L'objectif utilisateur "
                "ne peut pas être vide."
            )
        )


    return (
        normalized
    )


# ============================================================
# ENTITY ROUTING
# ============================================================


def _try_entity_outlier_route(
    *,
    objective: str,

    source_dataset_records: list[
        dict[
            str,
            Any,
        ]
    ],

    top_profile_limit: int,
) -> (
    EntityOutlierRequestReport
    | None
):
    """
    Try the explicit entity-outlier route.

    This resolver runs BEFORE the generic analytical intent
    resolver.

    This ordering is intentional:

        "Détecte les clients atypiques"

    contains an outlier concept, but it is not a request for a
    simple variable-level distribution.

    The explicit entity grain therefore has priority.
    """

    resolution = (
        resolve_entity_outlier_intent(
            objective
        )
    )


    if (
        resolution.status
        !=
        "matched"
    ):
        return None


    return (
        run_entity_outlier_request(
            objective=
                objective,

            source_dataset_records=
                source_dataset_records,

            top_profile_limit=
                top_profile_limit,
        )
    )


# ============================================================
# PUBLIC ROUTER
# ============================================================


def route_analytical_request(
    *,
    objective: str,

    source_dataset_records: list[
        dict[
            str,
            Any,
        ]
    ],

    catalog: PlannerCatalog,

    planner_model: str = (
        DEFAULT_AI_PLANNER_MODEL
    ),

    entity_top_profile_limit: int = 50,
) -> AnalyticalRequestRoutingReport:
    """
    Route a DataLens analytical request through the highest
    appropriate deterministic semantic layer.

    Priority order:

    1. Explicit entity-level analytical intents
       Example:
           "Détecte les clients atypiques."

       These requests may require:
       - validated analytical views;
       - entity-grain aggregation;
       - deterministic anomaly detection;
       - profile classification.

    2. Existing intent-routed analytical planner
       Example:
           "Détecte les outliers."

       This path already handles:
       - deterministic generic intents;
       - precise requests;
       - local AI fallback;
       - Python validation.

    Important:
    The router does NOT collapse entity results into an
    AIPlannerReport.

    The caller receives an explicit route_kind and therefore
    knows which canonical result contract is populated.
    """

    normalized_objective = (
        _validate_objective(
            objective
        )
    )


    if (
        entity_top_profile_limit
        <
        1
    ):
        raise ValueError(
            (
                "entity_top_profile_limit "
                "must be greater than zero."
            )
        )


    # ========================================================
    # 1. EXPLICIT ENTITY INTENT
    # ========================================================

    entity_report = (
        _try_entity_outlier_route(
            objective=
                normalized_objective,

            source_dataset_records=
                source_dataset_records,

            top_profile_limit=
                entity_top_profile_limit,
        )
    )


    if (
        entity_report
        is not None
    ):
        return (
            AnalyticalRequestRoutingReport(
                objective=
                    normalized_objective,

                route_kind=
                    "entity_outlier",

                entity_outlier_report=
                    entity_report,

                planner_report=
                    None,
            )
        )


    # ========================================================
    # 2. EXISTING GENERIC / AI PLANNER
    # ========================================================

    planner_report = (
        plan_analyses_with_intent_routing(
            objective=
                normalized_objective,

            catalog=
                catalog,

            model=
                planner_model,
        )
    )


    return (
        AnalyticalRequestRoutingReport(
            objective=
                normalized_objective,

            route_kind=
                "analytical_plan",

            entity_outlier_report=
                None,

            planner_report=
                planner_report,
        )
    )