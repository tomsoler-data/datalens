from __future__ import annotations

from typing import Any

from app.planning.schemas import (
    RequestedAnalysisPlan,
    RequestedAnalysisResolution,
)


REQUESTED_ANALYSIS_RESOLUTION_RULE_VERSION = (
    "requested_analysis_resolution_v0.3"
)


# ============================================================
# MODEL HELPERS
# ============================================================

def _model_dump(
    model: Any,
) -> dict[
    str,
    Any,
]:
    if hasattr(
        model,
        "model_dump",
    ):
        return dict(
            model.model_dump()
        )


    if hasattr(
        model,
        "dict",
    ):
        return dict(
            model.dict()
        )


    raise TypeError(
        "Requested analysis resolution requires "
        "a Pydantic-compatible model."
    )


def _matches_for_concept(
    plan: RequestedAnalysisPlan,
    concept: str,
):
    return [
        match

        for match
        in plan.matched_columns

        if (
            str(
                match.concept
            )
            ==
            concept
        )
    ]


def _single_match(
    plan: RequestedAnalysisPlan,
    concept: str,
):
    matches = (
        _matches_for_concept(
            plan,
            concept,
        )
    )


    if (
        len(
            matches
        )
        !=
        1
    ):
        return None


    return matches[
        0
    ]


# ============================================================
# RANKING RESOLUTION
# ============================================================

def _resolve_ranking_metric(
    *,
    plan: RequestedAnalysisPlan,
    resolution: RequestedAnalysisResolution,
) -> RequestedAnalysisPlan:
    product_match = (
        _single_match(
            plan,
            "product_id",
        )
    )


    metric = (
        resolution.ranking_metric
    )


    if metric is None:
        raise ValueError(
            "ranking_metric is required for "
            "ranking_metric resolution."
        )


    blockers: list[
        str
    ] = []


    if product_match is None:
        blockers.append(
            (
                "The product identifier is not "
                "resolved uniquely."
            )
        )


    if (
        metric
        ==
        "revenue"
    ):
        metric_match = (
            _single_match(
                plan,
                "amount",
            )
        )

        if metric_match is None:
            blockers.append(
                (
                    "Revenue ranking requires one "
                    "uniquely resolved monetary "
                    "measure."
                )
            )

        operation_label = (
            "Aggregate revenue by product."
        )

        reason_label = (
            "The user explicitly selected revenue "
            "as the ranking metric."
        )


    elif (
        metric
        ==
        "units"
    ):
        metric_match = (
            _single_match(
                plan,
                "quantity",
            )
        )

        if metric_match is None:
            blockers.append(
                (
                    "Units ranking requires one "
                    "uniquely resolved quantity "
                    "measure."
                )
            )

        operation_label = (
            "Aggregate units sold by product."
        )

        reason_label = (
            "The user explicitly selected units "
            "sold as the ranking metric."
        )


    elif (
        metric
        ==
        "transaction_count"
    ):
        transaction_match = (
            _single_match(
                plan,
                "transaction_id",
            )
        )

        session_match = (
            _single_match(
                plan,
                "session_id",
            )
        )


        if (
            transaction_match is None
            and
            session_match is None
        ):
            blockers.append(
                (
                    "Transaction-count ranking "
                    "requires one uniquely resolved "
                    "transaction or session "
                    "identifier."
                )
            )

        operation_label = (
            "Count distinct transactions by product."
        )

        reason_label = (
            "The user explicitly selected "
            "transaction count as the ranking "
            "metric."
        )


    else:
        raise ValueError(
            (
                "Unsupported ranking metric: "
                f"{metric}"
            )
        )


    status = (
        "blocked"
        if blockers
        else
        "ready"
    )


    direction = (
        "descending"
        if (
            plan.kind
            ==
            "top_products"
        )
        else
        "ascending"
    )


    payload = (
        _model_dump(
            plan
        )
    )


    payload.update(
        {
            "status":
                status,

            "resolution":
                _model_dump(
                    resolution
                ),

            "required_operations":
                [
                    operation_label,
                    (
                        "Rank products in "
                        f"{direction} order."
                    ),
                ],

            "reasons":
                [
                    *list(
                        plan.reasons
                    ),
                    reason_label,
                ],

            "blockers":
                blockers,
        }
    )


    return (
        RequestedAnalysisPlan(
            **payload
        )
    )


# ============================================================
# TIME-SERIES RESOLUTION
# ============================================================

def _resolve_time_series_parameters(
    *,
    plan: RequestedAnalysisPlan,
    resolution: RequestedAnalysisResolution,
) -> RequestedAnalysisPlan:
    granularity = (
        resolution.time_granularity
    )

    window = (
        resolution.moving_average_window
    )


    if granularity is None:
        raise ValueError(
            "time_granularity is required for "
            "time_series_parameters resolution."
        )


    if window is None:
        raise ValueError(
            "moving_average_window is required for "
            "time_series_parameters resolution."
        )


    amount_match = (
        _single_match(
            plan,
            "amount",
        )
    )

    time_match = (
        _single_match(
            plan,
            "time",
        )
    )


    blockers: list[
        str
    ] = []


    if amount_match is None:
        blockers.append(
            (
                "Time-series revenue analysis "
                "requires one uniquely resolved "
                "monetary measure."
            )
        )


    if time_match is None:
        blockers.append(
            (
                "Time-series revenue analysis "
                "requires one uniquely resolved "
                "temporal variable."
            )
        )


    if (
        amount_match is not None
        and
        time_match is not None
        and
        str(
            amount_match.dataset_id
        )
        !=
        str(
            time_match.dataset_id
        )
    ):
        blockers.append(
            (
                "The monetary measure and temporal "
                "variable must belong to the same "
                "server-resolved dataset."
            )
        )


    status = (
        "blocked"
        if blockers
        else
        "ready"
    )


    operation_labels = {
        "day":
            "Aggregate revenue by day.",

        "week":
            "Aggregate revenue by week.",

        "month":
            "Aggregate revenue by month.",

        "quarter":
            "Aggregate revenue by quarter.",

        "year":
            "Aggregate revenue by year.",
    }


    payload = (
        _model_dump(
            plan
        )
    )


    payload.update(
        {
            "status":
                status,

            "resolution":
                _model_dump(
                    resolution
                ),

            "required_operations":
                [
                    operation_labels[
                        granularity
                    ],
                    (
                        "Compute a moving average "
                        f"over {window} "
                        f"{granularity} period(s)."
                    ),
                ],

            "reasons":
                [
                    *list(
                        plan.reasons
                    ),
                    (
                        "The user explicitly selected "
                        f"time granularity={granularity} "
                        "and moving-average "
                        f"window={window}."
                    ),
                ],

            "blockers":
                blockers,
        }
    )


    return (
        RequestedAnalysisPlan(
            **payload
        )
    )


# ============================================================
# PUBLIC RESOLVER
# ============================================================

def resolve_requested_analysis(
    *,
    plan: RequestedAnalysisPlan,
    resolution: RequestedAnalysisResolution,
) -> RequestedAnalysisPlan:
    """
    Resolve one explicit ambiguity without changing the
    documentary request identity or provenance.

    Important invariants:

    - request_id is preserved;
    - request_text and documentary provenance are preserved;
    - a user choice does not force status='ready';
    - Python validates whether the selected interpretation is
      executable from the server-owned plan;
    - unsupported or insufficiently resolved inputs fail closed.
    """

    if (
        plan.status
        !=
        "ambiguous"
    ):
        raise ValueError(
            (
                "Only an ambiguous requested "
                "analysis can be resolved. "
                f"Current status={plan.status}."
            )
        )


    if (
        resolution.resolution_type
        ==
        "ranking_metric"
    ):
        if (
            plan.kind
            not in {
                "top_products",
                "flop_products",
            }
        ):
            raise ValueError(
                (
                    "Ranking-metric resolution is "
                    "only supported for top_products "
                    "and flop_products."
                )
            )


        resolved = (
            _resolve_ranking_metric(
                plan=
                    plan,

                resolution=
                    resolution,
            )
        )


    elif (
        resolution.resolution_type
        ==
        "time_series_parameters"
    ):
        if (
            plan.kind
            !=
            "revenue_moving_average"
        ):
            raise ValueError(
                (
                    "Time-series parameter resolution "
                    "is only supported for "
                    "revenue_moving_average."
                )
            )


        resolved = (
            _resolve_time_series_parameters(
                plan=
                    plan,

                resolution=
                    resolution,
            )
        )


    else:
        raise ValueError(
            (
                "Unsupported requested analysis "
                "resolution type."
            )
        )


    if (
        resolved.request_id
        !=
        plan.request_id
    ):
        raise RuntimeError(
            (
                "Requested analysis resolution "
                "changed request_id."
            )
        )


    if (
        resolved.request_text
        !=
        plan.request_text
    ):
        raise RuntimeError(
            (
                "Requested analysis resolution "
                "changed request_text."
            )
        )


    if (
        resolved.source_filename
        !=
        plan.source_filename
        or
        resolved.source_locator
        !=
        plan.source_locator
        or
        resolved.source_chunk_id
        !=
        plan.source_chunk_id
        or
        resolved.evidence_unit_id
        !=
        plan.evidence_unit_id
    ):
        raise RuntimeError(
            (
                "Requested analysis resolution "
                "changed documentary provenance."
            )
        )


    return resolved


# ============================================================
# REQUESTED ANALYSIS RECONFIGURATION
# ============================================================

_TIME_SERIES_SELECTION_REASON_PREFIX = (
    "The user explicitly selected time granularity="
)


def _without_previous_time_series_selection_reason(
    plan: RequestedAnalysisPlan,
) -> RequestedAnalysisPlan:
    """
    Remove only the deterministic reason generated by a previous
    time-series parameter choice.

    Documentary reasons and all other server-owned planning
    evidence remain untouched.
    """

    payload = (
        _model_dump(
            plan
        )
    )


    payload[
        "reasons"
    ] = [
        reason

        for reason
        in plan.reasons

        if not (
            str(
                reason
            )
            .startswith(
                _TIME_SERIES_SELECTION_REASON_PREFIX
            )
        )
    ]


    return (
        RequestedAnalysisPlan(
            **payload
        )
    )


def reconfigure_requested_analysis(
    *,
    plan: RequestedAnalysisPlan,
    resolution: RequestedAnalysisResolution,
) -> RequestedAnalysisPlan:
    """
    Reconfigure an already resolved requested time-series
    analysis without changing its documentary identity.

    This is intentionally different from
    resolve_requested_analysis():

    - resolve: ambiguous -> ready
    - reconfigure: ready -> ready

    Only explicit time-series parameters may be reconfigured.
    Dataset bindings, matched columns and documentary provenance
    remain server-owned.
    """

    if (
        plan.status
        !=
        "ready"
    ):
        raise ValueError(
            (
                "Only a ready requested analysis can be "
                "reconfigured. "
                f"Current status={plan.status}."
            )
        )


    if (
        plan.kind
        !=
        "revenue_moving_average"
    ):
        raise ValueError(
            (
                "Requested analysis reconfiguration is only "
                "supported for revenue_moving_average."
            )
        )


    previous_resolution = (
        plan.resolution
    )


    if (
        previous_resolution
        is None
        or
        previous_resolution.resolution_type
        !=
        "time_series_parameters"
    ):
        raise ValueError(
            (
                "The ready requested analysis does not contain "
                "server-owned time-series parameters that may "
                "be reconfigured."
            )
        )


    if (
        resolution.resolution_type
        !=
        "time_series_parameters"
    ):
        raise ValueError(
            (
                "Only time_series_parameters may reconfigure "
                "a revenue_moving_average request."
            )
        )


    clean_plan = (
        _without_previous_time_series_selection_reason(
            plan
        )
    )


    reconfigured = (
        _resolve_time_series_parameters(
            plan=
                clean_plan,

            resolution=
                resolution,
        )
    )


    # --------------------------------------------------------
    # Identity and documentary provenance must not mutate.
    # --------------------------------------------------------

    invariant_fields = (
        "request_id",
        "request_text",
        "context_text",
        "evidence_quote",
        "source_filename",
        "source_locator",
        "page_number",
        "source_chunk_id",
        "evidence_unit_id",
        "kind",
        "target_family",
        "matched_columns",
        "required_dataset_ids",
        "required_dataset_filenames",
    )


    for field_name in (
        invariant_fields
    ):
        if (
            getattr(
                reconfigured,
                field_name,
            )
            !=
            getattr(
                plan,
                field_name,
            )
        ):
            raise RuntimeError(
                (
                    "Requested analysis reconfiguration changed "
                    "a server-owned invariant: "
                    f"{field_name}."
                )
            )


    return (
        reconfigured
    )
