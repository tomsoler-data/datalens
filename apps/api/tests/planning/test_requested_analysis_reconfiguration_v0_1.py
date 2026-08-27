from app.planning.request_resolution import (
    REQUESTED_ANALYSIS_RESOLUTION_RULE_VERSION,
    reconfigure_requested_analysis,
    resolve_requested_analysis,
)
from app.planning.schemas import (
    RequestedAnalysisPlan,
    RequestedAnalysisResolution,
    RequestedColumnMatch,
)


def match(
    concept: str,
    column: str,
    analysis_kind: str,
) -> RequestedColumnMatch:
    return RequestedColumnMatch(
        concept=concept,
        dataset_id="dataset:transactions",
        dataset_filename="transactions.csv",
        column=column,
        analysis_kind=analysis_kind,
        match_score=100,
        reasons=[
            "test"
        ],
    )


def ambiguous_time_series_plan(
) -> RequestedAnalysisPlan:
    return RequestedAnalysisPlan(
        request_id=
            "request:test-revenue-moving-average",

        request_text=
            (
                "Afficher le chiffre d'affaires avec "
                "une moyenne mobile."
            ),

        evidence_quote=
            (
                "Afficher le chiffre d'affaires avec "
                "une moyenne mobile."
            ),

        source_filename=
            "brief.pdf",

        source_locator=
            "page 1",

        page_number=
            1,

        source_chunk_id=
            "chunk:test-time-series",

        evidence_unit_id=
            1,

        kind=
            "revenue_moving_average",

        status=
            "ambiguous",

        target_family=
            "time_series",

        matched_columns=[
            match(
                "amount",
                "price",
                "quantitative",
            ),
            match(
                "time",
                "date",
                "temporal",
            ),
        ],

        required_dataset_ids=[
            "dataset:transactions"
        ],

        required_dataset_filenames=[
            "transactions.csv"
        ],

        required_operations=[
            (
                "Resolve time granularity and "
                "moving-average window."
            )
        ],

        reasons=[
            "Time-series request detected."
        ],

        blockers=[
            (
                "Time granularity and moving-average "
                "window require explicit clarification."
            )
        ],
    )


print(
    "=== DATALENS REQUESTED ANALYSIS RECONFIGURATION v0.1 ==="
)
print()


original = (
    ambiguous_time_series_plan()
)


resolved = (
    resolve_requested_analysis(
        plan=
            original,

        resolution=
            RequestedAnalysisResolution(
                resolution_type=
                    "time_series_parameters",

                time_granularity=
                    "week",

                moving_average_window=
                    4,
            ),
    )
)


assert (
    resolved.status
    ==
    "ready"
)

assert (
    resolved.resolution
    is not None
)

assert (
    resolved.resolution.time_granularity
    ==
    "week"
)

assert (
    resolved.resolution.moving_average_window
    ==
    4
)


print(
    "[PASS] initial ambiguity resolves to week / 4"
)


reconfigured = (
    reconfigure_requested_analysis(
        plan=
            resolved,

        resolution=
            RequestedAnalysisResolution(
                resolution_type=
                    "time_series_parameters",

                time_granularity=
                    "month",

                moving_average_window=
                    3,
            ),
    )
)


assert (
    reconfigured.status
    ==
    "ready"
)

assert (
    reconfigured.resolution
    is not None
)

assert (
    reconfigured.resolution.time_granularity
    ==
    "month"
)

assert (
    reconfigured.resolution.moving_average_window
    ==
    3
)


print(
    "[PASS] ready analysis reconfigures to month / 3"
)


# ============================================================
# IDENTITY / PROVENANCE
# ============================================================

assert (
    reconfigured.request_id
    ==
    resolved.request_id
)

assert (
    reconfigured.request_text
    ==
    resolved.request_text
)

assert (
    reconfigured.context_text
    ==
    resolved.context_text
)

assert (
    reconfigured.evidence_quote
    ==
    resolved.evidence_quote
)

assert (
    reconfigured.source_filename
    ==
    resolved.source_filename
)

assert (
    reconfigured.source_locator
    ==
    resolved.source_locator
)

assert (
    reconfigured.page_number
    ==
    resolved.page_number
)

assert (
    reconfigured.source_chunk_id
    ==
    resolved.source_chunk_id
)

assert (
    reconfigured.evidence_unit_id
    ==
    resolved.evidence_unit_id
)

assert (
    reconfigured.matched_columns
    ==
    resolved.matched_columns
)

assert (
    reconfigured.required_dataset_ids
    ==
    resolved.required_dataset_ids
)

assert (
    reconfigured.required_dataset_filenames
    ==
    resolved.required_dataset_filenames
)


print(
    "[PASS] documentary identity and bindings are preserved"
)


# ============================================================
# PREVIOUS USER PARAMETER TRACE MUST BE REPLACED
# ============================================================

selection_reasons = [
    reason

    for reason
    in reconfigured.reasons

    if (
        reason.startswith(
            (
                "The user explicitly selected "
                "time granularity="
            )
        )
    )
]


assert (
    selection_reasons
    ==
    [
        (
            "The user explicitly selected "
            "time granularity=month "
            "and moving-average window=3."
        )
    ]
)

assert not any(
    "time granularity=week"
    in reason

    for reason
    in reconfigured.reasons
)


print(
    "[PASS] previous week / 4 trace is replaced, not accumulated"
)


# ============================================================
# ORIGINAL READY PLAN MUST REMAIN IMMUTABLE
# ============================================================

assert (
    resolved.resolution
    is not None
)

assert (
    resolved.resolution.time_granularity
    ==
    "week"
)

assert (
    resolved.resolution.moving_average_window
    ==
    4
)


print(
    "[PASS] original ready plan remains unchanged"
)


# ============================================================
# FAIL-CLOSED GUARDS
# ============================================================

ambiguous_failed = False


try:
    reconfigure_requested_analysis(
        plan=
            ambiguous_time_series_plan(),

        resolution=
            RequestedAnalysisResolution(
                resolution_type=
                    "time_series_parameters",

                time_granularity=
                    "month",

                moving_average_window=
                    3,
            ),
    )

except ValueError:
    ambiguous_failed = True


assert (
    ambiguous_failed
)


print(
    "[PASS] ambiguous request cannot use reconfiguration"
)


wrong_resolution_failed = False


try:
    reconfigure_requested_analysis(
        plan=
            resolved,

        resolution=
            RequestedAnalysisResolution(
                resolution_type=
                    "ranking_metric",

                ranking_metric=
                    "revenue",
            ),
    )

except ValueError:
    wrong_resolution_failed = True


assert (
    wrong_resolution_failed
)


print(
    "[PASS] ranking clarification cannot reconfigure time series"
)


wrong_kind_payload = (
    resolved.model_dump()
    if hasattr(
        resolved,
        "model_dump",
    )
    else
    resolved.dict()
)


wrong_kind_payload[
    "kind"
] = (
    "top_products"
)


wrong_kind = (
    RequestedAnalysisPlan(
        **wrong_kind_payload
    )
)


wrong_kind_failed = False


try:
    reconfigure_requested_analysis(
        plan=
            wrong_kind,

        resolution=
            RequestedAnalysisResolution(
                resolution_type=
                    "time_series_parameters",

                time_granularity=
                    "month",

                moving_average_window=
                    3,
            ),
    )

except ValueError:
    wrong_kind_failed = True


assert (
    wrong_kind_failed
)


print(
    "[PASS] unsupported requested kind fails closed"
)

print(
    "[PASS] rule version "
    +
    REQUESTED_ANALYSIS_RESOLUTION_RULE_VERSION
)

print()
print(
    "PASS - requested analysis reconfiguration v0.1"
)
