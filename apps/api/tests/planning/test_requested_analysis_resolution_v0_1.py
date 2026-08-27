from app.planning.request_resolution import (
    REQUESTED_ANALYSIS_RESOLUTION_RULE_VERSION,
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
) -> RequestedColumnMatch:
    return RequestedColumnMatch(
        concept=concept,
        dataset_id="dataset:transactions",
        dataset_filename="transactions.csv",
        column=column,
        analysis_kind="ranking",
        match_score=100,
        reasons=[
            "test"
        ],
    )


def ambiguous_plan() -> RequestedAnalysisPlan:
    return RequestedAnalysisPlan(
        request_id="request:test-top",
        request_text="les tops",
        evidence_quote="les tops",
        source_filename="brief.pdf",
        source_locator="page 1",
        page_number=1,
        source_chunk_id="chunk:test",
        evidence_unit_id=6,
        kind="top_products",
        status="ambiguous",
        target_family="ranking",
        matched_columns=[
            match(
                "product_id",
                "id_prod",
            ),
            match(
                "amount",
                "price",
            ),
        ],
        required_dataset_ids=[
            "dataset:transactions"
        ],
        required_dataset_filenames=[
            "transactions.csv"
        ],
        required_operations=[
            "Resolve ranking metric."
        ],
        reasons=[
            "Ranking intent detected."
        ],
        blockers=[
            "Ranking metric is ambiguous."
        ],
    )


print(
    "===== DATALENS REQUEST RESOLUTION v0.1 ====="
)
print()


original = (
    ambiguous_plan()
)

resolved = (
    resolve_requested_analysis(
        plan=
            original,

        resolution=
            RequestedAnalysisResolution(
                ranking_metric=
                    "revenue"
            ),
    )
)


assert (
    resolved.status
    ==
    "ready"
)

assert (
    resolved.request_id
    ==
    original.request_id
)

assert (
    resolved.request_text
    ==
    original.request_text
)

assert (
    resolved.source_filename
    ==
    original.source_filename
)

assert (
    resolved.source_chunk_id
    ==
    original.source_chunk_id
)

assert (
    resolved.evidence_unit_id
    ==
    original.evidence_unit_id
)

assert (
    resolved.resolution
    is not None
)

assert (
    resolved.resolution.ranking_metric
    ==
    "revenue"
)

assert (
    resolved.blockers
    ==
    []
)


print(
    "[PASS] revenue clarification resolves ambiguous -> ready"
)

print(
    "[PASS] request_id preserved"
)

print(
    "[PASS] request text preserved"
)

print(
    "[PASS] documentary provenance preserved"
)


blocked_units = (
    resolve_requested_analysis(
        plan=
            ambiguous_plan(),

        resolution=
            RequestedAnalysisResolution(
                ranking_metric=
                    "units"
            ),
    )
)


assert (
    blocked_units.status
    ==
    "blocked"
)

assert (
    blocked_units.request_id
    ==
    original.request_id
)

assert (
    blocked_units.resolution
    is not None
)

assert (
    blocked_units.resolution.ranking_metric
    ==
    "units"
)

assert (
    len(
        blocked_units.blockers
    )
    >
    0
)


print(
    "[PASS] unavailable units metric fails closed"
)


transaction_plan = (
    ambiguous_plan()
)

transaction_payload = (
    transaction_plan.model_dump()
    if hasattr(
        transaction_plan,
        "model_dump",
    )
    else
    transaction_plan.dict()
)

transaction_payload[
    "matched_columns"
] = [
    *transaction_plan.matched_columns,
    match(
        "session_id",
        "session_id",
    ),
]

transaction_plan = (
    RequestedAnalysisPlan(
        **transaction_payload
    )
)


resolved_transactions = (
    resolve_requested_analysis(
        plan=
            transaction_plan,

        resolution=
            RequestedAnalysisResolution(
                ranking_metric=
                    "transaction_count"
            ),
    )
)


assert (
    resolved_transactions.status
    ==
    "ready"
)

assert (
    resolved_transactions.request_id
    ==
    transaction_plan.request_id
)

assert (
    resolved_transactions.resolution
    is not None
)

assert (
    resolved_transactions.resolution.ranking_metric
    ==
    "transaction_count"
)

assert (
    resolved_transactions.blockers
    ==
    []
)


print(
    "[PASS] resolved session identifier enables transaction-count ranking"
)


# ============================================================
# TIME-SERIES PARAMETER RESOLUTION
# ============================================================

def ambiguous_time_series_plan(
    *,
    time_dataset_id: str =
        "dataset:transactions",
) -> RequestedAnalysisPlan:
    return RequestedAnalysisPlan(
        request_id=
            "request:test-revenue-moving-average",

        request_text=
            (
                "chiffre d'affaires avec moyenne "
                "mobile, choix jour semaine mois"
            ),

        evidence_quote=
            (
                "chiffre d'affaires avec moyenne "
                "mobile"
            ),

        source_filename=
            "brief.pdf",

        source_locator=
            "page 1",

        page_number=
            1,

        source_chunk_id=
            "chunk:time-series",

        evidence_unit_id=
            1,

        kind=
            "revenue_moving_average",

        status=
            "ambiguous",

        target_family=
            "time_series",

        matched_columns=[
            RequestedColumnMatch(
                concept=
                    "amount",

                dataset_id=
                    "dataset:transactions",

                dataset_filename=
                    "transactions.csv",

                column=
                    "price",

                analysis_kind=
                    "time_series",

                match_score=
                    100,

                reasons=[
                    "test"
                ],
            ),

            RequestedColumnMatch(
                concept=
                    "time",

                dataset_id=
                    time_dataset_id,

                dataset_filename=
                    "transactions.csv",

                column=
                    "date",

                analysis_kind=
                    "time_series",

                match_score=
                    100,

                reasons=[
                    "test"
                ],
            ),
        ],

        required_dataset_ids=[
            "dataset:transactions"
        ],

        required_dataset_filenames=[
            "transactions.csv"
        ],

        required_operations=[
            "Resolve time-series parameters."
        ],

        reasons=[
            (
                "Time-series request allows the "
                "user to choose the period."
            )
        ],

        blockers=[
            (
                "Time granularity and moving-average "
                "window require user resolution."
            )
        ],
    )


time_original = (
    ambiguous_time_series_plan()
)


time_resolved = (
    resolve_requested_analysis(
        plan=
            time_original,

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
    time_resolved.status
    ==
    "ready"
)

assert (
    time_resolved.request_id
    ==
    time_original.request_id
)

assert (
    time_resolved.request_text
    ==
    time_original.request_text
)

assert (
    time_resolved.source_filename
    ==
    time_original.source_filename
)

assert (
    time_resolved.source_chunk_id
    ==
    time_original.source_chunk_id
)

assert (
    time_resolved.evidence_unit_id
    ==
    time_original.evidence_unit_id
)

assert (
    time_resolved.resolution
    is not None
)

assert (
    time_resolved.resolution.resolution_type
    ==
    "time_series_parameters"
)

assert (
    time_resolved.resolution.time_granularity
    ==
    "week"
)

assert (
    time_resolved.resolution.moving_average_window
    ==
    4
)

assert (
    time_resolved.blockers
    ==
    []
)


print(
    "[PASS] weekly moving-average clarification resolves ambiguous -> ready"
)

print(
    "[PASS] time-series resolution preserves request identity and provenance"
)

print(
    "[PASS] time-series granularity and window are server-owned resolution data"
)


cross_dataset = (
    resolve_requested_analysis(
        plan=
            ambiguous_time_series_plan(
                time_dataset_id=
                    "dataset:other"
            ),

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
    cross_dataset.status
    ==
    "blocked"
)

assert (
    len(
        cross_dataset.blockers
    )
    >
    0
)


print(
    "[PASS] cross-dataset time/value resolution fails closed"
)


invalid_window_failed = False

try:
    RequestedAnalysisResolution(
        resolution_type=
            "time_series_parameters",

        time_granularity=
            "month",

        moving_average_window=
            0,
    )

except Exception:
    invalid_window_failed = True


assert (
    invalid_window_failed
)


print(
    "[PASS] moving-average window must be >= 1"
)


failed = False

try:
    resolve_requested_analysis(
        plan=
            resolved,

        resolution=
            RequestedAnalysisResolution(
                ranking_metric=
                    "revenue"
            ),
    )

except ValueError:
    failed = True


assert failed


print(
    "[PASS] non-ambiguous plan cannot be resolved again"
)

print(
    "[PASS] resolution does not bypass deterministic validation"
)

print(
    "[PASS] rule version "
    +
    REQUESTED_ANALYSIS_RESOLUTION_RULE_VERSION
)

print()
print(
    "PASS - requested analysis resolution v0.1"
)
