from app.planning.request_planner import (
    REQUEST_PLANNER_RULE_VERSION,
    _require_user_parameter_resolution,
)
from app.planning.schemas import (
    RequestedAnalysisPlan,
)


def plan(
    status: str,
) -> RequestedAnalysisPlan:
    return RequestedAnalysisPlan(
        request_id=
            "request:test-moving-average",

        request_text=
            (
                "chiffre d'affaires avec moyenne "
                "mobile, choix jour semaine mois"
            ),

        evidence_quote=
            "chiffre d'affaires avec moyenne mobile",

        source_filename=
            "brief.pdf",

        source_locator=
            "page 1",

        page_number=
            1,

        source_chunk_id=
            "chunk:moving-average",

        evidence_unit_id=
            1,

        kind=
            "revenue_moving_average",

        status=
            status,

        target_family=
            "time_series",

        matched_columns=[],

        required_dataset_ids=[
            "dataset:transactions"
        ],

        required_dataset_filenames=[
            "transactions.csv"
        ],

        required_operations=[
            "Aggregate revenue by resolved period."
        ],

        reasons=[
            "Revenue and time are structurally resolved."
        ],

        blockers=(
            []
            if status == "ready"
            else [
                "Temporal variable is unavailable."
            ]
        ),
    )


print(
    "===== DATALENS TIME-SERIES PLANNER AMBIGUITY v0.1 ====="
)
print()


original = plan(
    "ready"
)


resolved = (
    _require_user_parameter_resolution(
        plan=
            original,

        required_operation=
            "Resolve time-series parameters.",

        reason=
            "User choice is required.",

        blocker=
            "Time-series parameters require user resolution.",
    )
)


assert (
    resolved.status
    ==
    "ambiguous"
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
    resolved.blockers
    ==
    [
        "Time-series parameters require user resolution."
    ]
)

assert (
    resolved.required_operations[
        0
    ]
    ==
    "Resolve time-series parameters."
)


print(
    "[PASS] executable time-series request becomes ambiguous"
)

print(
    "[PASS] request identity preserved"
)

print(
    "[PASS] documentary provenance preserved"
)

print(
    "[PASS] user-resolution blocker is explicit"
)


blocked = plan(
    "blocked"
)


still_blocked = (
    _require_user_parameter_resolution(
        plan=
            blocked,

        required_operation=
            "Resolve time-series parameters.",

        reason=
            "User choice is required.",

        blocker=
            "Time-series parameters require user resolution.",
    )
)


assert (
    still_blocked.status
    ==
    "blocked"
)

assert (
    still_blocked.blockers
    ==
    blocked.blockers
)


print(
    "[PASS] structural blocker remains blocked"
)

print(
    "[PASS] ambiguity does not overwrite data-integrity failures"
)

print(
    "[PASS] planner rule version "
    +
    REQUEST_PLANNER_RULE_VERSION
)

print()
print(
    "PASS - time-series planner ambiguity v0.1"
)
