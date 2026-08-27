from __future__ import annotations

import os
import tempfile

from pathlib import Path
from unittest.mock import patch


# ============================================================
# ISOLATED STORES
# ============================================================

temporary = tempfile.TemporaryDirectory()

root = Path(
    temporary.name
)


os.environ[
    "DATALENS_ANALYSIS_ARTIFACT_STORE_PATH"
] = str(
    root
    /
    "analysis_artifacts.json"
)


os.environ[
    "DATALENS_REPORT_SELECTION_STORE_PATH"
] = str(
    root
    /
    "report_selection.json"
)


from app.reporting.analysis_artifact_store import (
    list_analysis_artifacts,
)

from app.reporting.report_selection_store import (
    get_report_selection,
)

from app.reporting.requested_adapter import (
    requested_analysis_id,
)

from app.reporting.unified_report_artifacts import (
    _stable_id,
    register_unresolved_requested_analysis_artifacts,
)


# ============================================================
# MINIMAL SERVER-OWNED TEST OBJECT
# ============================================================

class FakeModel:
    def __init__(
        self,
        **values,
    ):
        for key, value in values.items():
            setattr(
                self,
                key,
                value,
            )


    def model_dump(
        self,
        mode="python",
    ):
        return dict(
            self.__dict__
        )


def execution(
    *,
    request_id,
    request_text,
    plan_status,
    execution_status,
    warnings=None,
):
    return FakeModel(
        request_id=
            request_id,

        request_text=
            request_text,

        kind=
            "document_request",

        plan_status=
            plan_status,

        execution_status=
            execution_status,

        inferential_status=
            "not_applicable",

        analysis_mode=
            None,

        dataset_id=
            None,

        dataset_filename=
            None,

        analytical_grain=
            None,

        variables=
            {},

        warnings=
            list(
                warnings
                or
                []
            ),

        limitations=
            (
                [
                    (
                        "La demande n'a pas ete "
                        "executee car son plan "
                        "analytique n'est pas ready."
                    )
                ]
                if execution_status
                ==
                "not_executed"
                else
                []
            ),

        result=
            None,

        descriptive_statistics=
            None,
    )


def plan(
    *,
    request_id,
    request_text,
    status,
):
    return FakeModel(
        request_id=
            request_id,

        request_text=
            request_text,

        status=
            status,

        source_filename=
            "Brief.pdf",

        source_locator=
            "page 1",

        page_number=
            1,

        source_chunk_id=
            f"chunk:{request_id}",

        evidence_unit_id=
            f"evidence:{request_id}",

        evidence_quote=
            request_text,

        blockers=
            [],
    )


# ============================================================
# FIXTURE
# ============================================================

top_execution = execution(
    request_id=
        "request:top",

    request_text=
        "top produits",

    plan_status=
        "ambiguous",

    execution_status=
        "not_executed",

    warnings=[
        "Ranking criterion is ambiguous."
    ],
)


flop_execution = execution(
    request_id=
        "request:flop",

    request_text=
        "flop produits",

    plan_status=
        "ambiguous",

    execution_status=
        "not_executed",

    warnings=[
        "Ranking criterion is ambiguous."
    ],
)


btob_execution = execution(
    request_id=
        "request:btob",

    request_text=
        "repartition du CA BtoB",

    plan_status=
        "blocked",

    execution_status=
        "not_executed",

    warnings=[
        "No defensible BtoB identification rule."
    ],
)


complete_execution = execution(
    request_id=
        "request:revenue",

    request_text=
        "chiffre d'affaires",

    plan_status=
        "ready",

    execution_status=
        "complete",
)


executions = [
    top_execution,
    flop_execution,
    btob_execution,
    complete_execution,
]


plans = [
    plan(
        request_id=
            item.request_id,

        request_text=
            item.request_text,

        status=
            item.plan_status,
    )

    for item in executions
]


plan_map = {
    item.request_id:
        item

    for item in plans
}


execution_report = FakeModel(
    results=
        executions
)


plan_report = FakeModel(
    requests=
        plans
)


workflow_id = (
    "prep:test-unresolved-requests"
)


print()
print(
    "===== UNRESOLVED REQUEST ARTIFACTS v0.1 ====="
)
print()


# ============================================================
# REGISTER
# ============================================================

with patch(
    "app.reporting.requested_adapter."
    "build_request_plan_map",

    return_value=
        plan_map,
):
    registered = (
        register_unresolved_requested_analysis_artifacts(
            workflow_id=
                workflow_id,

            execution_report=
                execution_report,

            plan_report=
                plan_report,
        )
    )


# ============================================================
# 1. EXACTLY THREE UNRESOLVED
# ============================================================

assert (
    len(
        registered
    )
    ==
    3
)


print(
    "[PASS] exactly 3 unresolved requests registered"
)


# ============================================================
# 2. REPORTABLE RESULT WAS NOT REGISTERED HERE
# ============================================================

records = (
    list_analysis_artifacts(
        workflow_id=
            workflow_id
    )
)


assert (
    len(
        records
    )
    ==
    3
)


assert all(
    record.objective
    !=
    "chiffre d'affaires"

    for record in records
)


print(
    "[PASS] reportable request skipped by lifecycle registrar"
)


# ============================================================
# 3. ALL THREE ARE NON-EXECUTABLE ARTIFACTS
# ============================================================

assert all(
    record.source_type
    ==
    "document_request"

    for record in records
)


assert all(
    record.executed
    is False

    for record in records
)


assert all(
    record.executed_count
    ==
    0

    for record in records
)


print(
    "[PASS] unresolved artifacts remain non-executed"
)


# ============================================================
# 4. SAME STABLE IDENTITY AS FUTURE FINDING
# ============================================================

execution_by_request = {
    item.request_id:
        item

    for item in (
        top_execution,
        flop_execution,
        btob_execution,
    )
}


for record in records:
    lifecycle = (
        record.pipeline_payload[
            "request_lifecycle"
        ]
    )


    request_id = (
        lifecycle[
            "request_id"
        ]
    )


    original_execution = (
        execution_by_request[
            request_id
        ]
    )


    source_analysis_id = (
        requested_analysis_id(
            original_execution
        )
    )


    expected_id = (
        _stable_id(
            workflow_id=
                workflow_id,

            source_type=
                "document_request",

            source_analysis_id=
                source_analysis_id,
        )
    )


    assert (
        record.analysis_id
        ==
        expected_id
    )


print(
    "[PASS] unresolved requests use future finding identity"
)


# ============================================================
# 5. SERVER-OWNED LIFECYCLE PAYLOAD
# ============================================================

statuses = {}


for record in records:
    payload = (
        record.pipeline_payload
    )


    assert (
        payload[
            "artifact_kind"
        ]
        ==
        "requested_analysis_lifecycle"
    )


    lifecycle = (
        payload[
            "request_lifecycle"
        ]
    )


    assert (
        lifecycle[
            "execution_status"
        ]
        ==
        "not_executed"
    )


    assert lifecycle[
        "warnings"
    ]


    assert (
        lifecycle[
            "source_filename"
        ]
        ==
        "Brief.pdf"
    )


    statuses[
        lifecycle[
            "request_id"
        ]
    ] = (
        lifecycle[
            "plan_status"
        ]
    )


assert statuses == {
    "request:top":
        "ambiguous",

    "request:flop":
        "ambiguous",

    "request:btob":
        "blocked",
}


print(
    "[PASS] planner status, blockers and provenance persisted"
)


# ============================================================
# 6. REPORT SELECTION MUST STAY EMPTY
# ============================================================

selection = (
    get_report_selection(
        workflow_id=
            workflow_id
    )
)


assert (
    selection.selected_count
    ==
    0
)


print(
    "[PASS] unresolved requests remain outside report selection"
)


print()
print(
    "Persisted:",
    len(
        records
    ),
)

print(
    "Ambiguous:",
    sum(
        1

        for value in statuses.values()

        if value
        ==
        "ambiguous"
    ),
)

print(
    "Blocked:",
    sum(
        1

        for value in statuses.values()

        if value
        ==
        "blocked"
    ),
)

print(
    "Selected:",
    selection.selected_count,
)


print()
print(
    "PASS - unresolved requested artifacts v0.1"
)


temporary.cleanup()
