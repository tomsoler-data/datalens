from __future__ import annotations

from tests.preparation.test_preparation_combine_api_v0_1 import (
    PREPARATION_COMBINE_API_VERSION,
    build_ready_preparation,
    client,
)


def test_sequence_endpoint_executes_all_safe_server_derived_joins() -> None:
    session = build_ready_preparation()

    # --------------------------------------------------------
    # SERVER DISCOVERY OWNS THE FIRST REQUEST ID
    # --------------------------------------------------------

    discovery_response = client.post(
        "/preparation/combine/discover",
        json={
            "workflow_id":
                session.workflow_id,
        },
    )

    assert (
        discovery_response.status_code
        ==
        200
    )

    request_id = (
        discovery_response
        .json()[
            "discovery"
        ][
            "intent"
        ][
            "request_id"
        ]
    )

    # --------------------------------------------------------
    # ONE USER AUTHORIZATION FOR THE BOUNDED SEQUENCE
    # --------------------------------------------------------

    response = client.post(
        "/preparation/combine/sequence",
        json={
            "workflow_id":
                session.workflow_id,

            "request_id":
                request_id,

            "comment":
                (
                    "Assembler les tables liées "
                    "avec les contrôles serveur."
                ),
        },
    )

    print(
        "Combine sequence API status:",
        response.status_code,
    )

    assert (
        response.status_code
        ==
        200
    ), (
        "RED CONTRACT: "
        "POST /preparation/combine/sequence "
        "does not exist yet."
    )

    body = response.json()

    assert (
        body[
            "api_version"
        ]
        ==
        PREPARATION_COMBINE_API_VERSION
    )

    assert (
        body[
            "service_version"
        ]
    )

    # --------------------------------------------------------
    # TWO INDIVIDUALLY VALIDATED EXECUTIONS
    # --------------------------------------------------------

    executions = (
        body[
            "executions"
        ]
    )

    assert len(
        executions
    ) == 2

    assert all(
        execution[
            "validation"
        ][
            "valid_for_downstream"
        ]
        is True
        for execution
        in executions
    )

    # --------------------------------------------------------
    # FINAL FRONTIER
    # --------------------------------------------------------

    final_discovery = (
        body[
            "final_discovery"
        ]
    )

    assert (
        final_discovery[
            "has_candidate"
        ]
        is False
    )

    assert len(
        final_discovery[
            "active_dataset_ids"
        ]
    ) == 1

    # --------------------------------------------------------
    # SERVER-OWNED SESSION
    # --------------------------------------------------------

    combine_stage = next(
        stage
        for stage
        in body[
            "session"
        ][
            "snapshot"
        ][
            "stages"
        ]
        if stage[
            "stage"
        ]
        ==
        "combine"
    )

    assert (
        combine_stage[
            "status"
        ]
        ==
        "passed"
    )


if __name__ == "__main__":
    print(
        "=== DATALENS COMBINE SEQUENCE API CONTRACT v0.1 ==="
    )

    test_sequence_endpoint_executes_all_safe_server_derived_joins()

    print()
    print(
        "PASS - combine sequence API contract."
    )
