from __future__ import annotations

import pandas as pd

from app.preparation.dataset_identity import (
    profile_dataset_identity,
)

from app.preparation.preparation_artifact_store import (
    list_preparation_artifacts,
    put_preparation_artifact,
)

from app.preparation.preparation_combine_service import (
    approve_and_execute_combine_sequence,
    discover_next_combine,
)

from app.preparation.preparation_identity_resolution import (
    build_identity_resolution_request_id,
    record_continue_without_surrogate,
)

from app.preparation.preparation_session import (
    get_preparation_session,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
)

from tests.preparation.test_preparation_combine_service_v0_1 import (
    PRODUCTS,
    build_ready_preparation,
    stage_status,
)


def test_sequence_stops_before_blocked_many_to_many_join() -> None:
    """
    A safe first join may execute.

    If rediscovery then exposes a blocked relationship, the sequence
    must stop immediately without executing or materializing that
    second join.
    """

    session = build_ready_preparation()

    # --------------------------------------------------------
    # REPLACE PRODUCTS WITH AN INTENTIONALLY NON-UNIQUE TABLE
    # --------------------------------------------------------

    products = pd.DataFrame(
        {
            "product_id": [
                "P1",
                "P1",
                "P2",
                "P2",
            ],
            "category": [
                "A",
                "A",
                "B",
                "B",
            ],
        }
    )

    put_preparation_artifact(
        workflow_id=session.workflow_id,
        dataset_id=PRODUCTS,
        dataset_filename="products_test.csv",
        stage="source",
        dataframe=products,
        evidence_refs=[
            "test:blocked-products",
        ],
        replace=True,
    )

    # --------------------------------------------------------
    # EXPLICITLY RESOLVE ROW IDENTITY
    # --------------------------------------------------------
    #
    # This test concerns JOIN cardinality, not identity.
    # Products deliberately has no unique row identity, so the
    # analyst explicitly chooses to continue without a surrogate.
    # --------------------------------------------------------

    report = profile_dataset_identity(
        products,
        dataset_id=PRODUCTS,
        dataset_filename="products_test.csv",
    )

    assert (
        report.status
        ==
        "surrogate_recommended"
    )

    identity_request_id = (
        build_identity_resolution_request_id(
            workflow_id=session.workflow_id,
            dataset_id=PRODUCTS,
            dataset_filename="products_test.csv",
            artifact_stage="source",
            report=report,
        )
    )

    resolution = (
        record_continue_without_surrogate(
            workflow_id=session.workflow_id,
            dataset_id=PRODUCTS,
            dataset_filename="products_test.csv",
            artifact_stage="source",
            report=report,
            request_id=identity_request_id,
        )
    )

    assert (
        resolution.kind
        ==
        "continued_without_surrogate"
    )

    # --------------------------------------------------------
    # FIRST RELATION IS SAFE
    # --------------------------------------------------------

    first = discover_next_combine(
        session.workflow_id
    )

    assert first.has_candidate is True
    assert first.ready_for_approval is True
    assert first.intent is not None

    assert (
        first.intent.keys[0].left_column
        ==
        "customer_id"
    )

    assert (
        first.intent.expected_cardinality.value
        ==
        "many_to_one"
    )

    # --------------------------------------------------------
    # AUTHORIZE THE BOUNDED SEQUENCE
    # --------------------------------------------------------

    result = approve_and_execute_combine_sequence(
        workflow_id=session.workflow_id,
        request_id=first.intent.request_id,
        actor="test",
        comment=(
            "Fail-closed sequence regression."
        ),
    )

    # --------------------------------------------------------
    # ONLY THE SAFE FIRST JOIN EXECUTED
    # --------------------------------------------------------

    assert len(
        result.executions
    ) == 1

    first_execution = (
        result.executions[0]
    )

    assert (
        first_execution
        .validation
        .valid_for_downstream
        is True
    )

    # --------------------------------------------------------
    # SECOND RELATION EXISTS BUT IS BLOCKED
    # --------------------------------------------------------

    blocked = (
        result.final_discovery
    )

    assert blocked.has_candidate is True
    assert blocked.ready_for_approval is False

    assert blocked.intent is not None
    assert blocked.plan is not None

    assert (
        blocked.intent.keys[0].left_column
        ==
        "product_id"
    )

    assert (
        blocked.intent.expected_cardinality.value
        ==
        "many_to_many"
    )

    assert (
        blocked.plan.ready_for_approval
        is False
    )

    # --------------------------------------------------------
    # NO SECOND COMBINE ARTIFACT
    # --------------------------------------------------------

    combine_artifacts = [
        artifact
        for artifact
        in list_preparation_artifacts(
            workflow_id=session.workflow_id
        )
        if artifact.stage == "combine"
    ]

    assert len(
        combine_artifacts
    ) == 1

    # The frontier therefore still contains:
    #
    #   combined orders/customers
    #   products
    #
    assert len(
        blocked.active_dataset_ids
    ) == 2

    # --------------------------------------------------------
    # COMBINE IS SERVER-OWNED BLOCKED
    # --------------------------------------------------------

    current = get_preparation_session(
        session.workflow_id
    )

    assert (
        stage_status(
            current,
            PreparationStage.COMBINE,
        )
        ==
        "blocked"
    )


if __name__ == "__main__":
    print(
        "=== DATALENS COMBINE SEQUENCE FAIL-CLOSED v0.1 ==="
    )

    test_sequence_stops_before_blocked_many_to_many_join()

    print()
    print(
        "PASS - blocked subsequent join was not executed."
    )
