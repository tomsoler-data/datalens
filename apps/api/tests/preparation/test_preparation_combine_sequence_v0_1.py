from __future__ import annotations

from app.preparation import (
    preparation_combine_service
    as combine_service,
)

from app.preparation.preparation_artifact_store import (
    get_preparation_dataframe,
    list_preparation_artifacts,
)

from app.preparation.preparation_session import (
    get_preparation_session,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
)

from tests.preparation.test_preparation_combine_service_v0_1 import (
    build_ready_preparation,
    stage_status,
)


def test_one_authorization_executes_full_safe_combine_sequence() -> None:
    """
    One explicit analyst authorization may cover a bounded sequence
    of server-derived safe joins.

    The sequence must NOT bypass any existing authority:

        discover
        -> Join Planner
        -> approval
        -> execution
        -> post-join validation
        -> rediscover

    until no additional safe relationship remains.
    """

    sequence_fn = getattr(
        combine_service,
        "approve_and_execute_combine_sequence",
        None,
    )

    assert callable(sequence_fn), (
        "RED CONTRACT: "
        "approve_and_execute_combine_sequence() "
        "does not exist yet."
    )

    session = build_ready_preparation()

    first = combine_service.discover_next_combine(
        session.workflow_id
    )

    assert first.intent is not None
    assert first.ready_for_approval is True

    result = sequence_fn(
        workflow_id=session.workflow_id,
        request_id=first.intent.request_id,
        actor="test",
        comment=(
            "Approve the bounded server-derived "
            "combine sequence."
        ),
    )

    # --------------------------------------------------------
    # TWO SAFE JOINS
    # --------------------------------------------------------

    assert len(result.executions) == 2

    first_execution = result.executions[0]
    second_execution = result.executions[1]

    assert (
        first_execution
        .validation
        .valid_for_downstream
        is True
    )

    assert (
        second_execution
        .validation
        .valid_for_downstream
        is True
    )

    # --------------------------------------------------------
    # SERVER-DERIVED ORDER
    # --------------------------------------------------------

    assert (
        first_execution
        .parent_dataset_ids
        ==
        (
            "dataset:orders",
            "dataset:customers",
        )
    )

    assert (
        "dataset:products"
        in
        second_execution
        .parent_dataset_ids
    )

    # --------------------------------------------------------
    # FINAL MATERIALIZED FRONTIER
    # --------------------------------------------------------

    assert (
        result
        .final_discovery
        .has_candidate
        is False
    )

    assert len(
        result
        .final_discovery
        .active_dataset_ids
    ) == 1

    final_dataset_id = (
        result
        .final_discovery
        .active_dataset_ids[
            0
        ]
    )

    final_frame = get_preparation_dataframe(
        workflow_id=session.workflow_id,
        dataset_id=final_dataset_id,
    )

    assert final_frame.shape[0] == 4
    assert "segment" in final_frame.columns
    assert "category" in final_frame.columns

    # --------------------------------------------------------
    # EXACTLY TWO COMBINE MATERIALIZATIONS
    # --------------------------------------------------------

    combine_artifacts = [
        artifact
        for artifact
        in list_preparation_artifacts(
            workflow_id=session.workflow_id
        )
        if artifact.stage == "combine"
    ]

    assert len(combine_artifacts) == 2

    # --------------------------------------------------------
    # SERVER-OWNED STAGE COMPLETES
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
        "passed"
    )


if __name__ == "__main__":
    print(
        "=== DATALENS COMBINE SEQUENCE CONTRACT v0.1 ==="
    )

    test_one_authorization_executes_full_safe_combine_sequence()

    print()
    print(
        "PASS - bounded server-owned combine sequence."
    )
