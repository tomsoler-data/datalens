from __future__ import annotations

import pandas as pd

from app.preparation.dataset_identity import (
    profile_dataset_identity,
)

from app.preparation.preparation_identity_resolution import (
    build_identity_resolution_request_id,
    record_continue_without_surrogate,
    reset_preparation_identity_resolution_for_tests,
)

from app.preparation.preparation_artifact_store import (
    get_preparation_dataframe,
    list_preparation_artifacts,
    put_preparation_artifact,
    reset_preparation_artifact_store_for_tests,
)
from app.preparation.preparation_combine_service import (
    PREPARATION_COMBINE_SERVICE_VERSION,
    approve_and_execute_next_combine,
    discover_next_combine,
)
from app.preparation.preparation_session import (
    create_preparation_session,
    get_preparation_session,
    record_optional_stage_signal,
    record_required_stage_signal,
    reset_preparation_session_store_for_tests,
)
from app.preparation.preparation_workflow import PreparationStage


ORDERS = "dataset:orders"
CUSTOMERS = "dataset:customers"
PRODUCTS = "dataset:products"
ORDERS_WITH_ROW_ID = "transform:orders:row_id"


def orders_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["O1", "O2", "O3", "O4"],
            "customer_id": ["C1", "C1", "C2", "C3"],
            "product_id": ["P1", "P2", "P1", "P2"],
            "quantity": [1, 2, 1, 3],
        }
    )


def customers_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "segment": ["Premium", "Standard", "Premium"],
        }
    )


def products_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": ["P1", "P2"],
            "category": ["Electronics", "Accessories"],
        }
    )


def stage_status(session, stage: PreparationStage) -> str:
    record = next(
        item
        for item in session.snapshot.stages
        if item.stage == stage
    )
    return record.status.value


def build_ready_preparation():
    reset_preparation_session_store_for_tests()
    reset_preparation_artifact_store_for_tests()
    reset_preparation_identity_resolution_for_tests()

    root_ids = [ORDERS, CUSTOMERS, PRODUCTS]
    session = create_preparation_session(
        selected_analysis_dataset_ids=root_ids
    )

    for stage in [
        PreparationStage.IMPORT,
        PreparationStage.UNDERSTAND,
        PreparationStage.QUALITY,
    ]:
        record_required_stage_signal(
            workflow_id=session.workflow_id,
            stage=stage,
            completed=True,
            dataset_ids=root_ids,
            evidence_refs=[f"test:{stage.value}"],
            blocking_reasons=[],
        )

    put_preparation_artifact(
        workflow_id=session.workflow_id,
        dataset_id=ORDERS,
        dataset_filename="orders_test.csv",
        stage="source",
        dataframe=orders_frame(),
        evidence_refs=["test:source:orders"],
    )
    put_preparation_artifact(
        workflow_id=session.workflow_id,
        dataset_id=CUSTOMERS,
        dataset_filename="customers_test.csv",
        stage="source",
        dataframe=customers_frame(),
        evidence_refs=["test:source:customers"],
    )
    put_preparation_artifact(
        workflow_id=session.workflow_id,
        dataset_id=PRODUCTS,
        dataset_filename="products_test.csv",
        stage="source",
        dataframe=products_frame(),
        evidence_refs=["test:source:products"],
    )

    record_optional_stage_signal(
        workflow_id=session.workflow_id,
        stage=PreparationStage.CLEAN,
        required=False,
        completed=False,
        review_required=False,
        blocked=False,
        dataset_ids=root_ids,
        evidence_refs=["test:clean:not-required"],
        blocking_reasons=[],
    )
    record_optional_stage_signal(
        workflow_id=session.workflow_id,
        stage=PreparationStage.TRANSFORM,
        required=False,
        completed=False,
        review_required=False,
        blocked=False,
        dataset_ids=root_ids,
        evidence_refs=["test:transform:not-required"],
        blocking_reasons=[],
    )

    return get_preparation_session(session.workflow_id)


def test_first_join_requires_review():
    session = build_ready_preparation()
    discovery = discover_next_combine(session.workflow_id)

    assert discovery.has_candidate is True
    assert discovery.ready_for_approval is True
    assert discovery.intent is not None
    assert discovery.plan is not None
    assert discovery.intent.keys[0].left_column == "customer_id"
    assert discovery.intent.keys[0].right_column == "customer_id"
    assert discovery.intent.left_dataset_id == ORDERS
    assert discovery.intent.right_dataset_id == CUSTOMERS
    assert discovery.intent.join_type.value == "left"
    assert discovery.intent.expected_cardinality.value == "many_to_one"

    current = get_preparation_session(session.workflow_id)
    assert stage_status(current, PreparationStage.COMBINE) == "review_required"

    print("First deterministic join candidate: PASS")


def test_two_step_join_materializes_final_frontier():
    session = build_ready_preparation()
    original_orders = orders_frame()

    first = discover_next_combine(session.workflow_id)
    assert first.intent is not None

    first_execution = approve_and_execute_next_combine(
        workflow_id=session.workflow_id,
        request_id=first.intent.request_id,
        actor="test",
    )

    assert first_execution.validation.valid_for_downstream is True
    assert first_execution.rows == 4

    first_frame = get_preparation_dataframe(
        workflow_id=session.workflow_id,
        dataset_id=first_execution.output_dataset_id,
    )
    assert "segment" in first_frame.columns

    after_first = get_preparation_session(session.workflow_id)
    assert stage_status(after_first, PreparationStage.COMBINE) == "review_required"

    second = first_execution.next_discovery
    assert second.intent is not None
    assert second.ready_for_approval is True
    assert second.intent.keys[0].left_column == "product_id"
    assert second.intent.right_dataset_id == PRODUCTS

    second_execution = approve_and_execute_next_combine(
        workflow_id=session.workflow_id,
        request_id=second.intent.request_id,
        actor="test",
    )

    final_frame = get_preparation_dataframe(
        workflow_id=session.workflow_id,
        dataset_id=second_execution.output_dataset_id,
    )

    assert final_frame.shape[0] == 4
    assert "segment" in final_frame.columns
    assert "category" in final_frame.columns
    assert second_execution.next_discovery.has_candidate is False
    assert second_execution.next_discovery.active_dataset_ids == (
        second_execution.output_dataset_id,
    )

    final_session = get_preparation_session(session.workflow_id)
    assert stage_status(final_session, PreparationStage.COMBINE) == "passed"

    # Combine materializes data but does not decide the final Analysis scope.
    assert final_session.analysis_output_dataset_ids == []

    # Preparation roots remain immutable.
    assert final_session.selected_analysis_dataset_ids == [
        ORDERS,
        CUSTOMERS,
        PRODUCTS,
    ]

    stored_orders = get_preparation_dataframe(
        workflow_id=session.workflow_id,
        dataset_id=ORDERS,
    )
    pd.testing.assert_frame_equal(stored_orders, original_orders)

    combine_artifacts = [
        artifact
        for artifact in list_preparation_artifacts(
            workflow_id=session.workflow_id
        )
        if artifact.stage == "combine"
    ]
    assert len(combine_artifacts) == 2

    print("Two-step materialized combine frontier: PASS")


def test_transform_artifact_supersedes_source_on_combine_frontier():
    session = build_ready_preparation()

    transformed_orders = orders_frame().copy(
        deep=True
    )
    transformed_orders.insert(
        0,
        "row_id",
        range(
            1,
            len(transformed_orders) + 1,
        ),
    )

    put_preparation_artifact(
        workflow_id=session.workflow_id,
        dataset_id=ORDERS_WITH_ROW_ID,
        dataset_filename="orders_test__row_id.csv",
        stage="transform",
        dataframe=transformed_orders,
        parent_dataset_ids=[
            ORDERS,
        ],
        evidence_refs=[
            "test:identity-surrogate",
        ],
        replace=False,
    )

    record_optional_stage_signal(
        workflow_id=session.workflow_id,
        stage=PreparationStage.TRANSFORM,
        required=True,
        completed=True,
        review_required=False,
        blocked=False,
        dataset_ids=[
            ORDERS_WITH_ROW_ID,
        ],
        evidence_refs=[
            "test:transform:surrogate",
        ],
        blocking_reasons=[],
    )

    discovery = discover_next_combine(
        session.workflow_id
    )

    assert discovery.has_candidate is True
    assert discovery.ready_for_approval is True
    assert discovery.intent is not None

    assert ORDERS not in discovery.active_dataset_ids
    assert ORDERS_WITH_ROW_ID in discovery.active_dataset_ids
    assert CUSTOMERS in discovery.active_dataset_ids
    assert PRODUCTS in discovery.active_dataset_ids

    assert (
        discovery.intent.left_dataset_id
        ==
        ORDERS_WITH_ROW_ID
    )
    assert (
        discovery.intent.right_dataset_id
        ==
        CUSTOMERS
    )
    assert (
        discovery.intent.keys[0].left_column
        ==
        "customer_id"
    )

    first_execution = approve_and_execute_next_combine(
        workflow_id=session.workflow_id,
        request_id=discovery.intent.request_id,
        actor="test",
    )

    first_frame = get_preparation_dataframe(
        workflow_id=session.workflow_id,
        dataset_id=first_execution.output_dataset_id,
    )

    assert "row_id" in first_frame.columns
    assert "segment" in first_frame.columns

    print(
        "TRANSFORM output supersedes its source on COMBINE frontier: PASS"
    )


def test_in_place_parent_reference_does_not_consume_current_artifact():
    reset_preparation_session_store_for_tests()
    reset_preparation_artifact_store_for_tests()

    left_id = "dataset:left"
    right_id = "dataset:right"

    session = create_preparation_session(
        selected_analysis_dataset_ids=[
            left_id,
            right_id,
        ]
    )

    for stage in [
        PreparationStage.IMPORT,
        PreparationStage.UNDERSTAND,
        PreparationStage.QUALITY,
    ]:
        record_required_stage_signal(
            workflow_id=session.workflow_id,
            stage=stage,
            completed=True,
            dataset_ids=[
                left_id,
                right_id,
            ],
            evidence_refs=[
                f"test:{stage.value}"
            ],
            blocking_reasons=[],
        )

    put_preparation_artifact(
        workflow_id=session.workflow_id,
        dataset_id=left_id,
        dataset_filename="left_clean.csv",
        stage="clean",
        dataframe=pd.DataFrame(
            {
                "customer_id": [
                    "C1",
                    "C2",
                ],
                "value": [
                    10,
                    20,
                ],
            }
        ),
        parent_dataset_ids=[
            left_id,
        ],
        evidence_refs=[
            "test:clean:in-place",
        ],
        replace=True,
    )

    put_preparation_artifact(
        workflow_id=session.workflow_id,
        dataset_id=right_id,
        dataset_filename="right.csv",
        stage="source",
        dataframe=pd.DataFrame(
            {
                "customer_id": [
                    "C1",
                    "C2",
                ],
                "segment": [
                    "A",
                    "B",
                ],
            }
        ),
    )

    record_optional_stage_signal(
        workflow_id=session.workflow_id,
        stage=PreparationStage.CLEAN,
        required=True,
        completed=True,
        review_required=False,
        blocked=False,
        dataset_ids=[
            left_id,
            right_id,
        ],
        evidence_refs=[
            "test:clean",
        ],
        blocking_reasons=[],
    )

    record_optional_stage_signal(
        workflow_id=session.workflow_id,
        stage=PreparationStage.TRANSFORM,
        required=False,
        completed=False,
        review_required=False,
        blocked=False,
        dataset_ids=[
            left_id,
            right_id,
        ],
        evidence_refs=[],
        blocking_reasons=[],
    )

    discovery = discover_next_combine(
        session.workflow_id
    )

    assert discovery.has_candidate is True
    assert set(
        discovery.active_dataset_ids
    ) == {
        left_id,
        right_id,
    }

    print(
        "In-place lineage self-parent remains active: PASS"
    )


def test_stale_request_id_is_rejected():
    session = build_ready_preparation()
    discovery = discover_next_combine(session.workflow_id)
    assert discovery.intent is not None

    try:
        approve_and_execute_next_combine(
            workflow_id=session.workflow_id,
            request_id="join:stale-client-request",
            actor="test",
        )
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("Stale join request_id should be rejected.")

    assert not [
        artifact
        for artifact in list_preparation_artifacts(
            workflow_id=session.workflow_id
        )
        if artifact.stage == "combine"
    ]

    print("Stale client join approval rejected: PASS")


def test_many_to_many_candidate_fails_closed():
    reset_preparation_session_store_for_tests()
    reset_preparation_artifact_store_for_tests()
    reset_preparation_identity_resolution_for_tests()

    left_id = "dataset:left"
    right_id = "dataset:right"

    session = create_preparation_session(
        selected_analysis_dataset_ids=[left_id, right_id]
    )

    for stage in [
        PreparationStage.IMPORT,
        PreparationStage.UNDERSTAND,
        PreparationStage.QUALITY,
    ]:
        record_required_stage_signal(
            workflow_id=session.workflow_id,
            stage=stage,
            completed=True,
            dataset_ids=[left_id, right_id],
            evidence_refs=[f"test:{stage.value}"],
            blocking_reasons=[],
        )

    for dataset_id, filename, frame in [
        (
            left_id,
            "left.csv",
            pd.DataFrame({"customer_id": ["C1", "C1", "C2"]}),
        ),
        (
            right_id,
            "right.csv",
            pd.DataFrame({"customer_id": ["C1", "C1", "C2"]}),
        ),
    ]:
        put_preparation_artifact(
            workflow_id=session.workflow_id,
            dataset_id=dataset_id,
            dataset_filename=filename,
            stage="source",
            dataframe=frame,
        )

    record_optional_stage_signal(
        workflow_id=session.workflow_id,
        stage=PreparationStage.CLEAN,
        required=False,
        completed=False,
        review_required=False,
        blocked=False,
        dataset_ids=[left_id, right_id],
        evidence_refs=[],
        blocking_reasons=[],
    )
    record_optional_stage_signal(
        workflow_id=session.workflow_id,
        stage=PreparationStage.TRANSFORM,
        required=False,
        completed=False,
        review_required=False,
        blocked=False,
        dataset_ids=[left_id, right_id],
        evidence_refs=[],
        blocking_reasons=[],
    )

    # --------------------------------------------------------
    # Identity gate
    # --------------------------------------------------------
    #
    # This regression test is specifically about the JOIN
    # planner failing closed on a many-to-many relationship.
    #
    # Both fixtures intentionally have no unique row identity:
    #
    #   customer_id = C1, C1, C2
    #
    # Since Identity now runs before COMBINE, we must first
    # record the analyst's explicit decision to continue
    # without creating a surrogate key. Only then is COMBINE
    # allowed to evaluate join cardinality.
    # --------------------------------------------------------

    for (
        dataset_id,
        filename,
        frame,
    ) in [
        (
            left_id,
            "left.csv",
            pd.DataFrame(
                {
                    "customer_id": [
                        "C1",
                        "C1",
                        "C2",
                    ]
                }
            ),
        ),
        (
            right_id,
            "right.csv",
            pd.DataFrame(
                {
                    "customer_id": [
                        "C1",
                        "C1",
                        "C2",
                    ]
                }
            ),
        ),
    ]:
        report = profile_dataset_identity(
            frame,
            dataset_id=dataset_id,
            dataset_filename=filename,
        )

        assert (
            report.status
            ==
            "surrogate_recommended"
        )

        request_id = (
            build_identity_resolution_request_id(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                dataset_filename=
                    filename,

                artifact_stage=
                    "source",

                report=
                    report,
            )
        )

        resolution = (
            record_continue_without_surrogate(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                dataset_filename=
                    filename,

                artifact_stage=
                    "source",

                report=
                    report,

                request_id=
                    request_id,
            )
        )

        assert (
            resolution.kind
            ==
            "continued_without_surrogate"
        )

    discovery = discover_next_combine(
        session.workflow_id
    )

    assert discovery.has_candidate is True
    assert discovery.ready_for_approval is False

    current = get_preparation_session(session.workflow_id)
    assert stage_status(current, PreparationStage.COMBINE) == "blocked"

    print("Many-to-many automatic join blocked: PASS")


def main():
    print("=== DATALENS PREPARATION COMBINE SERVICE v0.3 ===")
    print()

    test_first_join_requires_review()
    test_two_step_join_materializes_final_frontier()
    test_transform_artifact_supersedes_source_on_combine_frontier()
    test_in_place_parent_reference_does_not_consume_current_artifact()
    test_stale_request_id_is_rejected()
    test_many_to_many_candidate_fails_closed()

    print()
    print(
        "Preparation Combine Service version:",
        PREPARATION_COMBINE_SERVICE_VERSION,
    )
    print("Preparation Combine Service v0.3: PASS")


if __name__ == "__main__":
    main()
