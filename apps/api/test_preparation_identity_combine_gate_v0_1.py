from __future__ import annotations

import pandas as pd

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.preparation_identity import (
    router as preparation_identity_router,
)

from app.preparation.preparation_artifact_store import (
    put_preparation_artifact,
    reset_preparation_artifact_store_for_tests,
)

from app.preparation.preparation_combine_service import (
    CombineIdentityResolutionRequiredError,
    discover_next_combine,
)

from app.preparation.preparation_identity_resolution import (
    reset_preparation_identity_resolution_for_tests,
)

from app.preparation.preparation_session import (
    create_preparation_session,
    record_optional_stage_signal,
    record_required_stage_signal,
    reset_preparation_session_store_for_tests,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
)


app = FastAPI()
app.include_router(
    preparation_identity_router
)

client = TestClient(
    app
)


def reset_state() -> None:
    reset_preparation_session_store_for_tests()
    reset_preparation_artifact_store_for_tests()
    reset_preparation_identity_resolution_for_tests()


def build_session():
    reset_state()

    orders_id = "dataset:orders"
    customers_id = "dataset:customers"

    session = create_preparation_session(
        selected_analysis_dataset_ids=[
            orders_id,
            customers_id,
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
                orders_id,
                customers_id,
            ],
            evidence_refs=[
                f"test:{stage.value}",
            ],
            blocking_reasons=[],
        )

    record_optional_stage_signal(
        workflow_id=session.workflow_id,
        stage=PreparationStage.CLEAN,
        required=False,
        completed=False,
        review_required=False,
        blocked=False,
        dataset_ids=[
            orders_id,
            customers_id,
        ],
        evidence_refs=[
            "test:clean:not-required",
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
            orders_id,
            customers_id,
        ],
        evidence_refs=[
            "test:transform:not-required",
        ],
        blocking_reasons=[],
    )

    put_preparation_artifact(
        workflow_id=session.workflow_id,
        dataset_id=orders_id,
        dataset_filename="orders.csv",
        stage="source",
        dataframe=pd.DataFrame(
            {
                "customer_id": [
                    "C1",
                    "C1",
                    "C2",
                ],
                "amount": [
                    10.0,
                    10.0,
                    20.0,
                ],
            }
        ),
    )

    put_preparation_artifact(
        workflow_id=session.workflow_id,
        dataset_id=customers_id,
        dataset_filename="customers.csv",
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

    return session


def test_combine_is_blocked_before_identity_resolution(
) -> None:
    session = build_session()

    try:
        discover_next_combine(
            session.workflow_id
        )

    except CombineIdentityResolutionRequiredError as error:
        assert "orders.csv" in str(
            error
        )

    else:
        raise AssertionError(
            "COMBINE should be blocked before identity resolution."
        )

    print(
        "COMBINE blocked before identity resolution: PASS"
    )


def test_explicit_continue_unlocks_combine(
) -> None:
    session = build_session()

    inspect = client.post(
        "/preparation/identity/inspect",
        json={
            "workflow_id":
                session.workflow_id,

            "dataset_id":
                "dataset:orders",

            "include_ai":
                False,
        },
    )

    assert inspect.status_code == 200

    body = inspect.json()

    assert (
        body[
            "report"
        ][
            "status"
        ]
        ==
        "surrogate_recommended"
    )

    assert (
        body[
            "identity_resolved"
        ]
        is False
    )

    request_id = body[
        "surrogate_request_id"
    ]

    continued = client.post(
        "/preparation/identity/continue",
        json={
            "workflow_id":
                session.workflow_id,

            "dataset_id":
                "dataset:orders",

            "request_id":
                request_id,
        },
    )

    assert continued.status_code == 200
    assert (
        continued.json()[
            "resolution_kind"
        ]
        ==
        "continued_without_surrogate"
    )

    rediscovery = discover_next_combine(
        session.workflow_id
    )

    assert rediscovery.has_candidate is True
    assert rediscovery.intent is not None
    assert (
        rediscovery.intent.keys[
            0
        ].left_column
        ==
        "customer_id"
    )

    print(
        "Explicit continue-without-surrogate unlocks COMBINE: PASS"
    )


def test_stale_identity_continue_is_rejected(
) -> None:
    session = build_session()

    response = client.post(
        "/preparation/identity/continue",
        json={
            "workflow_id":
                session.workflow_id,

            "dataset_id":
                "dataset:orders",

            "request_id":
                "identity-surrogate:stale",
        },
    )

    assert response.status_code == 409

    print(
        "Stale identity continuation rejected: PASS"
    )


def test_reliable_key_is_automatically_resolved(
) -> None:
    session = build_session()

    response = client.post(
        "/preparation/identity/inspect",
        json={
            "workflow_id":
                session.workflow_id,

            "dataset_id":
                "dataset:customers",

            "include_ai":
                False,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body[
        "report"
    ][
        "status"
    ] == "single_key"

    assert (
        body[
            "identity_resolved"
        ]
        is True
    )

    assert (
        body[
            "resolution_kind"
        ]
        ==
        "detected_key"
    )

    print(
        "Reliable Python key auto-resolves identity: PASS"
    )


def test_continue_route_registered(
) -> None:
    paths = app.openapi()[
        "paths"
    ]

    assert (
        "/preparation/identity/continue"
        in paths
    )

    print(
        "Identity continue route registered: PASS"
    )


def main() -> None:
    print(
        "=== DATALENS IDENTITY -> COMBINE GATE v0.1 ==="
    )

    print()

    test_combine_is_blocked_before_identity_resolution()
    test_explicit_continue_unlocks_combine()
    test_stale_identity_continue_is_rejected()
    test_reliable_key_is_automatically_resolved()
    test_continue_route_registered()

    print()

    print(
        "Identity -> COMBINE Gate v0.1: PASS"
    )


if __name__ == "__main__":
    main()
