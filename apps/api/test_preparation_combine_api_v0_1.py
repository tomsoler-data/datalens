from __future__ import annotations

import pandas as pd

from fastapi import (
    FastAPI,
)

from fastapi.testclient import (
    TestClient,
)

from app.api.preparation_combine import (
    PREPARATION_COMBINE_API_VERSION,
    router,
)
from app.preparation.preparation_artifact_store import (
    get_preparation_dataframe,
    list_preparation_artifacts,
    put_preparation_artifact,
    reset_preparation_artifact_store_for_tests,
)
from app.preparation.preparation_session import (
    create_preparation_session,
    get_preparation_session,
    record_optional_stage_signal,
    record_required_stage_signal,
    reset_preparation_session_store_for_tests,
)
from app.preparation.preparation_workflow import (
    PreparationStage,
)


# ============================================================
# ISOLATED API
# ============================================================


app = FastAPI()

app.include_router(
    router
)

client = TestClient(
    app
)


# ============================================================
# IDS
# ============================================================


ORDERS = (
    "dataset:orders"
)

CUSTOMERS = (
    "dataset:customers"
)

PRODUCTS = (
    "dataset:products"
)


# ============================================================
# DATA
# ============================================================


def orders_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "order_id": [
                    "O1",
                    "O2",
                    "O3",
                    "O4",
                ],

                "customer_id": [
                    "C1",
                    "C1",
                    "C2",
                    "C3",
                ],

                "product_id": [
                    "P1",
                    "P2",
                    "P1",
                    "P2",
                ],

                "quantity": [
                    1,
                    2,
                    1,
                    3,
                ],
            }
        )
    )


def customers_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "customer_id": [
                    "C1",
                    "C2",
                    "C3",
                ],

                "segment": [
                    "Premium",
                    "Standard",
                    "Premium",
                ],
            }
        )
    )


def products_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "product_id": [
                    "P1",
                    "P2",
                ],

                "category": [
                    "Electronics",
                    "Accessories",
                ],
            }
        )
    )


# ============================================================
# HELPERS
# ============================================================


def stage_status(
    workflow_id: str,
    stage: PreparationStage,
) -> str:
    session = (
        get_preparation_session(
            workflow_id
        )
    )

    record = next(
        item

        for item
        in session.snapshot.stages

        if item.stage ==
        stage
    )

    return (
        record.status.value
    )


def build_ready_preparation():
    reset_preparation_session_store_for_tests()

    reset_preparation_artifact_store_for_tests()


    root_ids = [
        ORDERS,
        CUSTOMERS,
        PRODUCTS,
    ]


    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=
                root_ids
        )
    )


    for stage in [
        PreparationStage.IMPORT,
        PreparationStage.UNDERSTAND,
        PreparationStage.QUALITY,
    ]:
        record_required_stage_signal(
            workflow_id=
                session.workflow_id,

            stage=
                stage,

            completed=
                True,

            dataset_ids=
                root_ids,

            evidence_refs=[
                f"test:{stage.value}",
            ],

            blocking_reasons=[],
        )


    put_preparation_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            ORDERS,

        dataset_filename=
            "orders_test.csv",

        stage=
            "source",

        dataframe=
            orders_frame(),

        evidence_refs=[
            "test:source:orders",
        ],
    )


    put_preparation_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            CUSTOMERS,

        dataset_filename=
            "customers_test.csv",

        stage=
            "source",

        dataframe=
            customers_frame(),

        evidence_refs=[
            "test:source:customers",
        ],
    )


    put_preparation_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            PRODUCTS,

        dataset_filename=
            "products_test.csv",

        stage=
            "source",

        dataframe=
            products_frame(),

        evidence_refs=[
            "test:source:products",
        ],
    )


    record_optional_stage_signal(
        workflow_id=
            session.workflow_id,

        stage=
            PreparationStage.CLEAN,

        required=
            False,

        completed=
            False,

        review_required=
            False,

        blocked=
            False,

        dataset_ids=
            root_ids,

        evidence_refs=[
            "test:clean:not-required",
        ],

        blocking_reasons=[],
    )


    record_optional_stage_signal(
        workflow_id=
            session.workflow_id,

        stage=
            PreparationStage.TRANSFORM,

        required=
            False,

        completed=
            False,

        review_required=
            False,

        blocked=
            False,

        dataset_ids=
            root_ids,

        evidence_refs=[
            "test:transform:not-required",
        ],

        blocking_reasons=[],
    )


    return (
        get_preparation_session(
            session.workflow_id
        )
    )


# ============================================================
# 1. DISCOVERY
# ============================================================


def test_discovery_exposes_server_candidate(
) -> None:
    session = (
        build_ready_preparation()
    )


    response = (
        client.post(
            "/preparation/combine/discover",

            json={
                "workflow_id":
                    session.workflow_id,
            },
        )
    )


    print(
        "Combine discovery status:",
        response.status_code,
    )


    assert (
        response.status_code
        ==
        200
    )


    body = (
        response.json()
    )


    assert (
        body[
            "api_version"
        ]
        ==
        PREPARATION_COMBINE_API_VERSION
    )


    discovery = (
        body[
            "discovery"
        ]
    )


    assert (
        discovery[
            "has_candidate"
        ]
        is True
    )


    assert (
        discovery[
            "ready_for_approval"
        ]
        is True
    )


    intent = (
        discovery[
            "intent"
        ]
    )


    assert (
        intent[
            "left_dataset_id"
        ]
        ==
        ORDERS
    )


    assert (
        intent[
            "right_dataset_id"
        ]
        ==
        CUSTOMERS
    )


    assert (
        intent[
            "join_type"
        ]
        ==
        "left"
    )


    assert (
        intent[
            "expected_cardinality"
        ]
        ==
        "many_to_one"
    )


    assert (
        intent[
            "keys"
        ][
            0
        ][
            "left_column"
        ]
        ==
        "customer_id"
    )


    assert (
        stage_status(
            session.workflow_id,
            PreparationStage.COMBINE,
        )
        ==
        "review_required"
    )


    print(
        "Server-derived join candidate exposed: PASS"
    )


# ============================================================
# 2. FIRST APPROVAL
# ============================================================


def test_first_approval_materializes_and_returns_next(
) -> None:
    session = (
        build_ready_preparation()
    )


    discovery_response = (
        client.post(
            "/preparation/combine/discover",

            json={
                "workflow_id":
                    session.workflow_id,
            },
        )
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


    response = (
        client.post(
            "/preparation/combine/approve",

            json={
                "workflow_id":
                    session.workflow_id,

                "request_id":
                    request_id,

                "comment":
                    "Jointure client approuvée.",
            },
        )
    )


    print(
        "First combine approval status:",
        response.status_code,
    )


    assert (
        response.status_code
        ==
        200
    )


    body = (
        response.json()
    )


    assert (
        body[
            "validation"
        ][
            "valid_for_downstream"
        ]
        is True
    )


    output_id = (
        body[
            "output_dataset_id"
        ]
    )


    frame = (
        get_preparation_dataframe(
            workflow_id=
                session.workflow_id,

            dataset_id=
                output_id,
        )
    )


    assert (
        "segment"
        in
        frame.columns
    )


    next_discovery = (
        body[
            "next_discovery"
        ]
    )


    assert (
        next_discovery[
            "has_candidate"
        ]
        is True
    )


    assert (
        next_discovery[
            "intent"
        ][
            "right_dataset_id"
        ]
        ==
        PRODUCTS
    )


    assert (
        next_discovery[
            "intent"
        ][
            "keys"
        ][
            0
        ][
            "left_column"
        ]
        ==
        "product_id"
    )


    assert (
        body[
            "session"
        ][
            "snapshot"
        ][
            "stages"
        ]
    )


    assert (
        stage_status(
            session.workflow_id,
            PreparationStage.COMBINE,
        )
        ==
        "review_required"
    )


    print(
        "First approved join materialized; next candidate returned: PASS"
    )


# ============================================================
# 3. SECOND APPROVAL
# ============================================================


def test_second_approval_completes_combine(
) -> None:
    session = (
        build_ready_preparation()
    )


    first_discovery = (
        client.post(
            "/preparation/combine/discover",

            json={
                "workflow_id":
                    session.workflow_id,
            },
        )
        .json()[
            "discovery"
        ]
    )


    first_execution = (
        client.post(
            "/preparation/combine/approve",

            json={
                "workflow_id":
                    session.workflow_id,

                "request_id":
                    first_discovery[
                        "intent"
                    ][
                        "request_id"
                    ],
            },
        )
    )


    assert (
        first_execution.status_code
        ==
        200
    )


    second_discovery = (
        first_execution
        .json()[
            "next_discovery"
        ]
    )


    response = (
        client.post(
            "/preparation/combine/approve",

            json={
                "workflow_id":
                    session.workflow_id,

                "request_id":
                    second_discovery[
                        "intent"
                    ][
                        "request_id"
                    ],
            },
        )
    )


    print(
        "Second combine approval status:",
        response.status_code,
    )


    assert (
        response.status_code
        ==
        200
    )


    body = (
        response.json()
    )


    assert (
        body[
            "next_discovery"
        ][
            "has_candidate"
        ]
        is False
    )


    final_id = (
        body[
            "output_dataset_id"
        ]
    )


    final_frame = (
        get_preparation_dataframe(
            workflow_id=
                session.workflow_id,

            dataset_id=
                final_id,
        )
    )


    assert (
        "segment"
        in
        final_frame.columns
    )


    assert (
        "category"
        in
        final_frame.columns
    )


    assert (
        stage_status(
            session.workflow_id,
            PreparationStage.COMBINE,
        )
        ==
        "passed"
    )


    final_session = (
        get_preparation_session(
            session.workflow_id
        )
    )


    assert (
        final_session
        .analysis_output_dataset_ids
        ==
        []
    )


    assert (
        final_session
        .selected_analysis_dataset_ids
        ==
        [
            ORDERS,
            CUSTOMERS,
            PRODUCTS,
        ]
    )


    combine_artifacts = [
        artifact

        for artifact
        in list_preparation_artifacts(
            workflow_id=
                session.workflow_id
        )

        if (
            artifact.stage
            ==
            "combine"
        )
    ]


    assert (
        len(
            combine_artifacts
        )
        ==
        2
    )


    print(
        "Second approved join completes COMBINE without auto-selecting Analysis output: PASS"
    )


# ============================================================
# 4. STALE REQUEST
# ============================================================


def test_stale_request_is_rejected(
) -> None:
    session = (
        build_ready_preparation()
    )


    response = (
        client.post(
            "/preparation/combine/approve",

            json={
                "workflow_id":
                    session.workflow_id,

                "request_id":
                    "join:stale-client-request",
            },
        )
    )


    print(
        "Stale request status:",
        response.status_code,
    )


    assert (
        response.status_code
        ==
        409
    )


    detail = (
        response.json()[
            "detail"
        ]
    )


    assert (
        detail[
            "error"
        ]
        ==
        "preparation_combine_approval_rejected"
    )


    assert not [
        artifact

        for artifact
        in list_preparation_artifacts(
            workflow_id=
                session.workflow_id
        )

        if (
            artifact.stage
            ==
            "combine"
        )
    ]


    print(
        "Stale browser approval rejected: PASS"
    )


# ============================================================
# 5. CLIENT CANNOT INJECT PLAN
# ============================================================


def test_client_cannot_inject_join_plan(
) -> None:
    session = (
        build_ready_preparation()
    )


    response = (
        client.post(
            "/preparation/combine/discover",

            json={
                "workflow_id":
                    session.workflow_id,

                "join_type":
                    "outer",

                "left_dataset_id":
                    PRODUCTS,

                "status":
                    "passed",
            },
        )
    )


    assert (
        response.status_code
        ==
        422
    )


    response = (
        client.post(
            "/preparation/combine/approve",

            json={
                "workflow_id":
                    session.workflow_id,

                "request_id":
                    "join:fake",

                "keys": [
                    {
                        "left_column":
                            "anything",

                        "right_column":
                            "anything",
                    }
                ],

                "ready_for_analysis":
                    True,
            },
        )
    )


    assert (
        response.status_code
        ==
        422
    )


    print(
        "Client cannot inject join plan or Preparation state: PASS"
    )


# ============================================================
# 6. UNKNOWN SESSION
# ============================================================


def test_unknown_session_is_404(
) -> None:
    response = (
        client.post(
            "/preparation/combine/discover",

            json={
                "workflow_id":
                    "prep:does-not-exist",
            },
        )
    )


    assert (
        response.status_code
        ==
        404
    )


    print(
        "Unknown Preparation session rejected: PASS"
    )


# ============================================================
# 7. ROUTES
# ============================================================


def test_routes_are_registered(
) -> None:
    paths = (
        app.openapi()[
            "paths"
        ]
    )


    assert (
        "/preparation/combine/discover"
        in
        paths
    )


    assert (
        "/preparation/combine/approve"
        in
        paths
    )


    print(
        "Combine API routes registered: PASS"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        "=== DATALENS PREPARATION COMBINE API v0.1 ==="
    )

    print()


    test_discovery_exposes_server_candidate()

    test_first_approval_materializes_and_returns_next()

    test_second_approval_completes_combine()

    test_stale_request_is_rejected()

    test_client_cannot_inject_join_plan()

    test_unknown_session_is_404()

    test_routes_are_registered()


    print()

    print(
        (
            "Preparation Combine API version: "
            f"{PREPARATION_COMBINE_API_VERSION}"
        )
    )

    print(
        "Preparation Combine API v0.1: PASS"
    )


if __name__ == "__main__":
    main()
