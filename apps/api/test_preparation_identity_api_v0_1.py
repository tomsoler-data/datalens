from __future__ import annotations

import pandas as pd

from fastapi import (
    FastAPI,
)

from fastapi.testclient import (
    TestClient,
)

from app.api.preparation_identity import (
    PREPARATION_IDENTITY_API_VERSION,
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
    record_validation_stage_signal,
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
# DATA
# ============================================================


def no_identity_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "city": [
                    "Paris",
                    "Paris",
                    "Lyon",
                ],

                "value": [
                    10,
                    10,
                    20,
                ],
            }
        )
    )


def reliable_identity_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "order_id": [
                    "O1",
                    "O2",
                    "O3",
                ],

                "amount": [
                    10.0,
                    20.0,
                    30.0,
                ],
            }
        )
    )


# ============================================================
# SESSION BUILDERS
# ============================================================


def reset_state() -> None:
    reset_preparation_session_store_for_tests()

    reset_preparation_artifact_store_for_tests()


def build_session(
    *,
    dataset_id: str,
    dataset_filename: str,
    dataframe: pd.DataFrame,
):
    reset_state()


    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                dataset_id,
            ]
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

            dataset_ids=[
                dataset_id,
            ],

            evidence_refs=[
                f"test:{stage.value}",
            ],

            blocking_reasons=[],
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

        dataset_ids=[
            dataset_id,
        ],

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

        dataset_ids=[
            dataset_id,
        ],

        evidence_refs=[
            "test:transform:not-required",
        ],

        blocking_reasons=[],
    )


    record_optional_stage_signal(
        workflow_id=
            session.workflow_id,

        stage=
            PreparationStage.COMBINE,

        required=
            False,

        completed=
            False,

        review_required=
            False,

        blocked=
            False,

        dataset_ids=[
            dataset_id,
        ],

        evidence_refs=[
            "test:combine:not-required",
        ],

        blocking_reasons=[],
    )


    put_preparation_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            dataset_id,

        dataset_filename=
            dataset_filename,

        stage=
            "source",

        dataframe=
            dataframe,

        evidence_refs=[
            "test:source",
        ],
    )


    return (
        get_preparation_session(
            session.workflow_id
        )
    )


# ============================================================
# 1. INSPECT WITHOUT AI
# ============================================================


def test_inspect_surrogate_candidate_without_ai(
) -> None:
    session = (
        build_session(
            dataset_id=
                "observations",

            dataset_filename=
                "observations.csv",

            dataframe=
                no_identity_frame(),
        )
    )


    response = (
        client.post(
            "/preparation/identity/inspect",

            json={
                "workflow_id":
                    session.workflow_id,

                "dataset_id":
                    "observations",

                "include_ai":
                    False,
            },
        )
    )


    print(
        "Identity inspect status:",
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
        PREPARATION_IDENTITY_API_VERSION
    )


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
            "report"
        ][
            "suggested_surrogate_column"
        ]
        ==
        "row_id"
    )


    assert (
        body[
            "surrogate_request_id"
        ]
        .startswith(
            "identity-surrogate:"
        )
    )


    assert (
        body[
            "can_create_surrogate"
        ]
        is True
    )


    assert (
        body[
            "explanation"
        ]
        is None
    )


    assert (
        body[
            "ai_error"
        ]
        is None
    )


    print(
        "Server-owned identity report and approval request generated: PASS"
    )


# ============================================================
# 2. RELIABLE KEY DOES NOT OFFER SURROGATE
# ============================================================


def test_reliable_key_does_not_offer_surrogate(
) -> None:
    session = (
        build_session(
            dataset_id=
                "orders",

            dataset_filename=
                "orders.csv",

            dataframe=
                reliable_identity_frame(),
        )
    )


    response = (
        client.post(
            "/preparation/identity/inspect",

            json={
                "workflow_id":
                    session.workflow_id,

                "dataset_id":
                    "orders",

                "include_ai":
                    False,
            },
        )
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
            "report"
        ][
            "status"
        ]
        ==
        "single_key"
    )


    assert (
        body[
            "surrogate_request_id"
        ]
        is None
    )


    assert (
        body[
            "can_create_surrogate"
        ]
        is False
    )


    print(
        "Reliable identity blocks unnecessary surrogate proposal: PASS"
    )


# ============================================================
# 3. CREATE SURROGATE
# ============================================================


def test_create_surrogate_materializes_transform_artifact(
) -> None:
    session = (
        build_session(
            dataset_id=
                "observations",

            dataset_filename=
                "observations.csv",

            dataframe=
                no_identity_frame(),
        )
    )


    original = (
        no_identity_frame()
        .copy(
            deep=True
        )
    )


    inspect_response = (
        client.post(
            "/preparation/identity/inspect",

            json={
                "workflow_id":
                    session.workflow_id,

                "dataset_id":
                    "observations",

                "include_ai":
                    False,
            },
        )
    )


    request_id = (
        inspect_response
        .json()[
            "surrogate_request_id"
        ]
    )


    response = (
        client.post(
            "/preparation/identity/create-surrogate",

            json={
                "workflow_id":
                    session.workflow_id,

                "dataset_id":
                    "observations",

                "request_id":
                    request_id,
            },
        )
    )


    print(
        "Create surrogate status:",
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
            "surrogate_column"
        ]
        ==
        "row_id"
    )


    assert (
        body[
            "parent_dataset_ids"
        ]
        ==
        [
            "observations",
        ]
    )


    output_dataset_id = (
        body[
            "output_dataset_id"
        ]
    )


    output = (
        get_preparation_dataframe(
            workflow_id=
                session.workflow_id,

            dataset_id=
                output_dataset_id,
        )
    )


    assert (
        output.columns[
            0
        ]
        ==
        "row_id"
    )


    assert (
        output[
            "row_id"
        ]
        .tolist()
        ==
        [
            1,
            2,
            3,
        ]
    )


    source = (
        get_preparation_dataframe(
            workflow_id=
                session.workflow_id,

            dataset_id=
                "observations",
        )
    )


    pd.testing.assert_frame_equal(
        source,
        original,
    )


    transform_artifacts = [
        artifact

        for artifact
        in list_preparation_artifacts(
            workflow_id=
                session.workflow_id
        )

        if artifact.stage ==
        "transform"
    ]


    assert (
        len(
            transform_artifacts
        )
        ==
        1
    )


    transform_stage = next(
        item

        for item
        in body[
            "session"
        ][
            "snapshot"
        ][
            "stages"
        ]

        if item[
            "stage"
        ]
        ==
        "transform"
    )


    assert (
        transform_stage[
            "status"
        ]
        ==
        "passed"
    )


    print(
        "Approved surrogate key materialized as non-mutating TRANSFORM artifact: PASS"
    )


# ============================================================
# 4. STALE REQUEST
# ============================================================


def test_stale_request_is_rejected(
) -> None:
    session = (
        build_session(
            dataset_id=
                "observations",

            dataset_filename=
                "observations.csv",

            dataframe=
                no_identity_frame(),
        )
    )


    response = (
        client.post(
            "/preparation/identity/create-surrogate",

            json={
                "workflow_id":
                    session.workflow_id,

                "dataset_id":
                    "observations",

                "request_id":
                    "identity-surrogate:stale",
            },
        )
    )


    assert (
        response.status_code
        ==
        409
    )


    assert not [
        artifact

        for artifact
        in list_preparation_artifacts(
            workflow_id=
                session.workflow_id
        )

        if artifact.stage ==
        "transform"
    ]


    print(
        "Stale surrogate approval rejected before mutation: PASS"
    )


# ============================================================
# 5. CLIENT CANNOT INJECT COLUMN / OUTPUT
# ============================================================


def test_client_cannot_inject_surrogate_details(
) -> None:
    session = (
        build_session(
            dataset_id=
                "observations",

            dataset_filename=
                "observations.csv",

            dataframe=
                no_identity_frame(),
        )
    )


    response = (
        client.post(
            "/preparation/identity/create-surrogate",

            json={
                "workflow_id":
                    session.workflow_id,

                "dataset_id":
                    "observations",

                "request_id":
                    "identity-surrogate:anything",

                "surrogate_column":
                    "magic_pk",

                "output_dataset_id":
                    "client:output",

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


    print(
        "Client cannot inject surrogate column, output or stage status: PASS"
    )


# ============================================================
# 6. POST-VALIDATE LOCK
# ============================================================


def test_post_validate_mutation_is_locked(
) -> None:
    session = (
        build_session(
            dataset_id=
                "observations",

            dataset_filename=
                "observations.csv",

            dataframe=
                no_identity_frame(),
        )
    )


    inspect_response = (
        client.post(
            "/preparation/identity/inspect",

            json={
                "workflow_id":
                    session.workflow_id,

                "dataset_id":
                    "observations",

                "include_ai":
                    False,
            },
        )
    )


    request_id = (
        inspect_response
        .json()[
            "surrogate_request_id"
        ]
    )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    record_validation_stage_signal(
        workflow_id=
            session.workflow_id,

        completed=
            True,

        passed=
            True,

        dataset_ids=[
            "observations",
        ],

        evidence_refs=[
            "test:validate",
        ],

        blocking_reasons=[],

        expected_revision=
            current.revision,
    )


    response = (
        client.post(
            "/preparation/identity/create-surrogate",

            json={
                "workflow_id":
                    session.workflow_id,

                "dataset_id":
                    "observations",

                "request_id":
                    request_id,
            },
        )
    )


    assert (
        response.status_code
        ==
        409
    )


    print(
        "Post-VALIDATE surrogate mutation rejected: PASS"
    )


# ============================================================
# 7. UNKNOWN DATASET
# ============================================================


def test_unknown_dataset_is_404(
) -> None:
    session = (
        build_session(
            dataset_id=
                "observations",

            dataset_filename=
                "observations.csv",

            dataframe=
                no_identity_frame(),
        )
    )


    response = (
        client.post(
            "/preparation/identity/inspect",

            json={
                "workflow_id":
                    session.workflow_id,

                "dataset_id":
                    "missing",

                "include_ai":
                    False,
            },
        )
    )


    assert (
        response.status_code
        ==
        404
    )


    print(
        "Unknown Preparation artifact rejected: PASS"
    )


# ============================================================
# 8. ROUTES
# ============================================================


def test_routes_registered(
) -> None:
    paths = (
        app.openapi()[
            "paths"
        ]
    )


    assert (
        "/preparation/identity/inspect"
        in
        paths
    )


    assert (
        "/preparation/identity/create-surrogate"
        in
        paths
    )


    print(
        "Preparation Identity API routes registered: PASS"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        "=== DATALENS PREPARATION IDENTITY API v0.1 ==="
    )

    print()


    test_inspect_surrogate_candidate_without_ai()

    test_reliable_key_does_not_offer_surrogate()

    test_create_surrogate_materializes_transform_artifact()

    test_stale_request_is_rejected()

    test_client_cannot_inject_surrogate_details()

    test_post_validate_mutation_is_locked()

    test_unknown_dataset_is_404()

    test_routes_registered()


    print()

    print(
        (
            "Preparation Identity API version: "
            f"{PREPARATION_IDENTITY_API_VERSION}"
        )
    )

    print(
        "Preparation Identity API v0.1: PASS"
    )


if __name__ == "__main__":
    main()
