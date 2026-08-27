from __future__ import annotations

import pandas as pd

from fastapi import (
    FastAPI,
)

from fastapi.testclient import (
    TestClient,
)


from app.api.preparation_identity import (
    router,
)

from app.preparation.preparation_artifact_store import (
    list_preparation_artifacts,
    put_preparation_artifact,
    reset_preparation_artifact_store_for_tests,
)

from app.preparation.preparation_identity_resolution import (
    reset_preparation_identity_resolution_for_tests,
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


# ========================================================
# API
# ========================================================


app = FastAPI()

app.include_router(
    router
)

client = TestClient(
    app
)


DATASET_ID = (
    "dataset:identity-clean-gate"
)


# ========================================================
# HELPERS
# ========================================================


def reset_state() -> None:
    reset_preparation_session_store_for_tests()

    reset_preparation_artifact_store_for_tests()

    reset_preparation_identity_resolution_for_tests()


def dataframe() -> pd.DataFrame:
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
        if item.stage == stage
    )

    return (
        record.status.value
    )


def build_session(
    clean_status: str,
):
    reset_state()


    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                DATASET_ID,
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
                DATASET_ID,
            ],

            evidence_refs=[
                f"test:{stage.value}",
            ],

            blocking_reasons=[],
        )


    if clean_status == "skipped":
        clean_signal = {
            "required":
                False,

            "completed":
                False,

            "review_required":
                False,

            "blocked":
                False,
        }

    elif clean_status == "passed":
        clean_signal = {
            "required":
                True,

            "completed":
                True,

            "review_required":
                False,

            "blocked":
                False,
        }

    elif clean_status == "not_started":
        clean_signal = {
            "required":
                True,

            "completed":
                False,

            "review_required":
                False,

            "blocked":
                False,
        }

    elif clean_status == "review_required":
        clean_signal = {
            "required":
                True,

            "completed":
                False,

            "review_required":
                True,

            "blocked":
                False,
        }

    elif clean_status == "blocked":
        clean_signal = {
            "required":
                True,

            "completed":
                False,

            "review_required":
                False,

            "blocked":
                True,
        }

    else:
        raise AssertionError(
            (
                "Unsupported CLEAN fixture status: "
                f"{clean_status}"
            )
        )


    record_optional_stage_signal(
        workflow_id=
            session.workflow_id,

        stage=
            PreparationStage.CLEAN,

        dataset_ids=[
            DATASET_ID,
        ],

        evidence_refs=[
            f"test:clean:{clean_status}",
        ],

        blocking_reasons=(
            [
                "test CLEAN blocker"
            ]
            if clean_status
            ==
            "blocked"
            else []
        ),

        **clean_signal,
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
            DATASET_ID,
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
            DATASET_ID,
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
            DATASET_ID,

        dataset_filename=
            "identity_clean_gate.csv",

        stage=
            "source",

        dataframe=
            dataframe(),

        evidence_refs=[
            "test:source",
        ],
    )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    actual_clean_status = (
        stage_status(
            session.workflow_id,
            PreparationStage.CLEAN,
        )
    )


    assert (
        actual_clean_status
        ==
        clean_status
    ), (
        actual_clean_status,
        clean_status,
    )


    return current


def inspect(
    workflow_id: str,
):
    response = client.post(
        "/preparation/identity/inspect",

        json={
            "workflow_id":
                workflow_id,

            "dataset_id":
                DATASET_ID,

            "include_ai":
                False,
        },
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
        "surrogate_recommended"
    )


    assert (
        body[
            "surrogate_request_id"
        ]
        is not None
    )


    return body


def transform_artifacts(
    workflow_id: str,
):
    return [
        artifact

        for artifact
        in list_preparation_artifacts(
            workflow_id=
                workflow_id
        )

        if artifact.stage
        ==
        "transform"
    ]


# ========================================================
# 1. UNRESOLVED CLEAN LOCKS IDENTITY MUTATION
# ========================================================


def test_unresolved_clean_statuses_lock_identity(
) -> None:
    for clean_status in [
        "not_started",
        "review_required",
        "blocked",
    ]:
        session = build_session(
            clean_status
        )


        body = inspect(
            session.workflow_id
        )


        assert (
            body[
                "mutation_locked"
            ]
            is True
        )


        assert (
            body[
                "can_create_surrogate"
            ]
            is False
        )


        assert (
            body[
                "can_continue_without_surrogate"
            ]
            is False
        )


        reason = (
            body[
                "mutation_lock_reason"
            ]
        )


        assert reason is not None

        assert (
            "CLEAN"
            in reason
        )

        assert (
            clean_status
            in reason
        )


        print(
            (
                "[PASS] Identity locked for "
                f"CLEAN={clean_status}"
            )
        )


# ========================================================
# 2. FAIL CLOSED BEFORE TRANSFORM ARTIFACT WRITE
# ========================================================


def test_surrogate_fails_before_artifact_write(
) -> None:
    for clean_status in [
        "not_started",
        "review_required",
        "blocked",
    ]:
        session = build_session(
            clean_status
        )


        body = inspect(
            session.workflow_id
        )


        request_id = (
            body[
                "surrogate_request_id"
            ]
        )


        response = client.post(
            "/preparation/identity/create-surrogate",

            json={
                "workflow_id":
                    session.workflow_id,

                "dataset_id":
                    DATASET_ID,

                "request_id":
                    request_id,
            },
        )


        assert (
            response.status_code
            ==
            409
        )


        assert (
            transform_artifacts(
                session.workflow_id
            )
            ==
            []
        )


        assert (
            stage_status(
                session.workflow_id,
                PreparationStage.TRANSFORM,
            )
            !=
            "passed"
        )


        print(
            (
                "[PASS] No TRANSFORM artifact for "
                f"CLEAN={clean_status}"
            )
        )


# ========================================================
# 3. CONTINUE ALSO RESPECTS CLEAN WINDOW
# ========================================================


def test_continue_is_locked_until_clean_resolved(
) -> None:
    session = build_session(
        "review_required"
    )


    body = inspect(
        session.workflow_id
    )


    response = client.post(
        "/preparation/identity/continue",

        json={
            "workflow_id":
                session.workflow_id,

            "dataset_id":
                DATASET_ID,

            "request_id":
                body[
                    "surrogate_request_id"
                ],
        },
    )


    assert (
        response.status_code
        ==
        409
    )


    print(
        "[PASS] continue-without-surrogate obeys CLEAN gate"
    )


# ========================================================
# 4. PASSED AND SKIPPED REMAIN VALID
# ========================================================


def test_resolved_clean_statuses_unlock_identity(
) -> None:
    for clean_status in [
        "passed",
        "skipped",
    ]:
        session = build_session(
            clean_status
        )


        body = inspect(
            session.workflow_id
        )


        assert (
            body[
                "mutation_locked"
            ]
            is False
        )


        assert (
            body[
                "can_create_surrogate"
            ]
            is True
        )


        assert (
            body[
                "can_continue_without_surrogate"
            ]
            is True
        )


        print(
            (
                "[PASS] Identity unlocked for "
                f"CLEAN={clean_status}"
            )
        )


# ========================================================
# MAIN
# ========================================================


def main() -> None:
    print()
    print(
        "=== DATALENS IDENTITY CLEAN GATE v0.1 ==="
    )
    print()


    test_unresolved_clean_statuses_lock_identity()

    test_surrogate_fails_before_artifact_write()

    test_continue_is_locked_until_clean_resolved()

    test_resolved_clean_statuses_unlock_identity()


    print()
    print(
        "PASS - Identity CLEAN Gate v0.1"
    )


if __name__ == "__main__":
    main()
