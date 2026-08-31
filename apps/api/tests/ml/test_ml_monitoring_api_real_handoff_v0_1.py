from __future__ import annotations


from fastapi.testclient import (
    TestClient,
)


from app.main import (
    app,
)


from app.ml.drift_evaluation_store import (
    list_ml_drift_evaluations_for_model,
)


from tests.ml.test_ml_monitoring_service_real_handoff_v0_1 import (
    build_ready_preparation_workflow,
    isolated_real_handoff_environment,
    persist_model_and_profile,
)


# ============================================================
# ROUTE REGISTRATION
# ============================================================


def test_monitoring_route_is_registered(
) -> None:

    # Use FastAPI's generated public contract instead of
    # assuming every internal app.routes entry exposes .path.
    #
    # Recent Starlette / FastAPI versions may include internal
    # router objects such as _IncludedRouter in app.routes.
    openapi_schema = (
        app.openapi()
    )


    paths = (
        openapi_schema.get(
            "paths",
            {}
        )
    )


    assert (
        "/ml-monitoring/evaluate"
        in
        paths
    )


    endpoint = (
        paths[
            "/ml-monitoring/evaluate"
        ]
    )


    assert (
        "post"
        in
        endpoint
    )


# ============================================================
# REAL HTTP -> REAL CONTROL PLANE
# ============================================================


def test_http_monitoring_real_handoff_end_to_end(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            dataframe,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            artifact,
            profile,
        ) = (
            persist_model_and_profile(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                preparation_revision=
                    session.revision,

                dataframe=
                    dataframe,
            )
        )


        # ====================================================
        # PUBLIC REQUEST
        #
        # The HTTP caller supplies identities only.
        # ====================================================


        request_payload = {
            "workflow_id":
                session.workflow_id,

            "model_id":
                artifact.model_id,

            "observed_dataset_id":
                dataset_id,
        }


        assert (
            set(
                request_payload
            )
            ==
            {
                "workflow_id",
                "model_id",
                "observed_dataset_id",
            }
        )


        with TestClient(
            app
        ) as client:

            response = (
                client.post(
                    "/ml-monitoring/evaluate",

                    json=
                        request_payload,
                )
            )


        assert (
            response.status_code
            ==
            200
        )


        payload = (
            response.json()
        )


        # ====================================================
        # SERVER-OWNED IDENTITY BINDINGS
        # ====================================================


        assert (
            payload[
                "workflow_id"
            ]
            ==
            session.workflow_id
        )


        assert (
            payload[
                "model_id"
            ]
            ==
            artifact.model_id
        )


        assert (
            payload[
                "profile_id"
            ]
            ==
            profile.profile_id
        )


        assert (
            payload[
                "reference_dataset_id"
            ]
            ==
            dataset_id
        )


        assert (
            payload[
                "observed_dataset_id"
            ]
            ==
            dataset_id
        )


        # ====================================================
        # TRAINING / OBSERVED SNAPSHOT BINDINGS
        # ====================================================


        assert (
            payload[
                "preparation_session_revision"
            ]
            ==
            session.revision
        )


        assert (
            payload[
                "observed_preparation_session_revision"
            ]
            ==
            session.revision
        )


        # ====================================================
        # EXACT MODEL FEATURE SURFACE
        # ====================================================


        feature_names = [
            item[
                "feature_name"
            ]

            for item
            in payload[
                "feature_results"
            ]
        ]


        assert (
            feature_names
            ==
            [
                "age",
                "segment",
            ]
        )


        # Extra dataset columns must never leak into Drift
        # evidence.
        serialized = str(
            payload
        )


        assert (
            "business_note"
            not in
            serialized
        )


        assert (
            "not-monitored"
            not in
            serialized
        )


        # The observed dataset is identical to the reference
        # feature distribution in this integration fixture.
        assert (
            payload[
                "overall_status"
            ]
            ==
            "ok"
        )


        # ====================================================
        # DURABLE HISTORY
        # ====================================================


        history = (
            list_ml_drift_evaluations_for_model(
                model_id=
                    artifact.model_id,

                workflow_id=
                    session.workflow_id,
            )
        )


        assert (
            len(
                history
            )
            ==
            1
        )


        persisted = (
            history[
                0
            ]
        )


        assert (
            persisted.evaluation_id
            ==
            payload[
                "evaluation_id"
            ]
        )


        assert (
            persisted.model_id
            ==
            artifact.model_id
        )


        assert (
            persisted.profile_id
            ==
            profile.profile_id
        )


        assert (
            persisted
            .observed_preparation_session_revision
            ==
            session.revision
        )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML MONITORING API "
            "REAL HANDOFF v0.1 ==="
        )
    )

    print()


    test_monitoring_route_is_registered()

    print(
        "[PASS] Monitoring API route registered"
    )


    test_http_monitoring_real_handoff_end_to_end()

    print(
        (
            "[PASS] HTTP -> Preparation Handoff -> "
            "Monitoring -> Drift history"
        )
    )


    print()

    print(
        (
            "PASS - ML Monitoring API "
            "Real Handoff v0.1"
        )
    )


if __name__ == "__main__":
    main()
