from __future__ import annotations


from contextlib import (
    contextmanager,
)


from fastapi.testclient import (
    TestClient,
)


import app.api.ml_monitoring as api_module


from app.api.ml_monitoring import (
    ML_MONITORING_API_VERSION,
)


from app.api.ml_monitoring_contracts import (
    ML_MONITORING_API_CONTRACT_RULE_VERSION,
    MLMonitoringRunRequest,
)


from app.main import (
    app,
)


from app.ml.monitoring_service import (
    MLMonitoringObservedDatasetError,
    MLMonitoringServiceAuthorityError,
    MLMonitoringServiceExecutionError,
    MLMonitoringServiceInputError,
)


from tests.ml.test_ml_drift_evaluation_store_v0_1 import (
    evaluation,
)


from tests.ml.test_ml_monitoring_profile_store_v0_1 import (
    isolated_environment,
    persisted_artifact_and_profile,
)


# ============================================================
# RESULT FIXTURE
# ============================================================


def valid_evaluation(
):

    with isolated_environment():

        (
            artifact,
            profile,
        ) = (
            persisted_artifact_and_profile()
        )


        return (
            evaluation(
                artifact=
                    artifact,

                profile=
                    profile,

                observed_dataset_id=
                    "dataset:observed",
            )
        )


# ============================================================
# SERVICE PATCH
# ============================================================


@contextmanager
def patched_service(
    callback,
):

    original = (
        api_module
        .run_ml_monitoring
    )


    api_module.run_ml_monitoring = (
        callback
    )


    try:
        yield

    finally:
        api_module.run_ml_monitoring = (
            original
        )


# ============================================================
# CONTRACT
# ============================================================


def test_public_request_contains_only_identifiers(
) -> None:

    assert (
        set(
            MLMonitoringRunRequest
            .model_fields
        )
        ==
        {
            "workflow_id",
            "model_id",
            "observed_dataset_id",
        }
    )


def test_request_is_frozen_and_extra_forbidden(
) -> None:

    request = (
        MLMonitoringRunRequest(
            workflow_id=
                " prep:api ",

            model_id=
                " model:abc ",

            observed_dataset_id=
                " dataset:observed ",
        )
    )


    assert (
        request.workflow_id
        ==
        "prep:api"
    )


    assert (
        request.model_id
        ==
        "model:abc"
    )


    assert (
        request.observed_dataset_id
        ==
        "dataset:observed"
    )


    try:
        MLMonitoringRunRequest.model_validate(
            {
                "workflow_id":
                    "prep:api",

                "model_id":
                    "model:abc",

                "observed_dataset_id":
                    "dataset:observed",

                "evaluation_id":
                    "client-controlled",
            }
        )

    except Exception:
        pass

    else:
        raise AssertionError(
            (
                "Monitoring API request must "
                "forbid extra authority fields."
            )
        )


# ============================================================
# SUCCESS
# ============================================================


def test_evaluate_endpoint_success(
) -> None:

    result = (
        valid_evaluation()
    )


    captured = {}


    def fake_run_ml_monitoring(
        *,
        workflow_id: str,
        model_id: str,
        observed_dataset_id: str,
    ):

        captured[
            "workflow_id"
        ] = workflow_id

        captured[
            "model_id"
        ] = model_id

        captured[
            "observed_dataset_id"
        ] = observed_dataset_id


        return result


    with isolated_environment():

        with patched_service(
            fake_run_ml_monitoring
        ):

            with TestClient(
                app
            ) as client:

                response = (
                    client.post(
                        "/ml-monitoring/evaluate",

                        json={
                            "workflow_id":
                                " prep:api ",

                            "model_id":
                                " model:abc ",

                            "observed_dataset_id":
                                " dataset:observed ",
                        },
                    )
                )


    assert (
        response.status_code
        ==
        200
    )


    assert captured == {
        "workflow_id":
            "prep:api",

        "model_id":
            "model:abc",

        "observed_dataset_id":
            "dataset:observed",
    }


    assert (
        response.json()
        ==
        result.model_dump(
            mode="json"
        )
    )


# ============================================================
# CLIENT CANNOT INJECT SERVER AUTHORITY
# ============================================================


def test_extra_authority_fields_are_rejected_by_http_contract(
) -> None:

    called = {
        "value":
            False
    }


    def fake_run_ml_monitoring(
        **_,
    ):

        called[
            "value"
        ] = True

        raise AssertionError(
            "Service must not be called."
        )


    with isolated_environment():

        with patched_service(
            fake_run_ml_monitoring
        ):

            with TestClient(
                app
            ) as client:

                response = (
                    client.post(
                        "/ml-monitoring/evaluate",

                        json={
                            "workflow_id":
                                "prep:api",

                            "model_id":
                                "model:abc",

                            "observed_dataset_id":
                                "dataset:observed",

                            "observed_preparation_session_revision":
                                999,

                            "training_contract_sha256":
                                "client-controlled",

                            "evaluation_id":
                                "client-controlled",
                        },
                    )
                )


    assert (
        response.status_code
        ==
        422
    )


    assert (
        called[
            "value"
        ]
        is False
    )


# ============================================================
# INPUT ERROR
# ============================================================


def test_service_input_error_maps_to_422(
) -> None:

    def failing_service(
        **_,
    ):

        raise (
            MLMonitoringServiceInputError(
                "invalid public monitoring input"
            )
        )


    with isolated_environment():

        with patched_service(
            failing_service
        ):

            with TestClient(
                app
            ) as client:

                response = (
                    client.post(
                        "/ml-monitoring/evaluate",

                        json={
                            "workflow_id":
                                "prep:api",

                            "model_id":
                                "model:abc",

                            "observed_dataset_id":
                                "dataset:observed",
                        },
                    )
                )


    assert (
        response.status_code
        ==
        422
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
        "monitoring_input_invalid"
    )


# ============================================================
# OBSERVED DATASET ERROR
# ============================================================


def test_observed_dataset_error_maps_to_422(
) -> None:

    def failing_service(
        **_,
    ):

        raise (
            MLMonitoringObservedDatasetError(
                "dataset not in validated scope"
            )
        )


    with isolated_environment():

        with patched_service(
            failing_service
        ):

            with TestClient(
                app
            ) as client:

                response = (
                    client.post(
                        "/ml-monitoring/evaluate",

                        json={
                            "workflow_id":
                                "prep:api",

                            "model_id":
                                "model:abc",

                            "observed_dataset_id":
                                "dataset:outside",
                        },
                    )
                )


    assert (
        response.status_code
        ==
        422
    )


    assert (
        response.json()[
            "detail"
        ][
            "error"
        ]
        ==
        "observed_dataset_invalid"
    )


# ============================================================
# AUTHORITY ERROR
# ============================================================


def test_authority_error_is_generic_and_non_enumerating(
) -> None:

    secret = (
        "SECRET-MODEL-EXISTS-IN-OTHER-WORKFLOW"
    )


    def failing_service(
        **_,
    ):

        raise (
            MLMonitoringServiceAuthorityError(
                secret
            )
        )


    with isolated_environment():

        with patched_service(
            failing_service
        ):

            with TestClient(
                app
            ) as client:

                response = (
                    client.post(
                        "/ml-monitoring/evaluate",

                        json={
                            "workflow_id":
                                "prep:api",

                            "model_id":
                                "model:abc",

                            "observed_dataset_id":
                                "dataset:observed",
                        },
                    )
                )


    assert (
        response.status_code
        ==
        409
    )


    payload = (
        response.json()
    )


    assert (
        payload[
            "detail"
        ][
            "error"
        ]
        ==
        "monitoring_authority_unavailable"
    )


    assert (
        secret
        not in
        str(
            payload
        )
    )


# ============================================================
# EXECUTION ERROR
# ============================================================


def test_execution_error_maps_to_409(
) -> None:

    secret = (
        "internal-drift-store-details"
    )


    def failing_service(
        **_,
    ):

        raise (
            MLMonitoringServiceExecutionError(
                secret
            )
        )


    with isolated_environment():

        with patched_service(
            failing_service
        ):

            with TestClient(
                app
            ) as client:

                response = (
                    client.post(
                        "/ml-monitoring/evaluate",

                        json={
                            "workflow_id":
                                "prep:api",

                            "model_id":
                                "model:abc",

                            "observed_dataset_id":
                                "dataset:observed",
                        },
                    )
                )


    assert (
        response.status_code
        ==
        409
    )


    payload = (
        response.json()
    )


    assert (
        payload[
            "detail"
        ][
            "error"
        ]
        ==
        "monitoring_execution_failed"
    )


    assert (
        secret
        not in
        str(
            payload
        )
    )


# ============================================================
# VERSION
# ============================================================


def test_versions(
) -> None:

    assert (
        ML_MONITORING_API_VERSION
        ==
        "ml_monitoring_api_v0.1"
    )


    assert (
        ML_MONITORING_API_CONTRACT_RULE_VERSION
        ==
        "ml_monitoring_api_contract_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML MONITORING API v0.1 ==="
    )

    print()


    tests = [
        (
            "Public request contains identifiers only",
            test_public_request_contains_only_identifiers,
        ),
        (
            "Request frozen / extra authority forbidden",
            test_request_is_frozen_and_extra_forbidden,
        ),
        (
            "POST evaluate success",
            test_evaluate_endpoint_success,
        ),
        (
            "HTTP authority injection blocked",
            test_extra_authority_fields_are_rejected_by_http_contract,
        ),
        (
            "Service input error -> 422",
            test_service_input_error_maps_to_422,
        ),
        (
            "Observed dataset error -> 422",
            test_observed_dataset_error_maps_to_422,
        ),
        (
            "Authority error -> generic 409",
            test_authority_error_is_generic_and_non_enumerating,
        ),
        (
            "Execution error -> generic 409",
            test_execution_error_maps_to_409,
        ),
        (
            "Monitoring API versions",
            test_versions,
        ),
    ]


    for (
        label,
        callback,
    ) in tests:

        callback()

        print(
            f"[PASS] {label}"
        )


    print()

    print(
        "PASS - ML Monitoring API v0.1"
    )


if __name__ == "__main__":
    main()
