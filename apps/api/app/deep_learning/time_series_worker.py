from __future__ import annotations


from contextlib import (
    redirect_stdout,
)


import json
import sys


from pydantic import (
    ValidationError,
)


from app.deep_learning.time_series_model_lab_executor import (
    DLTimeSeriesModelLabArtifactError,
    DLTimeSeriesModelLabEstimatorError,
    DLTimeSeriesModelLabExecutionError,
    DLTimeSeriesModelLabExecutionResult,
    DLTimeSeriesModelLabInputError,
    execute_time_series_model_lab,
)


from app.ml.time_series_worker_contracts import (
    ML_TIME_SERIES_WORKER_PROTOCOL_RULE_VERSION,
    MLTimeSeriesWorkerFailure,
    MLTimeSeriesWorkerMetrics,
    MLTimeSeriesWorkerRequest,
    MLTimeSeriesWorkerResponse,
    MLTimeSeriesWorkerResult,
    MLTimeSeriesWorkerSuccess,
)


# ============================================================
# VERSION
# ============================================================


DL_TIME_SERIES_WORKER_RULE_VERSION = (
    "dl_time_series_worker_v0.1"
)


# ============================================================
# SIZE AUTHORITY
# ============================================================


MAX_WORKER_REQUEST_BYTES = (
    1024
    *
    1024
)


# ============================================================
# STRICT JSON
# ============================================================


def _reject_duplicate_json_pairs(
    pairs,
):

    result = {}


    for (
        key,
        value,
    ) in pairs:

        if key in result:

            raise ValueError(
                (
                    "Duplicate JSON key is forbidden. "
                    f"key={key!r}"
                )
            )


        result[
            key
        ] = value


    return result


def _reject_json_constant(
    value: str,
):

    raise ValueError(
        (
            "Non-standard JSON numeric constant "
            f"is forbidden: {value!r}"
        )
    )


def _decode_worker_request(
    raw_text: object,
) -> MLTimeSeriesWorkerRequest:

    if not isinstance(
        raw_text,
        str,
    ):

        raise ValueError(
            "Worker request must be UTF-8 JSON text."
        )


    if not raw_text.strip():

        raise ValueError(
            "Worker request cannot be empty."
        )


    if (
        len(
            raw_text.encode(
                "utf-8"
            )
        )
        >
        MAX_WORKER_REQUEST_BYTES
    ):

        raise ValueError(
            "Worker request exceeds size boundary."
        )


    payload = json.loads(
        raw_text,
        object_pairs_hook=
            _reject_duplicate_json_pairs,
        parse_constant=
            _reject_json_constant,
    )


    if not isinstance(
        payload,
        dict,
    ):

        raise ValueError(
            "Worker request must contain a JSON object."
        )


    return (
        MLTimeSeriesWorkerRequest
        .model_validate(
            payload
        )
    )


# ============================================================
# RESPONSE ENCODING
# ============================================================


def _encode_worker_response(
    response: MLTimeSeriesWorkerResponse,
) -> str:

    payload = response.model_dump(
        mode="json"
    )


    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        allow_nan=False,
    )


# ============================================================
# FAILURE
# ============================================================


def _failure(
    *,
    error: str,
    message: str,
    retryable: bool = False,
) -> MLTimeSeriesWorkerFailure:

    return (
        MLTimeSeriesWorkerFailure(
            error=
                error,

            message=
                message,

            retryable=
                retryable,
        )
    )


# ============================================================
# RESULT PROJECTION
# ============================================================


def _worker_result_from_execution(
    execution: DLTimeSeriesModelLabExecutionResult,
) -> MLTimeSeriesWorkerResult:

    validated = (
        DLTimeSeriesModelLabExecutionResult
        .model_validate(
            execution
        )
    )


    return (
        MLTimeSeriesWorkerResult(
            workflow_id=
                validated.workflow_id,

            dataset_id=
                validated.dataset_id,

            preparation_session_revision=
                validated
                .preparation_session_revision,

            estimator_key=
                validated.estimator_key,

            train_rows=
                validated.train_rows,

            test_rows=
                validated.test_rows,

            metrics=
                MLTimeSeriesWorkerMetrics
                .model_validate(
                    validated.metrics
                ),

            naive_baseline_metrics=
                MLTimeSeriesWorkerMetrics
                .model_validate(
                    validated
                    .naive_baseline_metrics
                ),

            beats_naive_baseline=
                validated
                .beats_naive_baseline,

            rmse_delta_vs_naive=
                validated
                .rmse_delta_vs_naive,

            experiment_provenance=
                validated
                .experiment_provenance,

            model_artifact=
                validated
                .model_artifact,
        )
    )


# ============================================================
# PUBLIC WORKER EXECUTION
# ============================================================


def execute_time_series_worker_request(
    request: object,
    *,
    executor=None,
) -> MLTimeSeriesWorkerResponse:
    """
    Execute one validated worker request.

    The injectable executor exists only as a deterministic
    testing seam. Production execution always resolves to the
    A13-P2 forecasting Model Lab executor.
    """

    try:

        config = (
            MLTimeSeriesWorkerRequest
            .model_validate(
                request
            )
        )

    except (
        ValidationError,
        ValueError,
    ):

        return (
            _failure(
                error=
                    "worker_invalid_request",

                message=
                    (
                        "Forecast worker request "
                        "is invalid."
                    ),
            )
        )


    selected_executor = (
        execute_time_series_model_lab
        if executor is None
        else executor
    )


    try:

        execution = (
            selected_executor(
                training_contract=
                    config.training_contract,

                expected_preparation_session_revision=(
                    config
                    .expected_preparation_session_revision
                ),

                execution_device=
                    config.execution_device,
            )
        )

    except DLTimeSeriesModelLabInputError:

        return (
            _failure(
                error=
                    "forecast_input_invalid",

                message=
                    (
                        "Forecast worker could not "
                        "resolve a valid Preparation input."
                    ),
            )
        )

    except DLTimeSeriesModelLabEstimatorError:

        return (
            _failure(
                error=
                    "forecast_estimator_failed",

                message=
                    (
                        "Forecast estimator execution "
                        "could not be completed."
                    ),
            )
        )

    except DLTimeSeriesModelLabArtifactError:

        return (
            _failure(
                error=
                    "forecast_artifact_failed",

                message=
                    (
                        "Forecast training completed but "
                        "Artifact persistence failed."
                    ),
            )
        )

    except DLTimeSeriesModelLabExecutionError:

        return (
            _failure(
                error=
                    "forecast_execution_failed",

                message=
                    (
                        "Forecast Model Lab execution "
                        "could not be completed."
                    ),
            )
        )

    except Exception:

        return (
            _failure(
                error=
                    "worker_internal_error",

                message=
                    (
                        "Forecast worker encountered "
                        "an internal execution failure."
                    ),
            )
        )


    try:

        result = (
            _worker_result_from_execution(
                execution
            )
        )

    except Exception:

        return (
            _failure(
                error=
                    "worker_invalid_result",

                message=
                    (
                        "Forecast worker produced "
                        "an invalid result."
                    ),
            )
        )


    return (
        MLTimeSeriesWorkerSuccess(
            result=
                result
        )
    )


# ============================================================
# CLI
# ============================================================


def main() -> int:
    """
    stdin  -> exactly one JSON request
    stdout -> exactly one JSON response

    Internal execution stdout is redirected to stderr so that
    stdout remains a machine-only protocol channel.
    """

    try:

        raw_request = sys.stdin.read()


        try:

            request = (
                _decode_worker_request(
                    raw_request
                )
            )

        except Exception:

            response: MLTimeSeriesWorkerResponse = (
                _failure(
                    error=
                        "worker_invalid_request",

                    message=
                        (
                            "Forecast worker request "
                            "is invalid."
                        ),
                )
            )

        else:

            with redirect_stdout(
                sys.stderr
            ):

                response = (
                    execute_time_series_worker_request(
                        request
                    )
                )


        encoded = (
            _encode_worker_response(
                response
            )
        )


        sys.stdout.write(
            encoded
        )

        sys.stdout.write(
            "\n"
        )

        sys.stdout.flush()


        return 0

    except Exception:

        try:

            fallback = (
                _failure(
                    error=
                        "worker_protocol_failure",

                    message=
                        (
                            "Forecast worker protocol "
                            "failed unexpectedly."
                        ),
                )
            )


            sys.stdout.write(
                _encode_worker_response(
                    fallback
                )
            )

            sys.stdout.write(
                "\n"
            )

            sys.stdout.flush()

        except Exception:

            return 2


        return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
