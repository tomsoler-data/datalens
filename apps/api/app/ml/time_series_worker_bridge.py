from __future__ import annotations


import json
import math
import os
from pathlib import Path
import subprocess


from pydantic import (
    ValidationError,
)


from app.ml.time_series_worker_contracts import (
    MLTimeSeriesWorkerFailure,
    MLTimeSeriesWorkerRequest,
    MLTimeSeriesWorkerResult,
    MLTimeSeriesWorkerSuccess,
    validate_time_series_worker_response,
)


# ============================================================
# VERSION
# ============================================================


ML_TIME_SERIES_WORKER_BRIDGE_RULE_VERSION = (
    "ml_time_series_worker_bridge_v0.1"
)


# ============================================================
# CONFIG
# ============================================================


DATALENS_DL_PYTHON_PATH_ENV = (
    "DATALENS_DL_PYTHON_PATH"
)


DEFAULT_TIME_SERIES_WORKER_TIMEOUT_SECONDS = (
    900.0
)


MAX_WORKER_RESPONSE_BYTES = (
    4
    *
    1024
    *
    1024
)


# ============================================================
# ERRORS
# ============================================================


class MLTimeSeriesWorkerBridgeError(
    RuntimeError
):
    pass


class MLTimeSeriesWorkerUnavailableError(
    MLTimeSeriesWorkerBridgeError
):
    pass


class MLTimeSeriesWorkerProtocolError(
    MLTimeSeriesWorkerBridgeError
):
    pass


class MLTimeSeriesWorkerExecutionError(
    MLTimeSeriesWorkerBridgeError
):

    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        retryable: bool,
    ) -> None:

        super().__init__(
            message
        )


        self.error_code = (
            str(
                error_code
            )
        )


        self.retryable = bool(
            retryable
        )


# ============================================================
# API ROOT
# ============================================================


def _api_root() -> Path:

    return (
        Path(
            __file__
        )
        .resolve()
        .parents[
            2
        ]
    )


# ============================================================
# DL PYTHON
# ============================================================


def resolve_time_series_dl_python_executable() -> Path:
    """
    Resolve the isolated Deep Learning Python runtime.

    Optional deployment override:
        DATALENS_DL_PYTHON_PATH

    Local default:
        apps/api/.venv-dl
    """

    override = str(
        os.getenv(
            DATALENS_DL_PYTHON_PATH_ENV,
            "",
        )
    ).strip()


    if override:

        candidate = (
            Path(
                override
            )
            .expanduser()
            .resolve()
        )

    else:

        api_root = (
            _api_root()
        )


        if os.name == "nt":

            candidate = (
                api_root
                /
                ".venv-dl"
                /
                "Scripts"
                /
                "python.exe"
            )

        else:

            candidate = (
                api_root
                /
                ".venv-dl"
                /
                "bin"
                /
                "python"
            )


    if not candidate.exists():

        raise MLTimeSeriesWorkerUnavailableError(
            (
                "Deep Learning Python runtime "
                "was not found."
            )
        )


    if not candidate.is_file():

        raise MLTimeSeriesWorkerUnavailableError(
            (
                "Deep Learning Python runtime "
                "path is not a file."
            )
        )


    return candidate


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


def _decode_worker_response(
    raw_text: object,
):

    if not isinstance(
        raw_text,
        str,
    ):

        raise MLTimeSeriesWorkerProtocolError(
            "Worker response must be text."
        )


    if not raw_text.strip():

        raise MLTimeSeriesWorkerProtocolError(
            "Worker returned an empty response."
        )


    if (
        len(
            raw_text.encode(
                "utf-8"
            )
        )
        >
        MAX_WORKER_RESPONSE_BYTES
    ):

        raise MLTimeSeriesWorkerProtocolError(
            "Worker response exceeds size boundary."
        )


    try:

        payload = json.loads(
            raw_text,
            object_pairs_hook=
                _reject_duplicate_json_pairs,
            parse_constant=
                _reject_json_constant,
        )

    except Exception as error:

        raise MLTimeSeriesWorkerProtocolError(
            (
                "Worker returned invalid "
                "strict JSON."
            )
        ) from error


    if not isinstance(
        payload,
        dict,
    ):

        raise MLTimeSeriesWorkerProtocolError(
            (
                "Worker response must contain "
                "a JSON object."
            )
        )


    try:

        return (
            validate_time_series_worker_response(
                payload
            )
        )

    except (
        ValidationError,
        ValueError,
    ) as error:

        raise MLTimeSeriesWorkerProtocolError(
            (
                "Worker response violates "
                "the transport contract."
            )
        ) from error


# ============================================================
# REQUEST JSON
# ============================================================


def _encode_worker_request(
    request: MLTimeSeriesWorkerRequest,
) -> str:

    payload = request.model_dump(
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
# CHILD ENVIRONMENT
# ============================================================


def _worker_environment() -> dict[
    str,
    str,
]:

    environment = dict(
        os.environ
    )


    api_root = str(
        _api_root()
    )


    previous_pythonpath = str(
        environment.get(
            "PYTHONPATH",
            "",
        )
    ).strip()


    if previous_pythonpath:

        environment[
            "PYTHONPATH"
        ] = (
            api_root
            +
            os.pathsep
            +
            previous_pythonpath
        )

    else:

        environment[
            "PYTHONPATH"
        ] = api_root


    environment[
        "PYTHONDONTWRITEBYTECODE"
    ] = "1"


    environment[
        "PYTHONUTF8"
    ] = "1"


    return environment


# ============================================================
# PUBLIC BRIDGE
# ============================================================


def execute_time_series_worker(
    *,
    request: MLTimeSeriesWorkerRequest,
    python_executable: str | Path | None = None,
    timeout_seconds: float = (
        DEFAULT_TIME_SERIES_WORKER_TIMEOUT_SECONDS
    ),
) -> MLTimeSeriesWorkerResult:
    """
    Execute forecasting in the isolated `.venv-dl` process.

    This module must remain importable by the FastAPI runtime
    without importing PyTorch or any Deep Learning executor.
    """

    try:

        config = (
            MLTimeSeriesWorkerRequest
            .model_validate(
                request
            )
        )

    except Exception as error:

        raise MLTimeSeriesWorkerProtocolError(
            (
                "Forecast worker request "
                "is invalid."
            )
        ) from error


    if isinstance(
        timeout_seconds,
        bool,
    ):

        raise MLTimeSeriesWorkerProtocolError(
            "Worker timeout must be positive."
        )


    try:

        timeout = float(
            timeout_seconds
        )

    except Exception as error:

        raise MLTimeSeriesWorkerProtocolError(
            "Worker timeout must be positive."
        ) from error


    if (
        not math.isfinite(
            timeout
        )
        or
        timeout <= 0.0
    ):

        raise MLTimeSeriesWorkerProtocolError(
            "Worker timeout must be positive."
        )


    if python_executable is None:

        executable = (
            resolve_time_series_dl_python_executable()
        )

    else:

        executable = (
            Path(
                python_executable
            )
            .expanduser()
            .resolve()
        )


        if (
            not executable.exists()
            or
            not executable.is_file()
        ):

            raise MLTimeSeriesWorkerUnavailableError(
                (
                    "Configured Deep Learning Python "
                    "runtime is unavailable."
                )
            )


    request_json = (
        _encode_worker_request(
            config
        )
    )


    command = [
        str(
            executable
        ),
        "-m",
        (
            "app.deep_learning."
            "time_series_worker"
        ),
    ]


    try:

        completed = subprocess.run(
            command,
            input=
                request_json,
            text=True,
            encoding=
                "utf-8",
            errors=
                "strict",
            capture_output=True,
            timeout=
                timeout,
            cwd=
                str(
                    _api_root()
                ),
            env=
                _worker_environment(),
            check=False,
        )

    except subprocess.TimeoutExpired as error:

        raise MLTimeSeriesWorkerUnavailableError(
            (
                "Deep Learning worker exceeded "
                "its execution timeout."
            )
        ) from error

    except (
        OSError,
        UnicodeError,
    ) as error:

        raise MLTimeSeriesWorkerUnavailableError(
            (
                "Deep Learning worker process "
                "could not be executed."
            )
        ) from error


    if completed.returncode != 0:

        raise MLTimeSeriesWorkerUnavailableError(
            (
                "Deep Learning worker process "
                "terminated unexpectedly."
            )
        )


    response = (
        _decode_worker_response(
            completed.stdout
        )
    )


    if isinstance(
        response,
        MLTimeSeriesWorkerFailure,
    ):

        raise MLTimeSeriesWorkerExecutionError(
            response.message,
            error_code=
                response.error,
            retryable=
                response.retryable,
        )


    if not isinstance(
        response,
        MLTimeSeriesWorkerSuccess,
    ):

        raise MLTimeSeriesWorkerProtocolError(
            (
                "Worker returned an unsupported "
                "response family."
            )
        )


    result = (
        response.result
    )


    if (
        result.workflow_id
        !=
        config.training_contract.workflow_id
    ):

        raise MLTimeSeriesWorkerProtocolError(
            (
                "Worker result workflow identity "
                "does not match request."
            )
        )


    if (
        result.dataset_id
        !=
        config.training_contract.dataset_id
    ):

        raise MLTimeSeriesWorkerProtocolError(
            (
                "Worker result dataset identity "
                "does not match request."
            )
        )


    if (
        result.estimator_key
        !=
        config.training_contract.estimator_key
    ):

        raise MLTimeSeriesWorkerProtocolError(
            (
                "Worker result estimator identity "
                "does not match request."
            )
        )


    if (
        result.preparation_session_revision
        !=
        config.expected_preparation_session_revision
    ):

        raise MLTimeSeriesWorkerProtocolError(
            (
                "Worker result Preparation revision "
                "does not match request pin."
            )
        )


    return result
