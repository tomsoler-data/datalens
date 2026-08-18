from __future__ import annotations


import hashlib
import json
import os
import threading
import uuid

from datetime import (
    datetime,
    timezone,
)

from pathlib import (
    Path,
)

from typing import (
    Any,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


# ============================================================
# VERSION
# ============================================================

AI_TRACE_RULE_VERSION = (
    "ai_trace_v0.3"
)


# ============================================================
# LOCAL STORAGE
# ============================================================

DEFAULT_TRACE_RELATIVE_PATH = (
    Path(
        "data"
    )
    /
    "observability"
    /
    "ai_traces.jsonl"
)


_TRACE_WRITE_LOCK = (
    threading.Lock()
)


# ============================================================
# TRACE SCHEMAS
# ============================================================

class AITracePrivacy(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    storage_scope: str = (
        "local_jsonl"
    )

    contains_raw_dataset_rows: bool = (
        False
    )

    contains_uploaded_file_contents: bool = (
        False
    )

    contains_document_chunks: bool = (
        False
    )

    contains_objective_text: bool = (
        True
    )

    note: str = (
        "DataLens observability stores local analytical "
        "metadata and AI decision traces. It does not "
        "persist raw uploaded dataset rows in this trace."
    )


class AITraceTiming(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    ingestion_ms: float = Field(
        ge=0.0
    )

    planner_ms: float = Field(
        ge=0.0
    )

    native_pipeline_ms: float = Field(
        ge=0.0
    )

    total_ms: float = Field(
        ge=0.0
    )

    planner_prompt_construction_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    planner_model_inference_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    planner_structured_parse_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    planner_python_validation_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    planner_retry_feedback_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    tool_prompt_construction_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    tool_model_inference_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    tool_response_parse_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    tool_python_validation_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    deterministic_execution_ms: float = Field(
        default=0.0,
        ge=0.0,
    )


class AITraceRecord(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    trace_id: str

    created_at_utc: str

    trace_rule_version: str = (
        AI_TRACE_RULE_VERSION
    )

    objective: str

    objective_sha256: str

    datasets: list[
        dict[
            str,
            Any,
        ]
    ] = Field(
        default_factory=list
    )

    planner: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    native_pipeline: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    timings: AITraceTiming

    privacy: AITracePrivacy = Field(
        default_factory=
            AITracePrivacy
    )


class AITraceWriteResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    enabled: bool

    written: bool

    path: (
        str
        | None
    ) = None

    error: (
        str
        | None
    ) = None


# ============================================================
# GENERIC SERIALIZATION HELPERS
# ============================================================

def to_plain_data(
    value: Any,
) -> Any:
    if value is None:
        return None


    model_dump = getattr(
        value,
        "model_dump",
        None,
    )


    if callable(
        model_dump
    ):
        return model_dump(
            mode="json"
        )


    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                key
            ):
                to_plain_data(
                    item
                )

            for key, item
            in value.items()
        }


    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            to_plain_data(
                item
            )

            for item
            in value
        ]


    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )


    return value


def read_value(
    source: Any,
    key: str,
    default: Any = None,
) -> Any:
    if isinstance(
        source,
        dict,
    ):
        return source.get(
            key,
            default,
        )


    return getattr(
        source,
        key,
        default,
    )


# ============================================================
# TRACE ID / HASH
# ============================================================

def new_ai_trace_id() -> str:
    return (
        "ai:"
        +
        uuid.uuid4().hex
    )


def objective_sha256(
    objective: str,
) -> str:
    return (
        hashlib.sha256(
            objective.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


# ============================================================
# DATASET TRACE
# ============================================================

def build_dataset_trace(
    catalog: Any,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    datasets = (
        read_value(
            catalog,
            "datasets",
            [],
        )
        or []
    )


    traced: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for dataset in datasets:
        columns = (
            read_value(
                dataset,
                "columns",
                [],
            )
            or []
        )


        traced.append(
            {
                "dataset_id":
                    str(
                        read_value(
                            dataset,
                            "dataset_id",
                            "",
                        )
                    ),

                "filename":
                    str(
                        read_value(
                            dataset,
                            "filename",
                            "",
                        )
                    ),

                "row_count":
                    int(
                        read_value(
                            dataset,
                            "row_count",
                            0,
                        )
                        or 0
                    ),

                "column_count":
                    len(
                        columns
                    ),

                "columns":
                    [
                        {
                            "name":
                                str(
                                    read_value(
                                        column,
                                        "name",
                                        "",
                                    )
                                ),

                            "dtype":
                                str(
                                    read_value(
                                        column,
                                        "dtype",
                                        "",
                                    )
                                ),

                            "analysis_kind":
                                str(
                                    read_value(
                                        column,
                                        "analysis_kind",
                                        "unknown",
                                    )
                                ),
                        }

                        for column
                        in columns
                    ],
            }
        )


    return traced


# ============================================================
# PLANNER TRACE
# ============================================================

def build_planner_trace(
    planner_report: Any,
) -> dict[
    str,
    Any,
]:
    planner = (
        to_plain_data(
            planner_report
        )
        or {}
    )


    items: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for item in (
        planner.get(
            "items",
            [],
        )
        or []
    ):
        contract = (
            item.get(
                "contract"
            )
            or {}
        )


        items.append(
            {
                "proposal_index":
                    item.get(
                        "proposal_index"
                    ),

                "validation_status":
                    item.get(
                        "validation_status"
                    ),

                "raw_proposal":
                    item.get(
                        "raw_proposal"
                    ),

                "canonical_proposal":
                    item.get(
                        "proposal"
                    ),

                "errors":
                    list(
                        item.get(
                            "errors",
                            [],
                        )
                        or []
                    ),

                "warnings":
                    list(
                        item.get(
                            "warnings",
                            [],
                        )
                        or []
                    ),

                "normalizations":
                    list(
                        item.get(
                            "normalizations",
                            [],
                        )
                        or []
                    ),

                "contract":
                    (
                        {
                            "contract_id":
                                contract.get(
                                    "contract_id"
                                ),

                            "status":
                                contract.get(
                                    "status"
                                ),

                            "family":
                                contract.get(
                                    "family"
                                ),

                            "required_dataset_ids":
                                contract.get(
                                    "required_dataset_ids",
                                    [],
                                ),

                            "required_dataset_filenames":
                                contract.get(
                                    "required_dataset_filenames",
                                    [],
                                ),

                            "bindings":
                                contract.get(
                                    "bindings",
                                    [],
                                ),

                            "aggregation":
                                contract.get(
                                    "aggregation"
                                ),

                            "ranking":
                                contract.get(
                                    "ranking"
                                ),

                            "window":
                                contract.get(
                                    "window"
                                ),

                            "blockers":
                                contract.get(
                                    "blockers",
                                    [],
                                ),

                            "reasons":
                                contract.get(
                                    "reasons",
                                    [],
                                ),
                        }

                        if contract
                        else None
                    ),
            }
        )


    return {
        "status":
            planner.get(
                "status"
            ),

        "model":
            planner.get(
                "model"
            ),

        "planner_rule_version":
            planner.get(
                "planner_rule_version"
            ),

        "proposal_count":
            planner.get(
                "proposal_count"
            ),

        "validated_count":
            planner.get(
                "validated_count"
            ),

        "blocked_count":
            planner.get(
                "blocked_count"
            ),

        "ambiguous_count":
            planner.get(
                "ambiguous_count"
            ),

        "rejected_count":
            planner.get(
                "rejected_count"
            ),

        "attempt_count":
            planner.get(
                "attempt_count"
            ),

        "retry_count":
            planner.get(
                "retry_count"
            ),

        "retry_triggered":
            planner.get(
                "retry_triggered"
            ),

        "retry_feedback":
            planner.get(
                "retry_feedback",
                [],
            ),

        "normalization_count":
            planner.get(
                "normalization_count"
            ),

        "normalization_applied":
            planner.get(
                "normalization_applied"
            ),

        "timing":
            planner.get(
                "timing",
                {},
            ),

        "items":
            items,
    }


# ============================================================
# NATIVE PIPELINE TRACE
# ============================================================

def build_native_pipeline_trace(
    pipeline_report: Any,
) -> dict[
    str,
    Any,
]:
    pipeline = (
        to_plain_data(
            pipeline_report
        )
        or {}
    )


    items: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for item in (
        pipeline.get(
            "items",
            [],
        )
        or []
    ):
        native = (
            item.get(
                "native_tool"
            )
            or {}
        )


        execution = (
            native.get(
                "execution"
            )
            or {}
        )


        result = (
            execution.get(
                "result"
            )
            or {}
        )


        attempts = [
            {
                "attempt_index":
                    attempt.get(
                        "attempt_index"
                    ),

                "prompt_variant":
                    attempt.get(
                        "prompt_variant"
                    ),

                "tool_call_count":
                    attempt.get(
                        "tool_call_count"
                    ),

                "selected_tool_name":
                    attempt.get(
                        "selected_tool_name"
                    ),

                "errors":
                    attempt.get(
                        "errors",
                        [],
                    ),

                "prompt_construction_ms":
                    attempt.get(
                        "prompt_construction_ms",
                        0.0,
                    ),

                "model_inference_ms":
                    attempt.get(
                        "model_inference_ms",
                        0.0,
                    ),

                "response_parse_ms":
                    attempt.get(
                        "response_parse_ms",
                        0.0,
                    ),

                "total_ms":
                    attempt.get(
                        "total_ms",
                        0.0,
                    ),
            }

            for attempt
            in (
                native.get(
                    "attempts",
                    [],
                )
                or []
            )
        ]


        items.append(
            {
                "contract_id":
                    item.get(
                        "contract_id"
                    ),

                "family":
                    item.get(
                        "family"
                    ),

                "pipeline_status":
                    item.get(
                        "pipeline_status"
                    ),

                "pipeline_errors":
                    item.get(
                        "errors",
                        [],
                    ),

                "pipeline_warnings":
                    item.get(
                        "warnings",
                        [],
                    ),

                "tool_call":
                    (
                        {
                            "model":
                                native.get(
                                    "model"
                                ),

                            "native_tool_rule_version":
                                native.get(
                                    "native_tool_rule_version"
                                ),

                            "expected_tool":
                                native.get(
                                    "expected_tool"
                                ),

                            "tool_call_received":
                                native.get(
                                    "tool_call_received"
                                ),

                            "requested_tool":
                                native.get(
                                    "requested_tool"
                                ),

                            "requested_arguments":
                                native.get(
                                    "requested_arguments",
                                    {},
                                ),

                            "validation_status":
                                native.get(
                                    "validation_status"
                                ),

                            "validation_errors":
                                native.get(
                                    "validation_errors",
                                    [],
                                ),

                            "attempt_count":
                                native.get(
                                    "attempt_count"
                                ),

                            "retry_count":
                                native.get(
                                    "retry_count"
                                ),

                            "attempts":
                                attempts,

                            "timing":
                                native.get(
                                    "timing",
                                    {},
                                ),
                        }

                        if native
                        else None
                    ),

                "execution":
                    (
                        {
                            "execution_status":
                                execution.get(
                                    "execution_status"
                                ),

                            "tool_name":
                                execution.get(
                                    "tool_name"
                                ),

                            "dataset_id":
                                execution.get(
                                    "dataset_id"
                                ),

                            "dataset_filename":
                                execution.get(
                                    "dataset_filename"
                                ),

                            "arguments":
                                execution.get(
                                    "arguments",
                                    {},
                                ),

                            "result_status":
                                result.get(
                                    "execution_status"
                                ),

                            "chart_type":
                                result.get(
                                    "chart_type"
                                ),

                            "execution_rule_version":
                                result.get(
                                    "execution_rule_version"
                                ),
                        }

                        if execution
                        else None
                    ),
            }
        )


    return {
        "trace_id":
            pipeline.get(
                "trace_id"
            ),

        "status":
            pipeline.get(
                "status"
            ),

        "planner_model":
            pipeline.get(
                "planner_model"
            ),

        "tool_model":
            pipeline.get(
                "tool_model"
            ),

        "pipeline_rule_version":
            pipeline.get(
                "pipeline_rule_version"
            ),

        "timing":
            pipeline.get(
                "timing",
                {},
            ),

        "validated_contract_count":
            pipeline.get(
                "validated_contract_count"
            ),

        "pipeline_item_count":
            pipeline.get(
                "pipeline_item_count"
            ),

        "executed_count":
            pipeline.get(
                "executed_count"
            ),

        "not_supported_count":
            pipeline.get(
                "not_supported_count"
            ),

        "rejected_count":
            pipeline.get(
                "rejected_count"
            ),

        "items":
            items,
    }


# ============================================================
# TRACE CONSTRUCTION
# ============================================================

def build_ai_trace(
    *,
    trace_id: str,
    objective: str,
    catalog: Any,
    planner_report: Any,
    pipeline_report: Any,
    ingestion_ms: float,
    planner_ms: float,
    native_pipeline_ms: float,
    total_ms: float,
) -> AITraceRecord:
    planner_plain = (
        to_plain_data(
            planner_report
        )
        or {}
    )


    planner_timing = (
        planner_plain.get(
            "timing",
            {},
        )
        or {}
    )


    pipeline_plain = (
        to_plain_data(
            pipeline_report
        )
        or {}
    )


    pipeline_timing = (
        pipeline_plain.get(
            "timing",
            {},
        )
        or {}
    )


    def safe_timing(
        source: dict[
            str,
            Any,
        ],
        key: str,
    ) -> float:
        value = (
            source.get(
                key,
                0.0,
            )
            or 0.0
        )


        try:
            return max(
                0.0,
                float(
                    value
                ),
            )


        except (
            TypeError,
            ValueError,
        ):
            return 0.0


    return (
        AITraceRecord(
            trace_id=
                trace_id,

            created_at_utc=(
                datetime.now(
                    timezone.utc
                )
                .isoformat()
            ),

            objective=
                objective,

            objective_sha256=(
                objective_sha256(
                    objective
                )
            ),

            datasets=(
                build_dataset_trace(
                    catalog
                )
            ),

            planner=(
                build_planner_trace(
                    planner_report
                )
            ),

            native_pipeline=(
                build_native_pipeline_trace(
                    pipeline_report
                )
            ),

            timings=(
                AITraceTiming(
                    ingestion_ms=
                        round(
                            max(
                                0.0,
                                ingestion_ms,
                            ),
                            3,
                        ),

                    planner_ms=
                        round(
                            max(
                                0.0,
                                planner_ms,
                            ),
                            3,
                        ),

                    native_pipeline_ms=
                        round(
                            max(
                                0.0,
                                native_pipeline_ms,
                            ),
                            3,
                        ),

                    total_ms=
                        round(
                            max(
                                0.0,
                                total_ms,
                            ),
                            3,
                        ),

                    planner_prompt_construction_ms=
                        round(
                            safe_timing(
                                planner_timing,
                                "prompt_construction_ms",
                            ),
                            3,
                        ),

                    planner_model_inference_ms=
                        round(
                            safe_timing(
                                planner_timing,
                                "model_inference_ms",
                            ),
                            3,
                        ),

                    planner_structured_parse_ms=
                        round(
                            safe_timing(
                                planner_timing,
                                "structured_parse_ms",
                            ),
                            3,
                        ),

                    planner_python_validation_ms=
                        round(
                            safe_timing(
                                planner_timing,
                                "python_validation_ms",
                            ),
                            3,
                        ),

                    planner_retry_feedback_ms=
                        round(
                            safe_timing(
                                planner_timing,
                                "retry_feedback_ms",
                            ),
                            3,
                        ),

                    tool_prompt_construction_ms=
                        round(
                            safe_timing(
                                pipeline_timing,
                                "tool_prompt_construction_ms",
                            ),
                            3,
                        ),

                    tool_model_inference_ms=
                        round(
                            safe_timing(
                                pipeline_timing,
                                "tool_model_inference_ms",
                            ),
                            3,
                        ),

                    tool_response_parse_ms=
                        round(
                            safe_timing(
                                pipeline_timing,
                                "tool_response_parse_ms",
                            ),
                            3,
                        ),

                    tool_python_validation_ms=
                        round(
                            safe_timing(
                                pipeline_timing,
                                "tool_python_validation_ms",
                            ),
                            3,
                        ),

                    deterministic_execution_ms=
                        round(
                            safe_timing(
                                pipeline_timing,
                                "deterministic_execution_ms",
                            ),
                            3,
                        ),
                )
            ),
        )
    )


# ============================================================
# CONFIGURATION
# ============================================================

def ai_trace_enabled() -> bool:
    raw = (
        os.getenv(
            "DATALENS_AI_TRACE_ENABLED",
            "1",
        )
        .strip()
        .casefold()
    )


    return raw not in {
        "0",
        "false",
        "no",
        "off",
    }


def default_api_root() -> Path:
    return (
        Path(
            __file__
        )
        .resolve()
        .parents[
            2
        ]
    )


def resolve_ai_trace_path() -> Path:
    configured = (
        os.getenv(
            "DATALENS_AI_TRACE_PATH",
            "",
        )
        .strip()
    )


    if configured:
        configured_path = (
            Path(
                configured
            )
            .expanduser()
        )


        if (
            configured_path
            .is_absolute()
        ):
            return (
                configured_path
            )


        return (
            default_api_root()
            /
            configured_path
        ).resolve()


    return (
        default_api_root()
        /
        DEFAULT_TRACE_RELATIVE_PATH
    ).resolve()


# ============================================================
# JSONL WRITER
# ============================================================

def write_ai_trace(
    trace: AITraceRecord,
) -> AITraceWriteResult:
    if not ai_trace_enabled():
        return (
            AITraceWriteResult(
                enabled=False,
                written=False,
            )
        )


    path = (
        resolve_ai_trace_path()
    )


    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        serialized = (
            trace.model_dump_json()
            +
            "\n"
        )


        with _TRACE_WRITE_LOCK:
            prefix = ""


            if (
                path.exists()
                and
                path.stat().st_size > 0
            ):
                with path.open(
                    "rb"
                ) as existing:
                    existing.seek(
                        -1,
                        2,
                    )


                    last_byte = (
                        existing.read(
                            1
                        )
                    )


                if last_byte not in {
                    b"\n",
                    b"\r",
                }:
                    prefix = "\n"


            with path.open(
                "a",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                handle.write(
                    prefix
                    +
                    serialized
                )


        return (
            AITraceWriteResult(
                enabled=True,
                written=True,
                path=str(
                    path
                ),
            )
        )


    except OSError as error:
        # Observability must never break the analytical
        # execution path.
        return (
            AITraceWriteResult(
                enabled=True,
                written=False,
                path=str(
                    path
                ),
                error=(
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            )
        )
