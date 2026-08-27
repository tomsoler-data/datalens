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
    Literal,
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
    "ai_trace_v0.4"
)


AI_TRACE_PRIVACY_RULE_VERSION = (
    "ai_trace_privacy_v0.1"
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


    privacy_rule_version: str = (
        AI_TRACE_PRIVACY_RULE_VERSION
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

    contains_model_raw_output: bool = (
        False
    )

    contains_model_arguments: bool = (
        False
    )

    contains_internal_error_details: bool = (
        False
    )

    contains_trace_storage_path: bool = (
        False
    )

    note: str = (
        "DataLens observability stores local analytical "
        "metadata, validated decision metadata and "
        "timings. Raw dataset rows, uploaded contents, "
        "document chunks, raw model output, model "
        "arguments, internal error details and local "
        "trace storage filesystem paths are not persisted ""in traces."
    )


AITraceAnalysisSourceType = Literal[
    "initial_request",
    "follow_up_prompt",
    "document_request",
    "automatic",
]


AITraceRunStatus = Literal[
    "completed",
    "failed",
]


class AITraceFailure(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    stage: str = Field(
        min_length=1
    )

    error_type: str = Field(
        min_length=1
    )

    message_safe: str = Field(
        min_length=1
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

    workflow_id: (
        str
        | None
    ) = None

    analysis_id: (
        str
        | None
    ) = None

    analysis_source_type: (
        AITraceAnalysisSourceType
        | None
    ) = None

    run_status: AITraceRunStatus = (
        "completed"
    )

    failure: (
        AITraceFailure
        | None
    ) = None

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
# PRIVACY-SAFE TRACE METADATA
# ============================================================


def _trace_collection_count(
    value: Any,
) -> int:

    if isinstance(
        value,
        (
            dict,
            list,
            tuple,
            set,
        ),
    ):
        return len(
            value
        )


    return 0


def _trace_metadata_string(
    value: Any,
) -> (
    str
    | None
):

    if value is None:
        return None


    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return str(
            value
        )


    return None


def _trace_filename(
    value: Any,
) -> (
    str
    | None
):

    normalized = (
        _trace_metadata_string(
            value
        )
    )


    if not normalized:
        return None


    # Support both Windows and POSIX separators even when
    # tests run on a different operating system.
    safe = (
        normalized
        .replace(
            "\\",
            "/",
        )
        .rsplit(
            "/",
            1,
        )[
            -1
        ]
    )


    return (
        safe
        or
        None
    )


def _trace_string_list(
    value: Any,
    *,
    filenames: bool = False,
) -> list[
    str
]:

    if not isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return []


    result: list[
        str
    ] = []


    for item in value:

        normalized = (
            _trace_filename(
                item
            )
            if filenames
            else
            _trace_metadata_string(
                item
            )
        )


        if normalized:
            result.append(
                normalized
            )


        if len(
            result
        ) >= 100:
            break


    return result


def _trace_binding_metadata(
    binding: Any,
) -> (
    dict[
        str,
        Any,
    ]
    | None
):

    if not isinstance(
        binding,
        dict,
    ):
        return None


    safe = {
        "role":
            _trace_metadata_string(
                binding.get(
                    "role"
                )
            ),

        "dataset_id":
            _trace_metadata_string(
                binding.get(
                    "dataset_id"
                )
            ),

        "dataset_filename":
            _trace_filename(
                binding.get(
                    "dataset_filename"
                )
            ),

        "column":
            _trace_metadata_string(
                binding.get(
                    "column"
                )
            ),

        "analysis_kind":
            _trace_metadata_string(
                binding.get(
                    "analysis_kind"
                )
            ),
    }


    return {
        key:
            value

        for (
            key,
            value,
        ) in safe.items()

        if value is not None
    }


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
                    (
                        _trace_filename(
                            read_value(
                                dataset,
                                "filename",
                                "",
                            )
                        )
                        or
                        ""
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

        if not isinstance(
            item,
            dict,
        ):
            continue


        contract = (
            item.get(
                "contract"
            )
            or {}
        )


        if not isinstance(
            contract,
            dict,
        ):
            contract = {}


        required_dataset_ids = (
            _trace_string_list(
                contract.get(
                    "required_dataset_ids",
                    [],
                )
            )
        )


        required_dataset_filenames = (
            _trace_string_list(
                contract.get(
                    "required_dataset_filenames",
                    [],
                ),
                filenames=True,
            )
        )


        bindings: list[
            dict[
                str,
                Any,
            ]
        ] = []


        raw_bindings = (
            contract.get(
                "bindings",
                [],
            )
            or []
        )


        if isinstance(
            raw_bindings,
            (
                list,
                tuple,
            ),
        ):

            for binding in raw_bindings:

                safe_binding = (
                    _trace_binding_metadata(
                        binding
                    )
                )


                if safe_binding:
                    bindings.append(
                        safe_binding
                    )


                if len(
                    bindings
                ) >= 100:
                    break


        canonical_proposal = (
            {
                "family":
                    _trace_metadata_string(
                        contract.get(
                            "family"
                        )
                    ),

                "dataset_id":
                    (
                        required_dataset_ids[
                            0
                        ]
                        if required_dataset_ids
                        else None
                    ),
            }

            if contract
            else None
        )


        if canonical_proposal is not None:

            canonical_proposal = {
                key:
                    value

                for (
                    key,
                    value,
                ) in (
                    canonical_proposal
                    .items()
                )

                if value is not None
            }


        items.append(
            {
                "proposal_index":
                    item.get(
                        "proposal_index"
                    ),

                "validation_status":
                    _trace_metadata_string(
                        item.get(
                            "validation_status"
                        )
                    ),

                # Deliberately derived from the validated
                # AnalyticalContract rather than copied from
                # raw or model-produced proposal payloads.
                "canonical_proposal":
                    canonical_proposal,

                "error_count":
                    _trace_collection_count(
                        item.get(
                            "errors"
                        )
                    ),

                "warning_count":
                    _trace_collection_count(
                        item.get(
                            "warnings"
                        )
                    ),

                "normalization_count":
                    _trace_collection_count(
                        item.get(
                            "normalizations"
                        )
                    ),

                "contract":
                    (
                        {
                            "contract_id":
                                _trace_metadata_string(
                                    contract.get(
                                        "contract_id"
                                    )
                                ),

                            "status":
                                _trace_metadata_string(
                                    contract.get(
                                        "status"
                                    )
                                ),

                            "family":
                                _trace_metadata_string(
                                    contract.get(
                                        "family"
                                    )
                                ),

                            "required_dataset_ids":
                                required_dataset_ids,

                            "required_dataset_filenames":
                                required_dataset_filenames,

                            "bindings":
                                bindings,

                            "binding_count":
                                len(
                                    bindings
                                ),

                            "blocker_count":
                                _trace_collection_count(
                                    contract.get(
                                        "blockers"
                                    )
                                ),

                            "reason_count":
                                _trace_collection_count(
                                    contract.get(
                                        "reasons"
                                    )
                                ),
                        }

                        if contract
                        else None
                    ),
            }
        )


    return {
        "status":
            _trace_metadata_string(
                planner.get(
                    "status"
                )
            ),

        "model":
            _trace_metadata_string(
                planner.get(
                    "model"
                )
            ),

        "planner_rule_version":
            _trace_metadata_string(
                planner.get(
                    "planner_rule_version"
                )
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

        "retry_feedback_count":
            _trace_collection_count(
                planner.get(
                    "retry_feedback"
                )
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
            (
                planner.get(
                    "timing",
                    {},
                )
                if isinstance(
                    planner.get(
                        "timing",
                        {},
                    ),
                    dict,
                )
                else {}
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

        if not isinstance(
            item,
            dict,
        ):
            continue


        native = (
            item.get(
                "native_tool"
            )
            or {}
        )


        if not isinstance(
            native,
            dict,
        ):
            native = {}


        execution = (
            native.get(
                "execution"
            )
            or {}
        )


        if not isinstance(
            execution,
            dict,
        ):
            execution = {}


        result = (
            execution.get(
                "result"
            )
            or {}
        )


        if not isinstance(
            result,
            dict,
        ):
            result = {}


        attempts: list[
            dict[
                str,
                Any,
            ]
        ] = []


        raw_attempts = (
            native.get(
                "attempts",
                [],
            )
            or []
        )


        if isinstance(
            raw_attempts,
            (
                list,
                tuple,
            ),
        ):

            for attempt in raw_attempts:

                if not isinstance(
                    attempt,
                    dict,
                ):
                    continue


                attempts.append(
                    {
                        "attempt_index":
                            attempt.get(
                                "attempt_index"
                            ),

                        "tool_call_count":
                            attempt.get(
                                "tool_call_count"
                            ),

                        "selected_tool_name":
                            _trace_metadata_string(
                                attempt.get(
                                    "selected_tool_name"
                                )
                            ),

                        "error_count":
                            _trace_collection_count(
                                attempt.get(
                                    "errors"
                                )
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
                )


                if len(
                    attempts
                ) >= 20:
                    break


        items.append(
            {
                "contract_id":
                    _trace_metadata_string(
                        item.get(
                            "contract_id"
                        )
                    ),

                "family":
                    _trace_metadata_string(
                        item.get(
                            "family"
                        )
                    ),

                "pipeline_status":
                    _trace_metadata_string(
                        item.get(
                            "pipeline_status"
                        )
                    ),

                "pipeline_error_count":
                    _trace_collection_count(
                        item.get(
                            "errors"
                        )
                    ),

                "pipeline_warning_count":
                    _trace_collection_count(
                        item.get(
                            "warnings"
                        )
                    ),

                "tool_call":
                    (
                        {
                            "model":
                                _trace_metadata_string(
                                    native.get(
                                        "model"
                                    )
                                ),

                            "native_tool_rule_version":
                                _trace_metadata_string(
                                    native.get(
                                        "native_tool_rule_version"
                                    )
                                ),

                            "expected_tool":
                                _trace_metadata_string(
                                    native.get(
                                        "expected_tool"
                                    )
                                ),

                            "tool_call_received":
                                native.get(
                                    "tool_call_received"
                                ),

                            "requested_tool":
                                _trace_metadata_string(
                                    native.get(
                                        "requested_tool"
                                    )
                                ),

                            "validation_status":
                                _trace_metadata_string(
                                    native.get(
                                        "validation_status"
                                    )
                                ),

                            "validation_error_count":
                                _trace_collection_count(
                                    native.get(
                                        "validation_errors"
                                    )
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
                                (
                                    native.get(
                                        "timing",
                                        {},
                                    )
                                    if isinstance(
                                        native.get(
                                            "timing",
                                            {},
                                        ),
                                        dict,
                                    )
                                    else {}
                                ),
                        }

                        if native
                        else None
                    ),

                "execution":
                    (
                        {
                            "execution_status":
                                _trace_metadata_string(
                                    execution.get(
                                        "execution_status"
                                    )
                                ),

                            "tool_name":
                                _trace_metadata_string(
                                    execution.get(
                                        "tool_name"
                                    )
                                ),

                            "dataset_id":
                                _trace_metadata_string(
                                    execution.get(
                                        "dataset_id"
                                    )
                                ),

                            "dataset_filename":
                                _trace_filename(
                                    execution.get(
                                        "dataset_filename"
                                    )
                                ),

                            "result_status":
                                _trace_metadata_string(
                                    result.get(
                                        "execution_status"
                                    )
                                ),

                            "chart_type":
                                _trace_metadata_string(
                                    result.get(
                                        "chart_type"
                                    )
                                ),

                            "execution_rule_version":
                                _trace_metadata_string(
                                    result.get(
                                        "execution_rule_version"
                                    )
                                ),
                        }

                        if execution
                        else None
                    ),
            }
        )


    return {
        "trace_id":
            _trace_metadata_string(
                pipeline.get(
                    "trace_id"
                )
            ),

        "status":
            _trace_metadata_string(
                pipeline.get(
                    "status"
                )
            ),

        "planner_model":
            _trace_metadata_string(
                pipeline.get(
                    "planner_model"
                )
            ),

        "tool_model":
            _trace_metadata_string(
                pipeline.get(
                    "tool_model"
                )
            ),

        "pipeline_rule_version":
            _trace_metadata_string(
                pipeline.get(
                    "pipeline_rule_version"
                )
            ),

        "timing":
            (
                pipeline.get(
                    "timing",
                    {},
                )
                if isinstance(
                    pipeline.get(
                        "timing",
                        {},
                    ),
                    dict,
                )
                else {}
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
    workflow_id: str | None = None,
    run_status: str = "completed",
    failure: Any = None,
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

            workflow_id=
                workflow_id,

            run_status=
                run_status,

            failure=
                failure,

            analysis_id=(
                pipeline_plain.get(
                    "analysis_id"
                )
            ),

            analysis_source_type=(
                pipeline_plain.get(
                    "analysis_source_type"
                )
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
