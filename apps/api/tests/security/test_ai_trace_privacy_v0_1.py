from __future__ import annotations


import os

from pathlib import (
    Path,
)

from tempfile import (
    TemporaryDirectory,
)

from unittest.mock import (
    patch,
)


import app.api.analysis_run as analysis_run_module

from app.observability.ai_trace import (
    AI_TRACE_PRIVACY_RULE_VERSION,
    build_ai_trace,
    write_ai_trace,
)

from app.observability.trace_store import (
    AITraceListResponse,
    AITraceMetricsResponse,
)


TEST_RULE_VERSION = (
    "ai_trace_privacy_test_v0.1"
)


RAW_ROW_SECRET = (
    "POISON_RAW_ROW_7f91ac"
)

DOCUMENT_SECRET = (
    "POISON_DOCUMENT_CHUNK_5d2be1"
)

RAW_PROPOSAL_SECRET = (
    "POISON_RAW_PROPOSAL_41aa83"
)

REQUESTED_ARGUMENT_SECRET = (
    "POISON_REQUESTED_ARGUMENT_97ee14"
)

EXECUTION_ARGUMENT_SECRET = (
    "POISON_EXECUTION_ARGUMENT_b12cc8"
)

ERROR_SECRET = (
    "POISON_INTERNAL_ERROR_82f143"
)

LOCAL_PATH_SECRET = (
    "PRIVATE_HOME_8c41df"
)


def build_poisoned_trace():

    catalog = {
        "datasets": [
            {
                "dataset_id":
                    "dataset:0001",

                "filename":
                    (
                        f"C:\\\\{LOCAL_PATH_SECRET}"
                        "\\\\orders.csv"
                    ),

                "row_count":
                    1,

                "columns": [
                    {
                        "name":
                            "amount",

                        "dtype":
                            "float64",

                        "analysis_kind":
                            "quantitative",
                    }
                ],

                "rows": [
                    {
                        "customer_name":
                            RAW_ROW_SECRET
                    }
                ],

                "document_chunk":
                    DOCUMENT_SECRET,
            }
        ]
    }


    planner_report = {
        "status":
            "ready",

        "model":
            "gemma3:4b",

        "planner_rule_version":
            "privacy-probe",

        "proposal_count":
            1,

        "validated_count":
            1,

        "blocked_count":
            0,

        "ambiguous_count":
            0,

        "rejected_count":
            0,

        "attempt_count":
            1,

        "retry_count":
            1,

        "retry_triggered":
            True,

        "retry_feedback": [
            ERROR_SECRET
        ],

        "normalization_count":
            1,

        "normalization_applied":
            True,

        "timing":
            {},

        "items": [
            {
                "proposal_index":
                    0,

                "validation_status":
                    "validated",

                "raw_proposal": {
                    "family":
                        RAW_PROPOSAL_SECRET
                },

                "proposal": {
                    "family":
                        RAW_PROPOSAL_SECRET,

                    "dataset_id":
                        REQUESTED_ARGUMENT_SECRET,
                },

                "errors": [
                    ERROR_SECRET
                ],

                "warnings": [
                    ERROR_SECRET
                ],

                "normalizations": [
                    ERROR_SECRET
                ],

                "contract": {
                    "contract_id":
                        "contract:1",

                    "status":
                        "validated",

                    "family":
                        "distribution",

                    "required_dataset_ids": [
                        "dataset:0001"
                    ],

                    "required_dataset_filenames": [
                        (
                            f"C:\\\\{LOCAL_PATH_SECRET}"
                            "\\\\orders.csv"
                        )
                    ],

                    "bindings": [
                        {
                            "role":
                                "value",

                            "dataset_id":
                                "dataset:0001",

                            "dataset_filename":
                                (
                                    f"C:\\\\{LOCAL_PATH_SECRET}"
                                    "\\\\orders.csv"
                                ),

                            "column":
                                "amount",

                            "analysis_kind":
                                "quantitative",

                            # Must be dropped by the allowlist.
                            "sample_value":
                                RAW_ROW_SECRET,
                        }
                    ],

                    "blockers": [
                        ERROR_SECRET
                    ],

                    "reasons": [
                        ERROR_SECRET
                    ],

                    "aggregation": {
                        "poison":
                            RAW_ROW_SECRET
                    },
                },
            }
        ],
    }


    pipeline_report = {
        "trace_id":
            "pipeline:privacy-probe",

        "status":
            "completed",

        "planner_model":
            "gemma3:4b",

        "tool_model":
            "gemma3:4b",

        "pipeline_rule_version":
            "privacy-probe",

        "timing":
            {},

        "validated_contract_count":
            1,

        "pipeline_item_count":
            1,

        "executed_count":
            1,

        "not_supported_count":
            0,

        "rejected_count":
            0,

        "items": [
            {
                "contract_id":
                    "contract:1",

                "family":
                    "distribution",

                "pipeline_status":
                    "completed",

                "errors": [
                    ERROR_SECRET
                ],

                "warnings": [
                    ERROR_SECRET
                ],

                "native_tool": {
                    "model":
                        "gemma3:4b",

                    "native_tool_rule_version":
                        "privacy-probe",

                    "expected_tool":
                        "distribution",

                    "tool_call_received":
                        True,

                    "requested_tool":
                        "distribution",

                    "requested_arguments": {
                        "poison":
                            REQUESTED_ARGUMENT_SECRET
                    },

                    "validation_status":
                        "validated",

                    "validation_errors": [
                        ERROR_SECRET
                    ],

                    "attempt_count":
                        1,

                    "retry_count":
                        0,

                    "attempts": [
                        {
                            "attempt_index":
                                1,

                            "prompt_variant":
                                RAW_PROPOSAL_SECRET,

                            "tool_call_count":
                                1,

                            "selected_tool_name":
                                "distribution",

                            "errors": [
                                ERROR_SECRET
                            ],

                            "prompt_construction_ms":
                                1.0,

                            "model_inference_ms":
                                2.0,

                            "response_parse_ms":
                                1.0,

                            "total_ms":
                                4.0,
                        }
                    ],

                    "timing":
                        {},

                    "execution": {
                        "execution_status":
                            "completed",

                        "tool_name":
                            "distribution",

                        "dataset_id":
                            "dataset:0001",

                        "dataset_filename":
                            (
                                f"C:\\\\{LOCAL_PATH_SECRET}"
                                "\\\\orders.csv"
                            ),

                        "arguments": {
                            "poison":
                                EXECUTION_ARGUMENT_SECRET
                        },

                        "result": {
                            "execution_status":
                                "completed",

                            "chart_type":
                                "histogram",

                            "execution_rule_version":
                                "privacy-probe",
                        },
                    },
                },
            }
        ],
    }


    return build_ai_trace(
        trace_id=
            "ai:privacy-probe",

        objective=
            "Analyse distribution",

        catalog=
            catalog,

        planner_report=
            planner_report,

        pipeline_report=
            pipeline_report,

        ingestion_ms=
            1.0,

        planner_ms=
            2.0,

        native_pipeline_ms=
            3.0,

        total_ms=
            6.0,

        workflow_id=
            "prep:privacy-probe",
    )


def test_rule_version(
) -> None:

    assert (
        AI_TRACE_PRIVACY_RULE_VERSION
        ==
        "ai_trace_privacy_v0.1"
    )


def test_poison_values_are_not_serialized(
) -> None:

    trace = (
        build_poisoned_trace()
    )


    serialized = (
        trace.model_dump_json()
    )


    forbidden = [
        RAW_ROW_SECRET,
        DOCUMENT_SECRET,
        RAW_PROPOSAL_SECRET,
        REQUESTED_ARGUMENT_SECRET,
        EXECUTION_ARGUMENT_SECRET,
        ERROR_SECRET,
        LOCAL_PATH_SECRET,
    ]


    for marker in forbidden:

        assert (
            marker
            not in
            serialized
        ), (
            "Forbidden AI trace content "
            f"was persisted: {marker}"
        )


    # Objective text remains explicitly allowed.
    assert (
        "Analyse distribution"
        in
        serialized
    )


    # Validated analytical metadata remains useful.
    assert (
        "distribution"
        in
        serialized
    )

    assert (
        "dataset:0001"
        in
        serialized
    )

    assert (
        "orders.csv"
        in
        serialized
    )

    assert (
        "amount"
        in
        serialized
    )


def test_free_form_trace_keys_are_absent(
) -> None:

    trace = (
        build_poisoned_trace()
    )


    forbidden_keys = {
        "raw_proposal",
        "errors",
        "warnings",
        "normalizations",
        "retry_feedback",
        "requested_arguments",
        "validation_errors",
        "arguments",
        "prompt_variant",
        "aggregation",
        "ranking",
        "window",
        "blockers",
        "reasons",
    }


    def walk(
        value,
    ) -> None:

        if isinstance(
            value,
            dict,
        ):

            overlap = (
                forbidden_keys
                &
                set(
                    value
                    .keys()
                )
            )


            assert (
                not overlap
            ), (
                "Forbidden free-form trace keys "
                f"remain persisted: {sorted(overlap)}"
            )


            for item in (
                value
                .values()
            ):
                walk(
                    item
                )


        elif isinstance(
            value,
            list,
        ):

            for item in value:
                walk(
                    item
                )


    walk(
        trace.planner
    )

    walk(
        trace.native_pipeline
    )


def test_persisted_jsonl_respects_privacy_boundary(
) -> None:

    trace = (
        build_poisoned_trace()
    )


    with TemporaryDirectory() as directory:

        path = (
            Path(
                directory
            )
            /
            "ai-traces.jsonl"
        )


        with patch.dict(
            os.environ,
            {
                "DATALENS_AI_TRACE_ENABLED":
                    "1",

                "DATALENS_AI_TRACE_PATH":
                    str(
                        path
                    ),
            },
            clear=False,
        ):

            result = (
                write_ai_trace(
                    trace
                )
            )


        assert (
            result.written
            is True
        )


        persisted = (
            path.read_text(
                encoding="utf-8"
            )
        )


        for marker in (
            RAW_ROW_SECRET,
            DOCUMENT_SECRET,
            RAW_PROPOSAL_SECRET,
            REQUESTED_ARGUMENT_SECRET,
            EXECUTION_ARGUMENT_SECRET,
            ERROR_SECRET,
            LOCAL_PATH_SECRET,
        ):

            assert (
                marker
                not in
                persisted
            )


def test_privacy_declaration_matches_boundary(
) -> None:

    privacy = (
        build_poisoned_trace()
        .privacy
    )


    assert (
        privacy.contains_raw_dataset_rows
        is False
    )

    assert (
        privacy.contains_uploaded_file_contents
        is False
    )

    assert (
        privacy.contains_document_chunks
        is False
    )

    assert (
        privacy.contains_objective_text
        is True
    )

    assert (
        privacy.contains_model_raw_output
        is False
    )

    assert (
        privacy.contains_model_arguments
        is False
    )

    assert (
        privacy.contains_internal_error_details
        is False
    )

    assert (
        privacy.contains_trace_storage_path
        is False
    )


def test_http_models_do_not_expose_trace_path(
) -> None:

    assert (
        "path"
        not in
        AITraceListResponse
        .model_fields
    )

    assert (
        "path"
        not in
        AITraceMetricsResponse
        .model_fields
    )


def test_writer_diagnostics_are_not_publicly_consumed(
) -> None:

    source = (
        Path(
            analysis_run_module
            .__file__
        )
        .read_text(
            encoding="utf-8-sig"
        )
    )


    assert (
        "trace_write.path"
        not in
        source
    )

    assert (
        "trace_write.error"
        not in
        source
    )


def main(
) -> None:

    print(
        "=== DATALENS AI TRACE "
        "PRIVACY v0.1 ==="
    )

    print()


    tests = [
        (
            "AI trace privacy rule version",
            test_rule_version,
        ),
        (
            "Poison values excluded",
            test_poison_values_are_not_serialized,
        ),
        (
            "Free-form trace keys excluded",
            test_free_form_trace_keys_are_absent,
        ),
        (
            "Persisted JSONL privacy boundary",
            test_persisted_jsonl_respects_privacy_boundary,
        ),
        (
            "Privacy declaration matches boundary",
            test_privacy_declaration_matches_boundary,
        ),
        (
            "HTTP trace paths suppressed",
            test_http_models_do_not_expose_trace_path,
        ),
        (
            "Writer diagnostics remain internal",
            test_writer_diagnostics_are_not_publicly_consumed,
        ),
    ]


    for (
        label,
        test,
    ) in tests:

        test()

        print(
            f"[PASS] {label}"
        )


    print()

    print(
        "PASS - 7/7 AI trace "
        "privacy checks"
    )

    print(
        f"Rule: {TEST_RULE_VERSION}"
    )


if __name__ == "__main__":
    main()
