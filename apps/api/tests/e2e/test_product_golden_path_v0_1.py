from __future__ import annotations


import os
import sys
import tempfile


from pathlib import (
    Path,
)


from types import (
    SimpleNamespace,
)


from unittest.mock import (
    patch,
)


# ============================================================
# ISOLATED PRODUCT ENVIRONMENT
# ============================================================
#
# Environment variables must be configured BEFORE importing
# the DataLens application and persistence modules.
#
# Product E2E v0.1 must never read from or write to the
# developer's normal DataLens stores.
# ============================================================


_TEMP_DIRECTORY = tempfile.TemporaryDirectory(
    prefix="datalens-product-e2e-v0-1-"
)


_ROOT = Path(
    _TEMP_DIRECTORY.name
)


_DATABASE_PATH = (
    _ROOT
    /
    "datalens.sqlite3"
)


_PREPARATION_ARTIFACT_ROOT = (
    _ROOT
    /
    "preparation_artifacts"
)


_LEGACY_PREPARATION_SESSION_PATH = (
    _ROOT
    /
    "preparation_sessions.json"
)


_ANALYSIS_ARTIFACT_PATH = (
    _ROOT
    /
    "analysis_artifacts.json"
)


_REPORT_SELECTION_PATH = (
    _ROOT
    /
    "report_selection.json"
)


_RUNTIME_TRACE_PATH = (
    _ROOT
    /
    "runtime_requests.jsonl"
)


os.environ[
    "DATALENS_SQLITE_PATH"
] = str(
    _DATABASE_PATH
)


os.environ[
    "DATALENS_PREPARATION_ARTIFACT_STORE_PATH"
] = str(
    _PREPARATION_ARTIFACT_ROOT
)


os.environ[
    "DATALENS_PREPARATION_SESSION_STORE_PATH"
] = str(
    _LEGACY_PREPARATION_SESSION_PATH
)


os.environ[
    "DATALENS_ANALYSIS_ARTIFACT_STORE_PATH"
] = str(
    _ANALYSIS_ARTIFACT_PATH
)


os.environ[
    "DATALENS_REPORT_SELECTION_STORE_PATH"
] = str(
    _REPORT_SELECTION_PATH
)


# Runtime HTTP observability is independently tested elsewhere.
# Disable it here so the Product E2E owns only its intended
# temporary persistence surfaces.
os.environ[
    "DATALENS_RUNTIME_TRACE_ENABLED"
] = "0"


os.environ[
    "DATALENS_RUNTIME_TRACE_PATH"
] = str(
    _RUNTIME_TRACE_PATH
)


# ============================================================
# APPLICATION IMPORTS
# ============================================================


from fastapi.testclient import (
    TestClient,
)


from app.main import (
    app,
)


from app.planning.ai_analytical_planner import (
    AIPlannerProposal,
    RawAIPlannerOutput,
)


from app.ai.native_tool_calling import (
    NativeToolCallAttempt,
    NativeToolCallProposal,
    NativeToolCallRequestResult,
    expected_tool_arguments,
    native_tool_spec_for_contract,
)


from app.preparation.preparation_artifact_store import (
    PreparationArtifactStore,
    reset_preparation_artifact_store_for_tests,
)


from app.preparation.preparation_session import (
    PreparationSessionStore,
    reset_preparation_session_store_for_tests,
)


from app.reporting.analysis_artifact_store import (
    delete_analysis_artifacts,
)


from app.reporting.report_selection_store import (
    delete_report_selection,
)


# ============================================================
# PRODUCT CONTRACT
# ============================================================


PRODUCT_E2E_RULE_VERSION = (
    "product_e2e_benchmark_v0.1"
)


WORKFLOW_ROOT_DATASET_ID = (
    "dataset:0001"
)


OBJECTIVE = (
    "CA par catégorie"
)


PLANNER_MODEL = (
    "gemma3:4b"
)


TOOL_MODEL = (
    "qwen2.5:1.5b-instruct"
)


# ============================================================
# GOLDEN DATASET
# ============================================================
#
# Intentionally clean:
#
# - one dataset;
# - no duplicate row;
# - no missing value;
# - numeric gross_amount;
# - categorical category;
# - stable expected aggregation:
#
#   Accessories = 100 + 120 = 220
#   Electronics = 75 + 25 = 100
#
# This allows Product E2E v0.1 to exercise CLEAN skipped
# rather than introducing analyst approval into the first
# golden path.
# ============================================================


CSV_CONTENT = (
    "order_id,category,gross_amount\n"
    "O001,Accessories,100.0\n"
    "O002,Electronics,75.0\n"
    "O003,Accessories,120.0\n"
    "O004,Electronics,25.0\n"
)


# ============================================================
# RESET
# ============================================================


def reset_product_state() -> None:
    reset_preparation_session_store_for_tests()

    reset_preparation_artifact_store_for_tests()

    delete_analysis_artifacts()

    delete_report_selection()


# ============================================================
# HELPERS
# ============================================================


def require_dict(
    value,
    *,
    name: str,
) -> dict:
    if not isinstance(
        value,
        dict,
    ):
        raise AssertionError(
            (
                f"{name} must be a JSON object, "
                f"received {type(value).__name__}."
            )
        )

    return value


def require_list(
    value,
    *,
    name: str,
) -> list:
    if not isinstance(
        value,
        list,
    ):
        raise AssertionError(
            (
                f"{name} must be a JSON list, "
                f"received {type(value).__name__}."
            )
        )

    return value


def stage_map(
    session_body: dict,
) -> dict[
    str,
    dict,
]:
    snapshot = require_dict(
        session_body[
            "snapshot"
        ],
        name="snapshot",
    )

    stages = require_list(
        snapshot[
            "stages"
        ],
        name="snapshot.stages",
    )

    return {
        str(
            stage[
                "stage"
            ]
        ):
            require_dict(
                stage,
                name="stage",
            )

        for stage
        in stages
    }


# ============================================================
# DETERMINISTIC GEMMA BOUNDARY
# ============================================================
#
# Only model inference is replaced.
#
# Real production code still performs:
#
# - Planner Catalog resolution;
# - AIPlannerProposal validation;
# - contract construction;
# - semantic / structural validation;
# - native tool validation;
# - deterministic execution.
# ============================================================


def fake_generate_raw_ai_plan_with_timing(
    *,
    objective: str,
    catalog,
    model: str = PLANNER_MODEL,
    validation_feedback=None,
):
    del model
    del validation_feedback


    assert (
        objective.strip()
        ==
        OBJECTIVE
    )


    dataset_ids = [
        str(
            dataset.dataset_id
        )

        for dataset
        in catalog.datasets
    ]


    source_dataset_ids = [
        dataset_id

        for dataset_id
        in dataset_ids

        if not (
            dataset_id.startswith(
                "derived:"
            )
        )
    ]


    if (
        len(
            source_dataset_ids
        )
        !=
        1
    ):
        raise AssertionError(
            (
                "Product E2E v0.1 expects exactly one "
                "server-owned source dataset in the "
                "planner catalog; received "
                f"{source_dataset_ids!r}."
            )
        )


    source_dataset_id = (
        source_dataset_ids[
            0
        ]
    )


    proposal = (
        AIPlannerProposal(
            decision=
                "propose",

            title=
                "Chiffre d’affaires par catégorie",

            family=
                "aggregation",

            dataset_id=
                source_dataset_id,

            analytical_grain=
                "category",

            x_column=
                None,

            y_column=
                None,

            group_column=
                "category",

            value_column=
                "gross_amount",

            time_column=
                None,

            dimension_column=
                None,

            entity_column=
                None,

            aggregation_function=
                "sum",

            ranking_order=
                "none",

            ranking_limit=
                None,

            window_operation=
                "none",

            window_size=
                None,

            benchmark_reference=None,
            benchmark_operator=None,
            benchmark_selection=None,
            blockers=
                [],

            reasons=[
                (
                    "Deterministic Product E2E fixture. "
                    "Only Gemma inference is replaced; "
                    "the production Python planner must "
                    "validate the proposal."
                )
            ],

            confidence=
                0.95,
        )
    )


    return (
        RawAIPlannerOutput(
            proposals=[
                proposal
            ]
        ),
        0.0,
        0.0,
        0.0,
    )


# ============================================================
# DETERMINISTIC QWEN BOUNDARY
# ============================================================
#
# Qwen does not invent tool arguments in CI.
#
# The already validated production contract is converted into
# the expected proposal using DataLens production helpers.
#
# Production native-tool validation still runs afterwards.
# ============================================================


def fake_native_tool_request(
    *,
    contract,
    model: str = TOOL_MODEL,
) -> NativeToolCallRequestResult:
    del model


    spec = (
        native_tool_spec_for_contract(
            contract
        )
    )


    expected = (
        expected_tool_arguments(
            contract
        )
    )


    proposal = (
        NativeToolCallProposal(
            tool_name=
                spec.tool_name,

            arguments=
                expected.model_dump(),
        )
    )


    attempt = (
        NativeToolCallAttempt(
            attempt_index=
                1,

            prompt_variant=
                "standard",

            tool_call_count=
                1,

            assistant_content=
                "",

            selected_tool_name=
                spec.tool_name,

            errors=
                [],

            prompt_construction_ms=
                0.0,

            model_inference_ms=
                0.0,

            response_parse_ms=
                0.0,

            total_ms=
                0.0,
        )
    )


    return (
        NativeToolCallRequestResult(
            proposal=
                proposal,

            attempts=[
                attempt
            ],
        )
    )


# ============================================================
# AI TRACE BOUNDARY
# ============================================================
#
# AI observability has its own regression suite.
# Product E2E v0.1 keeps analysis/report persistence real but
# avoids creating unrelated trace files.
# ============================================================


def fake_trace_write(
    _trace,
):
    return (
        SimpleNamespace(
            enabled=
                False,

            written=
                False,

            error=
                None,
        )
    )


# ============================================================
# HTTP HELPER
# ============================================================


def post_analysis(
    client: TestClient,
    *,
    workflow_id: str,
):
    return (
        client.post(
            "/planning/ai-native-run",

            files={
                "workflow_id": (
                    None,
                    workflow_id,
                ),

                "objective": (
                    None,
                    OBJECTIVE,
                ),

                "planner_model": (
                    None,
                    PLANNER_MODEL,
                ),

                "tool_model": (
                    None,
                    TOOL_MODEL,
                ),
            },
        )
    )


# ============================================================
# 1. SESSION
# ============================================================


def create_product_session(
    client: TestClient,
) -> str:
    response = (
        client.post(
            "/preparation/sessions",

            json={
                "selected_analysis_dataset_ids": [
                    WORKFLOW_ROOT_DATASET_ID
                ]
            },
        )
    )


    assert (
        response.status_code
        ==
        201
    ), response.text


    body = (
        response.json()
    )


    workflow_id = str(
        body[
            "workflow_id"
        ]
    )


    assert (
        workflow_id.startswith(
            "prep:"
        )
    )


    assert (
        body[
            "snapshot"
        ][
            "next_stage"
        ]
        ==
        "import"
    )


    assert (
        body[
            "snapshot"
        ][
            "ready_for_analysis"
        ]
        is False
    )


    print(
        (
            "[PASS] server-owned Preparation "
            f"session created: {workflow_id}"
        )
    )


    return workflow_id


# ============================================================
# 2. REAL CSV → QUALITY
# ============================================================


def run_real_quality(
    client: TestClient,
    *,
    workflow_id: str,
) -> None:
    response = (
        client.post(
            "/preparation/quality",

            data={
                "workflow_id":
                    workflow_id,
            },

            files=[
                (
                    "dataset_files",
                    (
                        "sales.csv",
                        CSV_CONTENT,
                        "text/csv",
                    ),
                )
            ],
        )
    )


    assert (
        response.status_code
        ==
        200
    ), response.text


    session_response = (
        client.get(
            (
                "/preparation/sessions/"
                f"{workflow_id}"
            )
        )
    )


    assert (
        session_response.status_code
        ==
        200
    )


    session = (
        session_response.json()
    )


    stages = (
        stage_map(
            session
        )
    )


    for required_stage in [
        "import",
        "understand",
        "quality",
    ]:
        assert (
            stages[
                required_stage
            ][
                "status"
            ]
            ==
            "passed"
        )


    print(
        "[PASS] CSV crossed real IMPORT / UNDERSTAND / QUALITY"
    )


# ============================================================
# 3. CLEAN NOT REQUIRED
# ============================================================


def run_real_cleaning_plan(
    client: TestClient,
    *,
    workflow_id: str,
) -> None:
    response = (
        client.post(
            "/preparation/cleaning-plan",

            data={
                "workflow_id":
                    workflow_id,
            },

            files=[
                (
                    "dataset_files",
                    (
                        "sales.csv",
                        CSV_CONTENT,
                        "text/csv",
                    ),
                )
            ],
        )
    )


    assert (
        response.status_code
        ==
        200
    ), response.text


    body = (
        response.json()
    )


    assert (
        body[
            "action_count"
        ]
        ==
        0
    )


    assert (
        body[
            "protected_issue_count"
        ]
        ==
        0
    )


    session = (
        client.get(
            (
                "/preparation/sessions/"
                f"{workflow_id}"
            )
        )
        .json()
    )


    stages = (
        stage_map(
            session
        )
    )


    assert (
        stages[
            "clean"
        ][
            "status"
        ]
        ==
        "skipped"
    )


    assert (
        session[
            "snapshot"
        ][
            "next_stage"
        ]
        ==
        "validate"
    )


    print(
        "[PASS] CLEAN deterministically skipped for clean dataset"
    )


# ============================================================
# 4. ANALYSIS MUST FAIL BEFORE VALIDATE
# ============================================================


def require_analysis_blocked_before_validation(
    client: TestClient,
    *,
    workflow_id: str,
) -> None:
    response = (
        post_analysis(
            client,
            workflow_id=
                workflow_id,
        )
    )


    assert (
        response.status_code
        ==
        409
    ), response.text


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
        "analysis_not_ready"
    )


    print(
        "[PASS] Analysis gate blocks unvalidated Preparation"
    )


# ============================================================
# 5. SELECT REAL SERVER-OWNED OUTPUT
# ============================================================


def select_analysis_output(
    client: TestClient,
    *,
    workflow_id: str,
) -> None:
    candidates_response = (
        client.get(
            (
                "/preparation/sessions/"
                f"{workflow_id}"
                "/analysis-output-candidates"
            )
        )
    )


    assert (
        candidates_response.status_code
        ==
        200
    ), candidates_response.text


    candidates_body = (
        candidates_response.json()
    )


    candidates = (
        require_list(
            candidates_body[
                "candidates"
            ],
            name="analysis-output-candidates",
        )
    )


    matching = [
        candidate

        for candidate
        in candidates

        if (
            candidate[
                "dataset_id"
            ]
            ==
            WORKFLOW_ROOT_DATASET_ID
        )
    ]


    assert (
        len(
            matching
        )
        ==
        1
    )


    assert (
        matching[
            0
        ][
            "stage"
        ]
        ==
        "source"
    )


    selection_response = (
        client.post(
            "/preparation/analysis-output",

            json={
                "workflow_id":
                    workflow_id,

                "dataset_ids": [
                    WORKFLOW_ROOT_DATASET_ID
                ],
            },
        )
    )


    assert (
        selection_response.status_code
        ==
        200
    ), selection_response.text


    selection = (
        selection_response.json()
    )


    assert (
        selection[
            "analysis_output_dataset_ids"
        ]
        ==
        [
            WORKFLOW_ROOT_DATASET_ID
        ]
    )


    assert (
        selection[
            "snapshot"
        ][
            "ready_for_analysis"
        ]
        is False
    )


    assert (
        selection[
            "snapshot"
        ][
            "next_stage"
        ]
        ==
        "validate"
    )


    print(
        "[PASS] real server-owned source artifact selected"
    )


# ============================================================
# 6. FINAL VALIDATION
# ============================================================


def validate_preparation(
    client: TestClient,
    *,
    workflow_id: str,
) -> None:
    response = (
        client.post(
            "/preparation/validate",

            json={
                "workflow_id":
                    workflow_id
            },
        )
    )


    assert (
        response.status_code
        ==
        200
    ), response.text


    body = (
        response.json()
    )


    assert (
        body[
            "analysis_output_dataset_ids"
        ]
        ==
        [
            WORKFLOW_ROOT_DATASET_ID
        ]
    )


    assert (
        body[
            "snapshot"
        ][
            "validated_analysis_dataset_ids"
        ]
        ==
        [
            WORKFLOW_ROOT_DATASET_ID
        ]
    )


    assert (
        body[
            "snapshot"
        ][
            "ready_for_analysis"
        ]
        is True
    )


    assert (
        body[
            "snapshot"
        ][
            "next_stage"
        ]
        is None
    )


    assert (
        body[
            "snapshot"
        ][
            "blocking_reasons"
        ]
        ==
        []
    )


    print(
        "[PASS] Final Preparation Validation crossed"
    )


# ============================================================
# 7. DURABLE PREPARATION RESTORE
# ============================================================


def verify_preparation_persistence(
    *,
    workflow_id: str,
) -> None:
    assert (
        _DATABASE_PATH.exists()
    )


    fresh_session_store = (
        PreparationSessionStore()
    )


    restored_session = (
        fresh_session_store.get(
            workflow_id
        )
    )


    assert (
        restored_session
        .analysis_output_dataset_ids
        ==
        [
            WORKFLOW_ROOT_DATASET_ID
        ]
    )


    assert (
        restored_session
        .validate_stage
        .completed
        is True
    )


    assert (
        restored_session
        .validate_stage
        .passed
        is True
    )


    assert (
        restored_session
        .validate_stage
        .dataset_ids
        ==
        [
            WORKFLOW_ROOT_DATASET_ID
        ]
    )


    assert (
        restored_session
        .validate_stage
        .blocking_reasons
        ==
        []
    )


    fresh_artifact_store = (
        PreparationArtifactStore()
    )


    restored_dataframe = (
        fresh_artifact_store
        .get_dataframe(
            workflow_id=
                workflow_id,

            dataset_id=
                WORKFLOW_ROOT_DATASET_ID,
        )
    )


    assert (
        restored_dataframe.shape
        ==
        (
            4,
            3,
        )
    )


    assert (
        restored_dataframe[
            "category"
        ].tolist()
        ==
        [
            "Accessories",
            "Electronics",
            "Accessories",
            "Electronics",
        ]
    )


    assert (
        [
            float(
                value
            )

            for value
            in restored_dataframe[
                "gross_amount"
            ].tolist()
        ]
        ==
        [
            100.0,
            75.0,
            120.0,
            25.0,
        ]
    )


    print(
        "[PASS] fresh Preparation stores restore session + artifact"
    )


# ============================================================
# 8. REAL ANALYSIS PIPELINE
# ============================================================


def execute_real_analysis(
    client: TestClient,
    *,
    workflow_id: str,
) -> str:
    with (
        patch(
            (
                "app.planning.ai_analytical_planner."
                "_generate_raw_ai_plan_with_timing"
            ),
            side_effect=
                fake_generate_raw_ai_plan_with_timing,
        ),

        patch(
            (
                "app.ai.native_tool_calling."
                "request_native_tool_call"
            ),
            side_effect=
                fake_native_tool_request,
        ),

        patch(
            (
                "app.api.analysis_run."
                "write_ai_trace"
            ),
            side_effect=
                fake_trace_write,
        ),
    ):
        response = (
            post_analysis(
                client,
                workflow_id=
                    workflow_id,
            )
        )


    assert (
        response.status_code
        ==
        200
    ), response.text


    body = (
        response.json()
    )


    assert (
        body[
            "status"
        ]
        ==
        "ready"
    )


    assert (
        body[
            "executed_count"
        ]
        >=
        1
    )


    planner = (
        require_dict(
            body[
                "planner"
            ],
            name="planner",
        )
    )


    assert (
        planner[
            "validated_count"
        ]
        ==
        1
    )


    items = (
        require_list(
            body[
                "items"
            ],
            name="items",
        )
    )


    aggregation_items = [
        item

        for item
        in items

        if (
            isinstance(
                item,
                dict,
            )

            and

            item.get(
                "family"
            )
            ==
            "aggregation"
        )
    ]


    assert (
        len(
            aggregation_items
        )
        ==
        1
    )


    native_tool = (
        require_dict(
            aggregation_items[
                0
            ][
                "native_tool"
            ],
            name="native_tool",
        )
    )


    assert (
        native_tool[
            "requested_tool"
        ]
        ==
        "run_aggregation"
    )


    assert (
        native_tool[
            "validation_status"
        ]
        ==
        "validated"
    )


    execution = (
        require_dict(
            native_tool[
                "execution"
            ],
            name="execution",
        )
    )


    assert (
        execution[
            "execution_status"
        ]
        ==
        "executed"
    )


    result = (
        require_dict(
            execution[
                "result"
            ],
            name="execution.result",
        )
    )


    assert (
        result[
            "family"
        ]
        ==
        "aggregation"
    )


    assert (
        result[
            "execution_status"
        ]
        ==
        "complete"
    )


    assert (
        result[
            "chart_type"
        ]
        ==
        "bar"
    )


    chart_data = (
        require_list(
            result[
                "chart_data"
            ],
            name="chart_data",
        )
    )


    assert (
        len(
            chart_data
        )
        ==
        2
    )


    aggregated_values = sorted(
        round(
            float(
                row[
                    "value"
                ]
            ),
            2,
        )

        for row
        in chart_data
    )


    assert (
        aggregated_values
        ==
        [
            100.0,
            220.0,
        ]
    )


    analysis_id = (
        body.get(
            "analysis_id"
        )
    )


    assert (
        isinstance(
            analysis_id,
            str,
        )

        and

        bool(
            analysis_id
        )
    )


    assert (
        analysis_id.startswith(
            "analysis:"
        )
    )


    print(
        (
            "[PASS] real Analysis Handoff → planner validation "
            "→ native tool validation → executor"
        )
    )

    print(
        (
            "[PASS] deterministic aggregation values "
            "100.0 / 220.0"
        )
    )

    print(
        f"[PASS] server-owned analysis artifact: {analysis_id}"
    )


    return analysis_id


# ============================================================
# 9. REPORT SELECTION
# ============================================================

def verify_report_selection(
    client: TestClient,
    *,
    workflow_id: str,
    analysis_id: str,
) -> None:
    # ========================================================
    # ANALYSIS MUST EXIST IN SERVER-OWNED HISTORY
    # ========================================================

    analyses_response = (
        client.get(
            "/report/analyses/details",

            params={
                "workflow_id":
                    workflow_id,
            },
        )
    )


    assert (
        analyses_response.status_code
        ==
        200
    ), analyses_response.text


    available = (
        analyses_response.json()
    )


    available_analyses = (
        require_list(
            available[
                "analyses"
            ],
            name="available report analyses",
        )
    )


    matching_available = [
        analysis

        for analysis
        in available_analyses

        if (
            analysis[
                "analysis_id"
            ]
            ==
            analysis_id
        )
    ]


    assert (
        len(
            matching_available
        )
        ==
        1
    )


    assert (
        matching_available[
            0
        ][
            "executed"
        ]
        is True
    )


    print(
        "[PASS] executed analysis available in server-owned history"
    )


    # ========================================================
    # MANUAL-ONLY REPORT POLICY
    # ========================================================
    #
    # Analysis execution and report composition are deliberately
    # separated in AnalysisArtifact Store v0.3.
    #
    # A successful analysis must therefore NOT appear in the
    # report until the user explicitly selects it.
    # ========================================================

    before_response = (
        client.get(
            "/report/selection/details",

            params={
                "workflow_id":
                    workflow_id,
            },
        )
    )


    assert (
        before_response.status_code
        ==
        200
    ), before_response.text


    before = (
        before_response.json()
    )


    assert (
        before[
            "selected_count"
        ]
        ==
        0
    )


    assert (
        before[
            "analyses"
        ]
        ==
        []
    )


    print(
        "[PASS] analysis execution does not mutate report composition"
    )


    # ========================================================
    # PDF MUST FAIL WHILE SELECTION IS EMPTY
    # ========================================================

    empty_pdf_response = (
        client.post(
            "/report/export-pdf",

            json={
                "workflow_id":
                    workflow_id,
            },
        )
    )


    assert (
        empty_pdf_response.status_code
        ==
        409
    ), empty_pdf_response.text


    empty_pdf_detail = (
        empty_pdf_response
        .json()[
            "detail"
        ]
    )


    assert (
        empty_pdf_detail[
            "error"
        ]
        ==
        "report_selection_empty"
    )


    print(
        "[PASS] PDF export fails closed with empty report selection"
    )


    # ========================================================
    # EXPLICIT USER SELECTION
    # ========================================================

    add_response = (
        client.post(
            "/report/selection/add",

            json={
                "workflow_id":
                    workflow_id,

                "analysis_id":
                    analysis_id,
            },
        )
    )


    assert (
        add_response.status_code
        ==
        200
    ), add_response.text


    added = (
        add_response.json()
    )


    assert (
        added[
            "selected_count"
        ]
        ==
        1
    )


    assert (
        len(
            added[
                "analyses"
            ]
        )
        ==
        1
    )


    assert (
        added[
            "analyses"
        ][
            0
        ][
            "analysis_id"
        ]
        ==
        analysis_id
    )


    assert (
        added[
            "analyses"
        ][
            0
        ][
            "executed"
        ]
        is True
    )


    print(
        "[PASS] explicit server-owned report selection committed"
    )


    # ========================================================
    # DETAILS AFTER SELECTION
    # ========================================================

    details_response = (
        client.get(
            "/report/selection/details",

            params={
                "workflow_id":
                    workflow_id,
            },
        )
    )


    assert (
        details_response.status_code
        ==
        200
    ), details_response.text


    details = (
        details_response.json()
    )


    assert (
        details[
            "selected_count"
        ]
        ==
        1
    )


    analyses = (
        require_list(
            details[
                "analyses"
            ],
            name="report analyses",
        )
    )


    assert (
        len(
            analyses
        )
        ==
        1
    )


    detail = (
        require_dict(
            analyses[
                0
            ],
            name="selected analysis detail",
        )
    )


    selection = (
        require_dict(
            detail[
                "selection"
            ],
            name="selection",
        )
    )


    assert (
        selection[
            "analysis_id"
        ]
        ==
        analysis_id
    )


    assert (
        selection[
            "executed"
        ]
        is True
    )


    assert (
        selection[
            "report_order"
        ]
        ==
        1
    )


    pipeline_payload = (
        require_dict(
            detail[
                "pipeline_payload"
            ],
            name="pipeline_payload",
        )
    )


    planner = (
        require_dict(
            pipeline_payload[
                "planner"
            ],
            name="pipeline_payload.planner",
        )
    )


    assert (
        planner[
            "objective"
        ]
        ==
        OBJECTIVE
    )


    assert (
        pipeline_payload[
            "analysis_id"
        ]
        ==
        analysis_id
    )


    print(
        "[PASS] selected report detail restores persisted pipeline payload"
    )


# ============================================================
# 10. REAL PDF
# ============================================================


def export_real_pdf(
    client: TestClient,
    *,
    workflow_id: str,
) -> None:
    response = (
        client.post(
            "/report/export-pdf",

            json={
                "workflow_id":
                    workflow_id,
            },
        )
    )


    assert (
        response.status_code
        ==
        200
    ), response.text


    assert (
        response.headers[
            "content-type"
        ]
        ==
        "application/pdf"
    )


    assert (
        response.headers[
            "x-datalens-report-selection-count"
        ]
        ==
        "1"
    )


    assert (
        response.content[
            :
            4
        ]
        ==
        b"%PDF"
    )


    assert (
        len(
            response.content
        )
        >
        3000
    )


    print(
        (
            "[PASS] real server-owned PDF generated "
            f"({len(response.content)} bytes)"
        )
    )


# ============================================================
# GOLDEN PATH
# ============================================================


def test_product_golden_path_v0_1() -> None:
    reset_product_state()


    with TestClient(
        app
    ) as client:
        workflow_id = (
            create_product_session(
                client
            )
        )


        run_real_quality(
            client,
            workflow_id=
                workflow_id,
        )


        run_real_cleaning_plan(
            client,
            workflow_id=
                workflow_id,
        )


        require_analysis_blocked_before_validation(
            client,
            workflow_id=
                workflow_id,
        )


        select_analysis_output(
            client,
            workflow_id=
                workflow_id,
        )


        validate_preparation(
            client,
            workflow_id=
                workflow_id,
        )


        verify_preparation_persistence(
            workflow_id=
                workflow_id,
        )


        analysis_id = (
            execute_real_analysis(
                client,
                workflow_id=
                    workflow_id,
            )
        )


        verify_report_selection(
            client,
            workflow_id=
                workflow_id,

            analysis_id=
                analysis_id,
        )


        export_real_pdf(
            client,
            workflow_id=
                workflow_id,
        )


# ============================================================
# VERSION
# ============================================================


def test_product_e2e_rule_version() -> None:
    assert (
        PRODUCT_E2E_RULE_VERSION
        ==
        "product_e2e_benchmark_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> int:
    print()
    print(
        "=" * 78
    )

    print(
        "DATALENS PRODUCT E2E BENCHMARK v0.1"
    )

    print(
        "=" * 78
    )

    print(
        "Preparation : real FastAPI + SQLite + Artifact Store"
    )

    print(
        "Analysis    : real validated server-owned handoff"
    )

    print(
        "Gemma       : deterministic inference boundary"
    )

    print(
        "Qwen        : deterministic inference boundary"
    )

    print(
        "Executor    : real production deterministic engine"
    )

    print(
        "Reporting   : real Analysis Artifact + selection + PDF"
    )

    print()


    try:
        test_product_e2e_rule_version()

        print(
            "[PASS] Product E2E rule version"
        )


        test_product_golden_path_v0_1()


    except Exception as error:
        print()
        print(
            "=" * 78
        )

        print(
            (
                "FAIL - Product E2E Benchmark v0.1"
            )
        )

        print(
            (
                f"{type(error).__name__}: "
                f"{error}"
            )
        )

        print(
            "=" * 78
        )

        return 1


    print()
    print(
        "=" * 78
    )

    print(
        (
            "PASS - CSV → Preparation → VALIDATE → "
            "Analysis → Artifact → Report Selection → PDF"
        )
    )

    print(
        "=" * 78
    )


    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )