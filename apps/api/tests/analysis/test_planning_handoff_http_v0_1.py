from __future__ import annotations


from io import (
    StringIO,
)


from typing import (
    Callable,
    Optional,
)


import pandas as pd


from fastapi.testclient import (
    TestClient,
)


from main import (
    app,
)


import app.api.analysis_run as analysis_run_module


from app.preparation.preparation_artifact_store import (
    put_preparation_artifact,
    reset_preparation_artifact_store_for_tests,
)


from app.preparation.preparation_session import (
    create_preparation_session,
    get_preparation_session,
    record_analysis_output_selection,
    record_required_stage_signal,
    record_validation_stage_signal,
    reset_preparation_session_store_for_tests,
)


from app.preparation.preparation_workflow import (
    PreparationStage,
)


# ============================================================
# CLIENT
# ============================================================


client = TestClient(
    app
)


# ============================================================
# ROUTES UNDER TEST
# ============================================================


PLANNING_ROUTES = [
    "/planning/ai-preview",
    "/planning/ai-tool-run",
    "/planning/ai-native-run",
]


# ============================================================
# FIXTURE DATA
# ============================================================


VALIDATED_DATASET_CSV = """order_id,customer_segment,amount,quantity
O001,Premium,100,2
O002,Standard,75,1
O003,Premium,140,3
O004,Basic,45,1
O005,Standard,90,2
O006,Premium,180,4
O007,Basic,55,1
O008,Standard,110,2
O009,Premium,160,3
O010,Basic,60,1
O011,Standard,95,2
O012,Premium,210,4
"""


BROWSER_DATASET_CSV = """browser_only,poisoned_value
DO_NOT_ANALYZE,999999
"""


# ============================================================
# TEST SENTINEL
# ============================================================


class PlannerReached(Exception):
    """
    Test-only sentinel.

    The planning route is deliberately interrupted when the
    validated server-side dataset has successfully crossed the
    Preparation -> Analysis handoff and the analytical planner
    is about to run.

    This prevents the test from calling Ollama / Gemma / Qwen.
    """


# ============================================================
# DATAFRAME
# ============================================================


def validated_frame() -> pd.DataFrame:
    return (
        pd.read_csv(
            StringIO(
                VALIDATED_DATASET_CSV
            )
        )
    )


# ============================================================
# MULTIPART HELPERS
# ============================================================


def dataset_files():
    """
    Deliberately poisoned browser payload.

    The uploaded file is intentionally unrelated to the
    server-owned validated Preparation artifact.

    Once VALIDATE has passed, planning routes must not read
    these dataset bytes.
    """

    return [
        (
            "dataset_files",
            (
                "browser_payload.csv",
                BROWSER_DATASET_CSV.encode(
                    "utf-8"
                ),
                "text/csv",
            ),
        )
    ]


# ============================================================
# RESET
# ============================================================


def reset_state() -> None:
    reset_preparation_session_store_for_tests()

    reset_preparation_artifact_store_for_tests()


# ============================================================
# SESSION HELPERS
# ============================================================


def create_session(
    dataset_id: str = (
        "dataset:0001"
    ),
):
    return (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                dataset_id
            ]
        )
    )


def make_session_ready(
    *,
    workflow_id: str,
    dataset_id: str = (
        "dataset:0001"
    ),
):
    """
    Reproduces the same validated Preparation handoff used by
    the existing Analysis readiness HTTP test.
    """

    # ========================================================
    # IMPORT
    # ========================================================

    record_required_stage_signal(
        workflow_id=
            workflow_id,

        stage=
            PreparationStage.IMPORT,

        completed=
            True,

        dataset_ids=[
            dataset_id
        ],

        evidence_refs=[
            "csv_ingestion"
        ],

        blocking_reasons=[],
    )


    # ========================================================
    # UNDERSTAND
    # ========================================================

    record_required_stage_signal(
        workflow_id=
            workflow_id,

        stage=
            PreparationStage.UNDERSTAND,

        completed=
            True,

        dataset_ids=[
            dataset_id
        ],

        evidence_refs=[
            "dataset_profile"
        ],

        blocking_reasons=[],
    )


    # ========================================================
    # QUALITY
    # ========================================================

    record_required_stage_signal(
        workflow_id=
            workflow_id,

        stage=
            PreparationStage.QUALITY,

        completed=
            True,

        dataset_ids=[
            dataset_id
        ],

        evidence_refs=[
            "data_quality_engine_v0.2"
        ],

        blocking_reasons=[],
    )


    # ========================================================
    # SERVER-OWNED ARTIFACT
    # ========================================================

    put_preparation_artifact(
        workflow_id=
            workflow_id,

        dataset_id=
            dataset_id,

        dataset_filename=
            "orders_prepared.csv",

        stage=
            "source",

        dataframe=
            validated_frame(),

        parent_dataset_ids=[],

        evidence_refs=[
            "test:validated-artifact",
        ],
    )


    # ========================================================
    # FINAL ANALYSIS OUTPUT SELECTION
    # ========================================================

    before_selection = (
        get_preparation_session(
            workflow_id
        )
    )


    selected = (
        record_analysis_output_selection(
            workflow_id=
                workflow_id,

            analysis_output_dataset_ids=[
                dataset_id
            ],

            expected_revision=
                before_selection.revision,
        )
    )


    # ========================================================
    # VALIDATE
    # ========================================================

    ready = (
        record_validation_stage_signal(
            workflow_id=
                workflow_id,

            completed=
                True,

            passed=
                True,

            dataset_ids=[
                dataset_id
            ],

            evidence_refs=[
                "final_validation"
            ],

            blocking_reasons=[],

            expected_revision=
                selected.revision,
        )
    )


    assert (
        ready
        .snapshot
        .ready_for_analysis
        is True
    )


    return ready


# ============================================================
# OPENAPI HELPERS
# ============================================================


def resolve_openapi_schema(
    *,
    root_schema: dict,
    schema: dict,
) -> dict:
    """
    Resolve the common FastAPI $ref / allOf shapes used for
    multipart request bodies.
    """

    reference = (
        schema.get(
            "$ref"
        )
    )


    if (
        reference
    ):
        prefix = (
            "#/components/schemas/"
        )


        assert (
            reference.startswith(
                prefix
            )
        )


        schema_name = (
            reference[
                len(
                    prefix
                ):
            ]
        )


        return (
            root_schema[
                "components"
            ][
                "schemas"
            ][
                schema_name
            ]
        )


    all_of = (
        schema.get(
            "allOf"
        )
    )


    if (
        isinstance(
            all_of,
            list,
        )
        and
        len(
            all_of
        )
        ==
        1
    ):
        return (
            resolve_openapi_schema(
                root_schema=
                    root_schema,

                schema=
                    all_of[
                        0
                    ],
            )
        )


    return schema


def multipart_schema_for_route(
    path: str,
) -> dict:
    root_schema = (
        app.openapi()
    )


    operation = (
        root_schema[
            "paths"
        ][
            path
        ][
            "post"
        ]
    )


    request_body = (
        operation[
            "requestBody"
        ]
    )


    multipart_schema = (
        request_body[
            "content"
        ][
            "multipart/form-data"
        ][
            "schema"
        ]
    )


    return (
        resolve_openapi_schema(
            root_schema=
                root_schema,

            schema=
                multipart_schema,
        )
    )


# ============================================================
# OPENAPI WORKFLOW CONTRACT
# ============================================================


def test_workflow_id_required_on_all_planning_routes():
    """
    Every visible planning/execution route must be attached to
    one validated Preparation workflow.

    EXPECTED BEFORE MIGRATION:
        FAIL

    EXPECTED AFTER MIGRATION:
        PASS
    """

    for path in (
        PLANNING_ROUTES
    ):
        body_schema = (
            multipart_schema_for_route(
                path
            )
        )


        properties = (
            body_schema.get(
                "properties",
                {},
            )
        )


        required = (
            body_schema.get(
                "required",
                [],
            )
        )


        assert (
            "workflow_id"
            in
            properties
        ), (
            f"{path} does not expose workflow_id "
            "in its multipart contract."
        )


        assert (
            "workflow_id"
            in
            required
        ), (
            f"{path} exposes workflow_id but does "
            "not require it."
        )


    print(
        "\n=== PLANNING OPENAPI HANDOFF CONTRACT ==="
    )


    print(
        (
            "workflow_id is required on ai-preview, "
            "ai-tool-run and ai-native-run: True"
        )
    )


# ============================================================
# BROWSER UPLOAD BYPASS
# ============================================================


def assert_browser_upload_not_loaded_for_route(
    path: str,
) -> None:
    """
    Verify that a validated workflow-backed planning route does
    not use browser-uploaded dataset bytes.

    Before migration:
        load_uploaded_dataset_bundle() is reached -> FAIL.

    After migration:
        validated Artifact Store input is loaded and the
        analytical planner is reached -> PASS.

    The planner itself is interrupted so that no local LLM is
    required for this HTTP security test.
    """

    reset_state()


    session = (
        create_session()
    )


    make_session_ready(
        workflow_id=
            session.workflow_id,
    )


    original_loader = (
        analysis_run_module
        .load_uploaded_dataset_bundle
    )


    original_planner = (
        analysis_run_module
        .plan_analyses_with_intent_routing
    )


    state = {
        "upload_loader_called":
            False,

        "planner_reached":
            False,
    }


    def forbidden_upload_loader(
        *_args,
        **_kwargs,
    ):
        state[
            "upload_loader_called"
        ] = True


        raise AssertionError(
            (
                f"{path} attempted to read browser-uploaded "
                "dataset bytes after VALIDATE."
            )
        )


    def stop_at_planner(
        *_args,
        **_kwargs,
    ):
        state[
            "planner_reached"
        ] = True


        raise PlannerReached()


    analysis_run_module.load_uploaded_dataset_bundle = (
        forbidden_upload_loader
    )


    analysis_run_module.plan_analyses_with_intent_routing = (
        stop_at_planner
    )


    planner_interrupted = (
        False
    )


    try:
        try:
            client.post(
                path,

                files=
                    dataset_files(),

                data={
                    "workflow_id":
                        session.workflow_id,

                    "objective":
                        "Analyser les commandes",
                },
            )


        except PlannerReached:
            planner_interrupted = (
                True
            )


    finally:
        analysis_run_module.load_uploaded_dataset_bundle = (
            original_loader
        )


        analysis_run_module.plan_analyses_with_intent_routing = (
            original_planner
        )


    assert (
        state[
            "upload_loader_called"
        ]
        is False
    ), (
        f"{path} still trusts browser dataset bytes "
        "after Preparation VALIDATE."
    )


    assert (
        state[
            "planner_reached"
        ]
        is True
    ), (
        f"{path} did not reach analytical planning "
        "through the validated Preparation artifact."
    )


    assert (
        planner_interrupted
        is True
    )


    print(
        (
            f"{path}: browser dataset ignored; "
            "validated Artifact Store input reached planner."
        )
    )


# ============================================================
# AI PREVIEW
# ============================================================


def test_ai_preview_does_not_load_browser_dataset_after_validate():
    assert_browser_upload_not_loaded_for_route(
        "/planning/ai-preview"
    )


# ============================================================
# AI TOOL RUN
# ============================================================


def test_ai_tool_run_does_not_load_browser_dataset_after_validate():
    assert_browser_upload_not_loaded_for_route(
        "/planning/ai-tool-run"
    )


# ============================================================
# AI NATIVE RUN
# ============================================================


def test_ai_native_run_does_not_load_browser_dataset_after_validate():
    assert_browser_upload_not_loaded_for_route(
        "/planning/ai-native-run"
    )


# ============================================================
# ROUTES PRESERVED
# ============================================================


def test_planning_routes_are_preserved():
    paths = (
        app.openapi()[
            "paths"
        ]
    )


    for path in (
        PLANNING_ROUTES
    ):
        assert (
            path
            in
            paths
        )


    print(
        "\n=== PLANNING ROUTES PRESERVED ==="
    )


    print(
        "All three planning routes are present: True"
    )


# ============================================================
# TEST RUNNER
# ============================================================


def run_case(
    *,
    name: str,

    function: Callable[
        [],
        None,
    ],
) -> Optional[
    str
]:
    try:
        function()


        print(
            f"[PASS] {name}"
        )


        return None


    except Exception as error:
        print(
            f"[FAIL] {name}"
        )


        print(
            (
                f"       "
                f"{type(error).__name__}: "
                f"{error}"
            )
        )


        return (
            f"{name}: "
            f"{type(error).__name__}: "
            f"{error}"
        )


# ============================================================
# MAIN
# ============================================================


def main():
    reset_state()


    print(
        "\n========================================"
    )


    print(
        "DataLens Planning Handoff HTTP v0.1"
    )


    print(
        "========================================"
    )


    failures = []


    cases = [
        (
            "planning routes preserved",
            test_planning_routes_are_preserved,
        ),
        (
            "workflow_id required on planning routes",
            test_workflow_id_required_on_all_planning_routes,
        ),
        (
            "ai-preview ignores browser dataset",
            test_ai_preview_does_not_load_browser_dataset_after_validate,
        ),
        (
            "ai-tool-run ignores browser dataset",
            test_ai_tool_run_does_not_load_browser_dataset_after_validate,
        ),
        (
            "ai-native-run ignores browser dataset",
            test_ai_native_run_does_not_load_browser_dataset_after_validate,
        ),
    ]


    for (
        name,
        function,
    ) in cases:
        failure = (
            run_case(
                name=
                    name,

                function=
                    function,
            )
        )


        if (
            failure
            is not None
        ):
            failures.append(
                failure
            )


    reset_state()


    print(
        "\n========================================"
    )


    if (
        failures
    ):
        print(
            (
                "EXPECTED RED STATE - "
                f"{len(failures)} planning handoff "
                "guard(s) currently fail."
            )
        )


        for failure in (
            failures
        ):
            print(
                f"- {failure}"
            )


        print(
            "========================================"
        )


        raise AssertionError(
            (
                "Planning endpoints are not yet fully "
                "workflow-backed."
            )
        )


    print(
        "PASS - planning handoff HTTP v0.1"
    )


    print(
        "========================================"
    )


if __name__ == "__main__":
    main()