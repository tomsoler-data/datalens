from __future__ import annotations


from io import (
    StringIO,
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
    delete_preparation_artifacts,
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


DOCUMENT_TEXT = """
Analyse des commandes clients.
Les segments Premium, Standard et Basic doivent être
conservés comme catégories métier distinctes.
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
    Deliberately different from the validated Preparation
    artifact.

    The workflow-backed Analysis route must not read this
    content after VALIDATE.
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


def contextualized_files():
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
        ),
        (
            "document_files",
            (
                "business_rules.txt",
                DOCUMENT_TEXT.encode(
                    "utf-8"
                ),
                "text/plain",
            ),
        ),
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
    materialize_artifact: bool = True,
):
    # ========================================================
    # REQUIRED PREPARATION STAGES
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
    # SERVER-OWNED MATERIALIZED ARTIFACT
    # ========================================================

    if (
        materialize_artifact
    ):
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

    return (
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


# ============================================================
# OPENAPI CONTRACT
# ============================================================


def test_workflow_id_required_on_both_analysis_routes():
    schema = (
        app.openapi()
    )


    paths = (
        schema[
            "paths"
        ]
    )


    for path in [
        "/analysis/run",
        "/analysis/run-contextualized",
    ]:
        operation = (
            paths[
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


        assert (
            request_body[
                "required"
            ]
            is True
        )


    print(
        "\n=== OPENAPI ANALYSIS HANDOFF CONTRACT ==="
    )

    print(
        "workflow_id remains required on both routes: True"
    )


# ============================================================
# MISSING WORKFLOW ID
# ============================================================


def test_standard_analysis_requires_workflow_id():
    response = (
        client.post(
            "/analysis/run",

            files=
                dataset_files(),

            data={
                "objective":
                    "Analyser les commandes"
            },
        )
    )


    print(
        "\n=== MISSING WORKFLOW ID ==="
    )

    print(
        f"Status: "
        f"{response.status_code}"
    )


    assert (
        response.status_code
        ==
        422
    )


# ============================================================
# UNKNOWN SESSION
# ============================================================


def test_unknown_session_returns_404():
    response = (
        client.post(
            "/analysis/run",

            files=
                dataset_files(),

            data={
                "workflow_id":
                    "prep:does-not-exist",

                "objective":
                    "Analyser les commandes",
            },
        )
    )


    print(
        "\n=== UNKNOWN PREPARATION SESSION ==="
    )

    print(
        f"Status: "
        f"{response.status_code}"
    )


    assert (
        response.status_code
        ==
        404
    )


    body = (
        response.json()
    )


    assert (
        body[
            "detail"
        ][
            "error"
        ]
        ==
        "preparation_session_not_found"
    )


# ============================================================
# SESSION NOT READY
# ============================================================


def test_unready_session_returns_409():
    session = (
        create_session()
    )


    response = (
        client.post(
            "/analysis/run",

            files=
                dataset_files(),

            data={
                "workflow_id":
                    session.workflow_id,

                "objective":
                    "Analyser les commandes",
            },
        )
    )


    print(
        "\n=== PREPARATION NOT READY ==="
    )

    print(
        f"Status: "
        f"{response.status_code}"
    )


    assert (
        response.status_code
        ==
        409
    )


    body = (
        response.json()
    )


    detail = (
        body[
            "detail"
        ]
    )


    print(
        f"Error: "
        f"{detail['error']}"
    )

    print(
        f"Next stage: "
        f"{detail['next_stage']}"
    )


    assert (
        detail[
            "error"
        ]
        ==
        "analysis_not_ready"
    )


    assert (
        detail[
            "ready_for_analysis"
        ]
        is False
    )


    assert (
        detail[
            "next_stage"
        ]
        ==
        "import"
    )


# ============================================================
# CONTEXTUALIZED ROUTE IS ALSO GATED
# ============================================================


def test_contextualized_analysis_is_gated_before_rag():
    session = (
        create_session()
    )


    response = (
        client.post(
            "/analysis/run-contextualized",

            files=
                contextualized_files(),

            data={
                "workflow_id":
                    session.workflow_id,

                "objective":
                    "Analyser les commandes avec contexte",
            },
        )
    )


    print(
        "\n=== CONTEXTUALIZED ANALYSIS GATE ==="
    )

    print(
        f"Status: "
        f"{response.status_code}"
    )


    assert (
        response.status_code
        ==
        409
    )


    body = (
        response.json()
    )


    assert (
        body[
            "detail"
        ][
            "error"
        ]
        ==
        "analysis_not_ready"
    )


# ============================================================
# READY STANDARD ANALYSIS
# ============================================================


def test_ready_session_allows_standard_analysis():
    session = (
        create_session()
    )


    ready = (
        make_session_ready(
            workflow_id=
                session.workflow_id,
        )
    )


    assert (
        ready
        .snapshot
        .ready_for_analysis
        is True
    )


    response = (
        client.post(
            "/analysis/run",

            files=
                dataset_files(),

            data={
                "workflow_id":
                    session.workflow_id,

                "objective":
                    "Analyser les commandes",
            },
        )
    )


    print(
        "\n=== READY STANDARD ANALYSIS ==="
    )

    print(
        f"Status: "
        f"{response.status_code}"
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
        "report_rule_version"
        in
        body
    )


    assert (
        "datasets"
        in
        body
    )


# ============================================================
# BROWSER UPLOAD CANNOT REPLACE VALIDATED ARTIFACT
# ============================================================


def test_browser_upload_is_not_loaded_after_validate():
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


    def forbidden_upload_loader(
        *_args,
        **_kwargs,
    ):
        raise AssertionError(
            (
                "Workflow-backed Analysis must not read "
                "browser-uploaded dataset bytes after "
                "VALIDATE."
            )
        )


    analysis_run_module.load_uploaded_dataset_bundle = (
        forbidden_upload_loader
    )


    try:
        response = (
            client.post(
                "/analysis/run",

                files=
                    dataset_files(),

                data={
                    "workflow_id":
                        session.workflow_id,

                    "objective":
                        "Analyser les commandes",
                },
            )
        )


    finally:
        analysis_run_module.load_uploaded_dataset_bundle = (
            original_loader
        )


    print(
        "\n=== BROWSER DATASET BYPASS ATTEMPT ==="
    )

    print(
        f"Status: "
        f"{response.status_code}"
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
        "datasets"
        in
        body
    )


    print(
        (
            "Browser upload was not loaded; validated "
            "Artifact Store input remained authoritative: "
            "True"
        )
    )


# ============================================================
# POST-VALIDATION CLEANING OVERRIDE IS REJECTED
# ============================================================


def test_post_validation_cleaning_override_is_rejected():
    session = (
        create_session()
    )


    make_session_ready(
        workflow_id=
            session.workflow_id,
    )


    response = (
        client.post(
            "/analysis/run",

            files=
                dataset_files(),

            data={
                "workflow_id":
                    session.workflow_id,

                "objective":
                    "Analyser les commandes",

                "approved_action_ids_json":
                    '["clean:invented-action"]',
            },
        )
    )


    print(
        "\n=== POST-VALIDATION PREPARATION OVERRIDE ==="
    )

    print(
        f"Status: "
        f"{response.status_code}"
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
        "after VALIDATE"
        in
        detail
    )


# ============================================================
# MISSING VALIDATED ARTIFACT FAILS CLOSED
# ============================================================


def test_missing_validated_artifact_returns_409():
    session = (
        create_session()
    )


    make_session_ready(
        workflow_id=
            session.workflow_id,
    )


    delete_preparation_artifacts(
        workflow_id=
            session.workflow_id
    )


    response = (
        client.post(
            "/analysis/run",

            files=
                dataset_files(),

            data={
                "workflow_id":
                    session.workflow_id,

                "objective":
                    "Analyser les commandes",
            },
        )
    )


    print(
        "\n=== MISSING VALIDATED ARTIFACT ==="
    )

    print(
        f"Status: "
        f"{response.status_code}"
    )


    assert (
        response.status_code
        ==
        409
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
        "validated_analysis_artifact_unavailable"
    )


    assert (
        detail[
            "dataset_id"
        ]
        ==
        "dataset:0001"
    )


# ============================================================
# EMPTY LEGACY OVERRIDES REMAIN COMPATIBLE
# ============================================================


def test_empty_legacy_override_payloads_are_accepted():
    session = (
        create_session()
    )


    make_session_ready(
        workflow_id=
            session.workflow_id,
    )


    response = (
        client.post(
            "/analysis/run",

            files=
                dataset_files(),

            data={
                "workflow_id":
                    session.workflow_id,

                "objective":
                    "Analyser les commandes",

                "approved_action_ids_json":
                    "[]",

                "semantic_decisions_json":
                    "[]",

                "approved_semantic_choices_json":
                    "[]",
            },
        )
    )


    print(
        "\n=== EMPTY LEGACY OVERRIDES ==="
    )

    print(
        f"Status: "
        f"{response.status_code}"
    )


    assert (
        response.status_code
        ==
        200
    )


# ============================================================
# EXISTING ROUTES STILL PRESENT
# ============================================================


def test_analysis_routes_preserved():
    paths = (
        app.openapi()[
            "paths"
        ]
    )


    assert (
        "/analysis/run"
        in
        paths
    )


    assert (
        "/analysis/run-contextualized"
        in
        paths
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
        "DataLens Analysis Readiness HTTP v0.2"
    )

    print(
        "========================================"
    )


    test_workflow_id_required_on_both_analysis_routes()

    test_standard_analysis_requires_workflow_id()

    test_unknown_session_returns_404()

    test_unready_session_returns_409()

    test_contextualized_analysis_is_gated_before_rag()

    test_ready_session_allows_standard_analysis()

    test_browser_upload_is_not_loaded_after_validate()

    test_post_validation_cleaning_override_is_rejected()

    test_missing_validated_artifact_returns_409()

    test_empty_legacy_override_payloads_are_accepted()

    test_analysis_routes_preserved()


    print(
        "\n========================================"
    )

    print(
        "PASS - analysis readiness HTTP v0.2"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()
