from __future__ import annotations


import json


from pathlib import (
    Path,
)


from fastapi.testclient import (
    TestClient,
)


from starlette.datastructures import (
    UploadFile,
)


from main import (
    app,
)


from app.api.analysis_run import (
    build_entity_outlier_finding_if_requested,
)


from app.api.routes import (
    load_uploaded_dataset_bundle,
)


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
# CONFIGURATION
# ============================================================


DATA_DIR = Path(
    r"C:\Users\tomas\Documents\openclassrooms_data_analyst\Projet_9\Documents\Data"
)


DATASET_FILENAMES = [
    "customers.csv",
    "products.csv",
    "Transactions.csv",
]


DATASET_IDS = [
    "dataset:0001",
    "dataset:0002",
    "dataset:0003",
]


EXPECTED_PRIORITY_CLIENTS = {
    "c_1609",
    "c_3454",
    "c_4958",
    "c_6714",
}


client = TestClient(
    app
)


# ============================================================
# FIXTURE HELPERS
# ============================================================


def assert_source_files_exist() -> None:
    missing = [
        filename

        for filename
        in DATASET_FILENAMES

        if not (
            DATA_DIR
            /
            filename
        ).exists()
    ]


    if missing:
        raise FileNotFoundError(
            (
                "Lapage source file(s) missing: "
                f"{missing}. "
                f"Expected directory: {DATA_DIR}"
            )
        )


def lapage_files() -> list[
    tuple[
        str,
        tuple[
            str,
            bytes,
            str,
        ],
    ]
]:
    return [
        (
            "dataset_files",
            (
                filename,
                (
                    DATA_DIR
                    /
                    filename
                ).read_bytes(),
                "text/csv",
            ),
        )

        for filename
        in DATASET_FILENAMES
    ]


# ============================================================
# PREPARATION SESSION
# ============================================================


def make_ready_session():
    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=
                DATASET_IDS
        )
    )


    # ========================================================
    # REQUIRED PREPARATION STAGES
    # ========================================================

    for (
        stage,
        evidence_ref,
    ) in [
        (
            PreparationStage.IMPORT,
            "lapage_csv_ingestion",
        ),
        (
            PreparationStage.UNDERSTAND,
            "lapage_dataset_profile",
        ),
        (
            PreparationStage.QUALITY,
            "data_quality_engine_v0.2",
        ),
    ]:
        record_required_stage_signal(
            workflow_id=
                session.workflow_id,

            stage=
                stage,

            completed=
                True,

            dataset_ids=
                DATASET_IDS,

            evidence_refs=[
                evidence_ref
            ],

            blocking_reasons=[],
        )


    # ========================================================
    # SERVER-OWNED SOURCE ARTIFACTS
    # ========================================================

    uploads = [
        UploadFile(
            file=(
                DATA_DIR
                /
                filename
            ).open(
                "rb"
            ),

            filename=
                filename,
        )

        for filename
        in DATASET_FILENAMES
    ]


    (
        _,
        source_records,
    ) = (
        load_uploaded_dataset_bundle(
            uploads
        )
    )


    actual_dataset_ids = [
        str(
            record[
                "dataset_id"
            ]
        )

        for record
        in source_records
    ]


    assert (
        actual_dataset_ids
        ==
        DATASET_IDS
    )


    for record in (
        source_records
    ):
        put_preparation_artifact(
            workflow_id=
                session.workflow_id,

            dataset_id=
                str(
                    record[
                        "dataset_id"
                    ]
                ),

            dataset_filename=
                str(
                    record[
                        "filename"
                    ]
                ),

            stage=
                "source",

            dataframe=
                record[
                    "dataframe"
                ],

            parent_dataset_ids=[],

            evidence_refs=[
                "test:lapage-production-ingestion"
            ],
        )


    # ========================================================
    # FINAL ANALYSIS OUTPUT SELECTION
    # ========================================================

    before_selection = (
        get_preparation_session(
            session.workflow_id
        )
    )


    selected = (
        record_analysis_output_selection(
            workflow_id=
                session.workflow_id,

            analysis_output_dataset_ids=
                DATASET_IDS,

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
                session.workflow_id,

            completed=
                True,

            passed=
                True,

            dataset_ids=
                DATASET_IDS,

            evidence_refs=[
                "entity_outlier_http_integration"
            ],

            blocking_reasons=[],

            expected_revision=
                selected.revision,
        )
    )


    assert (
        ready.snapshot.ready_for_analysis
        is True
    )


    return ready


# ============================================================
# ROUTING REGRESSION — GENERIC OUTLIERS
# ============================================================


def test_generic_outlier_objective_is_not_intercepted() -> None:
    """
    The new final-report integration must not steal the generic
    variable-outlier request from the existing planner route.

    The helper intentionally performs its entity-intent pre-check
    before touching the dataset catalog, so an empty record list
    is sufficient for this regression test.
    """

    finding = (
        build_entity_outlier_finding_if_requested(
            objective=
                "Détecte les outliers.",

            source_dataset_records=[],
        )
    )


    print(
        "\n=== GENERIC OUTLIER ROUTE ==="
    )

    print(
        "Entity finding:",
        finding,
    )


    assert (
        finding
        is None
    )


# ============================================================
# HTTP INTEGRATION — EXPLICIT CUSTOMER OUTLIERS
# ============================================================


def test_explicit_customer_outliers_are_attached_to_analysis_report() -> None:
    session = (
        make_ready_session()
    )


    response = (
        client.post(
            "/analysis/run",

            files=
                lapage_files(),

            data={
                "workflow_id":
                    session.workflow_id,

                "objective":
                    "Détecte les clients atypiques.",
            },
        )
    )


    print(
        "\n=== HTTP RESPONSE ==="
    )

    print(
        "Status:",
        response.status_code,
    )


    if (
        response.status_code
        !=
        200
    ):
        print(
            response.text[
                :4000
            ]
        )


    assert (
        response.status_code
        ==
        200
    )


    body = (
        response.json()
    )


    # --------------------------------------------------------
    # EXISTING UNIFIED REPORT CONTRACT IS PRESERVED
    # --------------------------------------------------------

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

    assert (
        "main_findings"
        in
        body
    )

    assert (
        "additional_findings"
        in
        body
    )

    assert (
        "methodology_notes"
        in
        body
    )


    # --------------------------------------------------------
    # NEW OPTIONAL ROUTED FINDING
    # --------------------------------------------------------

    assert (
        "entity_outlier_finding"
        in
        body
    )


    finding = (
        body[
            "entity_outlier_finding"
        ]
    )


    assert (
        finding
        is not None
    )


    print(
        "\n=== ENTITY OUTLIER FINDING ==="
    )

    print(
        "Keys:",
        sorted(
            finding.keys()
        ),
    )

    print(
        "Status:",
        finding.get(
            "status"
        ),
    )

    print(
        "Family:",
        finding.get(
            "family"
        ),
    )

    print(
        "Kind:",
        finding.get(
            "kind"
        ),
    )

    print(
        "Entity count:",
        finding.get(
            "entity_count"
        ),
    )

    print(
        "Raw IQR flags:",
        finding.get(
            "raw_flagged_entity_count"
        ),
    )

    print(
        "Priority:",
        finding.get(
            "priority_profile_count"
        ),
    )

    print(
        "Secondary:",
        finding.get(
            "behavioral_signal_count"
        ),
    )


    assert (
        finding.get(
            "status"
        )
        ==
        "ready"
    )

    assert (
        finding.get(
            "family"
        )
        ==
        "entity_outlier"
    )

    assert (
        finding.get(
            "kind"
        )
        ==
        "customer_entity_outlier_detection"
    )

    assert (
        finding.get(
            "entity_count"
        )
        ==
        8600
    )

    assert (
        finding.get(
            "raw_flagged_entity_count"
        )
        ==
        1422
    )

    assert (
        finding.get(
            "priority_profile_count"
        )
        ==
        4
    )

    assert (
        finding.get(
            "behavioral_signal_count"
        )
        ==
        1418
    )


    # --------------------------------------------------------
    # EXPECTED PRIORITY CLIENTS ARE PRESENT
    #
    # We deliberately inspect the serialized finding rather
    # than coupling this HTTP test to the internal nested
    # profile field names.
    # --------------------------------------------------------

    serialized_finding = (
        json.dumps(
            finding,
            ensure_ascii=False,
        )
    )


    print(
        "\n=== EXPECTED PRIORITY CLIENTS ==="
    )


    for client_id in sorted(
        EXPECTED_PRIORITY_CLIENTS
    ):
        found = (
            client_id
            in
            serialized_finding
        )


        print(
            f"{client_id:<10}:",
            (
                "FOUND"
                if found
                else "MISSING"
            ),
        )


        assert found


    # --------------------------------------------------------
    # SAFETY — INTERNAL SCORE MUST STAY PRIVATE
    # --------------------------------------------------------

    lower_serialized = (
        serialized_finding.lower()
    )


    assert (
        "anomaly_score"
        not in
        lower_serialized
    )


    caveat_text = " ".join(
        str(
            value
        )

        for value
        in finding.get(
            "caveats",
            [],
        )
    ).lower()


    print(
        "\n=== SAFETY ==="
    )

    print(
        "Internal anomaly score exposed :",
        (
            "YES"
            if (
                "anomaly_score"
                in
                lower_serialized
            )
            else "NO"
        ),
    )

    print(
        "Fraud caveat present            :",
        (
            "YES"
            if (
                "fraud"
                in
                caveat_text
            )
            else "NO"
        ),
    )

    print(
        "B2B caveat present              :",
        (
            "YES"
            if (
                "b2b"
                in
                caveat_text
            )
            else "NO"
        ),
    )

    print(
        "Deletion caveat present         :",
        (
            "YES"
            if any(
                token
                in
                caveat_text

                for token
                in [
                    "suppression",
                    "supprim",
                ]
            )
            else "NO"
        ),
    )


    assert (
        "fraud"
        in
        caveat_text
    )

    assert (
        "b2b"
        in
        caveat_text
    )

    assert any(
        token
        in
        caveat_text

        for token
        in [
            "suppression",
            "supprim",
        ]
    )


# ============================================================
# OPENAPI CONTRACT
# ============================================================


def test_analysis_response_schema_exposes_optional_entity_finding() -> None:
    schema = (
        app.openapi()
    )


    operation = (
        schema[
            "paths"
        ][
            "/analysis/run"
        ][
            "post"
        ]
    )


    response_schema = (
        operation[
            "responses"
        ][
            "200"
        ][
            "content"
        ][
            "application/json"
        ][
            "schema"
        ]
    )


    print(
        "\n=== OPENAPI ==="
    )

    print(
        "200 response schema:",
        response_schema,
    )


    assert (
        "$ref"
        in
        response_schema
    )

    assert (
        response_schema[
            "$ref"
        ].endswith(
            "/RoutedUnifiedAnalysisReport"
        )
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    assert_source_files_exist()


    reset_preparation_session_store_for_tests()

    reset_preparation_artifact_store_for_tests()


    print()
    print(
        "=" * 66
    )
    print(
        "DataLens Analysis Entity Outlier HTTP · Lapage v0.1"
    )
    print(
        "=" * 66
    )


    test_generic_outlier_objective_is_not_intercepted()


    reset_preparation_session_store_for_tests()

    reset_preparation_artifact_store_for_tests()


    test_explicit_customer_outliers_are_attached_to_analysis_report()


    test_analysis_response_schema_exposes_optional_entity_finding()


    print()
    print(
        "=" * 66
    )
    print(
        "PASS - analysis entity outlier HTTP Lapage v0.1"
    )
    print(
        "=" * 66
    )


if (
    __name__
    ==
    "__main__"
):
    main()
