from __future__ import annotations


import pandas as pd


from fastapi.testclient import (
    TestClient,
)


from main import (
    app,
)


from app.preparation.preparation_artifact_store import (
    put_preparation_artifact,
    reset_preparation_artifact_store_for_tests,
)


from app.preparation.preparation_session import (
    create_preparation_session,
    get_preparation_session,
    record_optional_stage_signal,
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
# DATA
# ============================================================


ROOT_SALES_ID = (
    "sales"
)


ROOT_CUSTOMERS_ID = (
    "customers"
)


COMBINED_ID = (
    "sales_customers"
)


def sales_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "sale_id": [
                    1,
                    2,
                    3,
                ],

                "customer_id": [
                    "c1",
                    "c1",
                    "c2",
                ],

                "amount": [
                    10.0,
                    20.0,
                    30.0,
                ],
            }
        )
    )


def customers_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "customer_id": [
                    "c1",
                    "c2",
                ],

                "segment": [
                    "A",
                    "B",
                ],
            }
        )
    )


def combined_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "sale_id": [
                    1,
                    2,
                    3,
                ],

                "customer_id": [
                    "c1",
                    "c1",
                    "c2",
                ],

                "amount": [
                    10.0,
                    20.0,
                    30.0,
                ],

                "segment": [
                    "A",
                    "A",
                    "B",
                ],
            }
        )
    )


# ============================================================
# RESET
# ============================================================


def reset_state() -> None:
    reset_preparation_session_store_for_tests()

    reset_preparation_artifact_store_for_tests()


# ============================================================
# SESSION FIXTURE
# ============================================================


def build_combined_preparation():
    reset_state()


    root_ids = [
        ROOT_SALES_ID,
        ROOT_CUSTOMERS_ID,
    ]


    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=
                root_ids
        )
    )


    # ========================================================
    # REQUIRED STAGES
    # ========================================================

    for stage in [
        PreparationStage.IMPORT,
        PreparationStage.UNDERSTAND,
        PreparationStage.QUALITY,
    ]:
        record_required_stage_signal(
            workflow_id=
                session.workflow_id,

            stage=
                stage,

            completed=
                True,

            dataset_ids=
                root_ids,

            evidence_refs=[
                (
                    "test:"
                    f"{stage.value}"
                )
            ],

            blocking_reasons=[],
        )


    # ========================================================
    # SOURCE ARTIFACTS
    # ========================================================

    put_preparation_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            ROOT_SALES_ID,

        dataset_filename=
            "sales.csv",

        stage=
            "source",

        dataframe=
            sales_frame(),

        parent_dataset_ids=[],

        evidence_refs=[
            "test:source:sales",
        ],
    )


    put_preparation_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            ROOT_CUSTOMERS_ID,

        dataset_filename=
            "customers.csv",

        stage=
            "source",

        dataframe=
            customers_frame(),

        parent_dataset_ids=[],

        evidence_refs=[
            "test:source:customers",
        ],
    )


    # ========================================================
    # CLEAN — NOT REQUIRED
    # ========================================================

    record_optional_stage_signal(
        workflow_id=
            session.workflow_id,

        stage=
            PreparationStage.CLEAN,

        required=
            False,

        completed=
            False,

        review_required=
            False,

        blocked=
            False,

        dataset_ids=
            root_ids,

        evidence_refs=[
            "cleaning:not-required",
        ],

        blocking_reasons=[],
    )


    # ========================================================
    # TRANSFORM — NOT REQUIRED
    # ========================================================

    record_optional_stage_signal(
        workflow_id=
            session.workflow_id,

        stage=
            PreparationStage.TRANSFORM,

        required=
            False,

        completed=
            False,

        review_required=
            False,

        blocked=
            False,

        dataset_ids=
            root_ids,

        evidence_refs=[
            "transform:not-required",
        ],

        blocking_reasons=[],
    )


    # ========================================================
    # COMBINE ARTIFACT
    # ========================================================

    put_preparation_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            COMBINED_ID,

        dataset_filename=
            "sales_customers.csv",

        stage=
            "combine",

        dataframe=
            combined_frame(),

        parent_dataset_ids=[
            ROOT_SALES_ID,
            ROOT_CUSTOMERS_ID,
        ],

        evidence_refs=[
            "join:test",
            "post_join_validation:test",
        ],
    )


    # ========================================================
    # COMBINE PASSED
    # ========================================================

    record_optional_stage_signal(
        workflow_id=
            session.workflow_id,

        stage=
            PreparationStage.COMBINE,

        required=
            True,

        completed=
            True,

        review_required=
            False,

        blocked=
            False,

        dataset_ids=[
            COMBINED_ID,
        ],

        evidence_refs=[
            "join_plan:test",
            "join_execution:test",
            "post_join_validation:test",
        ],

        blocking_reasons=[],
    )


    return (
        get_preparation_session(
            session.workflow_id
        )
    )


# ============================================================
# 1. CAPABILITIES
# ============================================================


def test_capabilities_expose_controlled_output_selection(
) -> None:
    response = (
        client.get(
            "/preparation/sessions/capabilities"
        )
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
        body[
            "client_can_select_analysis_output"
        ]
        is True
    )


    assert (
        body[
            "client_can_set_ready_for_analysis"
        ]
        is False
    )


    assert (
        body[
            "client_can_update_stage_status"
        ]
        is False
    )


    assert (
        body[
            "api_version"
        ]
        ==
        "preparation_session_api_v0.2"
    )


    print(
        "Controlled output-selection capability: PASS"
    )


# ============================================================
# 2. CANDIDATE METADATA
# ============================================================


def test_analysis_output_candidates_are_exposed_without_data(
) -> None:
    session = (
        build_combined_preparation()
    )


    response = (
        client.get(
            (
                "/preparation/sessions/"
                f"{session.workflow_id}"
                "/analysis-output-candidates"
            )
        )
    )


    print(
        "Candidate GET status:",
        response.status_code,
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
        body[
            "workflow_id"
        ]
        ==
        session.workflow_id
    )


    assert (
        body[
            "candidate_count"
        ]
        ==
        3
    )


    assert (
        body[
            "analysis_output_dataset_ids"
        ]
        ==
        []
    )


    assert (
        body[
            "locked"
        ]
        is False
    )


    candidates = (
        body[
            "candidates"
        ]
    )


    # Derived final-stage artifact is presented first.
    assert (
        candidates[
            0
        ][
            "dataset_id"
        ]
        ==
        COMBINED_ID
    )


    assert (
        candidates[
            0
        ][
            "stage"
        ]
        ==
        "combine"
    )


    assert (
        candidates[
            0
        ][
            "rows"
        ]
        ==
        3
    )


    assert (
        candidates[
            0
        ][
            "columns"
        ]
        ==
        4
    )


    assert (
        candidates[
            0
        ][
            "parent_dataset_ids"
        ]
        ==
        [
            ROOT_SALES_ID,
            ROOT_CUSTOMERS_ID,
        ]
    )


    # No raw DataFrame may cross this read API.
    for candidate in (
        candidates
    ):
        assert (
            "dataframe"
            not in
            candidate
        )


    print(
        "Safe Artifact Store candidate metadata: PASS"
    )


# ============================================================
# 3. DERIVED OUTPUT COMMIT
# ============================================================


def test_derived_output_can_be_selected_via_api(
) -> None:
    session = (
        build_combined_preparation()
    )


    before_revision = (
        session.revision
    )


    response = (
        client.post(
            "/preparation/analysis-output",

            json={
                "workflow_id":
                    session.workflow_id,

                "dataset_ids": [
                    COMBINED_ID,
                ],
            },
        )
    )


    print(
        "Output selection POST status:",
        response.status_code,
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
        body[
            "analysis_output_dataset_ids"
        ]
        ==
        [
            COMBINED_ID,
        ]
    )


    assert (
        body[
            "snapshot"
        ][
            "analysis_output_dataset_ids"
        ]
        ==
        [
            COMBINED_ID,
        ]
    )


    assert (
        body[
            "selected_analysis_dataset_ids"
        ]
        ==
        [
            ROOT_SALES_ID,
            ROOT_CUSTOMERS_ID,
        ]
    )


    assert (
        body[
            "revision"
        ]
        ==
        before_revision
        +
        1
    )


    assert (
        body[
            "snapshot"
        ][
            "ready_for_analysis"
        ]
        is False
    )


    assert (
        body[
            "snapshot"
        ][
            "next_stage"
        ]
        ==
        "validate"
    )


    print(
        "Derived final output committed through server guard: PASS"
    )


# ============================================================
# 4. CANDIDATE REFLECTS SELECTION
# ============================================================


def test_candidate_read_model_reflects_selection(
) -> None:
    session = (
        build_combined_preparation()
    )


    selection_response = (
        client.post(
            "/preparation/analysis-output",

            json={
                "workflow_id":
                    session.workflow_id,

                "dataset_ids": [
                    COMBINED_ID,
                ],
            },
        )
    )


    assert (
        selection_response.status_code
        ==
        200
    )


    response = (
        client.get(
            (
                "/preparation/sessions/"
                f"{session.workflow_id}"
                "/analysis-output-candidates"
            )
        )
    )


    assert (
        response.status_code
        ==
        200
    )


    body = (
        response.json()
    )


    selected = [
        candidate

        for candidate
        in body[
            "candidates"
        ]

        if (
            candidate[
                "is_selected"
            ]
        )
    ]


    assert (
        len(
            selected
        )
        ==
        1
    )


    assert (
        selected[
            0
        ][
            "dataset_id"
        ]
        ==
        COMBINED_ID
    )


    print(
        "Candidate selection state reflected: PASS"
    )


# ============================================================
# 5. UNKNOWN ARTIFACT FAILS CLOSED
# ============================================================


def test_unknown_artifact_is_rejected(
) -> None:
    session = (
        build_combined_preparation()
    )


    response = (
        client.post(
            "/preparation/analysis-output",

            json={
                "workflow_id":
                    session.workflow_id,

                "dataset_ids": [
                    "invented_dataset",
                ],
            },
        )
    )


    print(
        "Unknown artifact status:",
        response.status_code,
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
        "analysis_output_selection_rejected"
    )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    assert (
        current
        .analysis_output_dataset_ids
        ==
        []
    )


    print(
        "Unknown output artifact rejected without commit: PASS"
    )


# ============================================================
# 6. CLIENT CANNOT INJECT READINESS
# ============================================================


def test_client_cannot_inject_preparation_state(
) -> None:
    session = (
        build_combined_preparation()
    )


    response = (
        client.post(
            "/preparation/analysis-output",

            json={
                "workflow_id":
                    session.workflow_id,

                "dataset_ids": [
                    COMBINED_ID,
                ],

                "ready_for_analysis":
                    True,

                "status":
                    "passed",
            },
        )
    )


    assert (
        response.status_code
        ==
        422
    )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    assert (
        current
        .analysis_output_dataset_ids
        ==
        []
    )


    print(
        "Client cannot inject readiness or stage status: PASS"
    )


# ============================================================
# 7. EMPTY SELECTION REJECTED
# ============================================================


def test_empty_output_selection_is_rejected(
) -> None:
    session = (
        build_combined_preparation()
    )


    response = (
        client.post(
            "/preparation/analysis-output",

            json={
                "workflow_id":
                    session.workflow_id,

                "dataset_ids": [],
            },
        )
    )


    assert (
        response.status_code
        ==
        422
    )


    print(
        "Empty output selection rejected by API contract: PASS"
    )


# ============================================================
# 8. UNKNOWN SESSION
# ============================================================


def test_unknown_session_is_rejected(
) -> None:
    response = (
        client.get(
            (
                "/preparation/sessions/"
                "prep:does-not-exist"
                "/analysis-output-candidates"
            )
        )
    )


    assert (
        response.status_code
        ==
        404
    )


    response = (
        client.post(
            "/preparation/analysis-output",

            json={
                "workflow_id":
                    "prep:does-not-exist",

                "dataset_ids": [
                    COMBINED_ID,
                ],
            },
        )
    )


    assert (
        response.status_code
        ==
        404
    )


    print(
        "Unknown Preparation session rejected: PASS"
    )


# ============================================================
# 9. VALIDATE LOCKS SELECTION
# ============================================================


def test_passed_validation_locks_output_selection(
) -> None:
    session = (
        build_combined_preparation()
    )


    selection_response = (
        client.post(
            "/preparation/analysis-output",

            json={
                "workflow_id":
                    session.workflow_id,

                "dataset_ids": [
                    COMBINED_ID,
                ],
            },
        )
    )


    assert (
        selection_response.status_code
        ==
        200
    )


    selected = (
        get_preparation_session(
            session.workflow_id
        )
    )


    ready = (
        record_validation_stage_signal(
            workflow_id=
                session.workflow_id,

            completed=
                True,

            passed=
                True,

            dataset_ids=[
                COMBINED_ID,
            ],

            evidence_refs=[
                "final_validation:test",
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


    # Attempt to replace the final output after VALIDATE.
    response = (
        client.post(
            "/preparation/analysis-output",

            json={
                "workflow_id":
                    session.workflow_id,

                "dataset_ids": [
                    ROOT_SALES_ID,
                ],
            },
        )
    )


    print(
        "Post-VALIDATE output change status:",
        response.status_code,
    )


    assert (
        response.status_code
        ==
        409
    )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    assert (
        current
        .analysis_output_dataset_ids
        ==
        [
            COMBINED_ID,
        ]
    )


    candidate_response = (
        client.get(
            (
                "/preparation/sessions/"
                f"{session.workflow_id}"
                "/analysis-output-candidates"
            )
        )
    )


    assert (
        candidate_response.status_code
        ==
        200
    )


    assert (
        candidate_response
        .json()[
            "locked"
        ]
        is True
    )


    print(
        "PASSED VALIDATE locks final output selection: PASS"
    )


# ============================================================
# 10. ROUTES
# ============================================================


def test_routes_are_registered(
) -> None:
    paths = (
        app.openapi()[
            "paths"
        ]
    )


    assert (
        "/preparation/analysis-output"
        in
        paths
    )


    assert (
        (
            "/preparation/sessions/"
            "{workflow_id}/analysis-output-candidates"
        )
        in
        paths
    )


    print(
        "Analysis output API routes registered: PASS"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        (
            "=== DATALENS PREPARATION "
            "ANALYSIS OUTPUT API v0.1 ==="
        )
    )

    print()


    test_capabilities_expose_controlled_output_selection()

    test_analysis_output_candidates_are_exposed_without_data()

    test_derived_output_can_be_selected_via_api()

    test_candidate_read_model_reflects_selection()

    test_unknown_artifact_is_rejected()

    test_client_cannot_inject_preparation_state()

    test_empty_output_selection_is_rejected()

    test_unknown_session_is_rejected()

    test_passed_validation_locks_output_selection()

    test_routes_are_registered()


    print()

    print(
        "Preparation Analysis Output API v0.1: PASS"
    )


if __name__ == "__main__":
    main()