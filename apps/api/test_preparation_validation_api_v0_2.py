from __future__ import annotations

import pandas as pd

from fastapi.testclient import (
    TestClient,
)

from main import (
    app,
)

from app.preparation.analysis_output_selection_commit import (
    commit_analysis_output_selection,
)

from app.preparation.preparation_artifact_store import (
    put_preparation_artifact,
    reset_preparation_artifact_store_for_tests,
)

from app.preparation.preparation_session import (
    PreparationSessionRevisionConflictError,
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


def joined_frame() -> pd.DataFrame:
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
# REQUIRED STAGES
# ============================================================


def pass_required_stages(
    *,
    workflow_id: str,
    dataset_ids: list[
        str
    ],
) -> None:
    for stage in [
        PreparationStage.IMPORT,
        PreparationStage.UNDERSTAND,
        PreparationStage.QUALITY,
    ]:
        record_required_stage_signal(
            workflow_id=
                workflow_id,

            stage=
                stage,

            completed=
                True,

            dataset_ids=
                dataset_ids,

            evidence_refs=[
                (
                    "test:"
                    f"{stage.value}"
                )
            ],

            blocking_reasons=[],
        )


# ============================================================
# JOINED PREPARATION
# ============================================================


def build_joined_preparation(
    *,
    commit_output: bool,
):
    reset_state()


    root_dataset_ids = [
        "sales",
        "customers",
    ]


    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=
                root_dataset_ids
        )
    )


    # ========================================================
    # REQUIRED
    # ========================================================

    pass_required_stages(
        workflow_id=
            session.workflow_id,

        dataset_ids=
            root_dataset_ids,
    )


    # ========================================================
    # CLEAN ARTIFACTS
    # ========================================================

    put_preparation_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            "sales",

        dataset_filename=
            "sales.csv",

        stage=
            "clean",

        dataframe=
            sales_frame(),

        parent_dataset_ids=[
            "sales",
        ],

        evidence_refs=[
            "cleaning:sales",
        ],
    )


    put_preparation_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            "customers",

        dataset_filename=
            "customers.csv",

        stage=
            "clean",

        dataframe=
            customers_frame(),

        parent_dataset_ids=[
            "customers",
        ],

        evidence_refs=[
            "cleaning:customers",
        ],
    )


    # ========================================================
    # CLEAN PASSED
    # ========================================================

    record_optional_stage_signal(
        workflow_id=
            session.workflow_id,

        stage=
            PreparationStage.CLEAN,

        required=
            True,

        completed=
            True,

        review_required=
            False,

        blocked=
            False,

        dataset_ids=
            root_dataset_ids,

        evidence_refs=[
            "cleaning_plan:test",
            "cleaning_execution:test",
        ],

        blocking_reasons=[],
    )


    # ========================================================
    # TRANSFORM SKIPPED
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
            root_dataset_ids,

        evidence_refs=[
            "transformation_plan:test",
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
            "sales_customers",

        dataset_filename=
            "sales_customers.csv",

        stage=
            "combine",

        dataframe=
            joined_frame(),

        parent_dataset_ids=[
            "sales",
            "customers",
        ],

        evidence_refs=[
            "join:validated",
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
            "sales_customers",
        ],

        evidence_refs=[
            "join_plan:test",
            "join_execution:test",
            "post_join_validation:test",
        ],

        blocking_reasons=[],
    )


    # ========================================================
    # OPTIONAL FINAL OUTPUT COMMIT
    # ========================================================

    if (
        commit_output
    ):
        commit_analysis_output_selection(
            workflow_id=
                session.workflow_id,

            requested_dataset_ids=[
                "sales_customers",
            ],
        )


    return (
        get_preparation_session(
            session.workflow_id
        )
    )


# ============================================================
# 1. DERIVED OUTPUT IS WRITTEN TO VALIDATE
# ============================================================


def test_derived_output_is_validated() -> None:
    session = (
        build_joined_preparation(
            commit_output=
                True
        )
    )


    response = (
        client.post(
            "/preparation/validate",

            json={
                "workflow_id":
                    session.workflow_id
            },
        )
    )


    print(
        "Derived output VALIDATE status:",
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


    # ========================================================
    # SESSION FINAL OUTPUT SCOPE
    # ========================================================

    assert (
        body[
            "analysis_output_dataset_ids"
        ]
        ==
        [
            "sales_customers",
        ]
    )


    # ========================================================
    # VALIDATE STAGE
    # ========================================================

    validate_stage = next(
        stage

        for stage
        in body[
            "snapshot"
        ][
            "stages"
        ]

        if (
            stage[
                "stage"
            ]
            ==
            "validate"
        )
    )


    assert (
        validate_stage[
            "status"
        ]
        ==
        "passed"
    )


    assert (
        validate_stage[
            "dataset_ids"
        ]
        ==
        [
            "sales_customers",
        ]
    )


    # ========================================================
    # SNAPSHOT FINAL OUTPUT CONTRACT
    # ========================================================

    assert (
        body[
            "snapshot"
        ][
            "analysis_output_dataset_ids"
        ]
        ==
        [
            "sales_customers",
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
            "sales_customers",
        ]
    )


    # ========================================================
    # READY FOR ANALYSIS
    #
    # Preparation roots and final analytical outputs are now
    # distinct scopes.
    #
    # sales + customers are the immutable Preparation roots.
    #
    # sales_customers is the materialized COMBINE output
    # explicitly selected for analytical execution.
    #
    # Because that exact final output has been certified by
    # VALIDATE, the workflow may cross the analysis boundary.
    # ========================================================

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
        (
            "Derived COMBINE output validated and "
            "READY FOR ANALYSIS: PASS"
        )
    )


# ============================================================
# 2. NO FINAL OUTPUT -> BLOCKED
# ============================================================


def test_missing_output_selection_is_blocked() -> None:
    session = (
        build_joined_preparation(
            commit_output=
                False
        )
    )


    response = (
        client.post(
            "/preparation/validate",

            json={
                "workflow_id":
                    session.workflow_id
            },
        )
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
        "final_preparation_validation_failed"
    )


    validation = (
        detail[
            "validation"
        ]
    )


    assert any(
        (
            check[
                "code"
            ]
            ==
            "analysis_outputs_selected"
            and
            check[
                "passed"
            ]
            is False
        )

        for check
        in validation[
            "checks"
        ]
    )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    validate_stage = next(
        stage

        for stage
        in current
        .snapshot
        .stages

        if (
            stage.stage
            ==
            PreparationStage.VALIDATE
        )
    )


    assert (
        validate_stage
        .status
        .value
        ==
        "blocked"
    )


    assert (
        current
        .snapshot
        .ready_for_analysis
        is False
    )


    print(
        "Missing output selection blocks VALIDATE: PASS"
    )


# ============================================================
# 3. CLIENT CANNOT INJECT VALIDATION STATE
# ============================================================


def test_client_cannot_inject_validation_state() -> None:
    session = (
        build_joined_preparation(
            commit_output=
                True
        )
    )


    response = (
        client.post(
            "/preparation/validate",

            json={
                "workflow_id":
                    session.workflow_id,

                "passed":
                    True,

                "dataset_ids": [
                    "invented_dataset",
                ],

                "ready_for_analysis":
                    True,
            },
        )
    )


    assert (
        response.status_code
        ==
        422
    )


    print(
        "Client cannot inject VALIDATE state: PASS"
    )


# ============================================================
# 4. ATOMIC REVISION GUARD
# ============================================================


def test_stale_validation_commit_is_rejected() -> None:
    reset_state()


    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                "dataset:orders",
            ]
        )
    )


    stale_revision = (
        session.revision
    )


    # Another legitimate backend operation changes the session
    # after validation would hypothetically have been
    # evaluated.
    newer = (
        record_required_stage_signal(
            workflow_id=
                session.workflow_id,

            stage=
                PreparationStage.IMPORT,

            completed=
                True,

            dataset_ids=[
                "dataset:orders",
            ],

            evidence_refs=[
                "test:import",
            ],

            blocking_reasons=[],
        )
    )


    assert (
        newer.revision
        ==
        stale_revision
        +
        1
    )


    try:
        record_validation_stage_signal(
            workflow_id=
                session.workflow_id,

            completed=
                True,

            passed=
                True,

            dataset_ids=[
                "dataset:orders",
            ],

            evidence_refs=[
                "final_validation:synthetic",
            ],

            blocking_reasons=[],

            expected_revision=
                stale_revision,
        )


    except PreparationSessionRevisionConflictError:
        pass


    else:
        raise AssertionError(
            (
                "Stale VALIDATE decision must never "
                "be committed."
            )
        )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    validate_stage = next(
        stage

        for stage
        in current
        .snapshot
        .stages

        if (
            stage.stage
            ==
            PreparationStage.VALIDATE
        )
    )


    assert (
        validate_stage
        .status
        .value
        ==
        "not_started"
    )


    assert (
        current
        .snapshot
        .ready_for_analysis
        is False
    )


    print(
        "Stale VALIDATE decision rejected atomically: PASS"
    )


# ============================================================
# 5. UNKNOWN SESSION
# ============================================================


def test_unknown_session_returns_404() -> None:
    response = (
        client.post(
            "/preparation/validate",

            json={
                "workflow_id":
                    "prep:does-not-exist"
            },
        )
    )


    assert (
        response.status_code
        ==
        404
    )


    assert (
        response.json()[
            "detail"
        ][
            "error"
        ]
        ==
        "preparation_session_not_found"
    )


    print(
        "Unknown Preparation session rejected: PASS"
    )


# ============================================================
# 6. ROUTE PRESERVED
# ============================================================


def test_route_preserved() -> None:
    paths = (
        app.openapi()[
            "paths"
        ]
    )


    assert (
        "/preparation/validate"
        in
        paths
    )


    print(
        "Preparation validation route preserved: PASS"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        (
            "=== DATALENS PREPARATION "
            "VALIDATION API v0.2 ==="
        )
    )

    print()


    test_derived_output_is_validated()

    test_missing_output_selection_is_blocked()

    test_client_cannot_inject_validation_state()

    test_stale_validation_commit_is_rejected()

    test_unknown_session_returns_404()

    test_route_preserved()


    print()

    print(
        "Preparation Validation API v0.2: PASS"
    )


if __name__ == "__main__":
    main()