from __future__ import annotations


import pandas as pd


from app.preparation.analysis_output_selection_commit import (
    commit_analysis_output_selection,
)

from app.preparation.final_validation_v0_2 import (
    FinalPreparationValidationV02BlockedError,
    evaluate_final_preparation_validation_v0_2,
    require_final_preparation_validation_v0_2,
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
    reset_preparation_session_store_for_tests,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
)


# ============================================================
# TEST DATA
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


def orders_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "order_id": [
                    1,
                    2,
                    3,
                ],

                "amount": [
                    15.0,
                    25.0,
                    35.0,
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
                ),
            ],

            blocking_reasons=[],
        )


# ============================================================
# CHECK LOOKUP
# ============================================================


def get_check(
    report,
    code: str,
):
    matches = [
        check

        for check
        in report.checks

        if (
            check.code
            ==
            code
        )
    ]


    assert (
        len(
            matches
        )
        ==
        1
    ), (
        "Expected exactly one validation check "
        f"with code={code!r}, "
        f"found={len(matches)}."
    )


    return (
        matches[
            0
        ]
    )


# ============================================================
# JOINED PREPARATION BUILDER
# ============================================================


def build_joined_preparation(
    *,
    selected_output_dataset_ids: list[
        str
    ],
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


    workflow_id = (
        session.workflow_id
    )


    # ========================================================
    # REQUIRED STAGES
    # ========================================================


    pass_required_stages(
        workflow_id=
            workflow_id,

        dataset_ids=
            root_dataset_ids,
    )


    # ========================================================
    # CLEAN ARTIFACT — SALES
    #
    # Self-parent lineage is intentional.
    # ========================================================


    put_preparation_artifact(
        workflow_id=
            workflow_id,

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


    # ========================================================
    # CLEAN ARTIFACT — CUSTOMERS
    # ========================================================


    put_preparation_artifact(
        workflow_id=
            workflow_id,

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
            workflow_id,

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
            workflow_id,

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
    #
    # This materialized descendant supersedes sales and
    # customers as terminal analytical outputs.
    # ========================================================


    put_preparation_artifact(
        workflow_id=
            workflow_id,

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
            workflow_id,

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
    # COMMIT REQUESTED ANALYTICAL OUTPUT
    # ========================================================


    commit_analysis_output_selection(
        workflow_id=
            workflow_id,

        requested_dataset_ids=
            selected_output_dataset_ids,
    )


    return (
        get_preparation_session(
            workflow_id
        )
    )


# ============================================================
# SELF-PARENT PREPARATION BUILDER
# ============================================================


def build_self_parent_clean_preparation():
    reset_state()


    root_dataset_ids = [
        "orders",
    ]


    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=
                root_dataset_ids
        )
    )


    workflow_id = (
        session.workflow_id
    )


    # ========================================================
    # REQUIRED STAGES
    # ========================================================


    pass_required_stages(
        workflow_id=
            workflow_id,

        dataset_ids=
            root_dataset_ids,
    )


    # ========================================================
    # CLEAN ARTIFACT
    #
    # The artifact is materialized in place:
    #
    #     dataset_id = orders
    #     parent_dataset_ids = [orders]
    #
    # This must NOT make orders "superseded".
    # ========================================================


    put_preparation_artifact(
        workflow_id=
            workflow_id,

        dataset_id=
            "orders",

        dataset_filename=
            "orders.csv",

        stage=
            "clean",

        dataframe=
            orders_frame(),

        parent_dataset_ids=[
            "orders",
        ],

        evidence_refs=[
            "cleaning:orders",
        ],
    )


    # ========================================================
    # CLEAN PASSED
    # ========================================================


    record_optional_stage_signal(
        workflow_id=
            workflow_id,

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

        dataset_ids=[
            "orders",
        ],

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
            workflow_id,

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

        dataset_ids=[
            "orders",
        ],

        evidence_refs=[
            "transformation_plan:test",
        ],

        blocking_reasons=[],
    )


    # ========================================================
    # COMBINE SKIPPED
    # ========================================================


    record_optional_stage_signal(
        workflow_id=
            workflow_id,

        stage=
            PreparationStage.COMBINE,

        required=
            False,

        completed=
            False,

        review_required=
            False,

        blocked=
            False,

        dataset_ids=[
            "orders",
        ],

        evidence_refs=[
            "combine_plan:test",
        ],

        blocking_reasons=[],
    )


    # ========================================================
    # OUTPUT SELECTION
    # ========================================================


    commit_analysis_output_selection(
        workflow_id=
            workflow_id,

        requested_dataset_ids=[
            "orders",
        ],
    )


    return (
        get_preparation_session(
            workflow_id
        )
    )


# ============================================================
# TEST 1
# TERMINAL COMBINE OUTPUT IS ACCEPTED
# ============================================================


def test_terminal_combine_output_passes() -> None:
    session = (
        build_joined_preparation(
            selected_output_dataset_ids=[
                "sales_customers",
            ]
        )
    )


    report = (
        evaluate_final_preparation_validation_v0_2(
            session
        )
    )


    terminal_check = (
        get_check(
            report,
            "analysis_outputs_terminal",
        )
    )


    assert (
        terminal_check.passed
        is
        True
    )


    assert (
        report.passed
        is
        True
    )


    required_report = (
        require_final_preparation_validation_v0_2(
            session
        )
    )


    assert (
        required_report.passed
        is
        True
    )


    assert (
        required_report
        .analysis_output_dataset_ids
        ==
        [
            "sales_customers",
        ]
    )


# ============================================================
# TEST 2
# SUPERSEDED ROOT OUTPUTS ARE BLOCKED
# ============================================================


def test_superseded_outputs_are_blocked() -> None:
    session = (
        build_joined_preparation(
            selected_output_dataset_ids=[
                "sales",
                "customers",
            ]
        )
    )


    # The selection layer currently allows these artifacts
    # because they still exist and their lineage is valid.
    #
    # Final Validation must add the stricter question:
    #
    #     Are they still terminal?
    #
    # They are not: sales_customers now consumes both.
    assert (
        session.analysis_output_dataset_ids
        ==
        [
            "sales",
            "customers",
        ]
    )


    report = (
        evaluate_final_preparation_validation_v0_2(
            session
        )
    )


    terminal_check = (
        get_check(
            report,
            "analysis_outputs_terminal",
        )
    )


    assert (
        terminal_check.passed
        is
        False
    )


    assert (
        report.passed
        is
        False
    )


    assert (
        "sales"
        in
        terminal_check.message
    )


    assert (
        "customers"
        in
        terminal_check.message
    )


    try:
        require_final_preparation_validation_v0_2(
            session
        )

    except FinalPreparationValidationV02BlockedError as error:
        blocked_report = (
            error.report
        )


        blocked_terminal_check = (
            get_check(
                blocked_report,
                "analysis_outputs_terminal",
            )
        )


        assert (
            blocked_terminal_check.passed
            is
            False
        )


        assert (
            blocked_report.passed
            is
            False
        )

    else:
        raise AssertionError(
            (
                "Superseded analytical outputs were "
                "incorrectly accepted by Final Validation."
            )
        )


# ============================================================
# TEST 3
# SELF-PARENT CLEAN ARTIFACT REMAINS TERMINAL
# ============================================================


def test_self_parent_clean_output_is_not_superseded() -> None:
    session = (
        build_self_parent_clean_preparation()
    )


    assert (
        session.analysis_output_dataset_ids
        ==
        [
            "orders",
        ]
    )


    report = (
        evaluate_final_preparation_validation_v0_2(
            session
        )
    )


    terminal_check = (
        get_check(
            report,
            "analysis_outputs_terminal",
        )
    )


    assert (
        terminal_check.passed
        is
        True
    )


    assert (
        report.passed
        is
        True
    )


    required_report = (
        require_final_preparation_validation_v0_2(
            session
        )
    )


    assert (
        required_report.passed
        is
        True
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        "=== DATALENS TERMINAL OUTPUT GUARD v0.2 ==="
    )

    print()


    test_terminal_combine_output_passes()

    print(
        "Terminal COMBINE output passes: PASS"
    )


    test_superseded_outputs_are_blocked()

    print(
        "Superseded outputs are blocked: PASS"
    )


    test_self_parent_clean_output_is_not_superseded()

    print(
        "Self-parent CLEAN output remains terminal: PASS"
    )


    print()

    print(
        "Terminal Output Guard v0.2: PASS"
    )


if __name__ == "__main__":
    main()