from __future__ import annotations

import pandas as pd

from app.preparation.analysis_output_selection_commit import (
    commit_analysis_output_selection,
)

from app.preparation.final_validation_v0_2 import (
    FINAL_PREPARATION_VALIDATION_V0_2_RULE_VERSION,
    FinalPreparationValidationV02BlockedError,
    evaluate_final_preparation_validation_v0_2,
    require_final_preparation_validation_v0_2,
)

from app.preparation.preparation_artifact_store import (
    delete_preparation_artifacts,
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
# DATA
# ============================================================


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
                    10.0,
                    20.0,
                    30.0,
                ],
            }
        )
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
# REQUIRED PREPARATION
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
# OPTIONAL STAGES
# ============================================================


def skip_clean(
    *,
    workflow_id: str,
    dataset_ids: list[
        str
    ],
):
    return (
        record_optional_stage_signal(
            workflow_id=
                workflow_id,

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
                dataset_ids,

            evidence_refs=[
                "cleaning_plan:test",
            ],

            blocking_reasons=[],
        )
    )


def pass_clean(
    *,
    workflow_id: str,
    dataset_ids: list[
        str
    ],
):
    return (
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
                dataset_ids,

            evidence_refs=[
                "cleaning_plan:test",
                "cleaning_execution:test",
            ],

            blocking_reasons=[],
        )
    )


def review_clean(
    *,
    workflow_id: str,
    dataset_ids: list[
        str
    ],
):
    return (
        record_optional_stage_signal(
            workflow_id=
                workflow_id,

            stage=
                PreparationStage.CLEAN,

            required=
                True,

            completed=
                False,

            review_required=
                True,

            blocked=
                False,

            dataset_ids=
                dataset_ids,

            evidence_refs=[
                "cleaning_plan:test",
            ],

            blocking_reasons=[
                "Analyst review remains.",
            ],
        )
    )


def skip_transform(
    *,
    workflow_id: str,
    dataset_ids: list[
        str
    ],
):
    return (
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
                dataset_ids,

            evidence_refs=[
                "transformation_plan:test",
            ],

            blocking_reasons=[],
        )
    )


def skip_combine(
    *,
    workflow_id: str,
    dataset_ids: list[
        str
    ],
):
    return (
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

            dataset_ids=
                dataset_ids,

            evidence_refs=[
                "join_plan:test",
            ],

            blocking_reasons=[],
        )
    )


# ============================================================
# SINGLE ROOT READY FOR OUTPUT SELECTION
# ============================================================


def build_single_root_session(
    *,
    clean_mode: str = "skip",
):
    reset_state()


    root_dataset_id = (
        "dataset:orders"
    )


    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                root_dataset_id,
            ]
        )
    )


    pass_required_stages(
        workflow_id=
            session.workflow_id,

        dataset_ids=[
            root_dataset_id,
        ],
    )


    if (
        clean_mode
        ==
        "skip"
    ):
        put_preparation_artifact(
            workflow_id=
                session.workflow_id,

            dataset_id=
                root_dataset_id,

            dataset_filename=
                "orders.csv",

            stage=
                "source",

            dataframe=
                orders_frame(),

            parent_dataset_ids=[],

            evidence_refs=[
                "source:test",
            ],
        )


        skip_clean(
            workflow_id=
                session.workflow_id,

            dataset_ids=[
                root_dataset_id,
            ],
        )


    elif (
        clean_mode
        ==
        "pass"
    ):
        put_preparation_artifact(
            workflow_id=
                session.workflow_id,

            dataset_id=
                root_dataset_id,

            dataset_filename=
                "orders.csv",

            stage=
                "clean",

            dataframe=
                orders_frame(),

            parent_dataset_ids=[
                root_dataset_id,
            ],

            evidence_refs=[
                "cleaning:test",
            ],
        )


        pass_clean(
            workflow_id=
                session.workflow_id,

            dataset_ids=[
                root_dataset_id,
            ],
        )


    elif (
        clean_mode
        ==
        "review"
    ):
        put_preparation_artifact(
            workflow_id=
                session.workflow_id,

            dataset_id=
                root_dataset_id,

            dataset_filename=
                "orders.csv",

            stage=
                "source",

            dataframe=
                orders_frame(),

            parent_dataset_ids=[],

            evidence_refs=[
                "source:test",
            ],
        )


        review_clean(
            workflow_id=
                session.workflow_id,

            dataset_ids=[
                root_dataset_id,
            ],
        )


    else:
        raise ValueError(
            (
                "Unsupported clean_mode="
                f"{clean_mode}"
            )
        )


    skip_transform(
        workflow_id=
            session.workflow_id,

        dataset_ids=[
            root_dataset_id,
        ],
    )


    skip_combine(
        workflow_id=
            session.workflow_id,

        dataset_ids=[
            root_dataset_id,
        ],
    )


    return (
        get_preparation_session(
            session.workflow_id
        )
    )


# ============================================================
# DERIVED COMBINE READY SESSION
# ============================================================


def build_joined_output_session():
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


    pass_required_stages(
        workflow_id=
            session.workflow_id,

        dataset_ids=
            root_dataset_ids,
    )


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


    pass_clean(
        workflow_id=
            session.workflow_id,

        dataset_ids=
            root_dataset_ids,
    )


    skip_transform(
        workflow_id=
            session.workflow_id,

        dataset_ids=
            root_dataset_ids,
    )


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


    return (
        get_preparation_session(
            session.workflow_id
        )
    )


# ============================================================
# 1. VERSION
# ============================================================


def test_version() -> None:
    assert (
        FINAL_PREPARATION_VALIDATION_V0_2_RULE_VERSION
        ==
        "final_preparation_validation_v0.2"
    )


    print(
        "Final Preparation Validation v0.2 version: PASS"
    )


# ============================================================
# 2. NO OUTPUT SELECTION
# ============================================================


def test_no_analysis_output_is_blocked() -> None:
    session = (
        build_single_root_session()
    )


    report = (
        evaluate_final_preparation_validation_v0_2(
            session
        )
    )


    assert not (
        report.passed
    )


    assert any(
        (
            check.code
            ==
            "analysis_outputs_selected"
            and
            check.passed
            is False
        )

        for check
        in report.checks
    )


    print(
        "Missing final analytical output is blocked: PASS"
    )


# ============================================================
# 3. ROOT OUTPUT
# ============================================================


def test_root_output_can_validate() -> None:
    session = (
        build_single_root_session(
            clean_mode=
                "skip"
        )
    )


    commit_analysis_output_selection(
        workflow_id=
            session.workflow_id,

        requested_dataset_ids=[
            "dataset:orders",
        ],
    )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    report = (
        require_final_preparation_validation_v0_2(
            current
        )
    )


    assert (
        report.passed
    )


    assert (
        report.analysis_output_dataset_ids
        ==
        [
            "dataset:orders",
        ]
    )


    print(
        "Prepared root output validates: PASS"
    )


# ============================================================
# 4. CLEAN OUTPUT
# ============================================================


def test_cleaned_root_output_can_validate() -> None:
    session = (
        build_single_root_session(
            clean_mode=
                "pass"
        )
    )


    commit_analysis_output_selection(
        workflow_id=
            session.workflow_id,

        requested_dataset_ids=[
            "dataset:orders",
        ],
    )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    report = (
        require_final_preparation_validation_v0_2(
            current
        )
    )


    assert (
        report.passed
    )


    print(
        "Cleaned root output validates: PASS"
    )


# ============================================================
# 5. DERIVED COMBINE OUTPUT
# ============================================================


def test_combine_output_validates_by_lineage() -> None:
    session = (
        build_joined_output_session()
    )


    commit_analysis_output_selection(
        workflow_id=
            session.workflow_id,

        requested_dataset_ids=[
            "sales_customers",
        ],
    )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    report = (
        require_final_preparation_validation_v0_2(
            current
        )
    )


    assert (
        report.passed
    )


    assert (
        report.preparation_root_dataset_ids
        ==
        [
            "sales",
            "customers",
        ]
    )


    assert (
        report.analysis_output_dataset_ids
        ==
        [
            "sales_customers",
        ]
    )


    legacy_combine_scope_checks = [
        check

        for check
        in report.checks

        if (
            check.code
            ==
            "combine_dataset_scope"
        )
    ]


    assert (
        legacy_combine_scope_checks
        ==
        []
    )


    assert any(
        (
            check.code
            ==
            "analysis_output_lineage_valid"
            and
            check.passed
            is True
        )

        for check
        in report.checks
    )


    print(
        "COMBINE output validates through lineage: PASS"
    )


# ============================================================
# 6. ARTIFACT DRIFT
# ============================================================


def test_missing_current_artifact_blocks_validation() -> None:
    session = (
        build_single_root_session()
    )


    commit_analysis_output_selection(
        workflow_id=
            session.workflow_id,

        requested_dataset_ids=[
            "dataset:orders",
        ],
    )


    delete_preparation_artifacts(
        workflow_id=
            session.workflow_id
    )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    report = (
        evaluate_final_preparation_validation_v0_2(
            current
        )
    )


    assert not (
        report.passed
    )


    assert any(
        (
            check.code
            ==
            "analysis_output_lineage_valid"
            and
            check.passed
            is False
        )

        for check
        in report.checks
    )


    print(
        "Missing current Artifact Store output is blocked: PASS"
    )


# ============================================================
# 7. CLEAN REVIEW STILL BLOCKS
# ============================================================


def test_clean_review_still_blocks() -> None:
    session = (
        build_single_root_session(
            clean_mode=
                "review"
        )
    )


    # Selection itself may be valid from a lineage perspective,
    # but Final Validation must still preserve the upstream
    # CLEAN review requirement.
    commit_analysis_output_selection(
        workflow_id=
            session.workflow_id,

        requested_dataset_ids=[
            "dataset:orders",
        ],
    )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    report = (
        evaluate_final_preparation_validation_v0_2(
            current
        )
    )


    assert not (
        report.passed
    )


    assert any(
        (
            check.code
            ==
            "clean_stage_resolved"
            and
            check.passed
            is False
        )

        for check
        in report.checks
    )


    try:
        require_final_preparation_validation_v0_2(
            current
        )


    except FinalPreparationValidationV02BlockedError:
        pass


    else:
        raise AssertionError(
            (
                "CLEAN review must remain blocking in "
                "Final Preparation Validation v0.2."
            )
        )


    print(
        "Existing CLEAN review guard remains active: PASS"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        (
            "=== DATALENS FINAL PREPARATION "
            "VALIDATION v0.2 ==="
        )
    )

    print()


    test_version()

    test_no_analysis_output_is_blocked()

    test_root_output_can_validate()

    test_cleaned_root_output_can_validate()

    test_combine_output_validates_by_lineage()

    test_missing_current_artifact_blocks_validation()

    test_clean_review_still_blocks()


    print()

    print(
        "Final Preparation Validation v0.2: PASS"
    )


if __name__ == "__main__":
    main()