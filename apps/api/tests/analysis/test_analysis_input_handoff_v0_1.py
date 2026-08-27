from __future__ import annotations


from unittest.mock import (
    patch,
)


import pandas as pd


from app.preparation.analysis_input_handoff import (
    ANALYSIS_INPUT_HANDOFF_RULE_VERSION,
    AnalysisInputHandoffError,
    AnalysisPreparedArtifactUnavailableError,
    load_validated_analysis_input,
)

from app.preparation.analysis_readiness_gate import (
    AnalysisNotReadyError,
    require_analysis_readiness as real_require_analysis_readiness,
)

from app.preparation.preparation_artifact_store import (
    delete_preparation_artifacts,
    get_preparation_artifact,
    put_preparation_artifact,
    reset_preparation_artifact_store_for_tests,
)

from app.preparation.preparation_session import (
    create_preparation_session,
    get_preparation_session,
    record_analysis_output_selection,
    record_optional_stage_signal,
    record_required_stage_signal,
    record_validation_stage_signal,
    reset_preparation_session_store_for_tests,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
)


# ============================================================
# DATASETS
# ============================================================


ROOT_DATASET_ID = (
    "dataset:orders"
)


FINAL_DATASET_ID = (
    "dataset:orders_prepared"
)


SECOND_ROOT_DATASET_ID = (
    "dataset:customers"
)


SECOND_FINAL_DATASET_ID = (
    "dataset:customers_prepared"
)


# ============================================================
# DATA
# ============================================================


def root_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "order_id": [
                    "O001",
                    "O002",
                    "O003",
                ],

                "amount": [
                    100.0,
                    200.0,
                    300.0,
                ],

                "segment": [
                    "A",
                    "B",
                    "A",
                ],
            }
        )
    )


def prepared_frame() -> pd.DataFrame:
    """
    Deliberately differs from the root DataFrame.

    This makes it possible to prove that the handoff uses the
    final server-owned artifact rather than the root dataset.
    """

    return (
        pd.DataFrame(
            {
                "order_id": [
                    "O001",
                    "O002",
                ],

                "amount": [
                    110.0,
                    220.0,
                ],

                "segment": [
                    "A",
                    "B",
                ],

                "validated_metric": [
                    1.5,
                    2.5,
                ],
            }
        )
    )


def second_root_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "customer_id": [
                    "C001",
                    "C002",
                    "C003",
                ],

                "customer_name": [
                    "Alice",
                    "Bob",
                    "Charlie",
                ],

                "segment": [
                    "A",
                    "B",
                    "A",
                ],
            }
        )
    )


def second_prepared_frame() -> pd.DataFrame:
    """
    Second independently materialized final analytical output.

    This dataset deliberately differs from its Preparation root
    so the multi-output handoff can prove that both final
    artifacts, and only those artifacts, cross into Analysis.
    """

    return (
        pd.DataFrame(
            {
                "customer_id": [
                    "C001",
                    "C002",
                ],

                "customer_name": [
                    "Alice",
                    "Bob",
                ],

                "segment": [
                    "A",
                    "B",
                ],

                "validated_customer_metric": [
                    10.0,
                    20.0,
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
    dataset_id: str,
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

            dataset_ids=[
                dataset_id,
            ],

            evidence_refs=[
                (
                    "test:"
                    f"{stage.value}"
                )
            ],

            blocking_reasons=[],
        )


def pass_required_stages_for_datasets(
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
                list(
                    dataset_ids
                ),

            evidence_refs=[
                (
                    "test:"
                    f"{stage.value}"
                )
            ],

            blocking_reasons=[],
        )


# ============================================================
# ROOT OUTPUT PREPARATION
# ============================================================


def build_ready_root_output():
    reset_state()


    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                ROOT_DATASET_ID,
            ]
        )
    )


    pass_required_stages(
        workflow_id=
            session.workflow_id,

        dataset_id=
            ROOT_DATASET_ID,
    )


    put_preparation_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            ROOT_DATASET_ID,

        dataset_filename=
            "orders.csv",

        stage=
            "source",

        dataframe=
            root_frame(),

        parent_dataset_ids=[],

        evidence_refs=[
            "test:source",
        ],
    )


    before_selection = (
        get_preparation_session(
            session.workflow_id
        )
    )


    selected = (
        record_analysis_output_selection(
            workflow_id=
                session.workflow_id,

            analysis_output_dataset_ids=[
                ROOT_DATASET_ID,
            ],

            expected_revision=
                before_selection.revision,
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
                ROOT_DATASET_ID,
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


    return (
        ready
    )


# ============================================================
# DERIVED OUTPUT PREPARATION
# ============================================================


def build_ready_derived_output():
    reset_state()


    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                ROOT_DATASET_ID,
            ]
        )
    )


    pass_required_stages(
        workflow_id=
            session.workflow_id,

        dataset_id=
            ROOT_DATASET_ID,
    )


    # ========================================================
    # ROOT MATERIALIZATION
    # ========================================================

    put_preparation_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            ROOT_DATASET_ID,

        dataset_filename=
            "orders.csv",

        stage=
            "source",

        dataframe=
            root_frame(),

        parent_dataset_ids=[],

        evidence_refs=[
            "test:source",
        ],
    )


    # ========================================================
    # FINAL CLEAN OUTPUT MATERIALIZATION
    # ========================================================

    put_preparation_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            FINAL_DATASET_ID,

        dataset_filename=
            "orders_prepared.csv",

        stage=
            "clean",

        dataframe=
            prepared_frame(),

        parent_dataset_ids=[
            ROOT_DATASET_ID,
        ],

        evidence_refs=[
            "test:clean",
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

        dataset_ids=[
            FINAL_DATASET_ID,
        ],

        evidence_refs=[
            "cleaning:test",
        ],

        blocking_reasons=[],
    )


    # TRANSFORM and COMBINE remain SKIPPED by default.


    # ========================================================
    # FINAL OUTPUT SELECTION
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

            analysis_output_dataset_ids=[
                FINAL_DATASET_ID,
            ],

            expected_revision=
                before_selection.revision,
        )
    )


    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    ready = (
        record_validation_stage_signal(
            workflow_id=
                session.workflow_id,

            completed=
                True,

            passed=
                True,

            dataset_ids=[
                FINAL_DATASET_ID,
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


    return (
        ready
    )


# ============================================================
# MULTI-OUTPUT PREPARATION
# ============================================================


def build_ready_multi_output():
    reset_state()


    root_dataset_ids = [
        ROOT_DATASET_ID,
        SECOND_ROOT_DATASET_ID,
    ]


    final_dataset_ids = [
        FINAL_DATASET_ID,
        SECOND_FINAL_DATASET_ID,
    ]


    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=
                root_dataset_ids
        )
    )


    pass_required_stages_for_datasets(
        workflow_id=
            session.workflow_id,

        dataset_ids=
            root_dataset_ids,
    )


    # ========================================================
    # ROOT ARTIFACTS
    # ========================================================

    put_preparation_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            ROOT_DATASET_ID,

        dataset_filename=
            "orders.csv",

        stage=
            "source",

        dataframe=
            root_frame(),

        parent_dataset_ids=[],

        evidence_refs=[
            "test:source:orders",
        ],
    )


    put_preparation_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            SECOND_ROOT_DATASET_ID,

        dataset_filename=
            "customers.csv",

        stage=
            "source",

        dataframe=
            second_root_frame(),

        parent_dataset_ids=[],

        evidence_refs=[
            "test:source:customers",
        ],
    )


    # ========================================================
    # FINAL ARTIFACTS
    # ========================================================

    put_preparation_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            FINAL_DATASET_ID,

        dataset_filename=
            "orders_prepared.csv",

        stage=
            "clean",

        dataframe=
            prepared_frame(),

        parent_dataset_ids=[
            ROOT_DATASET_ID,
        ],

        evidence_refs=[
            "test:clean:orders",
        ],
    )


    put_preparation_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            SECOND_FINAL_DATASET_ID,

        dataset_filename=
            "customers_prepared.csv",

        stage=
            "clean",

        dataframe=
            second_prepared_frame(),

        parent_dataset_ids=[
            SECOND_ROOT_DATASET_ID,
        ],

        evidence_refs=[
            "test:clean:customers",
        ],
    )


    # ========================================================
    # CLEAN PASSED FOR BOTH FINAL OUTPUTS
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
            final_dataset_ids,

        evidence_refs=[
            "cleaning:test:multi-output",
        ],

        blocking_reasons=[],
    )


    # ========================================================
    # FINAL OUTPUT SELECTION
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
                final_dataset_ids,

            expected_revision=
                before_selection.revision,
        )
    )


    # ========================================================
    # FINAL VALIDATION
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
                final_dataset_ids,

            evidence_refs=[
                "final_validation:test:multi-output",
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


    return (
        ready
    )


# ============================================================
# 1. ROOT OUTPUT HANDOFF
# ============================================================


def test_root_output_handoff() -> None:
    session = (
        build_ready_root_output()
    )


    handoff = (
        load_validated_analysis_input(
            workflow_id=
                session.workflow_id
        )
    )


    assert (
        handoff.workflow_id
        ==
        session.workflow_id
    )


    assert (
        handoff.dataset_ids
        ==
        (
            ROOT_DATASET_ID,
        )
    )


    assert (
        handoff.ingestion.dataset_count
        ==
        1
    )


    assert (
        handoff.ingestion.total_rows
        ==
        3
    )


    assert (
        len(
            handoff.dataset_records
        )
        ==
        1
    )


    record = (
        handoff.dataset_records[
            0
        ]
    )


    assert (
        record[
            "dataset_id"
        ]
        ==
        ROOT_DATASET_ID
    )


    assert (
        record[
            "filename"
        ]
        ==
        "orders.csv"
    )


    assert (
        record[
            "extension"
        ]
        ==
        ".csv"
    )


    assert (
        record[
            "preparation_stage"
        ]
        ==
        "source"
    )


    pd.testing.assert_frame_equal(
        record[
            "dataframe"
        ],

        root_frame(),
    )


# ============================================================
# 2. DERIVED OUTPUT REPLACES ROOT
# ============================================================


def test_derived_output_is_the_only_analysis_input(
) -> None:
    session = (
        build_ready_derived_output()
    )


    handoff = (
        load_validated_analysis_input(
            workflow_id=
                session.workflow_id
        )
    )


    assert (
        handoff.dataset_ids
        ==
        (
            FINAL_DATASET_ID,
        )
    )


    assert (
        len(
            handoff.dataset_records
        )
        ==
        1
    )


    record = (
        handoff.dataset_records[
            0
        ]
    )


    assert (
        record[
            "dataset_id"
        ]
        ==
        FINAL_DATASET_ID
    )


    assert (
        record[
            "dataset_id"
        ]
        !=
        ROOT_DATASET_ID
    )


    assert (
        record[
            "filename"
        ]
        ==
        "orders_prepared.csv"
    )


    assert (
        record[
            "preparation_stage"
        ]
        ==
        "clean"
    )


    assert (
        record[
            "preparation_parent_dataset_ids"
        ]
        ==
        [
            ROOT_DATASET_ID,
        ]
    )


    pd.testing.assert_frame_equal(
        record[
            "dataframe"
        ],

        prepared_frame(),
    )


    assert (
        list(
            record[
                "dataframe"
            ]
            .columns
        )
        ==
        [
            "order_id",
            "amount",
            "segment",
            "validated_metric",
        ]
    )


# ============================================================
# 3. MANIFEST MATCHES FINAL DATAFRAME
# ============================================================


def test_manifest_is_rebuilt_from_final_artifact(
) -> None:
    session = (
        build_ready_derived_output()
    )


    handoff = (
        load_validated_analysis_input(
            workflow_id=
                session.workflow_id
        )
    )


    manifest = (
        handoff.ingestion.datasets[
            0
        ]
    )


    assert (
        manifest.dataset_id
        ==
        FINAL_DATASET_ID
    )


    assert (
        manifest.filename
        ==
        "orders_prepared.csv"
    )


    assert (
        manifest.row_count
        ==
        2
    )


    assert (
        manifest.column_count
        ==
        4
    )


    column_names = [
        column.name

        for column
        in manifest.columns
    ]


    assert (
        column_names
        ==
        [
            "order_id",
            "amount",
            "segment",
            "validated_metric",
        ]
    )


# ============================================================
# 4. ARTIFACT STORE COPY ISOLATION
# ============================================================


def test_handoff_dataframe_does_not_mutate_store(
) -> None:
    session = (
        build_ready_derived_output()
    )


    handoff = (
        load_validated_analysis_input(
            workflow_id=
                session.workflow_id
        )
    )


    dataframe = (
        handoff
        .dataset_records[
            0
        ][
            "dataframe"
        ]
    )


    dataframe.loc[
        0,
        "amount",
    ] = (
        999999.0
    )


    stored = (
        get_preparation_artifact(
            workflow_id=
                session.workflow_id,

            dataset_id=
                FINAL_DATASET_ID,
        )
    )


    assert (
        float(
            stored
            .dataframe
            .loc[
                0,
                "amount",
            ]
        )
        ==
        110.0
    )


# ============================================================
# 5. MISSING ARTIFACT FAILS CLOSED
# ============================================================


def test_missing_validated_artifact_fails_closed(
) -> None:
    session = (
        build_ready_derived_output()
    )


    delete_preparation_artifacts(
        workflow_id=
            session.workflow_id
    )


    try:
        load_validated_analysis_input(
            workflow_id=
                session.workflow_id
        )


    except AnalysisPreparedArtifactUnavailableError as error:
        assert (
            error.workflow_id
            ==
            session.workflow_id
        )


        assert (
            error.dataset_id
            ==
            FINAL_DATASET_ID
        )


    else:
        raise AssertionError(
            (
                "Analysis must fail closed when the "
                "validated Preparation artifact has "
                "disappeared."
            )
        )


# ============================================================
# 6. UNREADY WORKFLOW CANNOT CROSS HANDOFF
# ============================================================


def test_unready_workflow_is_rejected(
) -> None:
    reset_state()


    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                ROOT_DATASET_ID,
            ]
        )
    )


    put_preparation_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            ROOT_DATASET_ID,

        dataset_filename=
            "orders.csv",

        stage=
            "source",

        dataframe=
            root_frame(),

        parent_dataset_ids=[],

        evidence_refs=[
            "test:source",
        ],
    )


    try:
        load_validated_analysis_input(
            workflow_id=
                session.workflow_id
        )


    except AnalysisNotReadyError:
        pass


    else:
        raise AssertionError(
            (
                "An unready Preparation workflow "
                "must not cross the analysis handoff."
            )
        )


# ============================================================
# 7. MULTIPLE FINAL OUTPUTS
# ============================================================


def test_multiple_final_outputs_are_handed_off_exactly(
) -> None:
    session = (
        build_ready_multi_output()
    )


    handoff = (
        load_validated_analysis_input(
            workflow_id=
                session.workflow_id
        )
    )


    assert (
        handoff.dataset_ids
        ==
        (
            FINAL_DATASET_ID,
            SECOND_FINAL_DATASET_ID,
        )
    )


    assert (
        handoff.ingestion.dataset_count
        ==
        2
    )


    assert (
        handoff.ingestion.total_rows
        ==
        4
    )


    assert (
        len(
            handoff.dataset_records
        )
        ==
        2
    )


    record_ids = [
        record[
            "dataset_id"
        ]

        for record
        in handoff.dataset_records
    ]


    assert (
        record_ids
        ==
        [
            FINAL_DATASET_ID,
            SECOND_FINAL_DATASET_ID,
        ]
    )


    assert (
        ROOT_DATASET_ID
        not in
        record_ids
    )


    assert (
        SECOND_ROOT_DATASET_ID
        not in
        record_ids
    )


    first_record = (
        handoff.dataset_records[
            0
        ]
    )


    second_record = (
        handoff.dataset_records[
            1
        ]
    )


    assert (
        first_record[
            "preparation_parent_dataset_ids"
        ]
        ==
        [
            ROOT_DATASET_ID,
        ]
    )


    assert (
        second_record[
            "preparation_parent_dataset_ids"
        ]
        ==
        [
            SECOND_ROOT_DATASET_ID,
        ]
    )


    pd.testing.assert_frame_equal(
        first_record[
            "dataframe"
        ],

        prepared_frame(),
    )


    pd.testing.assert_frame_equal(
        second_record[
            "dataframe"
        ],

        second_prepared_frame(),
    )


    manifest_ids = [
        manifest.dataset_id

        for manifest
        in handoff.ingestion.datasets
    ]


    assert (
        manifest_ids
        ==
        [
            FINAL_DATASET_ID,
            SECOND_FINAL_DATASET_ID,
        ]
    )


# ============================================================
# 8. SESSION REVISION CHANGE FAILS CLOSED
# ============================================================


def test_session_revision_change_during_handoff_fails_closed(
) -> None:
    session = (
        build_ready_derived_output()
    )


    call_count = (
        0
    )


    def changing_readiness(
        *,
        workflow_id: str,
        requested_analysis_dataset_ids=None,
    ):
        nonlocal call_count


        call_count += (
            1
        )


        decision = (
            real_require_analysis_readiness(
                workflow_id=
                    workflow_id,

                requested_analysis_dataset_ids=
                    requested_analysis_dataset_ids,
            )
        )


        if (
            call_count
            ==
            2
        ):
            return (
                decision.model_copy(
                    update={
                        "session_revision":
                            (
                                decision
                                .session_revision
                                +
                                1
                            )
                    }
                )
            )


        return (
            decision
        )


    with patch(
        (
            "app.preparation.analysis_input_handoff."
            "require_analysis_readiness"
        ),

        side_effect=
            changing_readiness,
    ):
        try:
            load_validated_analysis_input(
                workflow_id=
                    session.workflow_id
            )


        except AnalysisInputHandoffError as error:
            assert (
                "Preparation session changed"
                in
                str(
                    error
                )
            )


            assert (
                "initial_revision="
                in
                str(
                    error
                )
            )


            assert (
                "current_revision="
                in
                str(
                    error
                )
            )


        else:
            raise AssertionError(
                (
                    "Analysis handoff must fail closed "
                    "when the Preparation session revision "
                    "changes while artifacts are loaded."
                )
            )


    assert (
        call_count
        ==
        2
    )


# ============================================================
# 9. FINAL OUTPUT SCOPE CHANGE FAILS CLOSED
# ============================================================


def test_final_output_scope_change_during_handoff_fails_closed(
) -> None:
    session = (
        build_ready_derived_output()
    )


    call_count = (
        0
    )


    def changing_scope_readiness(
        *,
        workflow_id: str,
        requested_analysis_dataset_ids=None,
    ):
        nonlocal call_count


        call_count += (
            1
        )


        decision = (
            real_require_analysis_readiness(
                workflow_id=
                    workflow_id,

                requested_analysis_dataset_ids=
                    requested_analysis_dataset_ids,
            )
        )


        if (
            call_count
            ==
            2
        ):
            changed_scope = [
                ROOT_DATASET_ID,
            ]


            return (
                decision.model_copy(
                    update={
                        "analysis_output_dataset_ids":
                            changed_scope,

                        "authorized_analysis_dataset_ids":
                            changed_scope,

                        "validated_analysis_dataset_ids":
                            changed_scope,

                        "requested_analysis_dataset_ids":
                            changed_scope,
                    }
                )
            )


        return (
            decision
        )


    with patch(
        (
            "app.preparation.analysis_input_handoff."
            "require_analysis_readiness"
        ),

        side_effect=
            changing_scope_readiness,
    ):
        try:
            load_validated_analysis_input(
                workflow_id=
                    session.workflow_id
            )


        except AnalysisInputHandoffError as error:
            assert (
                (
                    "Preparation final analysis output "
                    "scope changed"
                )
                in
                str(
                    error
                )
            )


        else:
            raise AssertionError(
                (
                    "Analysis handoff must fail closed "
                    "when the final analytical output "
                    "scope changes while artifacts are "
                    "being loaded."
                )
            )


    assert (
        call_count
        ==
        2
    )


# ============================================================
# 10. VERSION
# ============================================================


def test_rule_version() -> None:
    assert (
        ANALYSIS_INPUT_HANDOFF_RULE_VERSION
        ==
        "analysis_input_handoff_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        "=== DATALENS ANALYSIS INPUT HANDOFF v0.1 ==="
    )

    print()


    test_root_output_handoff()

    print(
        "Validated root output handoff: PASS"
    )


    test_derived_output_is_the_only_analysis_input()

    print(
        "Derived output replaces Preparation root: PASS"
    )


    test_manifest_is_rebuilt_from_final_artifact()

    print(
        "Manifest rebuilt from final artifact: PASS"
    )


    test_handoff_dataframe_does_not_mutate_store()

    print(
        "Artifact Store copy isolation: PASS"
    )


    test_missing_validated_artifact_fails_closed()

    print(
        "Missing validated artifact fails closed: PASS"
    )


    test_unready_workflow_is_rejected()

    print(
        "Unready workflow cannot cross handoff: PASS"
    )


    test_multiple_final_outputs_are_handed_off_exactly()

    print(
        "Multiple final outputs handoff: PASS"
    )


    test_session_revision_change_during_handoff_fails_closed()

    print(
        "Session revision race fails closed: PASS"
    )


    test_final_output_scope_change_during_handoff_fails_closed()

    print(
        "Final output scope race fails closed: PASS"
    )


    test_rule_version()

    print(
        "Analysis Input Handoff version: PASS"
    )


    print()

    print(
        "Analysis Input Handoff v0.1: PASS"
    )


if __name__ == "__main__":
    main()