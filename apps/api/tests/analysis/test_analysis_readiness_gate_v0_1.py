from __future__ import annotations

from app.preparation.analysis_readiness_gate import (
    ANALYSIS_READINESS_GATE_RULE_VERSION,
    AnalysisDatasetNotAuthorizedError,
    AnalysisNotReadyError,
    evaluate_analysis_readiness,
    require_analysis_readiness,
)

from app.preparation.preparation_session import (
    PreparationSessionNotFoundError,
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
# HELPERS
# ============================================================


def create_session(
    dataset_id: str,
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
    root_dataset_id: str,
    output_dataset_id: str | None = None,
):
    final_output_dataset_id = (
        output_dataset_id
        or
        root_dataset_id
    )

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
            root_dataset_id
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
            root_dataset_id
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
            root_dataset_id
        ],

        evidence_refs=[
            "data_quality_engine_v0.2"
        ],

        blocking_reasons=[],
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
                final_output_dataset_id
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
                final_output_dataset_id
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
# INITIAL SESSION IS DENIED
# ============================================================


def test_initial_session_is_not_ready():
    dataset_id = (
        "dataset:orders"
    )

    session = (
        create_session(
            dataset_id
        )
    )

    decision = (
        evaluate_analysis_readiness(
            workflow_id=
                session.workflow_id,

            requested_analysis_dataset_ids=[
                dataset_id
            ],
        )
    )

    print(
        "\n=== INITIAL SESSION ==="
    )

    print(
        (
            "Preparation roots: "
            f"{decision.selected_analysis_dataset_ids}"
        )
    )

    print(
        (
            "Final outputs: "
            f"{decision.analysis_output_dataset_ids}"
        )
    )

    print(
        (
            "Authorized scope: "
            f"{decision.authorized_analysis_dataset_ids}"
        )
    )

    print(
        (
            "Workflow ready: "
            f"{decision.workflow_ready_for_analysis}"
        )
    )

    print(
        (
            "Dataset authorized: "
            f"{decision.dataset_scope_authorized}"
        )
    )

    print(
        (
            "Analysis ready: "
            f"{decision.ready_for_analysis}"
        )
    )

    print(
        (
            "Next stage: "
            f"{decision.next_stage}"
        )
    )

    assert (
        decision
        .selected_analysis_dataset_ids
        ==
        [
            dataset_id
        ]
    )

    assert (
        decision
        .analysis_output_dataset_ids
        ==
        []
    )

    # Before a final output exists, roots form only a
    # provisional authorization scope.
    assert (
        decision
        .authorized_analysis_dataset_ids
        ==
        [
            dataset_id
        ]
    )

    assert (
        decision
        .dataset_scope_authorized
        is True
    )

    assert (
        decision
        .workflow_ready_for_analysis
        is False
    )

    assert (
        decision
        .ready_for_analysis
        is False
    )

    assert (
        decision.next_stage
        ==
        PreparationStage.IMPORT
    )

    try:
        require_analysis_readiness(
            workflow_id=
                session.workflow_id,

            requested_analysis_dataset_ids=[
                dataset_id
            ],
        )

    except AnalysisNotReadyError as exc:
        assert (
            exc
            .decision
            .ready_for_analysis
            is False
        )

    else:
        raise AssertionError(
            (
                "Expected AnalysisNotReadyError "
                "for initial Preparation session."
            )
        )


# ============================================================
# READY ROOT OUTPUT IS AUTHORIZED
# ============================================================


def test_ready_session_is_authorized():
    dataset_id = (
        "dataset:customers"
    )

    session = (
        create_session(
            dataset_id
        )
    )

    ready_session = (
        make_session_ready(
            workflow_id=
                session.workflow_id,

            root_dataset_id=
                dataset_id,
        )
    )

    decision = (
        require_analysis_readiness(
            workflow_id=
                session.workflow_id,

            requested_analysis_dataset_ids=[
                dataset_id
            ],
        )
    )

    print(
        "\n=== READY ROOT OUTPUT ==="
    )

    print(
        (
            "Revision: "
            f"{decision.session_revision}"
        )
    )

    print(
        (
            "Final outputs: "
            f"{decision.analysis_output_dataset_ids}"
        )
    )

    print(
        (
            "Workflow ready: "
            f"{decision.workflow_ready_for_analysis}"
        )
    )

    print(
        (
            "Dataset authorized: "
            f"{decision.dataset_scope_authorized}"
        )
    )

    print(
        (
            "Analysis ready: "
            f"{decision.ready_for_analysis}"
        )
    )

    # 3 required stage writes
    # + final output selection
    # + VALIDATE
    assert (
        ready_session.revision
        ==
        5
    )

    assert (
        decision.session_revision
        ==
        5
    )

    assert (
        decision
        .workflow_ready_for_analysis
        is True
    )

    assert (
        decision
        .dataset_scope_authorized
        is True
    )

    assert (
        decision
        .requested_datasets_validated
        is True
    )

    assert (
        decision
        .ready_for_analysis
        is True
    )

    assert (
        decision
        .requested_analysis_dataset_ids
        ==
        [
            dataset_id
        ]
    )

    assert (
        decision
        .analysis_output_dataset_ids
        ==
        [
            dataset_id
        ]
    )

    assert (
        decision
        .authorized_analysis_dataset_ids
        ==
        [
            dataset_id
        ]
    )

    assert (
        decision
        .validated_analysis_dataset_ids
        ==
        [
            dataset_id
        ]
    )


# ============================================================
# DERIVED FINAL OUTPUT IS AUTHORIZED
# ============================================================


def test_derived_final_output_is_authorized():
    root_dataset_id = (
        "dataset:sales"
    )

    final_output_dataset_id = (
        "dataset:sales_customers"
    )

    session = (
        create_session(
            root_dataset_id
        )
    )

    make_session_ready(
        workflow_id=
            session.workflow_id,

        root_dataset_id=
            root_dataset_id,

        output_dataset_id=
            final_output_dataset_id,
    )

    decision = (
        require_analysis_readiness(
            workflow_id=
                session.workflow_id,

            requested_analysis_dataset_ids=[
                final_output_dataset_id
            ],
        )
    )

    print(
        "\n=== DERIVED FINAL OUTPUT ==="
    )

    print(
        (
            "Preparation roots: "
            f"{decision.selected_analysis_dataset_ids}"
        )
    )

    print(
        (
            "Final outputs: "
            f"{decision.analysis_output_dataset_ids}"
        )
    )

    print(
        (
            "Authorized scope: "
            f"{decision.authorized_analysis_dataset_ids}"
        )
    )

    assert (
        decision
        .selected_analysis_dataset_ids
        ==
        [
            root_dataset_id
        ]
    )

    assert (
        decision
        .analysis_output_dataset_ids
        ==
        [
            final_output_dataset_id
        ]
    )

    assert (
        decision
        .authorized_analysis_dataset_ids
        ==
        [
            final_output_dataset_id
        ]
    )

    assert (
        decision
        .validated_analysis_dataset_ids
        ==
        [
            final_output_dataset_id
        ]
    )

    assert (
        decision
        .requested_analysis_dataset_ids
        ==
        [
            final_output_dataset_id
        ]
    )

    assert (
        decision
        .ready_for_analysis
        is True
    )


# ============================================================
# ROOT CANNOT BYPASS DERIVED FINAL OUTPUT
# ============================================================


def test_root_cannot_bypass_derived_final_output():
    root_dataset_id = (
        "dataset:sales"
    )

    final_output_dataset_id = (
        "dataset:sales_customers"
    )

    session = (
        create_session(
            root_dataset_id
        )
    )

    make_session_ready(
        workflow_id=
            session.workflow_id,

        root_dataset_id=
            root_dataset_id,

        output_dataset_id=
            final_output_dataset_id,
    )

    decision = (
        evaluate_analysis_readiness(
            workflow_id=
                session.workflow_id,

            requested_analysis_dataset_ids=[
                root_dataset_id
            ],
        )
    )

    print(
        "\n=== ROOT BYPASS ATTEMPT ==="
    )

    print(
        (
            "Final output: "
            f"{decision.analysis_output_dataset_ids}"
        )
    )

    print(
        (
            "Requested root: "
            f"{decision.requested_analysis_dataset_ids}"
        )
    )

    print(
        (
            "Unauthorized datasets: "
            f"{decision.unauthorized_dataset_ids}"
        )
    )

    assert (
        decision
        .workflow_ready_for_analysis
        is True
    )

    assert (
        decision
        .dataset_scope_authorized
        is False
    )

    assert (
        decision
        .ready_for_analysis
        is False
    )

    assert (
        decision
        .unauthorized_dataset_ids
        ==
        [
            root_dataset_id
        ]
    )

    try:
        require_analysis_readiness(
            workflow_id=
                session.workflow_id,

            requested_analysis_dataset_ids=[
                root_dataset_id
            ],
        )

    except AnalysisDatasetNotAuthorizedError as exc:
        assert (
            exc
            .decision
            .dataset_scope_authorized
            is False
        )

    else:
        raise AssertionError(
            (
                "Preparation root must not bypass "
                "the selected final analytical output."
            )
        )


# ============================================================
# UNRELATED DATASET CANNOT BE REUSED
# ============================================================


def test_ready_session_cannot_authorize_other_dataset():
    selected_dataset_id = (
        "dataset:orders"
    )

    other_dataset_id = (
        "dataset:customers"
    )

    session = (
        create_session(
            selected_dataset_id
        )
    )

    make_session_ready(
        workflow_id=
            session.workflow_id,

        root_dataset_id=
            selected_dataset_id,
    )

    decision = (
        evaluate_analysis_readiness(
            workflow_id=
                session.workflow_id,

        requested_analysis_dataset_ids=[
                other_dataset_id
            ],
        )
    )

    print(
        "\n=== DATASET SCOPE ATTEMPT ==="
    )

    print(
        (
            "Workflow ready: "
            f"{decision.workflow_ready_for_analysis}"
        )
    )

    print(
        (
            "Dataset authorized: "
            f"{decision.dataset_scope_authorized}"
        )
    )

    print(
        (
            "Analysis ready: "
            f"{decision.ready_for_analysis}"
        )
    )

    print(
        (
            "Unauthorized datasets: "
            f"{decision.unauthorized_dataset_ids}"
        )
    )

    print(
        (
            "Unvalidated datasets: "
            f"{decision.unvalidated_dataset_ids}"
        )
    )

    assert (
        decision
        .workflow_ready_for_analysis
        is True
    )

    assert (
        decision
        .dataset_scope_authorized
        is False
    )

    assert (
        decision
        .ready_for_analysis
        is False
    )

    assert (
        decision
        .unauthorized_dataset_ids
        ==
        [
            other_dataset_id
        ]
    )

    assert (
        decision
        .unvalidated_dataset_ids
        ==
        [
            other_dataset_id
        ]
    )

    try:
        require_analysis_readiness(
            workflow_id=
                session.workflow_id,

            requested_analysis_dataset_ids=[
                other_dataset_id
            ],
        )

    except AnalysisDatasetNotAuthorizedError as exc:
        assert (
            exc
            .decision
            .dataset_scope_authorized
            is False
        )

    else:
        raise AssertionError(
            (
                "Expected "
                "AnalysisDatasetNotAuthorizedError."
            )
        )


# ============================================================
# DEFAULT REQUEST SCOPE USES FINAL OUTPUT
# ============================================================


def test_default_scope_uses_final_output_selection():
    root_dataset_id = (
        "dataset:orders"
    )

    final_output_dataset_id = (
        "dataset:orders_prepared"
    )

    session = (
        create_session(
            root_dataset_id
        )
    )

    make_session_ready(
        workflow_id=
            session.workflow_id,

        root_dataset_id=
            root_dataset_id,

        output_dataset_id=
            final_output_dataset_id,
    )

    decision = (
        require_analysis_readiness(
            workflow_id=
                session.workflow_id
        )
    )

    print(
        "\n=== DEFAULT FINAL OUTPUT SCOPE ==="
    )

    print(
        (
            "Requested datasets: "
            f"{decision.requested_analysis_dataset_ids}"
        )
    )

    assert (
        decision
        .requested_analysis_dataset_ids
        ==
        [
            final_output_dataset_id
        ]
    )

    assert (
        decision
        .authorized_analysis_dataset_ids
        ==
        [
            final_output_dataset_id
        ]
    )

    assert (
        decision
        .ready_for_analysis
        is True
    )


# ============================================================
# UNKNOWN SESSION
# ============================================================


def test_unknown_session_is_rejected():
    print(
        "\n=== UNKNOWN SESSION ==="
    )

    try:
        require_analysis_readiness(
            workflow_id=
                "prep:does-not-exist",

            requested_analysis_dataset_ids=[
                "dataset:orders"
            ],
        )

    except PreparationSessionNotFoundError:
        print(
            "Unknown session rejected: True"
        )

    else:
        raise AssertionError(
            (
                "Expected "
                "PreparationSessionNotFoundError."
            )
        )


# ============================================================
# EMPTY WORKFLOW ID
# ============================================================


def test_empty_workflow_id_is_rejected():
    try:
        require_analysis_readiness(
            workflow_id=
                "   ",

            requested_analysis_dataset_ids=[
                "dataset:orders"
            ],
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            (
                "Expected ValueError for empty "
                "workflow_id."
            )
        )


# ============================================================
# EMPTY DATASET ID
# ============================================================


def test_empty_dataset_id_is_rejected():
    dataset_id = (
        "dataset:orders-empty-test"
    )

    session = (
        create_session(
            dataset_id
        )
    )

    make_session_ready(
        workflow_id=
            session.workflow_id,

        root_dataset_id=
            dataset_id,
    )

    try:
        require_analysis_readiness(
            workflow_id=
                session.workflow_id,

            requested_analysis_dataset_ids=[
                "   "
            ],
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            (
                "Expected ValueError for empty "
                "analysis dataset_id."
            )
        )


# ============================================================
# VERSION
# ============================================================


def test_rule_version():
    assert (
        ANALYSIS_READINESS_GATE_RULE_VERSION
        ==
        "analysis_readiness_gate_v0.2"
    )


# ============================================================
# MAIN
# ============================================================


def main():
    reset_preparation_session_store_for_tests()

    print(
        "\n========================================"
    )

    print(
        "DataLens Analysis Readiness Gate v0.2"
    )

    print(
        "========================================"
    )

    test_initial_session_is_not_ready()

    test_ready_session_is_authorized()

    test_derived_final_output_is_authorized()

    test_root_cannot_bypass_derived_final_output()

    test_ready_session_cannot_authorize_other_dataset()

    test_default_scope_uses_final_output_selection()

    test_unknown_session_is_rejected()

    test_empty_workflow_id_is_rejected()

    test_empty_dataset_id_is_rejected()

    test_rule_version()

    print(
        "\n========================================"
    )

    print(
        "PASS - analysis readiness gate v0.2"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()