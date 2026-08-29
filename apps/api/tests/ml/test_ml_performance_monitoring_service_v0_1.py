from __future__ import annotations


import inspect


from contextlib import (
    contextmanager,
)


from dataclasses import (
    dataclass,
)


import pandas as pd


import app.ml.performance_monitoring_service as service_module


from app.ml.model_loader import (
    MLModelLoaderArtifactError,
)


from app.ml.performance_evaluation_store import (
    list_ml_performance_evaluations_for_model,
)


from app.ml.performance_monitoring_service import (
    ML_PERFORMANCE_MONITORING_SERVICE_RULE_VERSION,
    MLPerformanceMonitoringObservedDatasetError,
    MLPerformanceMonitoringServiceAuthorityError,
    MLPerformanceMonitoringServiceExecutionError,
    MLPerformanceMonitoringServiceInputError,
    MLPerformanceMonitoringTargetError,
    run_ml_performance_monitoring,
)


from app.preparation.preparation_session import (
    record_validation_stage_signal,
)


from tests.ml.test_ml_monitoring_service_real_handoff_v0_1 import (
    build_ready_preparation_workflow,
    isolated_real_handoff_environment,
)


from tests.ml.test_ml_performance_evaluator_v0_1 import (
    classification_model,
)


# ============================================================
# FAKE HANDOFF
# ============================================================


@dataclass(
    frozen=True
)
class FakePerformanceHandoff:

    workflow_id: str

    session_revision: int

    dataset_ids: tuple[
        str,
        ...
    ]

    dataset_records: tuple[
        dict,
        ...
    ]


# ============================================================
# HELPERS
# ============================================================


def expect_error(
    error_type,
    factory,
) -> None:

    try:
        factory()

    except error_type:
        return


    raise AssertionError(
        (
            "Expected "
            f"{error_type.__name__}."
        )
    )


@contextmanager
def patched_trusted_model(
    trusted_model,
):

    original = (
        service_module
        .load_trusted_ml_model
    )


    def fake_load_trusted_ml_model(
        *,
        workflow_id: str,
        model_id: str,
    ):

        artifact = (
            trusted_model.artifact
        )


        if (
            workflow_id
            !=
            artifact.workflow_id
            or
            model_id
            !=
            artifact.model_id
        ):
            raise MLModelLoaderArtifactError(
                "trusted model not found"
            )


        return trusted_model


    service_module.load_trusted_ml_model = (
        fake_load_trusted_ml_model
    )


    try:
        yield

    finally:
        service_module.load_trusted_ml_model = (
            original
        )


@contextmanager
def patched_handoff(
    *,
    handoff,
):

    original = (
        service_module
        .load_validated_analysis_input
    )


    def fake_load_validated_analysis_input(
        *,
        workflow_id: str,
    ):

        return handoff


    service_module.load_validated_analysis_input = (
        fake_load_validated_analysis_input
    )


    try:
        yield

    finally:
        service_module.load_validated_analysis_input = (
            original
        )


@contextmanager
def patched_register_with_revision_race(
    *,
    workflow_id: str,
    dataset_id: str,
    expected_revision: int,
):

    original = (
        service_module
        .register_ml_performance_evaluation
    )


    changed = {
        "done":
            False
    }


    def racing_register(
        *,
        evaluation,
    ):

        if not changed[
            "done"
        ]:

            record_validation_stage_signal(
                workflow_id=
                    workflow_id,

                completed=
                    True,

                passed=
                    False,

                dataset_ids=[
                    dataset_id
                ],

                evidence_refs=[
                    "test:performance-service-race"
                ],

                blocking_reasons=[
                    "test:revision-changed"
                ],

                expected_revision=
                    expected_revision,
            )


            changed[
                "done"
            ] = True


        return original(
            evaluation=
                evaluation
        )


    service_module.register_ml_performance_evaluation = (
        racing_register
    )


    try:
        yield

    finally:
        service_module.register_ml_performance_evaluation = (
            original
        )


def trusted_classifier(
    *,
    session,
    dataset_id: str,
):

    return (
        classification_model(
            workflow_id=
                session.workflow_id,

            dataset_id=
                dataset_id,

            revision=
                session.revision,

            predictions=[
                0,
                1,
                0,
                1,
                0,
                1,
            ],
        )
    )


# ============================================================
# PUBLIC SURFACE
# ============================================================


def test_service_accepts_identifiers_only(
) -> None:

    signature = inspect.signature(
        run_ml_performance_monitoring
    )


    assert (
        set(
            signature.parameters
        )
        ==
        {
            "workflow_id",
            "model_id",
            "observed_dataset_id",
        }
    )


# ============================================================
# SUCCESS ? REAL PREPARATION HANDOFF
# ============================================================


def test_server_owned_performance_roundtrip(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            trusted_model,
            estimator,
        ) = (
            trusted_classifier(
                session=
                    session,

                dataset_id=
                    dataset_id,
            )
        )


        with patched_trusted_model(
            trusted_model
        ):

            result = (
                run_ml_performance_monitoring(
                    workflow_id=
                        session.workflow_id,

                    model_id=
                        trusted_model
                        .artifact
                        .model_id,

                    observed_dataset_id=
                        dataset_id,
                )
            )


        assert (
            result.workflow_id
            ==
            session.workflow_id
        )


        assert (
            result.model_id
            ==
            trusted_model
            .artifact
            .model_id
        )


        assert (
            result.observed_dataset_id
            ==
            dataset_id
        )


        assert (
            result
            .observed_preparation_session_revision
            ==
            session.revision
        )


        assert (
            result.primary_metric
            ==
            "f1_macro"
        )


        assert (
            result.performance_status
            ==
            "ok"
        )


        assert (
            estimator.call_count
            ==
            1
        )


        history = (
            list_ml_performance_evaluations_for_model(
                model_id=
                    trusted_model
                    .artifact
                    .model_id,

                workflow_id=
                    session.workflow_id,
            )
        )


        assert history == [
            result
        ]


        serialized = str(
            result.model_dump(
                mode="json"
            )
        )


        assert (
            "business_note"
            not in
            serialized
        )


        assert (
            "not-monitored"
            not in
            serialized
        )


        assert (
            "y_true"
            not in
            serialized
        )


        assert (
            "predictions"
            not in
            serialized
        )


# ============================================================
# INPUT
# ============================================================


def test_empty_identities_blocked(
) -> None:

    for (
        workflow_id,
        model_id,
        observed_dataset_id,
    ) in [
        (
            "",
            "model:x",
            "dataset:x",
        ),
        (
            "prep:x",
            "",
            "dataset:x",
        ),
        (
            "prep:x",
            "model:x",
            "",
        ),
    ]:

        expect_error(
            MLPerformanceMonitoringServiceInputError,

            lambda workflow_id=workflow_id, model_id=model_id, observed_dataset_id=observed_dataset_id:
                run_ml_performance_monitoring(
                    workflow_id=
                        workflow_id,

                    model_id=
                        model_id,

                    observed_dataset_id=
                        observed_dataset_id,
                ),
        )


# ============================================================
# MODEL AUTHORITY
# ============================================================


def test_cross_identity_model_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            trusted_model,
            _,
        ) = (
            trusted_classifier(
                session=
                    session,

                dataset_id=
                    dataset_id,
            )
        )


        with patched_trusted_model(
            trusted_model
        ):

            expect_error(
                MLPerformanceMonitoringServiceAuthorityError,

                lambda:
                    run_ml_performance_monitoring(
                        workflow_id=
                            session.workflow_id,

                        model_id=(
                            "model:"
                            +
                            "f" * 32
                        ),

                        observed_dataset_id=
                            dataset_id,
                    ),
            )


# ============================================================
# OBSERVED DATASET SCOPE
# ============================================================


def test_unauthorized_observed_dataset_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            trusted_model,
            _,
        ) = (
            trusted_classifier(
                session=
                    session,

                dataset_id=
                    dataset_id,
            )
        )


        with patched_trusted_model(
            trusted_model
        ):

            expect_error(
                MLPerformanceMonitoringObservedDatasetError,

                lambda:
                    run_ml_performance_monitoring(
                        workflow_id=
                            session.workflow_id,

                        model_id=
                            trusted_model
                            .artifact
                            .model_id,

                        observed_dataset_id=
                            "dataset:not-authorized",
                    ),
            )


# ============================================================
# TRUE TARGET REQUIRED
# ============================================================


def test_missing_target_blocked_before_predict(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            dataframe,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            trusted_model,
            estimator,
        ) = (
            trusted_classifier(
                session=
                    session,

                dataset_id=
                    dataset_id,
            )
        )


        without_target = (
            dataframe.drop(
                columns=[
                    "target"
                ]
            )
            .copy(
                deep=True
            )
        )


        handoff = (
            FakePerformanceHandoff(
                workflow_id=
                    session.workflow_id,

                session_revision=
                    session.revision,

                dataset_ids=(
                    dataset_id,
                ),

                dataset_records=(
                    {
                        "dataset_id":
                            dataset_id,

                        "dataframe":
                            without_target,
                    },
                ),
            )
        )


        with (
            patched_trusted_model(
                trusted_model
            ),
            patched_handoff(
                handoff=
                    handoff
            ),
        ):

            expect_error(
                MLPerformanceMonitoringTargetError,

                lambda:
                    run_ml_performance_monitoring(
                        workflow_id=
                            session.workflow_id,

                        model_id=
                            trusted_model
                            .artifact
                            .model_id,

                        observed_dataset_id=
                            dataset_id,
                    ),
            )


        assert (
            estimator.call_count
            ==
            0
        )


        assert (
            list_ml_performance_evaluations_for_model(
                model_id=
                    trusted_model
                    .artifact
                    .model_id,

                workflow_id=
                    session.workflow_id,
            )
            ==
            []
        )


# ============================================================
# REQUIRED FEATURE
# ============================================================


def test_missing_required_feature_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            dataframe,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            trusted_model,
            estimator,
        ) = (
            trusted_classifier(
                session=
                    session,

                dataset_id=
                    dataset_id,
            )
        )


        without_feature = (
            dataframe.drop(
                columns=[
                    "age"
                ]
            )
            .copy(
                deep=True
            )
        )


        handoff = (
            FakePerformanceHandoff(
                workflow_id=
                    session.workflow_id,

                session_revision=
                    session.revision,

                dataset_ids=(
                    dataset_id,
                ),

                dataset_records=(
                    {
                        "dataset_id":
                            dataset_id,

                        "dataframe":
                            without_feature,
                    },
                ),
            )
        )


        with (
            patched_trusted_model(
                trusted_model
            ),
            patched_handoff(
                handoff=
                    handoff
            ),
        ):

            expect_error(
                MLPerformanceMonitoringObservedDatasetError,

                lambda:
                    run_ml_performance_monitoring(
                        workflow_id=
                            session.workflow_id,

                        model_id=
                            trusted_model
                            .artifact
                            .model_id,

                        observed_dataset_id=
                            dataset_id,
                    ),
            )


        assert (
            estimator.call_count
            ==
            0
        )


# ============================================================
# LATE PREPARATION REVISION RACE
# ============================================================


def test_late_preparation_revision_race_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            trusted_model,
            estimator,
        ) = (
            trusted_classifier(
                session=
                    session,

                dataset_id=
                    dataset_id,
            )
        )


        with (
            patched_trusted_model(
                trusted_model
            ),
            patched_register_with_revision_race(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                expected_revision=
                    session.revision,
            ),
        ):

            expect_error(
                MLPerformanceMonitoringServiceExecutionError,

                lambda:
                    run_ml_performance_monitoring(
                        workflow_id=
                            session.workflow_id,

                        model_id=
                            trusted_model
                            .artifact
                            .model_id,

                        observed_dataset_id=
                            dataset_id,
                    ),
            )


        assert (
            estimator.call_count
            ==
            1
        )


        assert (
            list_ml_performance_evaluations_for_model(
                model_id=
                    trusted_model
                    .artifact
                    .model_id,

                workflow_id=
                    session.workflow_id,
            )
            ==
            []
        )


# ============================================================
# VERSION
# ============================================================


def test_rule_versions(
) -> None:

    assert (
        ML_PERFORMANCE_MONITORING_SERVICE_RULE_VERSION
        ==
        "ml_performance_monitoring_service_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML PERFORMANCE "
            "MONITORING SERVICE v0.1 ==="
        )
    )

    print()


    tests = [
        (
            "Public service accepts identifiers only",
            test_service_accepts_identifiers_only,
        ),
        (
            "Server-owned Performance Monitoring roundtrip",
            test_server_owned_performance_roundtrip,
        ),
        (
            "Empty identities blocked",
            test_empty_identities_blocked,
        ),
        (
            "Cross-identity Model Artifact blocked",
            test_cross_identity_model_blocked,
        ),
        (
            "Unauthorized observed dataset blocked",
            test_unauthorized_observed_dataset_blocked,
        ),
        (
            "Missing true target blocked before predict",
            test_missing_target_blocked_before_predict,
        ),
        (
            "Missing required feature blocked",
            test_missing_required_feature_blocked,
        ),
        (
            "Late Preparation revision race blocked",
            test_late_preparation_revision_race_blocked,
        ),
        (
            "Performance Monitoring Service rule version",
            test_rule_versions,
        ),
    ]


    for (
        label,
        callback,
    ) in tests:

        callback()

        print(
            f"[PASS] {label}"
        )


    print()

    print(
        (
            "PASS - ML Performance "
            "Monitoring Service v0.1"
        )
    )


if __name__ == "__main__":
    main()
