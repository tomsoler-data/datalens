from __future__ import annotations


from dataclasses import (
    dataclass,
)


import numpy as np
import pandas as pd


from app.ml.model_loader import (
    LoadedMLModel,
)


from app.ml.performance_evaluation_store import (
    list_ml_performance_evaluations_for_model,
)


from app.ml.performance_monitoring_service import (
    ML_PERFORMANCE_MONITORING_SERVICE_RULE_VERSION,
    MLPerformanceMonitoringObservedDatasetError,
    MLPerformanceMonitoringServiceAuthorityError,
    MLPerformanceMonitoringServiceExecutionError,
    MLPerformanceMonitoringTargetError,
    run_ml_performance_monitoring,
)


from tests.ml.test_ml_monitoring_service_real_handoff_v0_1 import (
    build_ready_preparation_workflow,
    isolated_real_handoff_environment,
)


from tests.ml.test_ml_performance_evaluator_v0_1 import (
    classification_model,
)


from tests.ml.test_ml_performance_monitoring_service_v0_1 import (
    FakePerformanceHandoff,
    patched_handoff,
    patched_trusted_model,
)


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


def build_authority(
):

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


    return (
        session,
        dataset_id,
        dataframe,
        trusted_model,
        estimator,
    )


def fake_handoff(
    *,
    workflow_id: str,
    revision,
    dataset_ids,
    dataset_records,
) -> FakePerformanceHandoff:

    return (
        FakePerformanceHandoff(
            workflow_id=
                workflow_id,

            session_revision=
                revision,

            dataset_ids=
                dataset_ids,

            dataset_records=
                dataset_records,
        )
    )


def assert_no_history(
    *,
    trusted_model: LoadedMLModel,
) -> None:

    artifact = (
        trusted_model.artifact
    )


    history = (
        list_ml_performance_evaluations_for_model(
            model_id=
                artifact.model_id,

            workflow_id=
                artifact.workflow_id,
        )
    )


    assert (
        history
        ==
        []
    )


# ============================================================
# HANDOFF REVISION AUTHORITY
# ============================================================


def test_missing_handoff_revision_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            dataframe,
            trusted_model,
            estimator,
        ) = build_authority()


        handoff = (
            fake_handoff(
                workflow_id=
                    session.workflow_id,

                revision=
                    None,

                dataset_ids=(
                    dataset_id,
                ),

                dataset_records=(
                    {
                        "dataset_id":
                            dataset_id,

                        "dataframe":
                            dataframe.copy(
                                deep=True
                            ),
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
                MLPerformanceMonitoringServiceAuthorityError,

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


        assert_no_history(
            trusted_model=
                trusted_model
        )


def test_boolean_handoff_revision_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            dataframe,
            trusted_model,
            estimator,
        ) = build_authority()


        handoff = (
            fake_handoff(
                workflow_id=
                    session.workflow_id,

                revision=
                    True,

                dataset_ids=(
                    dataset_id,
                ),

                dataset_records=(
                    {
                        "dataset_id":
                            dataset_id,

                        "dataframe":
                            dataframe.copy(
                                deep=True
                            ),
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
                MLPerformanceMonitoringServiceAuthorityError,

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


        assert_no_history(
            trusted_model=
                trusted_model
        )


def test_negative_handoff_revision_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            dataframe,
            trusted_model,
            estimator,
        ) = build_authority()


        handoff = (
            fake_handoff(
                workflow_id=
                    session.workflow_id,

                revision=
                    -1,

                dataset_ids=(
                    dataset_id,
                ),

                dataset_records=(
                    {
                        "dataset_id":
                            dataset_id,

                        "dataframe":
                            dataframe.copy(
                                deep=True
                            ),
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
                MLPerformanceMonitoringServiceAuthorityError,

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


        assert_no_history(
            trusted_model=
                trusted_model
        )


# ============================================================
# HANDOFF WORKFLOW AUTHORITY
# ============================================================


def test_handoff_workflow_mismatch_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            dataframe,
            trusted_model,
            estimator,
        ) = build_authority()


        handoff = (
            fake_handoff(
                workflow_id=
                    "prep:wrong-workflow",

                revision=
                    session.revision,

                dataset_ids=(
                    dataset_id,
                ),

                dataset_records=(
                    {
                        "dataset_id":
                            dataset_id,

                        "dataframe":
                            dataframe.copy(
                                deep=True
                            ),
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
                MLPerformanceMonitoringServiceAuthorityError,

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


        assert_no_history(
            trusted_model=
                trusted_model
        )


# ============================================================
# DUPLICATE DATASET AUTHORITY
# ============================================================


def test_duplicate_handoff_dataset_ids_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            dataframe,
            trusted_model,
            estimator,
        ) = build_authority()


        handoff = (
            fake_handoff(
                workflow_id=
                    session.workflow_id,

                revision=
                    session.revision,

                dataset_ids=(
                    dataset_id,
                    dataset_id,
                ),

                dataset_records=(
                    {
                        "dataset_id":
                            dataset_id,

                        "dataframe":
                            dataframe.copy(
                                deep=True
                            ),
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
                MLPerformanceMonitoringServiceAuthorityError,

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


        assert_no_history(
            trusted_model=
                trusted_model
        )


def test_duplicate_handoff_records_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            dataframe,
            trusted_model,
            estimator,
        ) = build_authority()


        handoff = (
            fake_handoff(
                workflow_id=
                    session.workflow_id,

                revision=
                    session.revision,

                dataset_ids=(
                    dataset_id,
                ),

                dataset_records=(
                    {
                        "dataset_id":
                            dataset_id,

                        "dataframe":
                            dataframe.copy(
                                deep=True
                            ),
                    },
                    {
                        "dataset_id":
                            dataset_id,

                        "dataframe":
                            dataframe.copy(
                                deep=True
                            ),
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
                MLPerformanceMonitoringServiceAuthorityError,

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


        assert_no_history(
            trusted_model=
                trusted_model
        )


def test_handoff_scope_record_mismatch_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            dataframe,
            trusted_model,
            estimator,
        ) = build_authority()


        handoff = (
            fake_handoff(
                workflow_id=
                    session.workflow_id,

                revision=
                    session.revision,

                dataset_ids=(
                    dataset_id,
                ),

                dataset_records=(
                    {
                        "dataset_id":
                            "dataset:different",

                        "dataframe":
                            dataframe.copy(
                                deep=True
                            ),
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
                MLPerformanceMonitoringServiceAuthorityError,

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


        assert_no_history(
            trusted_model=
                trusted_model
        )


# ============================================================
# TRUSTED DATAFRAME AUTHORITY
# ============================================================


def test_non_dataframe_handoff_payload_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
            trusted_model,
            estimator,
        ) = build_authority()


        handoff = (
            fake_handoff(
                workflow_id=
                    session.workflow_id,

                revision=
                    session.revision,

                dataset_ids=(
                    dataset_id,
                ),

                dataset_records=(
                    {
                        "dataset_id":
                            dataset_id,

                        "dataframe": [
                            {
                                "age":
                                    20.0
                            }
                        ],
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
                MLPerformanceMonitoringServiceAuthorityError,

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


        assert_no_history(
            trusted_model=
                trusted_model
        )


# ============================================================
# OBSERVED FEATURE VALIDATION
# ============================================================


def test_non_finite_observed_feature_does_not_persist(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            dataframe,
            trusted_model,
            estimator,
        ) = build_authority()


        invalid = (
            dataframe.copy(
                deep=True
            )
        )


        invalid.loc[
            0,
            "age",
        ] = np.inf


        handoff = (
            fake_handoff(
                workflow_id=
                    session.workflow_id,

                revision=
                    session.revision,

                dataset_ids=(
                    dataset_id,
                ),

                dataset_records=(
                    {
                        "dataset_id":
                            dataset_id,

                        "dataframe":
                            invalid,
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


        assert_no_history(
            trusted_model=
                trusted_model
        )


# ============================================================
# GROUND TRUTH HARDENING
# ============================================================


def test_partially_missing_target_does_not_persist(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            dataframe,
            trusted_model,
            estimator,
        ) = build_authority()


        invalid = (
            dataframe.copy(
                deep=True
            )
        )


        invalid.loc[
            2,
            "target",
        ] = None


        handoff = (
            fake_handoff(
                workflow_id=
                    session.workflow_id,

                revision=
                    session.revision,

                dataset_ids=(
                    dataset_id,
                ),

                dataset_records=(
                    {
                        "dataset_id":
                            dataset_id,

                        "dataframe":
                            invalid,
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


        assert_no_history(
            trusted_model=
                trusted_model
        )


# ============================================================
# PREDICT FAILURE
# ============================================================


class ExplodingEstimator:

    def __init__(
        self,
    ) -> None:

        self.call_count = 0


    def predict(
        self,
        features,
    ):

        self.call_count += 1

        raise RuntimeError(
            "injected predict failure"
        )


def test_predict_failure_does_not_persist(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
            trusted_model,
            _,
        ) = build_authority()


        estimator = (
            ExplodingEstimator()
        )


        exploding_model = (
            LoadedMLModel(
                artifact=
                    trusted_model.artifact,

                estimator=
                    estimator,
            )
        )


        with patched_trusted_model(
            exploding_model
        ):

            expect_error(
                MLPerformanceMonitoringServiceExecutionError,

                lambda:
                    run_ml_performance_monitoring(
                        workflow_id=
                            session.workflow_id,

                        model_id=
                            exploding_model
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


        assert_no_history(
            trusted_model=
                exploding_model
        )


# ============================================================
# WRONG PREDICTION COUNT
# ============================================================


def test_wrong_prediction_count_does_not_persist(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
            _,
            _,
        ) = build_authority()


        (
            trusted_model,
            estimator,
        ) = (
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
                ],
            )
        )


        with patched_trusted_model(
            trusted_model
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


        assert_no_history(
            trusted_model=
                trusted_model
        )


# ============================================================
# VERSION
# ============================================================


def test_rule_version(
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
            "=== DATALENS ML PERFORMANCE MONITORING "
            "SERVICE HARDENING v0.1 ==="
        )
    )

    print()


    tests = [
        (
            "Missing Handoff revision blocked",
            test_missing_handoff_revision_blocked,
        ),
        (
            "Boolean Handoff revision blocked",
            test_boolean_handoff_revision_blocked,
        ),
        (
            "Negative Handoff revision blocked",
            test_negative_handoff_revision_blocked,
        ),
        (
            "Handoff workflow mismatch blocked",
            test_handoff_workflow_mismatch_blocked,
        ),
        (
            "Duplicate Handoff dataset identities blocked",
            test_duplicate_handoff_dataset_ids_blocked,
        ),
        (
            "Duplicate Handoff records blocked",
            test_duplicate_handoff_records_blocked,
        ),
        (
            "Handoff scope / record mismatch blocked",
            test_handoff_scope_record_mismatch_blocked,
        ),
        (
            "Non-DataFrame Handoff payload blocked",
            test_non_dataframe_handoff_payload_blocked,
        ),
        (
            "Non-finite observed feature leaves no history",
            test_non_finite_observed_feature_does_not_persist,
        ),
        (
            "Partially missing target leaves no history",
            test_partially_missing_target_does_not_persist,
        ),
        (
            "predict() failure leaves no history",
            test_predict_failure_does_not_persist,
        ),
        (
            "Wrong prediction count leaves no history",
            test_wrong_prediction_count_does_not_persist,
        ),
        (
            "Performance Monitoring Service rule version",
            test_rule_version,
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
            "PASS - ML Performance Monitoring "
            "Service Hardening v0.1"
        )
    )


if __name__ == "__main__":
    main()
