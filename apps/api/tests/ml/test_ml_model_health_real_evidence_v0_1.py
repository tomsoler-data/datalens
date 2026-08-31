from __future__ import annotations


import numpy as np


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.drift_evaluation_store import (
    list_ml_drift_evaluations_for_model,
)


from app.ml.model_artifact_store import (
    register_ml_model_artifact,
)


from app.ml.model_health_service import (
    get_ml_model_health_summary,
)


from app.ml.model_loader import (
    LoadedMLModel,
)


from app.ml.monitoring_profile_builder import (
    build_ml_monitoring_profile,
)


from app.ml.monitoring_profile_store import (
    register_ml_monitoring_profile,
)


from app.ml.monitoring_service import (
    run_ml_monitoring,
)


from app.ml.performance_evaluation_store import (
    list_ml_performance_evaluations_for_model,
    register_ml_performance_evaluation,
)


from app.ml.performance_evaluator import (
    evaluate_ml_performance,
)


from app.preparation.preparation_session import (
    record_validation_stage_signal,
)


from tests.ml.test_ml_monitoring_service_real_handoff_v0_1 import (
    build_ready_preparation_workflow,
    isolated_real_handoff_environment,
)


# ============================================================
# ESTIMATOR
# ============================================================


class DeterministicPerfectClassifier:

    def __init__(
        self,
    ) -> None:

        self.call_count = 0


    def predict(
        self,
        features,
    ):

        self.call_count += 1


        return np.asarray(
            [
                index % 2

                for index
                in range(
                    len(
                        features
                    )
                )
            ],
            dtype=int,
        )


# ============================================================
# SHARED MODEL AUTHORITY
# ============================================================


def persist_shared_model_and_profile(
    *,
    workflow_id: str,
    dataset_id: str,
    preparation_revision: int,
    dataframe,
):

    contract = (
        MLTrainingContract(
            workflow_id=
                workflow_id,

            dataset_id=
                dataset_id,

            problem_type=
                "classification",

            target_column=
                "target",

            feature_columns=[
                "age",
                "segment",
            ],

            categorical_feature_columns=[
                "segment"
            ],

            estimator_key=
                "logistic_regression",
        )
    )


    x_train = (
        dataframe[
            [
                "age",
                "segment",
            ]
        ]
        .copy(
            deep=True
        )
    )


    artifact = (
        register_ml_model_artifact(
            training_contract=
                contract,

            metrics={
                "accuracy":
                    1.0,

                "f1_macro":
                    1.0,

                "precision_macro":
                    1.0,

                "recall_macro":
                    1.0,

                "balanced_accuracy":
                    1.0,
            },

            train_rows=
                len(
                    x_train
                ),

            test_rows=
                2,

            model_bytes=
                b"MODEL-HEALTH-INTEGRATION",

            preparation_session_revision=
                preparation_revision,

            created_at_utc=
                "2026-08-29T21:00:00+00:00",
        )
    )


    profile = (
        build_ml_monitoring_profile(
            x_train=
                x_train,

            model_artifact=
                artifact,
        )
    )


    persisted_profile = (
        register_ml_monitoring_profile(
            profile=
                profile
        )
    )


    return (
        artifact,
        persisted_profile,
    )


# ============================================================
# REAL PERSISTED EVIDENCE
# ============================================================


def persist_real_drift_and_performance(
    *,
    session,
    dataset_id: str,
    dataframe,
):

    (
        artifact,
        profile,
    ) = (
        persist_shared_model_and_profile(
            workflow_id=
                session.workflow_id,

            dataset_id=
                dataset_id,

            preparation_revision=
                session.revision,

            dataframe=
                dataframe,
        )
    )


    # ========================================================
    # REAL DRIFT SERVICE
    #
    # Reads the actual validated Preparation Handoff and
    # persists one Drift Evaluation through the production
    # Monitoring Service.
    # ========================================================


    drift = (
        run_ml_monitoring(
            workflow_id=
                session.workflow_id,

            model_id=
                artifact.model_id,

            observed_dataset_id=
                dataset_id,
        )
    )


    # ========================================================
    # REAL PERFORMANCE EVALUATOR + STORE
    #
    # The same persisted Model Artifact is wrapped as a trusted
    # in-memory model for this integration test.
    #
    # No second artifact is created.
    # ========================================================


    estimator = (
        DeterministicPerfectClassifier()
    )


    trusted_model = (
        LoadedMLModel(
            artifact=
                artifact,

            estimator=
                estimator,
        )
    )


    performance = (
        evaluate_ml_performance(
            observed_dataframe=
                dataframe,

            observed_dataset_id=
                dataset_id,

            observed_preparation_session_revision=
                session.revision,

            trusted_model=
                trusted_model,
        )
    )


    persisted_performance = (
        register_ml_performance_evaluation(
            evaluation=
                performance
        )
    )


    assert (
        estimator.call_count
        ==
        1
    )


    return (
        artifact,
        profile,
        drift,
        persisted_performance,
    )


# ============================================================
# ALIGNED REAL EVIDENCE
# ============================================================


def test_real_persisted_drift_and_performance_produce_healthy_summary(
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
            artifact,
            profile,
            drift,
            performance,
        ) = (
            persist_real_drift_and_performance(
                session=
                    session,

                dataset_id=
                    dataset_id,

                dataframe=
                    dataframe,
            )
        )


        # ----------------------------------------------------
        # Both branches must have produced durable evidence.
        # ----------------------------------------------------


        drift_history = (
            list_ml_drift_evaluations_for_model(
                model_id=
                    artifact.model_id,

                workflow_id=
                    session.workflow_id,
            )
        )


        performance_history = (
            list_ml_performance_evaluations_for_model(
                model_id=
                    artifact.model_id,

                workflow_id=
                    session.workflow_id,
            )
        )


        assert drift_history == [
            drift
        ]


        assert performance_history == [
            performance
        ]


        # ----------------------------------------------------
        # Same server-owned observed snapshot.
        # ----------------------------------------------------


        assert (
            drift.observed_dataset_id
            ==
            performance.observed_dataset_id
            ==
            dataset_id
        )


        assert (
            drift
            .observed_preparation_session_revision
            ==
            performance
            .observed_preparation_session_revision
            ==
            session.revision
        )


        assert (
            drift.observed_row_count
            ==
            performance.observed_row_count
            ==
            len(
                dataframe
            )
        )


        assert (
            drift.model_id
            ==
            performance.model_id
            ==
            artifact.model_id
        )


        assert (
            drift.profile_id
            ==
            profile.profile_id
        )


        assert (
            drift.overall_status
            ==
            "ok"
        )


        assert (
            performance.performance_status
            ==
            "ok"
        )


        # ----------------------------------------------------
        # Model Health reads persisted evidence only.
        # ----------------------------------------------------


        health = (
            get_ml_model_health_summary(
                workflow_id=
                    session.workflow_id,

                model_id=
                    artifact.model_id,
            )
        )


        assert (
            health.health_status
            ==
            "healthy"
        )


        assert (
            health.health_reason
            ==
            "aligned_evidence_ok"
        )


        assert (
            health.evidence_alignment
            ==
            "aligned"
        )


        assert (
            health.joint_interpretation_allowed
            is True
        )


        assert (
            health.drift_evaluation_id
            ==
            drift.evaluation_id
        )


        assert (
            health.performance_evaluation_id
            ==
            performance
            .performance_evaluation_id
        )


        assert (
            health.drift_status
            ==
            "ok"
        )


        assert (
            health.performance_status
            ==
            "ok"
        )


        # ----------------------------------------------------
        # Privacy-minimal derived surface.
        # ----------------------------------------------------


        serialized = str(
            health.model_dump(
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
            "predictions"
            not in
            serialized
        )


        assert (
            "target"
            not in
            serialized
        )


# ============================================================
# HISTORY REMAINS VALID AFTER PREPARATION ADVANCES
# ============================================================


def test_real_health_summary_remains_readable_after_workflow_leaves_ready(
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
            artifact,
            _,
            drift,
            performance,
        ) = (
            persist_real_drift_and_performance(
                session=
                    session,

                dataset_id=
                    dataset_id,

                dataframe=
                    dataframe,
            )
        )


        healthy_before = (
            get_ml_model_health_summary(
                workflow_id=
                    session.workflow_id,

                model_id=
                    artifact.model_id,
            )
        )


        assert (
            healthy_before.health_status
            ==
            "healthy"
        )


        # ----------------------------------------------------
        # Preparation advances and is no longer READY.
        #
        # The monitoring evidence remains historical truth about
        # the earlier server-owned observed snapshot.
        # ----------------------------------------------------


        changed = (
            record_validation_stage_signal(
                workflow_id=
                    session.workflow_id,

                completed=
                    True,

                passed=
                    False,

                dataset_ids=[
                    dataset_id
                ],

                evidence_refs=[
                    "test:model-health-history"
                ],

                blocking_reasons=[
                    "test:workflow-no-longer-ready"
                ],

                expected_revision=
                    session.revision,
            )
        )


        assert (
            changed.revision
            >
            session.revision
        )


        assert (
            changed
            .snapshot
            .ready_for_analysis
            is False
        )


        # ----------------------------------------------------
        # No new Drift or Performance evaluation is executed.
        #
        # Model Health must still resolve from the persisted
        # historical evidence.
        # ----------------------------------------------------


        health_after = (
            get_ml_model_health_summary(
                workflow_id=
                    session.workflow_id,

                model_id=
                    artifact.model_id,
            )
        )


        assert (
            health_after
            ==
            healthy_before
        )


        assert (
            health_after.drift_evaluation_id
            ==
            drift.evaluation_id
        )


        assert (
            health_after.performance_evaluation_id
            ==
            performance
            .performance_evaluation_id
        )


        assert (
            health_after
            .drift_observed_preparation_session_revision
            ==
            session.revision
        )


        assert (
            health_after
            .performance_observed_preparation_session_revision
            ==
            session.revision
        )


        assert (
            health_after.health_status
            ==
            "healthy"
        )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML MODEL HEALTH "
            "REAL EVIDENCE v0.1 ==="
        )
    )

    print()


    tests = [
        (
            (
                "Real persisted Drift + Performance "
                "produce healthy summary"
            ),
            (
                test_real_persisted_drift_and_performance_produce_healthy_summary
            ),
        ),
        (
            (
                "Health remains readable after "
                "Preparation leaves READY"
            ),
            (
                test_real_health_summary_remains_readable_after_workflow_leaves_ready
            ),
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
            "PASS - ML Model Health "
            "Real Evidence v0.1"
        )
    )


if __name__ == "__main__":
    main()
