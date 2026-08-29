from __future__ import annotations


import io


import joblib
import numpy as np


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.model_artifact_store import (
    register_ml_model_artifact,
)


from app.ml.model_loader import (
    load_trusted_ml_model,
)


from app.ml.performance_evaluation_store import (
    list_ml_performance_evaluations_for_model,
)


from app.ml.performance_monitoring_service import (
    run_ml_performance_monitoring,
)


from tests.ml.test_ml_monitoring_service_real_handoff_v0_1 import (
    build_ready_preparation_workflow,
    isolated_real_handoff_environment,
)


# ============================================================
# REAL SERIALIZABLE ESTIMATOR
# ============================================================


class PersistedDeterministicClassifier:
    """
    Minimal deterministic predictor used only to prove the real
    joblib persistence / trusted-loader boundary.

    This is intentionally not a fake loader:
    the estimator itself is serialized into the Model Artifact
    Store and restored by load_trusted_ml_model().
    """

    def predict(
        self,
        features,
    ):

        row_count = len(
            features
        )


        return np.asarray(
            [
                index % 2

                for index
                in range(
                    row_count
                )
            ],
            dtype=int,
        )


# ============================================================
# JOBLIB
# ============================================================


def serialize_estimator(
    estimator,
) -> bytes:

    buffer = io.BytesIO()


    joblib.dump(
        estimator,
        buffer,
    )


    payload = (
        buffer.getvalue()
    )


    assert isinstance(
        payload,
        bytes,
    )


    assert (
        len(
            payload
        )
        >
        0
    )


    return payload


# ============================================================
# MODEL ARTIFACT
# ============================================================


def persist_real_joblib_model(
    *,
    workflow_id: str,
    dataset_id: str,
    preparation_revision: int,
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


    estimator = (
        PersistedDeterministicClassifier()
    )


    model_bytes = (
        serialize_estimator(
            estimator
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
                6,

            test_rows=
                2,

            model_bytes=
                model_bytes,

            preparation_session_revision=
                preparation_revision,

            created_at_utc=
                "2026-08-29T21:00:00+00:00",
        )
    )


    return artifact


# ============================================================
# TRUSTED LOADER PROOF
# ============================================================


def test_real_joblib_artifact_loads_through_trusted_loader(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        artifact = (
            persist_real_joblib_model(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                preparation_revision=
                    session.revision,
            )
        )


        loaded = (
            load_trusted_ml_model(
                workflow_id=
                    session.workflow_id,

                model_id=
                    artifact.model_id,
            )
        )


        assert (
            loaded.artifact
            ==
            artifact
        )


        assert isinstance(
            loaded.estimator,
            PersistedDeterministicClassifier,
        )


# ============================================================
# COMPLETE REAL SERVICE PATH
# ============================================================


def test_performance_service_uses_real_trusted_model_end_to_end(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            dataframe,
        ) = (
            build_ready_preparation_workflow()
        )


        artifact = (
            persist_real_joblib_model(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                preparation_revision=
                    session.revision,
            )
        )


        # No loader patch.
        # No Handoff patch.
        # No evaluator patch.
        # No store patch.
        result = (
            run_ml_performance_monitoring(
                workflow_id=
                    session.workflow_id,

                model_id=
                    artifact.model_id,

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
            artifact.model_id
        )


        assert (
            result.reference_dataset_id
            ==
            dataset_id
        )


        assert (
            result.observed_dataset_id
            ==
            dataset_id
        )


        assert (
            result
            .preparation_session_revision
            ==
            session.revision
        )


        assert (
            result
            .observed_preparation_session_revision
            ==
            session.revision
        )


        assert (
            result.problem_type
            ==
            "classification"
        )


        assert (
            result.target_column
            ==
            "target"
        )


        assert (
            result.observed_row_count
            ==
            len(
                dataframe
            )
        )


        assert (
            result.primary_metric
            ==
            "f1_macro"
        )


        assert (
            result
            .primary_metric_degradation_amount
            ==
            0.0
        )


        assert (
            result.performance_status
            ==
            "ok"
        )


        observed_metrics = {
            item.metric_name:
                item.observed_value

            for item
            in result.metric_results
        }


        assert observed_metrics == {
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
        }


        history = (
            list_ml_performance_evaluations_for_model(
                model_id=
                    artifact.model_id,

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
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML PERFORMANCE MONITORING "
            "REAL MODEL v0.1 ==="
        )
    )

    print()


    tests = [
        (
            "Real joblib Artifact loads through trusted loader",
            test_real_joblib_artifact_loads_through_trusted_loader,
        ),
        (
            "Performance Service uses real trusted model end-to-end",
            test_performance_service_uses_real_trusted_model_end_to_end,
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
            "Real Model v0.1"
        )
    )


if __name__ == "__main__":
    main()
