from __future__ import annotations


import math


from app.ml.baseline import (
    ML_BASELINE_RULE_VERSION,
)


from app.ml.classical_executor import (
    execute_classical_ml,
)


from tests.ml.test_classical_ml_executor_v0_1 import (
    classification_contract,
    classification_dataframe,
    isolated_environment,
    patched_handoff,
    regression_contract,
    regression_dataframe,
    seed_preparation_authority,
)


WORKFLOW_ID = (
    "prep:ml-executor"
)


DATASET_ID = (
    "dataset:validated"
)


# ============================================================
# REGRESSION
# ============================================================


def test_regression_executor_baseline(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,
        )


        dataframe = (
            regression_dataframe()
        )


        with patched_handoff(
            dataframe=
                dataframe,

            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,
        ):
            result = (
                execute_classical_ml(
                    training_contract=
                        regression_contract()
                )
            )


        assert (
            result.baseline.problem_type
            ==
            "regression"
        )


        assert (
            result.baseline.strategy
            ==
            "mean_train_target"
        )


        assert (
            result.baseline.primary_metric
            ==
            "rmse"
        )


        assert (
            result.baseline.train_rows
            ==
            result.train_rows
        )


        assert (
            result.baseline.test_rows
            ==
            result.test_rows
        )


        assert (
            set(
                result.baseline.metrics
            )
            ==
            {
                "mae",
                "rmse",
                "r2",
            }
        )


        assert (
            result.baseline.metrics[
                "rmse"
            ]
            >
            result.metrics[
                "rmse"
            ]
        )


        assert (
            result.baseline_comparison.beats_baseline
            is True
        )


        assert (
            result.baseline_comparison
            .absolute_improvement
            >
            0.0
        )


        assert (
            result.baseline_comparison
            .relative_improvement_pct
            is not None
        )


        assert (
            result.baseline_comparison
            .relative_improvement_pct
            >
            0.0
        )


        # Model Artifact remains the REAL fitted model only.
        assert (
            result.model_artifact.metrics
            ==
            result.metrics
        )


        artifact_payload = (
            result
            .model_artifact
            .model_dump()
        )


        assert (
            "baseline"
            not in
            artifact_payload
        )


        assert (
            "baseline_comparison"
            not in
            artifact_payload
        )


# ============================================================
# CLASSIFICATION
# ============================================================


def test_classification_executor_baseline(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,
        )


        dataframe = (
            classification_dataframe()
        )


        with patched_handoff(
            dataframe=
                dataframe,

            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,
        ):
            result = (
                execute_classical_ml(
                    training_contract=
                        classification_contract()
                )
            )


        assert (
            result.baseline.problem_type
            ==
            "classification"
        )


        assert (
            result.baseline.strategy
            ==
            "majority_train_class"
        )


        assert (
            result.baseline.primary_metric
            ==
            "f1_macro"
        )


        assert (
            set(
                result.baseline.metrics
            )
            ==
            {
                "accuracy",
                "f1_macro",
            }
        )


        assert (
            result.baseline_comparison
            .primary_metric
            ==
            "f1_macro"
        )


        assert (
            result.baseline_comparison.beats_baseline
            is True
        )


        assert (
            result.metrics[
                "f1_macro"
            ]
            >
            result.baseline.metrics[
                "f1_macro"
            ]
        )


# ============================================================
# DETERMINISM
# ============================================================


def test_baseline_is_deterministic(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,
        )


        dataframe = (
            regression_dataframe()
        )


        with patched_handoff(
            dataframe=
                dataframe,

            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,
        ):
            first = (
                execute_classical_ml(
                    training_contract=
                        regression_contract()
                )
            )


            second = (
                execute_classical_ml(
                    training_contract=
                        regression_contract()
                )
            )


        assert (
            first.baseline
            ==
            second.baseline
        )


        assert (
            first.baseline_comparison
            ==
            second.baseline_comparison
        )


        for value in (
            first.baseline.metrics.values()
        ):
            assert math.isfinite(
                float(
                    value
                )
            )


# ============================================================
# MODEL ARTIFACT BOUNDARY
# ============================================================


def test_baseline_is_not_a_model_artifact(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,
        )


        dataframe = (
            regression_dataframe()
        )


        with patched_handoff(
            dataframe=
                dataframe,

            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,
        ):
            result = (
                execute_classical_ml(
                    training_contract=
                        regression_contract()
                )
            )


        assert (
            result.model_artifact
            .training_contract
            .estimator_key
            ==
            "linear_regression"
        )


        assert (
            result.baseline.rule_version
            ==
            ML_BASELINE_RULE_VERSION
        )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML BASELINE EXECUTOR v0.1 ==="
    )

    print()


    test_regression_executor_baseline()

    print(
        "Regression model + same-split baseline: PASS"
    )


    test_classification_executor_baseline()

    print(
        "Classification model + same-split baseline: PASS"
    )


    test_baseline_is_deterministic()

    print(
        "Deterministic baseline evaluation: PASS"
    )


    test_baseline_is_not_a_model_artifact()

    print(
        "Baseline is not persisted as Model Artifact: PASS"
    )


    print()

    print(
        "ML Baseline Executor v0.1: PASS"
    )


if __name__ == "__main__":
    main()
