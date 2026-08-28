from __future__ import annotations


import math


from contextlib import (
    contextmanager,
)


from types import (
    SimpleNamespace,
)


from app.ml.contracts import (
    MLSplitContract,
    MLTrainingContract,
)


from app.ml.model_comparison_contracts import (
    MLModelComparisonContract,
)


from app.ml.model_comparison_executor import (
    MLModelComparisonCandidateError,
    MLModelComparisonSnapshotError,
    ML_MODEL_COMPARISON_EXECUTOR_RULE_VERSION,
    _ranking_key,
    execute_ml_model_comparison,
)


from app.ml.model_loader import (
    load_trusted_ml_model,
)


from tests.ml.test_classical_ml_executor_v0_1 import (
    classification_dataframe,
    isolated_environment,
    patched_handoff,
    regression_dataframe,
    seed_preparation_authority,
)


# ============================================================
# CONSTANTS
# ============================================================


WORKFLOW_ID = (
    "prep:ml-comparison"
)


DATASET_ID = (
    "dataset:validated"
)


# ============================================================
# COMPARISON READINESS PATCH
# ============================================================
#
# Classical ML executor unit tests deliberately use a minimal
# Preparation persistence fixture plus a patched handoff.
#
# Model Comparison additionally pins the server-owned
# Preparation revision before / between candidate executions.
#
# These unit tests therefore provide the corresponding stable
# readiness read model.
#
# The real E2E Golden Path does NOT use this patch.
# ============================================================


@contextmanager
def patched_comparison_readiness(
    *,
    session_revision: int = 0,
):
    import app.ml.model_comparison_executor as comparison_module


    original = (
        comparison_module
        .require_analysis_readiness
    )


    def fake_require_analysis_readiness(
        *,
        workflow_id: str,
    ):
        return (
            SimpleNamespace(
                workflow_id=
                    workflow_id,

                session_revision=
                    session_revision,

                requested_analysis_dataset_ids=[
                    DATASET_ID
                ],
            )
        )


    comparison_module.require_analysis_readiness = (
        fake_require_analysis_readiness
    )


    try:
        yield

    finally:
        comparison_module.require_analysis_readiness = (
            original
        )


# ============================================================
# CONTRACT HELPERS
# ============================================================


def regression_candidate(
    *,
    estimator_key: str,
    estimator_hyperparameters=None,
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

            problem_type=
                "regression",

            target_column=
                "revenue",

            feature_columns=[
                "age",
                "tenure",
            ],

            estimator_key=
                estimator_key,

            estimator_hyperparameters=
                estimator_hyperparameters,

            split=(
                MLSplitContract(
                    test_size=
                        0.20,

                    random_seed=
                        42,

                    shuffle=
                        True,

                    stratify=
                        False,
                )
            ),
        )
    )


def classification_candidate(
    *,
    estimator_key: str,
    estimator_hyperparameters=None,
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

            problem_type=
                "classification",

            target_column=
                "churned",

            feature_columns=[
                "signal",
                "aux",
            ],

            estimator_key=
                estimator_key,

            estimator_hyperparameters=
                estimator_hyperparameters,

            split=(
                MLSplitContract(
                    test_size=
                        0.25,

                    random_seed=
                        42,

                    shuffle=
                        True,

                    stratify=
                        True,
                )
            ),
        )
    )


# ============================================================
# REGRESSION COMPARISON
# ============================================================


def test_real_regression_model_comparison(
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


        contract = (
            MLModelComparisonContract(
                candidates=[
                    regression_candidate(
                        estimator_key=
                            "random_forest_regressor",

                        estimator_hyperparameters={
                            "kind":
                                "random_forest_regressor",

                            "n_estimators":
                                64,

                            "max_depth":
                                8,
                        },
                    ),

                    regression_candidate(
                        estimator_key=
                            "linear_regression"
                    ),

                    regression_candidate(
                        estimator_key=
                            "ridge_regression",

                        estimator_hyperparameters={
                            "kind":
                                "ridge_regression",

                            "alpha":
                                2.0,
                        },
                    ),
                ]
            )
        )


        with patched_comparison_readiness():

            with patched_handoff(
                dataframe=
                    dataframe,

                workflow_id=
                    WORKFLOW_ID,

                dataset_id=
                    DATASET_ID,
            ):
                result = (
                    execute_ml_model_comparison(
                        comparison_contract=
                            contract
                    )
                )


        assert (
            result.workflow_id
            ==
            WORKFLOW_ID
        )


        assert (
            result.dataset_id
            ==
            DATASET_ID
        )


        assert (
            result.problem_type
            ==
            "regression"
        )


        assert (
            result.primary_metric
            ==
            "rmse"
        )


        assert (
            result.ranking_policy
            ==
            "regression_rmse_v0.1"
        )


        assert (
            len(
                result.candidates
            )
            ==
            3
        )


        assert (
            [
                candidate.rank

                for candidate
                in result.candidates
            ]
            ==
            [
                1,
                2,
                3,
            ]
        )


        assert (
            {
                candidate.estimator_key

                for candidate
                in result.candidates
            }
            ==
            {
                "linear_regression",
                "ridge_regression",
                "random_forest_regressor",
            }
        )


        expected_winner = min(
            result.candidates,

            key=lambda candidate: (
                _ranking_key(
                    problem_type=
                        "regression",

                    estimator_key=
                        candidate.estimator_key,

                    metrics=
                        candidate.metrics,
                )
            ),
        )


        assert (
            result.selected_estimator_key
            ==
            expected_winner.estimator_key
        )


        assert (
            result.selected_model_id
            ==
            expected_winner
            .model_artifact
            .model_id
        )


        train_rows = {
            candidate.train_rows

            for candidate
            in result.candidates
        }


        test_rows = {
            candidate.test_rows

            for candidate
            in result.candidates
        }


        assert (
            len(
                train_rows
            )
            ==
            1
        )


        assert (
            len(
                test_rows
            )
            ==
            1
        )


        for candidate in (
            result.candidates
        ):

            assert (
                candidate.primary_metric
                ==
                "rmse"
            )


            assert (
                math.isfinite(
                    candidate
                    .primary_metric_value
                )
            )


            assert (
                candidate
                .model_artifact
                .training_contract
                .estimator_key
                ==
                candidate.estimator_key
            )


            loaded = (
                load_trusted_ml_model(
                    workflow_id=
                        WORKFLOW_ID,

                    model_id=
                        candidate
                        .model_artifact
                        .model_id,
                )
            )


            assert (
                loaded
                .artifact
                .model_id
                ==
                candidate
                .model_artifact
                .model_id
            )


# ============================================================
# CLASSIFICATION COMPARISON
# ============================================================


def test_real_classification_model_comparison(
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


        contract = (
            MLModelComparisonContract(
                candidates=[
                    classification_candidate(
                        estimator_key=
                            "random_forest_classifier",

                        estimator_hyperparameters={
                            "kind":
                                "random_forest_classifier",

                            "n_estimators":
                                80,

                            "max_depth":
                                6,
                        },
                    ),

                    classification_candidate(
                        estimator_key=
                            "logistic_regression"
                    ),
                ]
            )
        )


        with patched_comparison_readiness():

            with patched_handoff(
                dataframe=
                    dataframe,

                workflow_id=
                    WORKFLOW_ID,

                dataset_id=
                    DATASET_ID,
            ):
                result = (
                    execute_ml_model_comparison(
                        comparison_contract=
                            contract
                    )
                )


        assert (
            result.problem_type
            ==
            "classification"
        )


        assert (
            result.primary_metric
            ==
            "f1_macro"
        )


        assert (
            result.ranking_policy
            ==
            "classification_f1_macro_v0.1"
        )


        assert (
            len(
                result.candidates
            )
            ==
            2
        )


        expected_winner = min(
            result.candidates,

            key=lambda candidate: (
                _ranking_key(
                    problem_type=
                        "classification",

                    estimator_key=
                        candidate.estimator_key,

                    metrics=
                        candidate.metrics,
                )
            ),
        )


        assert (
            result.selected_estimator_key
            ==
            expected_winner.estimator_key
        )


        assert (
            result.selected_model_id
            ==
            expected_winner
            .model_artifact
            .model_id
        )


        for candidate in (
            result.candidates
        ):

            assert (
                0.0
                <=
                candidate.metrics[
                    "f1_macro"
                ]
                <=
                1.0
            )


            assert (
                0.0
                <=
                candidate.metrics[
                    "accuracy"
                ]
                <=
                1.0
            )


# ============================================================
# REGRESSION TIE-BREAKERS
# ============================================================


def test_regression_ranking_policy(
) -> None:

    better_rmse = (
        _ranking_key(
            problem_type=
                "regression",

            estimator_key=
                "random_forest_regressor",

            metrics={
                "rmse":
                    9.0,

                "mae":
                    8.0,

                "r2":
                    0.70,
            },
        )
    )


    worse_rmse = (
        _ranking_key(
            problem_type=
                "regression",

            estimator_key=
                "linear_regression",

            metrics={
                "rmse":
                    10.0,

                "mae":
                    1.0,

                "r2":
                    0.99,
            },
        )
    )


    assert (
        better_rmse
        <
        worse_rmse
    )


    better_mae = (
        _ranking_key(
            problem_type=
                "regression",

            estimator_key=
                "ridge_regression",

            metrics={
                "rmse":
                    10.0,

                "mae":
                    4.0,

                "r2":
                    0.20,
            },
        )
    )


    worse_mae = (
        _ranking_key(
            problem_type=
                "regression",

            estimator_key=
                "linear_regression",

            metrics={
                "rmse":
                    10.0,

                "mae":
                    5.0,

                "r2":
                    0.99,
            },
        )
    )


    assert (
        better_mae
        <
        worse_mae
    )


    better_r2 = (
        _ranking_key(
            problem_type=
                "regression",

            estimator_key=
                "ridge_regression",

            metrics={
                "rmse":
                    10.0,

                "mae":
                    5.0,

                "r2":
                    0.90,
            },
        )
    )


    worse_r2 = (
        _ranking_key(
            problem_type=
                "regression",

            estimator_key=
                "linear_regression",

            metrics={
                "rmse":
                    10.0,

                "mae":
                    5.0,

                "r2":
                    0.80,
            },
        )
    )


    assert (
        better_r2
        <
        worse_r2
    )


# ============================================================
# CLASSIFICATION TIE-BREAKERS
# ============================================================


def test_classification_ranking_policy(
) -> None:

    better_f1 = (
        _ranking_key(
            problem_type=
                "classification",

            estimator_key=
                "random_forest_classifier",

            metrics={
                "f1_macro":
                    0.90,

                "accuracy":
                    0.70,
            },
        )
    )


    worse_f1 = (
        _ranking_key(
            problem_type=
                "classification",

            estimator_key=
                "logistic_regression",

            metrics={
                "f1_macro":
                    0.80,

                "accuracy":
                    0.99,
            },
        )
    )


    assert (
        better_f1
        <
        worse_f1
    )


    better_accuracy = (
        _ranking_key(
            problem_type=
                "classification",

            estimator_key=
                "random_forest_classifier",

            metrics={
                "f1_macro":
                    0.80,

                "accuracy":
                    0.90,
            },
        )
    )


    worse_accuracy = (
        _ranking_key(
            problem_type=
                "classification",

            estimator_key=
                "logistic_regression",

            metrics={
                "f1_macro":
                    0.80,

                "accuracy":
                    0.70,
            },
        )
    )


    assert (
        better_accuracy
        <
        worse_accuracy
    )


# ============================================================
# LEXICAL FINAL TIE-BREAKER
# ============================================================


def test_final_tie_breaker_is_estimator_key(
) -> None:

    linear = (
        _ranking_key(
            problem_type=
                "regression",

            estimator_key=
                "linear_regression",

            metrics={
                "rmse":
                    10.0,

                "mae":
                    5.0,

                "r2":
                    0.80,
            },
        )
    )


    ridge = (
        _ranking_key(
            problem_type=
                "regression",

            estimator_key=
                "ridge_regression",

            metrics={
                "rmse":
                    10.0,

                "mae":
                    5.0,

                "r2":
                    0.80,
            },
        )
    )


    assert (
        linear
        <
        ridge
    )


# ============================================================
# NON-FINITE METRIC
# ============================================================


def test_non_finite_metric_is_blocked(
) -> None:

    try:
        _ranking_key(
            problem_type=
                "regression",

            estimator_key=
                "linear_regression",

            metrics={
                "rmse":
                    float(
                        "nan"
                    ),

                "mae":
                    1.0,

                "r2":
                    0.5,
            },
        )

    except Exception:
        return


    raise AssertionError(
        (
            "Non-finite ranking metrics "
            "must be blocked."
        )
    )


# ============================================================
# CANDIDATE FAILURE
# ============================================================


def test_candidate_execution_failure_is_fail_closed(
) -> None:

    import app.ml.model_comparison_executor as comparison_module


    original_execute = (
        comparison_module
        .execute_classical_ml
    )


    original_require = (
        comparison_module
        .require_analysis_readiness
    )


    call_count = 0


    def failing_execute(
        *,
        training_contract,
    ):
        nonlocal call_count


        call_count += 1


        raise (
            ClassicalComparisonTestError(
                "synthetic candidate failure"
            )
        )


    class ClassicalComparisonTestError(
        comparison_module
        .ClassicalMLExecutorError
    ):
        pass


    def stable_readiness(
        *,
        workflow_id: str,
    ):
        return (
            SimpleNamespace(
                workflow_id=
                    workflow_id,

                session_revision=
                    0,

                requested_analysis_dataset_ids=[
                    DATASET_ID
                ],
            )
        )


    comparison_module.require_analysis_readiness = (
        stable_readiness
    )


    comparison_module.execute_classical_ml = (
        failing_execute
    )


    try:
        contract = (
            MLModelComparisonContract(
                candidates=[
                    regression_candidate(
                        estimator_key=
                            "linear_regression"
                    ),

                    regression_candidate(
                        estimator_key=
                            "ridge_regression"
                    ),
                ]
            )
        )


        try:
            comparison_module.execute_ml_model_comparison(
                comparison_contract=
                    contract
            )

        except MLModelComparisonCandidateError:
            assert (
                call_count
                ==
                1
            )

            return


        raise AssertionError(
            (
                "Candidate execution failure "
                "must fail closed."
            )
        )


    finally:
        comparison_module.require_analysis_readiness = (
            original_require
        )


        comparison_module.execute_classical_ml = (
            original_execute
        )


# ============================================================
# PREPARATION SNAPSHOT RACE
# ============================================================


def test_preparation_revision_change_is_fail_closed(
) -> None:

    import app.ml.model_comparison_executor as comparison_module


    original_require = (
        comparison_module
        .require_analysis_readiness
    )


    original_execute = (
        comparison_module
        .execute_classical_ml
    )


    readiness_calls = 0
    execution_calls = 0


    def changing_readiness(
        *,
        workflow_id: str,
    ):
        nonlocal readiness_calls


        readiness_calls += 1


        revision = (
            10

            if readiness_calls
            ==
            1

            else 11
        )


        return (
            SimpleNamespace(
                workflow_id=
                    workflow_id,

                session_revision=
                    revision,

                requested_analysis_dataset_ids=[
                    DATASET_ID
                ],
            )
        )


    def should_not_execute(
        *,
        training_contract,
    ):
        nonlocal execution_calls


        execution_calls += 1


        raise AssertionError(
            (
                "Candidate execution should not start "
                "after Preparation revision changed."
            )
        )


    comparison_module.require_analysis_readiness = (
        changing_readiness
    )


    comparison_module.execute_classical_ml = (
        should_not_execute
    )


    try:
        contract = (
            MLModelComparisonContract(
                candidates=[
                    regression_candidate(
                        estimator_key=
                            "linear_regression"
                    ),

                    regression_candidate(
                        estimator_key=
                            "ridge_regression"
                    ),
                ]
            )
        )


        try:
            comparison_module.execute_ml_model_comparison(
                comparison_contract=
                    contract
            )

        except MLModelComparisonSnapshotError:

            assert (
                readiness_calls
                ==
                2
            )


            assert (
                execution_calls
                ==
                0
            )


            return


        raise AssertionError(
            (
                "Preparation revision change must "
                "fail closed before candidate training."
            )
        )


    finally:
        comparison_module.require_analysis_readiness = (
            original_require
        )


        comparison_module.execute_classical_ml = (
            original_execute
        )


# ============================================================
# RULE VERSION
# ============================================================


def test_model_comparison_executor_rule_version(
) -> None:

    assert (
        ML_MODEL_COMPARISON_EXECUTOR_RULE_VERSION
        ==
        "ml_model_comparison_executor_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML MODEL "
            "COMPARISON EXECUTOR v0.1 ==="
        )
    )

    print()


    test_real_regression_model_comparison()

    print(
        (
            "Regression fixed-candidate "
            "comparison + trusted reload: PASS"
        )
    )


    test_real_classification_model_comparison()

    print(
        (
            "Classification fixed-candidate "
            "comparison: PASS"
        )
    )


    test_regression_ranking_policy()

    print(
        "Regression deterministic ranking policy: PASS"
    )


    test_classification_ranking_policy()

    print(
        "Classification deterministic ranking policy: PASS"
    )


    test_final_tie_breaker_is_estimator_key()

    print(
        "Lexical final tie-breaker: PASS"
    )


    test_non_finite_metric_is_blocked()

    print(
        "Non-finite comparison metric is blocked: PASS"
    )


    test_candidate_execution_failure_is_fail_closed()

    print(
        "Candidate execution failure is fail-closed: PASS"
    )


    test_preparation_revision_change_is_fail_closed()

    print(
        "Preparation revision race is fail-closed: PASS"
    )


    test_model_comparison_executor_rule_version()

    print(
        "ML Model Comparison Executor rule version: PASS"
    )


    print()

    print(
        "ML Model Comparison Executor v0.1: PASS"
    )


if __name__ == "__main__":
    main()