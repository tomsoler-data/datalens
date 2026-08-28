from __future__ import annotations


from contextlib import (
    contextmanager,
)


import math


import pandas as pd


from app.ml.classical_executor import (
    _build_estimator as build_real_estimator,
    _split_dataset as split_real_dataset,
    _validate_and_extract_xy,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


from app.ml.hyperparameter_tuning import (
    MLHyperparameterSearchContract,
    expected_hyperparameter_metric_names,
    server_owned_hyperparameter_candidates,
)


import app.ml.hyperparameter_tuning_executor as tuning_executor


from app.ml.hyperparameter_tuning_executor import (
    ML_HYPERPARAMETER_TUNING_EXECUTOR_RULE_VERSION,
    MLHyperparameterTuningInputError,
    execute_ml_hyperparameter_tuning,
)


# ============================================================
# DATA
# ============================================================


def regression_dataframe(
    rows: int = 50,
) -> pd.DataFrame:

    records = []


    for index in range(
        rows
    ):

        x1 = float(
            index
        )


        x2 = float(
            (
                index
                %
                7
            )
            *
            1.5
        )


        target = (
            25.0
            +
            (
                4.0
                *
                x1
            )
            -
            (
                2.5
                *
                x2
            )
            +
            (
                0.03
                *
                x1
                *
                x1
            )
        )


        records.append(
            {
                "x1":
                    x1,

                "x2":
                    x2,

                "target":
                    target,
            }
        )


    return (
        pd.DataFrame(
            records
        )
    )


def classification_dataframe(
) -> pd.DataFrame:

    records = []


    for index in range(
        60
    ):

        if (
            index
            <
            30
        ):
            target = "A"
            x1 = float(
                index
            )
            x2 = float(
                index
                %
                5
            )

        else:
            target = "B"
            x1 = float(
                index
                +
                20
            )
            x2 = float(
                (
                    index
                    %
                    5
                )
                +
                5
            )


        records.append(
            {
                "x1":
                    x1,

                "x2":
                    x2,

                "target":
                    target,
            }
        )


    return (
        pd.DataFrame(
            records
        )
    )


# ============================================================
# CONTRACTS
# ============================================================


def regression_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:tuning-regression",

            dataset_id=
                "dataset:tuning-regression",

            problem_type=
                "regression",

            target_column=
                "target",

            feature_columns=[
                "x1",
                "x2",
            ],

            estimator_key=
                "ridge_regression",

            split={
                "strategy":
                    "holdout",

                "test_size":
                    0.20,

                "random_seed":
                    42,

                "shuffle":
                    True,

                "stratify":
                    False,
            },
        )
    )


def classification_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:tuning-classification",

            dataset_id=
                "dataset:tuning-classification",

            problem_type=
                "classification",

            target_column=
                "target",

            feature_columns=[
                "x1",
                "x2",
            ],

            estimator_key=
                "logistic_regression",

            split={
                "strategy":
                    "holdout",

                "test_size":
                    0.20,

                "random_seed":
                    42,

                "shuffle":
                    True,

                "stratify":
                    True,
            },
        )
    )


# ============================================================
# SERVER-OWNED DATAFRAME PATCH
# ============================================================


@contextmanager
def authorized_dataframe(
    dataframe: pd.DataFrame,
    *,
    revision: int = 7,
):

    original = (
        tuning_executor
        ._load_authorized_dataframe
    )


    def fake_load_authorized_dataframe(
        *,
        contract,
    ):
        return (
            dataframe.copy(
                deep=True
            ),
            revision,
        )


    tuning_executor._load_authorized_dataframe = (
        fake_load_authorized_dataframe
    )


    try:
        yield

    finally:
        tuning_executor._load_authorized_dataframe = (
            original
        )


# ============================================================
# RECORDING ESTIMATOR
# ============================================================


class RecordingEstimator:
    def __init__(
        self,
        *,
        inner,
        fit_index_sets,
        prediction_index_sets,
    ):

        self.inner = inner

        self.fit_index_sets = (
            fit_index_sets
        )

        self.prediction_index_sets = (
            prediction_index_sets
        )


    def fit(
        self,
        x,
        y,
    ):

        self.fit_index_sets.append(
            set(
                x.index.tolist()
            )
        )


        self.inner.fit(
            x,
            y,
        )


        return self


    def predict(
        self,
        x,
    ):

        self.prediction_index_sets.append(
            set(
                x.index.tolist()
            )
        )


        return (
            self.inner.predict(
                x
            )
        )


# ============================================================
# HOLDOUT BOUNDARY
# ============================================================


def test_regression_tunes_only_inside_outer_train(
) -> None:

    dataframe = (
        regression_dataframe()
    )


    contract = (
        regression_contract()
    )


    search = (
        MLHyperparameterSearchContract(
            folds=
                5,

            shuffle=
                True,

            random_seed=
                73,
        )
    )


    (
        x,
        y,
    ) = (
        _validate_and_extract_xy(
            dataframe=
                dataframe,

            contract=
                contract,
        )
    )


    (
        expected_outer_train,
        expected_holdout_test,
        _,
        _,
    ) = (
        split_real_dataset(
            x=
                x,

            y=
                y,

            contract=
                contract,
        )
    )


    outer_train_indexes = set(
        expected_outer_train.index.tolist()
    )


    holdout_test_indexes = set(
        expected_holdout_test.index.tolist()
    )


    assert (
        outer_train_indexes
        .isdisjoint(
            holdout_test_indexes
        )
    )


    fit_index_sets = []
    prediction_index_sets = []


    original_builder = (
        tuning_executor
        ._build_estimator
    )


    def recording_builder(
        *,
        contract,
    ):

        return (
            RecordingEstimator(
                inner=(
                    build_real_estimator(
                        contract=
                            contract
                    )
                ),

                fit_index_sets=
                    fit_index_sets,

                prediction_index_sets=
                    prediction_index_sets,
            )
        )


    tuning_executor._build_estimator = (
        recording_builder
    )


    try:
        with authorized_dataframe(
            dataframe,
            revision=11,
        ):
            result = (
                execute_ml_hyperparameter_tuning(
                    training_contract=
                        contract,

                    search_contract=
                        search,
                )
            )

    finally:
        tuning_executor._build_estimator = (
            original_builder
        )


    candidate_count = len(
        server_owned_hyperparameter_candidates(
            estimator_key=
                "ridge_regression"
        )
    )


    expected_model_fits = (
        candidate_count
        *
        search.folds
    )


    assert (
        len(
            fit_index_sets
        )
        ==
        expected_model_fits
    )


    assert (
        len(
            prediction_index_sets
        )
        ==
        expected_model_fits
    )


    for indexes in (
        fit_index_sets
        +
        prediction_index_sets
    ):

        assert (
            indexes
            <=
            outer_train_indexes
        )


        assert (
            indexes
            .isdisjoint(
                holdout_test_indexes
            )
        )


    assert (
        result.outer_train_rows
        ==
        len(
            expected_outer_train
        )
    )


    assert (
        result.holdout_test_rows
        ==
        len(
            expected_holdout_test
        )
    )


    assert (
        result.preparation_session_revision
        ==
        11
    )


    assert (
        result.base_training_contract_sha256
        ==
        ml_training_contract_sha256(
            contract
        )
    )


# ============================================================
# COMPLETE GRID + RANKING
# ============================================================


def test_regression_evaluates_complete_grid_and_five_metrics(
) -> None:

    contract = (
        regression_contract()
    )


    search = (
        MLHyperparameterSearchContract(
            folds=
                5,

            random_seed=
                123,
        )
    )


    with authorized_dataframe(
        regression_dataframe(),
        revision=5,
    ):
        result = (
            execute_ml_hyperparameter_tuning(
                training_contract=
                    contract,

                search_contract=
                    search,
            )
        )


    grid = (
        server_owned_hyperparameter_candidates(
            estimator_key=
                contract.estimator_key
        )
    )


    assert (
        result.candidate_count
        ==
        len(
            grid
        )
        ==
        3
    )


    assert (
        len(
            result.candidate_results
        )
        ==
        3
    )


    assert (
        [
            candidate.rank

            for candidate
            in result.candidate_results
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
            candidate.candidate_index

            for candidate
            in result.candidate_results
        }
        ==
        {
            1,
            2,
            3,
        }
    )


    expected_metrics = set(
        expected_hyperparameter_metric_names(
            problem_type=
                "regression"
        )
    )


    candidate_sha256 = set()


    for candidate in (
        result.candidate_results
    ):

        assert (
            set(
                candidate.metric_summary
            )
            ==
            expected_metrics
        )


        for summary in (
            candidate
            .metric_summary
            .values()
        ):

            assert math.isfinite(
                float(
                    summary.mean
                )
            )


            assert math.isfinite(
                float(
                    summary.std
                )
            )


            assert (
                summary.std
                >=
                0.0
            )


        candidate_sha256.add(
            candidate.training_contract_sha256
        )


    assert (
        len(
            candidate_sha256
        )
        ==
        3
    )


    assert (
        result.best_candidate_index
        ==
        result.candidate_results[
            0
        ].candidate_index
    )


# ============================================================
# CLASSIFICATION
# ============================================================


def test_classification_uses_inner_stratified_kfold_and_f1_macro(
) -> None:

    contract = (
        classification_contract()
    )


    search = (
        MLHyperparameterSearchContract(
            folds=
                4,

            shuffle=
                True,

            random_seed=
                456,
        )
    )


    with authorized_dataframe(
        classification_dataframe(),
        revision=9,
    ):
        result = (
            execute_ml_hyperparameter_tuning(
                training_contract=
                    contract,

                search_contract=
                    search,
            )
        )


    assert (
        result.problem_type
        ==
        "classification"
    )


    assert (
        result.validation_strategy
        ==
        "stratified_k_fold"
    )


    assert (
        result.primary_metric
        ==
        "f1_macro"
    )


    assert (
        result.metric_direction
        ==
        "maximize"
    )


    assert (
        result.candidate_count
        ==
        4
    )


    expected_metrics = set(
        expected_hyperparameter_metric_names(
            problem_type=
                "classification"
        )
    )


    for candidate in (
        result.candidate_results
    ):

        assert (
            set(
                candidate.metric_summary
            )
            ==
            expected_metrics
        )


# ============================================================
# DETERMINISM
# ============================================================


def test_same_seed_is_exactly_deterministic(
) -> None:

    dataframe = (
        regression_dataframe()
    )


    contract = (
        regression_contract()
    )


    search = (
        MLHyperparameterSearchContract(
            folds=
                5,

            shuffle=
                True,

            random_seed=
                999,
        )
    )


    with authorized_dataframe(
        dataframe,
        revision=3,
    ):
        first = (
            execute_ml_hyperparameter_tuning(
                training_contract=
                    contract,

                search_contract=
                    search,
            )
        )


    with authorized_dataframe(
        dataframe,
        revision=3,
    ):
        second = (
            execute_ml_hyperparameter_tuning(
                training_contract=
                    contract,

                search_contract=
                    search,
            )
        )


    assert (
        first.model_dump(
            mode="json"
        )
        ==
        second.model_dump(
            mode="json"
        )
    )


# ============================================================
# BASE CONTRACT IMMUTABILITY
# ============================================================


def test_base_training_contract_is_not_mutated(
) -> None:

    dataframe = (
        regression_dataframe()
    )


    contract = (
        regression_contract()
    )


    before = (
        contract.model_dump(
            mode="json"
        )
    )


    before_sha = (
        ml_training_contract_sha256(
            contract
        )
    )


    with authorized_dataframe(
        dataframe
    ):
        result = (
            execute_ml_hyperparameter_tuning(
                training_contract=
                    contract,

                search_contract=(
                    MLHyperparameterSearchContract(
                        folds=
                            5
                    )
                ),
            )
        )


    after = (
        contract.model_dump(
            mode="json"
        )
    )


    after_sha = (
        ml_training_contract_sha256(
            contract
        )
    )


    assert (
        before
        ==
        after
    )


    assert (
        before_sha
        ==
        after_sha
    )


    assert (
        result.base_training_contract_sha256
        ==
        before_sha
    )


# ============================================================
# FAIL CLOSED ? REGRESSION INNER CV
# ============================================================


def test_regression_outer_train_must_support_inner_folds(
) -> None:

    contract = (
        regression_contract()
    )


    search = (
        MLHyperparameterSearchContract(
            folds=
                5
        )
    )


    try:
        with authorized_dataframe(
            regression_dataframe(
                rows=12
            )
        ):
            execute_ml_hyperparameter_tuning(
                training_contract=
                    contract,

                search_contract=
                    search,
            )

    except MLHyperparameterTuningInputError:
        return


    raise AssertionError(
        (
            "Hyperparameter Tuning must reject "
            "an OUTER training split that cannot "
            "support the requested INNER folds."
        )
    )


# ============================================================
# FAIL CLOSED ? CLASSIFICATION INNER CV
# ============================================================


def test_classification_outer_train_requires_each_class_per_fold(
) -> None:

    dataframe = pd.DataFrame(
        {
            "x1":
                [
                    float(
                        index
                    )

                    for index
                    in range(
                        12
                    )
                ],

            "x2":
                [
                    0.0
                    for _
                    in range(
                        12
                    )
                ],

            "target":
                (
                    [
                        "A"
                    ]
                    *
                    9
                    +
                    [
                        "B"
                    ]
                    *
                    3
                ),
        }
    )


    contract = (
        classification_contract()
    )


    search = (
        MLHyperparameterSearchContract(
            folds=
                3
        )
    )


    try:
        with authorized_dataframe(
            dataframe
        ):
            execute_ml_hyperparameter_tuning(
                training_contract=
                    contract,

                search_contract=
                    search,
            )

    except MLHyperparameterTuningInputError:
        return


    raise AssertionError(
        (
            "Stratified INNER CV must reject "
            "a fold count larger than the "
            "smallest OUTER-training class."
        )
    )


# ============================================================
# PRIVACY
# ============================================================


def test_result_is_evaluation_only_and_privacy_minimal(
) -> None:

    contract = (
        regression_contract()
    )


    with authorized_dataframe(
        regression_dataframe()
    ):
        result = (
            execute_ml_hyperparameter_tuning(
                training_contract=
                    contract,

                search_contract=(
                    MLHyperparameterSearchContract(
                        folds=
                            5
                    )
                ),
            )
        )


    payload = (
        result.model_dump(
            mode="json"
        )
    )


    forbidden = {
        "raw_rows",
        "predictions",
        "fold_predictions",
        "holdout_predictions",
        "x_train",
        "x_test",
        "y_train",
        "y_test",
        "model_id",
        "experiment_id",
        "model_artifact",
        "model_bytes",
        "model_path",
        "estimator",
    }


    assert (
        forbidden
        .isdisjoint(
            payload
        )
    )


# ============================================================
# VERSION
# ============================================================


def test_executor_rule_version(
) -> None:

    assert (
        ML_HYPERPARAMETER_TUNING_EXECUTOR_RULE_VERSION
        ==
        "ml_hyperparameter_tuning_executor_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML HYPERPARAMETER TUNING EXECUTOR v0.1 ==="
    )


    tests = [
        (
            "Outer holdout is invisible to tuning",
            test_regression_tunes_only_inside_outer_train,
        ),
        (
            "Complete grid + five richer metrics",
            test_regression_evaluates_complete_grid_and_five_metrics,
        ),
        (
            "Classification StratifiedKFold + F1 macro",
            test_classification_uses_inner_stratified_kfold_and_f1_macro,
        ),
        (
            "Same seed is deterministic",
            test_same_seed_is_exactly_deterministic,
        ),
        (
            "Base Training Contract remains immutable",
            test_base_training_contract_is_not_mutated,
        ),
        (
            "Regression inner-fold feasibility fail-closed",
            test_regression_outer_train_must_support_inner_folds,
        ),
        (
            "Classification inner-fold feasibility fail-closed",
            test_classification_outer_train_requires_each_class_per_fold,
        ),
        (
            "Evaluation-only privacy-minimal result",
            test_result_is_evaluation_only_and_privacy_minimal,
        ),
        (
            "Executor rule version",
            test_executor_rule_version,
        ),
    ]


    for (
        label,
        test,
    ) in tests:

        test()

        print(
            f"[PASS] {label}"
        )


    print()
    print(
        "PASS - ML Hyperparameter Tuning Executor v0.1"
    )


if __name__ == "__main__":
    main()
