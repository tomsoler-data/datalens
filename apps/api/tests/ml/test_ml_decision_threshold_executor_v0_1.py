from __future__ import annotations


import math


import numpy as np
import pandas as pd


from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


import app.ml.decision_threshold_executor as threshold_executor


from app.ml.contracts import (
    MLSplitContract,
    MLTrainingContract,
)


from app.ml.decision_threshold import (
    MLDecisionThresholdContract,
)


from app.ml.decision_threshold_executor import (
    ML_DECISION_THRESHOLD_EXECUTOR_RULE_VERSION,
    MLDecisionThresholdArtifactError,
    MLDecisionThresholdExecutionError,
    MLDecisionThresholdInputError,
    execute_ml_decision_threshold,
)


from app.ml.experiment_provenance import (
    MLExperimentProvenanceRecord,
    ml_training_contract_sha256,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.ml.model_loader import (
    LoadedMLModel,
)


# ============================================================
# CONSTANTS
# ============================================================


WORKFLOW_ID = (
    "prep:decision-threshold"
)


DATASET_ID = (
    "dataset:classification"
)


MODEL_ID = (
    "model:decision-threshold"
)


EXPERIMENT_ID = (
    "experiment:"
    +
    (
        "a"
        *
        32
    )
)


PREPARATION_REVISION = 7


TRAIN_ROWS = 20


TEST_ROWS = 20


Y_TEST = pd.Series(
    [
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
    ],
    name="churned",
)


POSITIVE_PROBABILITIES = np.asarray(
    [
        0.10,
        0.20,
        0.30,
        0.40,
        0.45,
        0.49,
        0.00,
        0.25,
        0.60,
        0.80,
        0.40,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ],
    dtype=np.float64,
)


PROBABILITIES = np.column_stack(
    (
        1.0
        -
        POSITIVE_PROBABILITIES,

        POSITIVE_PROBABILITIES,
    )
)


NATIVE_PREDICTIONS = np.where(
    (
        POSITIVE_PROBABILITIES
        >=
        0.50
    ),
    "positive",
    "negative",
)


# ============================================================
# ASSERTION HELPERS
# ============================================================


def expect_exception(
    exception_type,
    factory,
) -> None:

    try:
        factory()

    except exception_type:
        return


    raise AssertionError(
        (
            "Expected exception: "
            f"{exception_type.__name__}"
        )
    )


# ============================================================
# FAKE TRUSTED PROBABILITY CLASSIFIER
# ============================================================


class ProbabilityOnlyClassifier:

    def __init__(
        self,
        *,
        classes,
        probabilities,
    ) -> None:

        self.classes_ = np.asarray(
            classes,
            dtype=object,
        )


        self._probabilities = np.asarray(
            probabilities
        )


        self.fit_calls = 0
        self.predict_calls = 0
        self.predict_proba_calls = 0


    def fit(
        self,
        *args,
        **kwargs,
    ):

        self.fit_calls += 1


        raise AssertionError(
            (
                "Decision Threshold must never "
                "fit or refit."
            )
        )


    def predict(
        self,
        *args,
        **kwargs,
    ):

        self.predict_calls += 1


        raise AssertionError(
            (
                "Decision Threshold must never "
                "use native predict()."
            )
        )


    def predict_proba(
        self,
        features,
    ):

        self.predict_proba_calls += 1


        return (
            self._probabilities.copy()
        )


class ClassesOnlyEstimator:

    def __init__(
        self,
        *,
        classes,
    ) -> None:

        self.classes_ = np.asarray(
            classes,
            dtype=object,
        )


class ProbabilityWithoutClassesEstimator:

    def __init__(
        self,
        *,
        probabilities,
    ) -> None:

        self._probabilities = np.asarray(
            probabilities
        )


        self.predict_proba_calls = 0


    def predict_proba(
        self,
        features,
    ):

        self.predict_proba_calls += 1


        return (
            self._probabilities.copy()
        )


# ============================================================
# TRAINING CONTRACTS
# ============================================================


def classification_training_contract(
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
                "score",
            ],

            estimator_key=
                "logistic_regression",

            split=(
                MLSplitContract(
                    test_size=
                        0.49,

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


def regression_training_contract(
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
                "target",

            feature_columns=[
                "score",
            ],

            estimator_key=
                "linear_regression",

            split=(
                MLSplitContract(
                    test_size=
                        0.49,

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


# ============================================================
# PERSISTED METRICS
# ============================================================


def classification_metrics(
) -> dict[
    str,
    float
]:

    return {
        "accuracy":
            float(
                accuracy_score(
                    Y_TEST,
                    NATIVE_PREDICTIONS,
                )
            ),

        "f1_macro":
            float(
                f1_score(
                    Y_TEST,
                    NATIVE_PREDICTIONS,
                    average="macro",
                    zero_division=0,
                )
            ),

        "precision_macro":
            float(
                precision_score(
                    Y_TEST,
                    NATIVE_PREDICTIONS,
                    average="macro",
                    zero_division=0,
                )
            ),

        "recall_macro":
            float(
                recall_score(
                    Y_TEST,
                    NATIVE_PREDICTIONS,
                    average="macro",
                    zero_division=0,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    Y_TEST,
                    NATIVE_PREDICTIONS,
                )
            ),
    }


def regression_metrics(
) -> dict[
    str,
    float
]:

    return {
        "mae":
            1.0,

        "rmse":
            1.0,

        "r2":
            0.5,

        "median_absolute_error":
            1.0,

        "explained_variance":
            0.5,
    }


# ============================================================
# MODEL ARTIFACT
# ============================================================


def build_artifact(
    *,
    training_contract=None,
    metrics=None,
) -> MLModelArtifactRecord:

    if training_contract is None:
        training_contract = (
            classification_training_contract()
        )


    if metrics is None:

        if (
            training_contract.problem_type
            ==
            "classification"
        ):
            metrics = (
                classification_metrics()
            )

        else:
            metrics = (
                regression_metrics()
            )


    provenance = (
        MLExperimentProvenanceRecord(
            experiment_id=
                EXPERIMENT_ID,

            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

            preparation_session_revision=(
                PREPARATION_REVISION
            ),

            training_contract_sha256=(
                ml_training_contract_sha256(
                    training_contract
                )
            ),

            model_id=
                MODEL_ID,

            train_rows=
                TRAIN_ROWS,

            test_rows=
                TEST_ROWS,

            metrics=
                metrics,
        )
    )


    return (
        MLModelArtifactRecord(
            model_id=
                MODEL_ID,

            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

            training_contract=
                training_contract,

            experiment_provenance=
                provenance,

            metrics=
                metrics,

            train_rows=
                TRAIN_ROWS,

            test_rows=
                TEST_ROWS,

            created_at_utc=
                "2026-08-29T00:00:00+00:00",

            serialization_format=
                "joblib",

            model_path=(
                "models/"
                "decision-threshold.joblib"
            ),

            model_file_bytes=
                123,

            model_sha256=(
                "b"
                *
                64
            ),
        )
    )


# ============================================================
# HOLDOUT
# ============================================================


def holdout_fixture(
):

    x_train = pd.DataFrame(
        {
            "score":
                list(
                    range(
                        TRAIN_ROWS
                    )
                )
        }
    )


    x_test = pd.DataFrame(
        {
            "score":
                list(
                    range(
                        100,
                        100
                        +
                        TEST_ROWS,
                    )
                )
        }
    )


    y_train = pd.Series(
        [
            "negative"
            if (
                index
                %
                2
                ==
                0
            )
            else
            "positive"

            for index
            in range(
                TRAIN_ROWS
            )
        ],
        name="churned",
    )


    y_test = (
        Y_TEST.copy(
            deep=True
        )
    )


    return (
        x_train,
        x_test,
        y_train,
        y_test,
    )


# ============================================================
# PATCHED TRUSTED RUNTIME
# ============================================================


def run_with_fake_runtime(
    *,
    estimator,
    threshold=0.50,
    artifact=None,
    current_revision=PREPARATION_REVISION,
    holdout=None,
):

    if artifact is None:
        artifact = (
            build_artifact()
        )


    if holdout is None:
        holdout = (
            holdout_fixture()
        )


    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = holdout


    loaded_model = (
        LoadedMLModel(
            artifact=
                artifact,

            estimator=
                estimator,
        )
    )


    original_loader = (
        threshold_executor
        .load_trusted_ml_model
    )


    original_dataframe = (
        threshold_executor
        ._load_authorized_dataframe
    )


    original_xy = (
        threshold_executor
        ._validate_and_extract_xy
    )


    original_split = (
        threshold_executor
        ._split_dataset
    )


    threshold_executor.load_trusted_ml_model = (
        lambda **kwargs:
            loaded_model
    )


    threshold_executor._load_authorized_dataframe = (
        lambda **kwargs:
            (
                pd.DataFrame(
                    {
                        "score":
                            list(
                                range(
                                    TRAIN_ROWS
                                    +
                                    TEST_ROWS
                                )
                            ),

                        "churned":
                            [
                                "negative",
                                "positive",
                            ]
                            *
                            (
                                (
                                    TRAIN_ROWS
                                    +
                                    TEST_ROWS
                                )
                                //
                                2
                            ),
                    }
                ),
                current_revision,
            )
    )


    threshold_executor._validate_and_extract_xy = (
        lambda **kwargs:
            (
                pd.concat(
                    [
                        x_train,
                        x_test,
                    ],
                    ignore_index=True,
                ),

                pd.concat(
                    [
                        y_train,
                        y_test,
                    ],
                    ignore_index=True,
                ),
            )
    )


    threshold_executor._split_dataset = (
        lambda **kwargs:
            (
                x_train,
                x_test,
                y_train,
                y_test,
            )
    )


    try:
        return (
            execute_ml_decision_threshold(
                workflow_id=
                    WORKFLOW_ID,

                model_id=
                    MODEL_ID,

                threshold_contract=(
                    MLDecisionThresholdContract(
                        threshold=
                            threshold
                    )
                ),
            )
        )

    finally:

        threshold_executor.load_trusted_ml_model = (
            original_loader
        )


        threshold_executor._load_authorized_dataframe = (
            original_dataframe
        )


        threshold_executor._validate_and_extract_xy = (
            original_xy
        )


        threshold_executor._split_dataset = (
            original_split
        )


# ============================================================
# HAPPY PATH
# ============================================================


def test_executor_builds_requested_threshold_evaluation(
) -> None:

    estimator = (
        ProbabilityOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            probabilities=
                PROBABILITIES,
        )
    )


    result = (
        run_with_fake_runtime(
            estimator=
                estimator,

            threshold=
                0.50,
        )
    )


    assert (
        result.threshold
        ==
        0.50
    )


    assert (
        result.negative_class_label
        ==
        "negative"
    )


    assert (
        result.positive_class_label
        ==
        "positive"
    )


    assert (
        result.confusion_matrix
        ==
        [
            [
                8,
                2,
            ],
            [
                1,
                9,
            ],
        ]
    )


    assert (
        result.true_negative
        ==
        8
    )


    assert (
        result.false_positive
        ==
        2
    )


    assert (
        result.false_negative
        ==
        1
    )


    assert (
        result.true_positive
        ==
        9
    )


    assert math.isclose(
        result.accuracy,
        0.85,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        result.balanced_accuracy,
        0.85,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        result.positive_prediction_rate,
        0.55,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert (
        estimator.predict_proba_calls
        ==
        1
    )


    assert (
        estimator.predict_calls
        ==
        0
    )


    assert (
        estimator.fit_calls
        ==
        0
    )


# ============================================================
# THRESHOLD SEMANTICS
# ============================================================


def test_threshold_changes_predictions_deterministically(
) -> None:

    estimator = (
        ProbabilityOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            probabilities=
                PROBABILITIES,
        )
    )


    result = (
        run_with_fake_runtime(
            estimator=
                estimator,

            threshold=
                0.70,
        )
    )


    assert (
        result.confusion_matrix
        ==
        [
            [
                9,
                1,
            ],
            [
                4,
                6,
            ],
        ]
    )


    assert (
        result.true_positive
        ==
        6
    )


    assert (
        result.false_negative
        ==
        4
    )


    assert (
        result.false_positive
        ==
        1
    )


    assert (
        result.true_negative
        ==
        9
    )


    # One positive observation has probability exactly 0.70.
    # It must therefore be classified positive because the
    # v0.1 operator is >= threshold.
    assert (
        result.positive_prediction_rate
        ==
        (
            7
            /
            20
        )
    )


def test_positive_class_is_estimator_classes_index_1(
) -> None:

    reversed_probabilities = (
        PROBABILITIES[
            :,
            ::-1
        ]
    )


    estimator = (
        ProbabilityOnlyClassifier(
            classes=[
                "positive",
                "negative",
            ],

            probabilities=
                reversed_probabilities,
        )
    )


    result = (
        run_with_fake_runtime(
            estimator=
                estimator,

            threshold=
                0.50,
        )
    )


    assert (
        result.negative_class_label
        ==
        "positive"
    )


    assert (
        result.positive_class_label
        ==
        "negative"
    )


    assert (
        estimator.predict_proba_calls
        ==
        1
    )


# ============================================================
# NO FIT / NO PREDICT
# ============================================================


def test_executor_uses_predict_proba_only(
) -> None:

    estimator = (
        ProbabilityOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            probabilities=
                PROBABILITIES,
        )
    )


    run_with_fake_runtime(
        estimator=
            estimator
    )


    assert (
        estimator.predict_proba_calls
        ==
        1
    )


    assert (
        estimator.predict_calls
        ==
        0
    )


    assert (
        estimator.fit_calls
        ==
        0
    )


# ============================================================
# CLASSIFICATION / BINARY GATES
# ============================================================


def test_regression_artifact_fails_closed(
) -> None:

    estimator = (
        ProbabilityOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            probabilities=
                PROBABILITIES,
        )
    )


    artifact = (
        build_artifact(
            training_contract=(
                regression_training_contract()
            )
        )
    )


    expect_exception(
        MLDecisionThresholdArtifactError,
        lambda:
            run_with_fake_runtime(
                estimator=
                    estimator,

                artifact=
                    artifact,
            ),
    )


    assert (
        estimator.predict_proba_calls
        ==
        0
    )


def test_multiclass_artifact_fails_closed(
) -> None:

    probabilities = np.tile(
        np.asarray(
            [
                [
                    0.20,
                    0.30,
                    0.50,
                ]
            ],
            dtype=np.float64,
        ),
        (
            TEST_ROWS,
            1,
        ),
    )


    estimator = (
        ProbabilityOnlyClassifier(
            classes=[
                "a",
                "b",
                "c",
            ],

            probabilities=
                probabilities,
        )
    )


    expect_exception(
        MLDecisionThresholdArtifactError,
        lambda:
            run_with_fake_runtime(
                estimator=
                    estimator
            ),
    )


    assert (
        estimator.predict_proba_calls
        ==
        0
    )


def test_missing_classes_fails_closed(
) -> None:

    estimator = (
        ProbabilityWithoutClassesEstimator(
            probabilities=
                PROBABILITIES
        )
    )


    expect_exception(
        MLDecisionThresholdArtifactError,
        lambda:
            run_with_fake_runtime(
                estimator=
                    estimator
            ),
    )


    assert (
        estimator.predict_proba_calls
        ==
        0
    )


def test_missing_predict_proba_fails_closed(
) -> None:

    estimator = (
        ClassesOnlyEstimator(
            classes=[
                "negative",
                "positive",
            ]
        )
    )


    expect_exception(
        MLDecisionThresholdArtifactError,
        lambda:
            run_with_fake_runtime(
                estimator=
                    estimator
            ),
    )


# ============================================================
# PREPARATION PINNING
# ============================================================


def test_stale_preparation_revision_blocks_before_scoring(
) -> None:

    estimator = (
        ProbabilityOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            probabilities=
                PROBABILITIES,
        )
    )


    expect_exception(
        MLDecisionThresholdInputError,
        lambda:
            run_with_fake_runtime(
                estimator=
                    estimator,

                current_revision=(
                    PREPARATION_REVISION
                    +
                    1
                ),
            ),
    )


    assert (
        estimator.predict_proba_calls
        ==
        0
    )


def test_holdout_shape_mismatch_blocks_before_scoring(
) -> None:

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = (
        holdout_fixture()
    )


    bad_holdout = (
        x_train,
        x_test.iloc[
            :-1
        ].copy(),
        y_train,
        y_test.iloc[
            :-1
        ].copy(),
    )


    estimator = (
        ProbabilityOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            probabilities=
                PROBABILITIES[
                    :-1
                ],
        )
    )


    expect_exception(
        MLDecisionThresholdInputError,
        lambda:
            run_with_fake_runtime(
                estimator=
                    estimator,

                holdout=
                    bad_holdout,
            ),
    )


    assert (
        estimator.predict_proba_calls
        ==
        0
    )


# ============================================================
# PROBABILITY VALIDATION
# ============================================================


def test_probability_shape_fails_closed(
) -> None:

    malformed = np.ones(
        (
            TEST_ROWS,
            1,
        ),
        dtype=np.float64,
    )


    estimator = (
        ProbabilityOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            probabilities=
                malformed,
        )
    )


    expect_exception(
        MLDecisionThresholdExecutionError,
        lambda:
            run_with_fake_runtime(
                estimator=
                    estimator
            ),
    )


    assert (
        estimator.predict_proba_calls
        ==
        1
    )


def test_non_finite_probabilities_fail_closed(
) -> None:

    malformed = (
        PROBABILITIES.copy()
    )


    malformed[
        0,
        1
    ] = np.nan


    estimator = (
        ProbabilityOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            probabilities=
                malformed,
        )
    )


    expect_exception(
        MLDecisionThresholdExecutionError,
        lambda:
            run_with_fake_runtime(
                estimator=
                    estimator
            ),
    )


def test_out_of_range_probabilities_fail_closed(
) -> None:

    malformed = (
        PROBABILITIES.copy()
    )


    malformed[
        0
    ] = [
        -0.1,
        1.1,
    ]


    estimator = (
        ProbabilityOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            probabilities=
                malformed,
        )
    )


    expect_exception(
        MLDecisionThresholdExecutionError,
        lambda:
            run_with_fake_runtime(
                estimator=
                    estimator
            ),
    )


def test_probability_rows_must_sum_to_one(
) -> None:

    malformed = (
        PROBABILITIES.copy()
    )


    malformed[
        0
    ] = [
        0.20,
        0.20,
    ]


    estimator = (
        ProbabilityOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            probabilities=
                malformed,
        )
    )


    expect_exception(
        MLDecisionThresholdExecutionError,
        lambda:
            run_with_fake_runtime(
                estimator=
                    estimator
            ),
    )


# ============================================================
# UNKNOWN HOLDOUT CLASS
# ============================================================


def test_unknown_holdout_class_fails_closed(
) -> None:

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = (
        holdout_fixture()
    )


    y_test = (
        y_test.copy(
            deep=True
        )
    )


    y_test.iloc[
        -1
    ] = "unknown"


    bad_holdout = (
        x_train,
        x_test,
        y_train,
        y_test,
    )


    estimator = (
        ProbabilityOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            probabilities=
                PROBABILITIES,
        )
    )


    expect_exception(
        MLDecisionThresholdExecutionError,
        lambda:
            run_with_fake_runtime(
                estimator=
                    estimator,

                holdout=
                    bad_holdout,
            ),
    )


    assert (
        estimator.predict_proba_calls
        ==
        1
    )


# ============================================================
# PROVENANCE
# ============================================================


def test_experiment_provenance_is_required(
) -> None:

    artifact = (
        build_artifact()
        .model_copy(
            update={
                "experiment_provenance":
                    None
            }
        )
    )


    estimator = (
        ProbabilityOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            probabilities=
                PROBABILITIES,
        )
    )


    expect_exception(
        MLDecisionThresholdArtifactError,
        lambda:
            run_with_fake_runtime(
                estimator=
                    estimator,

                artifact=
                    artifact,
            ),
    )


    assert (
        estimator.predict_proba_calls
        ==
        0
    )


def test_training_contract_sha_is_recomputed(
) -> None:

    artifact = (
        build_artifact()
    )


    provenance = (
        artifact
        .experiment_provenance
    )


    assert (
        provenance
        is not None
    )


    bad_provenance = (
        provenance.model_copy(
            update={
                "training_contract_sha256":
                    (
                        "c"
                        *
                        64
                    )
            }
        )
    )


    bad_artifact = (
        artifact.model_copy(
            update={
                "experiment_provenance":
                    bad_provenance
            }
        )
    )


    estimator = (
        ProbabilityOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            probabilities=
                PROBABILITIES,
        )
    )


    expect_exception(
        MLDecisionThresholdArtifactError,
        lambda:
            run_with_fake_runtime(
                estimator=
                    estimator,

                artifact=
                    bad_artifact,
            ),
    )


    assert (
        estimator.predict_proba_calls
        ==
        0
    )


# ============================================================
# RESULT AUTHORITY / PRIVACY
# ============================================================


def _all_keys(
    value,
) -> set[
    str
]:

    keys: set[
        str
    ] = set()


    if isinstance(
        value,
        dict,
    ):

        for (
            key,
            nested,
        ) in value.items():

            keys.add(
                str(
                    key
                )
            )


            keys.update(
                _all_keys(
                    nested
                )
            )


    elif isinstance(
        value,
        list,
    ):

        for nested in value:

            keys.update(
                _all_keys(
                    nested
                )
            )


    return keys


def test_result_is_bound_and_privacy_minimal(
) -> None:

    estimator = (
        ProbabilityOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            probabilities=
                PROBABILITIES,
        )
    )


    result = (
        run_with_fake_runtime(
            estimator=
                estimator,

            threshold=
                0.40,
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
        result.model_id
        ==
        MODEL_ID
    )


    assert (
        result.experiment_id
        ==
        EXPERIMENT_ID
    )


    assert (
        result.preparation_session_revision
        ==
        PREPARATION_REVISION
    )


    assert (
        result.training_contract_sha256
        ==
        ml_training_contract_sha256(
            classification_training_contract()
        )
    )


    assert (
        result.score_source
        ==
        "predict_proba"
    )


    assert (
        result.positive_class_policy
        ==
        "estimator_classes_index_1"
    )


    assert (
        result.comparison_operator
        ==
        "greater_than_or_equal"
    )


    assert (
        result.threshold_selection_policy
        ==
        "evaluate_requested_threshold_only"
    )


    payload = (
        result.model_dump(
            mode="json"
        )
    )


    forbidden = {
        "rows",
        "raw_rows",
        "predictions",
        "probabilities",
        "positive_probabilities",
        "negative_probabilities",
        "decision_scores",
        "scores",
        "y_true",
        "y_pred",
        "x_train",
        "x_test",
        "y_train",
        "y_test",
        "model_bytes",
        "model_path",
        "estimator",
        "training_contract",
    }


    assert (
        forbidden.isdisjoint(
            _all_keys(
                payload
            )
        )
    )


# ============================================================
# RULE VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_DECISION_THRESHOLD_EXECUTOR_RULE_VERSION
        ==
        "ml_decision_threshold_executor_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML DECISION THRESHOLD EXECUTOR v0.1 ==="
    )


    tests = [
        (
            "Requested threshold -> binary diagnostics",
            test_executor_builds_requested_threshold_evaluation,
        ),
        (
            "Threshold changes predictions deterministically",
            test_threshold_changes_predictions_deterministically,
        ),
        (
            "Positive class is estimator classes_[1]",
            test_positive_class_is_estimator_classes_index_1,
        ),
        (
            "Executor uses predict_proba only",
            test_executor_uses_predict_proba_only,
        ),
        (
            "Regression artifact fails closed",
            test_regression_artifact_fails_closed,
        ),
        (
            "Multiclass artifact fails closed",
            test_multiclass_artifact_fails_closed,
        ),
        (
            "Missing classes_ fails closed",
            test_missing_classes_fails_closed,
        ),
        (
            "Missing predict_proba fails closed",
            test_missing_predict_proba_fails_closed,
        ),
        (
            "Stale Preparation revision blocks",
            test_stale_preparation_revision_blocks_before_scoring,
        ),
        (
            "Holdout shape mismatch blocks",
            test_holdout_shape_mismatch_blocks_before_scoring,
        ),
        (
            "Probability shape fail-closed",
            test_probability_shape_fails_closed,
        ),
        (
            "Non-finite probabilities fail-closed",
            test_non_finite_probabilities_fail_closed,
        ),
        (
            "Out-of-range probabilities fail-closed",
            test_out_of_range_probabilities_fail_closed,
        ),
        (
            "Probability rows sum to one",
            test_probability_rows_must_sum_to_one,
        ),
        (
            "Unknown holdout class fails-closed",
            test_unknown_holdout_class_fails_closed,
        ),
        (
            "Experiment Provenance required",
            test_experiment_provenance_is_required,
        ),
        (
            "Training Contract SHA recomputed",
            test_training_contract_sha_is_recomputed,
        ),
        (
            "Privacy-minimal bound result",
            test_result_is_bound_and_privacy_minimal,
        ),
        (
            "Executor rule version",
            test_rule_version,
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
        "PASS - ML Decision Threshold Executor v0.1"
    )


if __name__ == "__main__":
    main()
