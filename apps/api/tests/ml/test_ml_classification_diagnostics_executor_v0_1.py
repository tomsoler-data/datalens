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


import app.ml.classification_diagnostics_executor as diagnostics_executor


from app.ml.classification_diagnostics import (
    MLClassificationDiagnosticsContract,
)


from app.ml.classification_diagnostics_executor import (
    ML_CLASSIFICATION_DIAGNOSTICS_EXECUTOR_RULE_VERSION,
    MLClassificationDiagnosticsArtifactError,
    MLClassificationDiagnosticsExecutionError,
    MLClassificationDiagnosticsInputError,
    execute_ml_classification_diagnostics,
)


from app.ml.contracts import (
    MLTrainingContract,
    MLSplitContract,
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
# FIXTURE
# ============================================================


WORKFLOW_ID = (
    "prep:classification-diagnostics"
)


DATASET_ID = (
    "dataset:classification"
)


MODEL_ID = (
    "model:classification-diagnostics"
)


EXPERIMENT_ID = (
    "experiment:"
    +
    "a"
    *
    32
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


PREDICTIONS = np.asarray(
    [
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
    ],
    dtype=object,
)


# ============================================================
# FAKE TRUSTED CLASSIFIER
# ============================================================


class PredictOnlyClassifier:

    def __init__(
        self,
        *,
        classes,
        predictions,
    ) -> None:

        self.classes_ = np.asarray(
            classes,
            dtype=object,
        )


        self._predictions = (
            np.asarray(
                predictions,
                dtype=object,
            )
        )


        self.predict_calls = 0
        self.fit_calls = 0


    def fit(
        self,
        *args,
        **kwargs,
    ):

        self.fit_calls += 1


        raise AssertionError(
            (
                "Classification Diagnostics "
                "must never fit or refit."
            )
        )


    def predict(
        self,
        features,
    ):

        self.predict_calls += 1


        return (
            self._predictions.copy()
        )


# ============================================================
# TRAINING CONTRACT
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


# ============================================================
# METRICS
# ============================================================


def classification_metrics(
    *,
    y_true=Y_TEST,
    predictions=PREDICTIONS,
) -> dict[
    str,
    float
]:

    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    predictions,
                )
            ),

        "f1_macro":
            float(
                f1_score(
                    y_true,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),

        "precision_macro":
            float(
                precision_score(
                    y_true,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),

        "recall_macro":
            float(
                recall_score(
                    y_true,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    predictions,
                )
            ),
    }


# ============================================================
# ARTIFACT
# ============================================================


def classification_artifact(
    *,
    metrics=None,
) -> MLModelArtifactRecord:

    training_contract = (
        classification_training_contract()
    )


    persisted_metrics = (
        classification_metrics()

        if metrics is None

        else dict(
            metrics
        )
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
                persisted_metrics,
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
                persisted_metrics,

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
                "classification-diagnostics.joblib"
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
# PATCH
# ============================================================


def run_with_fake_runtime(
    *,
    estimator,
    artifact=None,
    current_revision=PREPARATION_REVISION,
    holdout=None,
):

    if artifact is None:
        artifact = (
            classification_artifact()
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
        diagnostics_executor
        .load_trusted_ml_model
    )


    original_dataframe = (
        diagnostics_executor
        ._load_authorized_dataframe
    )


    original_xy = (
        diagnostics_executor
        ._validate_and_extract_xy
    )


    original_split = (
        diagnostics_executor
        ._split_dataset
    )


    diagnostics_executor.load_trusted_ml_model = (
        lambda **kwargs:
            loaded_model
    )


    diagnostics_executor._load_authorized_dataframe = (
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


    diagnostics_executor._validate_and_extract_xy = (
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


    diagnostics_executor._split_dataset = (
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
            execute_ml_classification_diagnostics(
                workflow_id=
                    WORKFLOW_ID,

                model_id=
                    MODEL_ID,

                diagnostics_contract=(
                    MLClassificationDiagnosticsContract()
                ),
            )
        )

    finally:

        diagnostics_executor.load_trusted_ml_model = (
            original_loader
        )


        diagnostics_executor._load_authorized_dataframe = (
            original_dataframe
        )


        diagnostics_executor._validate_and_extract_xy = (
            original_xy
        )


        diagnostics_executor._split_dataset = (
            original_split
        )


# ============================================================
# HAPPY PATH
# ============================================================


def test_executor_builds_holdout_diagnostics(
) -> None:

    estimator = (
        PredictOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            predictions=
                PREDICTIONS,
        )
    )


    result = (
        run_with_fake_runtime(
            estimator=
                estimator
        )
    )


    assert (
        estimator.fit_calls
        ==
        0
    )


    assert (
        estimator.predict_calls
        ==
        1
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
        result.problem_type
        ==
        "classification"
    )


    assert (
        result.evaluation_rows
        ==
        TEST_ROWS
    )


    assert (
        result.class_labels
        ==
        [
            "negative",
            "positive",
        ]
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


# ============================================================
# CLASS ORDER
# ============================================================


def test_class_order_is_estimator_owned(
) -> None:

    estimator = (
        PredictOnlyClassifier(
            classes=[
                "positive",
                "negative",
            ],

            predictions=
                PREDICTIONS,
        )
    )


    result = (
        run_with_fake_runtime(
            estimator=
                estimator
        )
    )


    assert (
        result.class_labels
        ==
        [
            "positive",
            "negative",
        ]
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
                2,
                8,
            ],
        ]
    )


# ============================================================
# NO REFIT
# ============================================================


def test_executor_never_refits(
) -> None:

    estimator = (
        PredictOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            predictions=
                PREDICTIONS,
        )
    )


    run_with_fake_runtime(
        estimator=
            estimator
    )


    assert (
        estimator.fit_calls
        ==
        0
    )


    assert (
        estimator.predict_calls
        ==
        1
    )


# ============================================================
# STALE REVISION
# ============================================================


def test_stale_preparation_revision_blocks_before_prediction(
) -> None:

    estimator = (
        PredictOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            predictions=
                PREDICTIONS,
        )
    )


    try:
        run_with_fake_runtime(
            estimator=
                estimator,

            current_revision=(
                PREPARATION_REVISION
                +
                1
            ),
        )

    except MLClassificationDiagnosticsInputError:
        pass

    else:
        raise AssertionError(
            (
                "Stale Preparation revision "
                "must fail closed."
            )
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
# HOLDOUT SHAPE
# ============================================================


def test_holdout_shape_mismatch_blocks_before_prediction(
) -> None:

    estimator = (
        PredictOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            predictions=
                PREDICTIONS,
        )
    )


    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = holdout_fixture()


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


    try:
        run_with_fake_runtime(
            estimator=
                estimator,

            holdout=
                bad_holdout,
        )

    except MLClassificationDiagnosticsInputError:
        pass

    else:
        raise AssertionError(
            (
                "Holdout shape mismatch "
                "must fail closed."
            )
        )


    assert (
        estimator.predict_calls
        ==
        0
    )


# ============================================================
# PERSISTED METRIC BINDING
# ============================================================


def test_persisted_metric_mismatch_fails_closed(
) -> None:

    estimator = (
        PredictOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            predictions=
                PREDICTIONS,
        )
    )


    metrics = (
        classification_metrics()
    )


    metrics[
        "accuracy"
    ] = 0.50


    artifact = (
        classification_artifact(
            metrics=
                metrics
        )
    )


    try:
        run_with_fake_runtime(
            estimator=
                estimator,

            artifact=
                artifact,
        )

    except MLClassificationDiagnosticsArtifactError:
        pass

    else:
        raise AssertionError(
            (
                "Diagnostics must be bound back "
                "to persisted classification metrics."
            )
        )


    assert (
        estimator.fit_calls
        ==
        0
    )


    assert (
        estimator.predict_calls
        ==
        1
    )


# ============================================================
# CLASS SURFACE
# ============================================================


def test_missing_estimator_classes_fails_closed(
) -> None:

    class NoClassesClassifier:

        def __init__(
            self,
        ) -> None:

            self.predict_calls = 0


        def predict(
            self,
            features,
        ):

            self.predict_calls += 1


            return (
                PREDICTIONS.copy()
            )


    estimator = (
        NoClassesClassifier()
    )


    try:
        run_with_fake_runtime(
            estimator=
                estimator
        )

    except MLClassificationDiagnosticsArtifactError:
        pass

    else:
        raise AssertionError(
            (
                "Trusted classifier without "
                "classes_ must fail closed."
            )
        )


    assert (
        estimator.predict_calls
        ==
        0
    )


# ============================================================
# UNKNOWN HOLDOUT CLASS
# ============================================================


def test_holdout_class_outside_estimator_classes_fails_closed(
) -> None:

    estimator = (
        PredictOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            predictions=
                PREDICTIONS,
        )
    )


    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = holdout_fixture()


    y_test = (
        y_test.copy(
            deep=True
        )
    )


    y_test.iloc[
        0
    ] = "unseen-class"


    bad_holdout = (
        x_train,
        x_test,
        y_train,
        y_test,
    )


    try:
        run_with_fake_runtime(
            estimator=
                estimator,

            holdout=
                bad_holdout,
        )

    except MLClassificationDiagnosticsExecutionError:
        pass

    else:
        raise AssertionError(
            (
                "Holdout classes outside fitted "
                "estimator classes_ must fail closed."
            )
        )


# ============================================================
# PRIVACY
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


def test_executor_result_is_privacy_minimal(
) -> None:

    estimator = (
        PredictOnlyClassifier(
            classes=[
                "negative",
                "positive",
            ],

            predictions=
                PREDICTIONS,
        )
    )


    payload = (
        run_with_fake_runtime(
            estimator=
                estimator
        )
        .model_dump(
            mode="json"
        )
    )


    forbidden = {
        "raw_rows",
        "rows",
        "predictions",
        "holdout_predictions",
        "probabilities",
        "decision_scores",
        "y_true",
        "y_pred",
        "x_train",
        "x_test",
        "y_train",
        "y_test",
        "model_bytes",
        "model_path",
        "estimator",
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


def test_executor_rule_version(
) -> None:

    assert (
        ML_CLASSIFICATION_DIAGNOSTICS_EXECUTOR_RULE_VERSION
        ==
        "ml_classification_diagnostics_executor_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML CLASSIFICATION DIAGNOSTICS EXECUTOR v0.1 ==="
    )


    tests = [
        (
            "Holdout confusion matrix + diagnostics",
            test_executor_builds_holdout_diagnostics,
        ),
        (
            "Class ordering is estimator-owned",
            test_class_order_is_estimator_owned,
        ),
        (
            "Diagnostics never fit or refit",
            test_executor_never_refits,
        ),
        (
            "Stale Preparation revision blocks",
            test_stale_preparation_revision_blocks_before_prediction,
        ),
        (
            "Holdout shape mismatch blocks",
            test_holdout_shape_mismatch_blocks_before_prediction,
        ),
        (
            "Persisted metrics bind diagnostics",
            test_persisted_metric_mismatch_fails_closed,
        ),
        (
            "Missing estimator classes fail-closed",
            test_missing_estimator_classes_fails_closed,
        ),
        (
            "Unknown holdout class fails-closed",
            test_holdout_class_outside_estimator_classes_fails_closed,
        ),
        (
            "Privacy-minimal executor result",
            test_executor_result_is_privacy_minimal,
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
        "PASS - ML Classification Diagnostics Executor v0.1"
    )


if __name__ == "__main__":
    main()
