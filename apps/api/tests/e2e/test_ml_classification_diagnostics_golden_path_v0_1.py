from __future__ import annotations


import math


import numpy as np


from fastapi.testclient import (
    TestClient,
)


from sklearn.linear_model import (
    LogisticRegression,
)


from sklearn.metrics import (
    confusion_matrix,
)


from sklearn.pipeline import (
    Pipeline,
)


# ============================================================
# REAL ISOLATED PRODUCT ENVIRONMENT
# ============================================================
#
# Import this module before the production ML modules below.
#
# It establishes:
# - isolated SQLite;
# - isolated Preparation Artifact Store;
# - isolated Model Artifact Store;
# - real FastAPI Preparation flow;
# - real 30-row mixed-type dataset.
# ============================================================


from tests.e2e.test_ml_preprocessing_golden_path_v0_1 import (
    WORKFLOW_ROOT_DATASET_ID,
    app,
    create_preparation_session,
    reset_product_state,
    run_real_cleaning_plan,
    run_real_quality,
    select_analysis_output,
    validate_preparation,
    verify_preparation_persistence,
    verify_real_handoff,
)


from app.ml.classical_executor import (
    _load_authorized_dataframe,
    _split_dataset,
    _validate_and_extract_xy,
    execute_classical_ml,
)


from app.ml.classification_diagnostics import (
    ML_CLASSIFICATION_DIAGNOSTICS_RULE_VERSION,
    MLClassificationDiagnosticsContract,
)


import app.ml.classification_diagnostics_executor as diagnostics_executor


from app.ml.classification_diagnostics_executor import (
    ML_CLASSIFICATION_DIAGNOSTICS_EXECUTOR_RULE_VERSION,
    execute_ml_classification_diagnostics,
)


from app.ml.contracts import (
    MLPreprocessingContract,
    MLSplitContract,
    MLTrainingContract,
)


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


from app.ml.model_loader import (
    load_trusted_ml_model,
)


from app.persistence.sqlite_database import (
    sqlite_connection,
)


from app.preparation.analysis_readiness_gate import (
    require_analysis_readiness,
)


# ============================================================
# VERSION
# ============================================================


ML_CLASSIFICATION_DIAGNOSTICS_GOLDEN_PATH_RULE_VERSION = (
    "ml_classification_diagnostics_golden_path_v0.1"
)


# ============================================================
# MODEL ARTIFACT COUNT
# ============================================================


def ml_model_artifact_count(
    *,
    workflow_id: str,
) -> int:

    with sqlite_connection(
        write=False
    ) as connection:

        row = (
            connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM ml_model_artifacts

                WHERE
                    workflow_id = ?
                """,
                (
                    workflow_id,
                ),
            )
            .fetchone()
        )


    assert (
        row
        is not None
    )


    return int(
        row[
            "count"
        ]
    )


# ============================================================
# CLASSIFICATION TRAINING CONTRACT
# ============================================================


def build_real_classification_contract(
    *,
    workflow_id: str,
) -> MLTrainingContract:
    """
    Reuse the real 30-row mixed Preparation artifact.

    Dataset:
    - age: numeric
    - tenure: numeric
    - segment: classification target
    - revenue: numeric

    Target:
        segment = standard / premium

    Features:
        age / tenure / revenue

    revenue contains the existing segment effect in this Golden
    dataset, making this a deterministic and intentionally simple
    classification integration fixture.

    The purpose here is not model difficulty. It is proving the
    complete trusted diagnostics boundary.
    """

    return (
        MLTrainingContract(
            workflow_id=
                workflow_id,

            dataset_id=
                WORKFLOW_ROOT_DATASET_ID,

            problem_type=
                "classification",

            target_column=
                "segment",

            feature_columns=[
                "age",
                "tenure",
                "revenue",
            ],

            categorical_feature_columns=[],

            estimator_key=
                "logistic_regression",

            preprocessing=(
                MLPreprocessingContract(
                    numeric_imputation=
                        "error",

                    categorical_imputation=
                        "error",

                    categorical_encoding=
                        "one_hot",

                    handle_unknown_categories=
                        "ignore",

                    scale_numeric=
                        True,
                )
            ),

            split=(
                MLSplitContract(
                    test_size=
                        0.20,

                    random_seed=
                        31,

                    shuffle=
                        True,

                    stratify=
                        True,
                )
            ),
        )
    )


# ============================================================
# REAL TRAINING
# ============================================================


def train_real_classifier(
    *,
    workflow_id: str,
    training_contract: MLTrainingContract,
):

    artifacts_before = (
        ml_model_artifact_count(
            workflow_id=
                workflow_id
        )
    )


    assert (
        artifacts_before
        ==
        0
    )


    readiness_before = (
        require_analysis_readiness(
            workflow_id=
                workflow_id
        )
    )


    result = (
        execute_classical_ml(
            training_contract=
                training_contract
        )
    )


    readiness_after = (
        require_analysis_readiness(
            workflow_id=
                workflow_id
        )
    )


    artifacts_after = (
        ml_model_artifact_count(
            workflow_id=
                workflow_id
        )
    )


    assert (
        artifacts_after
        ==
        1
    )


    assert (
        readiness_before
        .session_revision
        ==
        readiness_after
        .session_revision
    )


    assert (
        result.problem_type
        ==
        "classification"
    )


    assert (
        result.estimator_key
        ==
        "logistic_regression"
    )


    assert (
        result.train_rows
        ==
        24
    )


    assert (
        result.test_rows
        ==
        6
    )


    assert (
        result.model_artifact
        .training_contract
        ==
        training_contract
    )


    assert (
        result.experiment_provenance
        ==
        result.model_artifact
        .experiment_provenance
    )


    assert (
        result.experiment_provenance
        .preparation_session_revision
        ==
        readiness_before
        .session_revision
    )


    assert (
        result.experiment_provenance
        .training_contract_sha256
        ==
        ml_training_contract_sha256(
            training_contract
        )
    )


    expected_metric_names = {
        "accuracy",
        "f1_macro",
        "precision_macro",
        "recall_macro",
        "balanced_accuracy",
    }


    assert (
        set(
            result.metrics
        )
        ==
        expected_metric_names
    )


    for metric_value in (
        result.metrics.values()
    ):

        assert math.isfinite(
            float(
                metric_value
            )
        )


    print(
        (
            "[PASS] real validated Preparation "
            "trained one LogisticRegression classifier"
        )
    )


    print(
        (
            "[PASS] real classification training "
            "persisted exactly one Model Artifact"
        )
    )


    print(
        (
            "[PASS] classifier persisted five finite "
            "classification metrics + provenance"
        )
    )


    return result


# ============================================================
# TRUSTED RELOAD
# ============================================================


def verify_real_classifier_reload(
    *,
    workflow_id: str,
    execution_result,
):

    loaded = (
        load_trusted_ml_model(
            workflow_id=
                workflow_id,

            model_id=
                execution_result
                .model_artifact
                .model_id,
        )
    )


    assert (
        loaded.artifact
        .model_id
        ==
        execution_result
        .model_artifact
        .model_id
    )


    assert (
        loaded.artifact
        .training_contract
        ==
        execution_result
        .model_artifact
        .training_contract
    )


    assert isinstance(
        loaded.estimator,
        Pipeline,
    )


    assert (
        "estimator"
        in
        loaded.estimator
        .named_steps
    )


    classifier = (
        loaded.estimator
        .named_steps[
            "estimator"
        ]
    )


    assert isinstance(
        classifier,
        LogisticRegression,
    )


    assert (
        hasattr(
            classifier,
            "classes_",
        )
    )


    assert (
        set(
            str(
                value
            )

            for value
            in classifier.classes_
        )
        ==
        {
            "premium",
            "standard",
        }
    )


    print(
        (
            "[PASS] classification Model Artifact "
            "crossed trusted SHA-verified reload"
        )
    )


    print(
        (
            "[PASS] trusted reload restored fitted "
            "LogisticRegression classes_"
        )
    )


    return loaded


# ============================================================
# INDEPENDENT REAL HOLDOUT RECONSTRUCTION
# ============================================================


def reconstruct_real_holdout(
    *,
    training_contract: MLTrainingContract,
):

    (
        dataframe,
        preparation_revision,
    ) = (
        _load_authorized_dataframe(
            contract=
                training_contract
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
                training_contract,
        )
    )


    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = (
        _split_dataset(
            x=
                x,

            y=
                y,

            contract=
                training_contract,
        )
    )


    assert (
        len(
            x_train
        )
        ==
        24
    )


    assert (
        len(
            x_test
        )
        ==
        6
    )


    assert (
        len(
            y_train
        )
        ==
        24
    )


    assert (
        len(
            y_test
        )
        ==
        6
    )


    assert (
        set(
            str(
                value
            )

            for value
            in y_test.unique()
        )
        ==
        {
            "premium",
            "standard",
        }
    )


    print(
        (
            "[PASS] real deterministic stratified "
            "holdout reconstructed exact 24/6 split"
        )
    )


    return (
        preparation_revision,
        x_test,
        y_test,
    )


# ============================================================
# REAL DIAGNOSTICS
# ============================================================


def run_real_diagnostics(
    *,
    workflow_id: str,
    execution_result,
):

    artifact_count_before = (
        ml_model_artifact_count(
            workflow_id=
                workflow_id
        )
    )


    assert (
        artifact_count_before
        ==
        1
    )


    readiness_before = (
        require_analysis_readiness(
            workflow_id=
                workflow_id
        )
    )


    real_loader = (
        diagnostics_executor
        .load_trusted_ml_model
    )


    guarded_models = []


    def guarded_loader(
        **kwargs,
    ):

        loaded = (
            real_loader(
                **kwargs
            )
        )


        original_fit = (
            loaded.estimator.fit
        )


        def forbidden_fit(
            *args,
            **fit_kwargs,
        ):

            raise AssertionError(
                (
                    "Classification Diagnostics "
                    "must never fit or refit the "
                    "trusted persisted estimator."
                )
            )


        loaded.estimator.fit = (
            forbidden_fit
        )


        guarded_models.append(
            (
                loaded,
                original_fit,
            )
        )


        return loaded


    diagnostics_executor.load_trusted_ml_model = (
        guarded_loader
    )


    try:

        result = (
            execute_ml_classification_diagnostics(
                workflow_id=
                    workflow_id,

                model_id=(
                    execution_result
                    .model_artifact
                    .model_id
                ),

                diagnostics_contract=(
                    MLClassificationDiagnosticsContract()
                ),
            )
        )

    finally:

        diagnostics_executor.load_trusted_ml_model = (
            real_loader
        )


    readiness_after = (
        require_analysis_readiness(
            workflow_id=
                workflow_id
        )
    )


    artifact_count_after = (
        ml_model_artifact_count(
            workflow_id=
                workflow_id
        )
    )


    assert (
        len(
            guarded_models
        )
        ==
        1
    )


    assert (
        artifact_count_after
        ==
        artifact_count_before
        ==
        1
    )


    assert (
        readiness_after
        .session_revision
        ==
        readiness_before
        .session_revision
    )


    print(
        (
            "[PASS] diagnostics crossed real trusted "
            "Model Artifact reload"
        )
    )


    print(
        (
            "[PASS] real diagnostics completed with "
            "fit() explicitly forbidden"
        )
    )


    print(
        (
            "[PASS] diagnostics created zero additional "
            "Model Artifacts / Experiments"
        )
    )


    print(
        (
            "[PASS] diagnostics did not mutate "
            "Preparation revision"
        )
    )


    return result


# ============================================================
# RESULT AUTHORITY
# ============================================================


def verify_real_result_authority(
    *,
    workflow_id: str,
    training_contract: MLTrainingContract,
    execution_result,
    diagnostics_result,
) -> None:

    provenance = (
        execution_result
        .experiment_provenance
    )


    assert (
        diagnostics_result.workflow_id
        ==
        workflow_id
    )


    assert (
        diagnostics_result.dataset_id
        ==
        WORKFLOW_ROOT_DATASET_ID
    )


    assert (
        diagnostics_result.model_id
        ==
        execution_result
        .model_artifact
        .model_id
    )


    assert (
        diagnostics_result.experiment_id
        ==
        provenance
        .experiment_id
    )


    assert (
        diagnostics_result.problem_type
        ==
        "classification"
    )


    assert (
        diagnostics_result.target_column
        ==
        "segment"
    )


    assert (
        diagnostics_result.estimator_key
        ==
        "logistic_regression"
    )


    assert (
        diagnostics_result
        .preparation_session_revision
        ==
        provenance
        .preparation_session_revision
    )


    assert (
        diagnostics_result
        .training_contract_sha256
        ==
        ml_training_contract_sha256(
            training_contract
        )
    )


    assert (
        diagnostics_result.method
        ==
        "holdout_classification_diagnostics"
    )


    assert (
        diagnostics_result
        .label_order_policy
        ==
        "estimator_classes"
    )


    assert (
        diagnostics_result
        .zero_division_policy
        ==
        "zero"
    )


    print(
        (
            "[PASS] diagnostics result is bound to "
            "Model Artifact + Experiment Provenance + "
            "Training Contract SHA-256"
        )
    )


# ============================================================
# MATRIX + REAL HOLDOUT
# ============================================================


def verify_real_matrix(
    *,
    training_contract: MLTrainingContract,
    execution_result,
    loaded_model,
    diagnostics_result,
) -> None:

    (
        preparation_revision,
        x_test,
        y_test,
    ) = (
        reconstruct_real_holdout(
            training_contract=
                training_contract
        )
    )


    assert (
        preparation_revision
        ==
        diagnostics_result
        .preparation_session_revision
    )


    predictions = (
        loaded_model.predict(
            x_test
        )
    )


    classifier = (
        loaded_model
        .estimator
        .named_steps[
            "estimator"
        ]
    )


    raw_classes = [
        value.item()
        if isinstance(
            value,
            np.generic,
        )
        else value

        for value
        in classifier
        .classes_
        .tolist()
    ]


    expected_labels = [
        str(
            value
        ).strip()

        for value
        in raw_classes
    ]


    expected_matrix = (
        confusion_matrix(
            y_test,
            predictions,
            labels=
                raw_classes,
        )
        .astype(
            np.int64
        )
        .tolist()
    )


    assert (
        diagnostics_result
        .evaluation_rows
        ==
        6
    )


    assert (
        diagnostics_result
        .class_count
        ==
        2
    )


    assert (
        diagnostics_result
        .class_labels
        ==
        expected_labels
    )


    assert (
        diagnostics_result
        .confusion_matrix
        ==
        expected_matrix
    )


    assert (
        sum(
            sum(
                row
            )

            for row
            in diagnostics_result
            .confusion_matrix
        )
        ==
        6
    )


    assert (
        [
            item.class_label

            for item
            in diagnostics_result
            .per_class
        ]
        ==
        diagnostics_result
        .class_labels
    )


    assert (
        sum(
            item.support

            for item
            in diagnostics_result
            .per_class
        )
        ==
        6
    )


    for item in (
        diagnostics_result
        .per_class
    ):

        assert (
            item.support
            >
            0
        )


        assert (
            (
                item.true_positive
                +
                item.false_positive
                +
                item.false_negative
                +
                item.true_negative
            )
            ==
            6
        )


        for metric_value in (
            item.precision,
            item.recall,
            item.f1,
        ):

            assert (
                0.0
                <=
                metric_value
                <=
                1.0
            )


    print(
        (
            "[PASS] real confusion matrix uses fitted "
            "estimator classes_ ordering"
        )
    )


    print(
        (
            "[PASS] confusion matrix covers all six "
            "untouched holdout rows"
        )
    )


    print(
        (
            "[PASS] per-class TP / FP / FN / TN and "
            "support cover the real holdout"
        )
    )


# ============================================================
# PERSISTED METRIC CONSISTENCY
# ============================================================


def verify_real_metric_binding(
    *,
    execution_result,
    diagnostics_result,
) -> None:

    persisted = (
        execution_result.metrics
    )


    assert math.isclose(
        diagnostics_result.accuracy,
        persisted[
            "accuracy"
        ],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        diagnostics_result
        .macro_average
        .f1,
        persisted[
            "f1_macro"
        ],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        diagnostics_result
        .macro_average
        .precision,
        persisted[
            "precision_macro"
        ],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        diagnostics_result
        .macro_average
        .recall,
        persisted[
            "recall_macro"
        ],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        diagnostics_result
        .balanced_accuracy,
        persisted[
            "balanced_accuracy"
        ],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert (
        0.0
        <=
        diagnostics_result
        .weighted_average
        .precision
        <=
        1.0
    )


    assert (
        0.0
        <=
        diagnostics_result
        .weighted_average
        .recall
        <=
        1.0
    )


    assert (
        0.0
        <=
        diagnostics_result
        .weighted_average
        .f1
        <=
        1.0
    )


    print(
        (
            "[PASS] reconstructed diagnostics exactly "
            "match all five persisted classification metrics"
        )
    )


    print(
        (
            "[PASS] weighted precision / recall / F1 "
            "are finite holdout-only diagnostics"
        )
    )


# ============================================================
# DETERMINISTIC REPEAT
# ============================================================


def verify_real_repeat_is_deterministic(
    *,
    workflow_id: str,
    execution_result,
    first_result,
) -> None:

    artifacts_before = (
        ml_model_artifact_count(
            workflow_id=
                workflow_id
        )
    )


    second_result = (
        execute_ml_classification_diagnostics(
            workflow_id=
                workflow_id,

            model_id=(
                execution_result
                .model_artifact
                .model_id
            ),

            diagnostics_contract=(
                MLClassificationDiagnosticsContract()
            ),
        )
    )


    artifacts_after = (
        ml_model_artifact_count(
            workflow_id=
                workflow_id
        )
    )


    assert (
        first_result.model_dump(
            mode="json"
        )
        ==
        second_result.model_dump(
            mode="json"
        )
    )


    assert (
        artifacts_after
        ==
        artifacts_before
        ==
        1
    )


    print(
        (
            "[PASS] repeated real diagnostics are "
            "exactly deterministic"
        )
    )


    print(
        (
            "[PASS] deterministic repeat persisted "
            "zero additional Model Artifacts"
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


def verify_privacy_minimal(
    *,
    diagnostics_result,
) -> None:

    payload = (
        diagnostics_result
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


    print(
        (
            "[PASS] real diagnostics result remains "
            "privacy-minimal"
        )
    )


# ============================================================
# RULE VERSIONS
# ============================================================


def verify_rule_versions(
) -> None:

    assert (
        ML_CLASSIFICATION_DIAGNOSTICS_RULE_VERSION
        ==
        "ml_classification_diagnostics_v0.1"
    )


    assert (
        ML_CLASSIFICATION_DIAGNOSTICS_EXECUTOR_RULE_VERSION
        ==
        "ml_classification_diagnostics_executor_v0.1"
    )


    assert (
        ML_CLASSIFICATION_DIAGNOSTICS_GOLDEN_PATH_RULE_VERSION
        ==
        "ml_classification_diagnostics_golden_path_v0.1"
    )


    print(
        "[PASS] Classification Diagnostics rule versions"
    )


# ============================================================
# GOLDEN PATH
# ============================================================


def test_ml_classification_diagnostics_golden_path_v0_1(
) -> None:

    reset_product_state()


    with TestClient(
        app
    ) as client:

        # ----------------------------------------------------
        # REAL PREPARATION
        # ----------------------------------------------------

        workflow_id = (
            create_preparation_session(
                client
            )
        )


        run_real_quality(
            client,
            workflow_id=
                workflow_id,
        )


        run_real_cleaning_plan(
            client,
            workflow_id=
                workflow_id,
        )


        select_analysis_output(
            client,
            workflow_id=
                workflow_id,
        )


        validate_preparation(
            client,
            workflow_id=
                workflow_id,
        )


        verify_preparation_persistence(
            workflow_id=
                workflow_id,
        )


        verify_real_handoff(
            workflow_id=
                workflow_id,
        )


        # ----------------------------------------------------
        # REAL CLASSIFICATION TRAINING
        # ----------------------------------------------------

        training_contract = (
            build_real_classification_contract(
                workflow_id=
                    workflow_id
            )
        )


        execution_result = (
            train_real_classifier(
                workflow_id=
                    workflow_id,

                training_contract=
                    training_contract,
            )
        )


        # ----------------------------------------------------
        # REAL TRUSTED RELOAD
        # ----------------------------------------------------

        loaded_model = (
            verify_real_classifier_reload(
                workflow_id=
                    workflow_id,

                execution_result=
                    execution_result,
            )
        )


        # ----------------------------------------------------
        # REAL CLASSIFICATION DIAGNOSTICS
        # ----------------------------------------------------

        diagnostics_result = (
            run_real_diagnostics(
                workflow_id=
                    workflow_id,

                execution_result=
                    execution_result,
            )
        )


        verify_real_result_authority(
            workflow_id=
                workflow_id,

            training_contract=
                training_contract,

            execution_result=
                execution_result,

            diagnostics_result=
                diagnostics_result,
        )


        verify_real_matrix(
            training_contract=
                training_contract,

            execution_result=
                execution_result,

            loaded_model=
                loaded_model,

            diagnostics_result=
                diagnostics_result,
        )


        verify_real_metric_binding(
            execution_result=
                execution_result,

            diagnostics_result=
                diagnostics_result,
        )


        verify_real_repeat_is_deterministic(
            workflow_id=
                workflow_id,

            execution_result=
                execution_result,

            first_result=
                diagnostics_result,
        )


        verify_privacy_minimal(
            diagnostics_result=
                diagnostics_result
        )


        verify_rule_versions()


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print()


    print(
        "="
        *
        78
    )


    print(
        "DATALENS ML CLASSIFICATION DIAGNOSTICS GOLDEN PATH E2E v0.1"
    )


    print(
        "="
        *
        78
    )


    print(
        "Preparation : real validated 30-row mixed CSV"
    )


    print(
        "Target      : segment = premium / standard"
    )


    print(
        "Estimator   : real LogisticRegression"
    )


    print(
        "Holdout     : deterministic stratified 24/6"
    )


    print(
        "Persistence : exactly one Model Artifact + Experiment"
    )


    print(
        "Reload      : trusted SHA-verified joblib boundary"
    )


    print(
        "Diagnostics : confusion matrix + per-class TP/FP/FN/TN"
    )


    print(
        "Metrics     : persisted global + macro + weighted"
    )


    print(
        "Safety      : no refit / no additional persistence"
    )


    print()


    test_ml_classification_diagnostics_golden_path_v0_1()


    print()


    print(
        "="
        *
        78
    )


    print(
        (
            "PASS - Preparation -> Classification Training -> "
            "1 Model Artifact + Experiment Provenance -> "
            "Trusted Reload -> Exact 24/6 Holdout -> "
            "Confusion Matrix + Per-Class Diagnostics -> "
            "Persisted Metric Binding -> No Refit -> "
            "No Additional Persistence"
        )
    )


    print(
        "="
        *
        78
    )


if __name__ == "__main__":
    main()
