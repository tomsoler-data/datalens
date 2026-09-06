from __future__ import annotations


import io
import math


from typing import (
    Any,
    Literal,
)


import joblib
import numpy as np
import pandas as pd


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)


from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
)


from sklearn.pipeline import (
    Pipeline,
)


from app.ml.baseline import (
    MLBaselineComparisonResult,
    MLBaselineError,
    MLBaselineEvaluationResult,
    build_ml_baseline_evaluation,
    build_ml_baseline_predictions,
    compare_model_to_baseline,
)


from app.ml.contracts import (
    MLGroupHoldoutSplitContract,
    MLPurgedGroupTimeHoldoutSplitContract,
    MLSplitContract,
    MLTimeHoldoutSplitContract,
    MLTrainingContract,
)


from app.ml.training_input import (
    MLTrainingInputError,
    load_authorized_ml_dataframe,
    validate_and_extract_ml_xy,
)


from app.ml.splitting import (
    MLSplitInputError,
    MLSplitInvariantError,
    chronological_holdout_positions,
    split_ml_dataset,
    validated_group_values,
    validated_time_values,
)


from app.ml.experiment_provenance import (
    MLExperimentProvenanceRecord,
)


from app.ml.estimator_contracts import (
    MLLinearRegressionHyperparameters,
    MLLogisticRegressionHyperparameters,
    MLRandomForestClassifierHyperparameters,
    MLRandomForestRegressorHyperparameters,
    MLRidgeRegressionHyperparameters,
    estimator_problem_type,
)


from app.ml.preprocessing import (
    MLPreprocessingRuntimeError,
    build_ml_preprocessor,
)


from app.ml.model_artifact_store import (
    MLModelArtifactStoreError,
    delete_ml_model_artifact,
    register_ml_model_artifact,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.ml.model_metrics import (
    MLModelMetricsError,
    compute_ml_classification_metrics,
    compute_ml_regression_metrics,
    project_ml_baseline_metrics_v0_1,
)


from app.ml.monitoring_profile_builder import (
    build_ml_monitoring_profile,
)


from app.ml.monitoring_profile_store import (
    register_ml_monitoring_profile,
)


from app.preparation.analysis_input_handoff import (
    load_validated_analysis_input,
)


# ============================================================
# VERSION
# ============================================================


CLASSICAL_ML_EXECUTOR_RULE_VERSION = (
    "classical_ml_executor_v0.1"
)


ML_RICHER_METRICS_RULE_VERSION = (
    "ml_richer_metrics_v0.1"
)


ML_MONITORING_TRAINING_INTEGRATION_RULE_VERSION = (
    "ml_monitoring_training_integration_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class ClassicalMLExecutorError(
    RuntimeError
):
    pass


class ClassicalMLInputError(
    ClassicalMLExecutorError
):
    pass


class ClassicalMLEstimatorError(
    ClassicalMLExecutorError
):
    pass


# ============================================================
# RESULT
# ============================================================


class ClassicalMLExecutionResult(
    BaseModel
):
    """
    Privacy-minimal execution result.

    Predictions and raw train/test rows are deliberately not
    persisted in this result contract.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    workflow_id: str = Field(
        min_length=1,
    )


    dataset_id: str = Field(
        min_length=1,
    )


    problem_type: Literal[
        "regression",
        "classification",
    ]


    estimator_key: str = Field(
        min_length=1,
    )


    train_rows: int = Field(
        gt=0,
    )


    test_rows: int = Field(
        gt=0,
    )


    purged_rows: int = Field(
        default=0,
        ge=0,
    )


    metrics: dict[
        str,
        float,
    ]


    baseline: (
        MLBaselineEvaluationResult
    )


    baseline_comparison: (
        MLBaselineComparisonResult
    )


    experiment_provenance: (
        MLExperimentProvenanceRecord
    )


    model_artifact: MLModelArtifactRecord


    rule_version: Literal[
        "classical_ml_executor_v0.1"
    ] = CLASSICAL_ML_EXECUTOR_RULE_VERSION


# ============================================================
# DATASET RESOLUTION
# ============================================================


def _load_authorized_dataframe(
    *,
    contract: MLTrainingContract,
) -> tuple[
    pd.DataFrame,
    int,
]:

    try:
        return (
            load_authorized_ml_dataframe(
                contract=contract,
                handoff_loader=
                    load_validated_analysis_input,
                execution_label=
                    "Classical ML",
            )
        )

    except MLTrainingInputError as error:

        raise (
            ClassicalMLInputError(
                str(
                    error
                )
            )
        ) from error


# ============================================================
# SCHEMA VALIDATION
# ============================================================


def _validate_and_extract_xy(
    *,
    dataframe: pd.DataFrame,
    contract: MLTrainingContract,
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:

    try:
        return (
            validate_and_extract_ml_xy(
                dataframe=dataframe,
                contract=contract,
                execution_label=
                    "Classical ML",
            )
        )

    except MLTrainingInputError as error:

        raise (
            ClassicalMLInputError(
                str(
                    error
                )
            )
        ) from error


# ============================================================
# ESTIMATOR
# ============================================================


def _build_estimator(
    *,
    contract: MLTrainingContract,
) -> Pipeline:

    estimator_key = (
        contract.estimator_key
    )


    expected_problem_type = (
        estimator_problem_type(
            estimator_key
        )
    )


    if (
        expected_problem_type
        is None
    ):
        raise (
            ClassicalMLEstimatorError(
                (
                    "Unsupported Classical ML "
                    "estimator. "
                    f"estimator_key={estimator_key}"
                )
            )
        )


    if (
        expected_problem_type
        !=
        contract.problem_type
    ):
        raise (
            ClassicalMLEstimatorError(
                (
                    "Estimator/problem type mismatch. "
                    f"estimator_key={estimator_key}, "
                    "estimator_problem_type="
                    f"{expected_problem_type}, "
                    "contract_problem_type="
                    f"{contract.problem_type}"
                )
            )
        )


    hyperparameters = (
        contract
        .effective_estimator_hyperparameters
    )


    if (
        hyperparameters
        is None
    ):
        raise (
            ClassicalMLEstimatorError(
                (
                    "No server-validatable "
                    "hyperparameter contract is "
                    "available for estimator. "
                    f"estimator_key={estimator_key}"
                )
            )
        )


    # ========================================================
    # LINEAR REGRESSION
    # ========================================================


    if (
        estimator_key
        ==
        "linear_regression"
    ):
        if not isinstance(
            hyperparameters,
            MLLinearRegressionHyperparameters,
        ):
            raise (
                ClassicalMLEstimatorError(
                    (
                        "Linear Regression received "
                        "an incompatible estimator "
                        "hyperparameter contract."
                    )
                )
            )


        estimator = (
            LinearRegression(
                fit_intercept=
                    hyperparameters
                    .fit_intercept,
            )
        )


    # ========================================================
    # RIDGE REGRESSION
    # ========================================================


    elif (
        estimator_key
        ==
        "ridge_regression"
    ):
        if not isinstance(
            hyperparameters,
            MLRidgeRegressionHyperparameters,
        ):
            raise (
                ClassicalMLEstimatorError(
                    (
                        "Ridge Regression received "
                        "an incompatible estimator "
                        "hyperparameter contract."
                    )
                )
            )


        estimator = (
            Ridge(
                alpha=
                    hyperparameters
                    .alpha,

                fit_intercept=
                    hyperparameters
                    .fit_intercept,
            )
        )


    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================


    elif (
        estimator_key
        ==
        "logistic_regression"
    ):
        if not isinstance(
            hyperparameters,
            MLLogisticRegressionHyperparameters,
        ):
            raise (
                ClassicalMLEstimatorError(
                    (
                        "Logistic Regression received "
                        "an incompatible estimator "
                        "hyperparameter contract."
                    )
                )
            )


        estimator = (
            LogisticRegression(
                C=(
                    hyperparameters
                    .inverse_regularization_strength
                ),

                fit_intercept=(
                    hyperparameters
                    .fit_intercept
                ),

                max_iter=(
                    hyperparameters
                    .max_iter
                ),

                class_weight=(
                    hyperparameters
                    .class_weight
                ),

                solver="lbfgs",

                random_state=(
                    contract
                    .split
                    .random_seed
                ),
            )
        )


    # ========================================================
    # RANDOM FOREST REGRESSOR
    # ========================================================


    elif (
        estimator_key
        ==
        "random_forest_regressor"
    ):
        if not isinstance(
            hyperparameters,
            MLRandomForestRegressorHyperparameters,
        ):
            raise (
                ClassicalMLEstimatorError(
                    (
                        "Random Forest Regressor "
                        "received an incompatible "
                        "estimator hyperparameter "
                        "contract."
                    )
                )
            )


        estimator = (
            RandomForestRegressor(
                n_estimators=(
                    hyperparameters
                    .n_estimators
                ),

                max_depth=(
                    hyperparameters
                    .max_depth
                ),

                min_samples_split=(
                    hyperparameters
                    .min_samples_split
                ),

                min_samples_leaf=(
                    hyperparameters
                    .min_samples_leaf
                ),

                max_features=(
                    hyperparameters
                    .max_features
                ),

                bootstrap=(
                    hyperparameters
                    .bootstrap
                ),

                # Server-owned execution controls.
                random_state=(
                    contract
                    .split
                    .random_seed
                ),

                n_jobs=1,
            )
        )


    # ========================================================
    # RANDOM FOREST CLASSIFIER
    # ========================================================


    elif (
        estimator_key
        ==
        "random_forest_classifier"
    ):
        if not isinstance(
            hyperparameters,
            MLRandomForestClassifierHyperparameters,
        ):
            raise (
                ClassicalMLEstimatorError(
                    (
                        "Random Forest Classifier "
                        "received an incompatible "
                        "estimator hyperparameter "
                        "contract."
                    )
                )
            )


        estimator = (
            RandomForestClassifier(
                n_estimators=(
                    hyperparameters
                    .n_estimators
                ),

                max_depth=(
                    hyperparameters
                    .max_depth
                ),

                min_samples_split=(
                    hyperparameters
                    .min_samples_split
                ),

                min_samples_leaf=(
                    hyperparameters
                    .min_samples_leaf
                ),

                max_features=(
                    hyperparameters
                    .max_features
                ),

                bootstrap=(
                    hyperparameters
                    .bootstrap
                ),

                class_weight=(
                    hyperparameters
                    .class_weight
                ),

                # Server-owned execution controls.
                random_state=(
                    contract
                    .split
                    .random_seed
                ),

                n_jobs=1,
            )
        )


    else:
        raise (
            ClassicalMLEstimatorError(
                (
                    "Unsupported Classical ML "
                    "estimator. "
                    f"estimator_key={estimator_key}"
                )
            )
        )


    # ========================================================
    # LEAKAGE-SAFE PREPROCESSING
    # ========================================================


    try:
        preprocessor = (
            build_ml_preprocessor(
                contract=contract
            )
        )

    except MLPreprocessingRuntimeError as error:
        raise (
            ClassicalMLEstimatorError(
                (
                    "Classical ML preprocessing "
                    "pipeline could not be built."
                )
            )
        ) from error


    return (
        Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "estimator",
                    estimator,
                ),
            ]
        )
    )


# ============================================================
# SHARED SPLIT COMPATIBILITY WRAPPERS
# ============================================================


def _raise_classical_split_error(
    error: Exception,
) -> None:

    if isinstance(
        error,
        MLSplitInputError,
    ):
        raise (
            ClassicalMLInputError(
                str(
                    error
                )
            )
        ) from error


    if isinstance(
        error,
        MLSplitInvariantError,
    ):
        raise (
            ClassicalMLExecutorError(
                str(
                    error
                )
            )
        ) from error


    raise error


def _validated_group_values(
    *,
    dataframe: pd.DataFrame | None,
    x: pd.DataFrame,
    y: pd.Series,
    contract: MLTrainingContract,
) -> pd.Series:

    try:
        return (
            validated_group_values(
                dataframe=dataframe,
                x=x,
                y=y,
                contract=contract,
            )
        )

    except (
        MLSplitInputError,
        MLSplitInvariantError,
    ) as error:

        _raise_classical_split_error(
            error
        )

        raise AssertionError(
            "unreachable"
        )


def _validated_time_values(
    *,
    dataframe: pd.DataFrame | None,
    x: pd.DataFrame,
    y: pd.Series,
    contract: MLTrainingContract,
) -> pd.Series:

    try:
        return (
            validated_time_values(
                dataframe=dataframe,
                x=x,
                y=y,
                contract=contract,
            )
        )

    except (
        MLSplitInputError,
        MLSplitInvariantError,
    ) as error:

        _raise_classical_split_error(
            error
        )

        raise AssertionError(
            "unreachable"
        )


def _chronological_holdout_positions(
    *,
    time_values: pd.Series,
    test_size: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    try:
        return (
            chronological_holdout_positions(
                time_values=time_values,
                test_size=test_size,
            )
        )

    except (
        MLSplitInputError,
        MLSplitInvariantError,
    ) as error:

        _raise_classical_split_error(
            error
        )

        raise AssertionError(
            "unreachable"
        )


def _split_dataset(
    *,
    x: pd.DataFrame,
    y: pd.Series,
    contract: MLTrainingContract,
    dataframe: pd.DataFrame | None = None,
    return_group_partitions: bool = False,
    return_time_partitions: bool = False,
):

    try:
        return (
            split_ml_dataset(
                x=x,
                y=y,
                contract=contract,
                dataframe=dataframe,
                return_group_partitions=
                    return_group_partitions,
                return_time_partitions=
                    return_time_partitions,
            )
        )

    except (
        MLSplitInputError,
        MLSplitInvariantError,
    ) as error:

        _raise_classical_split_error(
            error
        )

        raise AssertionError(
            "unreachable"
        )


# ============================================================
# METRICS
# ============================================================


def _regression_metrics(
    *,
    y_true: pd.Series,
    predictions: Any,
) -> dict[
    str,
    float,
]:
    """
    Backward-compatible private alias.

    New production code must import the canonical public helper
    from app.ml.model_metrics.
    """

    return (
        compute_ml_regression_metrics(
            y_true=
                y_true,

            predictions=
                predictions,
        )
    )


def _classification_metrics(
    *,
    y_true: pd.Series,
    predictions: Any,
) -> dict[
    str,
    float,
]:
    """
    Backward-compatible private alias.

    New production code must import the canonical public helper
    from app.ml.model_metrics.
    """

    return (
        compute_ml_classification_metrics(
            y_true=
                y_true,

            predictions=
                predictions,
        )
    )


def _baseline_metrics_v0_1(
    *,
    problem_type: str,
    metrics: dict[
        str,
        float,
    ],
) -> dict[
    str,
    float,
]:
    """
    Backward-compatible private alias preserving the historical
    ClassicalMLExecutorError boundary.
    """

    try:
        return (
            project_ml_baseline_metrics_v0_1(
                problem_type=
                    problem_type,

                metrics=
                    metrics,
            )
        )

    except MLModelMetricsError as error:
        raise ClassicalMLExecutorError(
            str(
                error
            )
        ) from error


def _validate_metrics(
    metrics: dict[
        str,
        float,
    ],
) -> dict[
    str,
    float,
]:

    if not metrics:
        raise (
            ClassicalMLExecutorError(
                (
                    "Classical ML execution "
                    "produced no metrics."
                )
            )
        )


    normalized = {}


    for (
        metric_name,
        raw_value,
    ) in metrics.items():

        value = float(
            raw_value
        )


        if not (
            math.isfinite(
                value
            )
        ):
            raise (
                ClassicalMLExecutorError(
                    (
                        "Classical ML execution "
                        "produced a non-finite "
                        "metric. "
                        f"metric={metric_name}"
                    )
                )
            )


        normalized[
            str(
                metric_name
            )
        ] = value


    return normalized


# ============================================================
# SERIALIZATION
# ============================================================


def _serialize_fitted_estimator(
    estimator: Pipeline,
) -> bytes:
    """
    Serialize only a model that DataLens has just fitted.

    This function performs joblib.dump only.

    It never performs joblib.load and never consumes a
    user-supplied model file.
    """

    buffer = (
        io.BytesIO()
    )


    try:
        joblib.dump(
            estimator,
            buffer,
            compress=0,
            protocol=5,
        )

    except Exception as error:
        raise (
            ClassicalMLExecutorError(
                (
                    "Fitted Classical ML estimator "
                    "could not be serialized."
                )
            )
        ) from error


    model_bytes = (
        buffer.getvalue()
    )


    if not model_bytes:
        raise (
            ClassicalMLExecutorError(
                (
                    "Fitted Classical ML estimator "
                    "serialization produced an "
                    "empty payload."
                )
            )
        )


    return model_bytes


# ============================================================
# EXECUTION
# ============================================================


def execute_classical_ml(
    *,
    training_contract: MLTrainingContract,
    expected_preparation_session_revision: int | None = None,
) -> ClassicalMLExecutionResult:
    """
    Execute one deterministic Classical ML training request.

    Authority flow:

        MLTrainingContract
                ↓
        validated Preparation handoff
                ↓
        final server-owned DataFrame
                ↓
        schema / leakage guards
                ↓
        deterministic holdout split
                ↓
        scikit-learn Pipeline
                ↓
        metrics
                ↓
        server-owned Model Artifact Store

    No raw rows or predictions are persisted in the result.

    expected_preparation_session_revision is an optional
    server-owned execution guard.

    When supplied, Classical ML refuses to begin schema
    validation, splitting or fitting if the validated
    Preparation handoff revision no longer matches the
    expected revision.

    This is used by Tuned Model Promotion to prevent a
    tuning result computed on one Preparation snapshot from
    being promoted against a newer snapshot.
    """

    contract = (
        MLTrainingContract
        .model_validate(
            training_contract
        )
    )


    (
        dataframe,
        preparation_session_revision,
    ) = (
        _load_authorized_dataframe(
            contract=
                contract
        )
    )


    # ========================================================
    # OPTIONAL SERVER-OWNED PREPARATION REVISION PIN
    # ========================================================


    if (
        expected_preparation_session_revision
        is not None
    ):
        if (
            isinstance(
                expected_preparation_session_revision,
                bool,
            )
            or
            not isinstance(
                expected_preparation_session_revision,
                int,
            )
            or
            expected_preparation_session_revision
            <
            0
        ):
            raise (
                ClassicalMLInputError(
                    (
                        "Expected Preparation session "
                        "revision must be a non-negative "
                        "integer."
                    )
                )
            )


        if (
            int(
                preparation_session_revision
            )
            !=
            expected_preparation_session_revision
        ):
            raise (
                ClassicalMLInputError(
                    (
                        "Classical ML execution refused "
                        "because the validated Preparation "
                        "revision changed before training. "
                        "A server-owned caller requested "
                        "execution against an earlier "
                        "Preparation snapshot."
                    )
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
                contract,

            dataframe=
                dataframe,
        )
    )


    purged_rows = int(
        len(
            x
        )
        -
        len(
            x_train
        )
        -
        len(
            x_test
        )
    )


    if purged_rows < 0:

        raise (
            ClassicalMLExecutorError(
                (
                    "Classical ML holdout row "
                    "accounting became negative."
                )
            )
        )


    if (
        not isinstance(
            contract.split,
            MLPurgedGroupTimeHoldoutSplitContract,
        )
        and
        purged_rows
        !=
        0
    ):

        raise (
            ClassicalMLExecutorError(
                (
                    "A non-purged holdout strategy "
                    "discarded validated input rows."
                )
            )
        )


    try:
        baseline_prediction_bundle = (
            build_ml_baseline_predictions(
                problem_type=
                    contract.problem_type,

                y_train=
                    y_train,

                test_rows=
                    int(
                        len(
                            y_test
                        )
                    ),
            )
        )

    except MLBaselineError as error:
        raise (
            ClassicalMLExecutorError(
                (
                    "Classical ML baseline could not "
                    "be constructed from the training "
                    "split."
                )
            )
        ) from error


    estimator = (
        _build_estimator(
            contract=
                contract
        )
    )


    try:
        estimator.fit(
            x_train,
            y_train,
        )

    except Exception as error:
        raise (
            ClassicalMLExecutorError(
                (
                    "Classical ML estimator "
                    "training failed."
                )
            )
        ) from error


    try:
        predictions = (
            estimator.predict(
                x_test
            )
        )

    except Exception as error:
        raise (
            ClassicalMLExecutorError(
                (
                    "Classical ML estimator "
                    "prediction failed."
                )
            )
        ) from error


    if (
        contract.problem_type
        ==
        "regression"
    ):
        metrics = (
            _regression_metrics(
                y_true=
                    y_test,

                predictions=
                    predictions,
            )
        )

    else:
        metrics = (
            _classification_metrics(
                y_true=
                    y_test,

                predictions=
                    predictions,
            )
        )


    metrics = (
        _validate_metrics(
            metrics
        )
    )


    if (
        contract.problem_type
        ==
        "regression"
    ):
        baseline_metrics = (
            _regression_metrics(
                y_true=
                    y_test,

                predictions=(
                    baseline_prediction_bundle
                    .predictions
                ),
            )
        )

    else:
        baseline_metrics = (
            _classification_metrics(
                y_true=
                    y_test,

                predictions=(
                    baseline_prediction_bundle
                    .predictions
                ),
            )
        )


    baseline_metrics = (
        _validate_metrics(
            baseline_metrics
        )
    )


    baseline_metrics = (
        _baseline_metrics_v0_1(
            problem_type=
                contract.problem_type,

            metrics=
                baseline_metrics,
        )
    )


    try:
        baseline = (
            build_ml_baseline_evaluation(
                problem_type=
                    contract.problem_type,

                strategy=(
                    baseline_prediction_bundle
                    .strategy
                ),

                metrics=
                    baseline_metrics,

                train_rows=
                    int(
                        len(
                            y_train
                        )
                    ),

                test_rows=
                    int(
                        len(
                            y_test
                        )
                    ),
            )
        )


        baseline_comparison = (
            compare_model_to_baseline(
                problem_type=
                    contract.problem_type,

                model_metrics=
                    metrics,

                baseline_metrics=
                    baseline.metrics,
            )
        )

    except MLBaselineError as error:
        raise (
            ClassicalMLExecutorError(
                (
                    "Classical ML baseline evaluation "
                    "or model comparison failed."
                )
            )
        ) from error


    model_bytes = (
        _serialize_fitted_estimator(
            estimator
        )
    )


    try:
        model_artifact = (
            register_ml_model_artifact(
                training_contract=
                    contract,

                metrics=
                    metrics,

                train_rows=
                    int(
                        len(
                            x_train
                        )
                    ),

                test_rows=
                    int(
                        len(
                            x_test
                        )
                    ),

                model_bytes=
                    model_bytes,

                preparation_session_revision=
                    preparation_session_revision,
            )
        )

    except MLModelArtifactStoreError as error:
        raise (
            ClassicalMLExecutorError(
                (
                    "Classical ML training "
                    "completed but the resulting "
                    "server-owned Model Artifact "
                    "could not be persisted."
                )
            )
        ) from error


    experiment_provenance = (
        model_artifact
        .experiment_provenance
    )


    if (
        experiment_provenance
        is None
    ):
        raise (
            ClassicalMLExecutorError(
                (
                    "Current Classical ML execution "
                    "did not persist required "
                    "Experiment Provenance."
                )
            )
        )


    # ========================================================
    # ML MONITORING REFERENCE
    #
    # The exact x_train created by the deterministic holdout
    # split is reused here.
    #
    # No second split.
    # No second fit.
    # No holdout observations.
    # No raw rows persisted.
    # ========================================================


    try:
        monitoring_profile = (
            build_ml_monitoring_profile(
                x_train=
                    x_train,

                model_artifact=
                    model_artifact,
            )
        )


        register_ml_monitoring_profile(
            profile=
                monitoring_profile
        )


    except Exception as error:

        # ----------------------------------------------------
        # The Model Artifact already exists at this point.
        #
        # A training execution must not be returned as
        # successful if its required monitoring reference
        # could not be created.
        #
        # Compensate the newly-created Model Artifact.
        #
        # SQLite deletion also cascades any partially-created
        # Monitoring Profile.
        # ----------------------------------------------------


        cleanup_error = None


        try:
            delete_ml_model_artifact(
                model_id=
                    model_artifact.model_id,

                workflow_id=
                    model_artifact.workflow_id,
            )

        except Exception as candidate:
            cleanup_error = (
                candidate
            )


        if (
            cleanup_error
            is not None
        ):
            raise (
                ClassicalMLExecutorError(
                    (
                        "Classical ML training "
                        "completed, but Monitoring "
                        "Profile persistence failed "
                        "and Model Artifact "
                        "compensation could not "
                        "complete cleanly."
                    )
                )
            ) from error


        raise (
            ClassicalMLExecutorError(
                (
                    "Classical ML training "
                    "completed, but the required "
                    "Monitoring Profile could not "
                    "be persisted. The newly "
                    "created Model Artifact was "
                    "compensated."
                )
            )
        ) from error


    return (
        ClassicalMLExecutionResult(
            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,

            problem_type=
                contract.problem_type,

            estimator_key=
                contract.estimator_key,

            train_rows=
                int(
                    len(
                        x_train
                    )
                ),

            test_rows=
                int(
                    len(
                        x_test
                    )
                ),

            purged_rows=
                purged_rows,

            metrics=
                metrics,

            baseline=
                baseline,

            baseline_comparison=
                baseline_comparison,

            experiment_provenance=
                experiment_provenance,

            model_artifact=
                model_artifact,

            rule_version=
                CLASSICAL_ML_EXECUTOR_RULE_VERSION,
        )
    )