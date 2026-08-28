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


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


from sklearn.model_selection import (
    train_test_split,
)


from sklearn.pipeline import (
    Pipeline,
)


from app.ml.contracts import (
    MLTrainingContract,
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
    validate_ml_feature_frame,
)


from app.ml.model_artifact_store import (
    MLModelArtifactStoreError,
    register_ml_model_artifact,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.preparation.analysis_input_handoff import (
    AnalysisInputHandoffError,
    load_validated_analysis_input,
)


from app.preparation.analysis_readiness_gate import (
    AnalysisReadinessError,
)


# ============================================================
# VERSION
# ============================================================


CLASSICAL_ML_EXECUTOR_RULE_VERSION = (
    "classical_ml_executor_v0.1"
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


    metrics: dict[
        str,
        float,
    ]


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
) -> pd.DataFrame:
    """
    Resolve ML input through the exact same server-owned
    Preparation -> Analysis handoff used by deterministic
    analytical execution.

    The contract dataset_id is never sufficient on its own.
    It must also be present in the handoff-authorized final
    dataset scope.
    """

    try:
        handoff = (
            load_validated_analysis_input(
                workflow_id=
                    contract.workflow_id
            )
        )

    except (
        AnalysisInputHandoffError,
        AnalysisReadinessError,
    ) as error:
        raise (
            ClassicalMLInputError(
                (
                    "Classical ML execution refused "
                    "because Preparation did not "
                    "provide a valid READY analysis "
                    "input handoff."
                )
            )
        ) from error


    if (
        handoff.workflow_id
        !=
        contract.workflow_id
    ):
        raise (
            ClassicalMLInputError(
                (
                    "Analysis input handoff workflow "
                    "does not match the ML Training "
                    "Contract."
                )
            )
        )


    if (
        contract.dataset_id
        not in
        handoff.dataset_ids
    ):
        raise (
            ClassicalMLInputError(
                (
                    "ML Training Contract dataset "
                    "is outside the server-owned "
                    "validated analysis output scope. "
                    f"dataset_id={contract.dataset_id}"
                )
            )
        )


    matching_records = [
        record

        for record
        in handoff.dataset_records

        if (
            isinstance(
                record,
                dict,
            )
            and
            str(
                record.get(
                    "dataset_id",
                    "",
                )
            )
            ==
            contract.dataset_id
        )
    ]


    if (
        len(
            matching_records
        )
        !=
        1
    ):
        raise (
            ClassicalMLInputError(
                (
                    "Validated analysis handoff "
                    "does not contain exactly one "
                    "record for the requested ML "
                    "dataset."
                )
            )
        )


    dataframe = (
        matching_records[
            0
        ]
        .get(
            "dataframe"
        )
    )


    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise (
            ClassicalMLInputError(
                (
                    "Validated analysis handoff "
                    "record does not contain a "
                    "pandas DataFrame."
                )
            )
        )


    if dataframe.empty:
        raise (
            ClassicalMLInputError(
                "ML input dataset cannot be empty."
            )
        )


    return (
        dataframe.copy(
            deep=True
        )
    )


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

    required_columns = [
        *contract.feature_columns,
        contract.target_column,
    ]


    missing_columns = [
        column

        for column
        in required_columns

        if column
        not in
        dataframe.columns
    ]


    if missing_columns:
        raise (
            ClassicalMLInputError(
                (
                    "ML input dataset is missing "
                    "required contract columns: "
                    +
                    ", ".join(
                        missing_columns
                    )
                )
            )
        )


    selected = (
        dataframe.loc[
            :,
            required_columns,
        ]
        .copy(
            deep=True
        )
    )


    x = (
        selected.loc[
            :,
            contract.feature_columns,
        ]
        .copy(
            deep=True
        )
    )


    y = (
        selected.loc[
            :,
            contract.target_column,
        ]
        .copy(
            deep=True
        )
    )


    # ========================================================
    # TARGET MISSING VALUES
    # ========================================================


    if bool(
        y.isna().any()
    ):
        raise (
            ClassicalMLInputError(
                (
                    "ML target contains missing values. "
                    "Target imputation is never performed "
                    "by Classical ML."
                )
            )
        )


    # ========================================================
    # FEATURE STRUCTURE / PREPROCESSING POLICY
    # ========================================================


    try:
        x = (
            validate_ml_feature_frame(
                features=x,
                contract=contract,
            )
        )

    except MLPreprocessingRuntimeError as error:
        raise (
            ClassicalMLInputError(
                str(
                    error
                )
            )
        ) from error


    # ========================================================
    # REGRESSION TARGET
    # ========================================================


    if (
        contract.problem_type
        ==
        "regression"
    ):
        target_dtype = (
            y.dtype
        )


        if (
            pd.api.types
            .is_bool_dtype(
                target_dtype
            )
            or
            not pd.api.types
            .is_numeric_dtype(
                target_dtype
            )
        ):
            raise (
                ClassicalMLInputError(
                    (
                        "Regression target must be "
                        "numeric and non-boolean. "
                        f"target={contract.target_column}, "
                        f"dtype={target_dtype}"
                    )
                )
            )


        try:
            numeric_y = (
                y.to_numpy(
                    dtype=np.float64,
                    copy=True,
                )
            )

        except Exception as error:
            raise (
                ClassicalMLInputError(
                    (
                        "Regression target could "
                        "not be converted to "
                        "floating-point values."
                    )
                )
            ) from error


        if not (
            np.isfinite(
                numeric_y
            )
            .all()
        ):
            raise (
                ClassicalMLInputError(
                    (
                        "Regression target contains "
                        "non-finite values."
                    )
                )
            )


        if (
            int(
                y.nunique(
                    dropna=False
                )
            )
            <
            2
        ):
            raise (
                ClassicalMLInputError(
                    (
                        "Regression target must "
                        "contain at least two "
                        "distinct values."
                    )
                )
            )


    # ========================================================
    # CLASSIFICATION TARGET
    # ========================================================


    else:
        class_count = int(
            y.nunique(
                dropna=False
            )
        )


        if (
            class_count
            <
            2
        ):
            raise (
                ClassicalMLInputError(
                    (
                        "Classification target must "
                        "contain at least two classes."
                    )
                )
            )


        if (
            pd.api.types
            .is_numeric_dtype(
                y.dtype
            )
            and
            not pd.api.types
            .is_bool_dtype(
                y.dtype
            )
        ):
            try:
                numeric_y = (
                    y.to_numpy(
                        dtype=np.float64,
                        copy=True,
                    )
                )

            except Exception as error:
                raise (
                    ClassicalMLInputError(
                        (
                            "Numeric classification "
                            "target could not be "
                            "validated."
                        )
                    )
                ) from error


            if not (
                np.isfinite(
                    numeric_y
                )
                .all()
            ):
                raise (
                    ClassicalMLInputError(
                        (
                            "Classification target "
                            "contains non-finite "
                            "numeric values."
                        )
                    )
                )


    return (
        x,
        y,
    )


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
# SPLIT
# ============================================================


def _split_dataset(
    *,
    x: pd.DataFrame,
    y: pd.Series,
    contract: MLTrainingContract,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:

    if (
        contract.split.stratify
        and
        not contract.split.shuffle
    ):
        raise (
            ClassicalMLInputError(
                (
                    "stratify=True requires "
                    "shuffle=True for the "
                    "Classical ML holdout split."
                )
            )
        )


    stratify_values = (
        y

        if (
            contract.problem_type
            ==
            "classification"
            and
            contract.split.stratify
        )

        else None
    )


    random_state = (
        contract.split.random_seed

        if contract.split.shuffle

        else None
    )


    try:
        (
            x_train,
            x_test,
            y_train,
            y_test,
        ) = (
            train_test_split(
                x,
                y,

                test_size=
                    contract
                    .split
                    .test_size,

                random_state=
                    random_state,

                shuffle=
                    contract
                    .split
                    .shuffle,

                stratify=
                    stratify_values,
            )
        )

    except ValueError as error:
        raise (
            ClassicalMLInputError(
                (
                    "Deterministic train/test "
                    "split could not be created "
                    "from the ML Training Contract."
                )
            )
        ) from error


    if (
        len(
            x_train
        )
        <
        2
        or
        len(
            x_test
        )
        <
        2
    ):
        raise (
            ClassicalMLInputError(
                (
                    "Classical ML v0.1 requires "
                    "at least two training rows "
                    "and two test rows after "
                    "the holdout split."
                )
            )
        )


    if (
        contract.problem_type
        ==
        "classification"
        and
        int(
            y_train.nunique(
                dropna=False
            )
        )
        <
        2
    ):
        raise (
            ClassicalMLInputError(
                (
                    "Classification training split "
                    "contains fewer than two "
                    "classes."
                )
            )
        )


    return (
        x_train,
        x_test,
        y_train,
        y_test,
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

    mae = float(
        mean_absolute_error(
            y_true,
            predictions,
        )
    )


    mse = float(
        mean_squared_error(
            y_true,
            predictions,
        )
    )


    rmse = float(
        math.sqrt(
            mse
        )
    )


    r2 = float(
        r2_score(
            y_true,
            predictions,
        )
    )


    return {
        "mae":
            mae,

        "rmse":
            rmse,

        "r2":
            r2,
    }


def _classification_metrics(
    *,
    y_true: pd.Series,
    predictions: Any,
) -> dict[
    str,
    float,
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
    }


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
    """

    contract = (
        MLTrainingContract
        .model_validate(
            training_contract
        )
    )


    dataframe = (
        _load_authorized_dataframe(
            contract=
                contract
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
        )
    )


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

            metrics=
                metrics,

            model_artifact=
                model_artifact,

            rule_version=
                CLASSICAL_ML_EXECUTOR_RULE_VERSION,
        )
    )