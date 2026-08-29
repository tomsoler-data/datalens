from __future__ import annotations


import math


from datetime import (
    datetime,
    timezone,
)


from uuid import (
    uuid4,
)


import numpy as np
import pandas as pd


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.ml.model_loader import (
    LoadedMLModel,
)


from app.ml.model_metrics import (
    compute_ml_classification_metrics,
    compute_ml_regression_metrics,
    ml_model_metric_direction,
    ml_model_metric_names,
    ml_model_primary_metric,
)


from app.ml.performance_evaluation import (
    FLOAT_TOLERANCE,
    MLPerformanceEvaluationRecord,
    MLPerformanceMetricComparison,
    ml_performance_status_for_primary_metric,
)


from app.ml.preprocessing import (
    MLPreprocessingRuntimeError,
    validate_ml_feature_frame,
)


# ============================================================
# VERSION
# ============================================================


ML_PERFORMANCE_EVALUATOR_RULE_VERSION = (
    "ml_performance_evaluator_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLPerformanceEvaluatorError(
    RuntimeError
):
    pass


class MLPerformanceEvaluatorInputError(
    MLPerformanceEvaluatorError
):
    pass


class MLPerformanceEvaluatorAuthorityError(
    MLPerformanceEvaluatorError
):
    pass


class MLPerformanceTargetError(
    MLPerformanceEvaluatorError
):
    pass


class MLPerformancePredictionError(
    MLPerformanceEvaluatorError
):
    pass


class MLPerformanceMetricsError(
    MLPerformanceEvaluatorError
):
    pass


# ============================================================
# TIME / ID
# ============================================================


def _utc_now_iso(
) -> str:

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


def _new_performance_evaluation_id(
) -> str:

    return (
        "performance-evaluation:"
        +
        uuid4().hex
    )


# ============================================================
# TEXT / REVISION
# ============================================================


def _required_text(
    value: object,
    *,
    field_name: str,
) -> str:

    normalized = str(
        value
        if value is not None
        else ""
    ).strip()


    if not normalized:
        raise MLPerformanceEvaluatorInputError(
            (
                f"{field_name} "
                "cannot be empty."
            )
        )


    return normalized


def _observed_revision(
    value: object,
) -> int:

    if (
        not isinstance(
            value,
            int,
        )
        or
        isinstance(
            value,
            bool,
        )
        or
        value < 0
    ):
        raise MLPerformanceEvaluatorInputError(
            (
                "observed_preparation_session_revision "
                "must be a non-negative integer."
            )
        )


    return value


# ============================================================
# TRUSTED MODEL AUTHORITY
# ============================================================


def _validated_artifact(
    *,
    trusted_model: LoadedMLModel,
) -> MLModelArtifactRecord:

    if not isinstance(
        trusted_model,
        LoadedMLModel,
    ):
        raise MLPerformanceEvaluatorAuthorityError(
            (
                "Performance Evaluation requires "
                "a trusted LoadedMLModel."
            )
        )


    try:
        artifact = (
            MLModelArtifactRecord
            .model_validate(
                trusted_model.artifact
            )
        )

    except Exception as error:
        raise MLPerformanceEvaluatorAuthorityError(
            (
                "Trusted Model Artifact "
                "metadata is invalid."
            )
        ) from error


    provenance = (
        artifact.experiment_provenance
    )


    if provenance is None:
        raise MLPerformanceEvaluatorAuthorityError(
            (
                "Performance Evaluation requires "
                "trusted Experiment Provenance."
            )
        )


    expected_sha = (
        ml_training_contract_sha256(
            artifact.training_contract
        )
    )


    if (
        provenance.training_contract_sha256
        !=
        expected_sha
    ):
        raise MLPerformanceEvaluatorAuthorityError(
            (
                "Model Artifact provenance does "
                "not match its Training Contract."
            )
        )


    if (
        provenance.metrics
        !=
        artifact.metrics
    ):
        raise MLPerformanceEvaluatorAuthorityError(
            (
                "Model Artifact metrics do not "
                "match Experiment Provenance."
            )
        )


    if (
        provenance.test_rows
        !=
        artifact.test_rows
    ):
        raise MLPerformanceEvaluatorAuthorityError(
            (
                "Model Artifact holdout evidence "
                "does not match Experiment Provenance."
            )
        )


    return artifact


# ============================================================
# REFERENCE METRIC AUTHORITY
# ============================================================


def _reference_metrics(
    *,
    artifact: MLModelArtifactRecord,
) -> dict[
    str,
    float,
]:

    problem_type = (
        artifact
        .training_contract
        .problem_type
    )


    expected_names = list(
        ml_model_metric_names(
            problem_type=
                problem_type
        )
    )


    actual_names = set(
        artifact.metrics
    )


    if (
        actual_names
        !=
        set(
            expected_names
        )
    ):
        raise MLPerformanceEvaluatorAuthorityError(
            (
                "Persisted Model Artifact metrics "
                "do not match the complete canonical "
                "metric surface. "
                f"expected={expected_names}, "
                f"actual={sorted(actual_names)}"
            )
        )


    normalized = {}


    for metric_name in (
        expected_names
    ):

        try:
            value = float(
                artifact.metrics[
                    metric_name
                ]
            )

        except Exception as error:
            raise MLPerformanceEvaluatorAuthorityError(
                (
                    "Persisted Model Artifact "
                    "contains an invalid metric. "
                    f"metric_name={metric_name}"
                )
            ) from error


        if not math.isfinite(
            value
        ):
            raise MLPerformanceEvaluatorAuthorityError(
                (
                    "Persisted Model Artifact "
                    "contains a non-finite metric. "
                    f"metric_name={metric_name}"
                )
            )


        normalized[
            metric_name
        ] = value


    return normalized


# ============================================================
# REQUIRED OBSERVED COLUMN SURFACE
# ============================================================


def _require_unique_column(
    *,
    dataframe: pd.DataFrame,
    column_name: str,
    role: str,
) -> None:

    occurrences = sum(
        1

        for actual_name
        in dataframe.columns

        if (
            actual_name
            ==
            column_name
        )
    )


    if occurrences == 0:
        if role == "target":
            raise MLPerformanceTargetError(
                (
                    "Observed dataset does not "
                    "contain the true target required "
                    "for Performance Evaluation. "
                    f"target={column_name}"
                )
            )


        raise MLPerformanceEvaluatorInputError(
            (
                "Observed dataset is missing "
                "a required model feature. "
                f"feature={column_name}"
            )
        )


    if occurrences > 1:
        if role == "target":
            raise MLPerformanceTargetError(
                (
                    "Observed dataset contains "
                    "a duplicate true target column. "
                    f"target={column_name}"
                )
            )


        raise MLPerformanceEvaluatorInputError(
            (
                "Observed dataset contains "
                "a duplicate required model feature. "
                f"feature={column_name}"
            )
        )


# ============================================================
# TARGET VALIDATION
# ============================================================


def _validate_target(
    *,
    target: pd.Series,
    problem_type: str,
    target_column: str,
) -> pd.Series:

    if not isinstance(
        target,
        pd.Series,
    ):
        raise MLPerformanceTargetError(
            (
                "Observed true target must resolve "
                "to exactly one pandas Series."
            )
        )


    if target.empty:
        raise MLPerformanceTargetError(
            (
                "Observed true target cannot "
                "be empty."
            )
        )


    if bool(
        target.isna().any()
    ):
        raise MLPerformanceTargetError(
            (
                "Observed true target contains "
                "missing labels. Performance "
                "Evaluation never imputes ground truth. "
                f"target={target_column}"
            )
        )


    # --------------------------------------------------------
    # REGRESSION
    # --------------------------------------------------------


    if (
        problem_type
        ==
        "regression"
    ):

        if (
            pd.api.types
            .is_bool_dtype(
                target.dtype
            )
            or
            not pd.api.types
            .is_numeric_dtype(
                target.dtype
            )
        ):
            raise MLPerformanceTargetError(
                (
                    "Regression ground truth must "
                    "be numeric and non-boolean. "
                    f"target={target_column}, "
                    f"dtype={target.dtype}"
                )
            )


        try:
            values = (
                target.to_numpy(
                    dtype=np.float64,
                    copy=True,
                )
            )

        except Exception as error:
            raise MLPerformanceTargetError(
                (
                    "Regression ground truth "
                    "could not be converted to "
                    "finite float64 values."
                )
            ) from error


        if not (
            np.isfinite(
                values
            )
            .all()
        ):
            raise MLPerformanceTargetError(
                (
                    "Regression ground truth "
                    "contains non-finite values."
                )
            )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------


    elif (
        problem_type
        ==
        "classification"
    ):

        if (
            pd.api.types
            .is_numeric_dtype(
                target.dtype
            )
            and
            not pd.api.types
            .is_bool_dtype(
                target.dtype
            )
        ):
            try:
                values = (
                    target.to_numpy(
                        dtype=np.float64,
                        copy=True,
                    )
                )

            except Exception as error:
                raise MLPerformanceTargetError(
                    (
                        "Numeric classification "
                        "ground truth could not "
                        "be validated."
                    )
                ) from error


            if not (
                np.isfinite(
                    values
                )
                .all()
            ):
                raise MLPerformanceTargetError(
                    (
                        "Classification ground truth "
                        "contains non-finite numeric "
                        "values."
                    )
                )


    else:
        raise MLPerformanceEvaluatorAuthorityError(
            (
                "Trusted Model Artifact contains "
                "an unsupported problem type. "
                f"problem_type={problem_type}"
            )
        )


    return (
        target.copy(
            deep=True
        )
    )


# ============================================================
# OBSERVED SUPERVISED SURFACE
# ============================================================


def _validated_observed_surface(
    *,
    dataframe: pd.DataFrame,
    artifact: MLModelArtifactRecord,
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise MLPerformanceEvaluatorInputError(
            (
                "Performance Evaluation input "
                "must be a pandas DataFrame."
            )
        )


    if dataframe.empty:
        raise MLPerformanceEvaluatorInputError(
            (
                "Performance Evaluation observed "
                "dataset cannot be empty."
            )
        )


    # Two observations are required because the canonical
    # regression metric surface includes R?.
    if (
        len(
            dataframe
        )
        <
        2
    ):
        raise MLPerformanceEvaluatorInputError(
            (
                "Performance Evaluation requires "
                "at least two labeled observations."
            )
        )


    contract = (
        artifact.training_contract
    )


    feature_columns = list(
        contract.feature_columns
    )


    target_column = (
        contract.target_column
    )


    for feature_name in (
        feature_columns
    ):
        _require_unique_column(
            dataframe=
                dataframe,

            column_name=
                feature_name,

            role=
                "feature",
        )


    _require_unique_column(
        dataframe=
            dataframe,

        column_name=
            target_column,

        role=
            "target",
    )


    # Extra columns are deliberately ignored.
    # They never reach predict() and never enter the result.
    features = (
        dataframe.loc[
            :,
            feature_columns,
        ]
        .copy(
            deep=True
        )
    )


    target = (
        dataframe.loc[
            :,
            target_column,
        ]
        .copy(
            deep=True
        )
    )


    try:
        features = (
            validate_ml_feature_frame(
                features=
                    features,

                contract=
                    contract,
            )
        )

    except MLPreprocessingRuntimeError as error:
        raise MLPerformanceEvaluatorInputError(
            (
                "Observed feature surface is "
                "not compatible with the trusted "
                "ML Preprocessing Contract."
            )
        ) from error


    target = (
        _validate_target(
            target=
                target,

            problem_type=
                contract.problem_type,

            target_column=
                target_column,
        )
    )


    return (
        features,
        target,
    )


# ============================================================
# PREDICTION VALIDATION
# ============================================================


def _validated_predictions(
    *,
    trusted_model: LoadedMLModel,
    features: pd.DataFrame,
    problem_type: str,
) -> pd.Series:

    try:
        raw_predictions = (
            trusted_model.predict(
                features
            )
        )

    except Exception as error:
        raise MLPerformancePredictionError(
            (
                "Trusted model predict() "
                "execution failed."
            )
        ) from error


    try:
        predictions = pd.Series(
            raw_predictions
        )

    except Exception as error:
        raise MLPerformancePredictionError(
            (
                "Trusted model predictions "
                "could not be normalized to "
                "a one-dimensional sequence."
            )
        ) from error


    if (
        len(
            predictions
        )
        !=
        len(
            features
        )
    ):
        raise MLPerformancePredictionError(
            (
                "Trusted model prediction count "
                "does not match observed row count."
            )
        )


    if bool(
        predictions.isna().any()
    ):
        raise MLPerformancePredictionError(
            (
                "Trusted model predictions "
                "contain missing values."
            )
        )


    if (
        problem_type
        ==
        "regression"
    ):

        if (
            pd.api.types
            .is_bool_dtype(
                predictions.dtype
            )
            or
            not pd.api.types
            .is_numeric_dtype(
                predictions.dtype
            )
        ):
            raise MLPerformancePredictionError(
                (
                    "Regression predictions must "
                    "be numeric and non-boolean."
                )
            )


        try:
            values = (
                predictions.to_numpy(
                    dtype=np.float64,
                    copy=True,
                )
            )

        except Exception as error:
            raise MLPerformancePredictionError(
                (
                    "Regression predictions could "
                    "not be converted to float64."
                )
            ) from error


        if not (
            np.isfinite(
                values
            )
            .all()
        ):
            raise MLPerformancePredictionError(
                (
                    "Regression predictions contain "
                    "non-finite values."
                )
            )


    elif (
        problem_type
        ==
        "classification"
    ):

        if (
            pd.api.types
            .is_numeric_dtype(
                predictions.dtype
            )
            and
            not pd.api.types
            .is_bool_dtype(
                predictions.dtype
            )
        ):

            try:
                values = (
                    predictions.to_numpy(
                        dtype=np.float64,
                        copy=True,
                    )
                )

            except Exception as error:
                raise MLPerformancePredictionError(
                    (
                        "Numeric classification "
                        "predictions could not "
                        "be validated."
                    )
                ) from error


            if not (
                np.isfinite(
                    values
                )
                .all()
            ):
                raise MLPerformancePredictionError(
                    (
                        "Classification predictions "
                        "contain non-finite numeric "
                        "values."
                    )
                )


    else:
        raise MLPerformanceEvaluatorAuthorityError(
            (
                "Unsupported trusted model "
                "problem type."
            )
        )


    return (
        predictions.copy(
            deep=True
        )
    )


# ============================================================
# OBSERVED METRICS
# ============================================================


def _observed_metrics(
    *,
    problem_type: str,
    target: pd.Series,
    predictions: pd.Series,
) -> dict[
    str,
    float,
]:

    try:

        if (
            problem_type
            ==
            "regression"
        ):
            metrics = (
                compute_ml_regression_metrics(
                    y_true=
                        target,

                    predictions=
                        predictions,
                )
            )

        elif (
            problem_type
            ==
            "classification"
        ):
            metrics = (
                compute_ml_classification_metrics(
                    y_true=
                        target,

                    predictions=
                        predictions,
                )
            )

        else:
            raise MLPerformanceEvaluatorAuthorityError(
                (
                    "Unsupported trusted model "
                    "problem type."
                )
            )


    except MLPerformanceEvaluatorError:
        raise


    except Exception as error:
        raise MLPerformanceMetricsError(
            (
                "Canonical ML metrics could not "
                "be computed from observed labels "
                "and trusted predictions."
            )
        ) from error


    expected_names = list(
        ml_model_metric_names(
            problem_type=
                problem_type
        )
    )


    if (
        list(
            metrics
        )
        !=
        expected_names
    ):
        raise MLPerformanceMetricsError(
            (
                "Canonical observed metric surface "
                "is inconsistent."
            )
        )


    normalized = {}


    for metric_name in (
        expected_names
    ):

        try:
            value = float(
                metrics[
                    metric_name
                ]
            )

        except Exception as error:
            raise MLPerformanceMetricsError(
                (
                    "Observed metric could not "
                    "be normalized. "
                    f"metric_name={metric_name}"
                )
            ) from error


        if not math.isfinite(
            value
        ):
            raise MLPerformanceMetricsError(
                (
                    "Observed metric is non-finite. "
                    f"metric_name={metric_name}"
                )
            )


        normalized[
            metric_name
        ] = value


    return normalized


# ============================================================
# METRIC COMPARISONS
# ============================================================


def _metric_comparisons(
    *,
    problem_type: str,
    reference_metrics: dict[
        str,
        float,
    ],
    observed_metrics: dict[
        str,
        float,
    ],
) -> list[
    MLPerformanceMetricComparison
]:

    output = []


    for metric_name in (
        ml_model_metric_names(
            problem_type=
                problem_type
        )
    ):

        reference_value = float(
            reference_metrics[
                metric_name
            ]
        )


        observed_value = float(
            observed_metrics[
                metric_name
            ]
        )


        direction = (
            ml_model_metric_direction(
                problem_type=
                    problem_type,

                metric_name=
                    metric_name,
            )
        )


        delta = (
            observed_value
            -
            reference_value
        )


        if (
            direction
            ==
            "higher_is_better"
        ):
            degradation_amount = max(
                0.0,
                reference_value
                -
                observed_value,
            )

        else:
            degradation_amount = max(
                0.0,
                observed_value
                -
                reference_value,
            )


        output.append(
            MLPerformanceMetricComparison(
                metric_name=
                    metric_name,

                direction=
                    direction,

                reference_value=
                    reference_value,

                observed_value=
                    observed_value,

                delta=
                    delta,

                degradation_amount=
                    degradation_amount,
            )
        )


    return output


# ============================================================
# PUBLIC EVALUATOR
# ============================================================


def evaluate_ml_performance(
    *,
    observed_dataframe: pd.DataFrame,
    observed_dataset_id: str,
    observed_preparation_session_revision: int,
    trusted_model: LoadedMLModel,
) -> MLPerformanceEvaluationRecord:
    """
    Evaluate supervised model performance on one labeled
    observed dataset.

    No persistence occurs here.

    This function accepts only an already trusted LoadedMLModel.
    The future Monitoring Service is responsible for obtaining
    both the model and the observed DataFrame from server-owned
    authorities.

    A Performance Evaluation Record is produced only when a
    complete valid ground-truth target is available.
    """

    normalized_dataset_id = (
        _required_text(
            observed_dataset_id,
            field_name=
                "observed_dataset_id",
        )
    )


    observed_revision = (
        _observed_revision(
            observed_preparation_session_revision
        )
    )


    artifact = (
        _validated_artifact(
            trusted_model=
                trusted_model
        )
    )


    contract = (
        artifact.training_contract
    )


    reference_metrics = (
        _reference_metrics(
            artifact=
                artifact
        )
    )


    (
        validated_features,
        validated_target,
    ) = (
        _validated_observed_surface(
            dataframe=
                observed_dataframe,

            artifact=
                artifact,
        )
    )


    predictions = (
        _validated_predictions(
            trusted_model=
                trusted_model,

            features=
                validated_features,

            problem_type=
                contract.problem_type,
        )
    )


    observed_metrics = (
        _observed_metrics(
            problem_type=
                contract.problem_type,

            target=
                validated_target,

            predictions=
                predictions,
        )
    )


    metric_results = (
        _metric_comparisons(
            problem_type=
                contract.problem_type,

            reference_metrics=
                reference_metrics,

            observed_metrics=
                observed_metrics,
        )
    )


    primary_metric = (
        ml_model_primary_metric(
            problem_type=
                contract.problem_type
        )
    )


    primary_result = next(
        result

        for result
        in metric_results

        if (
            result.metric_name
            ==
            primary_metric
        )
    )


    if (
        contract.problem_type
        ==
        "classification"
    ):
        degradation_basis = (
            "absolute_points"
        )


        degradation_ratio = None


    else:
        degradation_basis = (
            "relative_increase"
        )


        reference_rmse = (
            primary_result.reference_value
        )


        if (
            reference_rmse
            <=
            FLOAT_TOLERANCE
        ):
            degradation_ratio = None

        else:
            degradation_ratio = (
                primary_result
                .degradation_amount
                /
                reference_rmse
            )


    performance_status = (
        ml_performance_status_for_primary_metric(
            problem_type=
                contract.problem_type,

            reference_value=
                primary_result.reference_value,

            observed_value=
                primary_result.observed_value,
        )
    )


    provenance = (
        artifact.experiment_provenance
    )


    if provenance is None:
        raise MLPerformanceEvaluatorAuthorityError(
            (
                "Experiment Provenance disappeared "
                "after authority validation."
            )
        )


    return (
        MLPerformanceEvaluationRecord(
            performance_evaluation_id=(
                _new_performance_evaluation_id()
            ),

            model_id=
                artifact.model_id,

            workflow_id=
                artifact.workflow_id,

            reference_dataset_id=
                artifact.dataset_id,

            observed_dataset_id=
                normalized_dataset_id,

            experiment_id=
                provenance.experiment_id,

            preparation_session_revision=(
                provenance
                .preparation_session_revision
            ),

            observed_preparation_session_revision=(
                observed_revision
            ),

            training_contract_sha256=(
                provenance
                .training_contract_sha256
            ),

            problem_type=
                contract.problem_type,

            target_column=
                contract.target_column,

            reference_evaluation_row_count=(
                artifact.test_rows
            ),

            observed_row_count=
                len(
                    validated_target
                ),

            evaluated_at_utc=
                _utc_now_iso(),

            metric_results=
                metric_results,

            primary_metric=
                primary_metric,

            primary_metric_degradation_amount=(
                primary_result
                .degradation_amount
            ),

            primary_metric_degradation_ratio=(
                degradation_ratio
            ),

            degradation_basis=
                degradation_basis,

            performance_status=
                performance_status,
        )
    )
