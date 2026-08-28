from __future__ import annotations


import math


import numpy as np
import pandas as pd


from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
)


from app.ml.classical_executor import (
    ClassicalMLEstimatorError,
    ClassicalMLExecutorError,
    ClassicalMLInputError,
    _build_estimator,
    _classification_metrics,
    _load_authorized_dataframe,
    _regression_metrics,
    _validate_and_extract_xy,
    _validate_metrics,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.cross_validation import (
    MLCrossValidationContract,
    MLCrossValidationEvaluationResult,
    MLCrossValidationFoldResult,
    MLCrossValidationMetricSummary,
    cross_validation_strategy,
)


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


# ============================================================
# ERRORS
# ============================================================


class MLCrossValidationError(
    RuntimeError
):
    pass


class MLCrossValidationInputError(
    MLCrossValidationError
):
    pass


class MLCrossValidationExecutionError(
    MLCrossValidationError
):
    pass


# ============================================================
# INPUT FEASIBILITY
# ============================================================


def _validate_cross_validation_feasibility(
    *,
    x: pd.DataFrame,
    y: pd.Series,
    training_contract: MLTrainingContract,
    cross_validation_contract: MLCrossValidationContract,
) -> None:

    folds = (
        cross_validation_contract
        .folds
    )


    row_count = int(
        len(
            x
        )
    )


    if (
        training_contract.problem_type
        ==
        "regression"
    ):
        # Richer regression metrics contain R?.
        #
        # R? is not defined for a validation fold containing
        # only one observation. Requiring at least two rows
        # per validation fold keeps every v0.1 metric finite.
        minimum_rows = (
            folds
            *
            2
        )


        if (
            row_count
            <
            minimum_rows
        ):
            raise (
                MLCrossValidationInputError(
                    (
                        "Regression Cross-Validation v0.1 "
                        "requires at least two validation "
                        "observations per fold because the "
                        "metric surface includes R?. "
                        f"rows={row_count}, "
                        f"folds={folds}, "
                        f"minimum_rows={minimum_rows}"
                    )
                )
            )


        return


    class_counts = (
        y
        .value_counts(
            dropna=False
        )
    )


    if (
        class_counts.empty
    ):
        raise (
            MLCrossValidationInputError(
                (
                    "Classification Cross-Validation "
                    "cannot operate on an empty "
                    "target distribution."
                )
            )
        )


    minimum_class_count = int(
        class_counts.min()
    )


    if (
        minimum_class_count
        <
        folds
    ):
        raise (
            MLCrossValidationInputError(
                (
                    "Stratified Cross-Validation v0.1 "
                    "requires every target class to "
                    "contain at least one observation "
                    "for every fold. "
                    f"minimum_class_count="
                    f"{minimum_class_count}, "
                    f"folds={folds}"
                )
            )
        )


# ============================================================
# SPLITTER
# ============================================================


def _build_cross_validation_splitter(
    *,
    training_contract: MLTrainingContract,
    cross_validation_contract: MLCrossValidationContract,
):

    random_state = (
        cross_validation_contract
        .random_seed

        if (
            cross_validation_contract
            .shuffle
        )

        else None
    )


    if (
        training_contract.problem_type
        ==
        "regression"
    ):
        return (
            KFold(
                n_splits=
                    cross_validation_contract
                    .folds,

                shuffle=
                    cross_validation_contract
                    .shuffle,

                random_state=
                    random_state,
            )
        )


    return (
        StratifiedKFold(
            n_splits=
                cross_validation_contract
                .folds,

            shuffle=
                cross_validation_contract
                .shuffle,

            random_state=
                random_state,
        )
    )


# ============================================================
# METRICS
# ============================================================


def _fold_metrics(
    *,
    training_contract: MLTrainingContract,
    y_true: pd.Series,
    predictions,
) -> dict[
    str,
    float,
]:

    if (
        training_contract.problem_type
        ==
        "regression"
    ):
        metrics = (
            _regression_metrics(
                y_true=
                    y_true,

                predictions=
                    predictions,
            )
        )

    else:
        metrics = (
            _classification_metrics(
                y_true=
                    y_true,

                predictions=
                    predictions,
            )
        )


    try:
        return (
            _validate_metrics(
                metrics
            )
        )

    except ClassicalMLExecutorError as error:
        raise (
            MLCrossValidationExecutionError(
                (
                    "Cross-validation fold produced "
                    "an invalid metric surface."
                )
            )
        ) from error


def _summarize_fold_metrics(
    *,
    fold_results: list[
        MLCrossValidationFoldResult
    ],
) -> dict[
    str,
    MLCrossValidationMetricSummary,
]:

    if not fold_results:
        raise (
            MLCrossValidationExecutionError(
                (
                    "Cross-validation produced "
                    "no fold results."
                )
            )
        )


    metric_names = list(
        fold_results[
            0
        ]
        .metrics
        .keys()
    )


    summary: dict[
        str,
        MLCrossValidationMetricSummary,
    ] = {}


    for metric_name in metric_names:

        values = np.asarray(
            [
                float(
                    fold
                    .metrics[
                        metric_name
                    ]
                )

                for fold
                in fold_results
            ],
            dtype=np.float64,
        )


        mean = float(
            np.mean(
                values
            )
        )


        std = float(
            np.std(
                values,
                ddof=0,
            )
        )


        if (
            not math.isfinite(
                mean
            )
            or
            not math.isfinite(
                std
            )
        ):
            raise (
                MLCrossValidationExecutionError(
                    (
                        "Cross-validation metric "
                        "summary is non-finite. "
                        f"metric={metric_name}"
                    )
                )
            )


        summary[
            metric_name
        ] = (
            MLCrossValidationMetricSummary(
                mean=
                    mean,

                std=
                    std,
            )
        )


    return summary


# ============================================================
# EXECUTION
# ============================================================


def execute_ml_cross_validation(
    *,
    training_contract: MLTrainingContract,
    cross_validation_contract: MLCrossValidationContract,
) -> MLCrossValidationEvaluationResult:
    """
    Evaluate one server-owned Classical ML training contract
    using deterministic cross-validation.

    Cross-Validation v0.1 is evaluation-only.

    It does NOT:
    - replace the existing holdout training execution;
    - persist fold models;
    - create Model Artifacts;
    - create new experiment IDs;
    - persist raw rows or fold predictions.

    A completely new scikit-learn Pipeline is constructed and
    fitted inside every fold. This keeps learned preprocessing
    statistics strictly inside the fold training boundary.
    """

    contract = (
        MLTrainingContract
        .model_validate(
            training_contract
        )
    )


    cv_contract = (
        MLCrossValidationContract
        .model_validate(
            cross_validation_contract
        )
    )


    try:
        (
            dataframe,
            preparation_session_revision,
        ) = (
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

    except ClassicalMLInputError as error:
        raise (
            MLCrossValidationInputError(
                (
                    "Cross-validation refused the "
                    "server-owned ML input. "
                    f"{error}"
                )
            )
        ) from error


    _validate_cross_validation_feasibility(
        x=
            x,

        y=
            y,

        training_contract=
            contract,

        cross_validation_contract=
            cv_contract,
    )


    splitter = (
        _build_cross_validation_splitter(
            training_contract=
                contract,

            cross_validation_contract=
                cv_contract,
        )
    )


    strategy = (
        cross_validation_strategy(
            problem_type=
                contract.problem_type
        )
    )


    if (
        contract.problem_type
        ==
        "classification"
    ):
        split_iterator = (
            splitter.split(
                x,
                y,
            )
        )

    else:
        split_iterator = (
            splitter.split(
                x
            )
        )


    fold_results: list[
        MLCrossValidationFoldResult
    ] = []


    try:
        split_pairs = list(
            split_iterator
        )

    except ValueError as error:
        raise (
            MLCrossValidationInputError(
                (
                    "Cross-validation folds could "
                    "not be constructed from the "
                    "validated dataset."
                )
            )
        ) from error


    if (
        len(
            split_pairs
        )
        !=
        cv_contract.folds
    ):
        raise (
            MLCrossValidationExecutionError(
                (
                    "Cross-validation splitter "
                    "did not produce the configured "
                    "number of folds."
                )
            )
        )


    for (
        zero_based_fold_index,
        (
            train_indices,
            validation_indices,
        ),
    ) in enumerate(
        split_pairs
    ):

        fold_index = (
            zero_based_fold_index
            +
            1
        )


        x_train = (
            x.iloc[
                train_indices
            ]
            .copy(
                deep=True
            )
        )


        y_train = (
            y.iloc[
                train_indices
            ]
            .copy(
                deep=True
            )
        )


        x_validation = (
            x.iloc[
                validation_indices
            ]
            .copy(
                deep=True
            )
        )


        y_validation = (
            y.iloc[
                validation_indices
            ]
            .copy(
                deep=True
            )
        )


        try:
            estimator = (
                _build_estimator(
                    contract=
                        contract
                )
            )

        except ClassicalMLEstimatorError as error:
            raise (
                MLCrossValidationExecutionError(
                    (
                        "Cross-validation could not "
                        "construct the estimator. "
                        f"fold={fold_index}"
                    )
                )
            ) from error


        try:
            estimator.fit(
                x_train,
                y_train,
            )

        except Exception as error:
            raise (
                MLCrossValidationExecutionError(
                    (
                        "Cross-validation estimator "
                        "training failed. "
                        f"fold={fold_index}"
                    )
                )
            ) from error


        try:
            predictions = (
                estimator.predict(
                    x_validation
                )
            )

        except Exception as error:
            raise (
                MLCrossValidationExecutionError(
                    (
                        "Cross-validation estimator "
                        "prediction failed. "
                        f"fold={fold_index}"
                    )
                )
            ) from error


        metrics = (
            _fold_metrics(
                training_contract=
                    contract,

                y_true=
                    y_validation,

                predictions=
                    predictions,
            )
        )


        fold_results.append(
            MLCrossValidationFoldResult(
                fold_index=
                    fold_index,

                train_rows=
                    int(
                        len(
                            x_train
                        )
                    ),

                validation_rows=
                    int(
                        len(
                            x_validation
                        )
                    ),

                metrics=
                    metrics,
            )
        )


    metric_summary = (
        _summarize_fold_metrics(
            fold_results=
                fold_results
        )
    )


    return (
        MLCrossValidationEvaluationResult(
            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,

            problem_type=
                contract.problem_type,

            estimator_key=
                contract.estimator_key,

            preparation_session_revision=
                int(
                    preparation_session_revision
                ),

            training_contract_sha256=
                ml_training_contract_sha256(
                    contract
                ),

            strategy=
                strategy,

            folds=
                cv_contract.folds,

            shuffle=
                cv_contract.shuffle,

            random_seed=
                cv_contract.random_seed,

            fold_results=
                fold_results,

            metric_summary=
                metric_summary,
        )
    )
