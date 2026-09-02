from __future__ import annotations


import math


import numpy as np
import pandas as pd


from app.ml.classical_executor import (
    ClassicalMLEstimatorError,
    ClassicalMLInputError,
    _build_estimator,
    _load_authorized_dataframe,
    _split_dataset,
    _validate_and_extract_xy,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.cross_validation import (
    MLCrossValidationContract,
)


from app.ml.cross_validation_executor import (
    MLCrossValidationExecutionError,
    MLCrossValidationInputError,
    _build_cross_validation_pairs,
    _fold_metrics,
)


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


from app.ml.hyperparameter_tuning import (
    MLHyperparameterCandidateResult,
    MLHyperparameterMetricSummary,
    MLHyperparameterSearchContract,
    MLHyperparameterSearchResult,
    expected_hyperparameter_metric_names,
    hyperparameter_metric_direction,
    hyperparameter_primary_metric,
    hyperparameter_validation_strategy,
    server_owned_hyperparameter_candidates,
)


# ============================================================
# VERSION
# ============================================================


ML_HYPERPARAMETER_TUNING_EXECUTOR_RULE_VERSION = (
    "ml_hyperparameter_tuning_executor_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLHyperparameterTuningError(
    RuntimeError
):
    pass


class MLHyperparameterTuningInputError(
    MLHyperparameterTuningError
):
    pass


class MLHyperparameterTuningExecutionError(
    MLHyperparameterTuningError
):
    pass


# ============================================================
# CANDIDATE CONTRACT
# ============================================================


def _candidate_training_contract(
    *,
    base_contract: MLTrainingContract,
    hyperparameters,
) -> MLTrainingContract:
    """
    Create one fully validated Training Contract for a
    server-owned candidate.

    The base contract itself is never mutated.

    Only estimator_hyperparameters changes.
    """

    payload = (
        base_contract.model_dump(
            mode="python"
        )
    )


    payload[
        "estimator_hyperparameters"
    ] = (
        hyperparameters.model_dump(
            mode="python"
        )
    )


    return (
        MLTrainingContract.model_validate(
            payload
        )
    )


# ============================================================
# INNER CV SPLITS
# ============================================================


def _build_inner_cv_pairs(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    training_contract: MLTrainingContract,
    search_contract: MLHyperparameterSearchContract,
    groups_train: pd.Series | None = None,
):
    """
    Build deterministic INNER CV folds from OUTER train only.

    x_test / y_test and holdout-test groups are intentionally
    not accepted by this function.

    Group-aware tuning therefore receives only groups belonging
    to the authoritative OUTER training partition.
    """

    cv_contract = (
        MLCrossValidationContract(
            folds=
                search_contract.folds,

            shuffle=
                search_contract.shuffle,

            random_seed=
                search_contract.random_seed,
        )
    )


    try:
        return (
            _build_cross_validation_pairs(
                x=
                    x_train,

                y=
                    y_train,

                training_contract=
                    training_contract,

                cross_validation_contract=
                    cv_contract,

                groups=
                    groups_train,
            )
        )


    except MLCrossValidationInputError as error:

        raise (
            MLHyperparameterTuningInputError(
                (
                    "Hyperparameter Tuning refused "
                    "the OUTER training split because "
                    "INNER Cross-Validation is not "
                    "feasible. "
                    f"{error}"
                )
            )
        ) from error


    except MLCrossValidationExecutionError as error:

        raise (
            MLHyperparameterTuningExecutionError(
                (
                    "Hyperparameter Tuning INNER "
                    "Cross-Validation violated the "
                    "validated fold invariants. "
                    f"{error}"
                )
            )
        ) from error


# ============================================================
# METRIC SUMMARY
# ============================================================


def _summarize_candidate_metrics(
    *,
    fold_metrics: list[
        dict[
            str,
            float,
        ]
    ],
    problem_type: str,
) -> dict[
    str,
    MLHyperparameterMetricSummary,
]:
    """
    Aggregate one candidate's fold metrics.

    Population standard deviation is used:
    ddof=0.
    """

    if not fold_metrics:
        raise (
            MLHyperparameterTuningExecutionError(
                (
                    "Hyperparameter candidate "
                    "produced no fold metrics."
                )
            )
        )


    expected_metric_names = (
        expected_hyperparameter_metric_names(
            problem_type=
                problem_type
        )
    )


    expected_metric_set = set(
        expected_metric_names
    )


    for metrics in fold_metrics:
        if (
            set(
                metrics
            )
            !=
            expected_metric_set
        ):
            raise (
                MLHyperparameterTuningExecutionError(
                    (
                        "Hyperparameter candidate "
                        "produced an invalid metric "
                        "surface."
                    )
                )
            )


    summary: dict[
        str,
        MLHyperparameterMetricSummary,
    ] = {}


    for metric_name in (
        expected_metric_names
    ):

        values = np.asarray(
            [
                float(
                    metrics[
                        metric_name
                    ]
                )

                for metrics
                in fold_metrics
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
                MLHyperparameterTuningExecutionError(
                    (
                        "Hyperparameter metric "
                        "summary is non-finite. "
                        f"metric={metric_name}"
                    )
                )
            )


        summary[
            metric_name
        ] = (
            MLHyperparameterMetricSummary(
                mean=
                    mean,

                std=
                    std,
            )
        )


    return summary


# ============================================================
# ONE CANDIDATE
# ============================================================


def _evaluate_candidate(
    *,
    candidate_index: int,
    hyperparameters,
    base_contract: MLTrainingContract,
    x_outer_train: pd.DataFrame,
    y_outer_train: pd.Series,
    inner_split_pairs,
):
    """
    Evaluate one complete server-owned candidate.

    A fresh Pipeline is built for every INNER fold.

    No fitted candidate survives this function.
    """

    candidate_contract = (
        _candidate_training_contract(
            base_contract=
                base_contract,

            hyperparameters=
                hyperparameters,
        )
    )


    candidate_sha256 = (
        ml_training_contract_sha256(
            candidate_contract
        )
    )


    candidate_fold_metrics: list[
        dict[
            str,
            float,
        ]
    ] = []


    for (
        zero_based_fold_index,
        (
            inner_train_indices,
            inner_validation_indices,
        ),
    ) in enumerate(
        inner_split_pairs
    ):

        fold_index = (
            zero_based_fold_index
            +
            1
        )


        x_inner_train = (
            x_outer_train.iloc[
                inner_train_indices
            ]
            .copy(
                deep=True
            )
        )


        y_inner_train = (
            y_outer_train.iloc[
                inner_train_indices
            ]
            .copy(
                deep=True
            )
        )


        x_inner_validation = (
            x_outer_train.iloc[
                inner_validation_indices
            ]
            .copy(
                deep=True
            )
        )


        y_inner_validation = (
            y_outer_train.iloc[
                inner_validation_indices
            ]
            .copy(
                deep=True
            )
        )


        if (
            len(
                x_inner_train
            )
            !=
            len(
                y_inner_train
            )
            or
            len(
                x_inner_validation
            )
            !=
            len(
                y_inner_validation
            )
        ):
            raise (
                MLHyperparameterTuningExecutionError(
                    (
                        "INNER Cross-Validation "
                        "feature/target row counts "
                        "do not match. "
                        f"candidate_index="
                        f"{candidate_index}, "
                        f"fold={fold_index}"
                    )
                )
            )


        try:
            estimator = (
                _build_estimator(
                    contract=
                        candidate_contract
                )
            )

        except ClassicalMLEstimatorError as error:
            raise (
                MLHyperparameterTuningExecutionError(
                    (
                        "Hyperparameter Tuning could "
                        "not construct a candidate "
                        "estimator. "
                        f"candidate_index="
                        f"{candidate_index}, "
                        f"fold={fold_index}"
                    )
                )
            ) from error


        try:
            estimator.fit(
                x_inner_train,
                y_inner_train,
            )

        except Exception as error:
            raise (
                MLHyperparameterTuningExecutionError(
                    (
                        "Hyperparameter candidate "
                        "training failed. "
                        f"candidate_index="
                        f"{candidate_index}, "
                        f"fold={fold_index}"
                    )
                )
            ) from error


        try:
            predictions = (
                estimator.predict(
                    x_inner_validation
                )
            )

        except Exception as error:
            raise (
                MLHyperparameterTuningExecutionError(
                    (
                        "Hyperparameter candidate "
                        "prediction failed. "
                        f"candidate_index="
                        f"{candidate_index}, "
                        f"fold={fold_index}"
                    )
                )
            ) from error


        metrics = (
            _fold_metrics(
                training_contract=
                    candidate_contract,

                y_true=
                    y_inner_validation,

                predictions=
                    predictions,
            )
        )


        candidate_fold_metrics.append(
            metrics
        )


    metric_summary = (
        _summarize_candidate_metrics(
            fold_metrics=
                candidate_fold_metrics,

            problem_type=
                base_contract.problem_type,
        )
    )


    return {
        "candidate_index":
            candidate_index,

        "hyperparameters":
            hyperparameters,

        "training_contract_sha256":
            candidate_sha256,

        "metric_summary":
            metric_summary,
    }


# ============================================================
# RANKING
# ============================================================


def _rank_candidates(
    *,
    evaluated_candidates,
    problem_type: str,
) -> list[
    MLHyperparameterCandidateResult
]:

    if not evaluated_candidates:
        raise (
            MLHyperparameterTuningExecutionError(
                (
                    "Hyperparameter Tuning produced "
                    "no evaluated candidates."
                )
            )
        )


    primary_metric = (
        hyperparameter_primary_metric(
            problem_type=
                problem_type
        )
    )


    metric_direction = (
        hyperparameter_metric_direction(
            problem_type=
                problem_type
        )
    )


    def ranking_key(
        candidate,
    ):

        summary = (
            candidate[
                "metric_summary"
            ][
                primary_metric
            ]
        )


        primary_value = (
            summary.mean

            if (
                metric_direction
                ==
                "minimize"
            )

            else (
                -
                summary.mean
            )
        )


        return (
            primary_value,
            summary.std,
            candidate[
                "candidate_index"
            ],
        )


    ordered = sorted(
        evaluated_candidates,
        key=
            ranking_key,
    )


    ranked_results: list[
        MLHyperparameterCandidateResult
    ] = []


    for (
        rank,
        candidate,
    ) in enumerate(
        ordered,
        start=1,
    ):

        ranked_results.append(
            MLHyperparameterCandidateResult(
                candidate_index=
                    candidate[
                        "candidate_index"
                    ],

                rank=
                    rank,

                hyperparameters=
                    candidate[
                        "hyperparameters"
                    ],

                training_contract_sha256=
                    candidate[
                        "training_contract_sha256"
                    ],

                metric_summary=
                    candidate[
                        "metric_summary"
                    ],
            )
        )


    return ranked_results


# ============================================================
# EXECUTION
# ============================================================


def execute_ml_hyperparameter_tuning(
    *,
    training_contract: MLTrainingContract,
    search_contract: MLHyperparameterSearchContract,
) -> MLHyperparameterSearchResult:
    """
    Exhaustively evaluate the complete DataLens v0.1
    server-owned hyperparameter grid.

    Leakage boundary:

        validated dataset
              |
              v
        OUTER holdout
        /           \
    train            test
      |               |
      v               X
    INNER CV       NEVER USED
      |
      v
    ranking

    The OUTER holdout test split is neither fitted nor scored
    during Hyperparameter Tuning.

    Hyperparameter Tuning v0.1 is evaluation-only.

    It does NOT:
    - persist candidate models;
    - create Model Artifacts;
    - create Experiment Provenance records;
    - score the OUTER holdout test;
    - expose raw rows or predictions.
    """

    contract = (
        MLTrainingContract.model_validate(
            training_contract
        )
    )


    tuning = (
        MLHyperparameterSearchContract
        .model_validate(
            search_contract
        )
    )


    base_training_contract_sha256 = (
        ml_training_contract_sha256(
            contract
        )
    )


    # ========================================================
    # SERVER-OWNED PREPARATION INPUT
    # ========================================================


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


        (
            x_outer_train,
            x_holdout_test,
            y_outer_train,
            y_holdout_test,
            outer_train_groups,
            holdout_test_groups,
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

                return_group_partitions=
                    True,
            )
        )

    except ClassicalMLInputError as error:
        raise (
            MLHyperparameterTuningInputError(
                (
                    "Hyperparameter Tuning refused "
                    "the server-owned ML input. "
                    f"{error}"
                )
            )
        ) from error


    # ========================================================
    # OUTER HOLDOUT INVARIANTS
    # ========================================================


    if (
        len(
            x_outer_train
        )
        !=
        len(
            y_outer_train
        )
        or
        len(
            x_holdout_test
        )
        !=
        len(
            y_holdout_test
        )
    ):
        raise (
            MLHyperparameterTuningExecutionError(
                (
                    "OUTER holdout feature/target "
                    "row counts do not match."
                )
            )
        )


    group_aware = (
        contract
        .split
        .strategy
        ==
        "group_holdout"
    )


    if group_aware:

        if (
            outer_train_groups
            is None
            or
            holdout_test_groups
            is None
        ):
            raise (
                MLHyperparameterTuningExecutionError(
                    (
                        "Entity-aware OUTER holdout "
                        "did not expose its validated "
                        "group partitions."
                    )
                )
            )


        if (
            set(
                outer_train_groups.tolist()
            )
            &
            set(
                holdout_test_groups.tolist()
            )
        ):
            raise (
                MLHyperparameterTuningExecutionError(
                    (
                        "Entity-aware OUTER holdout "
                        "contains overlapping train/"
                        "test entity groups."
                    )
                )
            )


    elif (
        outer_train_groups
        is not None
        or
        holdout_test_groups
        is not None
    ):
        raise (
            MLHyperparameterTuningExecutionError(
                (
                    "Row holdout unexpectedly "
                    "exposed entity group partitions."
                )
            )
        )


    outer_train_rows = int(
        len(
            x_outer_train
        )
    )


    holdout_test_rows = int(
        len(
            x_holdout_test
        )
    )


    # ========================================================
    # INNER CV ? TRAIN ONLY
    # ========================================================


    inner_split_pairs = (
        _build_inner_cv_pairs(
            x_train=
                x_outer_train,

            y_train=
                y_outer_train,

            training_contract=
                contract,

            search_contract=
                tuning,

            groups_train=
                outer_train_groups,
        )
    )


    # ========================================================
    # COMPLETE SERVER-OWNED GRID
    # ========================================================


    try:
        candidates = (
            server_owned_hyperparameter_candidates(
                estimator_key=
                    contract.estimator_key
            )
        )

    except ValueError as error:
        raise (
            MLHyperparameterTuningInputError(
                (
                    "Hyperparameter Tuning does not "
                    "support the requested estimator. "
                    f"estimator_key="
                    f"{contract.estimator_key}"
                )
            )
        ) from error


    if not candidates:
        raise (
            MLHyperparameterTuningExecutionError(
                (
                    "Server-owned Hyperparameter "
                    "Tuning grid is empty."
                )
            )
        )


    evaluated_candidates = []


    for (
        zero_based_candidate_index,
        hyperparameters,
    ) in enumerate(
        candidates
    ):

        candidate_index = (
            zero_based_candidate_index
            +
            1
        )


        evaluated_candidates.append(
            _evaluate_candidate(
                candidate_index=
                    candidate_index,

                hyperparameters=
                    hyperparameters,

                base_contract=
                    contract,

                x_outer_train=
                    x_outer_train,

                y_outer_train=
                    y_outer_train,

                inner_split_pairs=
                    inner_split_pairs,
            )
        )


    # ========================================================
    # DETERMINISTIC RANKING
    # ========================================================


    ranked_results = (
        _rank_candidates(
            evaluated_candidates=
                evaluated_candidates,

            problem_type=
                contract.problem_type,
        )
    )


    if not ranked_results:
        raise (
            MLHyperparameterTuningExecutionError(
                (
                    "Hyperparameter Tuning ranking "
                    "produced no candidates."
                )
            )
        )


    # ========================================================
    # PRIVACY-MINIMAL RESULT
    # ========================================================


    return (
        MLHyperparameterSearchResult(
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

            base_training_contract_sha256=
                base_training_contract_sha256,

            search_strategy=
                tuning.search_strategy,

            validation_strategy=(
                hyperparameter_validation_strategy(
                    problem_type=
                        contract.problem_type,

                    group_aware=
                        group_aware,
                )
            ),

            primary_metric=(
                hyperparameter_primary_metric(
                    problem_type=
                        contract.problem_type
                )
            ),

            metric_direction=(
                hyperparameter_metric_direction(
                    problem_type=
                        contract.problem_type
                )
            ),

            folds=
                tuning.folds,

            shuffle=
                tuning.shuffle,

            random_seed=
                tuning.random_seed,

            outer_train_rows=
                outer_train_rows,

            holdout_test_rows=
                holdout_test_rows,

            candidate_count=
                len(
                    candidates
                ),

            best_candidate_index=(
                ranked_results[
                    0
                ]
                .candidate_index
            ),

            candidate_results=
                ranked_results,
        )
    )
