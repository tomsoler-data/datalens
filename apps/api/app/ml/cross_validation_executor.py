from __future__ import annotations


import math


import numpy as np
import pandas as pd


from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)


from app.ml.classical_executor import (
    ClassicalMLEstimatorError,
    ClassicalMLExecutorError,
    ClassicalMLInputError,
    _build_estimator,
    _load_authorized_dataframe,
    _validate_and_extract_xy,
    _validate_metrics,
    _validated_group_values,
    _validated_time_values,
)


from app.ml.model_metrics import (
    compute_ml_classification_metrics,
    compute_ml_regression_metrics,
)


from app.ml.contracts import (
    MLGroupHoldoutSplitContract,
    MLPurgedGroupTimeHoldoutSplitContract,
    MLTimeHoldoutSplitContract,
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
    groups: pd.Series | None = None,
    times: pd.Series | None = None,
) -> None:

    folds = (
        cross_validation_contract.folds
    )


    row_count = int(
        len(
            x
        )
    )


    if isinstance(
        training_contract.split,
        MLPurgedGroupTimeHoldoutSplitContract,
    ):

        raise (
            MLCrossValidationInputError(
                (
                    "Purged group + temporal "
                    "Cross-Validation is deferred "
                    "to E15b."
                )
            )
        )


    group_aware = isinstance(
        training_contract.split,
        MLGroupHoldoutSplitContract,
    )


    temporal_aware = isinstance(
        training_contract.split,
        MLTimeHoldoutSplitContract,
    )


    if (
        group_aware
        and
        temporal_aware
    ):

        raise (
            MLCrossValidationInputError(
                (
                    "Group-aware and temporal-aware "
                    "Cross-Validation cannot be "
                    "combined in v0.1."
                )
            )
        )


    # ========================================================
    # TEMPORAL FEASIBILITY
    # ========================================================


    if temporal_aware:

        if groups is not None:

            raise (
                MLCrossValidationInputError(
                    (
                        "Temporal Cross-Validation "
                        "must not receive entity groups."
                    )
                )
            )


        if times is None:

            raise (
                MLCrossValidationInputError(
                    (
                        "Temporal Cross-Validation "
                        "requires validated observation "
                        "timestamps."
                    )
                )
            )


        if cross_validation_contract.shuffle:

            raise (
                MLCrossValidationInputError(
                    (
                        "Temporal Cross-Validation "
                        "requires shuffle=False."
                    )
                )
            )


        if (
            len(
                times
            )
            !=
            row_count
            or
            len(
                y
            )
            !=
            row_count
            or
            not times.index.equals(
                x.index
            )
            or
            not x.index.equals(
                y.index
            )
        ):

            raise (
                MLCrossValidationInputError(
                    (
                        "Temporal Cross-Validation "
                        "timestamp alignment is invalid."
                    )
                )
            )


        if bool(
            times
            .isna()
            .any()
        ):

            raise (
                MLCrossValidationInputError(
                    (
                        "Temporal Cross-Validation "
                        "timestamps contain missing "
                        "values."
                    )
                )
            )


        if not (
            pd.api.types
            .is_datetime64_any_dtype(
                times.dtype
            )
        ):

            raise (
                MLCrossValidationInputError(
                    (
                        "Temporal Cross-Validation "
                        "requires pandas datetime "
                        "timestamps."
                    )
                )
            )


        distinct_timestamp_count = int(
            times.nunique(
                dropna=True
            )
        )


        minimum_timestamps = (
            folds
            +
            1
        )


        if (
            distinct_timestamp_count
            <
            minimum_timestamps
        ):

            raise (
                MLCrossValidationInputError(
                    (
                        "Temporal Cross-Validation "
                        "requires at least folds + 1 "
                        "distinct timestamps. "
                        f"timestamps="
                        f"{distinct_timestamp_count}, "
                        f"folds={folds}"
                    )
                )
            )


        return


    # ========================================================
    # NON-TEMPORAL METADATA GUARD
    # ========================================================


    if times is not None:

        raise (
            MLCrossValidationInputError(
                (
                    "Non-temporal Cross-Validation "
                    "must not receive timestamp "
                    "metadata."
                )
            )
        )


    # ========================================================
    # HISTORICAL GROUP FEASIBILITY
    # ========================================================


    if group_aware:

        if groups is None:

            raise (
                MLCrossValidationInputError(
                    (
                        "Entity-aware Cross-Validation "
                        "requires validated group values."
                    )
                )
            )


        if (
            len(groups)
            !=
            row_count
            or
            not groups.index.equals(
                x.index
            )
            or
            not x.index.equals(
                y.index
            )
        ):

            raise (
                MLCrossValidationInputError(
                    (
                        "Entity-aware Cross-Validation "
                        "group alignment is invalid."
                    )
                )
            )


        group_count = int(
            groups.nunique(
                dropna=True
            )
        )


        if group_count < folds:

            raise (
                MLCrossValidationInputError(
                    (
                        "Entity-aware Cross-Validation "
                        "requires at least one distinct "
                        "entity group per fold. "
                        f"groups={group_count}, "
                        f"folds={folds}"
                    )
                )
            )


    elif groups is not None:

        raise (
            MLCrossValidationInputError(
                (
                    "Row-based Cross-Validation must "
                    "not receive entity group values."
                )
            )
        )


    # ========================================================
    # HISTORICAL ROW / GROUP METRIC FEASIBILITY
    # ========================================================


    if (
        training_contract.problem_type
        ==
        "regression"
    ):

        minimum_rows = (
            folds
            *
            2
        )


        if row_count < minimum_rows:

            raise (
                MLCrossValidationInputError(
                    (
                        "Regression Cross-Validation v0.1 "
                        "requires at least two validation "
                        "observations per fold because the "
                        "metric surface includes R2. "
                        f"rows={row_count}, "
                        f"folds={folds}, "
                        f"minimum_rows={minimum_rows}"
                    )
                )
            )


        return


    class_counts = (
        y.value_counts(
            dropna=False
        )
    )


    if class_counts.empty:

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


    if minimum_class_count < folds:

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


    if group_aware:

        assert groups is not None


        class_group_frame = (
            pd.DataFrame(
                {
                    "target":
                        y.to_numpy(
                            copy=True
                        ),

                    "group":
                        groups.to_numpy(
                            copy=True
                        ),
                }
            )
        )


        class_group_counts = (
            class_group_frame
            .groupby(
                "target",
                dropna=False,
            )["group"]
            .nunique(
                dropna=True
            )
        )


        minimum_class_group_count = int(
            class_group_counts.min()
        )


        if minimum_class_group_count < folds:

            raise (
                MLCrossValidationInputError(
                    (
                        "Stratified entity-aware "
                        "Cross-Validation requires "
                        "every target class to appear "
                        "in at least one distinct entity "
                        "group per fold. "
                        f"minimum_class_group_count="
                        f"{minimum_class_group_count}, "
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

    if isinstance(
        training_contract.split,
        MLPurgedGroupTimeHoldoutSplitContract,
    ):

        raise (
            MLCrossValidationInputError(
                (
                    "Purged group + temporal "
                    "Cross-Validation is deferred "
                    "to E15b."
                )
            )
        )


    temporal_aware = isinstance(
        training_contract.split,
        MLTimeHoldoutSplitContract,
    )


    if temporal_aware:

        if cross_validation_contract.shuffle:

            raise (
                MLCrossValidationInputError(
                    (
                        "TimeSeriesSplit requires "
                        "shuffle=False."
                    )
                )
            )


        return (
            TimeSeriesSplit(
                n_splits=
                    cross_validation_contract.folds,

                gap=
                    0,
            )
        )


    random_state = (
        cross_validation_contract.random_seed

        if cross_validation_contract.shuffle

        else None
    )


    group_aware = isinstance(
        training_contract.split,
        MLGroupHoldoutSplitContract,
    )


    if (
        training_contract.problem_type
        ==
        "regression"
    ):

        if group_aware:

            return (
                GroupKFold(
                    n_splits=
                        cross_validation_contract.folds,

                    shuffle=
                        cross_validation_contract.shuffle,

                    random_state=
                        random_state,
                )
            )


        return (
            KFold(
                n_splits=
                    cross_validation_contract.folds,

                shuffle=
                    cross_validation_contract.shuffle,

                random_state=
                    random_state,
            )
        )


    if group_aware:

        return (
            StratifiedGroupKFold(
                n_splits=
                    cross_validation_contract.folds,

                shuffle=
                    cross_validation_contract.shuffle,

                random_state=
                    random_state,
            )
        )


    return (
        StratifiedKFold(
            n_splits=
                cross_validation_contract.folds,

            shuffle=
                cross_validation_contract.shuffle,

            random_state=
                random_state,
        )
    )


# ============================================================
# SHARED FOLD AUTHORITY
# ============================================================


def _build_cross_validation_pairs(
    *,
    x: pd.DataFrame,
    y: pd.Series,
    training_contract: MLTrainingContract,
    cross_validation_contract: MLCrossValidationContract,
    groups: pd.Series | None = None,
    times: pd.Series | None = None,
):

    _validate_cross_validation_feasibility(
        x=
            x,

        y=
            y,

        training_contract=
            training_contract,

        cross_validation_contract=
            cross_validation_contract,

        groups=
            groups,

        times=
            times,
    )


    splitter = (
        _build_cross_validation_splitter(
            training_contract=
                training_contract,

            cross_validation_contract=
                cross_validation_contract,
        )
    )


    group_aware = isinstance(
        training_contract.split,
        MLGroupHoldoutSplitContract,
    )


    temporal_aware = isinstance(
        training_contract.split,
        MLTimeHoldoutSplitContract,
    )


    try:

        # ====================================================
        # TEMPORAL SPLIT OVER UNIQUE TIMESTAMPS
        # ====================================================

        if temporal_aware:

            assert times is not None


            ordered_positions = (
                np.argsort(
                    times.to_numpy(),
                    kind="stable",
                )
            )


            ordered_times = (
                times.iloc[
                    ordered_positions
                ]
                .reset_index(
                    drop=True
                )
            )


            unique_times = (
                ordered_times
                .drop_duplicates()
                .reset_index(
                    drop=True
                )
            )


            timestamp_split_pairs = list(
                splitter.split(
                    np.arange(
                        len(
                            unique_times
                        )
                    )
                )
            )


            split_pairs = []


            for (
                train_timestamp_indices,
                validation_timestamp_indices,
            ) in timestamp_split_pairs:

                train_timestamp_values = set(
                    unique_times.iloc[
                        train_timestamp_indices
                    ].tolist()
                )


                validation_timestamp_values = set(
                    unique_times.iloc[
                        validation_timestamp_indices
                    ].tolist()
                )


                train_ordered_offsets = (
                    np.flatnonzero(
                        ordered_times
                        .isin(
                            train_timestamp_values
                        )
                        .to_numpy()
                    )
                )


                validation_ordered_offsets = (
                    np.flatnonzero(
                        ordered_times
                        .isin(
                            validation_timestamp_values
                        )
                        .to_numpy()
                    )
                )


                train_indices = (
                    ordered_positions[
                        train_ordered_offsets
                    ]
                )


                validation_indices = (
                    ordered_positions[
                        validation_ordered_offsets
                    ]
                )


                split_pairs.append(
                    (
                        train_indices,
                        validation_indices,
                    )
                )


        # ====================================================
        # HISTORICAL GROUP SPLIT
        # ====================================================

        elif group_aware:

            assert groups is not None


            split_iterator = (
                splitter.split(
                    x,
                    y,
                    groups=
                        groups,
                )
            )


            split_pairs = list(
                split_iterator
            )


        # ====================================================
        # HISTORICAL ROW CLASSIFICATION
        # ====================================================

        elif (
            training_contract.problem_type
            ==
            "classification"
        ):

            split_iterator = (
                splitter.split(
                    x,
                    y,
                )
            )


            split_pairs = list(
                split_iterator
            )


        # ====================================================
        # HISTORICAL ROW REGRESSION
        # ====================================================

        else:

            split_iterator = (
                splitter.split(
                    x
                )
            )


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
        cross_validation_contract.folds
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


    # ========================================================
    # TEMPORAL FOLD INVARIANTS
    # ========================================================


    if temporal_aware:

        assert times is not None


        validation_timestamp_values_seen = set()

        previous_train_timestamp_values = None

        previous_validation_max = None


        expected_classes = (
            set(
                y.tolist()
            )

            if (
                training_contract.problem_type
                ==
                "classification"
            )

            else None
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


            if (
                len(
                    train_indices
                )
                ==
                0
                or
                len(
                    validation_indices
                )
                ==
                0
            ):

                raise (
                    MLCrossValidationExecutionError(
                        (
                            "Temporal Cross-Validation "
                            "produced an empty fold. "
                            f"fold={fold_index}"
                        )
                    )
                )


            train_times = (
                times.iloc[
                    train_indices
                ]
            )


            validation_times = (
                times.iloc[
                    validation_indices
                ]
            )


            train_timestamp_values = set(
                train_times.tolist()
            )


            validation_timestamp_values = set(
                validation_times.tolist()
            )


            if (
                train_timestamp_values
                &
                validation_timestamp_values
            ):

                raise (
                    MLCrossValidationExecutionError(
                        (
                            "Temporal Cross-Validation "
                            "split an equal timestamp "
                            "across train and validation. "
                            f"fold={fold_index}"
                        )
                    )
                )


            if not (
                train_times.max()
                <
                validation_times.min()
            ):

                raise (
                    MLCrossValidationExecutionError(
                        (
                            "Temporal Cross-Validation "
                            "violated the strict "
                            "train_time < validation_time "
                            "boundary. "
                            f"fold={fold_index}"
                        )
                    )
                )


            if (
                validation_timestamp_values_seen
                &
                validation_timestamp_values
            ):

                raise (
                    MLCrossValidationExecutionError(
                        (
                            "Temporal Cross-Validation "
                            "reused a timestamp in "
                            "multiple validation folds."
                        )
                    )
                )


            validation_timestamp_values_seen.update(
                validation_timestamp_values
            )


            if (
                previous_train_timestamp_values
                is not None
            ):

                if not (
                    previous_train_timestamp_values
                    <
                    train_timestamp_values
                ):

                    raise (
                        MLCrossValidationExecutionError(
                            (
                                "Temporal Cross-Validation "
                                "training windows are not "
                                "strictly expanding."
                            )
                        )
                    )


            if (
                previous_validation_max
                is not None
                and
                not (
                    previous_validation_max
                    <
                    validation_times.min()
                )
            ):

                raise (
                    MLCrossValidationExecutionError(
                        (
                            "Temporal validation windows "
                            "are not strictly ordered."
                        )
                    )
                )


            previous_train_timestamp_values = (
                train_timestamp_values
            )


            previous_validation_max = (
                validation_times.max()
            )


            if (
                training_contract.problem_type
                ==
                "regression"
            ):

                if (
                    len(
                        train_indices
                    )
                    <
                    2
                    or
                    len(
                        validation_indices
                    )
                    <
                    2
                ):

                    raise (
                        MLCrossValidationInputError(
                            (
                                "Temporal regression "
                                "Cross-Validation requires "
                                "at least two train rows "
                                "and two validation rows "
                                "per fold because the "
                                "metric surface includes R2. "
                                f"fold={fold_index}"
                            )
                        )
                    )


            elif expected_classes is not None:

                train_classes = set(
                    y.iloc[
                        train_indices
                    ].tolist()
                )


                validation_classes = set(
                    y.iloc[
                        validation_indices
                    ].tolist()
                )


                if (
                    train_classes
                    !=
                    expected_classes
                ):

                    raise (
                        MLCrossValidationInputError(
                            (
                                "Temporal classification "
                                "Cross-Validation produced "
                                "a training fold without "
                                "the complete target class "
                                "set. "
                                f"fold={fold_index}"
                            )
                        )
                    )


                if (
                    validation_classes
                    !=
                    expected_classes
                ):

                    raise (
                        MLCrossValidationInputError(
                            (
                                "Temporal classification "
                                "Cross-Validation produced "
                                "a validation fold without "
                                "the complete target class "
                                "set. "
                                f"fold={fold_index}"
                            )
                        )
                    )


        return split_pairs


    # ========================================================
    # HISTORICAL ROW SPLIT
    # ========================================================


    if not group_aware:

        return split_pairs


    # ========================================================
    # HISTORICAL GROUP FOLD INVARIANTS
    # ========================================================


    assert groups is not None


    all_group_values = set(
        groups.tolist()
    )


    validation_group_values_seen = set()


    expected_classes = (
        set(
            y.tolist()
        )

        if (
            training_contract.problem_type
            ==
            "classification"
        )

        else None
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


        train_group_values = set(
            groups.iloc[
                train_indices
            ].tolist()
        )


        validation_group_values = set(
            groups.iloc[
                validation_indices
            ].tolist()
        )


        if not train_group_values:

            raise (
                MLCrossValidationExecutionError(
                    (
                        "Entity-aware Cross-Validation "
                        "produced a fold with no "
                        "training entity groups. "
                        f"fold={fold_index}"
                    )
                )
            )


        if not validation_group_values:

            raise (
                MLCrossValidationExecutionError(
                    (
                        "Entity-aware Cross-Validation "
                        "produced a fold with no "
                        "validation entity groups. "
                        f"fold={fold_index}"
                    )
                )
            )


        if (
            train_group_values
            &
            validation_group_values
        ):

            raise (
                MLCrossValidationExecutionError(
                    (
                        "Entity-aware Cross-Validation "
                        "produced overlapping train/"
                        "validation entity groups. "
                        f"fold={fold_index}"
                    )
                )
            )


        if (
            validation_group_values_seen
            &
            validation_group_values
        ):

            raise (
                MLCrossValidationExecutionError(
                    (
                        "Entity-aware Cross-Validation "
                        "assigned an entity group to "
                        "multiple validation folds."
                    )
                )
            )


        validation_group_values_seen.update(
            validation_group_values
        )


        if (
            training_contract.problem_type
            ==
            "regression"
            and
            len(
                validation_indices
            )
            <
            2
        ):

            raise (
                MLCrossValidationInputError(
                    (
                        "Entity-aware regression "
                        "Cross-Validation produced "
                        "fewer than two validation "
                        "rows in a fold. "
                        f"fold={fold_index}"
                    )
                )
            )


        if expected_classes is not None:

            train_classes = set(
                y.iloc[
                    train_indices
                ].tolist()
            )


            validation_classes = set(
                y.iloc[
                    validation_indices
                ].tolist()
            )


            if (
                train_classes
                !=
                expected_classes
            ):

                raise (
                    MLCrossValidationInputError(
                        (
                            "Entity-aware classification "
                            "Cross-Validation produced "
                            "a training fold without the "
                            "complete target class set. "
                            f"fold={fold_index}"
                        )
                    )
                )


            if (
                validation_classes
                !=
                expected_classes
            ):

                raise (
                    MLCrossValidationInputError(
                        (
                            "Entity-aware classification "
                            "Cross-Validation produced "
                            "a validation fold without "
                            "the complete target class set. "
                            f"fold={fold_index}"
                        )
                    )
                )


    if (
        validation_group_values_seen
        !=
        all_group_values
    ):

        raise (
            MLCrossValidationExecutionError(
                (
                    "Entity-aware Cross-Validation "
                    "did not assign every entity "
                    "group to exactly one validation "
                    "fold."
                )
            )
        )


    return split_pairs


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
            compute_ml_regression_metrics(
                y_true=
                    y_true,

                predictions=
                    predictions,
            )
        )

    else:
        metrics = (
            compute_ml_classification_metrics(
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


    if isinstance(
        contract.split,
        MLPurgedGroupTimeHoldoutSplitContract,
    ):

        raise (
            MLCrossValidationInputError(
                (
                    "Purged group + temporal "
                    "Cross-Validation is deferred "
                    "to E15b."
                )
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


        groups = (
            _validated_group_values(
                dataframe=
                    dataframe,

                x=
                    x,

                y=
                    y,

                contract=
                    contract,
            )

            if isinstance(
                contract.split,
                MLGroupHoldoutSplitContract,
            )

            else None
        )


        times = (
            _validated_time_values(
                dataframe=
                    dataframe,

                x=
                    x,

                y=
                    y,

                contract=
                    contract,
            )

            if isinstance(
                contract.split,
                MLTimeHoldoutSplitContract,
            )

            else None
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


    split_pairs = (
        _build_cross_validation_pairs(
            x=
                x,

            y=
                y,

            training_contract=
                contract,

            cross_validation_contract=
                cv_contract,

            groups=
                groups,

            times=
                times,
        )
    )


    strategy = (
        cross_validation_strategy(
            problem_type=
                contract.problem_type,

            group_aware=
                groups is not None,

            temporal_aware=
                times is not None,
        )
    )


    fold_results: list[
        MLCrossValidationFoldResult
    ] = []


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
