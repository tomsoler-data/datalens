from __future__ import annotations


import math


from typing import (
    Any,
)


import numpy as np


from sklearn.metrics import (
    confusion_matrix,
)


from app.ml.classical_executor import (
    ClassicalMLInputError,
    _load_authorized_dataframe,
    _split_dataset,
    _validate_and_extract_xy,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.decision_threshold import (
    MLDecisionThresholdContract,
    MLDecisionThresholdResult,
)


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


from app.ml.model_loader import (
    MLModelLoaderError,
    load_trusted_ml_model,
)


# ============================================================
# VERSION
# ============================================================


ML_DECISION_THRESHOLD_EXECUTOR_RULE_VERSION = (
    "ml_decision_threshold_executor_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLDecisionThresholdExecutorError(
    RuntimeError
):
    pass


class MLDecisionThresholdArtifactError(
    MLDecisionThresholdExecutorError
):
    pass


class MLDecisionThresholdInputError(
    MLDecisionThresholdExecutorError
):
    pass


class MLDecisionThresholdExecutionError(
    MLDecisionThresholdExecutorError
):
    pass


# ============================================================
# IDENTIFIERS
# ============================================================


def _required_identifier(
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
        raise (
            MLDecisionThresholdInputError(
                (
                    f"{field_name} "
                    "cannot be empty."
                )
            )
        )


    return normalized


# ============================================================
# BASIC METRICS
# ============================================================


def _safe_ratio(
    numerator: int,
    denominator: int,
) -> float:

    if denominator <= 0:
        return 0.0


    return float(
        numerator
        /
        denominator
    )


def _f1(
    *,
    precision: float,
    recall: float,
) -> float:

    denominator = (
        precision
        +
        recall
    )


    if denominator <= 0.0:
        return 0.0


    return float(
        2.0
        *
        precision
        *
        recall
        /
        denominator
    )


# ============================================================
# BINARY ESTIMATOR CLASSES
# ============================================================


def _binary_estimator_classes(
    estimator: Any,
) -> tuple[
    list[
        Any
    ],
    list[
        str
    ],
]:
    """
    Resolve binary class ordering exclusively from the trusted
    fitted classifier.

    Decision Threshold v0.1 deliberately fixes:

        negative class = classes_[0]
        positive class = classes_[1]

    Caller-controlled class identities are not accepted.
    """

    raw_classes = getattr(
        estimator,
        "classes_",
        None,
    )


    if raw_classes is None:

        named_steps = getattr(
            estimator,
            "named_steps",
            None,
        )


        if named_steps is not None:

            try:
                final_estimator = (
                    named_steps[
                        "estimator"
                    ]
                )

            except Exception:
                final_estimator = None


            if final_estimator is not None:

                raw_classes = getattr(
                    final_estimator,
                    "classes_",
                    None,
                )


    if raw_classes is None:
        raise (
            MLDecisionThresholdArtifactError(
                (
                    "Trusted classification estimator "
                    "does not expose fitted classes_."
                )
            )
        )


    array = np.asarray(
        raw_classes,
        dtype=object,
    )


    if (
        array.ndim
        !=
        1
    ):
        raise (
            MLDecisionThresholdArtifactError(
                (
                    "Trusted classification estimator "
                    "classes_ must be one-dimensional."
                )
            )
        )


    if (
        len(
            array
        )
        !=
        2
    ):
        raise (
            MLDecisionThresholdArtifactError(
                (
                    "Decision Threshold v0.1 accepts "
                    "binary classification Model "
                    "Artifacts only. "
                    f"class_count={len(array)}"
                )
            )
        )


    raw = [
        value.item()
        if isinstance(
            value,
            np.generic,
        )
        else value

        for value
        in array.tolist()
    ]


    textual = [
        str(
            value
        ).strip()

        for value
        in raw
    ]


    if any(
        not value

        for value
        in textual
    ):
        raise (
            MLDecisionThresholdArtifactError(
                (
                    "Trusted estimator contains an "
                    "empty textual class identity."
                )
            )
        )


    if (
        len(
            set(
                textual
            )
        )
        !=
        2
    ):
        raise (
            MLDecisionThresholdArtifactError(
                (
                    "Distinct trusted estimator classes "
                    "collapse to the same textual "
                    "class identity."
                )
            )
        )


    return (
        raw,
        textual,
    )


# ============================================================
# PREDICT_PROBA INTERFACE
# ============================================================


def _require_predict_proba(
    estimator: Any,
):
    """
    Decision Threshold v0.1 supports probability-capable binary
    classifiers only.

    decision_function is deliberately NOT accepted in v0.1
    because its score scale is estimator-dependent and is not a
    probability threshold in [0, 1].
    """

    predict_proba = getattr(
        estimator,
        "predict_proba",
        None,
    )


    if not callable(
        predict_proba
    ):
        raise (
            MLDecisionThresholdArtifactError(
                (
                    "Decision Threshold v0.1 requires "
                    "a trusted fitted classifier exposing "
                    "callable predict_proba()."
                )
            )
        )


    return predict_proba


# ============================================================
# PROBABILITY VALIDATION
# ============================================================


def _positive_probabilities(
    *,
    predict_proba,
    x_test,
    expected_rows: int,
) -> np.ndarray:
    """
    Execute exactly one holdout predict_proba() call and validate
    standard sklearn binary probability semantics.

    Returned individual probabilities remain internal only.
    """

    try:
        raw_probabilities = (
            predict_proba(
                x_test
            )
        )

    except Exception as error:
        raise (
            MLDecisionThresholdExecutionError(
                (
                    "Trusted classifier predict_proba() "
                    "failed on the deterministic holdout."
                )
            )
        ) from error


    try:
        probabilities = np.asarray(
            raw_probabilities,
            dtype=np.float64,
        )

    except Exception as error:
        raise (
            MLDecisionThresholdExecutionError(
                (
                    "Trusted classifier predict_proba() "
                    "returned non-numeric probabilities."
                )
            )
        ) from error


    if (
        probabilities.ndim
        !=
        2
        or
        probabilities.shape
        !=
        (
            expected_rows,
            2,
        )
    ):
        raise (
            MLDecisionThresholdExecutionError(
                (
                    "Trusted binary classifier "
                    "predict_proba() must return exactly "
                    "(evaluation_rows, 2). "
                    f"expected=({expected_rows}, 2), "
                    f"actual={probabilities.shape}"
                )
            )
        )


    if not (
        np.isfinite(
            probabilities
        )
        .all()
    ):
        raise (
            MLDecisionThresholdExecutionError(
                (
                    "Trusted classifier predict_proba() "
                    "returned non-finite probabilities."
                )
            )
        )


    if (
        (
            probabilities
            <
            0.0
        )
        .any()
        or
        (
            probabilities
            >
            1.0
        )
        .any()
    ):
        raise (
            MLDecisionThresholdExecutionError(
                (
                    "Trusted classifier predict_proba() "
                    "returned values outside [0, 1]."
                )
            )
        )


    row_sums = (
        probabilities.sum(
            axis=1
        )
    )


    if not np.allclose(
        row_sums,
        np.ones(
            expected_rows,
            dtype=np.float64,
        ),
        rtol=1e-9,
        atol=1e-9,
    ):
        raise (
            MLDecisionThresholdExecutionError(
                (
                    "Trusted binary classifier "
                    "predict_proba() rows must sum to 1."
                )
            )
        )


    positive = (
        probabilities[
            :,
            1
        ]
        .copy()
    )


    if (
        positive.shape
        !=
        (
            expected_rows,
        )
    ):
        raise (
            MLDecisionThresholdExecutionError(
                (
                    "Positive probability vector has "
                    "an invalid shape."
                )
            )
        )


    return positive


# ============================================================
# CONFUSION MATRIX + METRICS
# ============================================================


def _build_threshold_result_values(
    *,
    y_test,
    thresholded_predictions: np.ndarray,
    raw_classes: list[
        Any
    ],
    evaluation_rows: int,
) -> tuple[
    list[
        list[
            int
        ]
    ],
    int,
    int,
    int,
    int,
    int,
    int,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:

    try:
        matrix = (
            confusion_matrix(
                y_test,
                thresholded_predictions,
                labels=
                    raw_classes,
            )
        )

    except Exception as error:
        raise (
            MLDecisionThresholdExecutionError(
                (
                    "Decision Threshold confusion matrix "
                    "could not be built from the trusted "
                    "binary class ordering."
                )
            )
        ) from error


    matrix_array = np.asarray(
        matrix,
        dtype=np.int64,
    )


    if (
        matrix_array.shape
        !=
        (
            2,
            2,
        )
    ):
        raise (
            MLDecisionThresholdExecutionError(
                (
                    "Decision Threshold confusion matrix "
                    "must be exactly 2x2."
                )
            )
        )


    matrix_total = int(
        matrix_array.sum()
    )


    if (
        matrix_total
        !=
        evaluation_rows
    ):
        raise (
            MLDecisionThresholdExecutionError(
                (
                    "Decision Threshold confusion matrix "
                    "does not cover the complete "
                    "deterministic holdout. This usually "
                    "means the holdout contains a class "
                    "outside the trusted fitted estimator "
                    "classes_."
                )
            )
        )


    true_negative = int(
        matrix_array[
            0,
            0
        ]
    )


    false_positive = int(
        matrix_array[
            0,
            1
        ]
    )


    false_negative = int(
        matrix_array[
            1,
            0
        ]
    )


    true_positive = int(
        matrix_array[
            1,
            1
        ]
    )


    negative_support = (
        true_negative
        +
        false_positive
    )


    positive_support = (
        false_negative
        +
        true_positive
    )


    precision = (
        _safe_ratio(
            true_positive,
            (
                true_positive
                +
                false_positive
            ),
        )
    )


    recall = (
        _safe_ratio(
            true_positive,
            (
                true_positive
                +
                false_negative
            ),
        )
    )


    f1 = (
        _f1(
            precision=
                precision,

            recall=
                recall,
        )
    )


    specificity = (
        _safe_ratio(
            true_negative,
            (
                true_negative
                +
                false_positive
            ),
        )
    )


    accuracy = (
        _safe_ratio(
            (
                true_positive
                +
                true_negative
            ),
            evaluation_rows,
        )
    )


    supported_recalls: list[
        float
    ] = []


    if (
        negative_support
        >
        0
    ):
        supported_recalls.append(
            specificity
        )


    if (
        positive_support
        >
        0
    ):
        supported_recalls.append(
            recall
        )


    if not supported_recalls:
        raise (
            MLDecisionThresholdExecutionError(
                (
                    "Decision Threshold holdout has no "
                    "supported binary class."
                )
            )
        )


    balanced_accuracy = float(
        sum(
            supported_recalls
        )
        /
        len(
            supported_recalls
        )
    )


    positive_prediction_rate = (
        _safe_ratio(
            (
                true_positive
                +
                false_positive
            ),
            evaluation_rows,
        )
    )


    matrix_values = [
        [
            true_negative,
            false_positive,
        ],
        [
            false_negative,
            true_positive,
        ],
    ]


    return (
        matrix_values,
        negative_support,
        positive_support,
        true_negative,
        false_positive,
        false_negative,
        true_positive,
        precision,
        recall,
        f1,
        specificity,
        accuracy,
        balanced_accuracy,
        positive_prediction_rate,
    )


# ============================================================
# EXECUTOR
# ============================================================


def execute_ml_decision_threshold(
    *,
    workflow_id: str,
    model_id: str,
    threshold_contract: MLDecisionThresholdContract,
) -> MLDecisionThresholdResult:
    """
    Evaluate one caller-requested probability threshold against
    the original deterministic holdout of one trusted persisted
    binary classification Model Artifact.

    Security / evaluation boundary
    ------------------------------

    Caller controls:
    - workflow_id;
    - model_id;
    - one threshold in [0, 1].

    Caller does NOT control:
    - Training Contract;
    - Preparation revision;
    - holdout rows;
    - class identities;
    - positive class;
    - probabilities;
    - predictions;
    - confusion matrix;
    - model bytes / path.

    Execution:
    - trusted server-owned model reload;
    - exact artifact-owned Training Contract;
    - exact Experiment Provenance;
    - current Preparation revision must match;
    - exact train/test split reconstruction;
    - exactly one predict_proba(x_test);
    - classes_[1] is the positive class;
    - probability >= threshold means positive;
    - no fit();
    - no refit();
    - no predict();
    - no persistence.
    """

    normalized_workflow_id = (
        _required_identifier(
            workflow_id,
            field_name=
                "workflow_id",
        )
    )


    normalized_model_id = (
        _required_identifier(
            model_id,
            field_name=
                "model_id",
        )
    )


    try:
        config = (
            MLDecisionThresholdContract
            .model_validate(
                threshold_contract
            )
        )

    except Exception as error:
        raise (
            MLDecisionThresholdInputError(
                (
                    "Decision Threshold request contract "
                    "is invalid."
                )
            )
        ) from error


    # ========================================================
    # TRUSTED MODEL ARTIFACT
    # ========================================================


    try:
        loaded_model = (
            load_trusted_ml_model(
                workflow_id=
                    normalized_workflow_id,

                model_id=
                    normalized_model_id,
            )
        )

    except MLModelLoaderError as error:
        raise (
            MLDecisionThresholdArtifactError(
                (
                    "Decision Threshold refused because "
                    "the trusted Model Artifact could "
                    "not be restored."
                )
            )
        ) from error


    artifact = (
        loaded_model.artifact
    )


    if (
        artifact.workflow_id
        !=
        normalized_workflow_id
    ):
        raise (
            MLDecisionThresholdArtifactError(
                (
                    "Trusted Model Artifact workflow "
                    "does not match threshold request."
                )
            )
        )


    if (
        artifact.model_id
        !=
        normalized_model_id
    ):
        raise (
            MLDecisionThresholdArtifactError(
                (
                    "Trusted Model Artifact identity "
                    "does not match threshold request."
                )
            )
        )


    # ========================================================
    # ARTIFACT-OWNED TRAINING CONTRACT
    # ========================================================


    try:
        training_contract = (
            MLTrainingContract.model_validate(
                artifact.training_contract
            )
        )

    except Exception as error:
        raise (
            MLDecisionThresholdArtifactError(
                (
                    "Trusted Model Artifact contains "
                    "an invalid ML Training Contract."
                )
            )
        ) from error


    if (
        training_contract.workflow_id
        !=
        artifact.workflow_id
    ):
        raise (
            MLDecisionThresholdArtifactError(
                (
                    "Model Artifact and ML Training "
                    "Contract workflow identities differ."
                )
            )
        )


    if (
        training_contract.dataset_id
        !=
        artifact.dataset_id
    ):
        raise (
            MLDecisionThresholdArtifactError(
                (
                    "Model Artifact and ML Training "
                    "Contract dataset identities differ."
                )
            )
        )


    # ========================================================
    # CLASSIFICATION-ONLY
    # ========================================================


    if (
        training_contract.problem_type
        !=
        "classification"
    ):
        raise (
            MLDecisionThresholdArtifactError(
                (
                    "Decision Threshold v0.1 accepts "
                    "classification Model Artifacts only. "
                    "problem_type="
                    f"{training_contract.problem_type}"
                )
            )
        )


    # ========================================================
    # EXPERIMENT PROVENANCE
    # ========================================================


    provenance = (
        artifact.experiment_provenance
    )


    if provenance is None:
        raise (
            MLDecisionThresholdArtifactError(
                (
                    "Decision Threshold v0.1 requires "
                    "Experiment Provenance on the "
                    "trusted Model Artifact."
                )
            )
        )


    expected_training_sha256 = (
        ml_training_contract_sha256(
            training_contract
        )
    )


    if (
        provenance.training_contract_sha256
        !=
        expected_training_sha256
    ):
        raise (
            MLDecisionThresholdArtifactError(
                (
                    "Experiment Provenance Training "
                    "Contract fingerprint does not match "
                    "the trusted Model Artifact."
                )
            )
        )


    if (
        provenance.model_id
        !=
        artifact.model_id
    ):
        raise (
            MLDecisionThresholdArtifactError(
                (
                    "Experiment Provenance model "
                    "identity does not match the trusted "
                    "Model Artifact."
                )
            )
        )


    if (
        provenance.workflow_id
        !=
        artifact.workflow_id
        or
        provenance.dataset_id
        !=
        artifact.dataset_id
    ):
        raise (
            MLDecisionThresholdArtifactError(
                (
                    "Experiment Provenance scope does "
                    "not match the trusted Model Artifact."
                )
            )
        )


    # ========================================================
    # BINARY CLASS ORDER + PROBABILITY CAPABILITY
    # ========================================================


    (
        raw_classes,
        class_labels,
    ) = (
        _binary_estimator_classes(
            loaded_model.estimator
        )
    )


    predict_proba = (
        _require_predict_proba(
            loaded_model.estimator
        )
    )


    negative_class = (
        raw_classes[
            0
        ]
    )


    positive_class = (
        raw_classes[
            1
        ]
    )


    negative_class_label = (
        class_labels[
            0
        ]
    )


    positive_class_label = (
        class_labels[
            1
        ]
    )


    # ========================================================
    # CURRENT SERVER-OWNED PREPARATION INPUT
    # ========================================================


    try:
        (
            dataframe,
            current_preparation_revision,
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

    except ClassicalMLInputError as error:
        raise (
            MLDecisionThresholdInputError(
                (
                    "Decision Threshold could not "
                    "reconstruct the validated ML "
                    "holdout from Preparation."
                )
            )
        ) from error


    # ========================================================
    # PREPARATION REVISION PINNING
    # ========================================================


    if (
        current_preparation_revision
        !=
        provenance.preparation_session_revision
    ):
        raise (
            MLDecisionThresholdInputError(
                (
                    "Preparation revision changed since "
                    "the evaluated Model Artifact was "
                    "trained. "
                    "artifact_revision="
                    f"{provenance.preparation_session_revision}, "
                    "current_revision="
                    f"{current_preparation_revision}"
                )
            )
        )


    # ========================================================
    # HOLDOUT SHAPE
    # ========================================================


    train_rows = int(
        len(
            x_train
        )
    )


    test_rows = int(
        len(
            x_test
        )
    )


    if (
        train_rows
        !=
        artifact.train_rows
        or
        test_rows
        !=
        artifact.test_rows
    ):
        raise (
            MLDecisionThresholdInputError(
                (
                    "Reconstructed holdout shape does "
                    "not match the persisted Model "
                    "Artifact. "
                    f"artifact_train={artifact.train_rows}, "
                    f"reconstructed_train={train_rows}, "
                    f"artifact_test={artifact.test_rows}, "
                    f"reconstructed_test={test_rows}"
                )
            )
        )


    if (
        provenance.train_rows
        !=
        train_rows
        or
        provenance.test_rows
        !=
        test_rows
    ):
        raise (
            MLDecisionThresholdInputError(
                (
                    "Reconstructed holdout shape does "
                    "not match Experiment Provenance."
                )
            )
        )


    if (
        len(
            y_train
        )
        !=
        train_rows
        or
        len(
            y_test
        )
        !=
        test_rows
    ):
        raise (
            MLDecisionThresholdInputError(
                (
                    "Reconstructed holdout features "
                    "and targets have inconsistent "
                    "shapes."
                )
            )
        )


    # ========================================================
    # HOLDOUT PROBABILITY ONLY
    #
    # IMPORTANT:
    # - no fit()
    # - no refit()
    # - no predict()
    # - no x_train score
    # ========================================================


    positive_probabilities = (
        _positive_probabilities(
            predict_proba=
                predict_proba,

            x_test=
                x_test,

            expected_rows=
                test_rows,
        )
    )


    thresholded_predictions = (
        np.where(
            (
                positive_probabilities
                >=
                config.threshold
            ),
            positive_class,
            negative_class,
        )
    )


    # ========================================================
    # BINARY CONFUSION MATRIX + METRICS
    # ========================================================


    (
        matrix_values,
        negative_support,
        positive_support,
        true_negative,
        false_positive,
        false_negative,
        true_positive,
        precision,
        recall,
        f1,
        specificity,
        accuracy,
        balanced_accuracy,
        positive_prediction_rate,
    ) = (
        _build_threshold_result_values(
            y_test=
                y_test,

            thresholded_predictions=(
                thresholded_predictions
            ),

            raw_classes=
                raw_classes,

            evaluation_rows=
                test_rows,
        )
    )


    # ========================================================
    # PRIVACY-MINIMAL RESULT
    # ========================================================


    return (
        MLDecisionThresholdResult(
            workflow_id=
                artifact.workflow_id,

            dataset_id=
                artifact.dataset_id,

            model_id=
                artifact.model_id,

            experiment_id=
                provenance.experiment_id,

            problem_type=
                "classification",

            target_column=
                training_contract.target_column,

            estimator_key=
                training_contract.estimator_key,

            preparation_session_revision=(
                current_preparation_revision
            ),

            training_contract_sha256=(
                expected_training_sha256
            ),

            evaluation_rows=
                test_rows,

            threshold=
                config.threshold,

            negative_class_label=(
                negative_class_label
            ),

            positive_class_label=(
                positive_class_label
            ),

            confusion_matrix=
                matrix_values,

            negative_support=
                negative_support,

            positive_support=
                positive_support,

            true_negative=
                true_negative,

            false_positive=
                false_positive,

            false_negative=
                false_negative,

            true_positive=
                true_positive,

            precision=
                precision,

            recall=
                recall,

            f1=
                f1,

            specificity=
                specificity,

            accuracy=
                accuracy,

            balanced_accuracy=
                balanced_accuracy,

            positive_prediction_rate=(
                positive_prediction_rate
            ),

            method=
                config.method,

            score_source=
                config.score_source,

            positive_class_policy=(
                config
                .positive_class_policy
            ),

            comparison_operator=(
                config
                .comparison_operator
            ),

            threshold_selection_policy=(
                config
                .threshold_selection_policy
            ),

            zero_division_policy=(
                config
                .zero_division_policy
            ),
        )
    )
