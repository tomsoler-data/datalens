from __future__ import annotations


import math


from typing import (
    Any,
)


import numpy as np
import pandas as pd


from sklearn.metrics import (
    confusion_matrix,
)


from app.ml.classical_executor import (
    ClassicalMLInputError,
    _load_authorized_dataframe,
    _split_dataset,
    _validate_and_extract_xy,
)


from app.ml.classification_diagnostics import (
    MLClassificationClassDiagnostics,
    MLClassificationDiagnosticsContract,
    MLClassificationDiagnosticsResult,
    MLClassificationMetricAverage,
)


from app.ml.contracts import (
    MLTrainingContract,
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


ML_CLASSIFICATION_DIAGNOSTICS_EXECUTOR_RULE_VERSION = (
    "ml_classification_diagnostics_executor_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLClassificationDiagnosticsExecutorError(
    RuntimeError
):
    pass


class MLClassificationDiagnosticsArtifactError(
    MLClassificationDiagnosticsExecutorError
):
    pass


class MLClassificationDiagnosticsInputError(
    MLClassificationDiagnosticsExecutorError
):
    pass


class MLClassificationDiagnosticsExecutionError(
    MLClassificationDiagnosticsExecutorError
):
    pass


# ============================================================
# TEXT
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
            MLClassificationDiagnosticsInputError(
                (
                    f"{field_name} "
                    "cannot be empty."
                )
            )
        )


    return normalized


# ============================================================
# FLOATS
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


def _assert_metric_close(
    *,
    actual: float,
    expected: float,
    metric_name: str,
) -> None:

    if not math.isclose(
        float(
            actual
        ),
        float(
            expected
        ),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise (
            MLClassificationDiagnosticsArtifactError(
                (
                    "Reconstructed Classification "
                    "Diagnostics metric does not match "
                    "the persisted Model Artifact. "
                    f"metric={metric_name}, "
                    f"artifact={actual}, "
                    f"reconstructed={expected}"
                )
            )
        )


# ============================================================
# ESTIMATOR CLASSES
# ============================================================


def _estimator_classes(
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
    Resolve class ordering exclusively from the trusted fitted
    classifier.

    Pipeline.classes_ is preferred when available.

    The named estimator step is accepted as a defensive fallback
    for compatible DataLens sklearn Pipelines.

    Caller-controlled labels are never accepted.
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
            MLClassificationDiagnosticsArtifactError(
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
        or
        len(
            array
        )
        <
        2
    ):
        raise (
            MLClassificationDiagnosticsArtifactError(
                (
                    "Trusted classification estimator "
                    "classes_ must be one-dimensional "
                    "and contain at least two classes."
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
            MLClassificationDiagnosticsArtifactError(
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
        len(
            textual
        )
    ):
        raise (
            MLClassificationDiagnosticsArtifactError(
                (
                    "Distinct trusted estimator classes "
                    "collapse to duplicate textual "
                    "identities."
                )
            )
        )


    return (
        raw,
        textual,
    )


# ============================================================
# PREDICTIONS
# ============================================================


def _validated_predictions(
    *,
    predictions: Any,
    expected_rows: int,
) -> np.ndarray:

    array = np.asarray(
        predictions,
        dtype=object,
    )


    if (
        array.ndim
        !=
        1
    ):
        raise (
            MLClassificationDiagnosticsExecutionError(
                (
                    "Trusted classifier predict() "
                    "returned a non-vector result."
                )
            )
        )


    if (
        len(
            array
        )
        !=
        expected_rows
    ):
        raise (
            MLClassificationDiagnosticsExecutionError(
                (
                    "Trusted classifier prediction "
                    "count does not match the "
                    "deterministic holdout. "
                    f"expected={expected_rows}, "
                    f"actual={len(array)}"
                )
            )
        )


    return array


# ============================================================
# MATRIX -> DIAGNOSTICS
# ============================================================


def _build_diagnostics(
    *,
    class_labels: list[
        str
    ],
    matrix: np.ndarray,
    evaluation_rows: int,
) -> tuple[
    list[
        list[
            int
        ]
    ],
    list[
        MLClassificationClassDiagnostics
    ],
    float,
    float,
    MLClassificationMetricAverage,
    MLClassificationMetricAverage,
]:

    class_count = len(
        class_labels
    )


    if (
        matrix.ndim
        !=
        2
        or
        matrix.shape
        !=
        (
            class_count,
            class_count,
        )
    ):
        raise (
            MLClassificationDiagnosticsExecutionError(
                (
                    "Confusion matrix shape does not "
                    "match trusted estimator classes."
                )
            )
        )


    matrix_values = [
        [
            int(
                matrix[
                    row_index,
                    column_index,
                ]
            )

            for column_index
            in range(
                class_count
            )
        ]

        for row_index
        in range(
            class_count
        )
    ]


    matrix_total = sum(
        sum(
            row
        )

        for row
        in matrix_values
    )


    if (
        matrix_total
        !=
        evaluation_rows
    ):
        raise (
            MLClassificationDiagnosticsExecutionError(
                (
                    "Confusion matrix does not cover "
                    "the complete deterministic holdout. "
                    "This usually means the holdout "
                    "contains a class outside the trusted "
                    "fitted estimator classes_."
                )
            )
        )


    per_class: list[
        MLClassificationClassDiagnostics
    ] = []


    for class_index in range(
        class_count
    ):

        row_total = sum(
            matrix_values[
                class_index
            ]
        )


        column_total = sum(
            matrix_values[
                row_index
            ][
                class_index
            ]

            for row_index
            in range(
                class_count
            )
        )


        true_positive = (
            matrix_values[
                class_index
            ][
                class_index
            ]
        )


        false_negative = (
            row_total
            -
            true_positive
        )


        false_positive = (
            column_total
            -
            true_positive
        )


        true_negative = (
            evaluation_rows
            -
            true_positive
            -
            false_negative
            -
            false_positive
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


        per_class.append(
            MLClassificationClassDiagnostics(
                class_label=
                    class_labels[
                        class_index
                    ],

                precision=
                    precision,

                recall=
                    recall,

                f1=(
                    _f1(
                        precision=
                            precision,

                        recall=
                            recall,
                    )
                ),

                support=
                    row_total,

                true_positive=
                    true_positive,

                false_positive=
                    false_positive,

                false_negative=
                    false_negative,

                true_negative=
                    true_negative,
            )
        )


    diagonal_total = sum(
        matrix_values[
            class_index
        ][
            class_index
        ]

        for class_index
        in range(
            class_count
        )
    )


    accuracy = (
        _safe_ratio(
            diagonal_total,
            evaluation_rows,
        )
    )


    supported_classes = [
        item

        for item
        in per_class

        if (
            item.support
            >
            0
        )
    ]


    if not supported_classes:
        raise (
            MLClassificationDiagnosticsExecutionError(
                (
                    "Deterministic holdout contains "
                    "no supported classification class."
                )
            )
        )


    balanced_accuracy = float(
        sum(
            item.recall

            for item
            in supported_classes
        )
        /
        len(
            supported_classes
        )
    )


    macro_average = (
        MLClassificationMetricAverage(
            precision=(
                sum(
                    item.precision

                    for item
                    in per_class
                )
                /
                class_count
            ),

            recall=(
                sum(
                    item.recall

                    for item
                    in per_class
                )
                /
                class_count
            ),

            f1=(
                sum(
                    item.f1

                    for item
                    in per_class
                )
                /
                class_count
            ),
        )
    )


    weighted_average = (
        MLClassificationMetricAverage(
            precision=(
                sum(
                    item.precision
                    *
                    item.support

                    for item
                    in per_class
                )
                /
                evaluation_rows
            ),

            recall=(
                sum(
                    item.recall
                    *
                    item.support

                    for item
                    in per_class
                )
                /
                evaluation_rows
            ),

            f1=(
                sum(
                    item.f1
                    *
                    item.support

                    for item
                    in per_class
                )
                /
                evaluation_rows
            ),
        )
    )


    return (
        matrix_values,
        per_class,
        accuracy,
        balanced_accuracy,
        macro_average,
        weighted_average,
    )


# ============================================================
# ARTIFACT METRIC BINDING
# ============================================================


def _verify_persisted_classification_metrics(
    *,
    artifact_metrics: dict[
        str,
        float
    ],
    accuracy: float,
    balanced_accuracy: float,
    macro_average: (
        MLClassificationMetricAverage
    ),
) -> None:
    """
    Bind reconstructed diagnostics back to the five richer
    classification metrics persisted by Classical ML v0.1.

    Extra future metrics are tolerated.

    Missing or inconsistent v0.1 metrics fail closed.
    """

    expected = {
        "accuracy":
            accuracy,

        "f1_macro":
            macro_average.f1,

        "precision_macro":
            macro_average.precision,

        "recall_macro":
            macro_average.recall,

        "balanced_accuracy":
            balanced_accuracy,
    }


    for (
        metric_name,
        reconstructed_value,
    ) in expected.items():

        if (
            metric_name
            not in
            artifact_metrics
        ):
            raise (
                MLClassificationDiagnosticsArtifactError(
                    (
                        "Trusted Model Artifact is missing "
                        "a required persisted classification "
                        "metric. "
                        f"metric={metric_name}"
                    )
                )
            )


        _assert_metric_close(
            actual=
                float(
                    artifact_metrics[
                        metric_name
                    ]
                ),

            expected=
                reconstructed_value,

            metric_name=
                metric_name,
        )


# ============================================================
# EXECUTION
# ============================================================


def execute_ml_classification_diagnostics(
    *,
    workflow_id: str,
    model_id: str,
    diagnostics_contract: (
        MLClassificationDiagnosticsContract
    ),
) -> MLClassificationDiagnosticsResult:
    """
    Evaluate one trusted persisted classification Model Artifact.

    Authority flow
    --------------

        workflow_id + model_id
                |
        trusted SHA-verified Model Artifact reload
                |
        artifact-owned MLTrainingContract
                |
        classification-only gate
                |
        Experiment Provenance SHA binding
                |
        current validated Preparation handoff
                |
        exact deterministic holdout reconstruction
                |
        exact Preparation revision + holdout shape
                |
        existing fitted classifier.predict(x_test)
                |
        estimator-owned classes_ ordering
                |
        confusion matrix + per-class diagnostics
                |
        persisted metric consistency verification
                |
        privacy-minimal result

    Security / leakage rules
    ------------------------

    This executor MUST NOT:

    - fit or refit any estimator;
    - train a new model;
    - accept arbitrary estimator objects;
    - accept arbitrary model bytes or paths;
    - accept caller-supplied class labels;
    - accept caller-supplied predictions;
    - accept caller-supplied confusion matrices;
    - expose individual holdout predictions;
    - persist a new Model Artifact;
    - create a new Experiment Provenance record;
    - inspect the training split for diagnostics.

    Decision thresholds and probabilities intentionally belong to
    a later milestone.
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


    config = (
        MLClassificationDiagnosticsContract
        .model_validate(
            diagnostics_contract
        )
    )


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
            MLClassificationDiagnosticsArtifactError(
                (
                    "Classification Diagnostics refused "
                    "because the trusted Model Artifact "
                    "could not be restored."
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
            MLClassificationDiagnosticsArtifactError(
                (
                    "Trusted Model Artifact workflow "
                    "does not match diagnostics request."
                )
            )
        )


    if (
        artifact.model_id
        !=
        normalized_model_id
    ):
        raise (
            MLClassificationDiagnosticsArtifactError(
                (
                    "Trusted Model Artifact identity "
                    "does not match diagnostics request."
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
            MLClassificationDiagnosticsArtifactError(
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
            MLClassificationDiagnosticsArtifactError(
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
            MLClassificationDiagnosticsArtifactError(
                (
                    "Model Artifact and ML Training "
                    "Contract dataset identities differ."
                )
            )
        )


    # ========================================================
    # CLASSIFICATION-ONLY GATE
    # ========================================================


    if (
        training_contract.problem_type
        !=
        "classification"
    ):
        raise (
            MLClassificationDiagnosticsArtifactError(
                (
                    "Classification Diagnostics v0.1 "
                    "accepts classification Model "
                    "Artifacts only. "
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
            MLClassificationDiagnosticsArtifactError(
                (
                    "Classification Diagnostics v0.1 "
                    "requires Experiment Provenance "
                    "on the trusted Model Artifact."
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
            MLClassificationDiagnosticsArtifactError(
                (
                    "Experiment Provenance Training "
                    "Contract fingerprint does not "
                    "match the trusted Model Artifact."
                )
            )
        )


    if (
        provenance.model_id
        !=
        artifact.model_id
    ):
        raise (
            MLClassificationDiagnosticsArtifactError(
                (
                    "Experiment Provenance model "
                    "identity does not match the "
                    "trusted Model Artifact."
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
            MLClassificationDiagnosticsArtifactError(
                (
                    "Experiment Provenance scope "
                    "does not match the trusted "
                    "Model Artifact."
                )
            )
        )


    # ========================================================
    # FITTED CLASS ORDER
    # ========================================================


    (
        raw_classes,
        class_labels,
    ) = (
        _estimator_classes(
            loaded_model.estimator
        )
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
            MLClassificationDiagnosticsInputError(
                (
                    "Classification Diagnostics "
                    "could not reconstruct the "
                    "validated ML holdout from "
                    "Preparation."
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
            MLClassificationDiagnosticsInputError(
                (
                    "Preparation revision changed "
                    "since the diagnosed Model Artifact "
                    "was trained. "
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
            MLClassificationDiagnosticsInputError(
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
            MLClassificationDiagnosticsInputError(
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
            MLClassificationDiagnosticsInputError(
                (
                    "Reconstructed holdout features "
                    "and targets have inconsistent "
                    "shapes."
                )
            )
        )


    # ========================================================
    # HOLDOUT PREDICTION ONLY
    #
    # IMPORTANT:
    # - no fit()
    # - no refit()
    # - no x_train prediction
    # ========================================================


    try:
        predictions = (
            loaded_model.predict(
                x_test
            )
        )

    except Exception as error:
        raise (
            MLClassificationDiagnosticsExecutionError(
                (
                    "Trusted classifier prediction "
                    "failed on the deterministic "
                    "holdout."
                )
            )
        ) from error


    prediction_array = (
        _validated_predictions(
            predictions=
                predictions,

            expected_rows=
                test_rows,
        )
    )


    # ========================================================
    # CONFUSION MATRIX
    #
    # Rows    = true class
    # Columns = predicted class
    #
    # Label order is fitted estimator classes_ only.
    # ========================================================


    try:
        matrix = (
            confusion_matrix(
                y_test,
                prediction_array,
                labels=
                    raw_classes,
            )
        )

    except Exception as error:
        raise (
            MLClassificationDiagnosticsExecutionError(
                (
                    "Confusion matrix could not be "
                    "built from the deterministic "
                    "holdout and trusted estimator "
                    "class order."
                )
            )
        ) from error


    (
        matrix_values,
        per_class,
        accuracy,
        balanced_accuracy,
        macro_average,
        weighted_average,
    ) = (
        _build_diagnostics(
            class_labels=
                class_labels,

            matrix=
                np.asarray(
                    matrix,
                    dtype=np.int64,
                ),

            evaluation_rows=
                test_rows,
        )
    )


    # ========================================================
    # BIND BACK TO PERSISTED TRAINING METRICS
    # ========================================================


    _verify_persisted_classification_metrics(
        artifact_metrics=
            artifact.metrics,

        accuracy=
            accuracy,

        balanced_accuracy=
            balanced_accuracy,

        macro_average=
            macro_average,
    )


    # ========================================================
    # PRIVACY-MINIMAL RESULT
    # ========================================================


    return (
        MLClassificationDiagnosticsResult(
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

            class_count=
                len(
                    class_labels
                ),

            class_labels=
                class_labels,

            confusion_matrix=
                matrix_values,

            per_class=
                per_class,

            accuracy=
                accuracy,

            balanced_accuracy=
                balanced_accuracy,

            macro_average=
                macro_average,

            weighted_average=
                weighted_average,

            method=
                config.method,

            label_order_policy=
                config.label_order_policy,

            zero_division_policy=
                config.zero_division_policy,
        )
    )
