from __future__ import annotations


import math


from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


from app.deep_learning.autoencoder_bundle import (
    TabularAutoencoderBundleError,
    serialize_tabular_autoencoder_bundle,
)


from app.deep_learning.autoencoder_executor import (
    TabularAutoencoderCoreExecutorError,
    TabularAutoencoderEstimatorError,
    TabularAutoencoderInputError,
    execute_tabular_autoencoder_core,
)


from app.ml.experiment_provenance import (
    MLExperimentProvenanceRecord,
)


from app.ml.model_artifact_store import (
    MLModelArtifactStoreError,
    register_ml_model_artifact,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.ml.anomaly_contracts import (
    MLAnomalyTrainingContract,
)


from app.ml.anomaly_training_input import (
    validate_and_extract_anomaly_x,
)


from app.ml.training_input import (
    MLTrainingInputError,
    load_authorized_ml_dataframe,
)


from app.preparation.analysis_input_handoff import (
    load_validated_analysis_input,
)


# ============================================================
# VERSION
# ============================================================


DL_TABULAR_AUTOENCODER_EXECUTOR_RULE_VERSION = (
    "dl_tabular_autoencoder_executor_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class TabularAutoencoderExecutionError(
    RuntimeError
):
    pass


class TabularAutoencoderExecutionInputError(
    TabularAutoencoderExecutionError
):
    pass


class TabularAutoencoderExecutionEstimatorError(
    TabularAutoencoderExecutionError
):
    pass


# ============================================================
# PRIVACY-MINIMAL RESULT
# ============================================================


class TabularAutoencoderExecutionResult(
    BaseModel
):
    """
    Privacy-minimal anomaly Model Lab execution result.

    Deliberately absent:
    - raw rows;
    - row positions;
    - reconstruction errors per row;
    - anomaly flags per row;
    - fitted preprocessing state;
    - fitted neural-network state;
    - per-epoch losses;
    - accuracy / precision / recall labels.

    Model Artifact metadata and Experiment Provenance are
    exposed through the shared Model Lab lifecycle.

    Raw learned runtime state remains private.
    """

    model_config = ConfigDict(
        extra=
            "forbid",
        frozen=
            True,
        allow_inf_nan=
            False,
    )


    workflow_id: str = Field(
        min_length=
            1,
    )


    dataset_id: str = Field(
        min_length=
            1,
    )


    preparation_session_revision: int = Field(
        ge=
            0,
        strict=
            True,
    )


    problem_type: Literal[
        "anomaly_detection"
    ] = "anomaly_detection"


    estimator_key: Literal[
        "tabular_autoencoder"
    ] = "tabular_autoencoder"


    train_rows: int = Field(
        gt=
            0,
        strict=
            True,
    )


    test_rows: int = Field(
        gt=
            0,
        strict=
            True,
    )


    purged_rows: int = Field(
        default=
            0,
        ge=
            0,
        strict=
            True,
    )


    train_reconstruction_mse: float = Field(
        ge=
            0.0,
    )


    test_reconstruction_mse: float = Field(
        ge=
            0.0,
    )


    threshold_quantile: float = Field(
        gt=
            0.0,
        lt=
            1.0,
    )


    anomaly_threshold: float = Field(
        ge=
            0.0,
    )


    train_anomaly_count: int = Field(
        ge=
            0,
        strict=
            True,
    )


    train_anomaly_rate: float = Field(
        ge=
            0.0,
        le=
            1.0,
    )


    test_anomaly_count: int = Field(
        ge=
            0,
        strict=
            True,
    )


    test_anomaly_rate: float = Field(
        ge=
            0.0,
        le=
            1.0,
    )


    experiment_provenance: (
        MLExperimentProvenanceRecord
    )


    model_artifact: MLModelArtifactRecord


    rule_version: Literal[
        "dl_tabular_autoencoder_executor_v0.1"
    ] = DL_TABULAR_AUTOENCODER_EXECUTOR_RULE_VERSION


    @model_validator(
        mode=
            "after"
    )
    def validate_derived_rates(
        self,
    ) -> "TabularAutoencoderExecutionResult":

        if (
            self.train_anomaly_count
            >
            self.train_rows
        ):

            raise ValueError(
                (
                    "train_anomaly_count cannot "
                    "exceed train_rows."
                )
            )


        if (
            self.test_anomaly_count
            >
            self.test_rows
        ):

            raise ValueError(
                (
                    "test_anomaly_count cannot "
                    "exceed test_rows."
                )
            )


        expected_train_rate = float(
            self.train_anomaly_count
            /
            self.train_rows
        )


        expected_test_rate = float(
            self.test_anomaly_count
            /
            self.test_rows
        )


        if not math.isclose(
            self.train_anomaly_rate,
            expected_train_rate,
            rel_tol=
                1e-12,
            abs_tol=
                1e-12,
        ):

            raise ValueError(
                (
                    "train_anomaly_rate does not "
                    "match train_anomaly_count/train_rows."
                )
            )


        if not math.isclose(
            self.test_anomaly_rate,
            expected_test_rate,
            rel_tol=
                1e-12,
            abs_tol=
                1e-12,
        ):

            raise ValueError(
                (
                    "test_anomaly_rate does not "
                    "match test_anomaly_count/test_rows."
                )
            )


        return self


# ============================================================
# PREPARATION REVISION PIN
# ============================================================


def _validate_preparation_revision_pin(
    *,
    actual_revision: int,
    expected_revision: int | None,
) -> None:

    if expected_revision is None:
        return


    if (
        isinstance(
            expected_revision,
            bool,
        )
        or
        not isinstance(
            expected_revision,
            int,
        )
        or
        expected_revision
        <
        0
    ):

        raise (
            TabularAutoencoderExecutionInputError(
                (
                    "Expected Preparation session "
                    "revision must be a non-negative "
                    "integer."
                )
            )
        )


    if (
        int(
            actual_revision
        )
        !=
        expected_revision
    ):

        raise (
            TabularAutoencoderExecutionInputError(
                (
                    "Tabular autoencoder execution "
                    "refused because the validated "
                    "Preparation revision changed "
                    "before training."
                )
            )
        )


# ============================================================
# PRODUCTION EXECUTION
# ============================================================


def execute_tabular_autoencoder(
    *,
    training_contract: MLAnomalyTrainingContract,
    expected_preparation_session_revision: (
        int
        |
        None
    ) = None,
    execution_device: (
        str
        |
        None
    ) = None,
) -> TabularAutoencoderExecutionResult:
    """
    Execute one anomaly-detection Model Lab request.

    The exact trained core state is serialized and persisted
    once after the single core execution.

    No second split, preprocessing fit, training pass or
    threshold fit is allowed during persistence.
    """

    contract = (
        MLAnomalyTrainingContract
        .model_validate(
            training_contract
        )
    )


    # ========================================================
    # AUTHORIZED PREPARATION INPUT
    # ========================================================


    try:

        (
            dataframe,
            preparation_session_revision,
        ) = (
            load_authorized_ml_dataframe(
                contract=
                    contract,
                handoff_loader=
                    load_validated_analysis_input,
                execution_label=
                    "Tabular Autoencoder",
            )
        )

    except MLTrainingInputError as error:

        raise (
            TabularAutoencoderExecutionInputError(
                str(
                    error
                )
            )
        ) from error


    # ========================================================
    # PREPARATION SNAPSHOT PIN
    # ========================================================


    _validate_preparation_revision_pin(
        actual_revision=
            preparation_session_revision,
        expected_revision=
            expected_preparation_session_revision,
    )


    # ========================================================
    # FEATURE-ONLY AUTHORITY
    # ========================================================


    try:

        x = (
            validate_and_extract_anomaly_x(
                dataframe=
                    dataframe,
                contract=
                    contract,
                execution_label=
                    "Tabular Autoencoder",
            )
        )

    except MLTrainingInputError as error:

        raise (
            TabularAutoencoderExecutionInputError(
                str(
                    error
                )
            )
        ) from error


    # ========================================================
    # PYTORCH CORE
    # ========================================================


    try:

        core_result = (
            execute_tabular_autoencoder_core(
                x=
                    x,
                contract=
                    contract,
                dataframe=
                    dataframe,
                execution_device=
                    execution_device,
            )
        )

    except TabularAutoencoderEstimatorError as error:

        raise (
            TabularAutoencoderExecutionEstimatorError(
                str(
                    error
                )
            )
        ) from error

    except TabularAutoencoderInputError as error:

        raise (
            TabularAutoencoderExecutionInputError(
                str(
                    error
                )
            )
        ) from error

    except TabularAutoencoderCoreExecutorError as error:

        raise (
            TabularAutoencoderExecutionError(
                str(
                    error
                )
            )
        ) from error


    # ========================================================
    # PYTORCH AUTOENCODER MODEL ARTIFACT
    #
    # Persist exactly the model, fitted TRAIN-only
    # preprocessor and frozen TRAIN threshold returned by
    # the single core execution above.
    #
    # No second execution.
    # No second split.
    # No preprocessing refit.
    # No threshold refit.
    # ========================================================


    try:

        model_bytes = (
            serialize_tabular_autoencoder_bundle(
                model=
                    core_result.model,

                preprocessor=
                    core_result.preprocessor,

                training_contract=
                    contract,

                threshold=
                    core_result.threshold,
            )
        )

    except TabularAutoencoderBundleError as error:

        raise (
            TabularAutoencoderExecutionError(
                (
                    "Tabular Autoencoder training completed "
                    "but the fitted PyTorch bundle could "
                    "not be serialized."
                )
            )
        ) from error


    artifact_metrics = {
        "train_reconstruction_mse":
            float(
                core_result
                .train_evaluation
                .mean_loss
            ),

        "test_reconstruction_mse":
            float(
                core_result
                .test_evaluation
                .mean_loss
            ),

        "threshold_quantile":
            float(
                core_result
                .threshold
                .quantile
            ),

        "anomaly_threshold":
            float(
                core_result
                .threshold
                .threshold
            ),

        "train_anomaly_rate":
            float(
                core_result
                .train_anomaly_rate
            ),

        "test_anomaly_rate":
            float(
                core_result
                .test_anomaly_rate
            ),
    }


    try:

        model_artifact = (
            register_ml_model_artifact(
                training_contract=
                    contract,

                metrics=
                    artifact_metrics,

                train_rows=
                    core_result.train_rows,

                test_rows=
                    core_result.test_rows,

                model_bytes=
                    model_bytes,

                serialization_format=
                    "pytorch_bundle",

                preparation_session_revision=
                    preparation_session_revision,
            )
        )

    except MLModelArtifactStoreError as error:

        raise (
            TabularAutoencoderExecutionError(
                (
                    "Tabular Autoencoder training completed "
                    "but the resulting server-owned "
                    "PyTorch Model Artifact could not be "
                    "persisted."
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
            TabularAutoencoderExecutionError(
                (
                    "Current Tabular Autoencoder execution "
                    "did not persist required Experiment "
                    "Provenance."
                )
            )
        )


    # ========================================================
    # PRIVACY-MINIMAL RESULT
    # ========================================================


    return (
        TabularAutoencoderExecutionResult(
            workflow_id=
                contract.workflow_id,
            dataset_id=
                contract.dataset_id,
            preparation_session_revision=
                int(
                    preparation_session_revision
                ),
            problem_type=
                "anomaly_detection",
            estimator_key=
                "tabular_autoencoder",
            train_rows=
                core_result.train_rows,
            test_rows=
                core_result.test_rows,
            purged_rows=
                core_result.purged_rows,
            train_reconstruction_mse=
                float(
                    core_result
                    .train_evaluation
                    .mean_loss
                ),
            test_reconstruction_mse=
                float(
                    core_result
                    .test_evaluation
                    .mean_loss
                ),
            threshold_quantile=
                float(
                    core_result
                    .threshold
                    .quantile
                ),
            anomaly_threshold=
                float(
                    core_result
                    .threshold
                    .threshold
                ),
            train_anomaly_count=
                core_result
                .train_anomaly_count,
            train_anomaly_rate=
                core_result
                .train_anomaly_rate,
            test_anomaly_count=
                core_result
                .test_anomaly_count,
            test_anomaly_rate=
                core_result
                .test_anomaly_rate,
            experiment_provenance=
                experiment_provenance,

            model_artifact=
                model_artifact,

            rule_version=
                DL_TABULAR_AUTOENCODER_EXECUTOR_RULE_VERSION,
        )
    )
