from __future__ import annotations


from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from app.deep_learning.model_bundle import (
    TabularMLPBundleError,
    serialize_tabular_mlp_bundle,
)


from app.deep_learning.tabular_executor import (
    TabularMLPCoreExecutorError,
    TabularMLPEstimatorError,
    TabularMLPInputError,
    execute_tabular_mlp_core,
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


from app.ml.baseline import (
    MLBaselineComparisonResult,
    MLBaselineError,
    MLBaselineEvaluationResult,
    build_ml_baseline_evaluation,
    build_ml_baseline_predictions,
    compare_model_to_baseline,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.model_metrics import (
    MLModelMetricsError,
    compute_ml_regression_metrics,
    project_ml_baseline_metrics_v0_1,
)


from app.ml.training_input import (
    MLTrainingInputError,
    load_authorized_ml_dataframe,
    validate_and_extract_ml_xy,
)


from app.preparation.analysis_input_handoff import (
    load_validated_analysis_input,
)


# ============================================================
# VERSION
# ============================================================


DL_TABULAR_MLP_EXECUTOR_RULE_VERSION = (
    "dl_tabular_mlp_executor_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class TabularMLPExecutionError(
    RuntimeError
):
    pass


class TabularMLPExecutionInputError(
    TabularMLPExecutionError
):
    pass


class TabularMLPExecutionEstimatorError(
    TabularMLPExecutionError
):
    pass


# ============================================================
# PRIVACY-MINIMAL PRODUCTION RESULT
# ============================================================


class TabularMLPExecutionResult(
    BaseModel
):
    """
    Privacy-minimal Model Lab Deep Learning result.

    Deliberately absent:
    - raw train/test rows;
    - row positions;
    - predictions;
    - fitted preprocessing state;
    - fitted neural-network state;
    - per-epoch losses.

    Model Artifact metadata and Experiment Provenance are
    exposed, matching the Classical Model Lab lifecycle.

    Raw learned runtime state remains private.
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


    preparation_session_revision: int = Field(
        ge=0,
    )


    problem_type: Literal[
        "regression",
    ]


    estimator_key: Literal[
        "tabular_mlp_regressor",
    ]


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
        "dl_tabular_mlp_executor_v0.1"
    ] = DL_TABULAR_MLP_EXECUTOR_RULE_VERSION


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
            TabularMLPExecutionInputError(
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
            TabularMLPExecutionInputError(
                (
                    "Tabular MLP execution refused "
                    "because the validated Preparation "
                    "revision changed before training. "
                    "A server-owned caller requested "
                    "execution against an earlier "
                    "Preparation snapshot."
                )
            )
        )


# ============================================================
# EXECUTION
# ============================================================


def execute_tabular_mlp(
    *,
    training_contract: MLTrainingContract,
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
) -> TabularMLPExecutionResult:
    """
    Execute one Model Lab tabular MLP regression request.

    Authority flow:

        MLTrainingContract
                ?
        validated Preparation handoff
                ?
        shared ML input validation
                ?
        shared exact holdout authority
                ?
        leakage-safe preprocessing
                ?
        PyTorch MLP core
                ?
        canonical Model Lab metrics
                ?
        Model Lab baseline

    execution_device is server-owned execution state.
    It is not part of the client Training Contract.
    """

    contract = (
        MLTrainingContract
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
                contract=contract,
                handoff_loader=
                    load_validated_analysis_input,
                execution_label=
                    "Tabular MLP",
            )
        )

    except MLTrainingInputError as error:

        raise (
            TabularMLPExecutionInputError(
                str(
                    error
                )
            )
        ) from error


    # ========================================================
    # SNAPSHOT PIN BEFORE TRAINING
    # ========================================================


    _validate_preparation_revision_pin(
        actual_revision=
            preparation_session_revision,

        expected_revision=
            expected_preparation_session_revision,
    )


    # ========================================================
    # SHARED MODEL LAB X / Y AUTHORITY
    # ========================================================


    try:

        (
            x,
            y,
        ) = (
            validate_and_extract_ml_xy(
                dataframe=dataframe,
                contract=contract,
                execution_label=
                    "Tabular MLP",
            )
        )

    except MLTrainingInputError as error:

        raise (
            TabularMLPExecutionInputError(
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
            execute_tabular_mlp_core(
                x=x,
                y=y,
                contract=contract,
                dataframe=dataframe,
                execution_device=
                    execution_device,
            )
        )

    except TabularMLPEstimatorError as error:

        raise (
            TabularMLPExecutionEstimatorError(
                str(
                    error
                )
            )
        ) from error

    except TabularMLPInputError as error:

        raise (
            TabularMLPExecutionInputError(
                str(
                    error
                )
            )
        ) from error

    except TabularMLPCoreExecutorError as error:

        raise (
            TabularMLPExecutionError(
                str(
                    error
                )
            )
        ) from error


    # ========================================================
    # EXACT BASELINE POPULATION
    #
    # Reuse the exact partition already used by the MLP.
    # No second split.
    # ========================================================


    y_train = (
        y.iloc[
            list(
                core_result
                .partition
                .train_positions
            )
        ]
        .copy(
            deep=True
        )
    )


    y_test = (
        y.iloc[
            list(
                core_result
                .partition
                .test_positions
            )
        ]
        .copy(
            deep=True
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
                    core_result.test_rows,
            )
        )


        baseline_full_metrics = (
            compute_ml_regression_metrics(
                y_true=
                    y_test,

                predictions=(
                    baseline_prediction_bundle
                    .predictions
                ),
            )
        )


        baseline_metrics = (
            project_ml_baseline_metrics_v0_1(
                problem_type=
                    contract.problem_type,

                metrics=
                    baseline_full_metrics,
            )
        )


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
                    core_result.train_rows,

                test_rows=
                    core_result.test_rows,
            )
        )


        baseline_comparison = (
            compare_model_to_baseline(
                problem_type=
                    contract.problem_type,

                model_metrics=
                    core_result.metrics,

                baseline_metrics=
                    baseline.metrics,
            )
        )

    except (
        MLBaselineError,
        MLModelMetricsError,
    ) as error:

        raise (
            TabularMLPExecutionError(
                (
                    "Tabular MLP baseline evaluation "
                    "or comparison failed."
                )
            )
        ) from error


    # ========================================================
    # PYTORCH MODEL ARTIFACT
    #
    # The exact trained model and fitted TRAIN-only
    # preprocessor returned by the core executor are serialized
    # once into the DataLens-owned PyTorch bundle.
    #
    # No second fit.
    # No second split.
    # ========================================================


    try:

        model_bytes = (
            serialize_tabular_mlp_bundle(
                model=
                    core_result.model,

                preprocessor=
                    core_result.preprocessor,

                training_contract=
                    contract,
            )
        )

    except TabularMLPBundleError as error:

        raise (
            TabularMLPExecutionError(
                (
                    "Tabular MLP training completed "
                    "but the fitted PyTorch bundle "
                    "could not be serialized."
                )
            )
        ) from error


    try:

        model_artifact = (
            register_ml_model_artifact(
                training_contract=
                    contract,

                metrics=
                    core_result.metrics,

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
            TabularMLPExecutionError(
                (
                    "Tabular MLP training completed "
                    "but the resulting server-owned "
                    "PyTorch Model Artifact could "
                    "not be persisted."
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
            TabularMLPExecutionError(
                (
                    "Current Tabular MLP execution "
                    "did not persist required "
                    "Experiment Provenance."
                )
            )
        )


    # ========================================================
    # PRIVACY-MINIMAL RESULT
    # ========================================================


    return (
        TabularMLPExecutionResult(
            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,

            preparation_session_revision=int(
                preparation_session_revision
            ),

            problem_type=
                "regression",

            estimator_key=
                "tabular_mlp_regressor",

            train_rows=
                core_result.train_rows,

            test_rows=
                core_result.test_rows,

            purged_rows=
                core_result.purged_rows,

            metrics=dict(
                core_result.metrics
            ),

            baseline=
                baseline,

            baseline_comparison=
                baseline_comparison,

            experiment_provenance=
                experiment_provenance,

            model_artifact=
                model_artifact,

            rule_version=
                DL_TABULAR_MLP_EXECUTOR_RULE_VERSION,
        )
    )
