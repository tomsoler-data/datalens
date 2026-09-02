from __future__ import annotations


import math


import numpy as np


from sklearn.inspection import (
    permutation_importance,
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


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


from app.ml.model_explainability import (
    MLFeatureImportanceResult,
    MLModelExplainabilityContract,
    MLModelExplainabilityResult,
    explainability_scoring,
)


from app.ml.model_loader import (
    MLModelLoaderError,
    load_trusted_ml_model,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_EXPLAINABILITY_EXECUTOR_RULE_VERSION = (
    "ml_model_explainability_executor_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLModelExplainabilityExecutorError(
    RuntimeError
):
    pass


class MLModelExplainabilityArtifactError(
    MLModelExplainabilityExecutorError
):
    pass


class MLModelExplainabilityInputError(
    MLModelExplainabilityExecutorError
):
    pass


class MLModelExplainabilityExecutionError(
    MLModelExplainabilityExecutorError
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
        raise MLModelExplainabilityInputError(
            (
                f"{field_name} "
                "cannot be empty."
            )
        )


    return normalized


# ============================================================
# FEATURE IMPORTANCE VALIDATION
# ============================================================


def _validated_importance_arrays(
    *,
    means,
    stds,
    expected_features: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    importance_means = np.asarray(
        means,
        dtype=np.float64,
    )


    importance_stds = np.asarray(
        stds,
        dtype=np.float64,
    )


    if (
        importance_means.ndim
        !=
        1
        or
        importance_stds.ndim
        !=
        1
    ):
        raise MLModelExplainabilityExecutionError(
            (
                "Permutation importance returned "
                "invalid array dimensions."
            )
        )


    if (
        len(
            importance_means
        )
        !=
        expected_features
        or
        len(
            importance_stds
        )
        !=
        expected_features
    ):
        raise MLModelExplainabilityExecutionError(
            (
                "Permutation importance feature count "
                "does not match ML Training Contract."
            )
        )


    if not (
        np.isfinite(
            importance_means
        ).all()
    ):
        raise MLModelExplainabilityExecutionError(
            (
                "Permutation importance produced "
                "non-finite mean values."
            )
        )


    if not (
        np.isfinite(
            importance_stds
        ).all()
    ):
        raise MLModelExplainabilityExecutionError(
            (
                "Permutation importance produced "
                "non-finite standard deviations."
            )
        )


    if bool(
        (
            importance_stds
            <
            0.0
        ).any()
    ):
        raise MLModelExplainabilityExecutionError(
            (
                "Permutation importance produced "
                "a negative standard deviation."
            )
        )


    return (
        importance_means,
        importance_stds,
    )


# ============================================================
# RANKING
# ============================================================


def _rank_feature_importances(
    *,
    feature_names: list[
        str
    ],
    importance_means: np.ndarray,
    importance_stds: np.ndarray,
) -> list[
    MLFeatureImportanceResult
]:

    raw = [
        (
            str(
                feature_name
            ),
            float(
                importance_means[
                    index
                ]
            ),
            float(
                importance_stds[
                    index
                ]
            ),
        )

        for (
            index,
            feature_name,
        )
        in enumerate(
            feature_names
        )
    ]


    ranked = sorted(
        raw,
        key=lambda item: (
            -
            item[
                1
            ],
            item[
                2
            ],
            item[
                0
            ],
        ),
    )


    return [
        MLFeatureImportanceResult(
            feature_name=
                feature_name,

            rank=
                rank,

            importance_mean=
                importance_mean,

            importance_std=
                importance_std,
        )

        for (
            rank,
            (
                feature_name,
                importance_mean,
                importance_std,
            ),
        )
        in enumerate(
            ranked,
            start=1,
        )
    ]


# ============================================================
# EXECUTION
# ============================================================


def execute_ml_model_explainability(
    *,
    workflow_id: str,
    model_id: str,
    explainability_contract: (
        MLModelExplainabilityContract
    ),
) -> MLModelExplainabilityResult:
    """
    Explain one persisted trusted DataLens Model Artifact.

    Authority flow:

        workflow_id + model_id
                ?
        trusted SHA-verified Model Artifact reload
                ?
        artifact-owned MLTrainingContract
                ?
        validated current Preparation handoff
                ?
        exact deterministic holdout reconstruction
                ?
        existing fitted estimator
                ?
        permutation importance on x_test / y_test only

    Security / leakage rules
    ------------------------

    This executor MUST NOT:

    - fit or refit the estimator;
    - calculate importance on x_train;
    - accept an arbitrary serialized estimator;
    - accept arbitrary model bytes or filesystem paths;
    - persist raw rows or permuted observations;
    - alter Model Artifact metadata;
    - create a new Experiment Provenance record.
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
        MLModelExplainabilityContract
        .model_validate(
            explainability_contract
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
        raise MLModelExplainabilityArtifactError(
            (
                "Model Explainability refused "
                "because the trusted Model Artifact "
                "could not be restored."
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
        raise MLModelExplainabilityArtifactError(
            (
                "Trusted Model Artifact workflow "
                "does not match explainability request."
            )
        )


    if (
        artifact.model_id
        !=
        normalized_model_id
    ):
        raise MLModelExplainabilityArtifactError(
            (
                "Trusted Model Artifact identity "
                "does not match explainability request."
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
        raise MLModelExplainabilityArtifactError(
            (
                "Trusted Model Artifact contains "
                "an invalid ML Training Contract."
            )
        ) from error


    if (
        training_contract.workflow_id
        !=
        artifact.workflow_id
    ):
        raise MLModelExplainabilityArtifactError(
            (
                "Model Artifact and ML Training "
                "Contract workflow identities differ."
            )
        )


    if (
        training_contract.dataset_id
        !=
        artifact.dataset_id
    ):
        raise MLModelExplainabilityArtifactError(
            (
                "Model Artifact and ML Training "
                "Contract dataset identities differ."
            )
        )


    # ========================================================
    # EXPERIMENT PROVENANCE
    # ========================================================

    provenance = (
        artifact.experiment_provenance
    )


    if provenance is None:
        raise MLModelExplainabilityArtifactError(
            (
                "Model Explainability v0.1 requires "
                "Experiment Provenance on the "
                "trusted Model Artifact."
            )
        )


    expected_training_sha256 = (
        ml_training_contract_sha256(
            training_contract
        )
    )


    if (
        provenance
        .training_contract_sha256
        !=
        expected_training_sha256
    ):
        raise MLModelExplainabilityArtifactError(
            (
                "Experiment Provenance training "
                "contract fingerprint does not match "
                "the trusted Model Artifact."
            )
        )


    if (
        provenance.model_id
        !=
        artifact.model_id
    ):
        raise MLModelExplainabilityArtifactError(
            (
                "Experiment Provenance model identity "
                "does not match Model Artifact."
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

                dataframe=
                    dataframe,
            )
        )

    except ClassicalMLInputError as error:
        raise MLModelExplainabilityInputError(
            (
                "Model Explainability could not "
                "reconstruct the validated ML "
                "holdout from Preparation."
            )
        ) from error


    # ========================================================
    # REVISION PINNING
    # ========================================================

    if (
        current_preparation_revision
        !=
        provenance
        .preparation_session_revision
    ):
        raise MLModelExplainabilityInputError(
            (
                "Preparation revision changed since "
                "the explained Model Artifact was "
                "trained. "
                "artifact_revision="
                f"{provenance.preparation_session_revision}, "
                "current_revision="
                f"{current_preparation_revision}"
            )
        )


    # ========================================================
    # HOLDOUT SHAPE CONSISTENCY
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
        raise MLModelExplainabilityInputError(
            (
                "Reconstructed holdout shape does not "
                "match the persisted Model Artifact. "
                f"artifact_train={artifact.train_rows}, "
                f"reconstructed_train={train_rows}, "
                f"artifact_test={artifact.test_rows}, "
                f"reconstructed_test={test_rows}"
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
        raise MLModelExplainabilityInputError(
            (
                "Reconstructed holdout feature and "
                "target shapes are inconsistent."
            )
        )


    # ========================================================
    # SERVER-OWNED SCORING
    # ========================================================

    scoring = (
        explainability_scoring(
            problem_type=
                training_contract
                .problem_type
        )
    )


    # ========================================================
    # PERMUTATION IMPORTANCE
    #
    # IMPORTANT:
    # only x_test / y_test are supplied.
    # No fit() occurs in this executor.
    # ========================================================

    try:
        permutation_result = (
            permutation_importance(
                loaded_model.estimator,
                x_test,
                y_test,
                scoring=
                    scoring,
                n_repeats=
                    config.n_repeats,
                random_state=
                    config.random_seed,
                n_jobs=
                    1,
            )
        )

    except Exception as error:
        raise MLModelExplainabilityExecutionError(
            (
                "Permutation Feature Importance "
                "execution failed on the deterministic "
                "holdout test set."
            )
        ) from error


    (
        importance_means,
        importance_stds,
    ) = (
        _validated_importance_arrays(
            means=
                permutation_result
                .importances_mean,

            stds=
                permutation_result
                .importances_std,

            expected_features=
                len(
                    training_contract
                    .feature_columns
                ),
        )
    )


    feature_importances = (
        _rank_feature_importances(
            feature_names=list(
                training_contract
                .feature_columns
            ),

            importance_means=
                importance_means,

            importance_stds=
                importance_stds,
        )
    )


    # ========================================================
    # PRIVACY-MINIMAL RESULT
    # ========================================================

    return (
        MLModelExplainabilityResult(
            workflow_id=
                artifact.workflow_id,

            dataset_id=
                artifact.dataset_id,

            model_id=
                artifact.model_id,

            experiment_id=
                provenance
                .experiment_id,

            problem_type=
                training_contract
                .problem_type,

            estimator_key=
                training_contract
                .estimator_key,

            preparation_session_revision=(
                current_preparation_revision
            ),

            training_contract_sha256=(
                expected_training_sha256
            ),

            method=
                config.method,

            scoring=
                scoring,

            n_repeats=
                config.n_repeats,

            random_seed=
                config.random_seed,

            evaluation_rows=
                test_rows,

            feature_importances=
                feature_importances,
        )
    )
