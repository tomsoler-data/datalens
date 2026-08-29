from __future__ import annotations


import math


from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


from app.ml.classical_executor import (
    ClassicalMLExecutorError,
    execute_classical_ml,
)


from app.ml.estimator_contracts import (
    MLEstimatorHyperparameters,
)


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


from app.ml.hyperparameter_tuning import (
    MLHyperparameterPrimaryMetric,
    expected_hyperparameter_metric_names,
)


from app.ml.hyperparameter_tuning_executor import (
    MLHyperparameterTuningError,
    execute_ml_hyperparameter_tuning,
)


from app.ml.tuned_model_promotion import (
    MLTunedModelPromotionAuthorityError,
    MLTunedModelPromotionContract,
    build_promoted_training_contract,
)


# ============================================================
# VERSION
# ============================================================


ML_TUNED_MODEL_PROMOTION_EXECUTOR_RULE_VERSION = (
    "ml_tuned_model_promotion_executor_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLTunedModelPromotionExecutorError(
    RuntimeError
):
    pass


class MLTunedModelPromotionExecutionError(
    MLTunedModelPromotionExecutorError
):
    pass


# ============================================================
# RESULT
# ============================================================


class MLTunedModelPromotionExecutionResult(
    BaseModel
):
    """
    Privacy-minimal result of one complete server-owned
    tuning -> rank-1 promotion -> final holdout training flow.

    No raw rows, predictions, estimator bytes or filesystem
    paths are exposed.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
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


    preparation_session_revision: int = Field(
        ge=0,
    )


    base_training_contract_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


    promoted_training_contract_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


    selected_candidate_index: int = Field(
        ge=1,
    )


    selected_hyperparameters: (
        MLEstimatorHyperparameters
    )


    tuning_primary_metric: (
        MLHyperparameterPrimaryMetric
    )


    tuning_primary_metric_mean: float


    tuning_primary_metric_std: float = Field(
        ge=0.0,
    )


    train_rows: int = Field(
        gt=0,
    )


    test_rows: int = Field(
        gt=0,
    )


    final_metrics: dict[
        str,
        float,
    ]


    model_id: str = Field(
        min_length=1,
    )


    experiment_id: str = Field(
        min_length=1,
    )


    selection_policy: Literal[
        "rank_1_only"
    ] = "rank_1_only"


    holdout_policy: Literal[
        "single_final_evaluation"
    ] = "single_final_evaluation"


    rule_version: Literal[
        "ml_tuned_model_promotion_executor_v0.1"
    ] = ML_TUNED_MODEL_PROMOTION_EXECUTOR_RULE_VERSION


    @field_validator(
        "tuning_primary_metric_mean",
        "tuning_primary_metric_std",
    )
    @classmethod
    def validate_finite_tuning_metric(
        cls,
        value: float,
    ) -> float:

        normalized = float(
            value
        )


        if not math.isfinite(
            normalized
        ):
            raise ValueError(
                (
                    "Tuned Model Promotion tuning "
                    "metric values must be finite."
                )
            )


        return normalized


    @field_validator(
        "final_metrics"
    )
    @classmethod
    def validate_finite_final_metrics(
        cls,
        value: dict[
            str,
            float,
        ],
    ) -> dict[
        str,
        float,
    ]:

        if not value:
            raise ValueError(
                (
                    "Tuned Model Promotion final "
                    "metrics cannot be empty."
                )
            )


        normalized: dict[
            str,
            float,
        ] = {}


        for (
            metric_name,
            raw_value,
        ) in value.items():

            name = str(
                metric_name
            ).strip()


            if not name:
                raise ValueError(
                    (
                        "Final metric names cannot "
                        "be empty."
                    )
                )


            metric_value = float(
                raw_value
            )


            if not math.isfinite(
                metric_value
            ):
                raise ValueError(
                    (
                        "Tuned Model Promotion final "
                        "metrics must be finite."
                    )
                )


            normalized[
                name
            ] = metric_value


        return normalized


    @model_validator(
        mode="after"
    )
    def validate_metric_surface(
        self,
    ) -> (
        "MLTunedModelPromotionExecutionResult"
    ):

        expected = set(
            expected_hyperparameter_metric_names(
                problem_type=
                    self.problem_type
            )
        )


        if (
            set(
                self.final_metrics
            )
            !=
            expected
        ):
            raise ValueError(
                (
                    "Final tuned-model metric surface "
                    "does not match problem_type."
                )
            )


        return self


# ============================================================
# EXECUTION
# ============================================================


def execute_ml_tuned_model_promotion(
    *,
    promotion_contract: (
        MLTunedModelPromotionContract
    ),
) -> MLTunedModelPromotionExecutionResult:
    """
    Execute one complete Tuned Model Promotion.

    Authority flow:

        Promotion Contract
               |
               v
        Hyperparameter Tuning
        OUTER train only
               |
               v
        deterministic rank #1
               |
               v
        promoted Training Contract
               |
               v
        Classical ML final training
        pinned to tuning Preparation revision
               |
               v
        OUTER holdout evaluated once
               |
               v
        Model Artifact + Experiment Provenance

    The caller never supplies the winner or winner
    hyperparameters.
    """

    if isinstance(
        promotion_contract,
        MLTunedModelPromotionContract,
    ):
        payload = (
            promotion_contract.model_dump(
                mode="python"
            )
        )

    else:
        payload = promotion_contract


    contract = (
        MLTunedModelPromotionContract
        .model_validate(
            payload
        )
    )


    base_contract = (
        contract.base_training_contract
    )


    # ========================================================
    # SERVER-OWNED TUNING
    # ========================================================


    try:
        tuning_result = (
            execute_ml_hyperparameter_tuning(
                training_contract=
                    base_contract,

                search_contract=
                    contract.search_contract,
            )
        )

    except MLHyperparameterTuningError as error:
        raise (
            MLTunedModelPromotionExecutionError(
                (
                    "Tuned Model Promotion failed "
                    "during server-owned "
                    "Hyperparameter Tuning."
                )
            )
        ) from error


    # ========================================================
    # SERVER-OWNED RANK #1 MATERIALIZATION
    # ========================================================


    try:
        promoted_contract = (
            build_promoted_training_contract(
                base_training_contract=
                    base_contract,

                tuning_result=
                    tuning_result,
            )
        )

    except (
        MLTunedModelPromotionAuthorityError,
        ValueError,
    ) as error:
        raise (
            MLTunedModelPromotionExecutionError(
                (
                    "Tuned Model Promotion could not "
                    "materialize a trusted rank-1 "
                    "Training Contract."
                )
            )
        ) from error


    winner = (
        tuning_result.candidate_results[
            0
        ]
    )


    promoted_sha256 = (
        ml_training_contract_sha256(
            promoted_contract
        )
    )


    if (
        promoted_sha256
        !=
        winner.training_contract_sha256
    ):
        raise (
            MLTunedModelPromotionExecutionError(
                (
                    "Promoted Training Contract "
                    "fingerprint does not match the "
                    "server-owned tuning winner."
                )
            )
        )


    # ========================================================
    # FINAL TRAINING
    # ========================================================
    #
    # Critical race protection:
    #
    # execute_classical_ml() must load exactly the same
    # Preparation revision used by tuning before it begins
    # schema validation, splitting or fitting.
    #
    # The existing Model Artifact persistence layer retains
    # its later atomic Preparation revision validation as a
    # second race boundary.
    # ========================================================


    try:
        final_execution = (
            execute_classical_ml(
                training_contract=
                    promoted_contract,

                expected_preparation_session_revision=(
                    tuning_result
                    .preparation_session_revision
                ),
            )
        )

    except ClassicalMLExecutorError as error:
        raise (
            MLTunedModelPromotionExecutionError(
                (
                    "Tuned Model Promotion final "
                    "training or persistence failed."
                )
            )
        ) from error


    # ========================================================
    # FINAL IDENTITY
    # ========================================================


    identity_checks = (
        (
            "workflow_id",
            base_contract.workflow_id,
            final_execution.workflow_id,
        ),
        (
            "dataset_id",
            base_contract.dataset_id,
            final_execution.dataset_id,
        ),
        (
            "problem_type",
            base_contract.problem_type,
            final_execution.problem_type,
        ),
        (
            "estimator_key",
            base_contract.estimator_key,
            final_execution.estimator_key,
        ),
    )


    for (
        field_name,
        expected,
        actual,
    ) in identity_checks:

        if actual != expected:
            raise (
                MLTunedModelPromotionExecutionError(
                    (
                        "Final tuned-model execution "
                        "identity does not match the "
                        "promotion authority. "
                        f"field={field_name}"
                    )
                )
            )


    # ========================================================
    # FINAL REVISION
    # ========================================================


    provenance = (
        final_execution
        .experiment_provenance
    )


    artifact = (
        final_execution
        .model_artifact
    )


    if (
        provenance
        .preparation_session_revision
        !=
        tuning_result
        .preparation_session_revision
    ):
        raise (
            MLTunedModelPromotionExecutionError(
                (
                    "Final tuned model was not "
                    "persisted against the same "
                    "Preparation revision used "
                    "during tuning."
                )
            )
        )


    # ========================================================
    # FINAL TRAINING CONTRACT PROVENANCE
    # ========================================================


    if (
        provenance
        .training_contract_sha256
        !=
        promoted_sha256
    ):
        raise (
            MLTunedModelPromotionExecutionError(
                (
                    "Final Experiment Provenance "
                    "does not reference the promoted "
                    "Training Contract SHA-256."
                )
            )
        )


    artifact_contract_sha256 = (
        ml_training_contract_sha256(
            artifact.training_contract
        )
    )


    if (
        artifact_contract_sha256
        !=
        promoted_sha256
    ):
        raise (
            MLTunedModelPromotionExecutionError(
                (
                    "Persisted Model Artifact does "
                    "not contain the promoted "
                    "Training Contract."
                )
            )
        )


    if (
        artifact.experiment_provenance
        !=
        provenance
    ):
        raise (
            MLTunedModelPromotionExecutionError(
                (
                    "Persisted Model Artifact "
                    "Experiment Provenance does not "
                    "match final execution provenance."
                )
            )
        )


    # ========================================================
    # HOLDOUT SHAPE
    # ========================================================


    if (
        final_execution.train_rows
        !=
        tuning_result.outer_train_rows
        or
        final_execution.test_rows
        !=
        tuning_result.holdout_test_rows
    ):
        raise (
            MLTunedModelPromotionExecutionError(
                (
                    "Final tuned-model holdout shape "
                    "does not match the outer holdout "
                    "used to isolate tuning."
                )
            )
        )


    # ========================================================
    # PRIVACY-MINIMAL RESULT
    # ========================================================


    primary_summary = (
        winner.metric_summary[
            tuning_result.primary_metric
        ]
    )


    return (
        MLTunedModelPromotionExecutionResult(
            workflow_id=
                base_contract.workflow_id,

            dataset_id=
                base_contract.dataset_id,

            problem_type=
                base_contract.problem_type,

            estimator_key=
                base_contract.estimator_key,

            preparation_session_revision=(
                tuning_result
                .preparation_session_revision
            ),

            base_training_contract_sha256=(
                tuning_result
                .base_training_contract_sha256
            ),

            promoted_training_contract_sha256=
                promoted_sha256,

            selected_candidate_index=
                winner.candidate_index,

            selected_hyperparameters=
                winner.hyperparameters,

            tuning_primary_metric=
                tuning_result.primary_metric,

            tuning_primary_metric_mean=
                primary_summary.mean,

            tuning_primary_metric_std=
                primary_summary.std,

            train_rows=
                final_execution.train_rows,

            test_rows=
                final_execution.test_rows,

            final_metrics=
                final_execution.metrics,

            model_id=
                artifact.model_id,

            experiment_id=
                provenance.experiment_id,
        )
    )
