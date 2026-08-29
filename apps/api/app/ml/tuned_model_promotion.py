from __future__ import annotations


from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


from app.ml.hyperparameter_tuning import (
    MLHyperparameterSearchContract,
    MLHyperparameterSearchResult,
    server_owned_hyperparameter_candidates,
)


# ============================================================
# VERSION
# ============================================================


ML_TUNED_MODEL_PROMOTION_RULE_VERSION = (
    "ml_tuned_model_promotion_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLTunedModelPromotionError(
    RuntimeError
):
    pass


class MLTunedModelPromotionAuthorityError(
    MLTunedModelPromotionError
):
    pass


# ============================================================
# CONTRACT
# ============================================================


class MLTunedModelPromotionContract(
    BaseModel
):
    """
    Server-validatable request to turn deterministic
    Hyperparameter Tuning into one final persisted model.

    The caller supplies only:

    - the base MLTrainingContract;
    - the Hyperparameter Search configuration.

    The caller DOES NOT supply:

    - a tuning result;
    - candidate metrics;
    - a candidate index;
    - winner hyperparameters;
    - model bytes;
    - filesystem paths.

    Promotion v0.1 will re-run tuning server-side and must
    promote only the deterministic rank-1 candidate.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    base_training_contract: MLTrainingContract


    search_contract: MLHyperparameterSearchContract


    selection_policy: Literal[
        "rank_1_only"
    ] = "rank_1_only"


    holdout_policy: Literal[
        "single_final_evaluation"
    ] = "single_final_evaluation"


    rule_version: Literal[
        "ml_tuned_model_promotion_v0.1"
    ] = ML_TUNED_MODEL_PROMOTION_RULE_VERSION


    @model_validator(
        mode="after"
    )
    def validate_promotion_contract(
        self,
    ) -> "MLTunedModelPromotionContract":

        # ----------------------------------------------------
        # FORCE NESTED REVALIDATION
        # ----------------------------------------------------
        #
        # Reconstruct from plain Python payloads so even a
        # model instance produced through model_copy(update=...)
        # cannot bypass the nested contract validators.
        # ----------------------------------------------------

        base_contract = (
            MLTrainingContract.model_validate(
                self
                .base_training_contract
                .model_dump(
                    mode="python"
                )
            )
        )


        MLHyperparameterSearchContract.model_validate(
            self
            .search_contract
            .model_dump(
                mode="python"
            )
        )


        # ----------------------------------------------------
        # TUNABLE ESTIMATOR AUTHORITY
        # ----------------------------------------------------

        try:
            candidates = (
                server_owned_hyperparameter_candidates(
                    estimator_key=
                        base_contract.estimator_key
                )
            )

        except ValueError as error:
            raise ValueError(
                (
                    "Tuned Model Promotion v0.1 "
                    "requires an estimator with a "
                    "server-owned Hyperparameter "
                    "Tuning grid. "
                    f"estimator_key="
                    f"{base_contract.estimator_key}"
                )
            ) from error


        if not candidates:
            raise ValueError(
                (
                    "Tuned Model Promotion v0.1 "
                    "requires a non-empty "
                    "server-owned candidate grid."
                )
            )


        return self


    @property
    def base_training_contract_sha256(
        self,
    ) -> str:

        return (
            ml_training_contract_sha256(
                self.base_training_contract
            )
        )


# ============================================================
# INTERNAL SERVER-OWNED WINNER MATERIALIZATION
# ============================================================


def build_promoted_training_contract(
    *,
    base_training_contract: MLTrainingContract,
    tuning_result: MLHyperparameterSearchResult,
) -> MLTrainingContract:
    """
    Materialize the final Training Contract from one
    SERVER-OWNED Hyperparameter Tuning result.

    This function is intentionally not a caller-selection API.

    The executor must obtain tuning_result directly from
    execute_ml_hyperparameter_tuning() during the same
    server-owned promotion flow.

    Security / provenance rules:

    - base contract is revalidated;
    - tuning result is fully revalidated;
    - workflow / dataset / problem / estimator must match;
    - tuning result must reference the exact base contract SHA;
    - rank #1 is selected automatically;
    - caller cannot substitute another candidate;
    - promoted contract is reconstructed from the base contract;
    - winner hyperparameters are the only replaced field;
    - resulting Training Contract SHA must equal the candidate
      SHA recorded during tuning.
    """

    base = (
        MLTrainingContract.model_validate(
            base_training_contract.model_dump(
                mode="python"
            )
        )
    )


    tuning = (
        MLHyperparameterSearchResult.model_validate(
            tuning_result.model_dump(
                mode="python"
            )
        )
    )


    base_sha256 = (
        ml_training_contract_sha256(
            base
        )
    )


    # ========================================================
    # BASE PROVENANCE
    # ========================================================


    if (
        tuning.base_training_contract_sha256
        !=
        base_sha256
    ):
        raise (
            MLTunedModelPromotionAuthorityError(
                (
                    "Hyperparameter Tuning result does "
                    "not belong to the supplied base "
                    "Training Contract."
                )
            )
        )


    # ========================================================
    # IDENTITY AUTHORITY
    # ========================================================


    identity_checks = (
        (
            "workflow_id",
            base.workflow_id,
            tuning.workflow_id,
        ),
        (
            "dataset_id",
            base.dataset_id,
            tuning.dataset_id,
        ),
        (
            "problem_type",
            base.problem_type,
            tuning.problem_type,
        ),
        (
            "estimator_key",
            base.estimator_key,
            tuning.estimator_key,
        ),
    )


    for (
        field_name,
        expected,
        actual,
    ) in identity_checks:

        if actual != expected:
            raise (
                MLTunedModelPromotionAuthorityError(
                    (
                        "Hyperparameter Tuning result "
                        "identity does not match the "
                        "base Training Contract. "
                        f"field={field_name}"
                    )
                )
            )


    # ========================================================
    # RANK #1 ONLY
    # ========================================================


    if not tuning.candidate_results:
        raise (
            MLTunedModelPromotionAuthorityError(
                (
                    "Hyperparameter Tuning result "
                    "contains no candidate results."
                )
            )
        )


    winner = (
        tuning.candidate_results[
            0
        ]
    )


    if winner.rank != 1:
        raise (
            MLTunedModelPromotionAuthorityError(
                (
                    "Tuned Model Promotion can only "
                    "materialize rank-1."
                )
            )
        )


    if (
        winner.candidate_index
        !=
        tuning.best_candidate_index
    ):
        raise (
            MLTunedModelPromotionAuthorityError(
                (
                    "Rank-1 candidate identity does "
                    "not match best_candidate_index."
                )
            )
        )


    # ========================================================
    # MATERIALIZE FROM BASE CONTRACT
    # ========================================================


    payload = (
        base.model_dump(
            mode="python"
        )
    )


    payload[
        "estimator_hyperparameters"
    ] = (
        winner.hyperparameters.model_dump(
            mode="python"
        )
    )


    promoted = (
        MLTrainingContract.model_validate(
            payload
        )
    )


    # ========================================================
    # CANDIDATE SHA AUTHORITY
    # ========================================================


    promoted_sha256 = (
        ml_training_contract_sha256(
            promoted
        )
    )


    if (
        promoted_sha256
        !=
        winner.training_contract_sha256
    ):
        raise (
            MLTunedModelPromotionAuthorityError(
                (
                    "Rank-1 candidate Training "
                    "Contract SHA-256 does not match "
                    "the contract reconstructed from "
                    "the trusted base contract and "
                    "winner hyperparameters."
                )
            )
        )


    return promoted
