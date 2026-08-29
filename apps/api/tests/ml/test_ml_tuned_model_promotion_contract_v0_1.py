from __future__ import annotations


from pydantic import (
    ValidationError,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


from app.ml.hyperparameter_tuning import (
    MLHyperparameterCandidateResult,
    MLHyperparameterMetricSummary,
    MLHyperparameterSearchContract,
    MLHyperparameterSearchResult,
    server_owned_hyperparameter_candidates,
)


from app.ml.tuned_model_promotion import (
    ML_TUNED_MODEL_PROMOTION_RULE_VERSION,
    MLTunedModelPromotionAuthorityError,
    MLTunedModelPromotionContract,
    build_promoted_training_contract,
)


# ============================================================
# HELPERS
# ============================================================


def expect_validation_error(
    factory,
) -> None:

    try:
        factory()

    except ValidationError:
        return


    raise AssertionError(
        "Expected ValidationError."
    )


def expect_authority_error(
    factory,
) -> None:

    try:
        factory()

    except MLTunedModelPromotionAuthorityError:
        return


    raise AssertionError(
        (
            "Expected "
            "MLTunedModelPromotionAuthorityError."
        )
    )


def base_training_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:tuned-promotion",

            dataset_id=
                "dataset:validated",

            problem_type=
                "regression",

            target_column=
                "revenue",

            feature_columns=[
                "age",
                "tenure",
                "segment",
            ],

            categorical_feature_columns=[
                "segment",
            ],

            estimator_key=
                "ridge_regression",

            preprocessing={
                "numeric_imputation":
                    "error",

                "categorical_imputation":
                    "error",

                "categorical_encoding":
                    "one_hot",

                "handle_unknown_categories":
                    "ignore",

                "scale_numeric":
                    True,
            },

            split={
                "strategy":
                    "holdout",

                "test_size":
                    0.20,

                "random_seed":
                    31,

                "shuffle":
                    True,

                "stratify":
                    False,
            },
        )
    )


def candidate_training_contract(
    *,
    base: MLTrainingContract,
    hyperparameters,
) -> MLTrainingContract:

    payload = (
        base.model_dump(
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


def regression_metrics(
    *,
    rmse_mean: float,
    rmse_std: float,
) -> dict[
    str,
    MLHyperparameterMetricSummary,
]:

    return {
        "mae":
            MLHyperparameterMetricSummary(
                mean=
                    rmse_mean
                    *
                    0.8,

                std=
                    rmse_std,
            ),

        "rmse":
            MLHyperparameterMetricSummary(
                mean=
                    rmse_mean,

                std=
                    rmse_std,
            ),

        "r2":
            MLHyperparameterMetricSummary(
                mean=
                    0.90,

                std=
                    0.02,
            ),

        "median_absolute_error":
            MLHyperparameterMetricSummary(
                mean=
                    rmse_mean
                    *
                    0.7,

                std=
                    rmse_std,
            ),

        "explained_variance":
            MLHyperparameterMetricSummary(
                mean=
                    0.91,

                std=
                    0.02,
            ),
    }


def valid_tuning_result(
    *,
    base: MLTrainingContract,
) -> MLHyperparameterSearchResult:

    grid = (
        server_owned_hyperparameter_candidates(
            estimator_key=
                base.estimator_key
        )
    )


    candidate_contracts = [
        candidate_training_contract(
            base=
                base,

            hyperparameters=
                hyperparameters,
        )

        for hyperparameters
        in grid
    ]


    # Deterministic ranking:
    #
    # candidate 2 -> rank 1, RMSE 1.00
    # candidate 1 -> rank 2, RMSE 1.20
    # candidate 3 -> rank 3, RMSE 1.50

    candidate_results = [
        MLHyperparameterCandidateResult(
            candidate_index=
                2,

            rank=
                1,

            hyperparameters=
                grid[
                    1
                ],

            training_contract_sha256=(
                ml_training_contract_sha256(
                    candidate_contracts[
                        1
                    ]
                )
            ),

            metric_summary=(
                regression_metrics(
                    rmse_mean=
                        1.00,

                    rmse_std=
                        0.10,
                )
            ),
        ),

        MLHyperparameterCandidateResult(
            candidate_index=
                1,

            rank=
                2,

            hyperparameters=
                grid[
                    0
                ],

            training_contract_sha256=(
                ml_training_contract_sha256(
                    candidate_contracts[
                        0
                    ]
                )
            ),

            metric_summary=(
                regression_metrics(
                    rmse_mean=
                        1.20,

                    rmse_std=
                        0.08,
                )
            ),
        ),

        MLHyperparameterCandidateResult(
            candidate_index=
                3,

            rank=
                3,

            hyperparameters=
                grid[
                    2
                ],

            training_contract_sha256=(
                ml_training_contract_sha256(
                    candidate_contracts[
                        2
                    ]
                )
            ),

            metric_summary=(
                regression_metrics(
                    rmse_mean=
                        1.50,

                    rmse_std=
                        0.05,
                )
            ),
        ),
    ]


    return (
        MLHyperparameterSearchResult(
            workflow_id=
                base.workflow_id,

            dataset_id=
                base.dataset_id,

            problem_type=
                base.problem_type,

            estimator_key=
                base.estimator_key,

            preparation_session_revision=
                7,

            base_training_contract_sha256=(
                ml_training_contract_sha256(
                    base
                )
            ),

            search_strategy=
                "server_owned_grid",

            validation_strategy=
                "k_fold",

            primary_metric=
                "rmse",

            metric_direction=
                "minimize",

            folds=
                5,

            shuffle=
                True,

            random_seed=
                73,

            outer_train_rows=
                80,

            holdout_test_rows=
                20,

            candidate_count=
                3,

            best_candidate_index=
                2,

            candidate_results=
                candidate_results,
        )
    )


# ============================================================
# CONTRACT
# ============================================================


def test_contract_defaults(
) -> None:

    base = (
        base_training_contract()
    )


    contract = (
        MLTunedModelPromotionContract(
            base_training_contract=
                base,

            search_contract=(
                MLHyperparameterSearchContract(
                    folds=
                        5,

                    random_seed=
                        73,
                )
            ),
        )
    )


    assert (
        contract.selection_policy
        ==
        "rank_1_only"
    )


    assert (
        contract.holdout_policy
        ==
        "single_final_evaluation"
    )


    assert (
        contract.rule_version
        ==
        ML_TUNED_MODEL_PROMOTION_RULE_VERSION
    )


    assert (
        contract.base_training_contract_sha256
        ==
        ml_training_contract_sha256(
            base
        )
    )


def test_contract_is_strict_and_frozen(
) -> None:

    base = (
        base_training_contract()
    )


    search = (
        MLHyperparameterSearchContract()
    )


    forbidden_payloads = [
        {
            "selected_candidate_index":
                2,
        },
        {
            "candidate_index":
                2,
        },
        {
            "tuning_result":
                {},
        },
        {
            "winner_hyperparameters":
                {
                    "alpha":
                        999.0,
                },
        },
        {
            "estimator_hyperparameters":
                {
                    "alpha":
                        999.0,
                },
        },
    ]


    for extra in (
        forbidden_payloads
    ):

        expect_validation_error(
            lambda extra=extra:
                MLTunedModelPromotionContract(
                    base_training_contract=
                        base,

                    search_contract=
                        search,

                    **extra,
                )
        )


    contract = (
        MLTunedModelPromotionContract(
            base_training_contract=
                base,

            search_contract=
                search,
        )
    )


    try:
        contract.selection_policy = (
            "something_else"
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "Tuned Model Promotion "
                "Contract must be frozen."
            )
        )


def test_contract_rejects_estimator_without_server_owned_grid(
) -> None:

    unsupported = (
        MLTrainingContract(
            workflow_id=
                "prep:unsupported",

            dataset_id=
                "dataset:unsupported",

            problem_type=
                "regression",

            target_column=
                "target",

            feature_columns=[
                "x1",
            ],

            estimator_key=
                "unsupported_estimator",
        )
    )


    expect_validation_error(
        lambda:
            MLTunedModelPromotionContract(
                base_training_contract=
                    unsupported,

                search_contract=(
                    MLHyperparameterSearchContract()
                ),
            )
    )


# ============================================================
# RANK-1 MATERIALIZATION
# ============================================================


def test_rank_1_is_materialized_server_side(
) -> None:

    base = (
        base_training_contract()
    )


    tuning = (
        valid_tuning_result(
            base=
                base
        )
    )


    promoted = (
        build_promoted_training_contract(
            base_training_contract=
                base,

            tuning_result=
                tuning,
        )
    )


    winner = (
        tuning.candidate_results[
            0
        ]
    )


    assert (
        winner.rank
        ==
        1
    )


    assert (
        winner.candidate_index
        ==
        tuning.best_candidate_index
        ==
        2
    )


    assert (
        promoted
        .estimator_hyperparameters
        .model_dump(
            mode="json"
        )
        ==
        winner
        .hyperparameters
        .model_dump(
            mode="json"
        )
    )


    assert (
        float(
            promoted
            .estimator_hyperparameters
            .alpha
        )
        ==
        1.0
    )


def test_promoted_contract_preserves_base_authority(
) -> None:

    base = (
        base_training_contract()
    )


    tuning = (
        valid_tuning_result(
            base=
                base
        )
    )


    promoted = (
        build_promoted_training_contract(
            base_training_contract=
                base,

            tuning_result=
                tuning,
        )
    )


    assert (
        promoted.workflow_id
        ==
        base.workflow_id
    )


    assert (
        promoted.dataset_id
        ==
        base.dataset_id
    )


    assert (
        promoted.problem_type
        ==
        base.problem_type
    )


    assert (
        promoted.target_column
        ==
        base.target_column
    )


    assert (
        promoted.feature_columns
        ==
        base.feature_columns
    )


    assert (
        promoted.categorical_feature_columns
        ==
        base.categorical_feature_columns
    )


    assert (
        promoted.estimator_key
        ==
        base.estimator_key
    )


    assert (
        promoted.preprocessing
        ==
        base.preprocessing
    )


    assert (
        promoted.split
        ==
        base.split
    )


def test_promoted_sha_matches_winner_sha(
) -> None:

    base = (
        base_training_contract()
    )


    tuning = (
        valid_tuning_result(
            base=
                base
        )
    )


    promoted = (
        build_promoted_training_contract(
            base_training_contract=
                base,

            tuning_result=
                tuning,
        )
    )


    winner = (
        tuning.candidate_results[
            0
        ]
    )


    assert (
        ml_training_contract_sha256(
            promoted
        )
        ==
        winner.training_contract_sha256
    )


# ============================================================
# BASE CONTRACT IMMUTABILITY
# ============================================================


def test_base_contract_is_never_mutated(
) -> None:

    base = (
        base_training_contract()
    )


    before = (
        base.model_dump(
            mode="json"
        )
    )


    before_sha = (
        ml_training_contract_sha256(
            base
        )
    )


    tuning = (
        valid_tuning_result(
            base=
                base
        )
    )


    _ = (
        build_promoted_training_contract(
            base_training_contract=
                base,

            tuning_result=
                tuning,
        )
    )


    after = (
        base.model_dump(
            mode="json"
        )
    )


    after_sha = (
        ml_training_contract_sha256(
            base
        )
    )


    assert (
        before
        ==
        after
    )


    assert (
        before_sha
        ==
        after_sha
    )


# ============================================================
# FAIL CLOSED ? BASE SHA
# ============================================================


def test_tuning_result_must_match_exact_base_sha(
) -> None:

    base = (
        base_training_contract()
    )


    tuning = (
        valid_tuning_result(
            base=
                base
        )
    )


    tampered = (
        tuning.model_copy(
            update={
                "base_training_contract_sha256":
                    (
                        "f"
                        *
                        64
                    ),
            }
        )
    )


    expect_authority_error(
        lambda:
            build_promoted_training_contract(
                base_training_contract=
                    base,

                tuning_result=
                    tampered,
            )
    )


# ============================================================
# FAIL CLOSED ? IDENTITY
# ============================================================


def test_tuning_identity_must_match_base_contract(
) -> None:

    base = (
        base_training_contract()
    )


    tuning = (
        valid_tuning_result(
            base=
                base
        )
    )


    mismatches = [
        {
            "workflow_id":
                "prep:other",
        },
        {
            "dataset_id":
                "dataset:other",
        },
        {
            "problem_type":
                "classification",
        },
        {
            "estimator_key":
                "linear_regression",
        },
    ]


    for update in mismatches:

        tampered = (
            tuning.model_copy(
                update=
                    update
            )
        )


        try:
            build_promoted_training_contract(
                base_training_contract=
                    base,

                tuning_result=
                    tampered,
            )

        except (
            ValidationError,
            MLTunedModelPromotionAuthorityError,
        ):
            continue


        raise AssertionError(
            (
                "Tuning/base identity mismatch "
                "must fail closed."
            )
        )


# ============================================================
# FAIL CLOSED ? WINNER CONTRACT SHA
# ============================================================


def test_winner_training_contract_sha_is_recomputed(
) -> None:

    base = (
        base_training_contract()
    )


    tuning = (
        valid_tuning_result(
            base=
                base
        )
    )


    candidates = list(
        tuning.candidate_results
    )


    candidates[
        0
    ] = (
        candidates[
            0
        ]
        .model_copy(
            update={
                "training_contract_sha256":
                    (
                        "e"
                        *
                        64
                    ),
            }
        )
    )


    tampered = (
        tuning.model_copy(
            update={
                "candidate_results":
                    candidates,
            }
        )
    )


    expect_authority_error(
        lambda:
            build_promoted_training_contract(
                base_training_contract=
                    base,

                tuning_result=
                    tampered,
            )
    )


# ============================================================
# PRIVACY / AUTHORITY SURFACE
# ============================================================


def test_promotion_contract_contains_no_result_or_model_payload(
) -> None:

    contract = (
        MLTunedModelPromotionContract(
            base_training_contract=(
                base_training_contract()
            ),

            search_contract=(
                MLHyperparameterSearchContract()
            ),
        )
    )


    payload = (
        contract.model_dump(
            mode="json"
        )
    )


    forbidden = {
        "tuning_result",
        "candidate_results",
        "selected_candidate_index",
        "winner_hyperparameters",
        "raw_rows",
        "predictions",
        "holdout_predictions",
        "model_id",
        "experiment_id",
        "model_bytes",
        "model_path",
    }


    assert (
        forbidden.isdisjoint(
            payload
        )
    )


# ============================================================
# VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_TUNED_MODEL_PROMOTION_RULE_VERSION
        ==
        "ml_tuned_model_promotion_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML TUNED MODEL PROMOTION CONTRACT v0.1 ==="
    )


    tests = [
        (
            "Contract defaults and base SHA",
            test_contract_defaults,
        ),
        (
            "Strict frozen no-candidate-override contract",
            test_contract_is_strict_and_frozen,
        ),
        (
            "Unsupported estimator fails closed",
            test_contract_rejects_estimator_without_server_owned_grid,
        ),
        (
            "Rank-1 materialized server-side",
            test_rank_1_is_materialized_server_side,
        ),
        (
            "Base authority preserved",
            test_promoted_contract_preserves_base_authority,
        ),
        (
            "Promoted SHA equals tuning winner SHA",
            test_promoted_sha_matches_winner_sha,
        ),
        (
            "Base Training Contract immutable",
            test_base_contract_is_never_mutated,
        ),
        (
            "Exact base SHA required",
            test_tuning_result_must_match_exact_base_sha,
        ),
        (
            "Tuning/base identity fail-closed",
            test_tuning_identity_must_match_base_contract,
        ),
        (
            "Winner Training Contract SHA recomputed",
            test_winner_training_contract_sha_is_recomputed,
        ),
        (
            "Privacy-minimal authority surface",
            test_promotion_contract_contains_no_result_or_model_payload,
        ),
        (
            "Rule version",
            test_rule_version,
        ),
    ]


    for (
        label,
        test,
    ) in tests:

        test()

        print(
            f"[PASS] {label}"
        )


    print()
    print(
        "PASS - ML Tuned Model Promotion Contract v0.1"
    )


if __name__ == "__main__":
    main()
