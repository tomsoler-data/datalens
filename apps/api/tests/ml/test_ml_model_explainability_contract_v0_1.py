from __future__ import annotations


from pydantic import (
    ValidationError,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.experiment_provenance import (
    canonical_ml_training_contract_json,
    ml_training_contract_sha256,
)


from app.ml.model_explainability import (
    ML_MODEL_EXPLAINABILITY_RULE_VERSION,
    MLFeatureImportanceResult,
    MLModelExplainabilityContract,
    MLModelExplainabilityResult,
    explainability_scoring,
)


# ============================================================
# HELPERS
# ============================================================


def training_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:explainability-contract",

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
                "linear_regression",
        )
    )


def feature_importances(
) -> list[
    MLFeatureImportanceResult
]:

    return [
        MLFeatureImportanceResult(
            feature_name=
                "age",

            rank=
                1,

            importance_mean=
                2.5,

            importance_std=
                0.10,
        ),

        MLFeatureImportanceResult(
            feature_name=
                "tenure",

            rank=
                2,

            importance_mean=
                1.1,

            importance_std=
                0.20,
        ),

        MLFeatureImportanceResult(
            feature_name=
                "segment",

            rank=
                3,

            importance_mean=
                -0.05,

            importance_std=
                0.01,
        ),
    ]


def explainability_result(
    **overrides,
) -> MLModelExplainabilityResult:

    payload = {
        "workflow_id":
            "prep:explainability-contract",

        "dataset_id":
            "dataset:validated",

        "model_id":
            "model:test-explainability",

        "experiment_id":
            (
                "experiment:"
                +
                (
                    "a"
                    *
                    32
                )
            ),

        "problem_type":
            "regression",

        "estimator_key":
            "linear_regression",

        "preparation_session_revision":
            7,

        "training_contract_sha256":
            (
                "b"
                *
                64
            ),

        "method":
            "permutation_importance",

        "scoring":
            "neg_root_mean_squared_error",

        "n_repeats":
            10,

        "random_seed":
            42,

        "evaluation_rows":
            20,

        "feature_importances":
            feature_importances(),
    }


    payload.update(
        overrides
    )


    return (
        MLModelExplainabilityResult(
            **payload
        )
    )


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


# ============================================================
# CONFIGURATION
# ============================================================


def test_contract_defaults(
) -> None:

    contract = (
        MLModelExplainabilityContract()
    )


    assert (
        contract.method
        ==
        "permutation_importance"
    )


    assert (
        contract.n_repeats
        ==
        10
    )


    assert (
        contract.random_seed
        ==
        42
    )


    assert (
        contract.rule_version
        ==
        ML_MODEL_EXPLAINABILITY_RULE_VERSION
    )


def test_contract_is_strict_frozen_and_bounded(
) -> None:

    expect_validation_error(
        lambda:
            MLModelExplainabilityContract(
                unknown_option=True
            )
    )


    expect_validation_error(
        lambda:
            MLModelExplainabilityContract(
                n_repeats=1
            )
    )


    expect_validation_error(
        lambda:
            MLModelExplainabilityContract(
                n_repeats=51
            )
    )


    contract = (
        MLModelExplainabilityContract()
    )


    try:
        contract.n_repeats = 20

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "Explainability contract "
                "must be frozen."
            )
        )


# ============================================================
# SERVER-OWNED SCORING
# ============================================================


def test_scoring_is_server_owned(
) -> None:

    assert (
        explainability_scoring(
            problem_type=
                "regression"
        )
        ==
        "neg_root_mean_squared_error"
    )


    assert (
        explainability_scoring(
            problem_type=
                "classification"
        )
        ==
        "f1_macro"
    )


    try:
        explainability_scoring(
            problem_type=
                "unsupported"
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            (
                "Unsupported explainability "
                "problem type must fail closed."
            )
        )


# ============================================================
# TRAINING CONTRACT FINGERPRINT ISOLATION
# ============================================================


def test_explainability_does_not_change_training_contract_fingerprint(
) -> None:

    contract = (
        training_contract()
    )


    canonical_before = (
        canonical_ml_training_contract_json(
            contract
        )
    )


    sha_before = (
        ml_training_contract_sha256(
            contract
        )
    )


    _ = (
        MLModelExplainabilityContract(
            n_repeats=
                17,

            random_seed=
                123,
        )
    )


    canonical_after = (
        canonical_ml_training_contract_json(
            contract
        )
    )


    sha_after = (
        ml_training_contract_sha256(
            contract
        )
    )


    assert (
        canonical_before
        ==
        canonical_after
    )


    assert (
        sha_before
        ==
        sha_after
    )


    training_payload = (
        contract.model_dump(
            mode="json"
        )
    )


    assert (
        "explainability"
        not in
        training_payload
    )


    assert (
        "model_explainability"
        not in
        training_payload
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================


def test_signed_feature_importance_is_supported(
) -> None:

    result = (
        MLFeatureImportanceResult(
            feature_name=
                "segment",

            rank=
                1,

            importance_mean=
                -0.25,

            importance_std=
                0.03,
        )
    )


    assert (
        result.importance_mean
        ==
        -0.25
    )


def test_non_finite_feature_importance_is_blocked(
) -> None:

    expect_validation_error(
        lambda:
            MLFeatureImportanceResult(
                feature_name=
                    "age",

                rank=
                    1,

                importance_mean=
                    float(
                        "nan"
                    ),

                importance_std=
                    0.0,
            )
    )


    expect_validation_error(
        lambda:
            MLFeatureImportanceResult(
                feature_name=
                    "age",

                rank=
                    1,

                importance_mean=
                    1.0,

                importance_std=
                    float(
                        "inf"
                    ),
            )
    )


    expect_validation_error(
        lambda:
            MLFeatureImportanceResult(
                feature_name=
                    "age",

                rank=
                    1,

                importance_mean=
                    1.0,

                importance_std=
                    -0.1,
            )
    )


# ============================================================
# RESULT
# ============================================================


def test_result_is_strict_and_privacy_minimal(
) -> None:

    result = (
        explainability_result()
    )


    assert (
        result.method
        ==
        "permutation_importance"
    )


    assert (
        result.scoring
        ==
        "neg_root_mean_squared_error"
    )


    assert (
        result.evaluation_rows
        ==
        20
    )


    assert (
        [
            item.feature_name

            for item
            in result.feature_importances
        ]
        ==
        [
            "age",
            "tenure",
            "segment",
        ]
    )


    payload = (
        result.model_dump(
            mode="json"
        )
    )


    forbidden_fields = {
        "raw_rows",
        "predictions",
        "permuted_rows",
        "estimator",
        "model_bytes",
        "model_path",
    }


    assert (
        forbidden_fields
        .isdisjoint(
            payload
        )
    )


    expect_validation_error(
        lambda:
            explainability_result(
                unexpected_field=True
            )
    )


def test_result_scoring_mismatch_fails_closed(
) -> None:

    expect_validation_error(
        lambda:
            explainability_result(
                scoring=
                    "f1_macro"
            )
    )


def test_result_requires_valid_experiment_identity_and_sha(
) -> None:

    expect_validation_error(
        lambda:
            explainability_result(
                experiment_id=
                    "experiment:not-valid"
            )
    )


    expect_validation_error(
        lambda:
            explainability_result(
                training_contract_sha256=
                    "not-a-sha"
            )
    )


# ============================================================
# DETERMINISTIC RANKING
# ============================================================


def test_result_requires_unique_feature_names(
) -> None:

    duplicate = [
        MLFeatureImportanceResult(
            feature_name=
                "age",

            rank=
                1,

            importance_mean=
                2.0,

            importance_std=
                0.1,
        ),

        MLFeatureImportanceResult(
            feature_name=
                "age",

            rank=
                2,

            importance_mean=
                1.0,

            importance_std=
                0.1,
        ),
    ]


    expect_validation_error(
        lambda:
            explainability_result(
                feature_importances=
                    duplicate
            )
    )


def test_result_requires_contiguous_ranks(
) -> None:

    invalid = [
        MLFeatureImportanceResult(
            feature_name=
                "age",

            rank=
                1,

            importance_mean=
                2.0,

            importance_std=
                0.1,
        ),

        MLFeatureImportanceResult(
            feature_name=
                "tenure",

            rank=
                3,

            importance_mean=
                1.0,

            importance_std=
                0.1,
        ),
    ]


    expect_validation_error(
        lambda:
            explainability_result(
                feature_importances=
                    invalid
            )
    )


def test_result_enforces_deterministic_ranking_policy(
) -> None:

    invalid_order = [
        MLFeatureImportanceResult(
            feature_name=
                "tenure",

            rank=
                1,

            importance_mean=
                1.0,

            importance_std=
                0.1,
        ),

        MLFeatureImportanceResult(
            feature_name=
                "age",

            rank=
                2,

            importance_mean=
                2.0,

            importance_std=
                0.1,
        ),
    ]


    expect_validation_error(
        lambda:
            explainability_result(
                feature_importances=
                    invalid_order
            )
    )


def test_ranking_final_tie_breaker_is_lexical(
) -> None:

    valid = [
        MLFeatureImportanceResult(
            feature_name=
                "age",

            rank=
                1,

            importance_mean=
                1.0,

            importance_std=
                0.1,
        ),

        MLFeatureImportanceResult(
            feature_name=
                "tenure",

            rank=
                2,

            importance_mean=
                1.0,

            importance_std=
                0.1,
        ),
    ]


    result = (
        explainability_result(
            feature_importances=
                valid
        )
    )


    assert (
        result.feature_importances[
            0
        ].feature_name
        ==
        "age"
    )


# ============================================================
# VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_MODEL_EXPLAINABILITY_RULE_VERSION
        ==
        "ml_model_explainability_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML MODEL EXPLAINABILITY CONTRACT v0.1 ==="
    )


    tests = [
        (
            "Contract defaults",
            test_contract_defaults,
        ),
        (
            "Strict frozen bounded contract",
            test_contract_is_strict_frozen_and_bounded,
        ),
        (
            "Server-owned scoring",
            test_scoring_is_server_owned,
        ),
        (
            "Training Contract fingerprint isolation",
            test_explainability_does_not_change_training_contract_fingerprint,
        ),
        (
            "Signed feature importance",
            test_signed_feature_importance_is_supported,
        ),
        (
            "Finite feature importance",
            test_non_finite_feature_importance_is_blocked,
        ),
        (
            "Strict privacy-minimal result",
            test_result_is_strict_and_privacy_minimal,
        ),
        (
            "Scoring mismatch fail-closed",
            test_result_scoring_mismatch_fails_closed,
        ),
        (
            "Experiment identity + SHA validation",
            test_result_requires_valid_experiment_identity_and_sha,
        ),
        (
            "Unique feature names",
            test_result_requires_unique_feature_names,
        ),
        (
            "Contiguous ranks",
            test_result_requires_contiguous_ranks,
        ),
        (
            "Deterministic ranking",
            test_result_enforces_deterministic_ranking_policy,
        ),
        (
            "Lexical ranking tie-breaker",
            test_ranking_final_tie_breaker_is_lexical,
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
        "PASS - ML Model Explainability Contract v0.1"
    )


if __name__ == "__main__":
    main()
