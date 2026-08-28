from __future__ import annotations


from pydantic import (
    ValidationError,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.cross_validation import (
    ML_CROSS_VALIDATION_RULE_VERSION,
    MLCrossValidationContract,
    MLCrossValidationEvaluationResult,
    MLCrossValidationFoldResult,
    MLCrossValidationMetricSummary,
    cross_validation_strategy,
)


from app.ml.experiment_provenance import (
    canonical_ml_training_contract_json,
    ml_training_contract_sha256,
)


# ============================================================
# HELPERS
# ============================================================


def training_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:cv-contract",

            dataset_id=
                "dataset:validated",

            problem_type=
                "regression",

            target_column=
                "revenue",

            feature_columns=[
                "age",
                "tenure",
            ],

            estimator_key=
                "linear_regression",
        )
    )


def richer_metrics(
    offset: float = 0.0,
) -> dict[
    str,
    float,
]:

    return {
        "mae":
            1.0
            +
            offset,

        "rmse":
            2.0
            +
            offset,

        "r2":
            0.8
            -
            (
                offset
                *
                0.01
            ),

        "median_absolute_error":
            0.9
            +
            offset,

        "explained_variance":
            0.85
            -
            (
                offset
                *
                0.01
            ),
    }


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
# TESTS
# ============================================================


def test_cross_validation_contract_defaults(
) -> None:

    contract = (
        MLCrossValidationContract()
    )


    assert (
        contract.folds
        ==
        5
    )


    assert (
        contract.shuffle
        is True
    )


    assert (
        contract.random_seed
        ==
        42
    )


    assert (
        contract.rule_version
        ==
        ML_CROSS_VALIDATION_RULE_VERSION
    )


def test_cross_validation_fold_bounds(
) -> None:

    expect_validation_error(
        lambda:
            MLCrossValidationContract(
                folds=1
            )
    )


    expect_validation_error(
        lambda:
            MLCrossValidationContract(
                folds=21
            )
    )


def test_cross_validation_contract_is_strict_and_frozen(
) -> None:

    expect_validation_error(
        lambda:
            MLCrossValidationContract(
                unknown_option=True
            )
    )


    contract = (
        MLCrossValidationContract()
    )


    try:
        contract.folds = 10

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "Cross-validation contract "
                "must be frozen."
            )
        )


def test_strategy_is_server_owned(
) -> None:

    assert (
        cross_validation_strategy(
            problem_type=
                "regression"
        )
        ==
        "k_fold"
    )


    assert (
        cross_validation_strategy(
            problem_type=
                "classification"
        )
        ==
        "stratified_k_fold"
    )


    try:
        cross_validation_strategy(
            problem_type=
                "unsupported"
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            (
                "Unsupported problem type "
                "must fail closed."
            )
        )


def test_training_contract_fingerprint_surface_is_unchanged(
) -> None:

    contract = (
        training_contract()
    )


    before_json = (
        canonical_ml_training_contract_json(
            contract
        )
    )


    before_sha = (
        ml_training_contract_sha256(
            contract
        )
    )


    _ = (
        MLCrossValidationContract(
            folds=7,
            shuffle=True,
            random_seed=123,
        )
    )


    after_json = (
        canonical_ml_training_contract_json(
            contract
        )
    )


    after_sha = (
        ml_training_contract_sha256(
            contract
        )
    )


    assert (
        before_json
        ==
        after_json
    )


    assert (
        before_sha
        ==
        after_sha
    )


    assert (
        "cross_validation"
        not in
        before_json
    )


def test_fold_metrics_must_be_finite(
) -> None:

    expect_validation_error(
        lambda:
            MLCrossValidationFoldResult(
                fold_index=1,
                train_rows=80,
                validation_rows=20,
                metrics={
                    "rmse":
                        float(
                            "nan"
                        )
                },
            )
    )


def test_evaluation_result_is_structurally_strict(
) -> None:

    fold_results = [
        MLCrossValidationFoldResult(
            fold_index=index,
            train_rows=80,
            validation_rows=20,
            metrics=
                richer_metrics(
                    float(
                        index
                    )
                    *
                    0.1
                ),
        )

        for index
        in range(
            1,
            6,
        )
    ]


    summaries = {
        metric_name:
            MLCrossValidationMetricSummary(
                mean=1.0,
                std=0.1,
            )

        for metric_name
        in richer_metrics()
    }


    result = (
        MLCrossValidationEvaluationResult(
            workflow_id=
                "prep:cv-contract",

            dataset_id=
                "dataset:validated",

            problem_type=
                "regression",

            estimator_key=
                "linear_regression",

            preparation_session_revision=
                4,

            training_contract_sha256=
                (
                    "a"
                    *
                    64
                ),

            strategy=
                "k_fold",

            folds=
                5,

            shuffle=
                True,

            random_seed=
                42,

            fold_results=
                fold_results,

            metric_summary=
                summaries,
        )
    )


    assert (
        result.strategy
        ==
        "k_fold"
    )


    assert (
        len(
            result.fold_results
        )
        ==
        5
    )


    assert (
        set(
            result.metric_summary
        )
        ==
        set(
            richer_metrics()
        )
    )


def test_strategy_mismatch_fails_closed(
) -> None:

    fold_results = [
        MLCrossValidationFoldResult(
            fold_index=index,
            train_rows=80,
            validation_rows=20,
            metrics={
                "rmse":
                    1.0
            },
        )

        for index
        in range(
            1,
            3,
        )
    ]


    expect_validation_error(
        lambda:
            MLCrossValidationEvaluationResult(
                workflow_id=
                    "prep:cv-contract",

                dataset_id=
                    "dataset:validated",

                problem_type=
                    "regression",

                estimator_key=
                    "linear_regression",

                preparation_session_revision=
                    1,

                training_contract_sha256=
                    (
                        "b"
                        *
                        64
                    ),

                strategy=
                    "stratified_k_fold",

                folds=
                    2,

                shuffle=
                    True,

                random_seed=
                    42,

                fold_results=
                    fold_results,

                metric_summary={
                    "rmse":
                        MLCrossValidationMetricSummary(
                            mean=1.0,
                            std=0.0,
                        )
                },
            )
    )


def test_fold_metric_surfaces_must_match(
) -> None:

    expect_validation_error(
        lambda:
            MLCrossValidationEvaluationResult(
                workflow_id=
                    "prep:cv-contract",

                dataset_id=
                    "dataset:validated",

                problem_type=
                    "regression",

                estimator_key=
                    "linear_regression",

                preparation_session_revision=
                    1,

                training_contract_sha256=
                    (
                        "c"
                        *
                        64
                    ),

                strategy=
                    "k_fold",

                folds=
                    2,

                shuffle=
                    True,

                random_seed=
                    42,

                fold_results=[
                    MLCrossValidationFoldResult(
                        fold_index=1,
                        train_rows=8,
                        validation_rows=2,
                        metrics={
                            "mae":
                                1.0,
                            "rmse":
                                2.0,
                        },
                    ),
                    MLCrossValidationFoldResult(
                        fold_index=2,
                        train_rows=8,
                        validation_rows=2,
                        metrics={
                            "rmse":
                                2.1,
                        },
                    ),
                ],

                metric_summary={
                    "mae":
                        MLCrossValidationMetricSummary(
                            mean=1.0,
                            std=0.0,
                        ),

                    "rmse":
                        MLCrossValidationMetricSummary(
                            mean=2.0,
                            std=0.1,
                        ),
                },
            )
    )


def test_rule_version(
) -> None:

    assert (
        ML_CROSS_VALIDATION_RULE_VERSION
        ==
        "ml_cross_validation_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML CROSS-VALIDATION CONTRACT v0.1 ==="
    )


    tests = [
        (
            "Contract defaults",
            test_cross_validation_contract_defaults,
        ),
        (
            "Fold bounds",
            test_cross_validation_fold_bounds,
        ),
        (
            "Strict frozen contract",
            test_cross_validation_contract_is_strict_and_frozen,
        ),
        (
            "Server-owned strategy",
            test_strategy_is_server_owned,
        ),
        (
            "Training Contract fingerprint isolation",
            test_training_contract_fingerprint_surface_is_unchanged,
        ),
        (
            "Finite fold metrics",
            test_fold_metrics_must_be_finite,
        ),
        (
            "Evaluation result structure",
            test_evaluation_result_is_structurally_strict,
        ),
        (
            "Strategy mismatch fail-closed",
            test_strategy_mismatch_fails_closed,
        ),
        (
            "Fold metric surface consistency",
            test_fold_metric_surfaces_must_match,
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
        "PASS - ML Cross-Validation Contract v0.1"
    )


if __name__ == "__main__":
    main()
