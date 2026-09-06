from __future__ import annotations


import sys


from pathlib import (
    Path,
)


from app.ml.baseline import (
    build_ml_baseline_evaluation,
    compare_model_to_baseline,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.experiment_provenance import (
    build_ml_experiment_provenance,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.ml.model_candidate_result import (
    MLModelCandidateResult,
)


from app.ml.model_comparison_contracts import (
    MLModelComparisonContract,
)


from app.ml.model_comparison_core import (
    ML_MODEL_COMPARISON_CORE_RULE_VERSION,
    MLModelComparisonCoreCandidateError,
    MLModelComparisonCoreError,
    aggregate_ml_model_candidates,
    ml_model_comparison_ranking_key,
)


# ============================================================
# CONTRACT
# ============================================================


def build_training_contract(
    *,
    estimator_key: str,
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:hybrid-comparison",

            dataset_id=
                "dataset:validated",

            problem_type=
                "regression",

            target_column=
                "target",

            feature_columns=[
                "feature_a",
                "feature_b",
            ],

            estimator_key=
                estimator_key,
        )
    )


def build_comparison_contract(
) -> MLModelComparisonContract:

    return (
        MLModelComparisonContract(
            candidates=[
                build_training_contract(
                    estimator_key=
                        "linear_regression"
                ),

                build_training_contract(
                    estimator_key=
                        "tabular_mlp_regressor"
                ),
            ]
        )
    )


# ============================================================
# CANDIDATE FIXTURE
# ============================================================


def build_candidate(
    *,
    estimator_key: str,
    model_id: str,
    rmse: float,
    mae: float,
    r2: float,
    baseline_rmse: float = 3.0,
    baseline_mae: float = 2.5,
    baseline_r2: float = 0.0,
) -> MLModelCandidateResult:

    contract = (
        build_training_contract(
            estimator_key=
                estimator_key
        )
    )


    metrics = {
        "mae":
            mae,

        "rmse":
            rmse,

        "r2":
            r2,

        "median_absolute_error":
            mae,

        "explained_variance":
            r2,
    }


    baseline = (
        build_ml_baseline_evaluation(
            problem_type=
                "regression",

            strategy=
                "mean_train_target",

            metrics={
                "mae":
                    baseline_mae,

                "rmse":
                    baseline_rmse,

                "r2":
                    baseline_r2,
            },

            train_rows=
                80,

            test_rows=
                20,
        )
    )


    baseline_comparison = (
        compare_model_to_baseline(
            problem_type=
                "regression",

            model_metrics=
                metrics,

            baseline_metrics=
                baseline.metrics,
        )
    )


    provenance = (
        build_ml_experiment_provenance(
            training_contract=
                contract,

            model_id=
                model_id,

            metrics=
                metrics,

            train_rows=
                80,

            test_rows=
                20,

            preparation_session_revision=
                7,
        )
    )


    if (
        estimator_key
        ==
        "tabular_mlp_regressor"
    ):

        serialization_format = (
            "pytorch_bundle"
        )

        model_path = (
            "data/model.ptbundle"
        )

    else:

        serialization_format = (
            "joblib"
        )

        model_path = (
            "data/model.joblib"
        )


    artifact = (
        MLModelArtifactRecord(
            model_id=
                model_id,

            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,

            training_contract=
                contract,

            experiment_provenance=
                provenance,

            metrics=
                metrics,

            train_rows=
                80,

            test_rows=
                20,

            created_at_utc=
                "2026-09-06T15:30:00+00:00",

            serialization_format=
                serialization_format,

            model_path=
                model_path,

            model_file_bytes=
                100,

            model_sha256=
                (
                    "a"
                    *
                    64
                ),
        )
    )


    return (
        MLModelCandidateResult(
            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,

            preparation_session_revision=
                7,

            problem_type=
                contract.problem_type,

            estimator_key=
                contract.estimator_key,

            train_rows=
                80,

            test_rows=
                20,

            purged_rows=
                0,

            metrics=
                metrics,

            baseline=
                baseline,

            baseline_comparison=
                baseline_comparison,

            experiment_provenance=
                provenance,

            model_artifact=
                artifact,
        )
    )


# ============================================================
# MIXED-FRAMEWORK RANKING
# ============================================================


def test_classical_and_mlp_candidates_rank_together(
) -> None:

    contract = (
        build_comparison_contract()
    )


    classical = (
        build_candidate(
            estimator_key=
                "linear_regression",

            model_id=
                "model:linear",

            rmse=
                2.0,

            mae=
                1.5,

            r2=
                0.75,
        )
    )


    mlp = (
        build_candidate(
            estimator_key=
                "tabular_mlp_regressor",

            model_id=
                "model:mlp",

            rmse=
                1.5,

            mae=
                1.2,

            r2=
                0.80,
        )
    )


    aggregation = (
        aggregate_ml_model_candidates(
            comparison_contract=
                contract,

            candidates=[
                classical,
                mlp,
            ],

            expected_preparation_session_revision=
                7,
        )
    )


    assert [
        candidate.estimator_key

        for candidate
        in aggregation.ranked_candidates
    ] == [
        "tabular_mlp_regressor",
        "linear_regression",
    ]


    assert (
        aggregation.winner.estimator_key
        ==
        "tabular_mlp_regressor"
    )


    assert (
        aggregation.baseline
        ==
        classical.baseline
    )


# ============================================================
# INPUT ORDER INDEPENDENCE
# ============================================================


def test_ranking_is_independent_of_execution_order(
) -> None:

    contract = (
        build_comparison_contract()
    )


    classical = (
        build_candidate(
            estimator_key=
                "linear_regression",

            model_id=
                "model:linear-order",

            rmse=
                2.0,

            mae=
                1.5,

            r2=
                0.75,
        )
    )


    mlp = (
        build_candidate(
            estimator_key=
                "tabular_mlp_regressor",

            model_id=
                "model:mlp-order",

            rmse=
                1.5,

            mae=
                1.2,

            r2=
                0.80,
        )
    )


    first = (
        aggregate_ml_model_candidates(
            comparison_contract=
                contract,

            candidates=[
                classical,
                mlp,
            ],

            expected_preparation_session_revision=
                7,
        )
    )


    second = (
        aggregate_ml_model_candidates(
            comparison_contract=
                contract,

            candidates=[
                mlp,
                classical,
            ],

            expected_preparation_session_revision=
                7,
        )
    )


    assert [
        candidate.estimator_key

        for candidate
        in first.ranked_candidates
    ] == [
        candidate.estimator_key

        for candidate
        in second.ranked_candidates
    ]


# ============================================================
# EXACT HISTORICAL TIE-BREAK
# ============================================================


def test_regression_lexical_final_tie_break_is_preserved(
) -> None:

    linear_key = (
        ml_model_comparison_ranking_key(
            problem_type=
                "regression",

            estimator_key=
                "linear_regression",

            metrics={
                "rmse":
                    2.0,

                "mae":
                    1.5,

                "r2":
                    0.8,
            },
        )
    )


    mlp_key = (
        ml_model_comparison_ranking_key(
            problem_type=
                "regression",

            estimator_key=
                "tabular_mlp_regressor",

            metrics={
                "rmse":
                    2.0,

                "mae":
                    1.5,

                "r2":
                    0.8,
            },
        )
    )


    assert (
        linear_key
        <
        mlp_key
    )


# ============================================================
# CONTRACT SET GUARD
# ============================================================


def test_executed_candidate_set_must_match_contract(
) -> None:

    contract = (
        build_comparison_contract()
    )


    classical = (
        build_candidate(
            estimator_key=
                "linear_regression",

            model_id=
                "model:linear-set",

            rmse=
                2.0,

            mae=
                1.5,

            r2=
                0.75,
        )
    )


    ridge = (
        build_candidate(
            estimator_key=
                "ridge_regression",

            model_id=
                "model:ridge-set",

            rmse=
                1.9,

            mae=
                1.4,

            r2=
                0.76,
        )
    )


    try:

        aggregate_ml_model_candidates(
            comparison_contract=
                contract,

            candidates=[
                classical,
                ridge,
            ],

            expected_preparation_session_revision=
                7,
        )

    except MLModelComparisonCoreCandidateError:
        return


    raise AssertionError(
        (
            "Executed candidate set mismatch "
            "must fail closed."
        )
    )


# ============================================================
# SHARED BASELINE GUARD
# ============================================================


def test_shared_baseline_must_be_identical(
) -> None:

    contract = (
        build_comparison_contract()
    )


    classical = (
        build_candidate(
            estimator_key=
                "linear_regression",

            model_id=
                "model:linear-baseline",

            rmse=
                2.0,

            mae=
                1.5,

            r2=
                0.75,
        )
    )


    mlp = (
        build_candidate(
            estimator_key=
                "tabular_mlp_regressor",

            model_id=
                "model:mlp-baseline",

            rmse=
                1.5,

            mae=
                1.2,

            r2=
                0.80,

            baseline_rmse=
                4.0,

            baseline_mae=
                3.5,
        )
    )


    try:

        aggregate_ml_model_candidates(
            comparison_contract=
                contract,

            candidates=[
                classical,
                mlp,
            ],

            expected_preparation_session_revision=
                7,
        )

    except MLModelComparisonCoreError as error:

        assert (
            "shared baseline"
            in
            str(
                error
            )
        )

        return


    raise AssertionError(
        (
            "Mixed-framework baseline divergence "
            "must fail closed."
        )
    )


# ============================================================
# PREPARATION REVISION GUARD
# ============================================================


def test_preparation_revision_is_pinned(
) -> None:

    contract = (
        build_comparison_contract()
    )


    classical = (
        build_candidate(
            estimator_key=
                "linear_regression",

            model_id=
                "model:linear-revision",

            rmse=
                2.0,

            mae=
                1.5,

            r2=
                0.75,
        )
    )


    mlp = (
        build_candidate(
            estimator_key=
                "tabular_mlp_regressor",

            model_id=
                "model:mlp-revision",

            rmse=
                1.5,

            mae=
                1.2,

            r2=
                0.80,
        )
    )


    try:

        aggregate_ml_model_candidates(
            comparison_contract=
                contract,

            candidates=[
                classical,
                mlp,
            ],

            expected_preparation_session_revision=
                8,
        )

    except MLModelComparisonCoreCandidateError:
        return


    raise AssertionError(
        (
            "Comparison Preparation revision "
            "mismatch must fail closed."
        )
    )


# ============================================================
# FRAMEWORK ISOLATION
# ============================================================


def test_core_is_framework_neutral(
) -> None:

    path = (
        Path(__file__)
        .parents[
            2
        ]
        /
        "app"
        /
        "ml"
        /
        "model_comparison_core.py"
    )


    source = path.read_text(
        encoding="utf-8"
    )


    forbidden = (
        "import torch",
        "from torch",
        "app.deep_learning",
        "ClassicalMLExecutionResult",
        "TabularMLPExecutionResult",
        "execute_classical_ml",
        "execute_tabular_mlp",
    )


    for token in forbidden:

        assert (
            token
            not in
            source
        )


    assert (
        "torch"
        not in
        sys.modules
    )


# ============================================================
# RULE VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_MODEL_COMPARISON_CORE_RULE_VERSION
        ==
        "ml_model_comparison_core_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS FRAMEWORK-NEUTRAL MODEL COMPARISON CORE v0.1 ==="
    )

    print()


    test_classical_and_mlp_candidates_rank_together()

    print(
        "Classical + MLP candidate aggregation: PASS"
    )


    test_ranking_is_independent_of_execution_order()

    print(
        "Execution-order-independent ranking: PASS"
    )


    test_regression_lexical_final_tie_break_is_preserved()

    print(
        "Historical regression ranking policy: PASS"
    )


    test_executed_candidate_set_must_match_contract()

    print(
        "Comparison candidate-set guard: PASS"
    )


    test_shared_baseline_must_be_identical()

    print(
        "Shared baseline consistency guard: PASS"
    )


    test_preparation_revision_is_pinned()

    print(
        "Preparation revision authority: PASS"
    )


    test_core_is_framework_neutral()

    print(
        "Framework-neutral / torch-free core: PASS"
    )


    test_rule_version()

    print(
        "Comparison core rule version: PASS"
    )


    print()

    print(
        "PASS - DataLens Framework-Neutral Model Comparison Core v0.1"
    )


if __name__ == "__main__":
    main()
