from __future__ import annotations


from contextlib import (
    contextmanager,
)


from types import (
    SimpleNamespace,
)


import math


from app.deep_learning.model_comparison_executor import (
    DL_HYBRID_MODEL_COMPARISON_EXECUTOR_RULE_VERSION,
    DLHybridModelComparisonCandidateError,
    execute_hybrid_ml_model_comparison,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.model_comparison_contracts import (
    MLModelComparisonContract,
)


from app.ml.model_comparison_core import (
    ml_model_comparison_ranking_key,
)


from app.ml.model_comparison_executor import (
    MLModelComparisonExecutionResult,
)


import app.deep_learning.model_lab_executor as dl_executor_module


import app.ml.classical_executor as classical_executor_module


import app.ml.model_comparison_executor as comparison_executor_module


from tests.deep_learning.test_tabular_mlp_model_lab_executor_v0_1 import (
    build_dataframe,
    build_contract as build_mlp_contract,
    patched_handoff,
)


# ============================================================
# READINESS FIXTURE
# ============================================================


@contextmanager
def patched_hybrid_readiness(
    *,
    workflow_id: str,
    dataset_id: str,
    revision: int,
):

    original = (
        comparison_executor_module
        .require_analysis_readiness
    )


    def fake_require_analysis_readiness(
        *,
        workflow_id: str,
    ):

        return (
            SimpleNamespace(
                workflow_id=
                    workflow_id,

                session_revision=
                    revision,

                requested_analysis_dataset_ids=[
                    dataset_id,
                ],
            )
        )


    comparison_executor_module.require_analysis_readiness = (
        fake_require_analysis_readiness
    )


    try:
        yield

    finally:

        comparison_executor_module.require_analysis_readiness = (
            original
        )


# ============================================================
# SHARED CONTRACT
# ============================================================


def build_hybrid_contract(
) -> MLModelComparisonContract:

    mlp = (
        build_mlp_contract()
    )


    classical = (
        MLTrainingContract(
            workflow_id=
                mlp.workflow_id,

            dataset_id=
                mlp.dataset_id,

            problem_type=
                mlp.problem_type,

            target_column=
                mlp.target_column,

            feature_columns=
                list(
                    mlp.feature_columns
                ),

            categorical_feature_columns=
                list(
                    mlp
                    .categorical_feature_columns
                ),

            estimator_key=
                "linear_regression",

            preprocessing=
                mlp.preprocessing,

            split=
                mlp.split,
        )
    )


    return (
        MLModelComparisonContract(
            candidates=[
                classical,
                mlp,
            ]
        )
    )


# ============================================================
# REAL HYBRID EXECUTION
# ============================================================


def test_real_classical_and_mlp_comparison(
) -> None:

    dataframe = (
        build_dataframe()
    )


    comparison_contract = (
        build_hybrid_contract()
    )


    mlp_contract = next(
        candidate

        for candidate
        in comparison_contract.candidates

        if (
            candidate.estimator_key
            ==
            "tabular_mlp_regressor"
        )
    )


    original_classical_handoff = (
        classical_executor_module
        .load_validated_analysis_input
    )


    with patched_handoff(
        dataframe=
            dataframe,

        contract=
            mlp_contract,

        revision=
            23,
    ):

        # Both framework executors must read the exact same
        # server-owned Preparation handoff.
        classical_executor_module.load_validated_analysis_input = (
            dl_executor_module
            .load_validated_analysis_input
        )


        try:

            with patched_hybrid_readiness(
                workflow_id=
                    comparison_contract.workflow_id,

                dataset_id=
                    comparison_contract.dataset_id,

                revision=
                    23,
            ):

                result = (
                    execute_hybrid_ml_model_comparison(
                        comparison_contract=
                            comparison_contract,

                        execution_device=
                            "cpu",
                    )
                )

        finally:

            classical_executor_module.load_validated_analysis_input = (
                original_classical_handoff
            )


        # ----------------------------------------------------
        # COMMON RESULT
        # ----------------------------------------------------


        assert isinstance(
            result,
            MLModelComparisonExecutionResult,
        )


        assert (
            result.preparation_session_revision
            ==
            23
        )


        assert (
            result.problem_type
            ==
            "regression"
        )


        assert (
            result.primary_metric
            ==
            "rmse"
        )


        assert (
            result.ranking_policy
            ==
            "regression_rmse_v0.1"
        )


        # ----------------------------------------------------
        # BOTH REAL FRAMEWORKS EXECUTED
        # ----------------------------------------------------


        assert {
            candidate.estimator_key

            for candidate
            in result.candidates
        } == {
            "linear_regression",
            "tabular_mlp_regressor",
        }


        assert (
            len(
                result.candidates
            )
            ==
            2
        )


        artifact_formats = {
            candidate.estimator_key:
                candidate
                .model_artifact
                .serialization_format

            for candidate
            in result.candidates
        }


        assert (
            artifact_formats[
                "linear_regression"
            ]
            ==
            "joblib"
        )


        assert (
            artifact_formats[
                "tabular_mlp_regressor"
            ]
            ==
            "pytorch_bundle"
        )


        # ----------------------------------------------------
        # SAME HOLDOUT / SAME BASELINE
        # ----------------------------------------------------


        assert (
            result.candidates[
                0
            ].train_rows
            ==
            result.candidates[
                1
            ].train_rows
        )


        assert (
            result.candidates[
                0
            ].test_rows
            ==
            result.candidates[
                1
            ].test_rows
        )


        assert (
            result.baseline.train_rows
            ==
            result.candidates[
                0
            ].train_rows
        )


        assert (
            result.baseline.test_rows
            ==
            result.candidates[
                0
            ].test_rows
        )


        for candidate in (
            result.candidates
        ):

            assert (
                candidate
                .baseline_comparison
                .baseline_primary_metric_value
                ==
                result.baseline.metrics[
                    "rmse"
                ]
            )


            assert math.isclose(
                candidate.primary_metric_value,
                candidate.metrics[
                    "rmse"
                ],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )


        # ----------------------------------------------------
        # SHARED DETERMINISTIC RANKING
        # ----------------------------------------------------


        expected = sorted(
            result.candidates,

            key=lambda candidate: (
                ml_model_comparison_ranking_key(
                    problem_type=
                        "regression",

                    estimator_key=
                        candidate.estimator_key,

                    metrics=
                        candidate.metrics,
                )
            ),
        )


        assert [
            candidate.estimator_key

            for candidate
            in result.candidates
        ] == [
            candidate.estimator_key

            for candidate
            in expected
        ]


        assert (
            result.selected_estimator_key
            ==
            result.candidates[
                0
            ].estimator_key
        )


        assert (
            result.selected_model_id
            ==
            result.candidates[
                0
            ].model_artifact.model_id
        )


        assert (
            result.selected_experiment_id
            ==
            result.candidates[
                0
            ]
            .experiment_provenance
            .experiment_id
        )


# ============================================================
# HYBRID-ONLY AUTHORITY
# ============================================================


def test_hybrid_executor_rejects_all_classical_contract(
) -> None:

    mlp = (
        build_mlp_contract()
    )


    linear = (
        MLTrainingContract(
            workflow_id=
                mlp.workflow_id,

            dataset_id=
                mlp.dataset_id,

            problem_type=
                "regression",

            target_column=
                mlp.target_column,

            feature_columns=
                list(
                    mlp.feature_columns
                ),

            categorical_feature_columns=
                list(
                    mlp
                    .categorical_feature_columns
                ),

            estimator_key=
                "linear_regression",

            preprocessing=
                mlp.preprocessing,

            split=
                mlp.split,
        )
    )


    ridge = (
        MLTrainingContract(
            workflow_id=
                mlp.workflow_id,

            dataset_id=
                mlp.dataset_id,

            problem_type=
                "regression",

            target_column=
                mlp.target_column,

            feature_columns=
                list(
                    mlp.feature_columns
                ),

            categorical_feature_columns=
                list(
                    mlp
                    .categorical_feature_columns
                ),

            estimator_key=
                "ridge_regression",

            preprocessing=
                mlp.preprocessing,

            split=
                mlp.split,
        )
    )


    contract = (
        MLModelComparisonContract(
            candidates=[
                linear,
                ridge,
            ]
        )
    )


    try:

        execute_hybrid_ml_model_comparison(
            comparison_contract=
                contract,

            execution_device=
                "cpu",
        )

    except DLHybridModelComparisonCandidateError:
        return


    raise AssertionError(
        (
            "Hybrid executor must not replace "
            "the historical all-Classical path."
        )
    )


# ============================================================
# RULE VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        DL_HYBRID_MODEL_COMPARISON_EXECUTOR_RULE_VERSION
        ==
        "dl_hybrid_model_comparison_executor_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS HYBRID CLASSICAL ML / MLP MODEL COMPARISON v0.1 ==="
    )

    print()


    test_real_classical_and_mlp_comparison()

    print(
        "Real scikit-learn + PyTorch candidate execution: PASS"
    )


    print(
        "Same Preparation snapshot / holdout authority: PASS"
    )


    print(
        "Joblib + PyTorch Model Artifact persistence: PASS"
    )


    print(
        "Shared baseline across frameworks: PASS"
    )


    print(
        "Framework-neutral deterministic ranking: PASS"
    )


    test_hybrid_executor_rejects_all_classical_contract()

    print(
        "Historical Classical execution boundary preserved: PASS"
    )


    test_rule_version()

    print(
        "Hybrid executor rule version: PASS"
    )


    print()

    print(
        "PASS - DataLens Hybrid Classical ML / MLP Model Comparison v0.1"
    )


if __name__ == "__main__":
    main()
