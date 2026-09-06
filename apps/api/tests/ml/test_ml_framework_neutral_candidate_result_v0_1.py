from __future__ import annotations


from pathlib import (
    Path,
)


from pydantic import (
    ValidationError,
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
    ML_MODEL_CANDIDATE_RESULT_RULE_VERSION,
    MLModelCandidateResult,
    project_ml_model_candidate_result,
)


# ============================================================
# FIXTURES
# ============================================================


def build_candidate_fixture(
    *,
    estimator_key: str,
    serialization_format: str,
    model_path: str,
):
    contract = (
        MLTrainingContract(
            workflow_id=
                "prep:candidate",

            dataset_id=
                "dataset:validated",

            problem_type=
                "regression",

            target_column=
                "target",

            feature_columns=[
                "feature",
            ],

            estimator_key=
                estimator_key,
        )
    )


    metrics = {
        "mae":
            1.0,

        "rmse":
            1.2,

        "r2":
            0.8,

        "median_absolute_error":
            0.9,

        "explained_variance":
            0.81,
    }


    baseline = (
        build_ml_baseline_evaluation(
            problem_type=
                "regression",

            strategy=
                "mean_train_target",

            metrics={
                "mae":
                    2.0,

                "rmse":
                    2.4,

                "r2":
                    0.0,
            },

            train_rows=
                80,

            test_rows=
                20,
        )
    )


    comparison = (
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
                "model:candidate",

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


    artifact = (
        MLModelArtifactRecord(
            model_id=
                "model:candidate",

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
                "2026-09-06T15:10:00+00:00",

            serialization_format=
                serialization_format,

            model_path=
                model_path,

            model_file_bytes=
                100,

            model_sha256=
                "a"
                *
                64,
        )
    )


    return (
        contract,
        metrics,
        baseline,
        comparison,
        provenance,
        artifact,
    )


class FakeExecutionResult:

    pass


def build_execution_result(
    *,
    estimator_key: str,
    serialization_format: str,
    model_path: str,
    explicit_revision: bool,
):

    (
        contract,
        metrics,
        baseline,
        comparison,
        provenance,
        artifact,
    ) = (
        build_candidate_fixture(
            estimator_key=
                estimator_key,

            serialization_format=
                serialization_format,

            model_path=
                model_path,
        )
    )


    result = (
        FakeExecutionResult()
    )


    result.workflow_id = (
        contract.workflow_id
    )

    result.dataset_id = (
        contract.dataset_id
    )

    result.problem_type = (
        contract.problem_type
    )

    result.estimator_key = (
        contract.estimator_key
    )

    result.train_rows = 80

    result.test_rows = 20

    result.purged_rows = 0

    result.metrics = metrics

    result.baseline = baseline

    result.baseline_comparison = comparison

    result.experiment_provenance = provenance

    result.model_artifact = artifact


    if explicit_revision:

        result.preparation_session_revision = 7


    return result


# ============================================================
# CLASSICAL-SHAPED PROJECTION
# ============================================================


def test_projection_derives_revision_from_provenance(
) -> None:

    execution = (
        build_execution_result(
            estimator_key=
                "linear_regression",

            serialization_format=
                "joblib",

            model_path=
                "data/model.joblib",

            explicit_revision=
                False,
        )
    )


    candidate = (
        project_ml_model_candidate_result(
            execution
        )
    )


    assert (
        candidate.preparation_session_revision
        ==
        7
    )


    assert (
        candidate.estimator_key
        ==
        "linear_regression"
    )


    assert (
        candidate.model_artifact.serialization_format
        ==
        "joblib"
    )


# ============================================================
# DEEP-LEARNING-SHAPED PROJECTION
# ============================================================


def test_projection_accepts_explicit_revision(
) -> None:

    execution = (
        build_execution_result(
            estimator_key=
                "tabular_mlp_regressor",

            serialization_format=
                "pytorch_bundle",

            model_path=
                "data/model.ptbundle",

            explicit_revision=
                True,
        )
    )


    candidate = (
        project_ml_model_candidate_result(
            execution
        )
    )


    assert (
        candidate.preparation_session_revision
        ==
        7
    )


    assert (
        candidate.estimator_key
        ==
        "tabular_mlp_regressor"
    )


    assert (
        candidate.model_artifact.serialization_format
        ==
        "pytorch_bundle"
    )


# ============================================================
# CONSISTENCY GUARD
# ============================================================


def test_candidate_scope_mismatch_fails_closed(
) -> None:

    execution = (
        build_execution_result(
            estimator_key=
                "linear_regression",

            serialization_format=
                "joblib",

            model_path=
                "data/model.joblib",

            explicit_revision=
                False,
        )
    )


    payload = (
        project_ml_model_candidate_result(
            execution
        )
        .model_dump(
            mode="json"
        )
    )


    payload[
        "workflow_id"
    ] = (
        "prep:wrong"
    )


    try:

        MLModelCandidateResult.model_validate(
            payload
        )

    except ValidationError:
        return


    raise AssertionError(
        (
            "Candidate / artifact scope mismatch "
            "must fail closed."
        )
    )


# ============================================================
# PRIVACY
# ============================================================


def test_candidate_surface_is_privacy_minimal(
) -> None:

    execution = (
        build_execution_result(
            estimator_key=
                "tabular_mlp_regressor",

            serialization_format=
                "pytorch_bundle",

            model_path=
                "data/model.ptbundle",

            explicit_revision=
                True,
        )
    )


    payload = (
        project_ml_model_candidate_result(
            execution
        )
        .model_dump(
            mode="json"
        )
    )


    forbidden = {
        "predictions",
        "model",
        "preprocessor",
        "partition",
        "train_positions",
        "test_positions",
        "purged_positions",
        "epoch_losses",
        "test_loss",
        "model_bytes",
        "state_dict",
    }


    assert (
        forbidden
        .isdisjoint(
            payload
        )
    )


# ============================================================
# FRAMEWORK ISOLATION
# ============================================================


def test_shared_candidate_module_is_framework_neutral(
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
        "model_candidate_result.py"
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


# ============================================================
# RULE VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_MODEL_CANDIDATE_RESULT_RULE_VERSION
        ==
        "ml_model_candidate_result_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS FRAMEWORK-NEUTRAL MODEL CANDIDATE RESULT v0.1 ==="
    )

    print()


    test_projection_derives_revision_from_provenance()

    print(
        "Classical-shaped candidate projection: PASS"
    )


    test_projection_accepts_explicit_revision()

    print(
        "Deep-Learning-shaped candidate projection: PASS"
    )


    test_candidate_scope_mismatch_fails_closed()

    print(
        "Candidate lifecycle consistency guard: PASS"
    )


    test_candidate_surface_is_privacy_minimal()

    print(
        "Privacy-minimal candidate surface: PASS"
    )


    test_shared_candidate_module_is_framework_neutral()

    print(
        "Framework-neutral shared module: PASS"
    )


    test_rule_version()

    print(
        "Candidate result rule version: PASS"
    )


    print()

    print(
        "PASS - DataLens Framework-Neutral Model Candidate Result v0.1"
    )


if __name__ == "__main__":
    main()
