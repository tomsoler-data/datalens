from __future__ import annotations


from pydantic import (
    ValidationError,
)


from app.deep_learning.model_lab_executor import (
    TabularMLPExecutionError,
    execute_tabular_mlp,
)


from app.ml.classical_executor import (
    ClassicalMLExecutorError,
    execute_classical_ml,
)


from app.ml.estimator_contracts import (
    estimator_problem_type,
)


from app.ml.model_candidate_result import (
    MLModelCandidateResult,
    project_ml_model_candidate_result,
)


from app.ml.model_comparison_contracts import (
    MLModelComparisonContract,
)


from app.ml.model_comparison_core import (
    MLModelComparisonCoreError,
    aggregate_ml_model_candidates,
)


from app.ml.model_comparison_executor import (
    ML_MODEL_COMPARISON_EXECUTOR_RULE_VERSION,
    MLModelComparisonCandidateResult,
    MLModelComparisonExecutionResult,
    MLModelComparisonSnapshotError,
    _assert_comparison_snapshot_unchanged,
    _read_comparison_snapshot_authority,
)


# ============================================================
# VERSION
# ============================================================


DL_HYBRID_MODEL_COMPARISON_EXECUTOR_RULE_VERSION = (
    "dl_hybrid_model_comparison_executor_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class DLHybridModelComparisonExecutorError(
    RuntimeError
):
    pass


class DLHybridModelComparisonCandidateError(
    DLHybridModelComparisonExecutorError
):
    pass


class DLHybridModelComparisonSnapshotError(
    DLHybridModelComparisonExecutorError
):
    pass


# ============================================================
# FRAMEWORK AUTHORITY
# ============================================================


def _candidate_framework(
    estimator_key: str,
) -> str:

    if (
        estimator_key
        ==
        "tabular_mlp_regressor"
    ):
        return "deep_learning"


    if (
        estimator_problem_type(
            estimator_key
        )
        is not None
    ):
        return "classical"


    raise (
        DLHybridModelComparisonCandidateError(
            (
                "Hybrid Model Comparison received "
                "an estimator with no v0.1 execution "
                "framework authority. "
                f"estimator_key={estimator_key}"
            )
        )
    )


def _validate_hybrid_candidate_set(
    contract: MLModelComparisonContract,
) -> None:
    """
    Hybrid v0.1 deliberately requires both frameworks.

    All-Classical execution remains owned by the historical
    runtime comparison executor.
    """

    frameworks = {
        _candidate_framework(
            candidate.estimator_key
        )

        for candidate
        in contract.candidates
    }


    if (
        frameworks
        !=
        {
            "classical",
            "deep_learning",
        }
    ):
        raise (
            DLHybridModelComparisonCandidateError(
                (
                    "Hybrid Model Comparison v0.1 "
                    "requires at least one Classical "
                    "candidate and one Deep Learning "
                    "candidate."
                )
            )
        )


# ============================================================
# ONE CANDIDATE EXECUTION
# ============================================================


def _execute_hybrid_candidate(
    *,
    candidate_contract,
    preparation_session_revision: int,
    execution_device: str,
):
    framework = (
        _candidate_framework(
            candidate_contract.estimator_key
        )
    )


    if (
        framework
        ==
        "deep_learning"
    ):

        try:

            return (
                execute_tabular_mlp(
                    training_contract=
                        candidate_contract,

                    expected_preparation_session_revision=
                        preparation_session_revision,

                    execution_device=
                        execution_device,
                )
            )

        except TabularMLPExecutionError as error:

            raise (
                DLHybridModelComparisonCandidateError(
                    (
                        "Deep Learning candidate "
                        "execution failed. "
                        "estimator_key="
                        f"{candidate_contract.estimator_key}"
                    )
                )
            ) from error


    try:

        return (
            execute_classical_ml(
                training_contract=
                    candidate_contract,

                expected_preparation_session_revision=
                    preparation_session_revision,
            )
        )

    except ClassicalMLExecutorError as error:

        raise (
            DLHybridModelComparisonCandidateError(
                (
                    "Classical candidate execution "
                    "failed. estimator_key="
                    f"{candidate_contract.estimator_key}"
                )
            )
        ) from error


# ============================================================
# PUBLIC EXECUTION
# ============================================================


def execute_hybrid_ml_model_comparison(
    *,
    comparison_contract: MLModelComparisonContract,
    execution_device: str = "cpu",
) -> MLModelComparisonExecutionResult:
    """
    Execute one deterministic mixed Classical / Deep Learning
    Model Lab comparison.

    This module belongs to the Deep Learning execution
    environment and may therefore import both scikit-learn
    and PyTorch execution paths.

    The historical FastAPI/runtime Model Comparison executor
    remains PyTorch-free.

    All candidates still share the exact comparison contract:
    workflow, dataset, target, ordered features, preprocessing
    policy and split policy.
    """

    contract = (
        MLModelComparisonContract
        .model_validate(
            comparison_contract
        )
    )


    _validate_hybrid_candidate_set(
        contract
    )


    # ========================================================
    # PIN PREPARATION SNAPSHOT
    # ========================================================


    try:

        (
            preparation_session_revision,
            preparation_dataset_ids,
        ) = (
            _read_comparison_snapshot_authority(
                contract=
                    contract
            )
        )

    except MLModelComparisonSnapshotError as error:

        raise (
            DLHybridModelComparisonSnapshotError(
                (
                    "Hybrid Model Comparison could "
                    "not pin Preparation authority."
                )
            )
        ) from error


    shared_candidates: list[
        MLModelCandidateResult
    ] = []


    # ========================================================
    # EXECUTE EXACT FIXED CANDIDATES
    # ========================================================


    for candidate_contract in (
        contract.candidates
    ):

        try:

            _assert_comparison_snapshot_unchanged(
                contract=
                    contract,

                expected_session_revision=
                    preparation_session_revision,

                expected_dataset_ids=
                    preparation_dataset_ids,
            )

        except MLModelComparisonSnapshotError as error:

            raise (
                DLHybridModelComparisonSnapshotError(
                    (
                        "Preparation authority changed "
                        "before hybrid candidate "
                        "execution."
                    )
                )
            ) from error


        execution_result = (
            _execute_hybrid_candidate(
                candidate_contract=
                    candidate_contract,

                preparation_session_revision=
                    preparation_session_revision,

                execution_device=
                    execution_device,
            )
        )


        try:

            shared_candidate = (
                project_ml_model_candidate_result(
                    execution_result
                )
            )

        except (
            TypeError,
            ValueError,
            ValidationError,
        ) as error:

            raise (
                DLHybridModelComparisonCandidateError(
                    (
                        "Hybrid candidate could not "
                        "be projected to the shared "
                        "Model Lab candidate result."
                    )
                )
            ) from error


        if (
            shared_candidate
            .model_artifact
            .training_contract
            !=
            candidate_contract
        ):
            raise (
                DLHybridModelComparisonCandidateError(
                    (
                        "Hybrid candidate artifact "
                        "does not reference its exact "
                        "comparison Training Contract."
                    )
                )
            )


        shared_candidates.append(
            shared_candidate
        )


        try:

            _assert_comparison_snapshot_unchanged(
                contract=
                    contract,

                expected_session_revision=
                    preparation_session_revision,

                expected_dataset_ids=
                    preparation_dataset_ids,
            )

        except MLModelComparisonSnapshotError as error:

            raise (
                DLHybridModelComparisonSnapshotError(
                    (
                        "Preparation authority changed "
                        "after hybrid candidate "
                        "execution."
                    )
                )
            ) from error


    # ========================================================
    # SHARED FRAMEWORK-NEUTRAL COMPARISON CORE
    # ========================================================


    try:

        aggregation = (
            aggregate_ml_model_candidates(
                comparison_contract=
                    contract,

                candidates=
                    shared_candidates,

                expected_preparation_session_revision=
                    preparation_session_revision,
            )
        )

    except MLModelComparisonCoreError as error:

        raise (
            DLHybridModelComparisonExecutorError(
                (
                    "Framework-neutral comparison "
                    "core rejected hybrid candidate "
                    "aggregation."
                )
            )
        ) from error


    # ========================================================
    # COMMON PUBLIC RESULT
    # ========================================================


    ranked_candidates: list[
        MLModelComparisonCandidateResult
    ] = []


    for (
        rank,
        candidate,
    ) in enumerate(
        aggregation.ranked_candidates,
        start=1,
    ):

        primary_metric_value = float(
            candidate.metrics[
                contract.primary_metric
            ]
        )


        ranked_candidates.append(
            MLModelComparisonCandidateResult(
                rank=
                    rank,

                estimator_key=
                    candidate.estimator_key,

                primary_metric=
                    contract.primary_metric,

                primary_metric_value=
                    primary_metric_value,

                metrics=
                    candidate.metrics,

                train_rows=
                    candidate.train_rows,

                test_rows=
                    candidate.test_rows,

                baseline_comparison=
                    candidate.baseline_comparison,

                experiment_provenance=
                    candidate.experiment_provenance,

                model_artifact=
                    candidate.model_artifact,
            )
        )


    winner = (
        ranked_candidates[
            0
        ]
    )


    try:

        return (
            MLModelComparisonExecutionResult(
                workflow_id=
                    contract.workflow_id,

                dataset_id=
                    contract.dataset_id,

                preparation_session_revision=
                    preparation_session_revision,

                problem_type=
                    contract.problem_type,

                comparison_contract=
                    contract,

                primary_metric=
                    contract.primary_metric,

                ranking_policy=
                    contract.ranking_policy,

                baseline=
                    aggregation.baseline,

                candidates=
                    ranked_candidates,

                selected_estimator_key=
                    winner.estimator_key,

                selected_experiment_id=(
                    winner
                    .experiment_provenance
                    .experiment_id
                ),

                selected_model_id=(
                    winner
                    .model_artifact
                    .model_id
                ),

                rule_version=(
                    ML_MODEL_COMPARISON_EXECUTOR_RULE_VERSION
                ),
            )
        )

    except ValidationError as error:

        raise (
            DLHybridModelComparisonExecutorError(
                (
                    "Hybrid comparison completed but "
                    "the shared public result failed "
                    "consistency validation."
                )
            )
        ) from error
