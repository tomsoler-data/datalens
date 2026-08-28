from __future__ import annotations


import app.ml.model_comparison_executor as comparison_module


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


from app.ml.model_comparison_contracts import (
    MLModelComparisonContract,
)


from app.ml.model_comparison_executor import (
    MLModelComparisonCandidateError,
    MLModelComparisonExecutorError,
    execute_ml_model_comparison,
)


from tests.ml.test_classical_ml_executor_v0_1 import (
    classification_dataframe,
    isolated_environment,
    patched_handoff,
    regression_dataframe,
    seed_preparation_authority,
)


from tests.ml.test_ml_model_comparison_executor_v0_1 import (
    DATASET_ID,
    WORKFLOW_ID,
    classification_candidate,
    patched_comparison_readiness,
    regression_candidate,
)


# ============================================================
# REGRESSION PROVENANCE
# ============================================================


def test_regression_comparison_exposes_candidate_experiments(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,
        )


        contract = (
            MLModelComparisonContract(
                candidates=[
                    regression_candidate(
                        estimator_key=
                            "linear_regression"
                    ),

                    regression_candidate(
                        estimator_key=
                            "ridge_regression"
                    ),

                    regression_candidate(
                        estimator_key=
                            "random_forest_regressor"
                    ),
                ]
            )
        )


        with patched_comparison_readiness():

            with patched_handoff(
                dataframe=
                    regression_dataframe(),

                workflow_id=
                    WORKFLOW_ID,

                dataset_id=
                    DATASET_ID,
            ):

                result = (
                    execute_ml_model_comparison(
                        comparison_contract=
                            contract
                    )
                )


        assert (
            result.preparation_session_revision
            ==
            0
        )


        experiment_ids = {
            candidate
            .experiment_provenance
            .experiment_id

            for candidate
            in result.candidates
        }


        assert (
            len(
                experiment_ids
            )
            ==
            3
        )


        for candidate in (
            result.candidates
        ):

            provenance = (
                candidate
                .experiment_provenance
            )


            assert (
                provenance
                ==
                candidate
                .model_artifact
                .experiment_provenance
            )


            assert (
                provenance.workflow_id
                ==
                WORKFLOW_ID
            )


            assert (
                provenance.dataset_id
                ==
                DATASET_ID
            )


            assert (
                provenance.preparation_session_revision
                ==
                result.preparation_session_revision
            )


            assert (
                provenance.model_id
                ==
                candidate
                .model_artifact
                .model_id
            )


            assert (
                provenance.train_rows
                ==
                candidate.train_rows
            )


            assert (
                provenance.test_rows
                ==
                candidate.test_rows
            )


            assert (
                provenance.metrics
                ==
                candidate.metrics
            )


            assert (
                provenance.training_contract_sha256
                ==
                ml_training_contract_sha256(
                    candidate
                    .model_artifact
                    .training_contract
                )
            )


        winner = (
            result.candidates[
                0
            ]
        )


        assert (
            result.selected_experiment_id
            ==
            winner
            .experiment_provenance
            .experiment_id
        )


        assert (
            result.selected_model_id
            ==
            winner
            .model_artifact
            .model_id
        )


# ============================================================
# CLASSIFICATION PROVENANCE
# ============================================================


def test_classification_comparison_exposes_candidate_experiments(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,
        )


        contract = (
            MLModelComparisonContract(
                candidates=[
                    classification_candidate(
                        estimator_key=
                            "logistic_regression"
                    ),

                    classification_candidate(
                        estimator_key=
                            "random_forest_classifier"
                    ),
                ]
            )
        )


        with patched_comparison_readiness():

            with patched_handoff(
                dataframe=
                    classification_dataframe(),

                workflow_id=
                    WORKFLOW_ID,

                dataset_id=
                    DATASET_ID,
            ):

                result = (
                    execute_ml_model_comparison(
                        comparison_contract=
                            contract
                    )
                )


        assert (
            len(
                {
                    candidate
                    .experiment_provenance
                    .experiment_id

                    for candidate
                    in result.candidates
                }
            )
            ==
            2
        )


        for candidate in (
            result.candidates
        ):

            assert (
                candidate
                .experiment_provenance
                .preparation_session_revision
                ==
                result.preparation_session_revision
            )


            assert (
                candidate
                .experiment_provenance
                ==
                candidate
                .model_artifact
                .experiment_provenance
            )


# ============================================================
# REVISION TAMPERING
# ============================================================


def test_candidate_experiment_revision_mismatch_is_fail_closed(
) -> None:

    original_execute = (
        comparison_module
        .execute_classical_ml
    )


    execution_count = 0


    def tampered_execute(
        *,
        training_contract,
    ):

        nonlocal execution_count


        execution_count += 1


        result = (
            original_execute(
                training_contract=
                    training_contract
            )
        )


        if (
            execution_count
            !=
            2
        ):
            return result


        tampered_provenance = (
            result
            .experiment_provenance
            .model_copy(
                update={
                    "preparation_session_revision":
                        1,
                }
            )
        )


        tampered_artifact = (
            result
            .model_artifact
            .model_copy(
                update={
                    "experiment_provenance":
                        tampered_provenance,
                }
            )
        )


        return (
            result.model_copy(
                update={
                    "experiment_provenance":
                        tampered_provenance,

                    "model_artifact":
                        tampered_artifact,
                }
            )
        )


    comparison_module.execute_classical_ml = (
        tampered_execute
    )


    try:

        with isolated_environment():

            seed_preparation_authority(
                workflow_id=
                    WORKFLOW_ID,

                dataset_id=
                    DATASET_ID,
            )


            contract = (
                MLModelComparisonContract(
                    candidates=[
                        regression_candidate(
                            estimator_key=
                                "linear_regression"
                        ),

                        regression_candidate(
                            estimator_key=
                                "ridge_regression"
                        ),
                    ]
                )
            )


            with patched_comparison_readiness():

                with patched_handoff(
                    dataframe=
                        regression_dataframe(),

                    workflow_id=
                        WORKFLOW_ID,

                    dataset_id=
                        DATASET_ID,
                ):

                    try:
                        execute_ml_model_comparison(
                            comparison_contract=
                                contract
                        )

                    except MLModelComparisonCandidateError:
                        return


        raise AssertionError(
            (
                "Candidate Experiment Provenance "
                "revision mismatch must fail closed."
            )
        )


    finally:
        comparison_module.execute_classical_ml = (
            original_execute
        )


# ============================================================
# DUPLICATE EXPERIMENT ID
# ============================================================


def test_duplicate_candidate_experiment_ids_are_fail_closed(
) -> None:

    original_execute = (
        comparison_module
        .execute_classical_ml
    )


    first_experiment_id = None


    def duplicate_execute(
        *,
        training_contract,
    ):

        nonlocal first_experiment_id


        result = (
            original_execute(
                training_contract=
                    training_contract
            )
        )


        if first_experiment_id is None:

            first_experiment_id = (
                result
                .experiment_provenance
                .experiment_id
            )


            return result


        duplicate_provenance = (
            result
            .experiment_provenance
            .model_copy(
                update={
                    "experiment_id":
                        first_experiment_id,
                }
            )
        )


        duplicate_artifact = (
            result
            .model_artifact
            .model_copy(
                update={
                    "experiment_provenance":
                        duplicate_provenance,
                }
            )
        )


        return (
            result.model_copy(
                update={
                    "experiment_provenance":
                        duplicate_provenance,

                    "model_artifact":
                        duplicate_artifact,
                }
            )
        )


    comparison_module.execute_classical_ml = (
        duplicate_execute
    )


    try:

        with isolated_environment():

            seed_preparation_authority(
                workflow_id=
                    WORKFLOW_ID,

                dataset_id=
                    DATASET_ID,
            )


            contract = (
                MLModelComparisonContract(
                    candidates=[
                        regression_candidate(
                            estimator_key=
                                "linear_regression"
                        ),

                        regression_candidate(
                            estimator_key=
                                "ridge_regression"
                        ),
                    ]
                )
            )


            with patched_comparison_readiness():

                with patched_handoff(
                    dataframe=
                        regression_dataframe(),

                    workflow_id=
                        WORKFLOW_ID,

                    dataset_id=
                        DATASET_ID,
                ):

                    try:
                        execute_ml_model_comparison(
                            comparison_contract=
                                contract
                        )

                    except MLModelComparisonExecutorError:
                        return


        raise AssertionError(
            (
                "Duplicate experiment_id values "
                "must fail Model Comparison closed."
            )
        )


    finally:
        comparison_module.execute_classical_ml = (
            original_execute
        )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML MODEL COMPARISON "
            "EXPERIMENT PROVENANCE v0.1 ==="
        )
    )

    print()


    test_regression_comparison_exposes_candidate_experiments()

    print(
        "Regression candidate Experiment Provenance: PASS"
    )


    test_classification_comparison_exposes_candidate_experiments()

    print(
        "Classification candidate Experiment Provenance: PASS"
    )


    test_candidate_experiment_revision_mismatch_is_fail_closed()

    print(
        "Candidate provenance revision mismatch is fail-closed: PASS"
    )


    test_duplicate_candidate_experiment_ids_are_fail_closed()

    print(
        "Duplicate experiment identities are fail-closed: PASS"
    )


    print()

    print(
        (
            "ML Model Comparison Experiment "
            "Provenance v0.1: PASS"
        )
    )


if __name__ == "__main__":
    main()
