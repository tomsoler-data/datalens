from __future__ import annotations


from types import (
    SimpleNamespace,
)


import pandas as pd


import app.ml.classical_executor as classical_executor
import app.ml.tuned_model_promotion_executor as promotion_executor


from app.ml.classical_executor import (
    ClassicalMLInputError,
    execute_classical_ml,
)


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


from app.ml.hyperparameter_tuning import (
    MLHyperparameterSearchContract,
)


from app.ml.tuned_model_promotion import (
    MLTunedModelPromotionContract,
    build_promoted_training_contract,
)


from app.ml.tuned_model_promotion_executor import (
    ML_TUNED_MODEL_PROMOTION_EXECUTOR_RULE_VERSION,
    MLTunedModelPromotionExecutionError,
    execute_ml_tuned_model_promotion,
)


from tests.ml.test_ml_tuned_model_promotion_contract_v0_1 import (
    base_training_contract,
    valid_tuning_result,
)


# ============================================================
# HELPERS
# ============================================================


def promotion_contract(
) -> MLTunedModelPromotionContract:

    return (
        MLTunedModelPromotionContract(
            base_training_contract=(
                base_training_contract()
            ),

            search_contract=(
                MLHyperparameterSearchContract(
                    folds=
                        5,

                    shuffle=
                        True,

                    random_seed=
                        73,
                )
            ),
        )
    )


def final_metrics(
) -> dict[
    str,
    float,
]:

    return {
        "mae":
            0.80,

        "rmse":
            1.00,

        "r2":
            0.90,

        "median_absolute_error":
            0.70,

        "explained_variance":
            0.91,
    }


def fake_final_execution(
    *,
    promoted_contract,
    revision: int,
    train_rows: int = 80,
    test_rows: int = 20,
):

    promoted_sha = (
        ml_training_contract_sha256(
            promoted_contract
        )
    )


    provenance = (
        SimpleNamespace(
            experiment_id=(
                "experiment:"
                +
                (
                    "a"
                    *
                    32
                )
            ),

            model_id=(
                "model:"
                +
                (
                    "b"
                    *
                    32
                )
            ),

            preparation_session_revision=
                revision,

            training_contract_sha256=
                promoted_sha,
        )
    )


    artifact = (
        SimpleNamespace(
            model_id=
                provenance.model_id,

            training_contract=
                promoted_contract,

            experiment_provenance=
                provenance,
        )
    )


    return (
        SimpleNamespace(
            workflow_id=
                promoted_contract.workflow_id,

            dataset_id=
                promoted_contract.dataset_id,

            problem_type=
                promoted_contract.problem_type,

            estimator_key=
                promoted_contract.estimator_key,

            train_rows=
                train_rows,

            test_rows=
                test_rows,

            metrics=
                final_metrics(),

            experiment_provenance=
                provenance,

            model_artifact=
                artifact,
        )
    )


# ============================================================
# CLASSICAL ML REVISION PIN
# ============================================================


def test_classical_ml_expected_revision_blocks_before_fit(
) -> None:

    contract = (
        base_training_contract()
    )


    original_loader = (
        classical_executor
        ._load_authorized_dataframe
    )


    original_builder = (
        classical_executor
        ._build_estimator
    )


    build_called = False


    def fake_loader(
        *,
        contract,
    ):

        return (
            pd.DataFrame(
                {
                    "irrelevant":
                        [
                            1.0,
                            2.0,
                        ]
                }
            ),
            8,
        )


    def recording_builder(
        *,
        contract,
    ):

        nonlocal build_called

        build_called = True

        return (
            original_builder(
                contract=
                    contract
            )
        )


    classical_executor._load_authorized_dataframe = (
        fake_loader
    )


    classical_executor._build_estimator = (
        recording_builder
    )


    try:
        try:
            execute_classical_ml(
                training_contract=
                    contract,

                expected_preparation_session_revision=
                    7,
            )

        except ClassicalMLInputError:
            pass

        else:
            raise AssertionError(
                (
                    "Expected Preparation revision "
                    "mismatch to fail closed."
                )
            )

    finally:
        classical_executor._load_authorized_dataframe = (
            original_loader
        )

        classical_executor._build_estimator = (
            original_builder
        )


    assert (
        build_called
        is False
    )


# ============================================================
# PROMOTION ORCHESTRATION
# ============================================================


def test_promotion_replays_tuning_and_promotes_rank_1(
) -> None:

    contract = (
        promotion_contract()
    )


    base = (
        contract.base_training_contract
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


    original_tuning = (
        promotion_executor
        .execute_ml_hyperparameter_tuning
    )


    original_training = (
        promotion_executor
        .execute_classical_ml
    )


    tuning_calls = 0
    training_calls = 0
    captured_expected_revision = None
    captured_training_contract = None


    def fake_tuning(
        *,
        training_contract,
        search_contract,
    ):

        nonlocal tuning_calls

        tuning_calls += 1

        assert (
            training_contract
            ==
            base
        )

        assert (
            search_contract
            ==
            contract.search_contract
        )

        return tuning


    def fake_training(
        *,
        training_contract,
        expected_preparation_session_revision=None,
    ):

        nonlocal training_calls
        nonlocal captured_expected_revision
        nonlocal captured_training_contract

        training_calls += 1

        captured_expected_revision = (
            expected_preparation_session_revision
        )

        captured_training_contract = (
            training_contract
        )

        return (
            fake_final_execution(
                promoted_contract=
                    training_contract,

                revision=
                    tuning
                    .preparation_session_revision,
            )
        )


    promotion_executor.execute_ml_hyperparameter_tuning = (
        fake_tuning
    )


    promotion_executor.execute_classical_ml = (
        fake_training
    )


    try:
        result = (
            execute_ml_tuned_model_promotion(
                promotion_contract=
                    contract
            )
        )

    finally:
        promotion_executor.execute_ml_hyperparameter_tuning = (
            original_tuning
        )

        promotion_executor.execute_classical_ml = (
            original_training
        )


    assert (
        tuning_calls
        ==
        1
    )


    assert (
        training_calls
        ==
        1
    )


    assert (
        captured_expected_revision
        ==
        tuning.preparation_session_revision
        ==
        7
    )


    assert (
        captured_training_contract
        ==
        promoted
    )


    winner = (
        tuning.candidate_results[
            0
        ]
    )


    assert (
        result.selected_candidate_index
        ==
        winner.candidate_index
        ==
        2
    )


    assert (
        result.selected_hyperparameters
        ==
        winner.hyperparameters
    )


    assert (
        result.promoted_training_contract_sha256
        ==
        winner.training_contract_sha256
    )


    assert (
        result.model_id
        ==
        (
            "model:"
            +
            (
                "b"
                *
                32
            )
        )
    )


    assert (
        result.experiment_id
        ==
        (
            "experiment:"
            +
            (
                "a"
                *
                32
            )
        )
    )


# ============================================================
# REVISION PROPAGATION
# ============================================================


def test_tuning_revision_is_required_by_final_training(
) -> None:

    contract = (
        promotion_contract()
    )


    base = (
        contract.base_training_contract
    )


    tuning = (
        valid_tuning_result(
            base=
                base
        )
    )


    original_tuning = (
        promotion_executor
        .execute_ml_hyperparameter_tuning
    )


    original_training = (
        promotion_executor
        .execute_classical_ml
    )


    received_revision = None


    def fake_tuning(
        *,
        training_contract,
        search_contract,
    ):

        return tuning


    def fake_training(
        *,
        training_contract,
        expected_preparation_session_revision=None,
    ):

        nonlocal received_revision

        received_revision = (
            expected_preparation_session_revision
        )

        return (
            fake_final_execution(
                promoted_contract=
                    training_contract,

                revision=
                    tuning
                    .preparation_session_revision,
            )
        )


    promotion_executor.execute_ml_hyperparameter_tuning = (
        fake_tuning
    )


    promotion_executor.execute_classical_ml = (
        fake_training
    )


    try:
        execute_ml_tuned_model_promotion(
            promotion_contract=
                contract
        )

    finally:
        promotion_executor.execute_ml_hyperparameter_tuning = (
            original_tuning
        )

        promotion_executor.execute_classical_ml = (
            original_training
        )


    assert (
        received_revision
        ==
        tuning.preparation_session_revision
    )


# ============================================================
# FINAL REVISION FAIL CLOSED
# ============================================================


def test_final_execution_revision_mismatch_fails_closed(
) -> None:

    contract = (
        promotion_contract()
    )


    base = (
        contract.base_training_contract
    )


    tuning = (
        valid_tuning_result(
            base=
                base
        )
    )


    original_tuning = (
        promotion_executor
        .execute_ml_hyperparameter_tuning
    )


    original_training = (
        promotion_executor
        .execute_classical_ml
    )


    def fake_tuning(
        *,
        training_contract,
        search_contract,
    ):

        return tuning


    def fake_training(
        *,
        training_contract,
        expected_preparation_session_revision=None,
    ):

        return (
            fake_final_execution(
                promoted_contract=
                    training_contract,

                revision=
                    (
                        tuning
                        .preparation_session_revision
                        +
                        1
                    ),
            )
        )


    promotion_executor.execute_ml_hyperparameter_tuning = (
        fake_tuning
    )


    promotion_executor.execute_classical_ml = (
        fake_training
    )


    try:
        try:
            execute_ml_tuned_model_promotion(
                promotion_contract=
                    contract
            )

        except MLTunedModelPromotionExecutionError:
            pass

        else:
            raise AssertionError(
                (
                    "Final revision mismatch "
                    "must fail closed."
                )
            )

    finally:
        promotion_executor.execute_ml_hyperparameter_tuning = (
            original_tuning
        )

        promotion_executor.execute_classical_ml = (
            original_training
        )


# ============================================================
# HOLDOUT SHAPE FAIL CLOSED
# ============================================================


def test_final_holdout_shape_must_match_tuning(
) -> None:

    contract = (
        promotion_contract()
    )


    base = (
        contract.base_training_contract
    )


    tuning = (
        valid_tuning_result(
            base=
                base
        )
    )


    original_tuning = (
        promotion_executor
        .execute_ml_hyperparameter_tuning
    )


    original_training = (
        promotion_executor
        .execute_classical_ml
    )


    def fake_tuning(
        *,
        training_contract,
        search_contract,
    ):

        return tuning


    def fake_training(
        *,
        training_contract,
        expected_preparation_session_revision=None,
    ):

        return (
            fake_final_execution(
                promoted_contract=
                    training_contract,

                revision=
                    tuning
                    .preparation_session_revision,

                train_rows=
                    79,

                test_rows=
                    21,
            )
        )


    promotion_executor.execute_ml_hyperparameter_tuning = (
        fake_tuning
    )


    promotion_executor.execute_classical_ml = (
        fake_training
    )


    try:
        try:
            execute_ml_tuned_model_promotion(
                promotion_contract=
                    contract
            )

        except MLTunedModelPromotionExecutionError:
            pass

        else:
            raise AssertionError(
                (
                    "Final holdout shape mismatch "
                    "must fail closed."
                )
            )

    finally:
        promotion_executor.execute_ml_hyperparameter_tuning = (
            original_tuning
        )

        promotion_executor.execute_classical_ml = (
            original_training
        )


# ============================================================
# BASE CONTRACT IMMUTABILITY
# ============================================================


def test_promotion_does_not_mutate_base_contract(
) -> None:

    contract = (
        promotion_contract()
    )


    base = (
        contract.base_training_contract
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


    original_tuning = (
        promotion_executor
        .execute_ml_hyperparameter_tuning
    )


    original_training = (
        promotion_executor
        .execute_classical_ml
    )


    def fake_tuning(
        *,
        training_contract,
        search_contract,
    ):

        return tuning


    def fake_training(
        *,
        training_contract,
        expected_preparation_session_revision=None,
    ):

        return (
            fake_final_execution(
                promoted_contract=
                    training_contract,

                revision=
                    tuning
                    .preparation_session_revision,
            )
        )


    promotion_executor.execute_ml_hyperparameter_tuning = (
        fake_tuning
    )


    promotion_executor.execute_classical_ml = (
        fake_training
    )


    try:
        execute_ml_tuned_model_promotion(
            promotion_contract=
                contract
        )

    finally:
        promotion_executor.execute_ml_hyperparameter_tuning = (
            original_tuning
        )

        promotion_executor.execute_classical_ml = (
            original_training
        )


    assert (
        base.model_dump(
            mode="json"
        )
        ==
        before
    )


    assert (
        ml_training_contract_sha256(
            base
        )
        ==
        before_sha
    )


# ============================================================
# RESULT PRIVACY
# ============================================================


def test_result_is_privacy_minimal(
) -> None:

    contract = (
        promotion_contract()
    )


    base = (
        contract.base_training_contract
    )


    tuning = (
        valid_tuning_result(
            base=
                base
        )
    )


    original_tuning = (
        promotion_executor
        .execute_ml_hyperparameter_tuning
    )


    original_training = (
        promotion_executor
        .execute_classical_ml
    )


    def fake_tuning(
        *,
        training_contract,
        search_contract,
    ):

        return tuning


    def fake_training(
        *,
        training_contract,
        expected_preparation_session_revision=None,
    ):

        return (
            fake_final_execution(
                promoted_contract=
                    training_contract,

                revision=
                    tuning
                    .preparation_session_revision,
            )
        )


    promotion_executor.execute_ml_hyperparameter_tuning = (
        fake_tuning
    )


    promotion_executor.execute_classical_ml = (
        fake_training
    )


    try:
        result = (
            execute_ml_tuned_model_promotion(
                promotion_contract=
                    contract
            )
        )

    finally:
        promotion_executor.execute_ml_hyperparameter_tuning = (
            original_tuning
        )

        promotion_executor.execute_classical_ml = (
            original_training
        )


    payload = (
        result.model_dump(
            mode="json"
        )
    )


    forbidden = {
        "raw_rows",
        "predictions",
        "fold_predictions",
        "holdout_predictions",
        "x_train",
        "x_test",
        "y_train",
        "y_test",
        "model_bytes",
        "model_path",
        "estimator",
        "tuning_result",
    }


    assert (
        forbidden.isdisjoint(
            payload
        )
    )


# ============================================================
# VERSION
# ============================================================


def test_executor_rule_version(
) -> None:

    assert (
        ML_TUNED_MODEL_PROMOTION_EXECUTOR_RULE_VERSION
        ==
        "ml_tuned_model_promotion_executor_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML TUNED MODEL PROMOTION EXECUTOR v0.1 ==="
    )


    tests = [
        (
            "Classical ML expected revision blocks before fit",
            test_classical_ml_expected_revision_blocks_before_fit,
        ),
        (
            "Promotion replays tuning and promotes rank-1",
            test_promotion_replays_tuning_and_promotes_rank_1,
        ),
        (
            "Tuning revision pinned into final training",
            test_tuning_revision_is_required_by_final_training,
        ),
        (
            "Final revision mismatch fail-closed",
            test_final_execution_revision_mismatch_fails_closed,
        ),
        (
            "Final holdout shape mismatch fail-closed",
            test_final_holdout_shape_must_match_tuning,
        ),
        (
            "Base Training Contract remains immutable",
            test_promotion_does_not_mutate_base_contract,
        ),
        (
            "Privacy-minimal promotion result",
            test_result_is_privacy_minimal,
        ),
        (
            "Executor rule version",
            test_executor_rule_version,
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
        "PASS - ML Tuned Model Promotion Executor v0.1"
    )


if __name__ == "__main__":
    main()
