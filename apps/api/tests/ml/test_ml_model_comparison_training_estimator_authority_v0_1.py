from __future__ import annotations


import sys


from pathlib import (
    Path,
)


from pydantic import (
    ValidationError,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.model_comparison_contracts import (
    ML_MODEL_COMPARISON_CONTRACT_RULE_VERSION,
    MLModelComparisonContract,
)


from app.ml.training_estimator_contracts import (
    ML_TRAINING_ESTIMATOR_BRIDGE_RULE_VERSION,
    training_estimator_problem_type,
)


# ============================================================
# CONTRACT HELPERS
# ============================================================


def regression_candidate(
    *,
    estimator_key: str,
    feature_columns: list[str] | None = None,
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:comparison",

            dataset_id=
                "dataset:validated",

            problem_type=
                "regression",

            target_column=
                "target",

            feature_columns=(
                feature_columns
                if feature_columns is not None
                else [
                    "feature_a",
                    "feature_b",
                ]
            ),

            estimator_key=
                estimator_key,
        )
    )


def classification_candidate(
    *,
    estimator_key: str,
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:comparison-classification",

            dataset_id=
                "dataset:validated",

            problem_type=
                "classification",

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


# ============================================================
# SHARED ESTIMATOR AUTHORITY
# ============================================================


def test_training_estimator_problem_type_authority(
) -> None:

    assert (
        training_estimator_problem_type(
            "linear_regression"
        )
        ==
        "regression"
    )


    assert (
        training_estimator_problem_type(
            "ridge_regression"
        )
        ==
        "regression"
    )


    assert (
        training_estimator_problem_type(
            "random_forest_regressor"
        )
        ==
        "regression"
    )


    assert (
        training_estimator_problem_type(
            "logistic_regression"
        )
        ==
        "classification"
    )


    assert (
        training_estimator_problem_type(
            "random_forest_classifier"
        )
        ==
        "classification"
    )


    assert (
        training_estimator_problem_type(
            "tabular_mlp_regressor"
        )
        ==
        "regression"
    )


    assert (
        training_estimator_problem_type(
            "unknown_estimator"
        )
        is None
    )


# ============================================================
# MIXED CLASSICAL / MLP CONTRACT
# ============================================================


def test_comparison_contract_accepts_classical_and_mlp_regression(
) -> None:

    contract = (
        MLModelComparisonContract(
            candidates=[
                regression_candidate(
                    estimator_key=
                        "linear_regression"
                ),

                regression_candidate(
                    estimator_key=
                        "tabular_mlp_regressor"
                ),
            ]
        )
    )


    assert (
        contract.problem_type
        ==
        "regression"
    )


    assert (
        contract.primary_metric
        ==
        "rmse"
    )


    assert (
        contract.ranking_policy
        ==
        "regression_rmse_v0.1"
    )


    assert [
        candidate.estimator_key

        for candidate
        in contract.candidates
    ] == [
        "linear_regression",
        "tabular_mlp_regressor",
    ]


# ============================================================
# CLASSIFICATION PRESERVATION
# ============================================================


def test_classical_classification_contract_is_preserved(
) -> None:

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


    assert (
        contract.primary_metric
        ==
        "f1_macro"
    )


    assert (
        contract.ranking_policy
        ==
        "classification_f1_macro_v0.1"
    )


# ============================================================
# UNKNOWN ESTIMATOR FAIL-CLOSED
# ============================================================


def test_unknown_estimator_remains_blocked(
) -> None:

    try:

        MLModelComparisonContract(
            candidates=[
                regression_candidate(
                    estimator_key=
                        "linear_regression"
                ),

                regression_candidate(
                    estimator_key=
                        "unknown_estimator"
                ),
            ]
        )

    except ValidationError as error:

        assert (
            "unsupported estimator"
            in
            str(
                error
            )
        )

        return


    raise AssertionError(
        "Unknown comparison estimator must fail closed."
    )


# ============================================================
# FAIRNESS PRESERVATION
# ============================================================


def test_fairness_constraints_remain_enforced(
) -> None:

    try:

        MLModelComparisonContract(
            candidates=[
                regression_candidate(
                    estimator_key=
                        "linear_regression"
                ),

                regression_candidate(
                    estimator_key=
                        "tabular_mlp_regressor",

                    feature_columns=[
                        "feature_b",
                        "feature_a",
                    ],
                ),
            ]
        )

    except ValidationError as error:

        assert (
            "feature_columns"
            in
            str(
                error
            )
        )

        return


    raise AssertionError(
        (
            "Mixed-framework comparison must still "
            "enforce identical ordered features."
        )
    )


# ============================================================
# RUNTIME / SOURCE ISOLATION
# ============================================================


def test_comparison_estimator_authority_is_torch_free(
) -> None:

    assert (
        "torch"
        not in
        sys.modules
    )


    bridge_path = (
        Path(__file__)
        .parents[
            2
        ]
        /
        "app"
        /
        "ml"
        /
        "training_estimator_contracts.py"
    )


    comparison_path = (
        Path(__file__)
        .parents[
            2
        ]
        /
        "app"
        /
        "ml"
        /
        "model_comparison_contracts.py"
    )


    bridge_source = bridge_path.read_text(
        encoding="utf-8"
    )


    comparison_source = (
        comparison_path.read_text(
            encoding="utf-8"
        )
    )


    for source in (
        bridge_source,
        comparison_source,
    ):

        assert (
            "import torch"
            not in
            source
        )


        assert (
            "from torch"
            not in
            source
        )


    assert (
        "execute_tabular_mlp"
        not in
        comparison_source
    )


    assert (
        "app.deep_learning.model_lab_executor"
        not in
        comparison_source
    )


# ============================================================
# RULE VERSIONS
# ============================================================


def test_rule_versions(
) -> None:

    assert (
        ML_TRAINING_ESTIMATOR_BRIDGE_RULE_VERSION
        ==
        "ml_training_estimator_bridge_v0.1"
    )


    assert (
        ML_MODEL_COMPARISON_CONTRACT_RULE_VERSION
        ==
        "ml_model_comparison_contract_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS MODEL COMPARISON TRAINING ESTIMATOR AUTHORITY v0.1 ==="
    )

    print()


    test_training_estimator_problem_type_authority()

    print(
        "Classical + MLP problem-type authority: PASS"
    )


    test_comparison_contract_accepts_classical_and_mlp_regression()

    print(
        "Mixed Classical / MLP comparison contract: PASS"
    )


    test_classical_classification_contract_is_preserved()

    print(
        "Classical classification comparison preservation: PASS"
    )


    test_unknown_estimator_remains_blocked()

    print(
        "Unknown estimator fail-closed guard: PASS"
    )


    test_fairness_constraints_remain_enforced()

    print(
        "Mixed-framework fairness constraints: PASS"
    )


    test_comparison_estimator_authority_is_torch_free()

    print(
        "Runtime / comparison contract torch isolation: PASS"
    )


    test_rule_versions()

    print(
        "Comparison estimator rule versions: PASS"
    )


    print()

    print(
        "PASS - DataLens Model Comparison Training Estimator Authority v0.1"
    )


if __name__ == "__main__":
    main()
