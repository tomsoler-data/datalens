from __future__ import annotations


import sys


from pathlib import (
    Path,
)


from app.ml.model_comparison_core import (
    ML_MODEL_COMPARISON_CORE_RULE_VERSION,
    ml_model_comparison_ranking_key,
)


from app.ml.model_comparison_executor import (
    ML_MODEL_COMPARISON_EXECUTOR_RULE_VERSION,
    _ranking_key,
)


# ============================================================
# REGRESSION RANKING PARITY
# ============================================================


def test_regression_ranking_key_matches_shared_core(
) -> None:

    cases = [
        (
            "linear_regression",
            {
                "rmse":
                    2.0,

                "mae":
                    1.5,

                "r2":
                    0.8,
            },
        ),

        (
            "ridge_regression",
            {
                "rmse":
                    2.0,

                "mae":
                    1.4,

                "r2":
                    0.7,
            },
        ),

        (
            "random_forest_regressor",
            {
                "rmse":
                    1.8,

                "mae":
                    1.6,

                "r2":
                    0.9,
            },
        ),
    ]


    for (
        estimator_key,
        metrics,
    ) in cases:

        historical = (
            _ranking_key(
                problem_type=
                    "regression",

                estimator_key=
                    estimator_key,

                metrics=
                    metrics,
            )
        )


        shared = (
            ml_model_comparison_ranking_key(
                problem_type=
                    "regression",

                estimator_key=
                    estimator_key,

                metrics=
                    metrics,
            )
        )


        assert (
            historical
            ==
            shared
        )


# ============================================================
# CLASSIFICATION RANKING PARITY
# ============================================================


def test_classification_ranking_key_matches_shared_core(
) -> None:

    cases = [
        (
            "logistic_regression",
            {
                "f1_macro":
                    0.81,

                "accuracy":
                    0.84,
            },
        ),

        (
            "random_forest_classifier",
            {
                "f1_macro":
                    0.83,

                "accuracy":
                    0.82,
            },
        ),
    ]


    for (
        estimator_key,
        metrics,
    ) in cases:

        historical = (
            _ranking_key(
                problem_type=
                    "classification",

                estimator_key=
                    estimator_key,

                metrics=
                    metrics,
            )
        )


        shared = (
            ml_model_comparison_ranking_key(
                problem_type=
                    "classification",

                estimator_key=
                    estimator_key,

                metrics=
                    metrics,
            )
        )


        assert (
            historical
            ==
            shared
        )


# ============================================================
# SOURCE WIRING
# ============================================================


def test_classical_executor_uses_shared_projection_and_core(
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
        "model_comparison_executor.py"
    )


    source = path.read_text(
        encoding="utf-8"
    )


    required = (
        "project_ml_model_candidate_result(",
        "aggregate_ml_model_candidates(",
        "aggregation.ranked_candidates",
        "legacy_ranked_estimator_keys",
        "shared_ranked_estimator_keys",
    )


    for token in required:

        assert (
            token
            in
            source
        )


# ============================================================
# RUNTIME ISOLATION
# ============================================================


def test_classical_shared_core_wiring_is_torch_free(
) -> None:

    assert (
        "torch"
        not in
        sys.modules
    )


    assert (
        "app.deep_learning.model_lab_executor"
        not in
        sys.modules
    )


    assert (
        "app.deep_learning.tabular_executor"
        not in
        sys.modules
    )


# ============================================================
# RULE VERSION PRESERVATION
# ============================================================


def test_rule_versions(
) -> None:

    assert (
        ML_MODEL_COMPARISON_EXECUTOR_RULE_VERSION
        ==
        "ml_model_comparison_executor_v0.1"
    )


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
        "=== DATALENS CLASSICAL MODEL COMPARISON / SHARED CORE WIRING v0.1 ==="
    )

    print()


    test_regression_ranking_key_matches_shared_core()

    print(
        "Regression ranking parity: PASS"
    )


    test_classification_ranking_key_matches_shared_core()

    print(
        "Classification ranking parity: PASS"
    )


    test_classical_executor_uses_shared_projection_and_core()

    print(
        "Historical executor -> shared core wiring: PASS"
    )


    test_classical_shared_core_wiring_is_torch_free()

    print(
        "Runtime / PyTorch isolation: PASS"
    )


    test_rule_versions()

    print(
        "Historical rule versions preserved: PASS"
    )


    print()

    print(
        "PASS - DataLens Classical Model Comparison / Shared Core Wiring v0.1"
    )


if __name__ == "__main__":
    main()
