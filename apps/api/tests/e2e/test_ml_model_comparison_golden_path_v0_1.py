from __future__ import annotations


import math


import numpy as np
import pandas as pd


from fastapi.testclient import (
    TestClient,
)


from sklearn.compose import (
    ColumnTransformer,
)


from sklearn.linear_model import (
    LinearRegression,
)


from sklearn.pipeline import (
    Pipeline,
)


# ============================================================
# REUSE REAL PREPARATION GOLDEN PATH
#
# Importing this module establishes the isolated product
# environment before app.main is imported.
# ============================================================


from tests.e2e.test_ml_preprocessing_golden_path_v0_1 import (
    WORKFLOW_ROOT_DATASET_ID,
    create_preparation_session,
    expected_revenue,
    reset_product_state,
    run_real_cleaning_plan,
    run_real_quality,
    select_analysis_output,
    validate_preparation,
    verify_preparation_persistence,
    verify_real_handoff,
)


from app.main import (
    app,
)


from app.ml.contracts import (
    MLPreprocessingContract,
    MLSplitContract,
    MLTrainingContract,
)


from app.ml.model_comparison_contracts import (
    MLModelComparisonContract,
)


from app.ml.model_comparison_executor import (
    MLModelComparisonExecutionResult,
    execute_ml_model_comparison,
)


from app.ml.model_loader import (
    load_trusted_ml_model,
)


from app.preparation.analysis_readiness_gate import (
    require_analysis_readiness,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_COMPARISON_GOLDEN_PATH_RULE_VERSION = (
    "ml_model_comparison_golden_path_v0.1"
)


# ============================================================
# COMMON ML AUTHORITY
# ============================================================


def build_common_preprocessing(
) -> MLPreprocessingContract:

    return (
        MLPreprocessingContract(
            numeric_imputation=
                "error",

            categorical_imputation=
                "error",

            categorical_encoding=
                "one_hot",

            handle_unknown_categories=
                "ignore",

            scale_numeric=
                True,
        )
    )


def build_common_split(
) -> MLSplitContract:

    return (
        MLSplitContract(
            test_size=
                0.20,

            random_seed=
                31,

            shuffle=
                True,

            stratify=
                False,
        )
    )


# ============================================================
# TRAINING CONTRACT HELPER
# ============================================================


def build_candidate(
    *,
    workflow_id: str,
    estimator_key: str,
    estimator_hyperparameters=None,
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                workflow_id,

            dataset_id=
                WORKFLOW_ROOT_DATASET_ID,

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
                estimator_key,

            estimator_hyperparameters=
                estimator_hyperparameters,

            preprocessing=
                build_common_preprocessing(),

            split=
                build_common_split(),
        )
    )


# ============================================================
# COMPARISON CONTRACT
# ============================================================


def build_comparison_contract(
    *,
    workflow_id: str,
) -> MLModelComparisonContract:
    """
    Candidate input order is deliberately NOT expected ranking
    order.

    This proves that selection comes from metrics + ranking
    policy rather than candidate insertion order.
    """

    return (
        MLModelComparisonContract(
            candidates=[
                build_candidate(
                    workflow_id=
                        workflow_id,

                    estimator_key=
                        "random_forest_regressor",

                    estimator_hyperparameters={
                        "kind":
                            "random_forest_regressor",

                        "n_estimators":
                            96,

                        "max_depth":
                            10,

                        "min_samples_split":
                            2,

                        "min_samples_leaf":
                            1,

                        "max_features":
                            "sqrt",

                        "bootstrap":
                            True,
                    },
                ),

                build_candidate(
                    workflow_id=
                        workflow_id,

                    estimator_key=
                        "ridge_regression",

                    estimator_hyperparameters={
                        "kind":
                            "ridge_regression",

                        "alpha":
                            2.0,

                        "fit_intercept":
                            True,
                    },
                ),

                build_candidate(
                    workflow_id=
                        workflow_id,

                    estimator_key=
                        "linear_regression",
                ),
            ]
        )
    )


# ============================================================
# REAL MODEL COMPARISON
# ============================================================


def run_real_model_comparison(
    *,
    workflow_id: str,
) -> tuple[
    MLModelComparisonContract,
    MLModelComparisonExecutionResult,
]:

    contract = (
        build_comparison_contract(
            workflow_id=
                workflow_id
        )
    )


    # --------------------------------------------------------
    # REAL SERVER-OWNED SNAPSHOT BEFORE COMPARISON
    # --------------------------------------------------------

    readiness_before = (
        require_analysis_readiness(
            workflow_id=
                workflow_id
        )
    )


    result = (
        execute_ml_model_comparison(
            comparison_contract=
                contract
        )
    )


    # --------------------------------------------------------
    # REAL SERVER-OWNED SNAPSHOT AFTER COMPARISON
    # --------------------------------------------------------

    readiness_after = (
        require_analysis_readiness(
            workflow_id=
                workflow_id
        )
    )


    assert (
        readiness_before.session_revision
        ==
        readiness_after.session_revision
    )


    assert (
        result.preparation_session_revision
        ==
        readiness_before.session_revision
    )


    assert (
        tuple(
            readiness_before
            .requested_analysis_dataset_ids
        )
        ==
        (
            WORKFLOW_ROOT_DATASET_ID,
        )
    )


    print(
        (
            "[PASS] complete comparison remained "
            "pinned to one Preparation revision"
        )
    )


    # --------------------------------------------------------
    # AUTHORITY
    # --------------------------------------------------------

    assert (
        result.workflow_id
        ==
        workflow_id
    )


    assert (
        result.dataset_id
        ==
        WORKFLOW_ROOT_DATASET_ID
    )


    assert (
        result.problem_type
        ==
        "regression"
    )


    assert (
        result.comparison_contract
        ==
        contract
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


    print(
        (
            "[PASS] real Model Comparison preserved "
            "dataset / preprocessing / split authority"
        )
    )


    # --------------------------------------------------------
    # COMPLETE RANKING
    # --------------------------------------------------------

    assert (
        len(
            result.candidates
        )
        ==
        3
    )


    assert (
        [
            candidate.rank

            for candidate
            in result.candidates
        ]
        ==
        [
            1,
            2,
            3,
        ]
    )


    assert (
        {
            candidate.estimator_key

            for candidate
            in result.candidates
        }
        ==
        {
            "linear_regression",
            "ridge_regression",
            "random_forest_regressor",
        }
    )


    assert (
        len(
            {
                candidate
                .model_artifact
                .model_id

                for candidate
                in result.candidates
            }
        )
        ==
        3
    )


    print(
        (
            "[PASS] three fixed estimators produced "
            "three independently persisted candidates"
        )
    )


    # --------------------------------------------------------
    # SAME HOLDOUT SHAPE + FINITE METRICS
    # --------------------------------------------------------

    assert (
        {
            candidate.train_rows

            for candidate
            in result.candidates
        }
        ==
        {
            24
        }
    )


    assert (
        {
            candidate.test_rows

            for candidate
            in result.candidates
        }
        ==
        {
            6
        }
    )


    for candidate in (
        result.candidates
    ):

        assert (
            candidate.primary_metric
            ==
            "rmse"
        )


        assert (
            candidate.primary_metric_value
            ==
            candidate.metrics[
                "rmse"
            ]
        )


        for metric_value in (
            candidate.metrics.values()
        ):

            assert (
                math.isfinite(
                    float(
                        metric_value
                    )
                )
            )


    print(
        (
            "[PASS] every estimator used the same "
            "deterministic 24/6 holdout shape"
        )
    )


    # --------------------------------------------------------
    # WINNER
    #
    # The golden dataset is deliberately exactly linear:
    #
    # revenue =
    #   50 + 2*age + 3*tenure + categorical effect
    #
    # LinearRegression should therefore dominate RMSE.
    # --------------------------------------------------------

    winner = (
        result.candidates[
            0
        ]
    )


    assert (
        winner.estimator_key
        ==
        "linear_regression"
    )


    assert (
        result.selected_estimator_key
        ==
        "linear_regression"
    )


    assert (
        result.selected_model_id
        ==
        winner
        .model_artifact
        .model_id
    )


    assert (
        winner.metrics[
            "rmse"
        ]
        <
        1e-8
    )


    minimum_rmse = min(
        candidate.metrics[
            "rmse"
        ]

        for candidate
        in result.candidates
    )


    assert (
        winner.metrics[
            "rmse"
        ]
        ==
        minimum_rmse
    )


    print(
        (
            "[PASS] deterministic RMSE ranking selected "
            "LinearRegression from non-ranked input order"
        )
    )


    return (
        contract,
        result,
    )


# ============================================================
# MODEL ARTIFACT PROVENANCE
# ============================================================


def verify_candidate_artifacts(
    *,
    workflow_id: str,
    contract: MLModelComparisonContract,
    result: MLModelComparisonExecutionResult,
) -> None:

    expected_contracts = {
        candidate.estimator_key:
            candidate

        for candidate
        in contract.candidates
    }


    for candidate in (
        result.candidates
    ):

        artifact = (
            candidate
            .model_artifact
        )


        assert (
            artifact.workflow_id
            ==
            workflow_id
        )


        assert (
            artifact.dataset_id
            ==
            WORKFLOW_ROOT_DATASET_ID
        )


        assert (
            artifact.training_contract
            ==
            expected_contracts[
                candidate.estimator_key
            ]
        )


        assert (
            artifact.metrics
            ==
            candidate.metrics
        )


        assert (
            artifact.serialization_format
            ==
            "joblib"
        )


        assert (
            artifact.model_file_bytes
            >
            0
        )


        assert (
            len(
                artifact.model_sha256
            )
            ==
            64
        )


    print(
        (
            "[PASS] every ranked candidate preserves "
            "server-owned Model Artifact provenance + SHA"
        )
    )


# ============================================================
# TRUSTED RELOAD
# ============================================================


def verify_all_candidates_reload(
    *,
    workflow_id: str,
    result: MLModelComparisonExecutionResult,
):
    selected_loaded = None


    for candidate in (
        result.candidates
    ):

        loaded = (
            load_trusted_ml_model(
                workflow_id=
                    workflow_id,

                model_id=
                    candidate
                    .model_artifact
                    .model_id,
            )
        )


        assert (
            loaded.artifact
            .model_id
            ==
            candidate
            .model_artifact
            .model_id
        )


        assert (
            loaded.artifact
            .training_contract
            ==
            candidate
            .model_artifact
            .training_contract
        )


        assert isinstance(
            loaded.estimator,
            Pipeline,
        )


        assert (
            "preprocessor"
            in
            loaded
            .estimator
            .named_steps
        )


        assert (
            "estimator"
            in
            loaded
            .estimator
            .named_steps
        )


        assert isinstance(
            loaded
            .estimator
            .named_steps[
                "preprocessor"
            ],
            ColumnTransformer,
        )


        if (
            candidate
            .model_artifact
            .model_id
            ==
            result.selected_model_id
        ):
            selected_loaded = (
                loaded
            )


    assert (
        selected_loaded
        is not None
    )


    print(
        (
            "[PASS] every comparison candidate crossed "
            "trusted SHA-verified reload"
        )
    )


    return (
        selected_loaded
    )


# ============================================================
# SELECTED PIPELINE
# ============================================================


def verify_selected_pipeline(
    *,
    selected_loaded,
) -> None:

    pipeline = (
        selected_loaded
        .estimator
    )


    assert isinstance(
        pipeline,
        Pipeline,
    )


    estimator = (
        pipeline
        .named_steps[
            "estimator"
        ]
    )


    assert isinstance(
        estimator,
        LinearRegression,
    )


    print(
        (
            "[PASS] selected Model Artifact reload "
            "restored LinearRegression pipeline"
        )
    )


# ============================================================
# SELECTED MODEL PREDICTION
# ============================================================


def verify_selected_known_predictions(
    *,
    selected_loaded,
) -> None:

    features = (
        pd.DataFrame(
            {
                "age": [
                    33.0,
                    44.0,
                ],

                "tenure": [
                    6.0,
                    11.0,
                ],

                "segment": [
                    "standard",
                    "premium",
                ],
            }
        )
    )


    expected = (
        np.asarray(
            [
                expected_revenue(
                    age=
                        33.0,

                    tenure=
                        6.0,

                    segment=
                        "standard",
                ),

                expected_revenue(
                    age=
                        44.0,

                    tenure=
                        11.0,

                    segment=
                        "premium",
                ),
            ],
            dtype=np.float64,
        )
    )


    predictions = (
        np.asarray(
            selected_loaded.predict(
                features
            ),
            dtype=np.float64,
        )
    )


    assert (
        predictions.shape
        ==
        expected.shape
    )


    assert (
        np.isfinite(
            predictions
        )
        .all()
    )


    assert (
        np.allclose(
            predictions,
            expected,
            rtol=1e-9,
            atol=1e-8,
        )
    )


    print(
        (
            "[PASS] selected trusted model predicts "
            "known mixed-type observations correctly"
        )
    )


# ============================================================
# UNSEEN CATEGORY
# ============================================================


def verify_selected_unseen_category(
    *,
    selected_loaded,
) -> None:

    features = (
        pd.DataFrame(
            {
                "age": [
                    36.0,
                ],

                "tenure": [
                    8.0,
                ],

                "segment": [
                    "enterprise",
                ],
            }
        )
    )


    predictions = (
        np.asarray(
            selected_loaded.predict(
                features
            ),
            dtype=np.float64,
        )
    )


    assert (
        predictions.shape
        ==
        (
            1,
        )
    )


    assert (
        np.isfinite(
            predictions
        )
        .all()
    )


    print(
        (
            "[PASS] selected persisted pipeline handles "
            "unseen category without refit"
        )
    )


# ============================================================
# RULE VERSION
# ============================================================


def verify_rule_version(
) -> None:

    assert (
        ML_MODEL_COMPARISON_GOLDEN_PATH_RULE_VERSION
        ==
        "ml_model_comparison_golden_path_v0.1"
    )


    print(
        (
            "[PASS] ML Model Comparison "
            "Golden Path rule version"
        )
    )


# ============================================================
# GOLDEN PATH
# ============================================================


def test_ml_model_comparison_golden_path_v0_1(
) -> None:

    reset_product_state()


    with TestClient(
        app
    ) as client:

        # ----------------------------------------------------
        # REAL PREPARATION
        # ----------------------------------------------------

        workflow_id = (
            create_preparation_session(
                client
            )
        )


        run_real_quality(
            client,
            workflow_id=
                workflow_id,
        )


        run_real_cleaning_plan(
            client,
            workflow_id=
                workflow_id,
        )


        select_analysis_output(
            client,
            workflow_id=
                workflow_id,
        )


        validate_preparation(
            client,
            workflow_id=
                workflow_id,
        )


        verify_preparation_persistence(
            workflow_id=
                workflow_id,
        )


        verify_real_handoff(
            workflow_id=
                workflow_id,
        )


        # ----------------------------------------------------
        # REAL MODEL COMPARISON
        # ----------------------------------------------------

        (
            comparison_contract,
            comparison_result,
        ) = (
            run_real_model_comparison(
                workflow_id=
                    workflow_id
            )
        )


        # ----------------------------------------------------
        # ALL SERVER-OWNED MODEL ARTIFACTS
        # ----------------------------------------------------

        verify_candidate_artifacts(
            workflow_id=
                workflow_id,

            contract=
                comparison_contract,

            result=
                comparison_result,
        )


        # ----------------------------------------------------
        # TRUSTED RELOAD
        # ----------------------------------------------------

        selected_loaded = (
            verify_all_candidates_reload(
                workflow_id=
                    workflow_id,

                result=
                    comparison_result,
            )
        )


        verify_selected_pipeline(
            selected_loaded=
                selected_loaded
        )


        # ----------------------------------------------------
        # REAL PREDICTIONS
        # ----------------------------------------------------

        verify_selected_known_predictions(
            selected_loaded=
                selected_loaded
        )


        verify_selected_unseen_category(
            selected_loaded=
                selected_loaded
        )


        verify_rule_version()


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print()

    print(
        "="
        *
        78
    )

    print(
        (
            "DATALENS ML MODEL COMPARISON "
            "GOLDEN PATH E2E v0.1"
        )
    )

    print(
        "="
        *
        78
    )

    print(
        "Preparation : real validated mixed-type CSV"
    )

    print(
        "Candidates   : Linear + Ridge + Random Forest"
    )

    print(
        "Snapshot     : pinned Preparation revision"
    )

    print(
        "Ranking      : deterministic RMSE policy"
    )

    print(
        "Selection    : rank #1 Model Artifact"
    )

    print(
        "Reload       : trusted SHA-verified joblib boundary"
    )

    print(
        "Prediction   : known + unseen category"
    )

    print()


    test_ml_model_comparison_golden_path_v0_1()


    print()

    print(
        "="
        *
        78
    )

    print(
        (
            "PASS - Mixed CSV → Preparation → VALIDATE → "
            "3 Models → Deterministic Ranking → "
            "Selected Model Artifact → Trusted Reload → Prediction"
        )
    )

    print(
        "="
        *
        78
    )


if __name__ == "__main__":
    main()