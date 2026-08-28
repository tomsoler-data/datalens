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


from sklearn.ensemble import (
    RandomForestRegressor,
)


from sklearn.pipeline import (
    Pipeline,
)


# ============================================================
# REUSE THE ALREADY-PROVEN REAL PREPARATION GOLDEN PATH
#
# Importing this module also establishes its isolated temporary
# product environment before app.main is imported.
#
# We reuse only non-test helpers so this E2E does not duplicate
# the full Preparation setup.
# ============================================================


from tests.e2e.test_ml_preprocessing_golden_path_v0_1 import (
    WORKFLOW_ROOT_DATASET_ID,
    create_preparation_session,
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


from app.ml.classical_executor import (
    execute_classical_ml,
)


from app.ml.contracts import (
    MLPreprocessingContract,
    MLSplitContract,
    MLTrainingContract,
)


from app.ml.model_loader import (
    load_trusted_ml_model,
)


# ============================================================
# VERSION
# ============================================================


ML_ESTIMATOR_GOLDEN_PATH_RULE_VERSION = (
    "ml_estimator_golden_path_v0.1"
)


# ============================================================
# TRAINING CONTRACT
# ============================================================


def build_random_forest_contract(
    *,
    workflow_id: str,
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

            preprocessing=(
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
            ),

            split=(
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
            ),
        )
    )


# ============================================================
# REAL RANDOM FOREST TRAINING
# ============================================================


def train_real_random_forest(
    *,
    workflow_id: str,
):
    contract = (
        build_random_forest_contract(
            workflow_id=
                workflow_id
        )
    )


    result = (
        execute_classical_ml(
            training_contract=
                contract
        )
    )


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
        result.estimator_key
        ==
        "random_forest_regressor"
    )


    assert (
        result.train_rows
        ==
        24
    )


    assert (
        result.test_rows
        ==
        6
    )


    assert (
        result.model_artifact
        .training_contract
        ==
        contract
    )


    assert (
        result.model_artifact
        .training_contract
        .estimator_key
        ==
        "random_forest_regressor"
    )


    hyperparameters = (
        result
        .model_artifact
        .training_contract
        .effective_estimator_hyperparameters
    )


    assert (
        hyperparameters
        is not None
    )


    assert (
        hyperparameters.n_estimators
        ==
        96
    )


    assert (
        hyperparameters.max_depth
        ==
        10
    )


    for (
        metric_name,
        metric_value,
    ) in result.metrics.items():

        assert (
            math.isfinite(
                float(
                    metric_value
                )
            )
        ), (
            metric_name,
            metric_value,
        )


    print(
        (
            "[PASS] real validated Preparation "
            "trained RandomForestRegressor"
        )
    )


    return (
        contract,
        result,
    )


# ============================================================
# TRUSTED MODEL ARTIFACT RELOAD
# ============================================================


def verify_random_forest_reload(
    *,
    workflow_id: str,
    contract: MLTrainingContract,
    execution_result,
):
    loaded = (
        load_trusted_ml_model(
            workflow_id=
                workflow_id,

            model_id=
                execution_result
                .model_artifact
                .model_id,
        )
    )


    assert (
        loaded.artifact
        .model_id
        ==
        execution_result
        .model_artifact
        .model_id
    )


    assert (
        loaded.artifact
        .training_contract
        ==
        contract
    )


    pipeline = (
        loaded.estimator
    )


    assert isinstance(
        pipeline,
        Pipeline,
    )


    assert (
        "preprocessor"
        in
        pipeline.named_steps
    )


    assert (
        "estimator"
        in
        pipeline.named_steps
    )


    preprocessor = (
        pipeline
        .named_steps[
            "preprocessor"
        ]
    )


    estimator = (
        pipeline
        .named_steps[
            "estimator"
        ]
    )


    assert isinstance(
        preprocessor,
        ColumnTransformer,
    )


    assert isinstance(
        estimator,
        RandomForestRegressor,
    )


    print(
        (
            "[PASS] trusted Model Artifact reload "
            "restores RandomForest pipeline"
        )
    )


    return (
        loaded,
        estimator,
    )


# ============================================================
# SERVER-OWNED EXECUTION CONTROLS
# ============================================================


def verify_server_owned_controls(
    *,
    contract: MLTrainingContract,
    estimator: RandomForestRegressor,
) -> None:

    assert (
        estimator.random_state
        ==
        contract
        .split
        .random_seed
    )


    assert (
        estimator.random_state
        ==
        31
    )


    assert (
        estimator.n_jobs
        ==
        1
    )


    assert (
        estimator.n_estimators
        ==
        96
    )


    assert (
        estimator.max_depth
        ==
        10
    )


    assert (
        estimator.min_samples_split
        ==
        2
    )


    assert (
        estimator.min_samples_leaf
        ==
        1
    )


    assert (
        estimator.max_features
        ==
        "sqrt"
    )


    assert (
        estimator.bootstrap
        is True
    )


    print(
        (
            "[PASS] persisted Random Forest keeps "
            "server-owned seed + execution controls"
        )
    )


# ============================================================
# KNOWN CATEGORY PREDICTION
# ============================================================


def verify_known_category_prediction(
    *,
    loaded,
) -> None:

    prediction_input = (
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


    predictions = (
        loaded.predict(
            prediction_input
        )
    )


    assert (
        len(
            predictions
        )
        ==
        2
    )


    numeric_predictions = (
        np.asarray(
            predictions,
            dtype=np.float64,
        )
    )


    assert (
        np.isfinite(
            numeric_predictions
        )
        .all()
    )


    print(
        (
            "[PASS] trusted Random Forest predicts "
            "known categorical feature values"
        )
    )


# ============================================================
# UNSEEN CATEGORY PREDICTION
# ============================================================


def verify_unseen_category_prediction(
    *,
    loaded,
) -> None:

    prediction_input = (
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
        loaded.predict(
            prediction_input
        )
    )


    assert (
        len(
            predictions
        )
        ==
        1
    )


    assert (
        np.isfinite(
            np.asarray(
                predictions,
                dtype=np.float64,
            )
        )
        .all()
    )


    print(
        (
            "[PASS] persisted Random Forest handles "
            "unseen category without refit"
        )
    )


# ============================================================
# MODEL ARTIFACT PROVENANCE
# ============================================================


def verify_model_artifact_provenance(
    *,
    workflow_id: str,
    contract: MLTrainingContract,
    execution_result,
) -> None:

    artifact = (
        execution_result
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
        contract
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
            "[PASS] Random Forest Model Artifact "
            "preserves server-owned provenance + SHA"
        )
    )


# ============================================================
# RULE VERSION
# ============================================================


def verify_rule_version(
) -> None:

    assert (
        ML_ESTIMATOR_GOLDEN_PATH_RULE_VERSION
        ==
        "ml_estimator_golden_path_v0.1"
    )


    print(
        "[PASS] ML Estimator Golden Path rule version"
    )


# ============================================================
# GOLDEN PATH
# ============================================================


def test_ml_estimator_golden_path_v0_1(
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
        # REAL RANDOM FOREST TRAINING
        # ----------------------------------------------------

        (
            contract,
            execution_result,
        ) = (
            train_real_random_forest(
                workflow_id=
                    workflow_id
            )
        )


        # ----------------------------------------------------
        # SERVER-OWNED ARTIFACT
        # ----------------------------------------------------

        verify_model_artifact_provenance(
            workflow_id=
                workflow_id,

            contract=
                contract,

            execution_result=
                execution_result,
        )


        # ----------------------------------------------------
        # TRUSTED RELOAD
        # ----------------------------------------------------

        (
            loaded,
            estimator,
        ) = (
            verify_random_forest_reload(
                workflow_id=
                    workflow_id,

                contract=
                    contract,

                execution_result=
                    execution_result,
            )
        )


        verify_server_owned_controls(
            contract=
                contract,

            estimator=
                estimator,
        )


        # ----------------------------------------------------
        # REAL PREDICTIONS
        # ----------------------------------------------------

        verify_known_category_prediction(
            loaded=
                loaded
        )


        verify_unseen_category_prediction(
            loaded=
                loaded
        )


        verify_rule_version()


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print()

    print(
        "=" * 78
    )


    print(
        "DATALENS ML ESTIMATOR GOLDEN PATH E2E v0.1"
    )


    print(
        "=" * 78
    )


    print(
        "Preparation : real validated mixed-type CSV"
    )


    print(
        "Preprocess  : real persisted ColumnTransformer"
    )


    print(
        "Estimator   : real RandomForestRegressor"
    )


    print(
        "Controls    : server-owned random_state + n_jobs"
    )


    print(
        "Persistence : real Model Artifact + SHA"
    )


    print(
        "Reload      : trusted joblib boundary"
    )


    print(
        "Prediction  : known + unseen category"
    )


    print()


    test_ml_estimator_golden_path_v0_1()


    print()

    print(
        "=" * 78
    )


    print(
        (
            "PASS - Mixed CSV → Preparation → VALIDATE → "
            "Random Forest → Model Artifact → Trusted Reload → "
            "Prediction"
        )
    )


    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()