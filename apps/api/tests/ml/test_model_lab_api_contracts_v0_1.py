from __future__ import annotations

import math

from pydantic import (
    ValidationError,
)

from app.api.model_lab_contracts import (
    MODEL_LAB_API_CONTRACT_RULE_VERSION,
    MODEL_LAB_PREDICTION_MAX_ROWS,
    ModelLabAPIErrorDetail,
    ModelLabEvaluateRequest,
    ModelLabModelCard,
    ModelLabModelDetail,
    ModelLabModelListResponse,
    ModelLabPredictRequest,
    ModelLabPredictResponse,
)

from app.ml.contracts import (
    MLPreprocessingContract,
    MLSplitContract,
)

from app.ml.decision_threshold import (
    MLDecisionThresholdContract,
)

from app.ml.estimator_contracts import (
    MLLogisticRegressionHyperparameters,
)


WORKFLOW_ID = "prep:model-lab"
DATASET_ID = "dataset:model-lab"
MODEL_ID = "model:model-lab"

EXPERIMENT_ID = (
    "experiment:"
    +
    (
        "a"
        *
        32
    )
)

TRAINING_SHA = (
    "b"
    *
    64
)


def expect_validation_error(
    factory,
) -> None:

    try:
        factory()

    except ValidationError:
        return

    raise AssertionError(
        "Expected pydantic ValidationError."
    )


def model_card(
    *,
    model_id: str = MODEL_ID,
    workflow_id: str = WORKFLOW_ID,
    with_provenance: bool = True,
) -> ModelLabModelCard:

    return (
        ModelLabModelCard(
            model_id=
                model_id,

            workflow_id=
                workflow_id,

            dataset_id=
                DATASET_ID,

            problem_type=
                "classification",

            target_column=
                "churned",

            estimator_key=
                "logistic_regression",

            feature_columns=[
                "age",
                "segment",
            ],

            categorical_feature_columns=[
                "segment",
            ],

            metrics={
                "accuracy":
                    0.80,

                "f1_macro":
                    0.78,

                "precision_macro":
                    0.79,

                "recall_macro":
                    0.77,

                "balanced_accuracy":
                    0.77,
            },

            train_rows=
                80,

            test_rows=
                20,

            created_at_utc=(
                "2026-08-29T10:00:00+00:00"
            ),

            experiment_id=(
                EXPERIMENT_ID
                if with_provenance
                else None
            ),

            preparation_session_revision=(
                9
                if with_provenance
                else None
            ),

            training_contract_sha256=(
                TRAINING_SHA
                if with_provenance
                else None
            ),

            has_experiment_provenance=(
                with_provenance
            ),
        )
    )


def test_valid_model_card(
) -> None:

    result = (
        model_card()
    )

    assert (
        result.model_id
        ==
        MODEL_ID
    )

    assert (
        result.problem_type
        ==
        "classification"
    )

    assert (
        result.has_experiment_provenance
        is True
    )


def test_legacy_model_card_without_provenance(
) -> None:

    result = (
        model_card(
            with_provenance=
                False
        )
    )

    assert (
        result.experiment_id
        is None
    )

    assert (
        result.training_contract_sha256
        is None
    )


def test_partial_provenance_fails_closed(
) -> None:

    payload = (
        model_card()
        .model_dump(
            mode="python"
        )
    )

    payload[
        "training_contract_sha256"
    ] = None

    expect_validation_error(
        lambda:
            ModelLabModelCard(
                **payload
            )
    )


def test_categorical_features_must_be_features(
) -> None:

    payload = (
        model_card()
        .model_dump(
            mode="python"
        )
    )

    payload[
        "categorical_feature_columns"
    ] = [
        "unknown_column"
    ]

    expect_validation_error(
        lambda:
            ModelLabModelCard(
                **payload
            )
    )


def test_model_card_excludes_internal_artifact_fields(
) -> None:

    fields = set(
        ModelLabModelCard
        .model_fields
    )

    forbidden = {
        "model_path",
        "model_file_bytes",
        "model_sha256",
        "model_bytes",
        "estimator",
        "training_contract",
    }

    assert (
        forbidden.isdisjoint(
            fields
        )
    )


def test_valid_model_detail(
) -> None:

    card = (
        model_card()
    )

    detail = (
        ModelLabModelDetail(
            **card.model_dump(
                mode="python"
            ),

            preprocessing=(
                MLPreprocessingContract()
            ),

            split=(
                MLSplitContract()
            ),

            effective_estimator_hyperparameters=(
                MLLogisticRegressionHyperparameters()
            ),
        )
    )

    assert (
        detail
        .effective_estimator_hyperparameters
        .kind
        ==
        "logistic_regression"
    )


def test_valid_model_list(
) -> None:

    first = (
        model_card()
    )

    second = (
        model_card(
            model_id=
                "model:second"
        )
    )

    result = (
        ModelLabModelListResponse(
            workflow_id=
                WORKFLOW_ID,

            model_count=
                2,

            models=[
                first,
                second,
            ],
        )
    )

    assert (
        result.model_count
        ==
        2
    )


def test_model_list_count_is_bound(
) -> None:

    expect_validation_error(
        lambda:
            ModelLabModelListResponse(
                workflow_id=
                    WORKFLOW_ID,

                model_count=
                    2,

                models=[
                    model_card()
                ],
            )
    )


def test_model_list_workflow_is_bound(
) -> None:

    expect_validation_error(
        lambda:
            ModelLabModelListResponse(
                workflow_id=
                    WORKFLOW_ID,

                model_count=
                    1,

                models=[
                    model_card(
                        workflow_id=
                            "prep:other"
                    )
                ],
            )
    )


def test_model_list_rejects_duplicate_ids(
) -> None:

    expect_validation_error(
        lambda:
            ModelLabModelListResponse(
                workflow_id=
                    WORKFLOW_ID,

                model_count=
                    2,

                models=[
                    model_card(),
                    model_card(),
                ],
            )
    )


def test_evaluate_request_defaults(
) -> None:

    request = (
        ModelLabEvaluateRequest(
            workflow_id=
                WORKFLOW_ID,

            model_id=
                MODEL_ID,
        )
    )

    assert (
        request.evaluation
        .decision_threshold
        is None
    )


def test_evaluate_request_accepts_explicit_threshold(
) -> None:

    request = (
        ModelLabEvaluateRequest(
            workflow_id=
                WORKFLOW_ID,

            model_id=
                MODEL_ID,

            evaluation={
                "decision_threshold":
                    {
                        "threshold":
                            0.70
                    }
            },
        )
    )

    assert (
        request
        .evaluation
        .decision_threshold
        ==
        MLDecisionThresholdContract(
            threshold=
                0.70
        )
    )


def test_evaluate_request_cannot_supply_selection_context(
) -> None:

    expect_validation_error(
        lambda:
            ModelLabEvaluateRequest
            .model_validate(
                {
                    "workflow_id":
                        WORKFLOW_ID,

                    "model_id":
                        MODEL_ID,

                    "selection_context":
                        {
                            "source":
                                "model_comparison"
                        },
                }
            )
    )


def test_valid_prediction_request(
) -> None:

    request = (
        ModelLabPredictRequest(
            workflow_id=
                WORKFLOW_ID,

            model_id=
                MODEL_ID,

            rows=[
                {
                    "age":
                        42,

                    "segment":
                        "premium",
                },
                {
                    "age":
                        31.5,

                    "segment":
                        "standard",
                },
            ],
        )
    )

    assert (
        len(
            request.rows
        )
        ==
        2
    )


def test_prediction_request_rejects_nested_values(
) -> None:

    expect_validation_error(
        lambda:
            ModelLabPredictRequest(
                workflow_id=
                    WORKFLOW_ID,

                model_id=
                    MODEL_ID,

                rows=[
                    {
                        "age":
                            {
                                "value":
                                    42
                            }
                    }
                ],
            )
    )


def test_prediction_request_rejects_nonfinite_values(
) -> None:

    expect_validation_error(
        lambda:
            ModelLabPredictRequest(
                workflow_id=
                    WORKFLOW_ID,

                model_id=
                    MODEL_ID,

                rows=[
                    {
                        "age":
                            math.nan
                    }
                ],
            )
    )


def test_prediction_request_rejects_whitespace_feature_names(
) -> None:

    expect_validation_error(
        lambda:
            ModelLabPredictRequest(
                workflow_id=
                    WORKFLOW_ID,

                model_id=
                    MODEL_ID,

                rows=[
                    {
                        " age ":
                            42
                    }
                ],
            )
    )


def test_prediction_request_row_limit(
) -> None:

    rows = [
        {
            "age":
                index
        }

        for index
        in range(
            MODEL_LAB_PREDICTION_MAX_ROWS
            +
            1
        )
    ]

    expect_validation_error(
        lambda:
            ModelLabPredictRequest(
                workflow_id=
                    WORKFLOW_ID,

                model_id=
                    MODEL_ID,

                rows=
                    rows,
            )
    )


def test_valid_prediction_response(
) -> None:

    result = (
        ModelLabPredictResponse(
            workflow_id=
                WORKFLOW_ID,

            model_id=
                MODEL_ID,

            problem_type=
                "classification",

            target_column=
                "churned",

            prediction_count=
                3,

            predictions=[
                "yes",
                "no",
                "yes",
            ],
        )
    )

    assert (
        result.prediction_count
        ==
        3
    )

    assert (
        result.method
        ==
        "trusted_native_predict"
    )


def test_prediction_count_is_bound(
) -> None:

    expect_validation_error(
        lambda:
            ModelLabPredictResponse(
                workflow_id=
                    WORKFLOW_ID,

                model_id=
                    MODEL_ID,

                problem_type=
                    "regression",

                target_column=
                    "revenue",

                prediction_count=
                    2,

                predictions=[
                    100.0
                ],
            )
    )


def test_prediction_response_rejects_nonfinite_values(
) -> None:

    expect_validation_error(
        lambda:
            ModelLabPredictResponse(
                workflow_id=
                    WORKFLOW_ID,

                model_id=
                    MODEL_ID,

                problem_type=
                    "regression",

                target_column=
                    "revenue",

                prediction_count=
                    1,

                predictions=[
                    math.inf
                ],
            )
    )


def test_structured_error_contract(
) -> None:

    detail = (
        ModelLabAPIErrorDetail(
            error=
                "model_not_found",

            message=
                "Model was not found.",

            workflow_id=
                WORKFLOW_ID,

            model_id=
                MODEL_ID,
        )
    )

    assert (
        detail.api_version
        ==
        "model_lab_api_v0.1"
    )

    assert (
        detail.retryable
        is False
    )


def test_public_contracts_are_strict(
) -> None:

    expect_validation_error(
        lambda:
            ModelLabPredictRequest
            .model_validate(
                {
                    "workflow_id":
                        WORKFLOW_ID,

                    "model_id":
                        MODEL_ID,

                    "rows":
                        [
                            {
                                "age":
                                    42
                            }
                        ],

                    "model_path":
                        "unsafe.joblib",
                }
            )
    )


def test_rule_version(
) -> None:

    assert (
        MODEL_LAB_API_CONTRACT_RULE_VERSION
        ==
        "model_lab_api_contract_v0.1"
    )


def main(
) -> None:

    print(
        "=== DATALENS MODEL LAB API CONTRACTS v0.1 ==="
    )

    tests = [
        (
            "Valid privacy-minimal model card",
            test_valid_model_card,
        ),
        (
            "Legacy model card without provenance",
            test_legacy_model_card_without_provenance,
        ),
        (
            "Partial provenance fails closed",
            test_partial_provenance_fails_closed,
        ),
        (
            "Categorical features remain feature subset",
            test_categorical_features_must_be_features,
        ),
        (
            "Model card excludes internal artifact fields",
            test_model_card_excludes_internal_artifact_fields,
        ),
        (
            "Valid expanded model detail",
            test_valid_model_detail,
        ),
        (
            "Valid workflow model list",
            test_valid_model_list,
        ),
        (
            "Model list count bound",
            test_model_list_count_is_bound,
        ),
        (
            "Model list workflow bound",
            test_model_list_workflow_is_bound,
        ),
        (
            "Model list duplicate ids blocked",
            test_model_list_rejects_duplicate_ids,
        ),
        (
            "Evaluate request defaults",
            test_evaluate_request_defaults,
        ),
        (
            "Evaluate request accepts explicit threshold",
            test_evaluate_request_accepts_explicit_threshold,
        ),
        (
            "Public evaluate request cannot supply selection context",
            test_evaluate_request_cannot_supply_selection_context,
        ),
        (
            "Valid bounded prediction request",
            test_valid_prediction_request,
        ),
        (
            "Prediction nested values blocked",
            test_prediction_request_rejects_nested_values,
        ),
        (
            "Prediction non-finite values blocked",
            test_prediction_request_rejects_nonfinite_values,
        ),
        (
            "Prediction whitespace feature names blocked",
            test_prediction_request_rejects_whitespace_feature_names,
        ),
        (
            "Prediction row limit enforced",
            test_prediction_request_row_limit,
        ),
        (
            "Valid native prediction response",
            test_valid_prediction_response,
        ),
        (
            "Prediction count bound",
            test_prediction_count_is_bound,
        ),
        (
            "Prediction non-finite response blocked",
            test_prediction_response_rejects_nonfinite_values,
        ),
        (
            "Structured API error contract",
            test_structured_error_contract,
        ),
        (
            "Public contracts reject extra authority",
            test_public_contracts_are_strict,
        ),
        (
            "Model Lab API contract rule version",
            test_rule_version,
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
        "PASS - Model Lab API Contracts v0.1"
    )


if __name__ == "__main__":
    main()
