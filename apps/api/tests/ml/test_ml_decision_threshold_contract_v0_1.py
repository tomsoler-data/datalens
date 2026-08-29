from __future__ import annotations


import math


from pydantic import (
    ValidationError,
)


from app.ml.decision_threshold import (
    ML_DECISION_THRESHOLD_RULE_VERSION,
    MLDecisionThresholdContract,
    MLDecisionThresholdResult,
)


# ============================================================
# CONSTANTS
# ============================================================


WORKFLOW_ID = (
    "prep:decision-threshold"
)


DATASET_ID = (
    "dataset:classification"
)


MODEL_ID = (
    "model:decision-threshold"
)


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


# ============================================================
# HELPERS
# ============================================================


def expect_validation_error(
    factory,
) -> None:

    try:

        factory()

    except ValidationError:
        return


    raise AssertionError(
        "Expected Pydantic ValidationError."
    )


def valid_result(
    **overrides,
) -> MLDecisionThresholdResult:

    true_negative = 8
    false_positive = 2
    false_negative = 1
    true_positive = 9


    precision = (
        true_positive
        /
        (
            true_positive
            +
            false_positive
        )
    )


    recall = (
        true_positive
        /
        (
            true_positive
            +
            false_negative
        )
    )


    f1 = (
        2.0
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
        )
    )


    specificity = (
        true_negative
        /
        (
            true_negative
            +
            false_positive
        )
    )


    payload = {
        "workflow_id":
            WORKFLOW_ID,

        "dataset_id":
            DATASET_ID,

        "model_id":
            MODEL_ID,

        "experiment_id":
            EXPERIMENT_ID,

        "problem_type":
            "classification",

        "target_column":
            "target",

        "estimator_key":
            "logistic_regression",

        "preparation_session_revision":
            7,

        "training_contract_sha256":
            TRAINING_SHA,

        "evaluation_rows":
            20,

        "threshold":
            0.50,

        "negative_class_label":
            "negative",

        "positive_class_label":
            "positive",

        "confusion_matrix": [
            [
                true_negative,
                false_positive,
            ],
            [
                false_negative,
                true_positive,
            ],
        ],

        "negative_support":
            10,

        "positive_support":
            10,

        "true_negative":
            true_negative,

        "false_positive":
            false_positive,

        "false_negative":
            false_negative,

        "true_positive":
            true_positive,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "specificity":
            specificity,

        "accuracy":
            0.85,

        "balanced_accuracy":
            0.85,

        "positive_prediction_rate":
            0.55,
    }


    payload.update(
        overrides
    )


    return (
        MLDecisionThresholdResult(
            **payload
        )
    )


# ============================================================
# CONTRACT
# ============================================================


def test_contract_requires_explicit_threshold(
) -> None:

    expect_validation_error(
        lambda:
            MLDecisionThresholdContract()
    )


def test_contract_defaults_and_authority(
) -> None:

    contract = (
        MLDecisionThresholdContract(
            threshold=
                0.40
        )
    )


    assert (
        contract.threshold
        ==
        0.40
    )


    assert (
        contract.method
        ==
        "holdout_binary_decision_threshold"
    )


    assert (
        contract.score_source
        ==
        "predict_proba"
    )


    assert (
        contract.positive_class_policy
        ==
        "estimator_classes_index_1"
    )


    assert (
        contract.comparison_operator
        ==
        "greater_than_or_equal"
    )


    assert (
        contract.threshold_selection_policy
        ==
        "evaluate_requested_threshold_only"
    )


    assert (
        contract.zero_division_policy
        ==
        "zero"
    )


    assert (
        set(
            contract.model_dump(
                mode="json"
            )
        )
        ==
        {
            "threshold",
            "method",
            "score_source",
            "positive_class_policy",
            "comparison_operator",
            "threshold_selection_policy",
            "zero_division_policy",
            "rule_version",
        }
    )


def test_contract_is_strict_and_frozen(
) -> None:

    expect_validation_error(
        lambda:
            MLDecisionThresholdContract(
                threshold=
                    0.50,

                positive_class_label=
                    "forbidden",
            )
    )


    contract = (
        MLDecisionThresholdContract(
            threshold=
                0.50
        )
    )


    try:

        contract.threshold = 0.30

    except ValidationError:
        return


    raise AssertionError(
        "Decision Threshold contract must be frozen."
    )


def test_threshold_boundaries_are_valid(
) -> None:

    assert (
        MLDecisionThresholdContract(
            threshold=
                0
        )
        .threshold
        ==
        0.0
    )


    assert (
        MLDecisionThresholdContract(
            threshold=
                1
        )
        .threshold
        ==
        1.0
    )


def test_invalid_thresholds_fail_closed(
) -> None:

    invalid_values = [
        -0.0001,
        1.0001,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        False,
        "0.5",
        None,
    ]


    for value in invalid_values:

        expect_validation_error(
            lambda value=value:
                MLDecisionThresholdContract(
                    threshold=
                        value
                )
        )


# ============================================================
# RESULT
# ============================================================


def test_valid_binary_threshold_result(
) -> None:

    result = (
        valid_result()
    )


    assert (
        result.confusion_matrix
        ==
        [
            [
                8,
                2,
            ],
            [
                1,
                9,
            ],
        ]
    )


    assert (
        result.true_negative
        ==
        8
    )


    assert (
        result.false_positive
        ==
        2
    )


    assert (
        result.false_negative
        ==
        1
    )


    assert (
        result.true_positive
        ==
        9
    )


    assert math.isclose(
        result.accuracy,
        0.85,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        result.balanced_accuracy,
        0.85,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        result.positive_prediction_rate,
        0.55,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_binary_class_labels_must_be_distinct(
) -> None:

    expect_validation_error(
        lambda:
            valid_result(
                positive_class_label=
                    "negative"
            )
    )


def test_confusion_matrix_shape_fails_closed(
) -> None:

    expect_validation_error(
        lambda:
            valid_result(
                confusion_matrix=[
                    [
                        8,
                        2,
                    ],
                ]
            )
    )


    expect_validation_error(
        lambda:
            valid_result(
                confusion_matrix=[
                    [
                        8,
                        2,
                        0,
                    ],
                    [
                        1,
                        9,
                        0,
                    ],
                ]
            )
    )


def test_confusion_matrix_cells_are_strict_integers(
) -> None:

    expect_validation_error(
        lambda:
            valid_result(
                confusion_matrix=[
                    [
                        8.0,
                        2,
                    ],
                    [
                        1,
                        9,
                    ],
                ]
            )
    )


    expect_validation_error(
        lambda:
            valid_result(
                confusion_matrix=[
                    [
                        True,
                        2,
                    ],
                    [
                        1,
                        9,
                    ],
                ]
            )
    )


def test_evaluation_rows_match_matrix_total(
) -> None:

    expect_validation_error(
        lambda:
            valid_result(
                evaluation_rows=
                    21
            )
    )


def test_support_is_derived_from_matrix(
) -> None:

    expect_validation_error(
        lambda:
            valid_result(
                negative_support=
                    9
            )
    )


    expect_validation_error(
        lambda:
            valid_result(
                positive_support=
                    11
            )
    )


def test_binary_counts_are_derived_from_matrix(
) -> None:

    for (
        field_name,
        bad_value,
    ) in [
        (
            "true_negative",
            7,
        ),
        (
            "false_positive",
            3,
        ),
        (
            "false_negative",
            2,
        ),
        (
            "true_positive",
            8,
        ),
    ]:

        expect_validation_error(
            lambda field_name=field_name, bad_value=bad_value:
                valid_result(
                    **{
                        field_name:
                            bad_value
                    }
                )
        )


def test_threshold_metrics_are_derived_from_matrix(
) -> None:

    for field_name in [
        "precision",
        "recall",
        "f1",
        "specificity",
        "accuracy",
        "balanced_accuracy",
        "positive_prediction_rate",
    ]:

        expect_validation_error(
            lambda field_name=field_name:
                valid_result(
                    **{
                        field_name:
                            0.123456
                    }
                )
        )


def test_balanced_accuracy_uses_supported_classes_only(
) -> None:

    result = (
        MLDecisionThresholdResult(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

            model_id=
                MODEL_ID,

            experiment_id=
                EXPERIMENT_ID,

            target_column=
                "target",

            estimator_key=
                "logistic_regression",

            preparation_session_revision=
                7,

            training_contract_sha256=
                TRAINING_SHA,

            evaluation_rows=
                10,

            threshold=
                0.40,

            negative_class_label=
                "negative",

            positive_class_label=
                "positive",

            confusion_matrix=[
                [
                    0,
                    0,
                ],
                [
                    2,
                    8,
                ],
            ],

            negative_support=
                0,

            positive_support=
                10,

            true_negative=
                0,

            false_positive=
                0,

            false_negative=
                2,

            true_positive=
                8,

            precision=
                1.0,

            recall=
                0.8,

            f1=(
                8.0
                /
                9.0
            ),

            specificity=
                0.0,

            accuracy=
                0.8,

            balanced_accuracy=
                0.8,

            positive_prediction_rate=
                0.8,
        )
    )


    assert math.isclose(
        result.balanced_accuracy,
        0.8,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_provenance_identifiers_are_validated(
) -> None:

    expect_validation_error(
        lambda:
            valid_result(
                experiment_id=
                    "experiment:not-valid"
            )
    )


    expect_validation_error(
        lambda:
            valid_result(
                training_contract_sha256=
                    "not-a-sha"
            )
    )


# ============================================================
# PRIVACY
# ============================================================


def _all_keys(
    value,
) -> set[
    str
]:

    keys: set[
        str
    ] = set()


    if isinstance(
        value,
        dict,
    ):

        for (
            key,
            nested,
        ) in value.items():

            keys.add(
                str(
                    key
                )
            )


            keys.update(
                _all_keys(
                    nested
                )
            )


    elif isinstance(
        value,
        list,
    ):

        for nested in value:

            keys.update(
                _all_keys(
                    nested
                )
            )


    return keys


def test_result_is_privacy_minimal(
) -> None:

    payload = (
        valid_result()
        .model_dump(
            mode="json"
        )
    )


    forbidden = {
        "rows",
        "raw_rows",
        "predictions",
        "probabilities",
        "positive_probabilities",
        "negative_probabilities",
        "decision_scores",
        "scores",
        "y_true",
        "y_pred",
        "x_train",
        "x_test",
        "y_train",
        "y_test",
        "model_bytes",
        "model_path",
        "estimator",
        "training_contract",
    }


    assert (
        forbidden.isdisjoint(
            _all_keys(
                payload
            )
        )
    )


# ============================================================
# VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_DECISION_THRESHOLD_RULE_VERSION
        ==
        "ml_decision_threshold_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML DECISION THRESHOLD CONTRACT v0.1 ==="
    )


    tests = [
        (
            "Contract requires explicit threshold",
            test_contract_requires_explicit_threshold,
        ),
        (
            "Contract defaults and authority",
            test_contract_defaults_and_authority,
        ),
        (
            "Strict frozen server-owned contract",
            test_contract_is_strict_and_frozen,
        ),
        (
            "Threshold boundaries accepted",
            test_threshold_boundaries_are_valid,
        ),
        (
            "Invalid thresholds fail-closed",
            test_invalid_thresholds_fail_closed,
        ),
        (
            "Valid binary threshold result",
            test_valid_binary_threshold_result,
        ),
        (
            "Binary class labels are distinct",
            test_binary_class_labels_must_be_distinct,
        ),
        (
            "Confusion matrix shape fail-closed",
            test_confusion_matrix_shape_fails_closed,
        ),
        (
            "Confusion matrix integer cells",
            test_confusion_matrix_cells_are_strict_integers,
        ),
        (
            "Evaluation rows match matrix total",
            test_evaluation_rows_match_matrix_total,
        ),
        (
            "Support derived from matrix",
            test_support_is_derived_from_matrix,
        ),
        (
            "TP / FP / FN / TN derived from matrix",
            test_binary_counts_are_derived_from_matrix,
        ),
        (
            "Threshold metrics derived from matrix",
            test_threshold_metrics_are_derived_from_matrix,
        ),
        (
            "Balanced accuracy uses supported classes",
            test_balanced_accuracy_uses_supported_classes_only,
        ),
        (
            "Provenance identifiers validated",
            test_provenance_identifiers_are_validated,
        ),
        (
            "Privacy-minimal result surface",
            test_result_is_privacy_minimal,
        ),
        (
            "Rule version",
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
        "PASS - ML Decision Threshold Contract v0.1"
    )


if __name__ == "__main__":
    main()
