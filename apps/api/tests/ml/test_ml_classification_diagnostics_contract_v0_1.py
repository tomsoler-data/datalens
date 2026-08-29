from __future__ import annotations


import math


from copy import (
    deepcopy,
)


from pydantic import (
    ValidationError,
)


from app.ml.classification_diagnostics import (
    ML_CLASSIFICATION_DIAGNOSTICS_RULE_VERSION,
    MLClassificationClassDiagnostics,
    MLClassificationDiagnosticsContract,
    MLClassificationDiagnosticsResult,
    MLClassificationMetricAverage,
)


# ============================================================
# HELPERS
# ============================================================


def _safe_ratio(
    numerator: int,
    denominator: int,
) -> float:

    if denominator <= 0:
        return 0.0


    return float(
        numerator
        /
        denominator
    )


def _f1(
    *,
    precision: float,
    recall: float,
) -> float:

    denominator = (
        precision
        +
        recall
    )


    if denominator <= 0.0:
        return 0.0


    return float(
        2.0
        *
        precision
        *
        recall
        /
        denominator
    )


def _class_result(
    *,
    class_label: str,
    support: int,
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
) -> MLClassificationClassDiagnostics:

    precision = (
        _safe_ratio(
            true_positive,
            (
                true_positive
                +
                false_positive
            ),
        )
    )


    recall = (
        _safe_ratio(
            true_positive,
            (
                true_positive
                +
                false_negative
            ),
        )
    )


    return (
        MLClassificationClassDiagnostics(
            class_label=
                class_label,

            precision=
                precision,

            recall=
                recall,

            f1=(
                _f1(
                    precision=
                        precision,

                    recall=
                        recall,
                )
            ),

            support=
                support,

            true_positive=
                true_positive,

            false_positive=
                false_positive,

            false_negative=
                false_negative,

            true_negative=
                true_negative,
        )
    )


def valid_result(
) -> MLClassificationDiagnosticsResult:

    # --------------------------------------------------------
    # rows=true / columns=predicted
    #
    #                 predicted
    #               neg      pos
    # true neg       8         2
    # true pos       1         9
    # --------------------------------------------------------

    negative = (
        _class_result(
            class_label=
                "negative",

            support=
                10,

            true_positive=
                8,

            false_positive=
                1,

            false_negative=
                2,

            true_negative=
                9,
        )
    )


    positive = (
        _class_result(
            class_label=
                "positive",

            support=
                10,

            true_positive=
                9,

            false_positive=
                2,

            false_negative=
                1,

            true_negative=
                8,
        )
    )


    per_class = [
        negative,
        positive,
    ]


    macro = (
        MLClassificationMetricAverage(
            precision=(
                (
                    negative.precision
                    +
                    positive.precision
                )
                /
                2
            ),

            recall=(
                (
                    negative.recall
                    +
                    positive.recall
                )
                /
                2
            ),

            f1=(
                (
                    negative.f1
                    +
                    positive.f1
                )
                /
                2
            ),
        )
    )


    weighted = (
        MLClassificationMetricAverage(
            precision=(
                (
                    negative.precision
                    *
                    negative.support
                    +
                    positive.precision
                    *
                    positive.support
                )
                /
                20
            ),

            recall=(
                (
                    negative.recall
                    *
                    negative.support
                    +
                    positive.recall
                    *
                    positive.support
                )
                /
                20
            ),

            f1=(
                (
                    negative.f1
                    *
                    negative.support
                    +
                    positive.f1
                    *
                    positive.support
                )
                /
                20
            ),
        )
    )


    return (
        MLClassificationDiagnosticsResult(
            workflow_id=
                "prep:test-workflow",

            dataset_id=
                "dataset:classification",

            model_id=
                "model:test-classifier",

            experiment_id=(
                "experiment:"
                +
                "a"
                *
                32
            ),

            problem_type=
                "classification",

            target_column=
                "churned",

            estimator_key=
                "logistic_regression",

            preparation_session_revision=
                7,

            training_contract_sha256=(
                "b"
                *
                64
            ),

            evaluation_rows=
                20,

            class_count=
                2,

            class_labels=[
                "negative",
                "positive",
            ],

            confusion_matrix=[
                [
                    8,
                    2,
                ],
                [
                    1,
                    9,
                ],
            ],

            per_class=
                per_class,

            accuracy=
                17.0
                /
                20.0,

            balanced_accuracy=(
                (
                    negative.recall
                    +
                    positive.recall
                )
                /
                2
            ),

            macro_average=
                macro,

            weighted_average=
                weighted,
        )
    )


def _invalid_payload(
):
    return deepcopy(
        valid_result()
        .model_dump(
            mode="json"
        )
    )


# ============================================================
# CONTRACT
# ============================================================


def test_contract_defaults_and_authority(
) -> None:

    contract = (
        MLClassificationDiagnosticsContract()
    )


    assert (
        contract.method
        ==
        "holdout_classification_diagnostics"
    )


    assert (
        contract.label_order_policy
        ==
        "estimator_classes"
    )


    assert (
        contract.zero_division_policy
        ==
        "zero"
    )


    payload = (
        contract.model_dump(
            mode="json"
        )
    )


    forbidden = {
        "workflow_id",
        "model_id",
        "class_labels",
        "predictions",
        "y_true",
        "y_pred",
        "confusion_matrix",
        "model_bytes",
        "model_path",
    }


    assert (
        forbidden.isdisjoint(
            payload
        )
    )


def test_contract_is_strict_and_frozen(
) -> None:

    try:
        MLClassificationDiagnosticsContract.model_validate(
            {
                "method":
                    "holdout_classification_diagnostics",

                "caller_labels": [
                    "no",
                    "yes",
                ],
            }
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "Unknown caller-controlled diagnostic "
                "fields must fail closed."
            )
        )


    try:
        MLClassificationDiagnosticsContract.model_validate(
            {
                "label_order_policy":
                    "caller_order",
            }
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "Caller-controlled class order "
                "must fail closed."
            )
        )


    contract = (
        MLClassificationDiagnosticsContract()
    )


    try:
        contract.method = "changed"

    except Exception:
        pass

    else:
        raise AssertionError(
            "Diagnostics contract must be frozen."
        )


# ============================================================
# VALID RESULT
# ============================================================


def test_valid_binary_result(
) -> None:

    result = (
        valid_result()
    )


    assert (
        result.problem_type
        ==
        "classification"
    )


    assert (
        result.evaluation_rows
        ==
        20
    )


    assert (
        result.class_count
        ==
        2
    )


    assert (
        result.class_labels
        ==
        [
            "negative",
            "positive",
        ]
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
        result.per_class[
            0
        ].false_negative
        ==
        2
    )


    assert (
        result.per_class[
            1
        ].false_positive
        ==
        2
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


# ============================================================
# MATRIX STRUCTURE
# ============================================================


def test_confusion_matrix_shape_fails_closed(
) -> None:

    payload = (
        _invalid_payload()
    )


    payload[
        "confusion_matrix"
    ] = [
        [
            8,
            2,
        ],
        [
            1,
        ],
    ]


    try:
        MLClassificationDiagnosticsResult.model_validate(
            payload
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "Non-square confusion matrix "
                "must fail closed."
            )
        )


def test_confusion_matrix_cell_type_fails_closed(
) -> None:

    payload = (
        _invalid_payload()
    )


    payload[
        "confusion_matrix"
    ][
        0
    ][
        0
    ] = True


    try:
        MLClassificationDiagnosticsResult.model_validate(
            payload
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "Boolean confusion-matrix cells "
                "must fail closed."
            )
        )


def test_evaluation_rows_must_match_matrix_total(
) -> None:

    payload = (
        _invalid_payload()
    )


    payload[
        "evaluation_rows"
    ] = 21


    try:
        MLClassificationDiagnosticsResult.model_validate(
            payload
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "evaluation_rows must match "
                "confusion-matrix total."
            )
        )


# ============================================================
# CLASS AUTHORITY
# ============================================================


def test_duplicate_class_labels_fail_closed(
) -> None:

    payload = (
        _invalid_payload()
    )


    payload[
        "class_labels"
    ] = [
        "negative",
        "negative",
    ]


    try:
        MLClassificationDiagnosticsResult.model_validate(
            payload
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "Duplicate class labels "
                "must fail closed."
            )
        )


def test_per_class_order_must_match_matrix_order(
) -> None:

    payload = (
        _invalid_payload()
    )


    payload[
        "per_class"
    ] = list(
        reversed(
            payload[
                "per_class"
            ]
        )
    )


    try:
        MLClassificationDiagnosticsResult.model_validate(
            payload
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "Per-class order must exactly "
                "match class_labels."
            )
        )


# ============================================================
# COUNT CONSISTENCY
# ============================================================


def test_support_mismatch_fails_closed(
) -> None:

    payload = (
        _invalid_payload()
    )


    payload[
        "per_class"
    ][
        0
    ][
        "support"
    ] = 11


    try:
        MLClassificationDiagnosticsResult.model_validate(
            payload
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "Per-class support must be derived "
                "from the confusion matrix."
            )
        )


def test_fp_fn_count_mismatch_fails_closed(
) -> None:

    payload = (
        _invalid_payload()
    )


    payload[
        "per_class"
    ][
        0
    ][
        "false_positive"
    ] = 2


    try:
        MLClassificationDiagnosticsResult.model_validate(
            payload
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "Per-class FP/FN counts must be "
                "derived from the confusion matrix."
            )
        )


# ============================================================
# METRIC CONSISTENCY
# ============================================================


def test_per_class_metric_mismatch_fails_closed(
) -> None:

    payload = (
        _invalid_payload()
    )


    payload[
        "per_class"
    ][
        0
    ][
        "precision"
    ] = 0.5


    try:
        MLClassificationDiagnosticsResult.model_validate(
            payload
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "Per-class metrics must match "
                "TP / FP / FN counts."
            )
        )


def test_global_metric_mismatch_fails_closed(
) -> None:

    payload = (
        _invalid_payload()
    )


    payload[
        "accuracy"
    ] = 0.5


    try:
        MLClassificationDiagnosticsResult.model_validate(
            payload
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "Global accuracy must match "
                "the confusion matrix."
            )
        )


def test_average_metric_mismatch_fails_closed(
) -> None:

    payload = (
        _invalid_payload()
    )


    payload[
        "macro_average"
    ][
        "f1"
    ] = 0.5


    try:
        MLClassificationDiagnosticsResult.model_validate(
            payload
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "Macro/weighted metrics must be "
                "derived from per-class diagnostics."
            )
        )


# ============================================================
# IDENTIFIERS + FINGERPRINT
# ============================================================


def test_provenance_identifiers_are_validated(
) -> None:

    payload = (
        _invalid_payload()
    )


    payload[
        "experiment_id"
    ] = "experiment:not-a-valid-id"


    try:
        MLClassificationDiagnosticsResult.model_validate(
            payload
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "Invalid experiment identity "
                "must fail closed."
            )
        )


    payload = (
        _invalid_payload()
    )


    payload[
        "training_contract_sha256"
    ] = "not-a-sha"


    try:
        MLClassificationDiagnosticsResult.model_validate(
            payload
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "Invalid Training Contract SHA "
                "must fail closed."
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
        "raw_rows",
        "rows",
        "predictions",
        "holdout_predictions",
        "probabilities",
        "decision_scores",
        "y_true",
        "y_pred",
        "x_train",
        "x_test",
        "y_train",
        "y_test",
        "model_bytes",
        "model_path",
        "estimator",
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
        ML_CLASSIFICATION_DIAGNOSTICS_RULE_VERSION
        ==
        "ml_classification_diagnostics_v0.1"
    )


    assert (
        MLClassificationDiagnosticsContract()
        .rule_version
        ==
        ML_CLASSIFICATION_DIAGNOSTICS_RULE_VERSION
    )


    assert (
        valid_result()
        .rule_version
        ==
        ML_CLASSIFICATION_DIAGNOSTICS_RULE_VERSION
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML CLASSIFICATION DIAGNOSTICS CONTRACT v0.1 ==="
    )


    tests = [
        (
            "Contract defaults and authority",
            test_contract_defaults_and_authority,
        ),
        (
            "Strict frozen server-owned contract",
            test_contract_is_strict_and_frozen,
        ),
        (
            "Valid binary diagnostic result",
            test_valid_binary_result,
        ),
        (
            "Confusion matrix shape fail-closed",
            test_confusion_matrix_shape_fails_closed,
        ),
        (
            "Confusion matrix integer cells",
            test_confusion_matrix_cell_type_fails_closed,
        ),
        (
            "Evaluation rows match matrix total",
            test_evaluation_rows_must_match_matrix_total,
        ),
        (
            "Unique class labels",
            test_duplicate_class_labels_fail_closed,
        ),
        (
            "Per-class order matches matrix",
            test_per_class_order_must_match_matrix_order,
        ),
        (
            "Support derived from matrix",
            test_support_mismatch_fails_closed,
        ),
        (
            "FP/FN counts derived from matrix",
            test_fp_fn_count_mismatch_fails_closed,
        ),
        (
            "Per-class metrics derived from counts",
            test_per_class_metric_mismatch_fails_closed,
        ),
        (
            "Global metrics derived from matrix",
            test_global_metric_mismatch_fails_closed,
        ),
        (
            "Macro/weighted metrics derived",
            test_average_metric_mismatch_fails_closed,
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
        "PASS - ML Classification Diagnostics Contract v0.1"
    )


if __name__ == "__main__":
    main()
