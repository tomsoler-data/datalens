from __future__ import annotations

from app.evaluation.schemas import (
    BinaryClassificationMetrics,
)


# ============================================================
# SAFE DIVISION
# ============================================================

def safe_ratio(
    numerator: int | float,
    denominator: int | float,
) -> float | None:
    if (
        denominator
        ==
        0
    ):
        return None


    return float(
        numerator
        /
        denominator
    )


# ============================================================
# BINARY CLASSIFICATION METRICS
# ============================================================

def compute_binary_classification_metrics(
    *,
    expected: list[
        bool
    ],
    predicted: list[
        bool
    ],
) -> BinaryClassificationMetrics:
    if (
        len(
            expected
        )
        !=
        len(
            predicted
        )
    ):
        raise ValueError(
            "expected and predicted must have the same length."
        )


    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0


    for truth, prediction in zip(
        expected,
        predicted,
        strict=True,
    ):
        if (
            truth
            and
            prediction
        ):
            true_positive += 1

        elif (
            not truth
            and
            prediction
        ):
            false_positive += 1

        elif (
            not truth
            and
            not prediction
        ):
            true_negative += 1

        else:
            false_negative += 1


    sample_count = len(
        expected
    )


    accuracy = (
        (
            true_positive
            +
            true_negative
        )
        /
        sample_count
        if sample_count
        else 0.0
    )


    precision = safe_ratio(
        true_positive,
        (
            true_positive
            +
            false_positive
        ),
    )


    recall = safe_ratio(
        true_positive,
        (
            true_positive
            +
            false_negative
        ),
    )


    specificity = safe_ratio(
        true_negative,
        (
            true_negative
            +
            false_positive
        ),
    )


    false_positive_rate = safe_ratio(
        false_positive,
        (
            false_positive
            +
            true_negative
        ),
    )


    false_negative_rate = safe_ratio(
        false_negative,
        (
            false_negative
            +
            true_positive
        ),
    )


    if (
        precision is None
        or
        recall is None
    ):
        f1 = None

    elif (
        precision
        +
        recall
        ==
        0
    ):
        f1 = 0.0

    else:
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


    return BinaryClassificationMetrics(
        sample_count=
            sample_count,

        true_positive=
            true_positive,

        false_positive=
            false_positive,

        true_negative=
            true_negative,

        false_negative=
            false_negative,

        accuracy=
            round(
                accuracy,
                6,
            ),

        precision=(
            round(
                precision,
                6,
            )
            if precision is not None
            else None
        ),

        recall=(
            round(
                recall,
                6,
            )
            if recall is not None
            else None
        ),

        specificity=(
            round(
                specificity,
                6,
            )
            if specificity is not None
            else None
        ),

        f1=(
            round(
                f1,
                6,
            )
            if f1 is not None
            else None
        ),

        false_positive_rate=(
            round(
                false_positive_rate,
                6,
            )
            if false_positive_rate
            is not None
            else None
        ),

        false_negative_rate=(
            round(
                false_negative_rate,
                6,
            )
            if false_negative_rate
            is not None
            else None
        ),
    )
