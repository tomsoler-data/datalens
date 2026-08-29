from __future__ import annotations


import math
import re


from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# VERSION
# ============================================================


ML_DECISION_THRESHOLD_RULE_VERSION = (
    "ml_decision_threshold_v0.1"
)


# ============================================================
# TYPES / FIXED POLICIES
# ============================================================


MLDecisionThresholdMethod = Literal[
    "holdout_binary_decision_threshold",
]


MLDecisionThresholdScoreSource = Literal[
    "predict_proba",
]


MLDecisionThresholdPositiveClassPolicy = Literal[
    "estimator_classes_index_1",
]


MLDecisionThresholdComparisonOperator = Literal[
    "greater_than_or_equal",
]


MLDecisionThresholdSelectionPolicy = Literal[
    "evaluate_requested_threshold_only",
]


MLDecisionThresholdZeroDivisionPolicy = Literal[
    "zero",
]


# ============================================================
# VALIDATION
# ============================================================


SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)


EXPERIMENT_ID_PATTERN = re.compile(
    r"^experiment:[0-9a-f]{32}$"
)


def _required_text(
    value: object,
    *,
    field_name: str,
) -> str:

    normalized = str(
        value
        if value is not None
        else ""
    ).strip()


    if not normalized:
        raise ValueError(
            (
                f"{field_name} "
                "cannot be empty."
            )
        )


    return normalized


def _threshold_value(
    value: object,
) -> float:

    if (
        isinstance(
            value,
            bool,
        )
        or
        not isinstance(
            value,
            (
                int,
                float,
            ),
        )
    ):
        raise ValueError(
            "threshold must be a numeric value"
        )


    normalized = float(
        value
    )


    if not math.isfinite(
        normalized
    ):
        raise ValueError(
            "threshold must be finite"
        )


    return normalized


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


def _assert_close(
    *,
    actual: float,
    expected: float,
    field_name: str,
) -> None:

    if not math.isclose(
        float(
            actual
        ),
        float(
            expected
        ),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            (
                f"{field_name} does not match "
                "the deterministic threshold "
                "confusion-matrix derivation."
            )
        )


# ============================================================
# REQUEST CONTRACT
# ============================================================


class MLDecisionThresholdContract(
    BaseModel
):
    """
    Server-validatable Decision Threshold v0.1 request.

    The caller is allowed to choose exactly one thing:
        threshold.

    Everything else is frozen server policy.

    This contract deliberately does NOT contain:
    - workflow_id;
    - model_id;
    - positive / negative labels;
    - probabilities;
    - predictions;
    - confusion-matrix values;
    - target values;
    - model bytes;
    - filesystem paths.

    v0.1 evaluates one explicitly requested threshold against the
    original deterministic holdout.

    It does NOT search for, optimize, rank, recommend or persist
    a threshold.

    This is important because selecting the best threshold from
    the final holdout would turn the holdout into tuning data.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


    threshold: float = Field(
        ge=0.0,
        le=1.0,
    )


    method: Literal[
        "holdout_binary_decision_threshold"
    ] = "holdout_binary_decision_threshold"


    score_source: Literal[
        "predict_proba"
    ] = "predict_proba"


    positive_class_policy: Literal[
        "estimator_classes_index_1"
    ] = "estimator_classes_index_1"


    comparison_operator: Literal[
        "greater_than_or_equal"
    ] = "greater_than_or_equal"


    threshold_selection_policy: Literal[
        "evaluate_requested_threshold_only"
    ] = "evaluate_requested_threshold_only"


    zero_division_policy: Literal[
        "zero"
    ] = "zero"


    rule_version: Literal[
        "ml_decision_threshold_v0.1"
    ] = ML_DECISION_THRESHOLD_RULE_VERSION


    @field_validator(
        "threshold",
        mode="before",
    )
    @classmethod
    def validate_threshold(
        cls,
        value: object,
    ) -> float:

        return (
            _threshold_value(
                value
            )
        )


# ============================================================
# RESULT
# ============================================================


class MLDecisionThresholdResult(
    BaseModel
):
    """
    Privacy-minimal binary threshold evaluation for one trusted
    persisted classification Model Artifact.

    Matrix convention
    -----------------

    Class order:
        [negative_class_label, positive_class_label]

    Rows:
        true class.

    Columns:
        thresholded predicted class.

    Therefore:

        [
            [TN, FP],
            [FN, TP],
        ]

    Positive classification policy
    ------------------------------

        positive class = fitted estimator classes_[1]

        positive prediction when:

            P(positive | x) >= threshold

    The result deliberately contains no individual:
    - probabilities;
    - predictions;
    - y_true values;
    - feature rows.

    It also contains no model bytes, model path or estimator.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


    workflow_id: str = Field(
        min_length=1,
    )


    dataset_id: str = Field(
        min_length=1,
    )


    model_id: str = Field(
        min_length=1,
    )


    experiment_id: str = Field(
        min_length=1,
    )


    problem_type: Literal[
        "classification"
    ] = "classification"


    target_column: str = Field(
        min_length=1,
    )


    estimator_key: str = Field(
        min_length=1,
    )


    preparation_session_revision: int = Field(
        ge=0,
        strict=True,
    )


    training_contract_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


    evaluation_rows: int = Field(
        gt=0,
        strict=True,
    )


    threshold: float = Field(
        ge=0.0,
        le=1.0,
    )


    negative_class_label: str = Field(
        min_length=1,
    )


    positive_class_label: str = Field(
        min_length=1,
    )


    confusion_matrix: list[
        list[
            int
        ]
    ] = Field(
        min_length=2,
        max_length=2,
    )


    negative_support: int = Field(
        ge=0,
        strict=True,
    )


    positive_support: int = Field(
        ge=0,
        strict=True,
    )


    true_negative: int = Field(
        ge=0,
        strict=True,
    )


    false_positive: int = Field(
        ge=0,
        strict=True,
    )


    false_negative: int = Field(
        ge=0,
        strict=True,
    )


    true_positive: int = Field(
        ge=0,
        strict=True,
    )


    precision: float = Field(
        ge=0.0,
        le=1.0,
    )


    recall: float = Field(
        ge=0.0,
        le=1.0,
    )


    f1: float = Field(
        ge=0.0,
        le=1.0,
    )


    specificity: float = Field(
        ge=0.0,
        le=1.0,
    )


    accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )


    balanced_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )


    positive_prediction_rate: float = Field(
        ge=0.0,
        le=1.0,
    )


    method: Literal[
        "holdout_binary_decision_threshold"
    ] = "holdout_binary_decision_threshold"


    score_source: Literal[
        "predict_proba"
    ] = "predict_proba"


    positive_class_policy: Literal[
        "estimator_classes_index_1"
    ] = "estimator_classes_index_1"


    comparison_operator: Literal[
        "greater_than_or_equal"
    ] = "greater_than_or_equal"


    threshold_selection_policy: Literal[
        "evaluate_requested_threshold_only"
    ] = "evaluate_requested_threshold_only"


    zero_division_policy: Literal[
        "zero"
    ] = "zero"


    rule_version: Literal[
        "ml_decision_threshold_v0.1"
    ] = ML_DECISION_THRESHOLD_RULE_VERSION


    # ========================================================
    # TEXT
    # ========================================================


    @field_validator(
        "workflow_id",
        "dataset_id",
        "model_id",
        "target_column",
        "estimator_key",
        "negative_class_label",
        "positive_class_label",
        mode="before",
    )
    @classmethod
    def normalize_required_text(
        cls,
        value: object,
        info,
    ) -> str:

        return (
            _required_text(
                value,
                field_name=
                    info.field_name,
            )
        )


    # ========================================================
    # THRESHOLD
    # ========================================================


    @field_validator(
        "threshold",
        mode="before",
    )
    @classmethod
    def validate_threshold(
        cls,
        value: object,
    ) -> float:

        return (
            _threshold_value(
                value
            )
        )


    # ========================================================
    # PROVENANCE
    # ========================================================


    @field_validator(
        "experiment_id",
        mode="before",
    )
    @classmethod
    def validate_experiment_id(
        cls,
        value: object,
    ) -> str:

        normalized = str(
            value
            if value is not None
            else ""
        ).strip().lower()


        if not (
            EXPERIMENT_ID_PATTERN
            .fullmatch(
                normalized
            )
        ):
            raise ValueError(
                (
                    "experiment_id must match "
                    "experiment:<32 lowercase hex chars>"
                )
            )


        return normalized


    @field_validator(
        "training_contract_sha256",
        mode="before",
    )
    @classmethod
    def validate_training_contract_sha256(
        cls,
        value: object,
    ) -> str:

        normalized = str(
            value
            if value is not None
            else ""
        ).strip().lower()


        if not (
            SHA256_PATTERN
            .fullmatch(
                normalized
            )
        ):
            raise ValueError(
                (
                    "training_contract_sha256 must be "
                    "64 lowercase hexadecimal characters"
                )
            )


        return normalized


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================


    @field_validator(
        "confusion_matrix",
        mode="before",
    )
    @classmethod
    def validate_confusion_matrix_shape_and_cells(
        cls,
        value: object,
    ) -> list[
        list[
            int
        ]
    ]:

        if not isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            raise ValueError(
                "confusion_matrix must be a 2x2 matrix"
            )


        if len(
            value
        ) != 2:
            raise ValueError(
                "confusion_matrix must contain exactly two rows"
            )


        normalized: list[
            list[
                int
            ]
        ] = []


        for row in value:

            if not isinstance(
                row,
                (
                    list,
                    tuple,
                ),
            ):
                raise ValueError(
                    "confusion_matrix rows must be sequences"
                )


            if len(
                row
            ) != 2:
                raise ValueError(
                    (
                        "confusion_matrix rows must contain "
                        "exactly two cells"
                    )
                )


            normalized_row: list[
                int
            ] = []


            for cell in row:

                if type(
                    cell
                ) is not int:
                    raise ValueError(
                        (
                            "confusion_matrix cells must be "
                            "strict integers"
                        )
                    )


                if cell < 0:
                    raise ValueError(
                        (
                            "confusion_matrix cells cannot "
                            "be negative"
                        )
                    )


                normalized_row.append(
                    cell
                )


            normalized.append(
                normalized_row
            )


        return normalized


    # ========================================================
    # COMPLETE DERIVATION
    # ========================================================


    @model_validator(
        mode="after"
    )
    def validate_deterministic_derivation(
        self,
    ) -> "MLDecisionThresholdResult":

        if (
            self.negative_class_label
            ==
            self.positive_class_label
        ):
            raise ValueError(
                (
                    "negative_class_label and "
                    "positive_class_label must be distinct"
                )
            )


        matrix = (
            self.confusion_matrix
        )


        true_negative = int(
            matrix[
                0
            ][
                0
            ]
        )


        false_positive = int(
            matrix[
                0
            ][
                1
            ]
        )


        false_negative = int(
            matrix[
                1
            ][
                0
            ]
        )


        true_positive = int(
            matrix[
                1
            ][
                1
            ]
        )


        matrix_total = (
            true_negative
            +
            false_positive
            +
            false_negative
            +
            true_positive
        )


        if (
            matrix_total
            !=
            self.evaluation_rows
        ):
            raise ValueError(
                (
                    "confusion_matrix total must equal "
                    "evaluation_rows"
                )
            )


        expected_negative_support = (
            true_negative
            +
            false_positive
        )


        expected_positive_support = (
            false_negative
            +
            true_positive
        )


        integer_expectations = {
            "negative_support":
                expected_negative_support,

            "positive_support":
                expected_positive_support,

            "true_negative":
                true_negative,

            "false_positive":
                false_positive,

            "false_negative":
                false_negative,

            "true_positive":
                true_positive,
        }


        for (
            field_name,
            expected,
        ) in integer_expectations.items():

            actual = int(
                getattr(
                    self,
                    field_name,
                )
            )


            if (
                actual
                !=
                expected
            ):
                raise ValueError(
                    (
                        f"{field_name} does not match "
                        "the confusion matrix"
                    )
                )


        expected_precision = (
            _safe_ratio(
                true_positive,
                (
                    true_positive
                    +
                    false_positive
                ),
            )
        )


        expected_recall = (
            _safe_ratio(
                true_positive,
                (
                    true_positive
                    +
                    false_negative
                ),
            )
        )


        expected_f1 = (
            _f1(
                precision=
                    expected_precision,

                recall=
                    expected_recall,
            )
        )


        expected_specificity = (
            _safe_ratio(
                true_negative,
                (
                    true_negative
                    +
                    false_positive
                ),
            )
        )


        expected_accuracy = (
            _safe_ratio(
                (
                    true_positive
                    +
                    true_negative
                ),
                self.evaluation_rows,
            )
        )


        supported_recalls: list[
            float
        ] = []


        if (
            expected_negative_support
            >
            0
        ):
            supported_recalls.append(
                expected_specificity
            )


        if (
            expected_positive_support
            >
            0
        ):
            supported_recalls.append(
                expected_recall
            )


        if not supported_recalls:
            raise ValueError(
                (
                    "Decision Threshold evaluation "
                    "must contain at least one supported class"
                )
            )


        expected_balanced_accuracy = float(
            sum(
                supported_recalls
            )
            /
            len(
                supported_recalls
            )
        )


        expected_positive_prediction_rate = (
            _safe_ratio(
                (
                    true_positive
                    +
                    false_positive
                ),
                self.evaluation_rows,
            )
        )


        metric_expectations = {
            "precision":
                expected_precision,

            "recall":
                expected_recall,

            "f1":
                expected_f1,

            "specificity":
                expected_specificity,

            "accuracy":
                expected_accuracy,

            "balanced_accuracy":
                expected_balanced_accuracy,

            "positive_prediction_rate":
                expected_positive_prediction_rate,
        }


        for (
            field_name,
            expected,
        ) in metric_expectations.items():

            _assert_close(
                actual=
                    float(
                        getattr(
                            self,
                            field_name,
                        )
                    ),

                expected=
                    expected,

                field_name=
                    field_name,
            )


        return self
