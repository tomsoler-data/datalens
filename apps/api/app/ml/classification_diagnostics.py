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


ML_CLASSIFICATION_DIAGNOSTICS_RULE_VERSION = (
    "ml_classification_diagnostics_v0.1"
)


# ============================================================
# TYPES
# ============================================================


MLClassificationDiagnosticsMethod = Literal[
    "holdout_classification_diagnostics",
]


MLClassificationLabelOrderPolicy = Literal[
    "estimator_classes",
]


MLClassificationZeroDivisionPolicy = Literal[
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


def _expected_f1(
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
                "the deterministic confusion-matrix "
                "derivation."
            )
        )


# ============================================================
# REQUEST CONTRACT
# ============================================================


class MLClassificationDiagnosticsContract(
    BaseModel
):
    """
    Server-validatable Classification Diagnostics v0.1 policy.

    This contract contains configuration only.

    It deliberately does NOT contain:
    - workflow_id;
    - model_id;
    - class labels;
    - predictions;
    - y_true;
    - confusion-matrix values;
    - metric values;
    - model bytes;
    - filesystem paths.

    Those values are derived server-side from one trusted
    persisted classification Model Artifact and its deterministic
    holdout.

    v0.1 policies are intentionally frozen:

    - diagnostics are evaluated on the original holdout only;
    - class ordering comes from the fitted estimator classes_;
    - undefined precision / recall / F1 values resolve to zero.

    Future threshold configuration belongs to the separate
    Decision Threshold milestone and must not enter this contract.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    method: Literal[
        "holdout_classification_diagnostics"
    ] = "holdout_classification_diagnostics"


    label_order_policy: Literal[
        "estimator_classes"
    ] = "estimator_classes"


    zero_division_policy: Literal[
        "zero"
    ] = "zero"


    rule_version: Literal[
        "ml_classification_diagnostics_v0.1"
    ] = ML_CLASSIFICATION_DIAGNOSTICS_RULE_VERSION


# ============================================================
# AVERAGE METRICS
# ============================================================


class MLClassificationMetricAverage(
    BaseModel
):
    """
    One deterministic aggregate precision / recall / F1 surface.

    Both macro and support-weighted averages reuse this shape.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
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


# ============================================================
# PER-CLASS DIAGNOSTICS
# ============================================================


class MLClassificationClassDiagnostics(
    BaseModel
):
    """
    Deterministic one-vs-rest diagnostics for one class.

    class_label is the privacy-minimal textual representation
    of the server-owned fitted estimator class label.

    The executor must fail closed if two distinct raw estimator
    labels would collapse to the same textual representation.

    Counts are derived from the confusion matrix:

        TP = diagonal cell
        FN = row total - TP
        FP = column total - TP
        TN = evaluation_rows - TP - FN - FP

    Metrics follow zero_division=0 semantics.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


    class_label: str = Field(
        min_length=1,
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


    support: int = Field(
        ge=0,
        strict=True,
    )


    true_positive: int = Field(
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


    true_negative: int = Field(
        ge=0,
        strict=True,
    )


    @field_validator(
        "class_label",
        mode="before",
    )
    @classmethod
    def normalize_class_label(
        cls,
        value: object,
    ) -> str:

        return (
            _required_text(
                value,
                field_name=
                    "class_label",
            )
        )


    @model_validator(
        mode="after"
    )
    def validate_metric_count_consistency(
        self,
    ) -> "MLClassificationClassDiagnostics":

        expected_precision = (
            _safe_ratio(
                self.true_positive,
                (
                    self.true_positive
                    +
                    self.false_positive
                ),
            )
        )


        expected_recall = (
            _safe_ratio(
                self.true_positive,
                (
                    self.true_positive
                    +
                    self.false_negative
                ),
            )
        )


        expected_f1 = (
            _expected_f1(
                precision=
                    expected_precision,

                recall=
                    expected_recall,
            )
        )


        _assert_close(
            actual=
                self.precision,

            expected=
                expected_precision,

            field_name=
                "precision",
        )


        _assert_close(
            actual=
                self.recall,

            expected=
                expected_recall,

            field_name=
                "recall",
        )


        _assert_close(
            actual=
                self.f1,

            expected=
                expected_f1,

            field_name=
                "f1",
        )


        return self


# ============================================================
# RESULT
# ============================================================


class MLClassificationDiagnosticsResult(
    BaseModel
):
    """
    Privacy-minimal deterministic diagnostics for one persisted
    trusted classification Model Artifact.

    Authority is bound to:
    - workflow;
    - dataset;
    - model;
    - experiment;
    - target column;
    - estimator;
    - Preparation revision;
    - canonical ML Training Contract SHA-256.

    Confusion matrix convention
    ---------------------------

    Rows:
        true class.

    Columns:
        predicted class.

    Both axes use class_labels in exactly the same order.

    label_order_policy="estimator_classes" means the future
    executor must obtain this order from the trusted fitted
    classifier, never from caller input.

    This result deliberately contains no:
    - raw rows;
    - individual y_true values;
    - individual predictions;
    - probabilities;
    - decision scores;
    - model bytes;
    - model filesystem path;
    - fitted estimator.
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


    class_count: int = Field(
        ge=2,
        strict=True,
    )


    class_labels: list[
        str
    ] = Field(
        min_length=2,
    )


    confusion_matrix: list[
        list[
            int
        ]
    ] = Field(
        min_length=2,
    )


    per_class: list[
        MLClassificationClassDiagnostics
    ] = Field(
        min_length=2,
    )


    accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )


    balanced_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )


    macro_average: (
        MLClassificationMetricAverage
    )


    weighted_average: (
        MLClassificationMetricAverage
    )


    method: Literal[
        "holdout_classification_diagnostics"
    ] = "holdout_classification_diagnostics"


    label_order_policy: Literal[
        "estimator_classes"
    ] = "estimator_classes"


    zero_division_policy: Literal[
        "zero"
    ] = "zero"


    rule_version: Literal[
        "ml_classification_diagnostics_v0.1"
    ] = ML_CLASSIFICATION_DIAGNOSTICS_RULE_VERSION


    # ========================================================
    # TEXT
    # ========================================================


    @field_validator(
        "workflow_id",
        "dataset_id",
        "model_id",
        "target_column",
        "estimator_key",
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
    # EXPERIMENT ID
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


        if (
            EXPERIMENT_ID_PATTERN
            .fullmatch(
                normalized
            )
            is None
        ):
            raise ValueError(
                (
                    "experiment_id must match "
                    "experiment:<32 lowercase hex characters>."
                )
            )


        return normalized


    # ========================================================
    # TRAINING CONTRACT SHA
    # ========================================================


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


        if (
            SHA256_PATTERN
            .fullmatch(
                normalized
            )
            is None
        ):
            raise ValueError(
                (
                    "training_contract_sha256 must be "
                    "a 64-character lowercase hex digest."
                )
            )


        return normalized


    # ========================================================
    # CLASS LABELS
    # ========================================================


    @field_validator(
        "class_labels",
        mode="before",
    )
    @classmethod
    def normalize_class_labels(
        cls,
        value: object,
    ) -> list[
        str
    ]:

        if not isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            raise ValueError(
                (
                    "class_labels must be "
                    "an ordered list."
                )
            )


        normalized = [
            _required_text(
                item,
                field_name=
                    "class_label",
            )

            for item
            in value
        ]


        if (
            len(
                set(
                    normalized
                )
            )
            !=
            len(
                normalized
            )
        ):
            raise ValueError(
                (
                    "class_labels must contain "
                    "unique textual class identities."
                )
            )


        return normalized


    # ========================================================
    # CONFUSION MATRIX INPUT
    # ========================================================


    @field_validator(
        "confusion_matrix",
        mode="before",
    )
    @classmethod
    def validate_confusion_matrix_cells(
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
                (
                    "confusion_matrix must be "
                    "a two-dimensional integer matrix."
                )
            )


        normalized = []


        for row in value:

            if not isinstance(
                row,
                (
                    list,
                    tuple,
                ),
            ):
                raise ValueError(
                    (
                        "confusion_matrix rows must "
                        "be ordered integer lists."
                    )
                )


            normalized_row = []


            for cell in row:

                if (
                    isinstance(
                        cell,
                        bool,
                    )
                    or
                    not isinstance(
                        cell,
                        int,
                    )
                ):
                    raise ValueError(
                        (
                            "confusion_matrix cells must "
                            "be non-negative integers."
                        )
                    )


                if cell < 0:
                    raise ValueError(
                        (
                            "confusion_matrix cells must "
                            "be non-negative integers."
                        )
                    )


                normalized_row.append(
                    int(
                        cell
                    )
                )


            normalized.append(
                normalized_row
            )


        return normalized


    # ========================================================
    # STRUCTURE + DETERMINISTIC DERIVATIONS
    # ========================================================


    @model_validator(
        mode="after"
    )
    def validate_structure(
        self,
    ) -> "MLClassificationDiagnosticsResult":

        if (
            len(
                self.class_labels
            )
            !=
            self.class_count
        ):
            raise ValueError(
                (
                    "class_count does not match "
                    "class_labels."
                )
            )


        if (
            len(
                self.per_class
            )
            !=
            self.class_count
        ):
            raise ValueError(
                (
                    "per_class length does not match "
                    "class_count."
                )
            )


        per_class_labels = [
            item.class_label

            for item
            in self.per_class
        ]


        if (
            per_class_labels
            !=
            self.class_labels
        ):
            raise ValueError(
                (
                    "per_class class order must exactly "
                    "match class_labels."
                )
            )


        if (
            len(
                self.confusion_matrix
            )
            !=
            self.class_count
        ):
            raise ValueError(
                (
                    "confusion_matrix row count does not "
                    "match class_count."
                )
            )


        for row in (
            self.confusion_matrix
        ):

            if (
                len(
                    row
                )
                !=
                self.class_count
            ):
                raise ValueError(
                    (
                        "confusion_matrix must be square "
                        "and match class_count."
                    )
                )


        matrix_total = sum(
            sum(
                row
            )

            for row
            in self.confusion_matrix
        )


        if (
            matrix_total
            !=
            self.evaluation_rows
        ):
            raise ValueError(
                (
                    "confusion_matrix total does not "
                    "match evaluation_rows."
                )
            )


        # ----------------------------------------------------
        # PER-CLASS COUNT CONSISTENCY
        # ----------------------------------------------------


        for class_index in range(
            self.class_count
        ):

            item = (
                self.per_class[
                    class_index
                ]
            )


            row_total = sum(
                self.confusion_matrix[
                    class_index
                ]
            )


            column_total = sum(
                self.confusion_matrix[
                    row_index
                ][
                    class_index
                ]

                for row_index
                in range(
                    self.class_count
                )
            )


            true_positive = (
                self.confusion_matrix[
                    class_index
                ][
                    class_index
                ]
            )


            false_negative = (
                row_total
                -
                true_positive
            )


            false_positive = (
                column_total
                -
                true_positive
            )


            true_negative = (
                self.evaluation_rows
                -
                true_positive
                -
                false_negative
                -
                false_positive
            )


            if (
                item.support
                !=
                row_total
            ):
                raise ValueError(
                    (
                        "Per-class support does not "
                        "match confusion-matrix row total."
                    )
                )


            if (
                item.true_positive
                !=
                true_positive
                or
                item.false_positive
                !=
                false_positive
                or
                item.false_negative
                !=
                false_negative
                or
                item.true_negative
                !=
                true_negative
            ):
                raise ValueError(
                    (
                        "Per-class TP / FP / FN / TN "
                        "counts do not match the "
                        "confusion matrix."
                    )
                )


            if (
                (
                    item.true_positive
                    +
                    item.false_positive
                    +
                    item.false_negative
                    +
                    item.true_negative
                )
                !=
                self.evaluation_rows
            ):
                raise ValueError(
                    (
                        "Per-class one-vs-rest counts "
                        "do not cover evaluation_rows."
                    )
                )


        # ----------------------------------------------------
        # ACCURACY
        # ----------------------------------------------------


        diagonal_total = sum(
            self.confusion_matrix[
                class_index
            ][
                class_index
            ]

            for class_index
            in range(
                self.class_count
            )
        )


        expected_accuracy = (
            _safe_ratio(
                diagonal_total,
                self.evaluation_rows,
            )
        )


        _assert_close(
            actual=
                self.accuracy,

            expected=
                expected_accuracy,

            field_name=
                "accuracy",
        )


        # ----------------------------------------------------
        # MACRO AVERAGE
        # ----------------------------------------------------


        expected_macro_precision = (
            sum(
                item.precision

                for item
                in self.per_class
            )
            /
            self.class_count
        )


        expected_macro_recall = (
            sum(
                item.recall

                for item
                in self.per_class
            )
            /
            self.class_count
        )


        expected_macro_f1 = (
            sum(
                item.f1

                for item
                in self.per_class
            )
            /
            self.class_count
        )


        _assert_close(
            actual=
                self.macro_average
                .precision,

            expected=
                expected_macro_precision,

            field_name=
                "macro_average.precision",
        )


        _assert_close(
            actual=
                self.macro_average
                .recall,

            expected=
                expected_macro_recall,

            field_name=
                "macro_average.recall",
        )


        _assert_close(
            actual=
                self.macro_average
                .f1,

            expected=
                expected_macro_f1,

            field_name=
                "macro_average.f1",
        )


        # ----------------------------------------------------
        # WEIGHTED AVERAGE
        # ----------------------------------------------------


        expected_weighted_precision = (
            sum(
                item.precision
                *
                item.support

                for item
                in self.per_class
            )
            /
            self.evaluation_rows
        )


        expected_weighted_recall = (
            sum(
                item.recall
                *
                item.support

                for item
                in self.per_class
            )
            /
            self.evaluation_rows
        )


        expected_weighted_f1 = (
            sum(
                item.f1
                *
                item.support

                for item
                in self.per_class
            )
            /
            self.evaluation_rows
        )


        _assert_close(
            actual=
                self.weighted_average
                .precision,

            expected=
                expected_weighted_precision,

            field_name=
                "weighted_average.precision",
        )


        _assert_close(
            actual=
                self.weighted_average
                .recall,

            expected=
                expected_weighted_recall,

            field_name=
                "weighted_average.recall",
        )


        _assert_close(
            actual=
                self.weighted_average
                .f1,

            expected=
                expected_weighted_f1,

            field_name=
                "weighted_average.f1",
        )


        # ----------------------------------------------------
        # BALANCED ACCURACY
        #
        # sklearn's balanced accuracy is the average recall over
        # classes represented in y_true.
        #
        # A fitted estimator class can legitimately have zero
        # support in one small holdout. Such a class therefore
        # remains visible in the matrix, but is excluded from
        # this balanced-accuracy denominator.
        # ----------------------------------------------------


        supported_classes = [
            item

            for item
            in self.per_class

            if (
                item.support
                >
                0
            )
        ]


        if not supported_classes:
            raise ValueError(
                (
                    "Classification diagnostics require "
                    "at least one supported holdout class."
                )
            )


        expected_balanced_accuracy = (
            sum(
                item.recall

                for item
                in supported_classes
            )
            /
            len(
                supported_classes
            )
        )


        _assert_close(
            actual=
                self.balanced_accuracy,

            expected=
                expected_balanced_accuracy,

            field_name=
                "balanced_accuracy",
        )


        return self
