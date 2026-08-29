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


from app.ml.model_metrics import (
    MLModelMetricDirection,
    MLModelPrimaryMetric,
    MLModelProblemType,
    ml_model_metric_direction,
    ml_model_metric_names,
    ml_model_primary_metric,
)


# ============================================================
# VERSION
# ============================================================


ML_PERFORMANCE_EVALUATION_RULE_VERSION = (
    "ml_performance_evaluation_v0.1"
)


# ============================================================
# POLICY
#
# These thresholds are operational v0.1 guardrails.
# They are not universal statistical constants.
#
# Classification:
#   primary = f1_macro
#   absolute deterioration in score points
#
# Regression:
#   primary = rmse
#   relative increase against reference holdout RMSE
# ============================================================


ML_PERFORMANCE_CLASSIFICATION_WARNING_DROP = (
    0.05
)


ML_PERFORMANCE_CLASSIFICATION_DEGRADED_DROP = (
    0.10
)


ML_PERFORMANCE_REGRESSION_WARNING_INCREASE_RATIO = (
    0.10
)


ML_PERFORMANCE_REGRESSION_DEGRADED_INCREASE_RATIO = (
    0.25
)


FLOAT_TOLERANCE = 1e-12


# ============================================================
# TYPES
# ============================================================


MLPerformanceStatus = Literal[
    "ok",
    "warning",
    "degraded",
]


MLPerformancePrivacyScope = Literal[
    "aggregate_only",
]


MLPerformanceDegradationBasis = Literal[
    "absolute_points",
    "relative_increase",
]


# ============================================================
# PATTERNS
# ============================================================


SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)


MODEL_ID_PATTERN = re.compile(
    r"^model:[0-9a-f]{32}$"
)


EXPERIMENT_ID_PATTERN = re.compile(
    r"^experiment:[0-9a-f]{32}$"
)


PERFORMANCE_EVALUATION_ID_PATTERN = re.compile(
    r"^performance-evaluation:[0-9a-f]{32}$"
)


# ============================================================
# HELPERS
# ============================================================


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
            f"{field_name} cannot be empty."
        )


    return normalized


def _finite_float(
    value: object,
    *,
    field_name: str,
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
            (
                f"{field_name} must be "
                "a finite number."
            )
        )


    normalized = float(
        value
    )


    if not math.isfinite(
        normalized
    ):
        raise ValueError(
            (
                f"{field_name} must "
                "be finite."
            )
        )


    return normalized


def _close(
    left: float,
    right: float,
) -> bool:

    return math.isclose(
        float(
            left
        ),
        float(
            right
        ),
        rel_tol=
            FLOAT_TOLERANCE,
        abs_tol=
            FLOAT_TOLERANCE,
    )


def _validate_metric_value(
    *,
    problem_type: str,
    metric_name: str,
    value: float,
) -> None:

    if (
        problem_type
        ==
        "classification"
    ):
        if (
            value < 0.0
            or
            value > 1.0
        ):
            raise ValueError(
                (
                    "Classification metrics must "
                    "remain between 0 and 1. "
                    f"metric_name={metric_name}"
                )
            )


        return


    if (
        metric_name
        in {
            "mae",
            "rmse",
            "median_absolute_error",
        }
        and
        value < 0.0
    ):
        raise ValueError(
            (
                "Regression error metrics "
                "cannot be negative. "
                f"metric_name={metric_name}"
            )
        )


# ============================================================
# PRIMARY PERFORMANCE POLICY
# ============================================================


def ml_performance_status_for_primary_metric(
    *,
    problem_type: str,
    reference_value: float,
    observed_value: float,
) -> MLPerformanceStatus:

    reference = (
        _finite_float(
            reference_value,
            field_name=
                "reference_value",
        )
    )


    observed = (
        _finite_float(
            observed_value,
            field_name=
                "observed_value",
        )
    )


    if (
        problem_type
        ==
        "classification"
    ):
        degradation = max(
            0.0,
            reference
            -
            observed,
        )


        if (
            degradation
            +
            FLOAT_TOLERANCE
            >=
            ML_PERFORMANCE_CLASSIFICATION_DEGRADED_DROP
        ):
            return "degraded"


        if (
            degradation
            +
            FLOAT_TOLERANCE
            >=
            ML_PERFORMANCE_CLASSIFICATION_WARNING_DROP
        ):
            return "warning"


        return "ok"


    if (
        problem_type
        ==
        "regression"
    ):
        if (
            reference < 0.0
            or
            observed < 0.0
        ):
            raise ValueError(
                (
                    "RMSE reference and observed "
                    "values cannot be negative."
                )
            )


        increase = max(
            0.0,
            observed
            -
            reference,
        )


        if (
            increase
            <=
            FLOAT_TOLERANCE
        ):
            return "ok"


        # Perfect reference RMSE followed by any meaningful
        # positive RMSE is fail-closed degraded.
        if (
            reference
            <=
            FLOAT_TOLERANCE
        ):
            return "degraded"


        ratio = (
            increase
            /
            reference
        )


        if (
            ratio
            +
            FLOAT_TOLERANCE
            >=
            ML_PERFORMANCE_REGRESSION_DEGRADED_INCREASE_RATIO
        ):
            return "degraded"


        if (
            ratio
            +
            FLOAT_TOLERANCE
            >=
            ML_PERFORMANCE_REGRESSION_WARNING_INCREASE_RATIO
        ):
            return "warning"


        return "ok"


    raise ValueError(
        (
            "Unsupported ML Performance "
            "problem type. "
            f"problem_type={problem_type}"
        )
    )


# ============================================================
# METRIC COMPARISON
# ============================================================


class MLPerformanceMetricComparison(
    BaseModel
):

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


    metric_name: str = Field(
        min_length=1,
    )


    direction: MLModelMetricDirection


    reference_value: float


    observed_value: float


    delta: float


    degradation_amount: float = Field(
        ge=0.0,
    )


    @field_validator(
        "metric_name",
        mode="before",
    )
    @classmethod
    def normalize_metric_name(
        cls,
        value: object,
    ) -> str:

        return (
            _required_text(
                value,
                field_name=
                    "metric_name",
            )
        )


    @field_validator(
        "reference_value",
        "observed_value",
        "delta",
        "degradation_amount",
        mode="before",
    )
    @classmethod
    def validate_finite_value(
        cls,
        value: object,
        info,
    ) -> float:

        return (
            _finite_float(
                value,
                field_name=
                    info.field_name,
            )
        )


    @model_validator(
        mode="after"
    )
    def validate_comparison(
        self,
    ) -> "MLPerformanceMetricComparison":

        expected_delta = (
            self.observed_value
            -
            self.reference_value
        )


        if not _close(
            self.delta,
            expected_delta,
        ):
            raise ValueError(
                (
                    "Metric delta does not match "
                    "observed - reference."
                )
            )


        if (
            self.direction
            ==
            "higher_is_better"
        ):
            expected_degradation = max(
                0.0,
                self.reference_value
                -
                self.observed_value,
            )

        else:
            expected_degradation = max(
                0.0,
                self.observed_value
                -
                self.reference_value,
            )


        if not _close(
            self.degradation_amount,
            expected_degradation,
        ):
            raise ValueError(
                (
                    "Metric degradation_amount "
                    "does not match metric direction."
                )
            )


        return self


# ============================================================
# PERFORMANCE EVALUATION RECORD
# ============================================================


class MLPerformanceEvaluationRecord(
    BaseModel
):
    """
    Privacy-minimal supervised performance evidence for one
    persisted trusted Model Artifact evaluated on one
    server-owned observed Preparation dataset.

    This contract represents a SUCCESSFUL labeled evaluation.

    If the observed dataset does not contain a valid true target,
    no MLPerformanceEvaluationRecord may be created.

    No raw:
    - rows;
    - feature values;
    - target values;
    - predictions;
    - probabilities;
    - estimator state;
    - model bytes;
    - filesystem paths

    belong to this record.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


    performance_evaluation_id: str = Field(
        min_length=1,
    )


    model_id: str = Field(
        min_length=1,
    )


    workflow_id: str = Field(
        min_length=1,
    )


    reference_dataset_id: str = Field(
        min_length=1,
    )


    observed_dataset_id: str = Field(
        min_length=1,
    )


    experiment_id: str = Field(
        min_length=1,
    )


    preparation_session_revision: int = Field(
        ge=0,
        strict=True,
    )


    observed_preparation_session_revision: int = Field(
        ge=0,
        strict=True,
    )


    training_contract_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


    problem_type: MLModelProblemType


    target_column: str = Field(
        min_length=1,
    )


    reference_evaluation_scope: Literal[
        "training_holdout"
    ] = "training_holdout"


    observed_evaluation_scope: Literal[
        "validated_observed_dataset"
    ] = "validated_observed_dataset"


    reference_evaluation_row_count: int = Field(
        gt=0,
        strict=True,
    )


    observed_row_count: int = Field(
        gt=0,
        strict=True,
    )


    evaluated_at_utc: str = Field(
        min_length=1,
    )


    metric_results: list[
        MLPerformanceMetricComparison
    ] = Field(
        min_length=1,
        max_length=16,
    )


    primary_metric: MLModelPrimaryMetric


    primary_metric_degradation_amount: float = Field(
        ge=0.0,
    )


    primary_metric_degradation_ratio: (
        float
        |
        None
    ) = Field(
        default=None,
        ge=0.0,
    )


    degradation_basis: MLPerformanceDegradationBasis


    performance_status: MLPerformanceStatus


    privacy_scope: (
        MLPerformancePrivacyScope
    ) = "aggregate_only"


    rule_version: Literal[
        "ml_performance_evaluation_v0.1"
    ] = ML_PERFORMANCE_EVALUATION_RULE_VERSION


    # ========================================================
    # TEXT
    # ========================================================


    @field_validator(
        "performance_evaluation_id",
        "model_id",
        "workflow_id",
        "reference_dataset_id",
        "observed_dataset_id",
        "experiment_id",
        "target_column",
        "evaluated_at_utc",
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
    # SERVER-SHAPED IDS
    # ========================================================


    @field_validator(
        "performance_evaluation_id",
    )
    @classmethod
    def validate_performance_evaluation_id(
        cls,
        value: str,
    ) -> str:

        if (
            PERFORMANCE_EVALUATION_ID_PATTERN
            .fullmatch(
                value
            )
            is None
        ):
            raise ValueError(
                (
                    "performance_evaluation_id "
                    "must be server-shaped."
                )
            )


        return value


    @field_validator(
        "model_id",
    )
    @classmethod
    def validate_model_id(
        cls,
        value: str,
    ) -> str:

        if (
            MODEL_ID_PATTERN
            .fullmatch(
                value
            )
            is None
        ):
            raise ValueError(
                (
                    "model_id must be a "
                    "server-shaped model identifier."
                )
            )


        return value


    @field_validator(
        "experiment_id",
    )
    @classmethod
    def validate_experiment_id(
        cls,
        value: str,
    ) -> str:

        if (
            EXPERIMENT_ID_PATTERN
            .fullmatch(
                value
            )
            is None
        ):
            raise ValueError(
                (
                    "experiment_id must be a "
                    "server-shaped experiment "
                    "identifier."
                )
            )


        return value


    # ========================================================
    # SHA
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
                    "training_contract_sha256 "
                    "must be a lowercase "
                    "64-character SHA-256 digest."
                )
            )


        return normalized


    # ========================================================
    # PRIMARY NUMBERS
    # ========================================================


    @field_validator(
        "primary_metric_degradation_amount",
        "primary_metric_degradation_ratio",
        mode="before",
    )
    @classmethod
    def validate_primary_numeric(
        cls,
        value: object,
        info,
    ) -> (
        float
        |
        None
    ):

        if value is None:
            return None


        return (
            _finite_float(
                value,
                field_name=
                    info.field_name,
            )
        )


    # ========================================================
    # COMPLETE CONSISTENCY
    # ========================================================


    @model_validator(
        mode="after"
    )
    def validate_evaluation(
        self,
    ) -> "MLPerformanceEvaluationRecord":

        expected_names = list(
            ml_model_metric_names(
                problem_type=
                    self.problem_type
            )
        )


        actual_names = [
            result.metric_name

            for result
            in self.metric_results
        ]


        # Deterministic ordered surface.
        if (
            actual_names
            !=
            expected_names
        ):
            raise ValueError(
                (
                    "Performance metric results "
                    "must exactly match the "
                    "canonical ordered metric surface."
                )
            )


        for result in (
            self.metric_results
        ):
            expected_direction = (
                ml_model_metric_direction(
                    problem_type=
                        self.problem_type,

                    metric_name=
                        result.metric_name,
                )
            )


            if (
                result.direction
                !=
                expected_direction
            ):
                raise ValueError(
                    (
                        "Performance metric direction "
                        "does not match canonical "
                        "metric semantics. "
                        f"metric_name="
                        f"{result.metric_name}"
                    )
                )


            _validate_metric_value(
                problem_type=
                    self.problem_type,

                metric_name=
                    result.metric_name,

                value=
                    result.reference_value,
            )


            _validate_metric_value(
                problem_type=
                    self.problem_type,

                metric_name=
                    result.metric_name,

                value=
                    result.observed_value,
            )


        expected_primary = (
            ml_model_primary_metric(
                problem_type=
                    self.problem_type
            )
        )


        if (
            self.primary_metric
            !=
            expected_primary
        ):
            raise ValueError(
                (
                    "primary_metric does not "
                    "match canonical ML policy."
                )
            )


        primary_result = next(
            result
            for result
            in self.metric_results
            if (
                result.metric_name
                ==
                expected_primary
            )
        )


        expected_degradation = (
            primary_result
            .degradation_amount
        )


        if not _close(
            self.primary_metric_degradation_amount,
            expected_degradation,
        ):
            raise ValueError(
                (
                    "Primary metric degradation "
                    "does not match primary "
                    "metric comparison."
                )
            )


        if (
            self.problem_type
            ==
            "classification"
        ):
            if (
                self.degradation_basis
                !=
                "absolute_points"
            ):
                raise ValueError(
                    (
                        "Classification performance "
                        "must use absolute_points."
                    )
                )


            if (
                self.primary_metric_degradation_ratio
                is not None
            ):
                raise ValueError(
                    (
                        "Classification v0.1 does not "
                        "persist a relative degradation "
                        "ratio."
                    )
                )


        else:
            if (
                self.degradation_basis
                !=
                "relative_increase"
            ):
                raise ValueError(
                    (
                        "Regression performance "
                        "must use relative_increase."
                    )
                )


            reference_rmse = (
                primary_result
                .reference_value
            )


            if (
                reference_rmse
                <=
                FLOAT_TOLERANCE
            ):
                expected_ratio = (
                    None
                )

            else:
                expected_ratio = (
                    expected_degradation
                    /
                    reference_rmse
                )


            if (
                expected_ratio
                is None
            ):
                if (
                    self.primary_metric_degradation_ratio
                    is not None
                ):
                    raise ValueError(
                        (
                            "Regression degradation "
                            "ratio must be absent for "
                            "zero reference RMSE."
                        )
                    )

            else:
                if (
                    self.primary_metric_degradation_ratio
                    is None
                    or
                    not _close(
                        self.primary_metric_degradation_ratio,
                        expected_ratio,
                    )
                ):
                    raise ValueError(
                        (
                            "Regression degradation "
                            "ratio is inconsistent."
                        )
                    )


        expected_status = (
            ml_performance_status_for_primary_metric(
                problem_type=
                    self.problem_type,

                reference_value=
                    primary_result
                    .reference_value,

                observed_value=
                    primary_result
                    .observed_value,
            )
        )


        if (
            self.performance_status
            !=
            expected_status
        ):
            raise ValueError(
                (
                    "performance_status does not "
                    "match Performance policy."
                )
            )


        return self
