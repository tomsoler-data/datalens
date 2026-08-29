from __future__ import annotations

import math
import re

from typing import (
    Annotated,
    Literal,
    Union,
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


ML_DRIFT_EVALUATION_RULE_VERSION = (
    "ml_drift_evaluation_v0.1"
)


# ============================================================
# POLICY
# ============================================================


ML_DRIFT_PSI_EPSILON = 1e-6

ML_DRIFT_PSI_WARNING_THRESHOLD = 0.10
ML_DRIFT_PSI_DRIFT_THRESHOLD = 0.25

ML_DRIFT_MISSING_RATE_WARNING_THRESHOLD = 0.05
ML_DRIFT_MISSING_RATE_DRIFT_THRESHOLD = 0.10

ML_DRIFT_OUTSIDE_RANGE_WARNING_THRESHOLD = 0.05
ML_DRIFT_OUTSIDE_RANGE_DRIFT_THRESHOLD = 0.10

RATE_TOLERANCE = 1e-9


# ============================================================
# TYPES
# ============================================================


MLDriftStatus = Literal[
    "ok",
    "warning",
    "drift",
]

MLDriftDistributionStatus = Literal[
    "ok",
    "warning",
    "drift",
    "not_evaluable",
]

MLDriftFeatureKind = Literal[
    "numeric",
    "categorical",
]

MLDriftPrivacyScope = Literal[
    "aggregate_only",
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

MONITORING_PROFILE_ID_PATTERN = re.compile(
    r"^monitoring-profile:[0-9a-f]{32}$"
)

DRIFT_EVALUATION_ID_PATTERN = re.compile(
    r"^drift-evaluation:[0-9a-f]{32}$"
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


def _validate_sha256(
    value: object,
    *,
    field_name: str,
) -> str:

    normalized = str(
        value
        if value is not None
        else ""
    ).strip().lower()

    if (
        SHA256_PATTERN.fullmatch(
            normalized
        )
        is None
    ):
        raise ValueError(
            (
                f"{field_name} must be a "
                "lowercase 64-character "
                "SHA-256 digest."
            )
        )

    return normalized


def _rates_close(
    left: float,
    right: float,
) -> bool:

    return math.isclose(
        left,
        right,
        rel_tol=0.0,
        abs_tol=RATE_TOLERANCE,
    )


def ml_drift_status_for_psi(
    score: float,
) -> MLDriftStatus:

    if score >= ML_DRIFT_PSI_DRIFT_THRESHOLD:
        return "drift"

    if score >= ML_DRIFT_PSI_WARNING_THRESHOLD:
        return "warning"

    return "ok"


def ml_drift_status_for_rate_shift(
    absolute_delta: float,
    *,
    warning_threshold: float,
    drift_threshold: float,
) -> MLDriftStatus:

    if absolute_delta >= drift_threshold:
        return "drift"

    if absolute_delta >= warning_threshold:
        return "warning"

    return "ok"


def ml_drift_max_status(
    statuses: list[
        MLDriftStatus
    ],
) -> MLDriftStatus:

    rank = {
        "ok": 0,
        "warning": 1,
        "drift": 2,
    }

    return max(
        statuses,
        key=lambda status: rank[status],
    )


# ============================================================
# NUMERIC RESULT
# ============================================================


class MLNumericDriftFeatureResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    feature_name: str = Field(
        min_length=1,
    )

    kind: Literal[
        "numeric"
    ] = "numeric"

    reference_missing_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    observed_total_count: int = Field(
        gt=0,
    )

    observed_non_missing_count: int = Field(
        ge=0,
    )

    observed_missing_count: int = Field(
        ge=0,
    )

    observed_missing_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    missing_rate_delta: float = Field(
        ge=-1.0,
        le=1.0,
    )

    absolute_missing_rate_delta: float = Field(
        ge=0.0,
        le=1.0,
    )

    population_stability_index: (
        float
        |
        None
    ) = Field(
        default=None,
        ge=0.0,
    )

    distribution_status: (
        MLDriftDistributionStatus
    )

    missingness_status: MLDriftStatus

    outside_reference_range_count: int = Field(
        ge=0,
    )

    outside_reference_range_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    range_status: MLDriftStatus

    status: MLDriftStatus

    @field_validator(
        "feature_name",
        mode="before",
    )
    @classmethod
    def normalize_feature_name(
        cls,
        value: object,
    ) -> str:

        return _required_text(
            value,
            field_name="feature_name",
        )

    @model_validator(
        mode="after"
    )
    def validate_numeric_result(
        self,
    ) -> "MLNumericDriftFeatureResult":

        if (
            self.observed_non_missing_count
            +
            self.observed_missing_count
            !=
            self.observed_total_count
        ):
            raise ValueError(
                (
                    "Observed numeric counts must "
                    "sum to observed_total_count."
                )
            )

        expected_missing_rate = (
            self.observed_missing_count
            /
            self.observed_total_count
        )

        if not _rates_close(
            self.observed_missing_rate,
            expected_missing_rate,
        ):
            raise ValueError(
                (
                    "Observed numeric missing_rate "
                    "does not match counts."
                )
            )

        expected_delta = (
            self.observed_missing_rate
            -
            self.reference_missing_rate
        )

        if not _rates_close(
            self.missing_rate_delta,
            expected_delta,
        ):
            raise ValueError(
                (
                    "Numeric missing_rate_delta "
                    "does not match reference and "
                    "observed rates."
                )
            )

        if not _rates_close(
            self.absolute_missing_rate_delta,
            abs(
                expected_delta
            ),
        ):
            raise ValueError(
                (
                    "Numeric absolute missing-rate "
                    "delta is inconsistent."
                )
            )

        expected_missingness_status = (
            ml_drift_status_for_rate_shift(
                self.absolute_missing_rate_delta,
                warning_threshold=(
                    ML_DRIFT_MISSING_RATE_WARNING_THRESHOLD
                ),
                drift_threshold=(
                    ML_DRIFT_MISSING_RATE_DRIFT_THRESHOLD
                ),
            )
        )

        if (
            self.missingness_status
            !=
            expected_missingness_status
        ):
            raise ValueError(
                (
                    "Numeric missingness_status "
                    "does not match policy."
                )
            )

        if (
            self.outside_reference_range_count
            >
            self.observed_non_missing_count
        ):
            raise ValueError(
                (
                    "outside_reference_range_count "
                    "cannot exceed observed "
                    "non-missing count."
                )
            )

        expected_outside_rate = (
            (
                self.outside_reference_range_count
                /
                self.observed_non_missing_count
            )
            if self.observed_non_missing_count > 0
            else 0.0
        )

        if not _rates_close(
            self.outside_reference_range_rate,
            expected_outside_rate,
        ):
            raise ValueError(
                (
                    "outside_reference_range_rate "
                    "does not match counts."
                )
            )

        expected_range_status = (
            ml_drift_status_for_rate_shift(
                self.outside_reference_range_rate,
                warning_threshold=(
                    ML_DRIFT_OUTSIDE_RANGE_WARNING_THRESHOLD
                ),
                drift_threshold=(
                    ML_DRIFT_OUTSIDE_RANGE_DRIFT_THRESHOLD
                ),
            )
        )

        if (
            self.range_status
            !=
            expected_range_status
        ):
            raise ValueError(
                (
                    "Numeric range_status does not "
                    "match policy."
                )
            )

        if (
            self.population_stability_index
            is None
        ):
            if (
                self.distribution_status
                !=
                "not_evaluable"
            ):
                raise ValueError(
                    (
                        "Missing numeric PSI requires "
                        "distribution_status="
                        "not_evaluable."
                    )
                )

            if (
                self.observed_non_missing_count
                !=
                0
            ):
                raise ValueError(
                    (
                        "Numeric PSI can be absent "
                        "only when no observed "
                        "non-missing value exists."
                    )
                )

        else:
            expected_distribution_status = (
                ml_drift_status_for_psi(
                    self.population_stability_index
                )
            )

            if (
                self.distribution_status
                !=
                expected_distribution_status
            ):
                raise ValueError(
                    (
                        "Numeric distribution_status "
                        "does not match PSI policy."
                    )
                )

        expected_status: MLDriftStatus

        if (
            self.observed_non_missing_count
            ==
            0
        ):
            expected_status = "drift"

        else:
            distribution_status: MLDriftStatus = (
                self.distribution_status
            )

            expected_status = (
                ml_drift_max_status(
                    [
                        distribution_status,
                        self.missingness_status,
                        self.range_status,
                    ]
                )
            )

        if self.status != expected_status:
            raise ValueError(
                (
                    "Numeric feature status does "
                    "not match component statuses."
                )
            )

        return self


# ============================================================
# CATEGORICAL RESULT
# ============================================================


class MLCategoricalDriftFeatureResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    feature_name: str = Field(
        min_length=1,
    )

    kind: Literal[
        "categorical"
    ] = "categorical"

    reference_missing_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    observed_total_count: int = Field(
        gt=0,
    )

    observed_non_missing_count: int = Field(
        ge=0,
    )

    observed_missing_count: int = Field(
        ge=0,
    )

    observed_missing_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    missing_rate_delta: float = Field(
        ge=-1.0,
        le=1.0,
    )

    absolute_missing_rate_delta: float = Field(
        ge=0.0,
        le=1.0,
    )

    population_stability_index: (
        float
        |
        None
    ) = Field(
        default=None,
        ge=0.0,
    )

    distribution_status: (
        MLDriftDistributionStatus
    )

    missingness_status: MLDriftStatus

    reference_other_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    observed_untracked_count: int = Field(
        ge=0,
    )

    observed_untracked_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    untracked_rate_delta: float = Field(
        ge=-1.0,
        le=1.0,
    )

    absolute_untracked_rate_delta: float = Field(
        ge=0.0,
        le=1.0,
    )

    status: MLDriftStatus

    @field_validator(
        "feature_name",
        mode="before",
    )
    @classmethod
    def normalize_feature_name(
        cls,
        value: object,
    ) -> str:

        return _required_text(
            value,
            field_name="feature_name",
        )

    @model_validator(
        mode="after"
    )
    def validate_categorical_result(
        self,
    ) -> "MLCategoricalDriftFeatureResult":

        if (
            self.observed_non_missing_count
            +
            self.observed_missing_count
            !=
            self.observed_total_count
        ):
            raise ValueError(
                (
                    "Observed categorical counts "
                    "must sum to "
                    "observed_total_count."
                )
            )

        expected_missing_rate = (
            self.observed_missing_count
            /
            self.observed_total_count
        )

        if not _rates_close(
            self.observed_missing_rate,
            expected_missing_rate,
        ):
            raise ValueError(
                (
                    "Observed categorical "
                    "missing_rate does not match "
                    "counts."
                )
            )

        expected_missing_delta = (
            self.observed_missing_rate
            -
            self.reference_missing_rate
        )

        if not _rates_close(
            self.missing_rate_delta,
            expected_missing_delta,
        ):
            raise ValueError(
                (
                    "Categorical missing_rate_delta "
                    "is inconsistent."
                )
            )

        if not _rates_close(
            self.absolute_missing_rate_delta,
            abs(
                expected_missing_delta
            ),
        ):
            raise ValueError(
                (
                    "Categorical absolute "
                    "missing-rate delta is "
                    "inconsistent."
                )
            )

        expected_missingness_status = (
            ml_drift_status_for_rate_shift(
                self.absolute_missing_rate_delta,
                warning_threshold=(
                    ML_DRIFT_MISSING_RATE_WARNING_THRESHOLD
                ),
                drift_threshold=(
                    ML_DRIFT_MISSING_RATE_DRIFT_THRESHOLD
                ),
            )
        )

        if (
            self.missingness_status
            !=
            expected_missingness_status
        ):
            raise ValueError(
                (
                    "Categorical missingness_status "
                    "does not match policy."
                )
            )

        if (
            self.observed_untracked_count
            >
            self.observed_non_missing_count
        ):
            raise ValueError(
                (
                    "observed_untracked_count "
                    "cannot exceed observed "
                    "non-missing count."
                )
            )

        expected_untracked_rate = (
            (
                self.observed_untracked_count
                /
                self.observed_non_missing_count
            )
            if self.observed_non_missing_count > 0
            else 0.0
        )

        if not _rates_close(
            self.observed_untracked_rate,
            expected_untracked_rate,
        ):
            raise ValueError(
                (
                    "observed_untracked_rate "
                    "does not match counts."
                )
            )

        expected_untracked_delta = (
            self.observed_untracked_rate
            -
            self.reference_other_rate
        )

        if not _rates_close(
            self.untracked_rate_delta,
            expected_untracked_delta,
        ):
            raise ValueError(
                (
                    "untracked_rate_delta is "
                    "inconsistent."
                )
            )

        if not _rates_close(
            self.absolute_untracked_rate_delta,
            abs(
                expected_untracked_delta
            ),
        ):
            raise ValueError(
                (
                    "absolute_untracked_rate_delta "
                    "is inconsistent."
                )
            )

        if (
            self.population_stability_index
            is None
        ):
            if (
                self.distribution_status
                !=
                "not_evaluable"
            ):
                raise ValueError(
                    (
                        "Missing categorical PSI "
                        "requires "
                        "distribution_status="
                        "not_evaluable."
                    )
                )

            if (
                self.observed_non_missing_count
                !=
                0
            ):
                raise ValueError(
                    (
                        "Categorical PSI can be "
                        "absent only when no "
                        "observed non-missing "
                        "value exists."
                    )
                )

        else:
            expected_distribution_status = (
                ml_drift_status_for_psi(
                    self.population_stability_index
                )
            )

            if (
                self.distribution_status
                !=
                expected_distribution_status
            ):
                raise ValueError(
                    (
                        "Categorical "
                        "distribution_status does "
                        "not match PSI policy."
                    )
                )

        expected_status: MLDriftStatus

        if (
            self.observed_non_missing_count
            ==
            0
        ):
            expected_status = "drift"

        else:
            distribution_status: MLDriftStatus = (
                self.distribution_status
            )

            expected_status = (
                ml_drift_max_status(
                    [
                        distribution_status,
                        self.missingness_status,
                    ]
                )
            )

        if self.status != expected_status:
            raise ValueError(
                (
                    "Categorical feature status "
                    "does not match component "
                    "statuses."
                )
            )

        return self


# ============================================================
# FEATURE UNION
# ============================================================


MLDriftFeatureResult = Annotated[
    Union[
        MLNumericDriftFeatureResult,
        MLCategoricalDriftFeatureResult,
    ],
    Field(
        discriminator="kind"
    ),
]


# ============================================================
# EVALUATION RECORD
# ============================================================


class MLDriftEvaluationRecord(
    BaseModel
):
    """
    Privacy-minimal drift result for one trusted model/profile.

    No raw rows, raw category labels, predictions, model bytes,
    or fitted estimator state are allowed in this contract.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    evaluation_id: str = Field(
        min_length=1,
    )

    profile_id: str = Field(
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
    )

    training_contract_sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    evaluated_at_utc: str = Field(
        min_length=1,
    )

    observed_row_count: int = Field(
        gt=0,
    )

    feature_results: list[
        MLDriftFeatureResult
    ] = Field(
        min_length=1,
        max_length=256,
    )

    warning_feature_count: int = Field(
        ge=0,
    )

    drift_feature_count: int = Field(
        ge=0,
    )

    overall_status: MLDriftStatus

    privacy_scope: (
        MLDriftPrivacyScope
    ) = "aggregate_only"

    rule_version: Literal[
        "ml_drift_evaluation_v0.1"
    ] = ML_DRIFT_EVALUATION_RULE_VERSION

    @field_validator(
        "evaluation_id",
        "profile_id",
        "model_id",
        "workflow_id",
        "reference_dataset_id",
        "observed_dataset_id",
        "experiment_id",
        "evaluated_at_utc",
        mode="before",
    )
    @classmethod
    def normalize_required_text(
        cls,
        value: object,
        info,
    ) -> str:

        return _required_text(
            value,
            field_name=info.field_name,
        )

    @field_validator(
        "evaluation_id",
    )
    @classmethod
    def validate_evaluation_id(
        cls,
        value: str,
    ) -> str:

        if (
            DRIFT_EVALUATION_ID_PATTERN
            .fullmatch(
                value
            )
            is None
        ):
            raise ValueError(
                (
                    "evaluation_id must be a "
                    "server-shaped drift "
                    "evaluation identifier."
                )
            )

        return value

    @field_validator(
        "profile_id",
    )
    @classmethod
    def validate_profile_id(
        cls,
        value: str,
    ) -> str:

        if (
            MONITORING_PROFILE_ID_PATTERN
            .fullmatch(
                value
            )
            is None
        ):
            raise ValueError(
                (
                    "profile_id must be a "
                    "server-shaped monitoring "
                    "profile identifier."
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
            MODEL_ID_PATTERN.fullmatch(
                value
            )
            is None
        ):
            raise ValueError(
                (
                    "model_id must be a "
                    "server-shaped model "
                    "identifier."
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

    @field_validator(
        "training_contract_sha256",
        mode="before",
    )
    @classmethod
    def validate_training_contract_sha256(
        cls,
        value: object,
    ) -> str:

        return _validate_sha256(
            value,
            field_name=(
                "training_contract_sha256"
            ),
        )

    @model_validator(
        mode="after"
    )
    def validate_evaluation(
        self,
    ) -> "MLDriftEvaluationRecord":

        feature_names = [
            result.feature_name
            for result
            in self.feature_results
        ]

        if (
            len(feature_names)
            !=
            len(set(feature_names))
        ):
            raise ValueError(
                (
                    "Drift feature names must "
                    "be unique."
                )
            )

        for result in self.feature_results:
            if (
                result.observed_total_count
                !=
                self.observed_row_count
            ):
                raise ValueError(
                    (
                        "Every drift feature result "
                        "must describe the complete "
                        "observed row surface."
                    )
                )

        expected_warning_count = sum(
            1
            for result
            in self.feature_results
            if result.status == "warning"
        )

        expected_drift_count = sum(
            1
            for result
            in self.feature_results
            if result.status == "drift"
        )

        if (
            self.warning_feature_count
            !=
            expected_warning_count
        ):
            raise ValueError(
                (
                    "warning_feature_count does "
                    "not match feature results."
                )
            )

        if (
            self.drift_feature_count
            !=
            expected_drift_count
        ):
            raise ValueError(
                (
                    "drift_feature_count does "
                    "not match feature results."
                )
            )

        expected_overall_status = (
            "drift"
            if expected_drift_count > 0
            else
            (
                "warning"
                if expected_warning_count > 0
                else "ok"
            )
        )

        if (
            self.overall_status
            !=
            expected_overall_status
        ):
            raise ValueError(
                (
                    "overall_status does not "
                    "match feature results."
                )
            )

        return self
