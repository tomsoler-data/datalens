from __future__ import annotations

import math
import re

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.ml.baseline import (
    MLBaselineComparisonResult,
    MLBaselineEvaluationResult,
)

from app.ml.classification_diagnostics import (
    MLClassificationDiagnosticsResult,
)

from app.ml.decision_threshold import (
    MLDecisionThresholdContract,
    MLDecisionThresholdResult,
)

from app.ml.model_explainability import (
    MLModelExplainabilityResult,
)


from app.ml.model_metrics import (
    ml_model_primary_metric,
)


# ============================================================
# VERSIONS
# ============================================================


ML_MODEL_EVALUATION_SUMMARY_RULE_VERSION = (
    "ml_model_evaluation_summary_v0.1"
)

ML_MODEL_SELECTION_EVIDENCE_RULE_VERSION = (
    "ml_model_selection_evidence_v0.1"
)


# ============================================================
# TYPES
# ============================================================


MLModelEvaluationProblemType = Literal[
    "regression",
    "classification",
]

MLModelEvaluationPrimaryMetric = Literal[
    "rmse",
    "f1_macro",
]

MLModelSelectionSource = Literal[
    "standalone_model",
    "model_comparison",
    "tuned_model_promotion",
]

MLModelSelectionStatus = Literal[
    "selection_not_available",
    "verified_selected",
]

MLModelSelectionMetricScope = Literal[
    "not_available",
    "final_holdout",
    "inner_cross_validation",
]

MLModelSelectionPolicy = Literal[
    "regression_rmse_v0.1",
    "classification_f1_macro_v0.1",
    "rank_1_only",
]

MLModelEvaluationLimitation = Literal[
    "single_holdout_evaluation",
    "no_external_validation",
    "feature_importance_not_causal",
    "selection_evidence_not_available",
    "decision_threshold_not_included",
    "requested_threshold_not_optimized",
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
            f"{field_name} cannot be empty."
        )

    return normalized


def _finite_float(
    value: object,
    *,
    field_name: str,
) -> float:

    if (
        isinstance(value, bool)
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
            f"{field_name} must be a finite number."
        )

    normalized = float(value)

    if not math.isfinite(normalized):
        raise ValueError(
            f"{field_name} must be finite."
        )

    return normalized


def _assert_close(
    *,
    actual: float,
    expected: float,
    field_name: str,
) -> None:

    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            (
                f"{field_name} is inconsistent with "
                "the trusted evaluation evidence."
            )
        )


def model_evaluation_primary_metric(
    *,
    problem_type: str,
) -> MLModelEvaluationPrimaryMetric:
    """
    Backward-compatible Model Evaluation wrapper around the
    canonical ML metric semantics.
    """

    return (
        ml_model_primary_metric(
            problem_type=
                problem_type
        )
    )


def expected_model_evaluation_limitations(
    *,
    problem_type: str,
    selection_source: MLModelSelectionSource,
    threshold_requested: bool,
) -> list[
    MLModelEvaluationLimitation
]:

    limitations: list[
        MLModelEvaluationLimitation
    ] = [
        "single_holdout_evaluation",
        "no_external_validation",
        "feature_importance_not_causal",
    ]

    if selection_source == "standalone_model":
        limitations.append(
            "selection_evidence_not_available"
        )

    if problem_type == "classification":

        if threshold_requested:
            limitations.append(
                "requested_threshold_not_optimized"
            )

        else:
            limitations.append(
                "decision_threshold_not_included"
            )

    return limitations


# ============================================================
# SUMMARY REQUEST CONTRACT
# ============================================================


class MLModelEvaluationSummaryContract(
    BaseModel
):
    """
    Configuration for one deterministic evaluation summary of an
    already persisted trusted Model Artifact.

    The caller does NOT supply:
    - metrics;
    - baseline results;
    - selection rank;
    - classification diagnostics;
    - predictions;
    - probabilities;
    - feature importances;
    - model bytes;
    - model path;
    - Training Contract;
    - Experiment Provenance.

    These are reconstructed or verified server-side.

    The only optional evaluation intervention in v0.1 is one
    explicit Decision Threshold contract.

    No threshold search or optimization is performed.

    The Summary must never:
    - train;
    - refit;
    - re-rank models;
    - override an upstream selection decision.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    decision_threshold: (
        MLDecisionThresholdContract
        |
        None
    ) = None

    method: Literal[
        "trusted_model_evaluation_summary"
    ] = "trusted_model_evaluation_summary"

    evaluation_scope: Literal[
        "persisted_model_holdout"
    ] = "persisted_model_holdout"

    evidence_policy: Literal[
        "server_reconstructed_only"
    ] = "server_reconstructed_only"

    selection_policy: Literal[
        "preserve_upstream_selection_only"
    ] = "preserve_upstream_selection_only"

    explainability_policy: Literal[
        "default_permutation_importance_v0.1"
    ] = "default_permutation_importance_v0.1"

    threshold_policy: Literal[
        "explicit_requested_threshold_only"
    ] = "explicit_requested_threshold_only"

    rule_version: Literal[
        "ml_model_evaluation_summary_v0.1"
    ] = ML_MODEL_EVALUATION_SUMMARY_RULE_VERSION

    @model_validator(mode="after")
    def revalidate_nested_contracts(
        self,
    ) -> "MLModelEvaluationSummaryContract":

        if self.decision_threshold is not None:

            MLDecisionThresholdContract.model_validate(
                self
                .decision_threshold
                .model_dump(
                    mode="python"
                )
            )

        return self


# ============================================================
# VERIFIED SELECTION EVIDENCE
# ============================================================


class MLModelSelectionEvidence(
    BaseModel
):
    """
    Structured evidence describing how the evaluated Model
    Artifact became the model under review.

    This model does not itself select anything.

    standalone_model
        No upstream server-owned selection result is available.

    model_comparison
        The artifact was the deterministic rank-1 result of a
        Model Comparison. Its selection metric is measured on the
        shared final holdout.

    tuned_model_promotion
        The artifact was promoted from the deterministic rank-1
        Hyperparameter Tuning candidate. Its selection metric
        belongs to the inner cross-validation surface and is
        intentionally distinct from final holdout performance.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    source: MLModelSelectionSource

    status: MLModelSelectionStatus

    rank: (
        int
        |
        None
    ) = Field(
        default=None,
        ge=1,
        strict=True,
    )

    selection_policy: (
        MLModelSelectionPolicy
        |
        None
    ) = None

    primary_metric: (
        MLModelEvaluationPrimaryMetric
        |
        None
    ) = None

    primary_metric_value: (
        float
        |
        None
    ) = None

    metric_scope: MLModelSelectionMetricScope

    rule_version: Literal[
        "ml_model_selection_evidence_v0.1"
    ] = ML_MODEL_SELECTION_EVIDENCE_RULE_VERSION

    @field_validator(
        "primary_metric_value",
        mode="before",
    )
    @classmethod
    def validate_primary_metric_value(
        cls,
        value: object,
    ) -> (
        float
        |
        None
    ):

        if value is None:
            return None

        return _finite_float(
            value,
            field_name="primary_metric_value",
        )

    @model_validator(mode="after")
    def validate_selection_evidence(
        self,
    ) -> "MLModelSelectionEvidence":

        if self.source == "standalone_model":

            if (
                self.status
                !=
                "selection_not_available"
            ):
                raise ValueError(
                    (
                        "Standalone model selection status "
                        "must be selection_not_available."
                    )
                )

            if self.rank is not None:
                raise ValueError(
                    (
                        "Standalone model cannot claim "
                        "a selection rank."
                    )
                )

            if self.selection_policy is not None:
                raise ValueError(
                    (
                        "Standalone model cannot claim "
                        "a selection policy."
                    )
                )

            if self.primary_metric is not None:
                raise ValueError(
                    (
                        "Standalone model cannot claim "
                        "a selection metric."
                    )
                )

            if (
                self.primary_metric_value
                is not None
            ):
                raise ValueError(
                    (
                        "Standalone model cannot claim "
                        "a selection metric value."
                    )
                )

            if self.metric_scope != "not_available":
                raise ValueError(
                    (
                        "Standalone model metric_scope "
                        "must be not_available."
                    )
                )

            return self

        # ----------------------------------------------------
        # VERIFIED UPSTREAM SELECTION
        # ----------------------------------------------------

        if self.status != "verified_selected":
            raise ValueError(
                (
                    "Upstream selection evidence must "
                    "have status verified_selected."
                )
            )

        if self.rank != 1:
            raise ValueError(
                (
                    "Model Evaluation Summary v0.1 accepts "
                    "only verified rank-1 selection evidence."
                )
            )

        if self.primary_metric is None:
            raise ValueError(
                (
                    "Verified selection evidence requires "
                    "a primary metric."
                )
            )

        if self.primary_metric_value is None:
            raise ValueError(
                (
                    "Verified selection evidence requires "
                    "a primary metric value."
                )
            )

        if self.source == "model_comparison":

            expected_policy = (
                "regression_rmse_v0.1"
                if self.primary_metric == "rmse"
                else
                "classification_f1_macro_v0.1"
            )

            if (
                self.selection_policy
                !=
                expected_policy
            ):
                raise ValueError(
                    (
                        "Model Comparison selection policy "
                        "does not match the primary metric."
                    )
                )

            if self.metric_scope != "final_holdout":
                raise ValueError(
                    (
                        "Model Comparison selection metric "
                        "must belong to the final holdout."
                    )
                )

            return self

        # ----------------------------------------------------
        # TUNED MODEL PROMOTION
        # ----------------------------------------------------

        if self.selection_policy != "rank_1_only":
            raise ValueError(
                (
                    "Tuned Model Promotion selection policy "
                    "must be rank_1_only."
                )
            )

        if (
            self.metric_scope
            !=
            "inner_cross_validation"
        ):
            raise ValueError(
                (
                    "Tuned Model Promotion selection metric "
                    "must belong to inner cross-validation."
                )
            )

        return self


# ============================================================
# FINAL STRUCTURED SUMMARY
# ============================================================


class MLModelEvaluationSummaryResult(
    BaseModel
):
    """
    Privacy-minimal deterministic evaluation bundle for one
    already persisted trusted Model Artifact.

    It intentionally exposes aggregate evidence only.

    No raw:
    - rows;
    - y_true;
    - predictions;
    - probabilities;
    - decision scores;
    - model bytes;
    - model paths;
    - estimator state;
    - Training Contract

    crosses this result boundary.
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

    problem_type: MLModelEvaluationProblemType

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

    train_rows: int = Field(
        gt=0,
        strict=True,
    )

    test_rows: int = Field(
        gt=0,
        strict=True,
    )

    summary_contract: (
        MLModelEvaluationSummaryContract
    )

    metrics: dict[
        str,
        float,
    ]

    baseline: (
        MLBaselineEvaluationResult
    )

    baseline_comparison: (
        MLBaselineComparisonResult
    )

    selection_evidence: (
        MLModelSelectionEvidence
    )

    classification_diagnostics: (
        MLClassificationDiagnosticsResult
        |
        None
    ) = None

    decision_threshold_evaluation: (
        MLDecisionThresholdResult
        |
        None
    ) = None

    explainability: (
        MLModelExplainabilityResult
    )

    limitations: list[
        MLModelEvaluationLimitation
    ] = Field(
        min_length=3,
        max_length=5,
    )

    evaluation_status: Literal[
        "complete"
    ] = "complete"

    method: Literal[
        "trusted_model_evaluation_summary"
    ] = "trusted_model_evaluation_summary"

    rule_version: Literal[
        "ml_model_evaluation_summary_v0.1"
    ] = ML_MODEL_EVALUATION_SUMMARY_RULE_VERSION

    # ========================================================
    # TEXT / PROVENANCE
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

        return _required_text(
            value,
            field_name=info.field_name,
        )

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
            .fullmatch(normalized)
            is None
        ):
            raise ValueError(
                (
                    "experiment_id must match "
                    "experiment:<32 lowercase hex characters>."
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

        if (
            SHA256_PATTERN
            .fullmatch(normalized)
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
    # METRICS
    # ========================================================

    @field_validator(
        "metrics",
        mode="before",
    )
    @classmethod
    def validate_metrics(
        cls,
        value: object,
    ) -> dict[
        str,
        float,
    ]:

        if not isinstance(value, dict):
            raise ValueError(
                "metrics must be an object."
            )

        if not value:
            raise ValueError(
                "metrics cannot be empty."
            )

        normalized: dict[
            str,
            float
        ] = {}

        for (
            raw_name,
            raw_value,
        ) in value.items():

            metric_name = _required_text(
                raw_name,
                field_name="metric_name",
            )

            if metric_name in normalized:
                raise ValueError(
                    (
                        "Metric names must remain unique "
                        "after normalization."
                    )
                )

            normalized[
                metric_name
            ] = _finite_float(
                raw_value,
                field_name=(
                    f"metrics.{metric_name}"
                ),
            )

        return normalized

    # ========================================================
    # CROSS-EVIDENCE CONSISTENCY
    # ========================================================

    @model_validator(mode="after")
    def validate_summary_consistency(
        self,
    ) -> "MLModelEvaluationSummaryResult":

        # ----------------------------------------------------
        # FORCE NESTED REVALIDATION
        # ----------------------------------------------------

        summary_contract = (
            MLModelEvaluationSummaryContract
            .model_validate(
                self
                .summary_contract
                .model_dump(
                    mode="python"
                )
            )
        )

        baseline = (
            MLBaselineEvaluationResult
            .model_validate(
                self
                .baseline
                .model_dump(
                    mode="python"
                )
            )
        )

        baseline_comparison = (
            MLBaselineComparisonResult
            .model_validate(
                self
                .baseline_comparison
                .model_dump(
                    mode="python"
                )
            )
        )

        selection_evidence = (
            MLModelSelectionEvidence
            .model_validate(
                self
                .selection_evidence
                .model_dump(
                    mode="python"
                )
            )
        )

        explainability = (
            MLModelExplainabilityResult
            .model_validate(
                self
                .explainability
                .model_dump(
                    mode="python"
                )
            )
        )

        diagnostics = None

        if (
            self.classification_diagnostics
            is not None
        ):
            diagnostics = (
                MLClassificationDiagnosticsResult
                .model_validate(
                    self
                    .classification_diagnostics
                    .model_dump(
                        mode="python"
                    )
                )
            )

        threshold_result = None

        if (
            self.decision_threshold_evaluation
            is not None
        ):
            threshold_result = (
                MLDecisionThresholdResult
                .model_validate(
                    self
                    .decision_threshold_evaluation
                    .model_dump(
                        mode="python"
                    )
                )
            )

        # ----------------------------------------------------
        # PRIMARY METRIC / METRIC SURFACE
        # ----------------------------------------------------

        primary_metric = (
            model_evaluation_primary_metric(
                problem_type=
                    self.problem_type
            )
        )

        required_metrics = (
            {
                "mae",
                "rmse",
                "r2",
                "median_absolute_error",
                "explained_variance",
            }
            if self.problem_type
            ==
            "regression"
            else
            {
                "accuracy",
                "f1_macro",
                "precision_macro",
                "recall_macro",
                "balanced_accuracy",
            }
        )

        missing_metrics = (
            required_metrics
            -
            set(self.metrics)
        )

        if missing_metrics:
            raise ValueError(
                (
                    "Model Evaluation metrics are missing "
                    "required persisted metrics: "
                    f"{sorted(missing_metrics)}"
                )
            )

        model_primary_value = float(
            self.metrics[
                primary_metric
            ]
        )

        # ----------------------------------------------------
        # BASELINE BINDING
        # ----------------------------------------------------

        if (
            baseline.problem_type
            !=
            self.problem_type
        ):
            raise ValueError(
                (
                    "Baseline problem_type does not match "
                    "the evaluated Model Artifact."
                )
            )

        if (
            baseline.primary_metric
            !=
            primary_metric
        ):
            raise ValueError(
                (
                    "Baseline primary metric does not "
                    "match the Model Evaluation policy."
                )
            )

        if (
            baseline.train_rows
            !=
            self.train_rows
            or
            baseline.test_rows
            !=
            self.test_rows
        ):
            raise ValueError(
                (
                    "Baseline holdout shape does not match "
                    "the evaluated Model Artifact."
                )
            )

        if (
            baseline_comparison.problem_type
            !=
            self.problem_type
        ):
            raise ValueError(
                (
                    "Baseline comparison problem_type "
                    "does not match the evaluated model."
                )
            )

        if (
            baseline_comparison.primary_metric
            !=
            primary_metric
        ):
            raise ValueError(
                (
                    "Baseline comparison primary metric "
                    "does not match the evaluation policy."
                )
            )

        _assert_close(
            actual=(
                baseline_comparison
                .model_primary_metric_value
            ),
            expected=model_primary_value,
            field_name=(
                "baseline_comparison."
                "model_primary_metric_value"
            ),
        )

        _assert_close(
            actual=(
                baseline_comparison
                .baseline_primary_metric_value
            ),
            expected=float(
                baseline.metrics[
                    primary_metric
                ]
            ),
            field_name=(
                "baseline_comparison."
                "baseline_primary_metric_value"
            ),
        )

        # ----------------------------------------------------
        # SELECTION EVIDENCE
        # ----------------------------------------------------

        if (
            selection_evidence.primary_metric
            is not None
            and
            selection_evidence.primary_metric
            !=
            primary_metric
        ):
            raise ValueError(
                (
                    "Selection evidence primary metric "
                    "does not match problem_type."
                )
            )

        if (
            selection_evidence.source
            ==
            "model_comparison"
        ):
            expected_policy = (
                "regression_rmse_v0.1"
                if self.problem_type
                ==
                "regression"
                else
                "classification_f1_macro_v0.1"
            )

            if (
                selection_evidence.selection_policy
                !=
                expected_policy
            ):
                raise ValueError(
                    (
                        "Model Comparison selection policy "
                        "does not match problem_type."
                    )
                )

            _assert_close(
                actual=float(
                    selection_evidence
                    .primary_metric_value
                ),
                expected=model_primary_value,
                field_name=(
                    "selection_evidence."
                    "primary_metric_value"
                ),
            )

        # ----------------------------------------------------
        # EXPLAINABILITY IDENTITY
        # ----------------------------------------------------

        identity_pairs = (
            (
                "workflow_id",
                explainability.workflow_id,
                self.workflow_id,
            ),
            (
                "dataset_id",
                explainability.dataset_id,
                self.dataset_id,
            ),
            (
                "model_id",
                explainability.model_id,
                self.model_id,
            ),
            (
                "experiment_id",
                explainability.experiment_id,
                self.experiment_id,
            ),
            (
                "problem_type",
                explainability.problem_type,
                self.problem_type,
            ),
            (
                "estimator_key",
                explainability.estimator_key,
                self.estimator_key,
            ),
            (
                "preparation_session_revision",
                explainability
                .preparation_session_revision,
                self
                .preparation_session_revision,
            ),
            (
                "training_contract_sha256",
                explainability
                .training_contract_sha256,
                self
                .training_contract_sha256,
            ),
        )

        for (
            field_name,
            actual,
            expected,
        ) in identity_pairs:

            if actual != expected:
                raise ValueError(
                    (
                        "Explainability evidence identity "
                        "does not match Model Evaluation "
                        f"Summary. field={field_name}"
                    )
                )

        if (
            explainability.evaluation_rows
            !=
            self.test_rows
        ):
            raise ValueError(
                (
                    "Explainability evaluation rows do "
                    "not match the persisted holdout."
                )
            )

        # ----------------------------------------------------
        # PROBLEM-SPECIFIC EVIDENCE
        # ----------------------------------------------------

        requested_threshold = (
            summary_contract
            .decision_threshold
        )

        if self.problem_type == "regression":

            if diagnostics is not None:
                raise ValueError(
                    (
                        "Regression Model Evaluation cannot "
                        "contain classification diagnostics."
                    )
                )

            if requested_threshold is not None:
                raise ValueError(
                    (
                        "Regression Model Evaluation cannot "
                        "request a Decision Threshold."
                    )
                )

            if threshold_result is not None:
                raise ValueError(
                    (
                        "Regression Model Evaluation cannot "
                        "contain Decision Threshold evidence."
                    )
                )

        else:

            if diagnostics is None:
                raise ValueError(
                    (
                        "Classification Model Evaluation "
                        "requires Classification Diagnostics."
                    )
                )

            diagnostic_identity_pairs = (
                (
                    "workflow_id",
                    diagnostics.workflow_id,
                    self.workflow_id,
                ),
                (
                    "dataset_id",
                    diagnostics.dataset_id,
                    self.dataset_id,
                ),
                (
                    "model_id",
                    diagnostics.model_id,
                    self.model_id,
                ),
                (
                    "experiment_id",
                    diagnostics.experiment_id,
                    self.experiment_id,
                ),
                (
                    "target_column",
                    diagnostics.target_column,
                    self.target_column,
                ),
                (
                    "estimator_key",
                    diagnostics.estimator_key,
                    self.estimator_key,
                ),
                (
                    "preparation_session_revision",
                    diagnostics
                    .preparation_session_revision,
                    self
                    .preparation_session_revision,
                ),
                (
                    "training_contract_sha256",
                    diagnostics
                    .training_contract_sha256,
                    self
                    .training_contract_sha256,
                ),
            )

            for (
                field_name,
                actual,
                expected,
            ) in diagnostic_identity_pairs:

                if actual != expected:
                    raise ValueError(
                        (
                            "Classification Diagnostics "
                            "identity does not match Model "
                            "Evaluation Summary. "
                            f"field={field_name}"
                        )
                    )

            if (
                diagnostics.evaluation_rows
                !=
                self.test_rows
            ):
                raise ValueError(
                    (
                        "Classification Diagnostics rows "
                        "do not match the persisted holdout."
                    )
                )

            diagnostic_metric_pairs = (
                (
                    "accuracy",
                    diagnostics.accuracy,
                ),
                (
                    "balanced_accuracy",
                    diagnostics
                    .balanced_accuracy,
                ),
                (
                    "precision_macro",
                    diagnostics
                    .macro_average
                    .precision,
                ),
                (
                    "recall_macro",
                    diagnostics
                    .macro_average
                    .recall,
                ),
                (
                    "f1_macro",
                    diagnostics
                    .macro_average
                    .f1,
                ),
            )

            for (
                metric_name,
                diagnostic_value,
            ) in diagnostic_metric_pairs:

                _assert_close(
                    actual=float(
                        diagnostic_value
                    ),
                    expected=float(
                        self.metrics[
                            metric_name
                        ]
                    ),
                    field_name=(
                        "classification_diagnostics."
                        f"{metric_name}"
                    ),
                )

            if requested_threshold is None:

                if threshold_result is not None:
                    raise ValueError(
                        (
                            "Decision Threshold evidence "
                            "cannot be present when no "
                            "threshold was requested."
                        )
                    )

            else:

                if threshold_result is None:
                    raise ValueError(
                        (
                            "Requested Decision Threshold "
                            "requires threshold evidence."
                        )
                    )

                if diagnostics.class_count != 2:
                    raise ValueError(
                        (
                            "Decision Threshold v0.1 requires "
                            "binary Classification Diagnostics."
                        )
                    )

                threshold_identity_pairs = (
                    (
                        "workflow_id",
                        threshold_result.workflow_id,
                        self.workflow_id,
                    ),
                    (
                        "dataset_id",
                        threshold_result.dataset_id,
                        self.dataset_id,
                    ),
                    (
                        "model_id",
                        threshold_result.model_id,
                        self.model_id,
                    ),
                    (
                        "experiment_id",
                        threshold_result.experiment_id,
                        self.experiment_id,
                    ),
                    (
                        "target_column",
                        threshold_result.target_column,
                        self.target_column,
                    ),
                    (
                        "estimator_key",
                        threshold_result.estimator_key,
                        self.estimator_key,
                    ),
                    (
                        "preparation_session_revision",
                        threshold_result
                        .preparation_session_revision,
                        self
                        .preparation_session_revision,
                    ),
                    (
                        "training_contract_sha256",
                        threshold_result
                        .training_contract_sha256,
                        self
                        .training_contract_sha256,
                    ),
                )

                for (
                    field_name,
                    actual,
                    expected,
                ) in threshold_identity_pairs:

                    if actual != expected:
                        raise ValueError(
                            (
                                "Decision Threshold evidence "
                                "identity does not match Model "
                                "Evaluation Summary. "
                                f"field={field_name}"
                            )
                        )

                if (
                    threshold_result.evaluation_rows
                    !=
                    self.test_rows
                ):
                    raise ValueError(
                        (
                            "Decision Threshold rows do not "
                            "match the persisted holdout."
                        )
                    )

                _assert_close(
                    actual=threshold_result.threshold,
                    expected=(
                        requested_threshold.threshold
                    ),
                    field_name=(
                        "decision_threshold_evaluation."
                        "threshold"
                    ),
                )

                expected_threshold_labels = [
                    threshold_result
                    .negative_class_label,
                    threshold_result
                    .positive_class_label,
                ]

                if (
                    expected_threshold_labels
                    !=
                    diagnostics.class_labels
                ):
                    raise ValueError(
                        (
                            "Decision Threshold class ordering "
                            "does not match Classification "
                            "Diagnostics estimator class order."
                        )
                    )

        # ----------------------------------------------------
        # LIMITATIONS ARE SERVER-DERIVED
        # ----------------------------------------------------

        expected_limitations = (
            expected_model_evaluation_limitations(
                problem_type=
                    self.problem_type,

                selection_source=(
                    selection_evidence.source
                ),

                threshold_requested=(
                    requested_threshold
                    is not None
                ),
            )
        )

        if (
            self.limitations
            !=
            expected_limitations
        ):
            raise ValueError(
                (
                    "Model Evaluation limitations must "
                    "match the deterministic server policy."
                )
            )

        return self
