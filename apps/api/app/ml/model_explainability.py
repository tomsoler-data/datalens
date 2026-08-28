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


ML_MODEL_EXPLAINABILITY_RULE_VERSION = (
    "ml_model_explainability_v0.1"
)


# ============================================================
# TYPES
# ============================================================


MLExplainabilityMethod = Literal[
    "permutation_importance",
]


MLExplainabilityScoring = Literal[
    "neg_root_mean_squared_error",
    "f1_macro",
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


# ============================================================
# CONFIGURATION CONTRACT
# ============================================================


class MLModelExplainabilityContract(
    BaseModel
):
    """
    Deterministic Model Explainability configuration.

    This contract is deliberately separate from MLTrainingContract.

    MLTrainingContract continues to represent the exact model
    training authority and therefore keeps its canonical
    provenance fingerprint unchanged.

    Explainability v0.1 supports only global permutation feature
    importance evaluated on the deterministic holdout test set of
    an already persisted trusted Model Artifact.

    The scoring policy is server-owned and cannot be selected by
    callers:
    - regression -> neg_root_mean_squared_error;
    - classification -> f1_macro.

    v0.1 intentionally excludes:
    - SHAP;
    - local explanations;
    - partial dependence;
    - native estimator coefficient/importances;
    - training-set importance.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    method: Literal[
        "permutation_importance"
    ] = "permutation_importance"


    n_repeats: int = Field(
        default=10,
        ge=2,
        le=50,
    )


    random_seed: int = Field(
        default=42,
        ge=0,
        le=2_147_483_647,
    )


    rule_version: Literal[
        "ml_model_explainability_v0.1"
    ] = ML_MODEL_EXPLAINABILITY_RULE_VERSION


# ============================================================
# FEATURE IMPORTANCE
# ============================================================


class MLFeatureImportanceResult(
    BaseModel
):
    """
    One deterministic global feature-importance result.

    importance_mean is deliberately signed.

    Positive:
        model performance degraded when the feature was shuffled.

    Zero:
        no average measurable change.

    Negative:
        shuffled values improved the scorer on average.

    Negative values are therefore valid and MUST NOT be clipped.

    importance_std is the population standard deviation returned
    across deterministic permutation repeats.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


    feature_name: str = Field(
        min_length=1,
    )


    rank: int = Field(
        ge=1,
    )


    importance_mean: float


    importance_std: float = Field(
        ge=0.0,
    )


    @field_validator(
        "feature_name",
        mode="before",
    )
    @classmethod
    def normalize_feature_name(
        cls,
        value: object,
    ) -> str:

        return (
            _required_text(
                value,
                field_name=
                    "feature_name",
            )
        )


    @field_validator(
        "importance_mean",
        "importance_std",
    )
    @classmethod
    def validate_finite_importance(
        cls,
        value: float,
    ) -> float:

        normalized = float(
            value
        )


        if not math.isfinite(
            normalized
        ):
            raise ValueError(
                (
                    "Feature importance values "
                    "must be finite."
                )
            )


        return normalized


# ============================================================
# EXPLAINABILITY RESULT
# ============================================================


class MLModelExplainabilityResult(
    BaseModel
):
    """
    Privacy-minimal deterministic global explanation for one
    persisted DataLens Model Artifact.

    The result is bound to:
    - workflow;
    - dataset;
    - model;
    - experiment;
    - Preparation revision;
    - canonical ML Training Contract SHA-256.

    It contains no:
    - raw rows;
    - predictions;
    - permuted observations;
    - estimator bytes;
    - model filesystem path.

    Ranking policy v0.1:
    1. higher importance_mean first;
    2. lower importance_std first;
    3. lexical feature_name final tie-breaker.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
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
        "regression",
        "classification",
    ]


    estimator_key: str = Field(
        min_length=1,
    )


    preparation_session_revision: int = Field(
        ge=0,
    )


    training_contract_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


    method: Literal[
        "permutation_importance"
    ] = "permutation_importance"


    scoring: MLExplainabilityScoring


    n_repeats: int = Field(
        ge=2,
        le=50,
    )


    random_seed: int = Field(
        ge=0,
        le=2_147_483_647,
    )


    evaluation_rows: int = Field(
        gt=0,
    )


    feature_importances: list[
        MLFeatureImportanceResult
    ] = Field(
        min_length=1,
    )


    rule_version: Literal[
        "ml_model_explainability_v0.1"
    ] = ML_MODEL_EXPLAINABILITY_RULE_VERSION


    # ========================================================
    # TEXT
    # ========================================================


    @field_validator(
        "workflow_id",
        "dataset_id",
        "model_id",
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
    # CONTRACT SHA
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
    # STRUCTURE
    # ========================================================


    @model_validator(
        mode="after"
    )
    def validate_structure(
        self,
    ) -> "MLModelExplainabilityResult":

        expected_scoring = (
            explainability_scoring(
                problem_type=
                    self.problem_type
            )
        )


        if (
            self.scoring
            !=
            expected_scoring
        ):
            raise ValueError(
                (
                    "Explainability scoring does not "
                    "match problem_type. "
                    f"problem_type={self.problem_type}, "
                    f"scoring={self.scoring}"
                )
            )


        feature_names = [
            item.feature_name

            for item
            in self.feature_importances
        ]


        if (
            len(
                feature_names
            )
            !=
            len(
                set(
                    feature_names
                )
            )
        ):
            raise ValueError(
                (
                    "Explainability feature names "
                    "must be unique."
                )
            )


        ranks = [
            item.rank

            for item
            in self.feature_importances
        ]


        expected_ranks = list(
            range(
                1,
                len(
                    self.feature_importances
                )
                +
                1,
            )
        )


        if (
            ranks
            !=
            expected_ranks
        ):
            raise ValueError(
                (
                    "Explainability ranks must be "
                    "contiguous and ordered from 1."
                )
            )


        expected_order = sorted(
            self.feature_importances,
            key=lambda item: (
                -
                item.importance_mean,
                item.importance_std,
                item.feature_name,
            ),
        )


        expected_feature_names = [
            item.feature_name

            for item
            in expected_order
        ]


        if (
            feature_names
            !=
            expected_feature_names
        ):
            raise ValueError(
                (
                    "Explainability feature_importances "
                    "do not follow the deterministic "
                    "ranking policy."
                )
            )


        return self


# ============================================================
# SERVER-OWNED SCORING
# ============================================================


def explainability_scoring(
    *,
    problem_type: str,
) -> MLExplainabilityScoring:
    """
    Return the server-owned permutation-importance scorer.

    Regression intentionally aligns with the existing DataLens
    model-comparison RMSE objective by using sklearn's
    neg_root_mean_squared_error scorer.

    Classification intentionally aligns with the existing
    Model Comparison primary objective through f1_macro.

    permutation_importance computes the decrease in scorer value,
    so higher positive importance means greater performance loss
    when the feature is shuffled.
    """

    if (
        problem_type
        ==
        "regression"
    ):
        return (
            "neg_root_mean_squared_error"
        )


    if (
        problem_type
        ==
        "classification"
    ):
        return (
            "f1_macro"
        )


    raise ValueError(
        (
            "Unsupported problem type for "
            "Model Explainability v0.1. "
            f"problem_type={problem_type}"
        )
    )
