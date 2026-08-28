from __future__ import annotations


import math


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


ML_CROSS_VALIDATION_RULE_VERSION = (
    "ml_cross_validation_v0.1"
)


# ============================================================
# TYPES
# ============================================================


MLCrossValidationStrategy = Literal[
    "k_fold",
    "stratified_k_fold",
]


# ============================================================
# CONFIGURATION CONTRACT
# ============================================================


class MLCrossValidationContract(
    BaseModel
):
    """
    Deterministic Cross-Validation configuration.

    This contract is deliberately separate from MLTrainingContract.

    MLTrainingContract continues to describe the persisted holdout
    training execution and therefore keeps its existing canonical
    provenance fingerprint unchanged.

    Cross-validation is an evaluation layer around that training
    contract.

    The fold strategy is server-owned:
    - regression -> KFold;
    - classification -> StratifiedKFold.

    Callers cannot select an incompatible strategy.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    folds: int = Field(
        default=5,
        ge=2,
        le=20,
    )


    shuffle: bool = True


    random_seed: int = Field(
        default=42,
        ge=0,
        le=2_147_483_647,
    )


    rule_version: Literal[
        "ml_cross_validation_v0.1"
    ] = ML_CROSS_VALIDATION_RULE_VERSION


# ============================================================
# FOLD RESULT
# ============================================================


class MLCrossValidationFoldResult(
    BaseModel
):
    """
    Privacy-minimal result for one validation fold.

    Raw rows and predictions must never be persisted here.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    fold_index: int = Field(
        ge=1,
    )


    train_rows: int = Field(
        gt=0,
    )


    validation_rows: int = Field(
        gt=0,
    )


    metrics: dict[
        str,
        float,
    ]


    @field_validator(
        "metrics"
    )
    @classmethod
    def validate_metrics(
        cls,
        value: dict[
            str,
            float,
        ],
    ) -> dict[
        str,
        float,
    ]:

        if not value:
            raise ValueError(
                (
                    "Cross-validation fold metrics "
                    "cannot be empty."
                )
            )


        normalized: dict[
            str,
            float,
        ] = {}


        for (
            raw_name,
            raw_value,
        ) in value.items():

            name = str(
                raw_name
            ).strip()


            if not name:
                raise ValueError(
                    (
                        "Cross-validation metric "
                        "names cannot be empty."
                    )
                )


            if isinstance(
                raw_value,
                bool,
            ):
                raise ValueError(
                    (
                        "Cross-validation metric "
                        "values cannot be boolean."
                    )
                )


            metric_value = float(
                raw_value
            )


            if not math.isfinite(
                metric_value
            ):
                raise ValueError(
                    (
                        "Cross-validation metric "
                        "values must be finite."
                    )
                )


            normalized[
                name
            ] = metric_value


        return normalized


# ============================================================
# METRIC SUMMARY
# ============================================================


class MLCrossValidationMetricSummary(
    BaseModel
):
    """
    Aggregate statistic for one metric across validation folds.

    std is the population standard deviation of fold scores
    (ddof=0) in Cross-Validation v0.1.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


    mean: float


    std: float = Field(
        ge=0.0,
    )


# ============================================================
# EVALUATION RESULT
# ============================================================


class MLCrossValidationEvaluationResult(
    BaseModel
):
    """
    Server-owned deterministic Cross-Validation result.

    It contains only configuration/provenance identifiers,
    fold sizes and aggregate evaluation metrics.

    It intentionally contains no raw observations,
    predictions or fitted estimator bytes.
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


    strategy: MLCrossValidationStrategy


    folds: int = Field(
        ge=2,
        le=20,
    )


    shuffle: bool


    random_seed: int = Field(
        ge=0,
        le=2_147_483_647,
    )


    fold_results: list[
        MLCrossValidationFoldResult
    ] = Field(
        min_length=2,
    )


    metric_summary: dict[
        str,
        MLCrossValidationMetricSummary,
    ]


    rule_version: Literal[
        "ml_cross_validation_v0.1"
    ] = ML_CROSS_VALIDATION_RULE_VERSION


    @field_validator(
        "training_contract_sha256"
    )
    @classmethod
    def validate_training_contract_sha256(
        cls,
        value: str,
    ) -> str:

        normalized = (
            value
            .strip()
            .lower()
        )


        if len(
            normalized
        ) != 64:
            raise ValueError(
                (
                    "training_contract_sha256 "
                    "must contain 64 hexadecimal "
                    "characters."
                )
            )


        if any(
            character
            not in
            "0123456789abcdef"

            for character
            in normalized
        ):
            raise ValueError(
                (
                    "training_contract_sha256 "
                    "must be hexadecimal."
                )
            )


        return normalized


    @model_validator(
        mode="after"
    )
    def validate_structure(
        self,
    ) -> (
        "MLCrossValidationEvaluationResult"
    ):

        expected_strategy = (
            "k_fold"

            if (
                self.problem_type
                ==
                "regression"
            )

            else
            "stratified_k_fold"
        )


        if (
            self.strategy
            !=
            expected_strategy
        ):
            raise ValueError(
                (
                    "Cross-validation strategy does "
                    "not match problem_type. "
                    f"problem_type={self.problem_type}, "
                    f"strategy={self.strategy}"
                )
            )


        if (
            len(
                self.fold_results
            )
            !=
            self.folds
        ):
            raise ValueError(
                (
                    "Cross-validation result must "
                    "contain exactly one fold result "
                    "per configured fold."
                )
            )


        fold_indices = [
            fold.fold_index

            for fold
            in self.fold_results
        ]


        expected_indices = list(
            range(
                1,
                self.folds
                +
                1,
            )
        )


        if (
            fold_indices
            !=
            expected_indices
        ):
            raise ValueError(
                (
                    "Cross-validation fold indices "
                    "must be contiguous and ordered "
                    "from 1 to folds."
                )
            )


        first_metric_names = set(
            self
            .fold_results[
                0
            ]
            .metrics
        )


        for fold in (
            self.fold_results
        ):
            if (
                set(
                    fold.metrics
                )
                !=
                first_metric_names
            ):
                raise ValueError(
                    (
                        "Every cross-validation fold "
                        "must expose the same metric "
                        "surface."
                    )
                )


        if (
            set(
                self.metric_summary
            )
            !=
            first_metric_names
        ):
            raise ValueError(
                (
                    "Cross-validation metric summary "
                    "must exactly match the fold "
                    "metric surface."
                )
            )


        return self


# ============================================================
# SERVER-OWNED STRATEGY
# ============================================================


def cross_validation_strategy(
    *,
    problem_type: str,
) -> MLCrossValidationStrategy:

    if (
        problem_type
        ==
        "regression"
    ):
        return (
            "k_fold"
        )


    if (
        problem_type
        ==
        "classification"
    ):
        return (
            "stratified_k_fold"
        )


    raise ValueError(
        (
            "Unsupported problem type for "
            "Cross-Validation v0.1. "
            f"problem_type={problem_type}"
        )
    )
