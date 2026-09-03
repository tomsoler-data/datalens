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


from app.ml.estimator_contracts import (
    MLEstimatorHyperparameters,
    MLLinearRegressionHyperparameters,
    MLLogisticRegressionHyperparameters,
    MLRandomForestClassifierHyperparameters,
    MLRandomForestRegressorHyperparameters,
    MLRidgeRegressionHyperparameters,
    estimator_problem_type,
)


# ============================================================
# VERSION
# ============================================================


ML_HYPERPARAMETER_TUNING_RULE_VERSION = (
    "ml_hyperparameter_tuning_v0.1"
)


# ============================================================
# TYPES
# ============================================================


MLHyperparameterSearchStrategy = Literal[
    "server_owned_grid",
]


MLHyperparameterValidationStrategy = Literal[
    "k_fold",
    "stratified_k_fold",
    "group_k_fold",
    "stratified_group_k_fold",
    "time_series_split",
]


MLHyperparameterPrimaryMetric = Literal[
    "rmse",
    "f1_macro",
]


MLHyperparameterMetricDirection = Literal[
    "minimize",
    "maximize",
]


# ============================================================
# VALIDATION
# ============================================================


SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
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
# SEARCH CONFIGURATION
# ============================================================


class MLHyperparameterSearchContract(
    BaseModel
):
    """
    Deterministic Hyperparameter Tuning configuration.

    This contract is deliberately separate from MLTrainingContract.

    The base MLTrainingContract continues to describe the
    server-validatable holdout training request and therefore
    retains its existing canonical provenance SHA-256.

    Hyperparameter Tuning v0.1:

    - uses one fixed server-owned finite candidate grid;
    - performs inner Cross-Validation on the OUTER TRAIN split;
    - never exposes the OUTER holdout test split to tuning;
    - never accepts arbitrary sklearn parameter grids;
    - never accepts arbitrary scorers;
    - does not persist candidate models.

    Validation strategy is server-owned from problem_type and
    Training Contract split semantics.

    Primary metric remains server-owned from problem_type.

    Neither is caller-configurable.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    search_strategy: Literal[
        "server_owned_grid"
    ] = "server_owned_grid"


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
        "ml_hyperparameter_tuning_v0.1"
    ] = ML_HYPERPARAMETER_TUNING_RULE_VERSION


# ============================================================
# METRIC SUMMARY
# ============================================================


class MLHyperparameterMetricSummary(
    BaseModel
):
    """
    Aggregate inner-CV statistic for one metric.

    std uses population standard deviation (ddof=0) in v0.1.
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


    @field_validator(
        "mean",
        "std",
    )
    @classmethod
    def validate_finite_metric(
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
                    "Hyperparameter metric summary "
                    "values must be finite."
                )
            )


        return normalized


# ============================================================
# CANDIDATE RESULT
# ============================================================


class MLHyperparameterCandidateResult(
    BaseModel
):
    """
    Privacy-minimal evaluation for one server-owned candidate.

    candidate_index identifies the candidate's fixed position
    inside the server-owned grid.

    rank identifies its deterministic performance ranking.

    No fold rows, predictions, estimator bytes or fitted
    preprocessing state are exposed.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    candidate_index: int = Field(
        ge=1,
    )


    rank: int = Field(
        ge=1,
    )


    hyperparameters: MLEstimatorHyperparameters


    training_contract_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


    metric_summary: dict[
        str,
        MLHyperparameterMetricSummary,
    ]


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
            SHA256_PATTERN.fullmatch(
                normalized
            )
            is None
        ):
            raise ValueError(
                (
                    "training_contract_sha256 must "
                    "be a lowercase 64-character "
                    "SHA-256 digest."
                )
            )


        return normalized


    @field_validator(
        "metric_summary"
    )
    @classmethod
    def validate_metric_summary(
        cls,
        value: dict[
            str,
            MLHyperparameterMetricSummary,
        ],
    ) -> dict[
        str,
        MLHyperparameterMetricSummary,
    ]:

        if not value:
            raise ValueError(
                (
                    "Hyperparameter candidate "
                    "metric_summary cannot be empty."
                )
            )


        normalized: dict[
            str,
            MLHyperparameterMetricSummary,
        ] = {}


        for (
            raw_name,
            raw_summary,
        ) in value.items():

            name = _required_text(
                raw_name,
                field_name=
                    "metric name",
            )


            if name in normalized:
                raise ValueError(
                    (
                        "Hyperparameter metric names "
                        "must be unique."
                    )
                )


            normalized[
                name
            ] = (
                MLHyperparameterMetricSummary
                .model_validate(
                    raw_summary
                )
            )


        return normalized


# ============================================================
# SERVER-OWNED POLICY
# ============================================================


def hyperparameter_validation_strategy(
    *,
    problem_type: str,
    group_aware: bool = False,
    temporal_aware: bool = False,
) -> MLHyperparameterValidationStrategy:

    if (
        group_aware
        and
        temporal_aware
    ):

        raise ValueError(
            (
                "Group-aware and temporal-aware "
                "Hyperparameter Tuning cannot be "
                "combined in v0.1."
            )
        )


    if (
        problem_type
        not in
        {
            "regression",
            "classification",
        }
    ):

        raise ValueError(
            (
                "Unsupported problem type for "
                "Hyperparameter Tuning v0.1. "
                f"problem_type={problem_type}"
            )
        )


    if temporal_aware:

        return (
            "time_series_split"
        )


    if (
        problem_type
        ==
        "regression"
    ):

        return (
            "group_k_fold"

            if group_aware

            else
            "k_fold"
        )


    return (
        "stratified_group_k_fold"

        if group_aware

        else
        "stratified_k_fold"
    )


def hyperparameter_primary_metric(
    *,
    problem_type: str,
) -> MLHyperparameterPrimaryMetric:

    if (
        problem_type
        ==
        "regression"
    ):
        return "rmse"


    if (
        problem_type
        ==
        "classification"
    ):
        return "f1_macro"


    raise ValueError(
        (
            "Unsupported problem type for "
            "Hyperparameter Tuning v0.1. "
            f"problem_type={problem_type}"
        )
    )


def hyperparameter_metric_direction(
    *,
    problem_type: str,
) -> MLHyperparameterMetricDirection:

    if (
        problem_type
        ==
        "regression"
    ):
        return "minimize"


    if (
        problem_type
        ==
        "classification"
    ):
        return "maximize"


    raise ValueError(
        (
            "Unsupported problem type for "
            "Hyperparameter Tuning v0.1. "
            f"problem_type={problem_type}"
        )
    )


def expected_hyperparameter_metric_names(
    *,
    problem_type: str,
) -> tuple[
    str,
    ...,
]:

    if (
        problem_type
        ==
        "regression"
    ):
        return (
            "mae",
            "rmse",
            "r2",
            "median_absolute_error",
            "explained_variance",
        )


    if (
        problem_type
        ==
        "classification"
    ):
        return (
            "accuracy",
            "f1_macro",
            "precision_macro",
            "recall_macro",
            "balanced_accuracy",
        )


    raise ValueError(
        (
            "Unsupported problem type for "
            "Hyperparameter metric surface. "
            f"problem_type={problem_type}"
        )
    )


# ============================================================
# SERVER-OWNED FINITE GRIDS
# ============================================================


def server_owned_hyperparameter_candidates(
    *,
    estimator_key: str,
) -> tuple[
    MLEstimatorHyperparameters,
    ...,
]:
    """
    Return the complete deterministic v0.1 candidate grid.

    Callers cannot provide or extend this grid.

    The grids are deliberately small so that:
    - tuning remains auditable;
    - CI remains bounded;
    - every candidate can be evaluated exhaustively;
    - no uncontrolled sklearn kwargs cross the DataLens boundary.
    """

    normalized = _required_text(
        estimator_key,
        field_name=
            "estimator_key",
    )


    # ========================================================
    # LINEAR REGRESSION
    # ========================================================

    if (
        normalized
        ==
        "linear_regression"
    ):
        return (
            MLLinearRegressionHyperparameters(
                fit_intercept=True,
            ),
            MLLinearRegressionHyperparameters(
                fit_intercept=False,
            ),
        )


    # ========================================================
    # RIDGE REGRESSION
    # ========================================================

    if (
        normalized
        ==
        "ridge_regression"
    ):
        return (
            MLRidgeRegressionHyperparameters(
                alpha=0.1,
                fit_intercept=True,
            ),
            MLRidgeRegressionHyperparameters(
                alpha=1.0,
                fit_intercept=True,
            ),
            MLRidgeRegressionHyperparameters(
                alpha=10.0,
                fit_intercept=True,
            ),
        )


    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    if (
        normalized
        ==
        "logistic_regression"
    ):
        return (
            MLLogisticRegressionHyperparameters(
                inverse_regularization_strength=
                    0.1,
                fit_intercept=
                    True,
                max_iter=
                    1000,
                class_weight=
                    None,
            ),
            MLLogisticRegressionHyperparameters(
                inverse_regularization_strength=
                    1.0,
                fit_intercept=
                    True,
                max_iter=
                    1000,
                class_weight=
                    None,
            ),
            MLLogisticRegressionHyperparameters(
                inverse_regularization_strength=
                    10.0,
                fit_intercept=
                    True,
                max_iter=
                    1000,
                class_weight=
                    None,
            ),
            MLLogisticRegressionHyperparameters(
                inverse_regularization_strength=
                    1.0,
                fit_intercept=
                    True,
                max_iter=
                    1000,
                class_weight=
                    "balanced",
            ),
        )


    # ========================================================
    # RANDOM FOREST REGRESSOR
    # ========================================================

    if (
        normalized
        ==
        "random_forest_regressor"
    ):
        return (
            MLRandomForestRegressorHyperparameters(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                bootstrap=True,
            ),
            MLRandomForestRegressorHyperparameters(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                bootstrap=True,
            ),
            MLRandomForestRegressorHyperparameters(
                n_estimators=200,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                bootstrap=True,
            ),
            MLRandomForestRegressorHyperparameters(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=2,
                max_features="sqrt",
                bootstrap=True,
            ),
        )


    # ========================================================
    # RANDOM FOREST CLASSIFIER
    # ========================================================

    if (
        normalized
        ==
        "random_forest_classifier"
    ):
        return (
            MLRandomForestClassifierHyperparameters(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                bootstrap=True,
                class_weight=None,
            ),
            MLRandomForestClassifierHyperparameters(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                bootstrap=True,
                class_weight=None,
            ),
            MLRandomForestClassifierHyperparameters(
                n_estimators=200,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                bootstrap=True,
                class_weight=None,
            ),
            MLRandomForestClassifierHyperparameters(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=2,
                max_features="sqrt",
                bootstrap=True,
                class_weight=None,
            ),
            MLRandomForestClassifierHyperparameters(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                bootstrap=True,
                class_weight="balanced",
            ),
        )


    raise ValueError(
        (
            "Unsupported estimator for "
            "Hyperparameter Tuning v0.1. "
            f"estimator_key={normalized}"
        )
    )


# ============================================================
# SEARCH RESULT
# ============================================================


class MLHyperparameterSearchResult(
    BaseModel
):
    """
    Privacy-minimal deterministic Hyperparameter Tuning result.

    The OUTER holdout is created first from MLTrainingContract.

    Only outer_train_rows are eligible for inner Cross-Validation.

    holdout_test_rows are recorded for auditability but are never
    evaluated, scored or inspected during the search.

    candidate_results must contain the complete server-owned grid.

    Deterministic ranking:
    - regression: lower RMSE mean first;
    - classification: higher F1 macro mean first;
    - then lower primary metric std;
    - then lower server-owned candidate_index.
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


    base_training_contract_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


    search_strategy: Literal[
        "server_owned_grid"
    ] = "server_owned_grid"


    validation_strategy: (
        MLHyperparameterValidationStrategy
    )


    primary_metric: (
        MLHyperparameterPrimaryMetric
    )


    metric_direction: (
        MLHyperparameterMetricDirection
    )


    folds: int = Field(
        ge=2,
        le=20,
    )


    shuffle: bool


    random_seed: int = Field(
        ge=0,
        le=2_147_483_647,
    )


    outer_train_rows: int = Field(
        gt=0,
    )


    holdout_test_rows: int = Field(
        gt=0,
    )


    candidate_count: int = Field(
        gt=0,
    )


    best_candidate_index: int = Field(
        ge=1,
    )


    candidate_results: list[
        MLHyperparameterCandidateResult
    ] = Field(
        min_length=1,
    )


    rule_version: Literal[
        "ml_hyperparameter_tuning_v0.1"
    ] = ML_HYPERPARAMETER_TUNING_RULE_VERSION


    # ========================================================
    # TEXT
    # ========================================================


    @field_validator(
        "workflow_id",
        "dataset_id",
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
            field_name=
                info.field_name,
        )


    # ========================================================
    # BASE CONTRACT SHA
    # ========================================================


    @field_validator(
        "base_training_contract_sha256",
        mode="before",
    )
    @classmethod
    def validate_base_training_contract_sha256(
        cls,
        value: object,
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
                    "base_training_contract_sha256 "
                    "must be a lowercase "
                    "64-character SHA-256 digest."
                )
            )


        return normalized


    # ========================================================
    # RESULT CONSISTENCY
    # ========================================================


    @model_validator(
        mode="after"
    )
    def validate_result_consistency(
        self,
    ) -> "MLHyperparameterSearchResult":

        expected_problem_type = (
            estimator_problem_type(
                self.estimator_key
            )
        )


        if (
            expected_problem_type
            is None
        ):
            raise ValueError(
                (
                    "Hyperparameter result contains "
                    "an unsupported estimator."
                )
            )


        if (
            expected_problem_type
            !=
            self.problem_type
        ):
            raise ValueError(
                (
                    "Hyperparameter estimator/problem "
                    "type mismatch."
                )
            )


        group_aware = (
            self.validation_strategy
            in
            {
                "group_k_fold",
                "stratified_group_k_fold",
            }
        )


        temporal_aware = (
            self.validation_strategy
            ==
            "time_series_split"
        )


        expected_validation_strategy = (
            hyperparameter_validation_strategy(
                problem_type=
                    self.problem_type,

                group_aware=
                    group_aware,

                temporal_aware=
                    temporal_aware,
            )
        )


        if (
            self.validation_strategy
            !=
            expected_validation_strategy
        ):
            raise ValueError(
                (
                    "Hyperparameter validation strategy "
                    "does not match problem_type."
                )
            )


        if (
            temporal_aware
            and
            self.shuffle
        ):
            raise ValueError(
                (
                    "Temporal Hyperparameter Tuning "
                    "must use shuffle=False."
                )
            )


        expected_primary_metric = (
            hyperparameter_primary_metric(
                problem_type=
                    self.problem_type
            )
        )


        if (
            self.primary_metric
            !=
            expected_primary_metric
        ):
            raise ValueError(
                (
                    "Hyperparameter primary metric "
                    "does not match problem_type."
                )
            )


        expected_direction = (
            hyperparameter_metric_direction(
                problem_type=
                    self.problem_type
            )
        )


        if (
            self.metric_direction
            !=
            expected_direction
        ):
            raise ValueError(
                (
                    "Hyperparameter metric direction "
                    "does not match problem_type."
                )
            )


        expected_candidates = (
            server_owned_hyperparameter_candidates(
                estimator_key=
                    self.estimator_key
            )
        )


        if (
            self.candidate_count
            !=
            len(
                expected_candidates
            )
        ):
            raise ValueError(
                (
                    "candidate_count does not match "
                    "the complete server-owned grid."
                )
            )


        if (
            len(
                self.candidate_results
            )
            !=
            self.candidate_count
        ):
            raise ValueError(
                (
                    "candidate_results length does not "
                    "match candidate_count."
                )
            )


        candidate_indexes = [
            item.candidate_index

            for item
            in self.candidate_results
        ]


        if (
            set(
                candidate_indexes
            )
            !=
            set(
                range(
                    1,
                    self.candidate_count
                    +
                    1,
                )
            )
        ):
            raise ValueError(
                (
                    "Hyperparameter candidate indexes "
                    "must exactly cover the server-owned "
                    "grid from 1..candidate_count."
                )
            )


        expected_metric_names = set(
            expected_hyperparameter_metric_names(
                problem_type=
                    self.problem_type
            )
        )


        candidate_by_index = {
            item.candidate_index:
                item

            for item
            in self.candidate_results
        }


        for (
            expected_index,
            expected_hyperparameters,
        ) in enumerate(
            expected_candidates,
            start=1,
        ):

            candidate = (
                candidate_by_index[
                    expected_index
                ]
            )


            if (
                candidate
                .hyperparameters
                .model_dump(
                    mode="json"
                )
                !=
                expected_hyperparameters
                .model_dump(
                    mode="json"
                )
            ):
                raise ValueError(
                    (
                        "Hyperparameter candidate does "
                        "not match its server-owned "
                        "grid position. "
                        f"candidate_index={expected_index}"
                    )
                )


            if (
                set(
                    candidate
                    .metric_summary
                    .keys()
                )
                !=
                expected_metric_names
            ):
                raise ValueError(
                    (
                        "Hyperparameter candidate metric "
                        "surface does not match "
                        "problem_type."
                    )
                )


        ranks = [
            item.rank

            for item
            in self.candidate_results
        ]


        if (
            ranks
            !=
            list(
                range(
                    1,
                    self.candidate_count
                    +
                    1,
                )
            )
        ):
            raise ValueError(
                (
                    "Hyperparameter candidate results "
                    "must be stored in rank order with "
                    "contiguous ranks from 1."
                )
            )


        def ranking_key(
            candidate: (
                MLHyperparameterCandidateResult
            ),
        ):

            summary = (
                candidate.metric_summary[
                    self.primary_metric
                ]
            )


            primary_value = (
                summary.mean

                if (
                    self.metric_direction
                    ==
                    "minimize"
                )

                else (
                    -
                    summary.mean
                )
            )


            return (
                primary_value,
                summary.std,
                candidate.candidate_index,
            )


        expected_ranked = sorted(
            self.candidate_results,
            key=
                ranking_key,
        )


        if (
            [
                item.candidate_index
                for item
                in self.candidate_results
            ]
            !=
            [
                item.candidate_index
                for item
                in expected_ranked
            ]
        ):
            raise ValueError(
                (
                    "Hyperparameter candidate results "
                    "do not follow deterministic "
                    "ranking policy."
                )
            )


        if (
            self.best_candidate_index
            !=
            self.candidate_results[
                0
            ].candidate_index
        ):
            raise ValueError(
                (
                    "best_candidate_index must identify "
                    "the rank-1 candidate."
                )
            )


        return self
