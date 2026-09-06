from __future__ import annotations


import math


from dataclasses import (
    dataclass,
)


from typing import (
    Any,
)


from app.ml.baseline import (
    MLBaselineEvaluationResult,
)


from app.ml.model_candidate_result import (
    MLModelCandidateResult,
)


from app.ml.model_comparison_contracts import (
    MLModelComparisonContract,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_COMPARISON_CORE_RULE_VERSION = (
    "ml_model_comparison_core_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLModelComparisonCoreError(
    RuntimeError
):
    pass


class MLModelComparisonCoreCandidateError(
    MLModelComparisonCoreError
):
    pass


class MLModelComparisonCoreRankingError(
    MLModelComparisonCoreError
):
    pass


# ============================================================
# AGGREGATION RESULT
# ============================================================


@dataclass(
    frozen=True,
)
class MLModelComparisonAggregation:
    """
    Framework-neutral aggregation result.

    ranked_candidates contains privacy-minimal candidate
    projections only.

    Raw rows, predictions and fitted framework objects are
    deliberately absent.
    """

    baseline: MLBaselineEvaluationResult

    ranked_candidates: tuple[
        MLModelCandidateResult,
        ...
    ]

    rule_version: str = (
        ML_MODEL_COMPARISON_CORE_RULE_VERSION
    )


    @property
    def winner(
        self,
    ) -> MLModelCandidateResult:

        if not self.ranked_candidates:
            raise (
                MLModelComparisonCoreError(
                    (
                        "Model Comparison aggregation "
                        "contains no ranked winner."
                    )
                )
            )

        return (
            self.ranked_candidates[
                0
            ]
        )


# ============================================================
# REQUIRED METRICS
# ============================================================


def required_ml_model_comparison_metrics(
    *,
    problem_type: str,
) -> tuple[
    str,
    ...
]:

    if (
        problem_type
        ==
        "regression"
    ):
        return (
            "rmse",
            "mae",
            "r2",
        )


    if (
        problem_type
        ==
        "classification"
    ):
        return (
            "f1_macro",
            "accuracy",
        )


    raise (
        MLModelComparisonCoreRankingError(
            (
                "Unsupported problem type for "
                "Model Comparison ranking."
            )
        )
    )


# ============================================================
# METRIC VALIDATION
# ============================================================


def validated_ml_model_comparison_metric(
    *,
    metrics: dict[
        str,
        float,
    ],
    metric_name: str,
) -> float:

    if (
        metric_name
        not in
        metrics
    ):
        raise (
            MLModelComparisonCoreRankingError(
                (
                    "Model Comparison candidate "
                    "is missing a required metric. "
                    f"metric={metric_name}"
                )
            )
        )


    try:

        value = float(
            metrics[
                metric_name
            ]
        )

    except Exception as error:

        raise (
            MLModelComparisonCoreRankingError(
                (
                    "Model Comparison candidate "
                    "metric is not numeric. "
                    f"metric={metric_name}"
                )
            )
        ) from error


    if not math.isfinite(
        value
    ):
        raise (
            MLModelComparisonCoreRankingError(
                (
                    "Model Comparison candidate "
                    "metric is not finite. "
                    f"metric={metric_name}"
                )
            )
        )


    return value


# ============================================================
# RANKING KEY
# ============================================================


def ml_model_comparison_ranking_key(
    *,
    problem_type: str,
    estimator_key: str,
    metrics: dict[
        str,
        float,
    ],
) -> tuple[
    Any,
    ...
]:
    """
    Exact DataLens deterministic ranking authority.

    Regression:
        RMSE asc
        MAE asc
        R? desc
        estimator_key asc

    Classification:
        F1 macro desc
        Accuracy desc
        estimator_key asc
    """

    if (
        problem_type
        ==
        "regression"
    ):

        rmse = (
            validated_ml_model_comparison_metric(
                metrics=
                    metrics,

                metric_name=
                    "rmse",
            )
        )


        mae = (
            validated_ml_model_comparison_metric(
                metrics=
                    metrics,

                metric_name=
                    "mae",
            )
        )


        r2 = (
            validated_ml_model_comparison_metric(
                metrics=
                    metrics,

                metric_name=
                    "r2",
            )
        )


        return (
            rmse,
            mae,
            -r2,
            estimator_key,
        )


    if (
        problem_type
        ==
        "classification"
    ):

        f1_macro = (
            validated_ml_model_comparison_metric(
                metrics=
                    metrics,

                metric_name=
                    "f1_macro",
            )
        )


        accuracy = (
            validated_ml_model_comparison_metric(
                metrics=
                    metrics,

                metric_name=
                    "accuracy",
            )
        )


        return (
            -f1_macro,
            -accuracy,
            estimator_key,
        )


    raise (
        MLModelComparisonCoreRankingError(
            (
                "Unsupported problem type for "
                "Model Comparison ranking."
            )
        )
    )


# ============================================================
# CANDIDATE / CONTRACT CONSISTENCY
# ============================================================


def _validate_candidate_against_comparison(
    *,
    comparison_contract: (
        MLModelComparisonContract
    ),
    candidate: MLModelCandidateResult,
    expected_preparation_session_revision: int,
    expected_contract_by_estimator: dict[
        str,
        object,
    ],
) -> None:

    if (
        candidate.workflow_id
        !=
        comparison_contract.workflow_id
    ):
        raise (
            MLModelComparisonCoreCandidateError(
                (
                    "Candidate workflow does not "
                    "match comparison authority."
                )
            )
        )


    if (
        candidate.dataset_id
        !=
        comparison_contract.dataset_id
    ):
        raise (
            MLModelComparisonCoreCandidateError(
                (
                    "Candidate dataset does not "
                    "match comparison authority."
                )
            )
        )


    if (
        candidate.problem_type
        !=
        comparison_contract.problem_type
    ):
        raise (
            MLModelComparisonCoreCandidateError(
                (
                    "Candidate problem type does not "
                    "match comparison authority."
                )
            )
        )


    expected_contract = (
        expected_contract_by_estimator.get(
            candidate.estimator_key
        )
    )


    if expected_contract is None:
        raise (
            MLModelComparisonCoreCandidateError(
                (
                    "Candidate estimator is outside "
                    "the comparison contract. "
                    f"estimator_key={candidate.estimator_key}"
                )
            )
        )


    if (
        candidate
        .model_artifact
        .training_contract
        !=
        expected_contract
    ):
        raise (
            MLModelComparisonCoreCandidateError(
                (
                    "Candidate Model Artifact training "
                    "contract does not match the "
                    "comparison candidate."
                )
            )
        )


    if (
        candidate
        .preparation_session_revision
        !=
        expected_preparation_session_revision
    ):
        raise (
            MLModelComparisonCoreCandidateError(
                (
                    "Candidate does not reference "
                    "the pinned Preparation revision."
                )
            )
        )


    # --------------------------------------------------------
    # REQUIRED FINITE METRICS
    # --------------------------------------------------------


    for metric_name in (
        required_ml_model_comparison_metrics(
            problem_type=
                comparison_contract.problem_type
        )
    ):

        validated_ml_model_comparison_metric(
            metrics=
                candidate.metrics,

            metric_name=
                metric_name,
        )


    # --------------------------------------------------------
    # BASELINE AUTHORITY
    # --------------------------------------------------------


    baseline = (
        candidate.baseline
    )


    if (
        baseline.problem_type
        !=
        comparison_contract.problem_type
    ):
        raise (
            MLModelComparisonCoreCandidateError(
                (
                    "Candidate baseline problem type "
                    "does not match comparison."
                )
            )
        )


    if (
        baseline.primary_metric
        !=
        comparison_contract.primary_metric
    ):
        raise (
            MLModelComparisonCoreCandidateError(
                (
                    "Candidate baseline primary metric "
                    "does not match comparison."
                )
            )
        )


    for metric_name in (
        required_ml_model_comparison_metrics(
            problem_type=
                comparison_contract.problem_type
        )
    ):

        validated_ml_model_comparison_metric(
            metrics=
                baseline.metrics,

            metric_name=
                metric_name,
        )


    baseline_comparison = (
        candidate
        .baseline_comparison
    )


    if (
        baseline_comparison.primary_metric
        !=
        comparison_contract.primary_metric
    ):
        raise (
            MLModelComparisonCoreCandidateError(
                (
                    "Candidate model-to-baseline "
                    "primary metric does not match "
                    "comparison authority."
                )
            )
        )


# ============================================================
# AGGREGATION
# ============================================================


def aggregate_ml_model_candidates(
    *,
    comparison_contract: (
        MLModelComparisonContract
    ),
    candidates: list[
        MLModelCandidateResult
    ],
    expected_preparation_session_revision: int,
) -> MLModelComparisonAggregation:
    """
    Validate and rank framework-neutral Model Lab candidates.

    This function does not execute models.

    Execution belongs to framework-specific orchestration.

    This core owns only cross-candidate comparability,
    shared-baseline validation and deterministic ranking.
    """

    contract = (
        MLModelComparisonContract
        .model_validate(
            comparison_contract
        )
    )


    if (
        isinstance(
            expected_preparation_session_revision,
            bool,
        )
        or
        not isinstance(
            expected_preparation_session_revision,
            int,
        )
        or
        expected_preparation_session_revision
        <
        0
    ):
        raise (
            MLModelComparisonCoreError(
                (
                    "Expected Preparation revision "
                    "must be a non-negative integer."
                )
            )
        )


    if (
        len(
            candidates
        )
        !=
        len(
            contract.candidates
        )
    ):
        raise (
            MLModelComparisonCoreCandidateError(
                (
                    "Executed candidate count does not "
                    "match comparison contract."
                )
            )
        )


    expected_contract_by_estimator = {
        candidate_contract.estimator_key:
            candidate_contract

        for candidate_contract
        in contract.candidates
    }


    actual_estimator_keys = [
        candidate.estimator_key

        for candidate
        in candidates
    ]


    if (
        len(
            set(
                actual_estimator_keys
            )
        )
        !=
        len(
            actual_estimator_keys
        )
    ):
        raise (
            MLModelComparisonCoreCandidateError(
                (
                    "Executed comparison candidates "
                    "must have unique estimator keys."
                )
            )
        )


    if (
        set(
            actual_estimator_keys
        )
        !=
        set(
            expected_contract_by_estimator
        )
    ):
        raise (
            MLModelComparisonCoreCandidateError(
                (
                    "Executed candidate estimator set "
                    "does not match comparison contract."
                )
            )
        )


    for candidate in (
        candidates
    ):

        _validate_candidate_against_comparison(
            comparison_contract=
                contract,

            candidate=
                candidate,

            expected_preparation_session_revision=
                expected_preparation_session_revision,

            expected_contract_by_estimator=
                expected_contract_by_estimator,
        )


    # ========================================================
    # IDENTICAL HOLDOUT SHAPE
    # ========================================================


    reference = (
        candidates[
            0
        ]
    )


    for candidate in (
        candidates[
            1:
        ]
    ):

        if (
            candidate.train_rows
            !=
            reference.train_rows
            or
            candidate.test_rows
            !=
            reference.test_rows
        ):
            raise (
                MLModelComparisonCoreError(
                    (
                        "Comparison candidates did not "
                        "produce identical train/test "
                        "row counts."
                    )
                )
            )


    # ========================================================
    # UNIQUE EXPERIMENT IDENTITIES
    # ========================================================


    experiment_ids = [
        candidate
        .experiment_provenance
        .experiment_id

        for candidate
        in candidates
    ]


    if (
        len(
            set(
                experiment_ids
            )
        )
        !=
        len(
            experiment_ids
        )
    ):
        raise (
            MLModelComparisonCoreError(
                (
                    "Comparison candidates produced "
                    "duplicate experiment_id values."
                )
            )
        )


    # ========================================================
    # ONE SHARED BASELINE
    # ========================================================


    shared_baseline = (
        reference.baseline
    )


    for candidate in (
        candidates[
            1:
        ]
    ):

        if (
            candidate.baseline
            !=
            shared_baseline
        ):
            raise (
                MLModelComparisonCoreError(
                    (
                        "Comparison candidates did not "
                        "produce one identical shared "
                        "baseline."
                    )
                )
            )


    # ========================================================
    # DETERMINISTIC RANKING
    # ========================================================


    ranked_candidates = tuple(
        sorted(
            candidates,

            key=lambda candidate: (
                ml_model_comparison_ranking_key(
                    problem_type=
                        contract.problem_type,

                    estimator_key=
                        candidate.estimator_key,

                    metrics=
                        candidate.metrics,
                )
            ),
        )
    )


    return (
        MLModelComparisonAggregation(
            baseline=
                shared_baseline,

            ranked_candidates=
                ranked_candidates,
        )
    )
