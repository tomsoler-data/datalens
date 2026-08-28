from __future__ import annotations


import math


from typing import (
    Any,
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


from app.ml.classical_executor import (
    ClassicalMLExecutionResult,
    ClassicalMLExecutorError,
    execute_classical_ml,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.ml.model_comparison_contracts import (
    MLModelComparisonContract,
    MLModelComparisonPrimaryMetric,
    MLModelComparisonRankingPolicy,
)


from app.preparation.analysis_readiness_gate import (
    AnalysisReadinessError,
    require_analysis_readiness,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_COMPARISON_EXECUTOR_RULE_VERSION = (
    "ml_model_comparison_executor_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLModelComparisonExecutorError(
    RuntimeError
):
    pass


class MLModelComparisonCandidateError(
    MLModelComparisonExecutorError
):
    pass


class MLModelComparisonRankingError(
    MLModelComparisonExecutorError
):
    pass


class MLModelComparisonSnapshotError(
    MLModelComparisonExecutorError
):
    pass


# ============================================================
# CANDIDATE RESULT
# ============================================================


class MLModelComparisonCandidateResult(
    BaseModel
):
    """
    Privacy-minimal ranked result for one fixed estimator
    candidate.

    Raw rows and predictions are deliberately absent.

    model_artifact preserves the complete server-owned training
    provenance and trusted reload identity.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    rank: int = Field(
        ge=1,
    )


    estimator_key: str = Field(
        min_length=1,
    )


    primary_metric: (
        MLModelComparisonPrimaryMetric
    )


    primary_metric_value: float


    metrics: dict[
        str,
        float,
    ]


    train_rows: int = Field(
        gt=0,
    )


    test_rows: int = Field(
        gt=0,
    )


    model_artifact: (
        MLModelArtifactRecord
    )


# ============================================================
# COMPARISON RESULT
# ============================================================


class MLModelComparisonExecutionResult(
    BaseModel
):
    """
    Deterministic outcome of one Model Comparison execution.

    candidates is stored in final ranking order.

    candidates[0] is therefore the selected model according to
    the fixed v0.1 ranking policy.
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


    preparation_session_revision: int = Field(
        ge=0,
    )


    problem_type: Literal[
        "regression",
        "classification",
    ]


    comparison_contract: (
        MLModelComparisonContract
    )


    primary_metric: (
        MLModelComparisonPrimaryMetric
    )


    ranking_policy: (
        MLModelComparisonRankingPolicy
    )


    candidates: list[
        MLModelComparisonCandidateResult
    ] = Field(
        min_length=2,
    )


    selected_estimator_key: str = Field(
        min_length=1,
    )


    selected_model_id: str = Field(
        min_length=1,
    )


    rule_version: Literal[
        "ml_model_comparison_executor_v0.1"
    ] = (
        ML_MODEL_COMPARISON_EXECUTOR_RULE_VERSION
    )


    @model_validator(
        mode="after"
    )
    def validate_result_consistency(
        self,
    ) -> "MLModelComparisonExecutionResult":

        # ----------------------------------------------------
        # COMPARISON AUTHORITY
        # ----------------------------------------------------

        if (
            self.workflow_id
            !=
            self.comparison_contract.workflow_id
        ):
            raise ValueError(
                (
                    "Comparison result workflow_id "
                    "does not match comparison contract."
                )
            )


        if (
            self.dataset_id
            !=
            self.comparison_contract.dataset_id
        ):
            raise ValueError(
                (
                    "Comparison result dataset_id "
                    "does not match comparison contract."
                )
            )


        if (
            self.problem_type
            !=
            self.comparison_contract.problem_type
        ):
            raise ValueError(
                (
                    "Comparison result problem_type "
                    "does not match comparison contract."
                )
            )


        if (
            self.primary_metric
            !=
            self.comparison_contract.primary_metric
        ):
            raise ValueError(
                (
                    "Comparison result primary_metric "
                    "does not match comparison contract."
                )
            )


        if (
            self.ranking_policy
            !=
            self.comparison_contract.ranking_policy
        ):
            raise ValueError(
                (
                    "Comparison result ranking_policy "
                    "does not match comparison contract."
                )
            )


        # ----------------------------------------------------
        # COMPLETE CANDIDATE SET
        # ----------------------------------------------------

        expected_keys = {
            candidate.estimator_key

            for candidate
            in self.comparison_contract.candidates
        }


        actual_keys = {
            candidate.estimator_key

            for candidate
            in self.candidates
        }


        if (
            actual_keys
            !=
            expected_keys
        ):
            raise ValueError(
                (
                    "Comparison result candidate set "
                    "does not match comparison contract."
                )
            )


        if (
            len(
                self.candidates
            )
            !=
            len(
                self.comparison_contract.candidates
            )
        ):
            raise ValueError(
                (
                    "Comparison result candidate count "
                    "does not match comparison contract."
                )
            )


        # ----------------------------------------------------
        # RANKS
        # ----------------------------------------------------

        ranks = [
            candidate.rank

            for candidate
            in self.candidates
        ]


        expected_ranks = list(
            range(
                1,
                len(
                    self.candidates
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
                    "Comparison candidate ranks must be "
                    "contiguous and already sorted."
                )
            )


        # ----------------------------------------------------
        # SELECTED MODEL
        # ----------------------------------------------------

        winner = (
            self.candidates[
                0
            ]
        )


        if (
            self.selected_estimator_key
            !=
            winner.estimator_key
        ):
            raise ValueError(
                (
                    "selected_estimator_key must match "
                    "ranked candidate #1."
                )
            )


        if (
            self.selected_model_id
            !=
            winner
            .model_artifact
            .model_id
        ):
            raise ValueError(
                (
                    "selected_model_id must match "
                    "ranked candidate #1 Model Artifact."
                )
            )


        return self


# ============================================================
# REQUIRED METRICS
# ============================================================


def _required_metric_names(
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
        MLModelComparisonRankingError(
            (
                "Unsupported problem type for "
                "Model Comparison ranking."
            )
        )
    )


# ============================================================
# METRIC VALIDATION
# ============================================================


def _validated_metric(
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
            MLModelComparisonRankingError(
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
            MLModelComparisonRankingError(
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
            MLModelComparisonRankingError(
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


def _ranking_key(
    *,
    problem_type: str,
    estimator_key: str,
    metrics: dict[
        str,
        float,
    ],
) -> tuple[
    Any,
    ...,
]:
    """
    Convert DataLens ranking policy to a Python ascending sort
    key.

    Regression:
        RMSE asc
        MAE asc
        R² desc
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
            _validated_metric(
                metrics=
                    metrics,

                metric_name=
                    "rmse",
            )
        )


        mae = (
            _validated_metric(
                metrics=
                    metrics,

                metric_name=
                    "mae",
            )
        )


        r2 = (
            _validated_metric(
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
            _validated_metric(
                metrics=
                    metrics,

                metric_name=
                    "f1_macro",
            )
        )


        accuracy = (
            _validated_metric(
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
        MLModelComparisonRankingError(
            (
                "Unsupported problem type for "
                "Model Comparison ranking."
            )
        )
    )


# ============================================================
# CANDIDATE EXECUTION VALIDATION
# ============================================================


def _validate_candidate_execution(
    *,
    comparison_contract: (
        MLModelComparisonContract
    ),
    candidate_contract,
    execution_result: (
        ClassicalMLExecutionResult
    ),
) -> None:

    # --------------------------------------------------------
    # AUTHORITY
    # --------------------------------------------------------

    if (
        execution_result.workflow_id
        !=
        comparison_contract.workflow_id
    ):
        raise (
            MLModelComparisonCandidateError(
                (
                    "Candidate execution workflow "
                    "does not match comparison authority."
                )
            )
        )


    if (
        execution_result.dataset_id
        !=
        comparison_contract.dataset_id
    ):
        raise (
            MLModelComparisonCandidateError(
                (
                    "Candidate execution dataset "
                    "does not match comparison authority."
                )
            )
        )


    if (
        execution_result.problem_type
        !=
        comparison_contract.problem_type
    ):
        raise (
            MLModelComparisonCandidateError(
                (
                    "Candidate execution problem type "
                    "does not match comparison authority."
                )
            )
        )


    if (
        execution_result.estimator_key
        !=
        candidate_contract.estimator_key
    ):
        raise (
            MLModelComparisonCandidateError(
                (
                    "Candidate execution estimator_key "
                    "does not match candidate contract."
                )
            )
        )


    # --------------------------------------------------------
    # MODEL ARTIFACT PROVENANCE
    # --------------------------------------------------------

    artifact = (
        execution_result
        .model_artifact
    )


    if (
        artifact.training_contract
        !=
        candidate_contract
    ):
        raise (
            MLModelComparisonCandidateError(
                (
                    "Candidate Model Artifact training "
                    "contract does not match the "
                    "comparison candidate."
                )
            )
        )


    if (
        artifact.workflow_id
        !=
        comparison_contract.workflow_id
    ):
        raise (
            MLModelComparisonCandidateError(
                (
                    "Candidate Model Artifact workflow "
                    "scope does not match comparison."
                )
            )
        )


    if (
        artifact.dataset_id
        !=
        comparison_contract.dataset_id
    ):
        raise (
            MLModelComparisonCandidateError(
                (
                    "Candidate Model Artifact dataset "
                    "scope does not match comparison."
                )
            )
        )


    if (
        artifact.metrics
        !=
        execution_result.metrics
    ):
        raise (
            MLModelComparisonCandidateError(
                (
                    "Candidate Model Artifact metrics "
                    "do not match execution metrics."
                )
            )
        )


    # --------------------------------------------------------
    # REQUIRED FINITE METRICS
    # --------------------------------------------------------

    for metric_name in (
        _required_metric_names(
            problem_type=
                comparison_contract.problem_type
        )
    ):
        _validated_metric(
            metrics=
                execution_result.metrics,

            metric_name=
                metric_name,
        )


# ============================================================
# SPLIT CONSISTENCY
# ============================================================


def _validate_execution_split_sizes(
    *,
    execution_results: list[
        ClassicalMLExecutionResult
    ],
) -> None:

    if not execution_results:
        raise (
            MLModelComparisonExecutorError(
                (
                    "Model Comparison produced "
                    "no candidate executions."
                )
            )
        )


    reference = (
        execution_results[
            0
        ]
    )


    for execution_result in (
        execution_results[
            1:
        ]
    ):

        if (
            execution_result.train_rows
            !=
            reference.train_rows
            or
            execution_result.test_rows
            !=
            reference.test_rows
        ):
            raise (
                MLModelComparisonExecutorError(
                    (
                        "Comparison candidates did not "
                        "produce identical train/test "
                        "row counts."
                    )
                )
            )


# ============================================================
# PREPARATION SNAPSHOT AUTHORITY
# ============================================================


def _read_comparison_snapshot_authority(
    *,
    contract: MLModelComparisonContract,
) -> tuple[
    int,
    tuple[
        str,
        ...,
    ],
]:
    """
    Pin the server-owned Preparation revision and final dataset
    scope used by the complete model comparison.

    Every candidate execution must remain inside this same
    authority snapshot.

    Individual Classical ML executions already protect their own
    handoff reads against revision races.

    This additional guard protects the interval BETWEEN
    candidates.
    """

    try:
        decision = (
            require_analysis_readiness(
                workflow_id=
                    contract.workflow_id
            )
        )

    except AnalysisReadinessError as error:
        raise (
            MLModelComparisonSnapshotError(
                (
                    "Model Comparison could not pin "
                    "a READY Preparation snapshot."
                )
            )
        ) from error


    if (
        decision.workflow_id
        !=
        contract.workflow_id
    ):
        raise (
            MLModelComparisonSnapshotError(
                (
                    "Preparation snapshot workflow "
                    "does not match comparison authority."
                )
            )
        )


    dataset_ids = tuple(
        decision
        .requested_analysis_dataset_ids
    )


    if (
        contract.dataset_id
        not in
        dataset_ids
    ):
        raise (
            MLModelComparisonSnapshotError(
                (
                    "Comparison dataset is not present "
                    "in the validated Preparation "
                    "analysis-output scope."
                )
            )
        )


    return (
        int(
            decision.session_revision
        ),
        dataset_ids,
    )


def _assert_comparison_snapshot_unchanged(
    *,
    contract: MLModelComparisonContract,
    expected_session_revision: int,
    expected_dataset_ids: tuple[
        str,
        ...,
    ],
) -> None:
    """
    Fail closed if Preparation changed while candidates were
    being compared.

    Preparation session revisions are server-owned. A change of
    revision or final analysis-output scope invalidates metric
    comparability for the current comparison.
    """

    try:
        decision = (
            require_analysis_readiness(
                workflow_id=
                    contract.workflow_id
            )
        )

    except AnalysisReadinessError as error:
        raise (
            MLModelComparisonSnapshotError(
                (
                    "Preparation stopped being READY "
                    "during Model Comparison."
                )
            )
        ) from error


    if (
        decision.workflow_id
        !=
        contract.workflow_id
    ):
        raise (
            MLModelComparisonSnapshotError(
                (
                    "Preparation workflow authority "
                    "changed during Model Comparison."
                )
            )
        )


    if (
        int(
            decision.session_revision
        )
        !=
        expected_session_revision
    ):
        raise (
            MLModelComparisonSnapshotError(
                (
                    "Preparation session revision "
                    "changed during Model Comparison. "
                    "Comparison metrics are therefore "
                    "not considered comparable."
                )
            )
        )


    current_dataset_ids = tuple(
        decision
        .requested_analysis_dataset_ids
    )


    if (
        current_dataset_ids
        !=
        expected_dataset_ids
    ):
        raise (
            MLModelComparisonSnapshotError(
                (
                    "Preparation final analysis-output "
                    "scope changed during Model "
                    "Comparison."
                )
            )
        )


# ============================================================
# EXECUTION
# ============================================================


def execute_ml_model_comparison(
    *,
    comparison_contract: (
        MLModelComparisonContract
    ),
) -> MLModelComparisonExecutionResult:
    """
    Execute deterministic comparison of fixed Classical ML
    candidate contracts.

    Each candidate is delegated to the existing trusted
    Classical ML executor.

    Therefore every candidate still crosses:

        validated Preparation handoff
                ↓
        deterministic split
                ↓
        leakage-safe preprocessing
                ↓
        server-controlled estimator
                ↓
        metrics
                ↓
        server-owned Model Artifact

    Because the comparison contract requires identical dataset,
    features, preprocessing and split policy, all candidates are
    evaluated under the same deterministic holdout authority.

    v0.1 deliberately does not perform hyperparameter search.
    """

    contract = (
        MLModelComparisonContract
        .model_validate(
            comparison_contract
        )
    )


    (
        preparation_session_revision,
        preparation_dataset_ids,
    ) = (
        _read_comparison_snapshot_authority(
            contract=
                contract
        )
    )


    execution_results: list[
        ClassicalMLExecutionResult
    ] = []


    # ========================================================
    # EXECUTE FIXED CANDIDATES
    # ========================================================

    for (
        candidate_index,
        candidate_contract,
    ) in enumerate(
        contract.candidates
    ):

        _assert_comparison_snapshot_unchanged(
            contract=
                contract,

            expected_session_revision=
                preparation_session_revision,

            expected_dataset_ids=
                preparation_dataset_ids,
        )


        try:
            execution_result = (
                execute_classical_ml(
                    training_contract=
                        candidate_contract
                )
            )

        except ClassicalMLExecutorError as error:
            raise (
                MLModelComparisonCandidateError(
                    (
                        "Model Comparison candidate "
                        "execution failed. "
                        "candidate_index="
                        f"{candidate_index}, "
                        "estimator_key="
                        f"{candidate_contract.estimator_key}"
                    )
                )
            ) from error


        _validate_candidate_execution(
            comparison_contract=
                contract,

            candidate_contract=
                candidate_contract,

            execution_result=
                execution_result,
        )


        _assert_comparison_snapshot_unchanged(
            contract=
                contract,

            expected_session_revision=
                preparation_session_revision,

            expected_dataset_ids=
                preparation_dataset_ids,
        )


        execution_results.append(
            execution_result
        )


    # ========================================================
    # SAME DETERMINISTIC HOLDOUT SHAPE
    # ========================================================

    _validate_execution_split_sizes(
        execution_results=
            execution_results
    )


    # ========================================================
    # DETERMINISTIC RANKING
    # ========================================================

    ranked_executions = sorted(
        execution_results,

        key=lambda result: (
            _ranking_key(
                problem_type=
                    contract.problem_type,

                estimator_key=
                    result.estimator_key,

                metrics=
                    result.metrics,
            )
        ),
    )


    # ========================================================
    # PRIVACY-MINIMAL RANKED RESULT
    # ========================================================

    ranked_candidates: list[
        MLModelComparisonCandidateResult
    ] = []


    for (
        rank,
        execution_result,
    ) in enumerate(
        ranked_executions,
        start=1,
    ):

        primary_metric_value = (
            _validated_metric(
                metrics=
                    execution_result.metrics,

                metric_name=
                    contract.primary_metric,
            )
        )


        ranked_candidates.append(
            MLModelComparisonCandidateResult(
                rank=
                    rank,

                estimator_key=
                    execution_result.estimator_key,

                primary_metric=
                    contract.primary_metric,

                primary_metric_value=
                    primary_metric_value,

                metrics=
                    execution_result.metrics,

                train_rows=
                    execution_result.train_rows,

                test_rows=
                    execution_result.test_rows,

                model_artifact=
                    execution_result
                    .model_artifact,
            )
        )


    winner = (
        ranked_candidates[
            0
        ]
    )


    return (
        MLModelComparisonExecutionResult(
            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,

            preparation_session_revision=
                preparation_session_revision,

            problem_type=
                contract.problem_type,

            comparison_contract=
                contract,

            primary_metric=
                contract.primary_metric,

            ranking_policy=
                contract.ranking_policy,

            candidates=
                ranked_candidates,

            selected_estimator_key=
                winner.estimator_key,

            selected_model_id=
                winner
                .model_artifact
                .model_id,

            rule_version=(
                ML_MODEL_COMPARISON_EXECUTOR_RULE_VERSION
            ),
        )
    )