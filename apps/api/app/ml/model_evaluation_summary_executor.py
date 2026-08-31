from __future__ import annotations


from typing import (
    Any,
)


from app.ml.baseline import (
    MLBaselineError,
    build_ml_baseline_evaluation,
    build_ml_baseline_predictions,
    compare_model_to_baseline,
)


from app.ml.classical_executor import (
    ClassicalMLInputError,
    _load_authorized_dataframe,
    _split_dataset,
    _validate_and_extract_xy,
)


from app.ml.model_metrics import (
    compute_ml_classification_metrics,
    compute_ml_regression_metrics,
    project_ml_baseline_metrics_v0_1,
)


from app.ml.classification_diagnostics import (
    MLClassificationDiagnosticsContract,
)


from app.ml.classification_diagnostics_executor import (
    MLClassificationDiagnosticsExecutorError,
    execute_ml_classification_diagnostics,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.decision_threshold_executor import (
    MLDecisionThresholdExecutorError,
    execute_ml_decision_threshold,
)


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


from app.ml.model_comparison_executor import (
    MLModelComparisonExecutionResult,
)


from app.ml.model_evaluation_summary import (
    MLModelEvaluationSummaryContract,
    MLModelEvaluationSummaryResult,
    MLModelSelectionEvidence,
    expected_model_evaluation_limitations,
)


from app.ml.model_explainability import (
    MLModelExplainabilityContract,
)


from app.ml.model_explainability_executor import (
    MLModelExplainabilityExecutorError,
    execute_ml_model_explainability,
)


from app.ml.model_loader import (
    MLModelLoaderError,
    load_trusted_ml_model,
)


from app.ml.tuned_model_promotion_executor import (
    MLTunedModelPromotionExecutionResult,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_EVALUATION_SUMMARY_EXECUTOR_RULE_VERSION = (
    "ml_model_evaluation_summary_executor_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLModelEvaluationSummaryExecutorError(
    RuntimeError
):
    pass


class MLModelEvaluationSummaryArtifactError(
    MLModelEvaluationSummaryExecutorError
):
    pass


class MLModelEvaluationSummaryInputError(
    MLModelEvaluationSummaryExecutorError
):
    pass


class MLModelEvaluationSummarySelectionError(
    MLModelEvaluationSummaryExecutorError
):
    pass


class MLModelEvaluationSummaryExecutionError(
    MLModelEvaluationSummaryExecutorError
):
    pass


# ============================================================
# IDENTIFIERS
# ============================================================


def _required_identifier(
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
        raise (
            MLModelEvaluationSummaryInputError(
                (
                    f"{field_name} "
                    "cannot be empty."
                )
            )
        )


    return normalized


# ============================================================
# MODEL COMPARISON SELECTION EVIDENCE
# ============================================================


def _selection_from_model_comparison(
    *,
    artifact,
    provenance,
    training_contract_sha256: str,
    comparison_context: (
        MLModelComparisonExecutionResult
    ),
    baseline,
) -> MLModelSelectionEvidence:

    try:
        comparison = (
            MLModelComparisonExecutionResult
            .model_validate(
                comparison_context
                .model_dump(
                    mode="python"
                )
            )
        )

    except Exception as error:
        raise (
            MLModelEvaluationSummarySelectionError(
                (
                    "Model Comparison selection context "
                    "is invalid."
                )
            )
        ) from error


    if (
        comparison.selected_model_id
        !=
        artifact.model_id
    ):
        raise (
            MLModelEvaluationSummarySelectionError(
                (
                    "Model Comparison did not select "
                    "the Model Artifact under evaluation."
                )
            )
        )


    if (
        comparison.selected_experiment_id
        !=
        provenance.experiment_id
    ):
        raise (
            MLModelEvaluationSummarySelectionError(
                (
                    "Model Comparison selected experiment "
                    "does not match the evaluated "
                    "Model Artifact."
                )
            )
        )


    identity_checks = (
        (
            "workflow_id",
            comparison.workflow_id,
            artifact.workflow_id,
        ),
        (
            "dataset_id",
            comparison.dataset_id,
            artifact.dataset_id,
        ),
        (
            "problem_type",
            comparison.problem_type,
            artifact
            .training_contract
            .problem_type,
        ),
        (
            "preparation_session_revision",
            comparison
            .preparation_session_revision,
            provenance
            .preparation_session_revision,
        ),
    )


    for (
        field_name,
        actual,
        expected,
    ) in identity_checks:

        if actual != expected:
            raise (
                MLModelEvaluationSummarySelectionError(
                    (
                        "Model Comparison selection "
                        "context does not match the "
                        "evaluated artifact. "
                        f"field={field_name}"
                    )
                )
            )


    winner = (
        comparison.candidates[
            0
        ]
    )


    if winner.rank != 1:
        raise (
            MLModelEvaluationSummarySelectionError(
                (
                    "Model Comparison winner must "
                    "remain rank #1."
                )
            )
        )


    if (
        winner.model_artifact.model_id
        !=
        artifact.model_id
    ):
        raise (
            MLModelEvaluationSummarySelectionError(
                (
                    "Model Comparison rank #1 artifact "
                    "does not match the evaluated model."
                )
            )
        )


    if (
        winner
        .experiment_provenance
        .experiment_id
        !=
        provenance.experiment_id
    ):
        raise (
            MLModelEvaluationSummarySelectionError(
                (
                    "Model Comparison rank #1 experiment "
                    "does not match the evaluated model."
                )
            )
        )


    winner_sha256 = (
        ml_training_contract_sha256(
            winner
            .model_artifact
            .training_contract
        )
    )


    if (
        winner_sha256
        !=
        training_contract_sha256
    ):
        raise (
            MLModelEvaluationSummarySelectionError(
                (
                    "Model Comparison rank #1 Training "
                    "Contract SHA does not match the "
                    "evaluated Model Artifact."
                )
            )
        )


    if (
        winner.metrics
        !=
        artifact.metrics
    ):
        raise (
            MLModelEvaluationSummarySelectionError(
                (
                    "Model Comparison rank #1 metrics "
                    "do not match persisted artifact "
                    "metrics."
                )
            )
        )


    if (
        winner.train_rows
        !=
        artifact.train_rows
        or
        winner.test_rows
        !=
        artifact.test_rows
    ):
        raise (
            MLModelEvaluationSummarySelectionError(
                (
                    "Model Comparison rank #1 holdout "
                    "shape does not match the evaluated "
                    "Model Artifact."
                )
            )
        )


    if (
        comparison.baseline
        !=
        baseline
    ):
        raise (
            MLModelEvaluationSummarySelectionError(
                (
                    "Reconstructed baseline does not "
                    "match the server-owned Model "
                    "Comparison baseline."
                )
            )
        )


    return (
        MLModelSelectionEvidence(
            source=
                "model_comparison",

            status=
                "verified_selected",

            rank=
                1,

            selection_policy=
                comparison.ranking_policy,

            primary_metric=
                comparison.primary_metric,

            primary_metric_value=(
                winner
                .primary_metric_value
            ),

            metric_scope=
                "final_holdout",
        )
    )


# ============================================================
# TUNED PROMOTION SELECTION EVIDENCE
# ============================================================


def _selection_from_tuned_promotion(
    *,
    artifact,
    provenance,
    training_contract_sha256: str,
    promotion_context: (
        MLTunedModelPromotionExecutionResult
    ),
) -> MLModelSelectionEvidence:

    try:
        promotion = (
            MLTunedModelPromotionExecutionResult
            .model_validate(
                promotion_context
                .model_dump(
                    mode="python"
                )
            )
        )

    except Exception as error:
        raise (
            MLModelEvaluationSummarySelectionError(
                (
                    "Tuned Model Promotion selection "
                    "context is invalid."
                )
            )
        ) from error


    identity_checks = (
        (
            "workflow_id",
            promotion.workflow_id,
            artifact.workflow_id,
        ),
        (
            "dataset_id",
            promotion.dataset_id,
            artifact.dataset_id,
        ),
        (
            "problem_type",
            promotion.problem_type,
            artifact
            .training_contract
            .problem_type,
        ),
        (
            "estimator_key",
            promotion.estimator_key,
            artifact
            .training_contract
            .estimator_key,
        ),
        (
            "preparation_session_revision",
            promotion
            .preparation_session_revision,
            provenance
            .preparation_session_revision,
        ),
        (
            "model_id",
            promotion.model_id,
            artifact.model_id,
        ),
        (
            "experiment_id",
            promotion.experiment_id,
            provenance.experiment_id,
        ),
        (
            "promoted_training_contract_sha256",
            promotion
            .promoted_training_contract_sha256,
            training_contract_sha256,
        ),
        (
            "train_rows",
            promotion.train_rows,
            artifact.train_rows,
        ),
        (
            "test_rows",
            promotion.test_rows,
            artifact.test_rows,
        ),
    )


    for (
        field_name,
        actual,
        expected,
    ) in identity_checks:

        if actual != expected:
            raise (
                MLModelEvaluationSummarySelectionError(
                    (
                        "Tuned Model Promotion context "
                        "does not match the evaluated "
                        "Model Artifact. "
                        f"field={field_name}"
                    )
                )
            )


    if (
        promotion.final_metrics
        !=
        artifact.metrics
    ):
        raise (
            MLModelEvaluationSummarySelectionError(
                (
                    "Tuned Model Promotion final metrics "
                    "do not match persisted artifact "
                    "metrics."
                )
            )
        )


    return (
        MLModelSelectionEvidence(
            source=
                "tuned_model_promotion",

            status=
                "verified_selected",

            rank=
                1,

            selection_policy=
                "rank_1_only",

            primary_metric=(
                promotion
                .tuning_primary_metric
            ),

            primary_metric_value=(
                promotion
                .tuning_primary_metric_mean
            ),

            metric_scope=(
                "inner_cross_validation"
            ),
        )
    )


# ============================================================
# SELECTION AUTHORITY
# ============================================================


def _build_selection_evidence(
    *,
    artifact,
    provenance,
    training_contract_sha256: str,
    baseline,
    selection_context: Any,
) -> MLModelSelectionEvidence:

    if selection_context is None:

        return (
            MLModelSelectionEvidence(
                source=
                    "standalone_model",

                status=(
                    "selection_not_available"
                ),

                metric_scope=
                    "not_available",
            )
        )


    if isinstance(
        selection_context,
        MLModelComparisonExecutionResult,
    ):

        return (
            _selection_from_model_comparison(
                artifact=
                    artifact,

                provenance=
                    provenance,

                training_contract_sha256=(
                    training_contract_sha256
                ),

                comparison_context=(
                    selection_context
                ),

                baseline=
                    baseline,
            )
        )


    if isinstance(
        selection_context,
        MLTunedModelPromotionExecutionResult,
    ):

        return (
            _selection_from_tuned_promotion(
                artifact=
                    artifact,

                provenance=
                    provenance,

                training_contract_sha256=(
                    training_contract_sha256
                ),

                promotion_context=(
                    selection_context
                ),
            )
        )


    raise (
        MLModelEvaluationSummarySelectionError(
            (
                "Unsupported server-owned selection "
                "context. Model Evaluation Summary "
                "will not infer selection provenance "
                "from artifact characteristics."
            )
        )
    )


# ============================================================
# BASELINE RECONSTRUCTION
# ============================================================


def _reconstruct_baseline(
    *,
    problem_type: str,
    y_train,
    y_test,
    model_metrics: dict[
        str,
        float
    ],
    train_rows: int,
    test_rows: int,
):

    try:
        baseline_prediction_bundle = (
            build_ml_baseline_predictions(
                problem_type=
                    problem_type,

                y_train=
                    y_train,

                test_rows=
                    test_rows,
            )
        )


        if (
            problem_type
            ==
            "regression"
        ):

            richer_baseline_metrics = (
                compute_ml_regression_metrics(
                    y_true=
                        y_test,

                    predictions=(
                        baseline_prediction_bundle
                        .predictions
                    ),
                )
            )

        else:

            richer_baseline_metrics = (
                compute_ml_classification_metrics(
                    y_true=
                        y_test,

                    predictions=(
                        baseline_prediction_bundle
                        .predictions
                    ),
                )
            )


        baseline_metrics = (
            project_ml_baseline_metrics_v0_1(
                problem_type=
                    problem_type,

                metrics=(
                    richer_baseline_metrics
                ),
            )
        )


        baseline = (
            build_ml_baseline_evaluation(
                problem_type=
                    problem_type,

                strategy=(
                    baseline_prediction_bundle
                    .strategy
                ),

                metrics=
                    baseline_metrics,

                train_rows=
                    train_rows,

                test_rows=
                    test_rows,
            )
        )


        comparison = (
            compare_model_to_baseline(
                problem_type=
                    problem_type,

                model_metrics=
                    model_metrics,

                baseline_metrics=(
                    baseline.metrics
                ),
            )
        )


    except (
        MLBaselineError,
        Exception,
    ) as error:

        if isinstance(
            error,
            MLModelEvaluationSummaryExecutorError,
        ):
            raise


        raise (
            MLModelEvaluationSummaryExecutionError(
                (
                    "Model Evaluation Summary could "
                    "not reconstruct the original "
                    "training-only baseline."
                )
            )
        ) from error


    return (
        baseline,
        comparison,
    )


# ============================================================
# EXECUTION
# ============================================================


def execute_ml_model_evaluation_summary(
    *,
    workflow_id: str,
    model_id: str,
    summary_contract: (
        MLModelEvaluationSummaryContract
    ),
    selection_context: Any = None,
) -> MLModelEvaluationSummaryResult:
    """
    Build one deterministic privacy-minimal evaluation summary
    for one already persisted trusted Model Artifact.

    Public request authority
    ------------------------
    Caller-controlled:
    - workflow_id;
    - model_id;
    - MLModelEvaluationSummaryContract;
    - optional explicit Decision Threshold inside that contract.

    `selection_context` is NOT public request authority.

    It is an optional server-owned orchestration argument that may
    contain a validated Model Comparison or Tuned Model Promotion
    execution result from the same trusted server flow.

    If it is absent, selection evidence is deliberately reported
    as unavailable rather than inferred.

    This executor MUST NOT:
    - train;
    - fit/refit;
    - re-rank models;
    - select another model;
    - infer selection provenance;
    - optimize a Decision Threshold;
    - persist a new Model Artifact or Experiment.
    """

    normalized_workflow_id = (
        _required_identifier(
            workflow_id,
            field_name=
                "workflow_id",
        )
    )


    normalized_model_id = (
        _required_identifier(
            model_id,
            field_name=
                "model_id",
        )
    )


    try:
        config = (
            MLModelEvaluationSummaryContract
            .model_validate(
                summary_contract
            )
        )

    except Exception as error:
        raise (
            MLModelEvaluationSummaryInputError(
                (
                    "Model Evaluation Summary "
                    "contract is invalid."
                )
            )
        ) from error


    # ========================================================
    # TRUSTED MODEL ARTIFACT
    # ========================================================


    try:
        loaded_model = (
            load_trusted_ml_model(
                workflow_id=(
                    normalized_workflow_id
                ),

                model_id=(
                    normalized_model_id
                ),
            )
        )

    except MLModelLoaderError as error:
        raise (
            MLModelEvaluationSummaryArtifactError(
                (
                    "Model Evaluation Summary refused "
                    "because the trusted Model Artifact "
                    "could not be restored."
                )
            )
        ) from error


    artifact = (
        loaded_model.artifact
    )


    if (
        artifact.workflow_id
        !=
        normalized_workflow_id
    ):
        raise (
            MLModelEvaluationSummaryArtifactError(
                (
                    "Trusted Model Artifact workflow "
                    "does not match Summary request."
                )
            )
        )


    if (
        artifact.model_id
        !=
        normalized_model_id
    ):
        raise (
            MLModelEvaluationSummaryArtifactError(
                (
                    "Trusted Model Artifact identity "
                    "does not match Summary request."
                )
            )
        )


    # ========================================================
    # ARTIFACT-OWNED TRAINING CONTRACT
    # ========================================================


    try:
        training_contract = (
            MLTrainingContract
            .model_validate(
                artifact.training_contract
            )
        )

    except Exception as error:
        raise (
            MLModelEvaluationSummaryArtifactError(
                (
                    "Trusted Model Artifact contains "
                    "an invalid ML Training Contract."
                )
            )
        ) from error


    if (
        training_contract.workflow_id
        !=
        artifact.workflow_id
        or
        training_contract.dataset_id
        !=
        artifact.dataset_id
    ):
        raise (
            MLModelEvaluationSummaryArtifactError(
                (
                    "Model Artifact and Training "
                    "Contract scope identities differ."
                )
            )
        )


    # ========================================================
    # EXPERIMENT PROVENANCE
    # ========================================================


    provenance = (
        artifact.experiment_provenance
    )


    if provenance is None:
        raise (
            MLModelEvaluationSummaryArtifactError(
                (
                    "Model Evaluation Summary requires "
                    "Experiment Provenance."
                )
            )
        )


    training_contract_sha256 = (
        ml_training_contract_sha256(
            training_contract
        )
    )


    if (
        provenance.training_contract_sha256
        !=
        training_contract_sha256
    ):
        raise (
            MLModelEvaluationSummaryArtifactError(
                (
                    "Experiment Provenance Training "
                    "Contract SHA does not match the "
                    "trusted Model Artifact."
                )
            )
        )


    if (
        provenance.workflow_id
        !=
        artifact.workflow_id
        or
        provenance.dataset_id
        !=
        artifact.dataset_id
        or
        provenance.model_id
        !=
        artifact.model_id
    ):
        raise (
            MLModelEvaluationSummaryArtifactError(
                (
                    "Experiment Provenance identity "
                    "does not match the trusted "
                    "Model Artifact."
                )
            )
        )


    if (
        provenance.metrics
        !=
        artifact.metrics
    ):
        raise (
            MLModelEvaluationSummaryArtifactError(
                (
                    "Experiment Provenance metrics "
                    "do not match persisted Model "
                    "Artifact metrics."
                )
            )
        )


    if (
        provenance.train_rows
        !=
        artifact.train_rows
        or
        provenance.test_rows
        !=
        artifact.test_rows
    ):
        raise (
            MLModelEvaluationSummaryArtifactError(
                (
                    "Experiment Provenance holdout "
                    "shape does not match persisted "
                    "Model Artifact."
                )
            )
        )


    # ========================================================
    # PROBLEM-SPECIFIC REQUEST GATE
    # ========================================================


    if (
        training_contract.problem_type
        ==
        "regression"
        and
        config.decision_threshold
        is not None
    ):
        raise (
            MLModelEvaluationSummaryInputError(
                (
                    "Decision Threshold evaluation is "
                    "available only for binary "
                    "classification Model Artifacts."
                )
            )
        )


    # ========================================================
    # CURRENT PREPARATION + EXACT HOLDOUT
    # ========================================================


    try:
        (
            dataframe,
            current_preparation_revision,
        ) = (
            _load_authorized_dataframe(
                contract=
                    training_contract
            )
        )


        (
            x,
            y,
        ) = (
            _validate_and_extract_xy(
                dataframe=
                    dataframe,

                contract=
                    training_contract,
            )
        )


        (
            x_train,
            x_test,
            y_train,
            y_test,
        ) = (
            _split_dataset(
                x=
                    x,

                y=
                    y,

                contract=
                    training_contract,
            )
        )


    except ClassicalMLInputError as error:
        raise (
            MLModelEvaluationSummaryInputError(
                (
                    "Model Evaluation Summary could "
                    "not reconstruct the validated "
                    "training / holdout split."
                )
            )
        ) from error


    if (
        current_preparation_revision
        !=
        provenance
        .preparation_session_revision
    ):
        raise (
            MLModelEvaluationSummaryInputError(
                (
                    "Preparation revision changed "
                    "since the evaluated Model Artifact "
                    "was trained."
                )
            )
        )


    train_rows = int(
        len(
            x_train
        )
    )


    test_rows = int(
        len(
            x_test
        )
    )


    if (
        train_rows
        !=
        artifact.train_rows
        or
        test_rows
        !=
        artifact.test_rows
    ):
        raise (
            MLModelEvaluationSummaryInputError(
                (
                    "Reconstructed holdout shape does "
                    "not match the trusted Model "
                    "Artifact."
                )
            )
        )


    if (
        len(
            y_train
        )
        !=
        train_rows
        or
        len(
            y_test
        )
        !=
        test_rows
    ):
        raise (
            MLModelEvaluationSummaryInputError(
                (
                    "Reconstructed feature and target "
                    "split shapes are inconsistent."
                )
            )
        )


    # ========================================================
    # BASELINE
    #
    # Baseline learns ONLY from y_train.
    # No persisted estimator fit/refit occurs.
    # ========================================================


    (
        baseline,
        baseline_comparison,
    ) = (
        _reconstruct_baseline(
            problem_type=(
                training_contract
                .problem_type
            ),

            y_train=
                y_train,

            y_test=
                y_test,

            model_metrics=
                artifact.metrics,

            train_rows=
                train_rows,

            test_rows=
                test_rows,
        )
    )


    # ========================================================
    # SELECTION EVIDENCE
    # ========================================================


    selection_evidence = (
        _build_selection_evidence(
            artifact=
                artifact,

            provenance=
                provenance,

            training_contract_sha256=(
                training_contract_sha256
            ),

            baseline=
                baseline,

            selection_context=(
                selection_context
            ),
        )
    )


    # ========================================================
    # EXPLAINABILITY
    # ========================================================


    try:
        explainability = (
            execute_ml_model_explainability(
                workflow_id=(
                    artifact.workflow_id
                ),

                model_id=(
                    artifact.model_id
                ),

                explainability_contract=(
                    MLModelExplainabilityContract()
                ),
            )
        )

    except MLModelExplainabilityExecutorError as error:
        raise (
            MLModelEvaluationSummaryExecutionError(
                (
                    "Model Evaluation Summary failed "
                    "while reconstructing trusted "
                    "Explainability evidence."
                )
            )
        ) from error


    # ========================================================
    # CLASSIFICATION EVIDENCE
    # ========================================================


    classification_diagnostics = None
    decision_threshold_evaluation = None


    if (
        training_contract.problem_type
        ==
        "classification"
    ):

        try:
            classification_diagnostics = (
                execute_ml_classification_diagnostics(
                    workflow_id=(
                        artifact.workflow_id
                    ),

                    model_id=(
                        artifact.model_id
                    ),

                    diagnostics_contract=(
                        MLClassificationDiagnosticsContract()
                    ),
                )
            )

        except (
            MLClassificationDiagnosticsExecutorError
        ) as error:
            raise (
                MLModelEvaluationSummaryExecutionError(
                    (
                        "Model Evaluation Summary failed "
                        "while reconstructing trusted "
                        "Classification Diagnostics."
                    )
                )
            ) from error


        if (
            config.decision_threshold
            is not None
        ):

            try:
                decision_threshold_evaluation = (
                    execute_ml_decision_threshold(
                        workflow_id=(
                            artifact.workflow_id
                        ),

                        model_id=(
                            artifact.model_id
                        ),

                        threshold_contract=(
                            config
                            .decision_threshold
                        ),
                    )
                )

            except (
                MLDecisionThresholdExecutorError
            ) as error:
                raise (
                    MLModelEvaluationSummaryExecutionError(
                        (
                            "Model Evaluation Summary "
                            "failed while reconstructing "
                            "trusted Decision Threshold "
                            "evidence."
                        )
                    )
                ) from error


    # ========================================================
    # SERVER-DERIVED LIMITATIONS
    # ========================================================


    limitations = (
        expected_model_evaluation_limitations(
            problem_type=(
                training_contract
                .problem_type
            ),

            selection_source=(
                selection_evidence
                .source
            ),

            threshold_requested=(
                config.decision_threshold
                is not None
            ),
        )
    )


    # ========================================================
    # FINAL CROSS-EVIDENCE VALIDATION
    # ========================================================


    try:
        return (
            MLModelEvaluationSummaryResult(
                workflow_id=
                    artifact.workflow_id,

                dataset_id=
                    artifact.dataset_id,

                model_id=
                    artifact.model_id,

                experiment_id=(
                    provenance
                    .experiment_id
                ),

                problem_type=(
                    training_contract
                    .problem_type
                ),

                target_column=(
                    training_contract
                    .target_column
                ),

                estimator_key=(
                    training_contract
                    .estimator_key
                ),

                preparation_session_revision=(
                    current_preparation_revision
                ),

                training_contract_sha256=(
                    training_contract_sha256
                ),

                train_rows=
                    train_rows,

                test_rows=
                    test_rows,

                summary_contract=
                    config,

                metrics=
                    dict(
                        artifact.metrics
                    ),

                baseline=
                    baseline,

                baseline_comparison=(
                    baseline_comparison
                ),

                selection_evidence=(
                    selection_evidence
                ),

                classification_diagnostics=(
                    classification_diagnostics
                ),

                decision_threshold_evaluation=(
                    decision_threshold_evaluation
                ),

                explainability=
                    explainability,

                limitations=
                    limitations,
            )
        )

    except Exception as error:
        raise (
            MLModelEvaluationSummaryExecutionError(
                (
                    "Model Evaluation Summary evidence "
                    "failed final cross-source "
                    "consistency validation."
                )
            )
        ) from error
