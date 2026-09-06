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


from app.ml.baseline import (
    MLBaselineComparisonResult,
    MLBaselineEvaluationResult,
)


from app.ml.experiment_provenance import (
    MLExperimentProvenanceRecord,
    ml_training_contract_sha256,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_CANDIDATE_RESULT_RULE_VERSION = (
    "ml_model_candidate_result_v0.1"
)


# ============================================================
# RESULT
# ============================================================


class MLModelCandidateResult(
    BaseModel
):
    """
    Framework-neutral, privacy-minimal result for one trained
    Model Lab candidate.

    It intentionally does not expose:
    - raw rows;
    - holdout row positions;
    - predictions;
    - fitted preprocessing objects;
    - fitted estimator / neural-network objects;
    - serialized model bytes.

    Framework-specific executors project into this contract.
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


    estimator_key: str = Field(
        min_length=1,
    )


    train_rows: int = Field(
        gt=0,
    )


    test_rows: int = Field(
        gt=0,
    )


    purged_rows: int = Field(
        default=0,
        ge=0,
    )


    metrics: dict[
        str,
        float,
    ]


    baseline: MLBaselineEvaluationResult


    baseline_comparison: (
        MLBaselineComparisonResult
    )


    experiment_provenance: (
        MLExperimentProvenanceRecord
    )


    model_artifact: MLModelArtifactRecord


    rule_version: Literal[
        "ml_model_candidate_result_v0.1"
    ] = ML_MODEL_CANDIDATE_RESULT_RULE_VERSION


    @model_validator(
        mode="after"
    )
    def validate_candidate_consistency(
        self,
    ) -> "MLModelCandidateResult":

        artifact = (
            self.model_artifact
        )


        provenance = (
            self.experiment_provenance
        )


        contract = (
            artifact.training_contract
        )


        # ----------------------------------------------------
        # EXECUTION / CONTRACT AUTHORITY
        # ----------------------------------------------------


        if (
            self.workflow_id
            !=
            contract.workflow_id
        ):
            raise ValueError(
                (
                    "Candidate workflow_id does not "
                    "match Model Artifact contract."
                )
            )


        if (
            self.dataset_id
            !=
            contract.dataset_id
        ):
            raise ValueError(
                (
                    "Candidate dataset_id does not "
                    "match Model Artifact contract."
                )
            )


        if (
            self.problem_type
            !=
            contract.problem_type
        ):
            raise ValueError(
                (
                    "Candidate problem_type does not "
                    "match Model Artifact contract."
                )
            )


        if (
            self.estimator_key
            !=
            contract.estimator_key
        ):
            raise ValueError(
                (
                    "Candidate estimator_key does not "
                    "match Model Artifact contract."
                )
            )


        # ----------------------------------------------------
        # METRICS / ROW COUNTS
        # ----------------------------------------------------


        if (
            self.train_rows
            !=
            artifact.train_rows
            or
            self.test_rows
            !=
            artifact.test_rows
        ):
            raise ValueError(
                (
                    "Candidate holdout shape does not "
                    "match Model Artifact."
                )
            )


        if (
            self.metrics
            !=
            artifact.metrics
        ):
            raise ValueError(
                (
                    "Candidate metrics do not match "
                    "Model Artifact metrics."
                )
            )


        for value in (
            self.metrics.values()
        ):

            if not math.isfinite(
                float(
                    value
                )
            ):
                raise ValueError(
                    (
                        "Candidate metrics must "
                        "be finite."
                    )
                )


        # ----------------------------------------------------
        # BASELINE AUTHORITY
        # ----------------------------------------------------


        if (
            self.baseline.problem_type
            !=
            self.problem_type
        ):
            raise ValueError(
                (
                    "Candidate baseline problem type "
                    "does not match candidate."
                )
            )


        if (
            self.baseline.train_rows
            !=
            self.train_rows
            or
            self.baseline.test_rows
            !=
            self.test_rows
        ):
            raise ValueError(
                (
                    "Candidate baseline holdout shape "
                    "does not match candidate."
                )
            )


        if (
            self.baseline_comparison.problem_type
            !=
            self.problem_type
        ):
            raise ValueError(
                (
                    "Candidate baseline comparison "
                    "problem type does not match."
                )
            )


        primary_metric = (
            self.baseline_comparison
            .primary_metric
        )


        if (
            primary_metric
            not in
            self.metrics
        ):
            raise ValueError(
                (
                    "Candidate baseline comparison "
                    "primary metric is absent from "
                    "candidate metrics."
                )
            )


        if not math.isclose(
            self.baseline_comparison
            .model_primary_metric_value,
            float(
                self.metrics[
                    primary_metric
                ]
            ),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                (
                    "Candidate baseline comparison "
                    "does not reference candidate "
                    "primary metric."
                )
            )


        # ----------------------------------------------------
        # PROVENANCE AUTHORITY
        # ----------------------------------------------------


        if (
            provenance
            !=
            artifact.experiment_provenance
        ):
            raise ValueError(
                (
                    "Candidate and Model Artifact "
                    "must expose identical "
                    "Experiment Provenance."
                )
            )


        if (
            provenance.workflow_id
            !=
            self.workflow_id
            or
            provenance.dataset_id
            !=
            self.dataset_id
        ):
            raise ValueError(
                (
                    "Candidate Experiment Provenance "
                    "scope does not match candidate."
                )
            )


        if (
            provenance
            .preparation_session_revision
            !=
            self.preparation_session_revision
        ):
            raise ValueError(
                (
                    "Candidate Preparation revision "
                    "does not match Experiment "
                    "Provenance."
                )
            )


        if (
            provenance.model_id
            !=
            artifact.model_id
        ):
            raise ValueError(
                (
                    "Candidate Experiment Provenance "
                    "model_id does not match artifact."
                )
            )


        if (
            provenance.train_rows
            !=
            self.train_rows
            or
            provenance.test_rows
            !=
            self.test_rows
        ):
            raise ValueError(
                (
                    "Candidate Experiment Provenance "
                    "holdout shape does not match."
                )
            )


        if (
            provenance.metrics
            !=
            self.metrics
        ):
            raise ValueError(
                (
                    "Candidate Experiment Provenance "
                    "metrics do not match."
                )
            )


        if (
            provenance.training_contract_sha256
            !=
            ml_training_contract_sha256(
                contract
            )
        ):
            raise ValueError(
                (
                    "Candidate Experiment Provenance "
                    "Training Contract fingerprint "
                    "does not match artifact."
                )
            )


        return self


# ============================================================
# FRAMEWORK-NEUTRAL PROJECTION
# ============================================================


def project_ml_model_candidate_result(
    execution_result: Any,
) -> MLModelCandidateResult:
    """
    Project one privacy-minimal Model Lab execution result into
    the shared candidate surface.

    This module deliberately does not import Classical or Deep
    Learning executor implementations.
    """

    required_attributes = (
        "workflow_id",
        "dataset_id",
        "problem_type",
        "estimator_key",
        "train_rows",
        "test_rows",
        "purged_rows",
        "metrics",
        "baseline",
        "baseline_comparison",
        "experiment_provenance",
        "model_artifact",
    )


    missing = [
        attribute

        for attribute
        in required_attributes

        if not hasattr(
            execution_result,
            attribute,
        )
    ]


    if missing:
        raise ValueError(
            (
                "Model Lab execution result cannot "
                "be projected to a shared candidate. "
                "Missing attributes: "
                +
                ", ".join(
                    missing
                )
            )
        )


    provenance = getattr(
        execution_result,
        "experiment_provenance",
    )


    preparation_session_revision = getattr(
        execution_result,
        "preparation_session_revision",
        None,
    )


    if (
        preparation_session_revision
        is None
    ):

        preparation_session_revision = getattr(
            provenance,
            "preparation_session_revision",
            None,
        )


    return (
        MLModelCandidateResult(
            workflow_id=
                execution_result.workflow_id,

            dataset_id=
                execution_result.dataset_id,

            preparation_session_revision=
                preparation_session_revision,

            problem_type=
                execution_result.problem_type,

            estimator_key=
                execution_result.estimator_key,

            train_rows=
                execution_result.train_rows,

            test_rows=
                execution_result.test_rows,

            purged_rows=
                execution_result.purged_rows,

            metrics=
                execution_result.metrics,

            baseline=
                execution_result.baseline,

            baseline_comparison=
                execution_result.baseline_comparison,

            experiment_provenance=
                execution_result.experiment_provenance,

            model_artifact=
                execution_result.model_artifact,
        )
    )
