from __future__ import annotations


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


from app.ml.drift_evaluation import (
    MLDriftEvaluationRecord,
    MLDriftStatus,
)


from app.ml.performance_evaluation import (
    MLPerformanceEvaluationRecord,
    MLPerformanceStatus,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_HEALTH_RULE_VERSION = (
    "ml_model_health_v0.1"
)


# ============================================================
# TYPES
# ============================================================


MLModelHealthStatus = Literal[
    "insufficient_evidence",
    "healthy",
    "attention",
    "critical",
]


MLModelHealthEvidenceAlignment = Literal[
    "none",
    "single_source",
    "aligned",
    "misaligned",
    "unverifiable",
]


MLModelHealthReason = Literal[
    "no_monitoring_evidence",
    "drift_only_ok",
    "performance_only_ok",
    "aligned_evidence_ok",
    "evidence_misaligned",
    "evidence_unverifiable",
    "drift_signal",
    "performance_warning",
    "performance_degraded",
]


MLModelHealthPrivacyScope = Literal[
    "aggregate_only",
]


# ============================================================
# ERRORS
# ============================================================


class MLModelHealthError(
    RuntimeError
):
    pass


class MLModelHealthInputError(
    MLModelHealthError
):
    pass


class MLModelHealthAuthorityError(
    MLModelHealthError
):
    pass


# ============================================================
# TEXT
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
        raise MLModelHealthInputError(
            (
                f"{field_name} "
                "cannot be empty."
            )
        )


    return normalized


# ============================================================
# ALIGNMENT
# ============================================================


def _evidence_alignment(
    *,
    drift: MLDriftEvaluationRecord | None,
    performance: MLPerformanceEvaluationRecord | None,
) -> MLModelHealthEvidenceAlignment:

    if (
        drift is None
        and
        performance is None
    ):
        return "none"


    if (
        drift is None
        or
        performance is None
    ):
        return "single_source"


    if (
        drift
        .observed_preparation_session_revision
        is None
    ):
        return "unverifiable"


    if (
        drift.observed_dataset_id
        ==
        performance.observed_dataset_id
        and
        drift.observed_preparation_session_revision
        ==
        performance
        .observed_preparation_session_revision
        and
        drift.observed_row_count
        ==
        performance.observed_row_count
    ):
        return "aligned"


    return "misaligned"


# ============================================================
# HEALTH POLICY
# ============================================================


def _health_policy(
    *,
    drift_status: MLDriftStatus | None,
    performance_status: MLPerformanceStatus | None,
    evidence_alignment: MLModelHealthEvidenceAlignment,
) -> tuple[
    MLModelHealthStatus,
    MLModelHealthReason,
]:

    # --------------------------------------------------------
    # PERFORMANCE DEGRADATION HAS HIGHEST OPERATIONAL PRIORITY
    #
    # It remains critical even when Drift evidence describes a
    # different snapshot. In that case the degradation itself
    # is valid, but no causal Drift interpretation is allowed.
    # --------------------------------------------------------


    if (
        performance_status
        ==
        "degraded"
    ):
        return (
            "critical",
            "performance_degraded",
        )


    # --------------------------------------------------------
    # PERFORMANCE WARNING
    # --------------------------------------------------------


    if (
        performance_status
        ==
        "warning"
    ):
        return (
            "attention",
            "performance_warning",
        )


    # --------------------------------------------------------
    # DATA DRIFT SIGNAL
    # --------------------------------------------------------


    if (
        drift_status
        in {
            "warning",
            "drift",
        }
    ):
        return (
            "attention",
            "drift_signal",
        )


    # --------------------------------------------------------
    # NO EVIDENCE
    # --------------------------------------------------------


    if (
        drift_status is None
        and
        performance_status is None
    ):
        return (
            "insufficient_evidence",
            "no_monitoring_evidence",
        )


    # --------------------------------------------------------
    # ONLY PERFORMANCE OK
    #
    # No Drift evidence means DataLens cannot claim that the
    # current feature distribution is also healthy.
    # --------------------------------------------------------


    if (
        drift_status is None
    ):
        return (
            "insufficient_evidence",
            "performance_only_ok",
        )


    # --------------------------------------------------------
    # ONLY DRIFT OK
    #
    # No supervised Performance evidence means DataLens cannot
    # claim that predictive quality remains healthy.
    # --------------------------------------------------------


    if (
        performance_status is None
    ):
        return (
            "insufficient_evidence",
            "drift_only_ok",
        )


    # --------------------------------------------------------
    # BOTH SOURCES ARE OK
    #
    # Healthy requires evidence describing the exact same
    # server-owned observed snapshot.
    # --------------------------------------------------------


    if (
        evidence_alignment
        ==
        "aligned"
    ):
        return (
            "healthy",
            "aligned_evidence_ok",
        )


    if (
        evidence_alignment
        ==
        "misaligned"
    ):
        return (
            "insufficient_evidence",
            "evidence_misaligned",
        )


    if (
        evidence_alignment
        ==
        "unverifiable"
    ):
        return (
            "insufficient_evidence",
            "evidence_unverifiable",
        )


    raise MLModelHealthAuthorityError(
        (
            "Model Health policy received "
            "an impossible evidence state."
        )
    )


# ============================================================
# SUMMARY CONTRACT
# ============================================================


class MLModelHealthSummary(
    BaseModel
):
    """
    Aggregate-only operational interpretation of the latest
    persisted Drift and Performance evidence for one Model
    Artifact.

    This is derived evidence.

    It contains no:
    - raw rows;
    - raw feature values;
    - raw targets;
    - predictions;
    - probabilities;
    - model bytes;
    - filesystem paths.

    It does not itself prove causality between Drift and
    Performance degradation.

    joint_interpretation_allowed is True only when Drift and
    Performance describe the same observed server-owned
    Preparation snapshot.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    workflow_id: str = Field(
        min_length=1,
    )


    model_id: str = Field(
        min_length=1,
    )


    health_status: MLModelHealthStatus


    health_reason: MLModelHealthReason


    evidence_alignment: (
        MLModelHealthEvidenceAlignment
    )


    joint_interpretation_allowed: bool


    # ========================================================
    # LATEST DRIFT EVIDENCE
    # ========================================================


    drift_evaluation_id: (
        str
        |
        None
    ) = None


    drift_status: (
        MLDriftStatus
        |
        None
    ) = None


    drift_observed_dataset_id: (
        str
        |
        None
    ) = None


    drift_observed_preparation_session_revision: (
        int
        |
        None
    ) = Field(
        default=None,
        ge=0,
    )


    drift_observed_row_count: (
        int
        |
        None
    ) = Field(
        default=None,
        gt=0,
    )


    drift_evaluated_at_utc: (
        str
        |
        None
    ) = None


    # ========================================================
    # LATEST PERFORMANCE EVIDENCE
    # ========================================================


    performance_evaluation_id: (
        str
        |
        None
    ) = None


    performance_status: (
        MLPerformanceStatus
        |
        None
    ) = None


    performance_observed_dataset_id: (
        str
        |
        None
    ) = None


    performance_observed_preparation_session_revision: (
        int
        |
        None
    ) = Field(
        default=None,
        ge=0,
    )


    performance_observed_row_count: (
        int
        |
        None
    ) = Field(
        default=None,
        gt=0,
    )


    performance_evaluated_at_utc: (
        str
        |
        None
    ) = None


    privacy_scope: (
        MLModelHealthPrivacyScope
    ) = "aggregate_only"


    rule_version: Literal[
        "ml_model_health_v0.1"
    ] = ML_MODEL_HEALTH_RULE_VERSION


    # ========================================================
    # REQUIRED TEXT
    # ========================================================


    @field_validator(
        "workflow_id",
        "model_id",
        mode="before",
    )
    @classmethod
    def normalize_required_text(
        cls,
        value: object,
        info,
    ) -> str:

        normalized = str(
            value
            if value is not None
            else ""
        ).strip()


        if not normalized:
            raise ValueError(
                (
                    f"{info.field_name} "
                    "cannot be empty."
                )
            )


        return normalized


    # ========================================================
    # INTERNAL CONSISTENCY
    # ========================================================


    @model_validator(
        mode="after"
    )
    def validate_summary(
        self,
    ) -> "MLModelHealthSummary":

        drift_present = (
            self.drift_evaluation_id
            is not None
        )


        performance_present = (
            self.performance_evaluation_id
            is not None
        )


        # ----------------------------------------------------
        # DRIFT EVIDENCE GROUP
        #
        # Revision may legitimately remain None for restored
        # pre-v12 Drift records.
        # ----------------------------------------------------


        drift_required = [
            self.drift_status,
            self.drift_observed_dataset_id,
            self.drift_observed_row_count,
            self.drift_evaluated_at_utc,
        ]


        if drift_present:

            if any(
                value is None

                for value
                in drift_required
            ):
                raise ValueError(
                    (
                        "Drift evidence fields must "
                        "be complete when a Drift "
                        "Evaluation identity exists."
                    )
                )

        else:

            if any(
                value is not None

                for value
                in drift_required
            ):
                raise ValueError(
                    (
                        "Drift evidence fields cannot "
                        "exist without a Drift "
                        "Evaluation identity."
                    )
                )


            if (
                self
                .drift_observed_preparation_session_revision
                is not None
            ):
                raise ValueError(
                    (
                        "Drift Preparation revision "
                        "cannot exist without Drift "
                        "evidence."
                    )
                )


        # ----------------------------------------------------
        # PERFORMANCE EVIDENCE GROUP
        # ----------------------------------------------------


        performance_required = [
            self.performance_status,
            self.performance_observed_dataset_id,
            (
                self
                .performance_observed_preparation_session_revision
            ),
            self.performance_observed_row_count,
            self.performance_evaluated_at_utc,
        ]


        if performance_present:

            if any(
                value is None

                for value
                in performance_required
            ):
                raise ValueError(
                    (
                        "Performance evidence fields "
                        "must be complete when a "
                        "Performance Evaluation "
                        "identity exists."
                    )
                )

        else:

            if any(
                value is not None

                for value
                in performance_required
            ):
                raise ValueError(
                    (
                        "Performance evidence fields "
                        "cannot exist without a "
                        "Performance Evaluation "
                        "identity."
                    )
                )


        # ----------------------------------------------------
        # EXPECTED ALIGNMENT
        # ----------------------------------------------------


        if (
            not drift_present
            and
            not performance_present
        ):
            expected_alignment = (
                "none"
            )

        elif (
            not drift_present
            or
            not performance_present
        ):
            expected_alignment = (
                "single_source"
            )

        elif (
            self
            .drift_observed_preparation_session_revision
            is None
        ):
            expected_alignment = (
                "unverifiable"
            )

        elif (
            self.drift_observed_dataset_id
            ==
            self.performance_observed_dataset_id
            and
            self
            .drift_observed_preparation_session_revision
            ==
            self
            .performance_observed_preparation_session_revision
            and
            self.drift_observed_row_count
            ==
            self.performance_observed_row_count
        ):
            expected_alignment = (
                "aligned"
            )

        else:
            expected_alignment = (
                "misaligned"
            )


        if (
            self.evidence_alignment
            !=
            expected_alignment
        ):
            raise ValueError(
                (
                    "Model Health evidence_alignment "
                    "does not match evidence "
                    "identities."
                )
            )


        expected_joint = (
            expected_alignment
            ==
            "aligned"
        )


        if (
            self.joint_interpretation_allowed
            is not
            expected_joint
        ):
            raise ValueError(
                (
                    "joint_interpretation_allowed "
                    "does not match evidence "
                    "alignment."
                )
            )


        # ----------------------------------------------------
        # EXPECTED HEALTH POLICY
        # ----------------------------------------------------


        expected_status, expected_reason = (
            _health_policy(
                drift_status=
                    self.drift_status,

                performance_status=
                    self.performance_status,

                evidence_alignment=
                    expected_alignment,
            )
        )


        if (
            self.health_status
            !=
            expected_status
        ):
            raise ValueError(
                (
                    "Model Health status does not "
                    "match v0.1 policy."
                )
            )


        if (
            self.health_reason
            !=
            expected_reason
        ):
            raise ValueError(
                (
                    "Model Health reason does not "
                    "match v0.1 policy."
                )
            )


        return self


# ============================================================
# TRAINING AUTHORITY
# ============================================================


def _assert_shared_training_authority(
    *,
    drift: MLDriftEvaluationRecord,
    performance: MLPerformanceEvaluationRecord,
) -> None:

    bindings = [
        (
            "model_id",
            drift.model_id,
            performance.model_id,
        ),
        (
            "workflow_id",
            drift.workflow_id,
            performance.workflow_id,
        ),
        (
            "reference_dataset_id",
            drift.reference_dataset_id,
            performance.reference_dataset_id,
        ),
        (
            "experiment_id",
            drift.experiment_id,
            performance.experiment_id,
        ),
        (
            "preparation_session_revision",
            drift.preparation_session_revision,
            performance.preparation_session_revision,
        ),
        (
            "training_contract_sha256",
            drift.training_contract_sha256,
            performance.training_contract_sha256,
        ),
    ]


    for (
        field_name,
        drift_value,
        performance_value,
    ) in bindings:

        if (
            drift_value
            !=
            performance_value
        ):
            raise MLModelHealthAuthorityError(
                (
                    "Drift and Performance evidence "
                    "do not share the same Model "
                    "Artifact training authority. "
                    f"field={field_name}"
                )
            )


# ============================================================
# BUILDER
# ============================================================


def build_ml_model_health_summary(
    *,
    workflow_id: str,
    model_id: str,
    latest_drift: (
        MLDriftEvaluationRecord
        |
        None
    ),
    latest_performance: (
        MLPerformanceEvaluationRecord
        |
        None
    ),
) -> MLModelHealthSummary:
    """
    Derive one deterministic Model Health Summary from already
    persisted aggregate monitoring evidence.

    This function performs no:
    - model loading;
    - prediction;
    - Drift computation;
    - metric computation;
    - persistence.
    """

    normalized_workflow_id = (
        _required_text(
            workflow_id,
            field_name=
                "workflow_id",
        )
    )


    normalized_model_id = (
        _required_text(
            model_id,
            field_name=
                "model_id",
        )
    )


    # ========================================================
    # TYPE AUTHORITY
    # ========================================================


    if (
        latest_drift is not None
        and
        not isinstance(
            latest_drift,
            MLDriftEvaluationRecord,
        )
    ):
        raise MLModelHealthInputError(
            (
                "latest_drift must be a validated "
                "MLDriftEvaluationRecord."
            )
        )


    if (
        latest_performance is not None
        and
        not isinstance(
            latest_performance,
            MLPerformanceEvaluationRecord,
        )
    ):
        raise MLModelHealthInputError(
            (
                "latest_performance must be a "
                "validated "
                "MLPerformanceEvaluationRecord."
            )
        )


    # ========================================================
    # REQUEST IDENTITY AUTHORITY
    # ========================================================


    for (
        evidence_name,
        evidence,
    ) in [
        (
            "Drift",
            latest_drift,
        ),
        (
            "Performance",
            latest_performance,
        ),
    ]:

        if evidence is None:
            continue


        if (
            evidence.workflow_id
            !=
            normalized_workflow_id
        ):
            raise MLModelHealthAuthorityError(
                (
                    f"{evidence_name} evidence "
                    "does not belong to the "
                    "requested workflow."
                )
            )


        if (
            evidence.model_id
            !=
            normalized_model_id
        ):
            raise MLModelHealthAuthorityError(
                (
                    f"{evidence_name} evidence "
                    "does not belong to the "
                    "requested Model Artifact."
                )
            )


    # ========================================================
    # SHARED TRAINING AUTHORITY
    # ========================================================


    if (
        latest_drift is not None
        and
        latest_performance is not None
    ):
        _assert_shared_training_authority(
            drift=
                latest_drift,

            performance=
                latest_performance,
        )


    alignment = (
        _evidence_alignment(
            drift=
                latest_drift,

            performance=
                latest_performance,
        )
    )


    drift_status = (
        latest_drift.overall_status

        if latest_drift is not None

        else None
    )


    performance_status = (
        latest_performance.performance_status

        if latest_performance is not None

        else None
    )


    health_status, health_reason = (
        _health_policy(
            drift_status=
                drift_status,

            performance_status=
                performance_status,

            evidence_alignment=
                alignment,
        )
    )


    return (
        MLModelHealthSummary(
            workflow_id=
                normalized_workflow_id,

            model_id=
                normalized_model_id,

            health_status=
                health_status,

            health_reason=
                health_reason,

            evidence_alignment=
                alignment,

            joint_interpretation_allowed=(
                alignment
                ==
                "aligned"
            ),

            drift_evaluation_id=(
                latest_drift.evaluation_id

                if latest_drift is not None

                else None
            ),

            drift_status=
                drift_status,

            drift_observed_dataset_id=(
                latest_drift.observed_dataset_id

                if latest_drift is not None

                else None
            ),

            drift_observed_preparation_session_revision=(
                latest_drift
                .observed_preparation_session_revision

                if latest_drift is not None

                else None
            ),

            drift_observed_row_count=(
                latest_drift.observed_row_count

                if latest_drift is not None

                else None
            ),

            drift_evaluated_at_utc=(
                latest_drift.evaluated_at_utc

                if latest_drift is not None

                else None
            ),

            performance_evaluation_id=(
                latest_performance
                .performance_evaluation_id

                if latest_performance is not None

                else None
            ),

            performance_status=
                performance_status,

            performance_observed_dataset_id=(
                latest_performance
                .observed_dataset_id

                if latest_performance is not None

                else None
            ),

            performance_observed_preparation_session_revision=(
                latest_performance
                .observed_preparation_session_revision

                if latest_performance is not None

                else None
            ),

            performance_observed_row_count=(
                latest_performance.observed_row_count

                if latest_performance is not None

                else None
            ),

            performance_evaluated_at_utc=(
                latest_performance.evaluated_at_utc

                if latest_performance is not None

                else None
            ),
        )
    )
