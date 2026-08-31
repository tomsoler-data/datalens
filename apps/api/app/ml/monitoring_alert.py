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


from app.ml.model_health import (
    MLModelHealthEvidenceAlignment,
    MLModelHealthReason,
    MLModelHealthStatus,
    MLModelHealthSummary,
)


# ============================================================
# VERSION
# ============================================================


ML_MONITORING_ALERT_RULE_VERSION = (
    "ml_monitoring_alert_v0.1"
)


# ============================================================
# TYPES
# ============================================================


MLMonitoringAlertCategory = Literal[
    "none",
    "monitoring_gap",
    "evidence_alignment_gap",
    "data_shift",
    "performance_warning",
    "performance_degradation",
]


MLMonitoringAlertSeverity = Literal[
    "none",
    "info",
    "warning",
    "critical",
]


MLMonitoringAlertAction = Literal[
    "no_action",
    "establish_monitoring_evidence",
    "complete_monitoring_evidence",
    "align_monitoring_snapshots",
    "review_observed_data_distribution",
    "review_model_performance",
    "investigate_model_degradation",
]


MLMonitoringAlertPrivacyScope = Literal[
    "aggregate_only",
]


# ============================================================
# ERRORS
# ============================================================


class MLMonitoringAlertError(
    RuntimeError
):
    pass


class MLMonitoringAlertInputError(
    MLMonitoringAlertError
):
    pass


# ============================================================
# POLICY
# ============================================================


def _alert_policy(
    *,
    health_status: MLModelHealthStatus,
    health_reason: MLModelHealthReason,
) -> tuple[
    MLMonitoringAlertCategory,
    MLMonitoringAlertSeverity,
    MLMonitoringAlertAction,
    bool,
]:
    """
    Return:

        category,
        severity,
        recommended_action,
        notification_recommended

    This policy intentionally separates:

    - operational monitoring gaps;
    - data distribution shifts;
    - predictive degradation.

    A Drift signal alone never becomes a critical model-quality
    alert.
    """

    # --------------------------------------------------------
    # HEALTHY
    # --------------------------------------------------------


    if (
        health_status
        ==
        "healthy"
        and
        health_reason
        ==
        "aligned_evidence_ok"
    ):
        return (
            "none",
            "none",
            "no_action",
            False,
        )


    # --------------------------------------------------------
    # NO MONITORING EVIDENCE AT ALL
    # --------------------------------------------------------


    if (
        health_status
        ==
        "insufficient_evidence"
        and
        health_reason
        ==
        "no_monitoring_evidence"
    ):
        return (
            "monitoring_gap",
            "warning",
            "establish_monitoring_evidence",
            True,
        )


    # --------------------------------------------------------
    # PARTIAL MONITORING COVERAGE
    #
    # One evidence branch is healthy, but the other branch is
    # missing.
    #
    # This should remain visible operationally without becoming
    # a noisy notification by default.
    # --------------------------------------------------------


    if (
        health_status
        ==
        "insufficient_evidence"
        and
        health_reason
        in {
            "drift_only_ok",
            "performance_only_ok",
        }
    ):
        return (
            "monitoring_gap",
            "info",
            "complete_monitoring_evidence",
            False,
        )


    # --------------------------------------------------------
    # EVIDENCE ALIGNMENT GAP
    # --------------------------------------------------------


    if (
        health_status
        ==
        "insufficient_evidence"
        and
        health_reason
        in {
            "evidence_misaligned",
            "evidence_unverifiable",
        }
    ):
        return (
            "evidence_alignment_gap",
            "warning",
            "align_monitoring_snapshots",
            True,
        )


    # --------------------------------------------------------
    # DATA SHIFT
    #
    # Includes Drift warning and strong Drift.
    #
    # Model Health deliberately keeps these at attention unless
    # Performance itself is degraded.
    # --------------------------------------------------------


    if (
        health_status
        ==
        "attention"
        and
        health_reason
        ==
        "drift_signal"
    ):
        return (
            "data_shift",
            "warning",
            "review_observed_data_distribution",
            True,
        )


    # --------------------------------------------------------
    # PERFORMANCE WARNING
    # --------------------------------------------------------


    if (
        health_status
        ==
        "attention"
        and
        health_reason
        ==
        "performance_warning"
    ):
        return (
            "performance_warning",
            "warning",
            "review_model_performance",
            True,
        )


    # --------------------------------------------------------
    # PERFORMANCE DEGRADATION
    # --------------------------------------------------------


    if (
        health_status
        ==
        "critical"
        and
        health_reason
        ==
        "performance_degraded"
    ):
        return (
            "performance_degradation",
            "critical",
            "investigate_model_degradation",
            True,
        )


    raise MLMonitoringAlertInputError(
        (
            "Model Health state is not supported "
            "by Monitoring Alert policy v0.1. "
            f"health_status={health_status}, "
            f"health_reason={health_reason}"
        )
    )


# ============================================================
# ALERT DECISION
# ============================================================


class MLMonitoringAlertDecision(
    BaseModel
):
    """
    Derived aggregate-only operational decision.

    This is not an alert-delivery event and is not persisted.

    It contains references to the monitoring evidence used by
    Model Health, but no raw data, predictions, targets,
    probabilities, estimator state, model bytes or file paths.
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


    alert_active: bool


    alert_category: (
        MLMonitoringAlertCategory
    )


    severity: (
        MLMonitoringAlertSeverity
    )


    recommended_action: (
        MLMonitoringAlertAction
    )


    notification_recommended: bool


    drift_evaluation_id: (
        str
        |
        None
    ) = None


    performance_evaluation_id: (
        str
        |
        None
    ) = None


    privacy_scope: (
        MLMonitoringAlertPrivacyScope
    ) = "aggregate_only"


    rule_version: Literal[
        "ml_monitoring_alert_v0.1"
    ] = ML_MONITORING_ALERT_RULE_VERSION


    # ========================================================
    # TEXT
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
    # POLICY CONSISTENCY
    # ========================================================


    @model_validator(
        mode="after"
    )
    def validate_policy(
        self,
    ) -> "MLMonitoringAlertDecision":

        (
            expected_category,
            expected_severity,
            expected_action,
            expected_notification,
        ) = (
            _alert_policy(
                health_status=
                    self.health_status,

                health_reason=
                    self.health_reason,
            )
        )


        if (
            self.alert_category
            !=
            expected_category
        ):
            raise ValueError(
                (
                    "alert_category does not "
                    "match Monitoring Alert "
                    "policy v0.1."
                )
            )


        if (
            self.severity
            !=
            expected_severity
        ):
            raise ValueError(
                (
                    "severity does not match "
                    "Monitoring Alert policy "
                    "v0.1."
                )
            )


        if (
            self.recommended_action
            !=
            expected_action
        ):
            raise ValueError(
                (
                    "recommended_action does not "
                    "match Monitoring Alert "
                    "policy v0.1."
                )
            )


        if (
            self.notification_recommended
            is not
            expected_notification
        ):
            raise ValueError(
                (
                    "notification_recommended "
                    "does not match Monitoring "
                    "Alert policy v0.1."
                )
            )


        expected_active = (
            expected_category
            !=
            "none"
        )


        if (
            self.alert_active
            is not
            expected_active
        ):
            raise ValueError(
                (
                    "alert_active does not match "
                    "Monitoring Alert policy "
                    "v0.1."
                )
            )


        if (
            self.alert_active
            is False
        ):

            if (
                self.severity
                !=
                "none"
            ):
                raise ValueError(
                    (
                        "Inactive alert requires "
                        "severity=none."
                    )
                )


            if (
                self.notification_recommended
                is not False
            ):
                raise ValueError(
                    (
                        "Inactive alert cannot "
                        "recommend notification."
                    )
                )


        return self


# ============================================================
# BUILDER
# ============================================================


def build_ml_monitoring_alert_decision(
    *,
    model_health: MLModelHealthSummary,
) -> MLMonitoringAlertDecision:
    """
    Derive one operational Monitoring Alert decision from one
    validated Model Health Summary.

    No monitoring computation, model loading, persistence or
    notification delivery occurs here.
    """

    if not isinstance(
        model_health,
        MLModelHealthSummary,
    ):
        raise MLMonitoringAlertInputError(
            (
                "model_health must be a validated "
                "MLModelHealthSummary."
            )
        )


    (
        category,
        severity,
        action,
        notification,
    ) = (
        _alert_policy(
            health_status=
                model_health.health_status,

            health_reason=
                model_health.health_reason,
        )
    )


    return (
        MLMonitoringAlertDecision(
            workflow_id=
                model_health.workflow_id,

            model_id=
                model_health.model_id,

            health_status=
                model_health.health_status,

            health_reason=
                model_health.health_reason,

            evidence_alignment=
                model_health.evidence_alignment,

            joint_interpretation_allowed=(
                model_health
                .joint_interpretation_allowed
            ),

            alert_active=(
                category
                !=
                "none"
            ),

            alert_category=
                category,

            severity=
                severity,

            recommended_action=
                action,

            notification_recommended=
                notification,

            drift_evaluation_id=(
                model_health
                .drift_evaluation_id
            ),

            performance_evaluation_id=(
                model_health
                .performance_evaluation_id
            ),
        )
    )
