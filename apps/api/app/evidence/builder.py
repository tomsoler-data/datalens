from typing import (
    Any,
)

from pydantic import (
    BaseModel,
)

from app.dashboard.schemas import (
    DashboardSpec,
)

from app.evidence.schemas import (
    AnalysisEvidenceBundle,
    EvidenceRecord,
)

from app.statistics.schemas import (
    CorrelationExecution,
    CorrelationTestDecision,
)

from app.visualization.schemas import (
    VisualizationDecision,
)


# ============================================================
# CANONICAL IDS
# ============================================================

DECISION_EVIDENCE_ID = (
    "decision:0001"
)

STATISTIC_EVIDENCE_ID = (
    "statistic:0001"
)

VISUALIZATION_EVIDENCE_ID = (
    "visualization:0001"
)

DASHBOARD_EVIDENCE_ID = (
    "dashboard:0001"
)


# ============================================================
# ERROR
# ============================================================

class EvidenceBuildError(
    ValueError
):
    """
    Raised when DataLens cannot safely create
    a canonical evidence bundle.
    """

    pass


# ============================================================
# MODEL SERIALIZATION
# ============================================================

def model_to_evidence_data(
    model: BaseModel,
) -> dict[
    str,
    Any,
]:
    """
    Convert a Pydantic model into JSON-safe
    deterministic evidence data.
    """

    return model.model_dump(
        mode="json",
    )


# ============================================================
# CONSISTENCY VALIDATION
# ============================================================

def validate_evidence_inputs(
    decision: CorrelationTestDecision,
    visualization: VisualizationDecision,
    dashboard: DashboardSpec,
    execution: CorrelationExecution | None,
) -> None:
    """
    Validate the dependency chain before
    creating canonical evidence records.

    This is intentionally redundant with
    upstream validation.

    Evidence is the trust boundary.
    """

    # ========================================================
    # VARIABLES
    # ========================================================

    if (
        visualization.x_column
        != decision.x_column
        or visualization.y_column
        != decision.y_column
    ):
        raise EvidenceBuildError(
            (
                "Visualization variables do not "
                "match the statistical decision."
            )
        )

    if (
        dashboard.x_column
        != decision.x_column
        or dashboard.y_column
        != decision.y_column
    ):
        raise EvidenceBuildError(
            (
                "Dashboard variables do not match "
                "the statistical decision."
            )
        )

    # ========================================================
    # VISUALIZATION ID
    # ========================================================

    if (
        visualization.visualization_id
        != VISUALIZATION_EVIDENCE_ID
    ):
        raise EvidenceBuildError(
            (
                "Unexpected visualization "
                "evidence ID: "
                f"{visualization.visualization_id!r}."
            )
        )

    # ========================================================
    # DASHBOARD ID
    # ========================================================

    if (
        dashboard.dashboard_id
        != DASHBOARD_EVIDENCE_ID
    ):
        raise EvidenceBuildError(
            (
                "Unexpected dashboard evidence "
                "ID: "
                f"{dashboard.dashboard_id!r}."
            )
        )

    # ========================================================
    # DASHBOARD REFERENCES
    # ========================================================

    if (
        dashboard.evidence.decision
        != DECISION_EVIDENCE_ID
    ):
        raise EvidenceBuildError(
            (
                "Dashboard decision reference "
                "does not match the canonical "
                "decision evidence ID."
            )
        )

    if (
        dashboard.evidence.visualization
        != VISUALIZATION_EVIDENCE_ID
    ):
        raise EvidenceBuildError(
            (
                "Dashboard visualization "
                "reference does not match the "
                "canonical visualization "
                "evidence ID."
            )
        )

    # ========================================================
    # NO EXECUTION
    # ========================================================

    if execution is None:
        if (
            dashboard.evidence.statistic
            is not None
        ):
            raise EvidenceBuildError(
                (
                    "Dashboard references a "
                    "statistical result even "
                    "though no execution exists."
                )
            )

        if (
            dashboard.statistical_result
            is not None
        ):
            raise EvidenceBuildError(
                (
                    "Dashboard contains a "
                    "statistical result even "
                    "though no execution exists."
                )
            )

        return

    # ========================================================
    # EXECUTION EXISTS
    # ========================================================

    if (
        dashboard.evidence.statistic
        != STATISTIC_EVIDENCE_ID
    ):
        raise EvidenceBuildError(
            (
                "Dashboard statistical reference "
                "does not match the canonical "
                "statistical evidence ID."
            )
        )

    if (
        dashboard.statistical_result
        is None
    ):
        raise EvidenceBuildError(
            (
                "Execution exists but the "
                "dashboard has no statistical "
                "result block."
            )
        )

    if (
        execution.x_column
        != decision.x_column
        or execution.y_column
        != decision.y_column
    ):
        raise EvidenceBuildError(
            (
                "Execution variables do not "
                "match the statistical decision."
            )
        )

    if (
        decision.status
        != "selected"
    ):
        raise EvidenceBuildError(
            (
                "A statistical execution exists "
                "for a decision that did not "
                "select a test."
            )
        )

    if (
        execution.selected_test
        != decision.selected_test
    ):
        raise EvidenceBuildError(
            (
                "Execution test does not match "
                "the selected statistical test."
            )
        )

    if (
        execution.result.test
        != execution.selected_test
    ):
        raise EvidenceBuildError(
            (
                "Execution result does not match "
                "the executed statistical test."
            )
        )


# ============================================================
# DECISION EVIDENCE
# ============================================================

def build_decision_evidence(
    dataset: str,
    decision: CorrelationTestDecision,
) -> EvidenceRecord:
    """
    Why was the test selected or refused?
    """

    return EvidenceRecord(
        evidence_id=
            DECISION_EVIDENCE_ID,

        source_type=
            "statistical_decision",

        dataset=
            dataset,

        producer=
            "python",

        rule_version=
            decision.decision_rule_version,

        depends_on=[],

        data=
            model_to_evidence_data(
                decision
            ),
    )


# ============================================================
# STATISTICAL EVIDENCE
# ============================================================

def build_statistic_evidence(
    dataset: str,
    execution: CorrelationExecution,
) -> EvidenceRecord:
    """
    What did the selected test calculate?

    The nested decision stored inside the
    CorrelationExecution object is deliberately
    not duplicated here.

    It is represented by the explicit dependency
    on decision:0001 instead.
    """

    data = {
        "x_column":
            execution.x_column,

        "y_column":
            execution.y_column,

        "selected_test":
            execution.selected_test,

        "inference_method_used":
            execution.inference_method_used,

        "permutation_mode":
            execution.permutation_mode,

        "permutation_resamples_requested":
            execution
            .permutation_resamples_requested,

        "random_seed":
            execution.random_seed,

        "n_total":
            execution.n_total,

        "n_valid":
            execution.n_valid,

        "n_excluded":
            execution.n_excluded,

        "result":
            execution.result.model_dump(
                mode="json",
            ),

        "warnings":
            list(
                execution.warnings
            ),

        "execution_rule_version":
            execution
            .execution_rule_version,
    }

    return EvidenceRecord(
        evidence_id=
            STATISTIC_EVIDENCE_ID,

        source_type=
            "statistical_result",

        dataset=
            dataset,

        producer=
            "python",

        rule_version=
            execution
            .execution_rule_version,

        depends_on=[
            DECISION_EVIDENCE_ID,
        ],

        data=
            data,
    )


# ============================================================
# VISUALIZATION EVIDENCE
# ============================================================

def build_visualization_evidence(
    dataset: str,
    visualization: VisualizationDecision,
) -> EvidenceRecord:
    """
    Why was this visualization selected?
    """

    return EvidenceRecord(
        evidence_id=
            VISUALIZATION_EVIDENCE_ID,

        source_type=
            "visualization_decision",

        dataset=
            dataset,

        producer=
            "python",

        rule_version=
            visualization
            .visualization_rule_version,

        depends_on=[
            DECISION_EVIDENCE_ID,
        ],

        data=
            model_to_evidence_data(
                visualization
            ),
    )


# ============================================================
# DASHBOARD EVIDENCE
# ============================================================

def build_dashboard_evidence(
    dataset: str,
    dashboard: DashboardSpec,
    execution: CorrelationExecution | None,
) -> EvidenceRecord:
    """
    How were the validated analytical outputs
    composed into a dashboard specification?
    """

    dependencies = [
        DECISION_EVIDENCE_ID,
        VISUALIZATION_EVIDENCE_ID,
    ]

    if execution is not None:
        dependencies.insert(
            1,
            STATISTIC_EVIDENCE_ID,
        )

    return EvidenceRecord(
        evidence_id=
            DASHBOARD_EVIDENCE_ID,

        source_type=
            "dashboard_spec",

        dataset=
            dataset,

        producer=
            "python",

        rule_version=
            dashboard
            .dashboard_rule_version,

        depends_on=
            dependencies,

        data=
            model_to_evidence_data(
                dashboard
            ),
    )


# ============================================================
# COMPLETE BUNDLE
# ============================================================

def build_analysis_evidence_bundle(
    dataset: str,
    decision: CorrelationTestDecision,
    visualization: VisualizationDecision,
    dashboard: DashboardSpec,
    execution: CorrelationExecution | None = None,
) -> AnalysisEvidenceBundle:
    """
    Build the canonical DataLens evidence bundle
    for one association analysis.

    Ordering is intentional:

    decision
        ↓
    statistic, when available
        ↓
    visualization
        ↓
    dashboard
    """

    if not (
        dataset.strip()
    ):
        raise EvidenceBuildError(
            (
                "dataset must not be empty."
            )
        )

    validate_evidence_inputs(
        decision=
            decision,

        visualization=
            visualization,

        dashboard=
            dashboard,

        execution=
            execution,
    )

    evidence = [
        build_decision_evidence(
            dataset=
                dataset,

            decision=
                decision,
        )
    ]

    if execution is not None:
        evidence.append(
            build_statistic_evidence(
                dataset=
                    dataset,

                execution=
                    execution,
            )
        )

    evidence.append(
        build_visualization_evidence(
            dataset=
                dataset,

            visualization=
                visualization,
        )
    )

    evidence.append(
        build_dashboard_evidence(
            dataset=
                dataset,

            dashboard=
                dashboard,

            execution=
                execution,
        )
    )

    return AnalysisEvidenceBundle(
        dataset=
            dataset,

        evidence=
            evidence,
    )