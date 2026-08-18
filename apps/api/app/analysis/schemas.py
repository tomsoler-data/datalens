from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)

from app.dashboard.schemas import (
    DashboardSpec,
)

from app.evidence.schemas import (
    AnalysisEvidenceBundle,
)

from app.statistics.schemas import (
    CorrelationExecution,
    CorrelationTestDecision,
)

from app.visualization.schemas import (
    VisualizationDecision,
)


# ============================================================
# ANALYSIS STATUS
# ============================================================

AnalysisRunStatus = Literal[
    "complete",
    "needs_information",
    "not_applicable",
]


# ============================================================
# COMPLETE ANALYSIS RUN
# ============================================================

class CorrelationAnalysisRun(
    BaseModel
):
    """
    Complete deterministic DataLens correlation
    analysis.

    Pipeline:

        statistical decision
            ↓
        statistical execution, when allowed
            ↓
        visualization decision
            ↓
        dashboard composition
            ↓
        canonical evidence bundle

    No LLM calculation or decision occurs here.
    """

    analysis_id: str = Field(
        default="analysis:0001",
        min_length=1,
    )

    dataset: str = Field(
        min_length=1,
    )

    status: AnalysisRunStatus

    decision: CorrelationTestDecision

    execution: (
        CorrelationExecution | None
    ) = None

    visualization: VisualizationDecision

    dashboard: DashboardSpec

    evidence: AnalysisEvidenceBundle

    pipeline_rule_version: str = (
        "correlation_analysis_pipeline_v0.1"
    )

    @model_validator(
        mode="after"
    )
    def validate_pipeline_consistency(
        self,
    ):
        """
        Validate the complete deterministic
        analysis chain.
        """

        # ====================================================
        # DATASET
        # ====================================================

        if (
            self.evidence.dataset
            != self.dataset
        ):
            raise ValueError(
                (
                    "Evidence bundle dataset does "
                    "not match the analysis dataset."
                )
            )

        # ====================================================
        # VARIABLES
        # ====================================================

        if (
            self.visualization.x_column
            != self.decision.x_column
        ):
            raise ValueError(
                (
                    "Visualization x_column does "
                    "not match the statistical "
                    "decision."
                )
            )

        if (
            self.visualization.y_column
            != self.decision.y_column
        ):
            raise ValueError(
                (
                    "Visualization y_column does "
                    "not match the statistical "
                    "decision."
                )
            )

        if (
            self.dashboard.x_column
            != self.decision.x_column
        ):
            raise ValueError(
                (
                    "Dashboard x_column does not "
                    "match the statistical "
                    "decision."
                )
            )

        if (
            self.dashboard.y_column
            != self.decision.y_column
        ):
            raise ValueError(
                (
                    "Dashboard y_column does not "
                    "match the statistical "
                    "decision."
                )
            )

        # ====================================================
        # COMPLETE
        # ====================================================

        if (
            self.status
            == "complete"
        ):
            if (
                self.decision.status
                != "selected"
            ):
                raise ValueError(
                    (
                        "A complete analysis must "
                        "originate from a selected "
                        "statistical decision."
                    )
                )

            if (
                self.execution
                is None
            ):
                raise ValueError(
                    (
                        "A complete analysis must "
                        "contain a statistical "
                        "execution."
                    )
                )

            if (
                self.dashboard.status
                != "complete"
            ):
                raise ValueError(
                    (
                        "A complete analysis must "
                        "contain a complete "
                        "dashboard."
                    )
                )

        # ====================================================
        # NEEDS INFORMATION
        # ====================================================

        if (
            self.status
            == "needs_information"
        ):
            if (
                self.decision.status
                != "needs_information"
            ):
                raise ValueError(
                    (
                        "A needs_information "
                        "analysis must originate "
                        "from a needs_information "
                        "decision."
                    )
                )

            if (
                self.execution
                is not None
            ):
                raise ValueError(
                    (
                        "A needs_information "
                        "analysis must not contain "
                        "a statistical execution."
                    )
                )

            if (
                self.dashboard.status
                != "needs_information"
            ):
                raise ValueError(
                    (
                        "The dashboard must also "
                        "have needs_information "
                        "status."
                    )
                )

        # ====================================================
        # NOT APPLICABLE
        # ====================================================

        if (
            self.status
            == "not_applicable"
        ):
            if (
                self.decision.status
                != "not_applicable"
            ):
                raise ValueError(
                    (
                        "A not_applicable analysis "
                        "must originate from a "
                        "not_applicable decision."
                    )
                )

            if (
                self.execution
                is not None
            ):
                raise ValueError(
                    (
                        "A not_applicable analysis "
                        "must not contain a "
                        "statistical execution."
                    )
                )

            if (
                self.dashboard.status
                != "not_applicable"
            ):
                raise ValueError(
                    (
                        "The dashboard must also "
                        "have not_applicable "
                        "status."
                    )
                )

        # ====================================================
        # EXECUTION
        # ====================================================

        if (
            self.execution
            is not None
        ):
            if (
                self.execution.x_column
                != self.decision.x_column
            ):
                raise ValueError(
                    (
                        "Execution x_column does "
                        "not match the statistical "
                        "decision."
                    )
                )

            if (
                self.execution.y_column
                != self.decision.y_column
            ):
                raise ValueError(
                    (
                        "Execution y_column does "
                        "not match the statistical "
                        "decision."
                    )
                )

            if (
                self.execution.selected_test
                != self.decision.selected_test
            ):
                raise ValueError(
                    (
                        "Executed test does not "
                        "match the selected test."
                    )
                )

        # ====================================================
        # EVIDENCE IDS
        # ====================================================

        evidence_ids = {
            item.evidence_id
            for item
            in self.evidence.evidence
        }

        required_ids = {
            "decision:0001",
            "visualization:0001",
            "dashboard:0001",
        }

        missing_ids = (
            required_ids
            - evidence_ids
        )

        if missing_ids:
            raise ValueError(
                (
                    "Analysis evidence is missing "
                    "required records: "
                    f"{sorted(missing_ids)}."
                )
            )

        has_statistic = (
            "statistic:0001"
            in evidence_ids
        )

        if (
            self.execution
            is None
            and has_statistic
        ):
            raise ValueError(
                (
                    "Statistical evidence exists "
                    "without statistical "
                    "execution."
                )
            )

        if (
            self.execution
            is not None
            and not has_statistic
        ):
            raise ValueError(
                (
                    "Statistical execution exists "
                    "without statistical "
                    "evidence."
                )
            )

        return self