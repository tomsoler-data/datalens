import pandas as pd

from app.analysis.schemas import (
    CorrelationAnalysisRun,
)

from app.dashboard import (
    compose_correlation_dashboard,
)

from app.evidence import (
    build_analysis_evidence_bundle,
)

from app.statistics import (
    DEFAULT_ALPHA,
    DEFAULT_PERMUTATION_RESAMPLES,
    DEFAULT_RANDOM_SEED,
    decide_correlation_test,
    execute_correlation_decision,
)

from app.statistics.schemas import (
    AnalysisGoal,
    AnalysisMode,
    VariableKind,
)

from app.visualization import (
    decide_correlation_visualization,
)


# ============================================================
# PIPELINE ERROR
# ============================================================

class AnalysisPipelineError(
    ValueError
):
    """
    Raised when DataLens cannot safely build
    the deterministic analysis pipeline.
    """

    pass


# ============================================================
# STATUS
# ============================================================

def determine_analysis_status(
    decision_status: str,
) -> str:
    """
    Convert a statistical decision status into
    the final analysis lifecycle status.
    """

    if (
        decision_status
        == "selected"
    ):
        return "complete"

    if (
        decision_status
        == "needs_information"
    ):
        return "needs_information"

    if (
        decision_status
        == "not_applicable"
    ):
        return "not_applicable"

    raise AnalysisPipelineError(
        (
            "Unsupported statistical decision "
            f"status: {decision_status!r}."
        )
    )


# ============================================================
# MAIN CORRELATION PIPELINE
# ============================================================

def run_correlation_analysis(
    dataframe: pd.DataFrame,
    dataset: str,
    x_column: str,
    y_column: str,
    analysis_goal: AnalysisGoal,
    analysis_mode: AnalysisMode,
    x_kind: VariableKind,
    y_kind: VariableKind,
    observations_independent: (
        bool | None
    ),
    alpha: float = DEFAULT_ALPHA,
    permutation_resamples: int = (
        DEFAULT_PERMUTATION_RESAMPLES
    ),
    random_seed: int = (
        DEFAULT_RANDOM_SEED
    ),
) -> CorrelationAnalysisRun:
    """
    Run the complete deterministic DataLens
    correlation analysis.

    Pipeline:

        dataframe
            ↓
        Statistical Decision Engine
            ↓
        Statistical Executor
            ↓
        Visualization Decision Engine
            ↓
        Dashboard Composer
            ↓
        Evidence Layer
            ↓
        CorrelationAnalysisRun

    The LLM is intentionally absent.

    The future local LLM consumes the resulting
    evidence rather than controlling the
    calculations.
    """

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    dataset_name = (
        dataset.strip()
    )

    if not dataset_name:
        raise AnalysisPipelineError(
            "dataset must not be empty."
        )

    if not (
        x_column.strip()
    ):
        raise AnalysisPipelineError(
            "x_column must not be empty."
        )

    if not (
        y_column.strip()
    ):
        raise AnalysisPipelineError(
            "y_column must not be empty."
        )

    if (
        x_column
        == y_column
    ):
        raise AnalysisPipelineError(
            (
                "x_column and y_column must refer "
                "to different variables."
            )
        )

    # ========================================================
    # 1. STATISTICAL DECISION
    # ========================================================

    decision = (
        decide_correlation_test(
            dataframe=
                dataframe,

            x_column=
                x_column,

            y_column=
                y_column,

            analysis_goal=
                analysis_goal,

            analysis_mode=
                analysis_mode,

            x_kind=
                x_kind,

            y_kind=
                y_kind,

            observations_independent=
                observations_independent,
        )
    )

    # ========================================================
    # 2. STATISTICAL EXECUTION
    # ========================================================

    execution = None

    if (
        decision.status
        == "selected"
    ):
        execution = (
            execute_correlation_decision(
                dataframe=
                    dataframe,

                decision=
                    decision,

                alpha=
                    alpha,

                permutation_resamples=
                    permutation_resamples,

                random_seed=
                    random_seed,
            )
        )

    # ========================================================
    # 3. VISUALIZATION DECISION
    # ========================================================

    visualization = (
        decide_correlation_visualization(
            decision
        )
    )

    # ========================================================
    # 4. DASHBOARD COMPOSITION
    # ========================================================

    dashboard = (
        compose_correlation_dashboard(
            decision=
                decision,

            visualization=
                visualization,

            execution=
                execution,
        )
    )

    # ========================================================
    # 5. CANONICAL EVIDENCE
    # ========================================================

    evidence = (
        build_analysis_evidence_bundle(
            dataset=
                dataset_name,

            decision=
                decision,

            visualization=
                visualization,

            dashboard=
                dashboard,

            execution=
                execution,
        )
    )

    # ========================================================
    # 6. FINAL STATUS
    # ========================================================

    status = (
        determine_analysis_status(
            decision.status
        )
    )

    # ========================================================
    # 7. FINAL VALIDATED ANALYSIS
    # ========================================================

    return CorrelationAnalysisRun(
        dataset=
            dataset_name,

        status=
            status,

        decision=
            decision,

        execution=
            execution,

        visualization=
            visualization,

        dashboard=
            dashboard,

        evidence=
            evidence,
    )