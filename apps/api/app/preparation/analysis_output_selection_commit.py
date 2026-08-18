from __future__ import annotations

from typing import (
    List,
)

from pydantic import (
    BaseModel,
)

from app.preparation.analysis_output_selection import (
    ANALYSIS_OUTPUT_SELECTION_RULE_VERSION,
    AnalysisOutputSelectionReport,
    require_analysis_output_selection,
)

from app.preparation.preparation_session import (
    PreparationSessionView,
    get_preparation_session,
    record_analysis_output_selection,
)


# ============================================================
# VERSION
# ============================================================


ANALYSIS_OUTPUT_SELECTION_COMMIT_VERSION = (
    "analysis_output_selection_commit_v0.1"
)


# ============================================================
# RESULT
# ============================================================


class AnalysisOutputSelectionCommitResult(
    BaseModel,
):
    workflow_id: str

    previous_revision: int

    committed_revision: int

    analysis_output_dataset_ids: List[
        str
    ]

    selection: AnalysisOutputSelectionReport

    session: PreparationSessionView

    rule_version: str = (
        ANALYSIS_OUTPUT_SELECTION_COMMIT_VERSION
    )


# ============================================================
# COMMIT
# ============================================================


def commit_analysis_output_selection(
    *,
    workflow_id: str,
    requested_dataset_ids: List[
        str
    ],
) -> AnalysisOutputSelectionCommitResult:
    """
    Validate and atomically commit the final analytical output
    scope.

    Trust sequence:

        PreparationSession
                ↓
        PreparationArtifactStore
                ↓
        deterministic lineage validation
                ↓
        selection report PASS
                ↓
        optimistic revision check
                ↓
        server-owned session commit

    The browser cannot directly write
    analysis_output_dataset_ids through this function.
    """

    session = (
        get_preparation_session(
            workflow_id
        )
    )


    selection = (
        require_analysis_output_selection(
            session=
                session,

            requested_dataset_ids=
                requested_dataset_ids,
        )
    )


    updated_session = (
        record_analysis_output_selection(
            workflow_id=
                session.workflow_id,

            analysis_output_dataset_ids=(
                selection
                .selected_analysis_output_dataset_ids
            ),

            expected_revision=
                session.revision,
        )
    )


    return (
        AnalysisOutputSelectionCommitResult(
            workflow_id=
                session.workflow_id,

            previous_revision=
                session.revision,

            committed_revision=
                updated_session.revision,

            analysis_output_dataset_ids=list(
                updated_session
                .analysis_output_dataset_ids
            ),

            selection=
                selection,

            session=
                updated_session,
        )
    )