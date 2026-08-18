from __future__ import annotations

from typing import (
    List,
    Optional,
    Set,
)

from pydantic import (
    BaseModel,
    Field,
)

from app.preparation.preparation_session import (
    get_preparation_session,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
)


# ============================================================
# VERSION
# ============================================================


ANALYSIS_READINESS_GATE_RULE_VERSION = (
    "analysis_readiness_gate_v0.2"
)


# ============================================================
# ERRORS
# ============================================================


class AnalysisReadinessError(
    RuntimeError,
):
    pass


class AnalysisNotReadyError(
    AnalysisReadinessError,
):
    """
    Raised when a requested analytical dataset belongs to the
    authorized Preparation scope but the workflow or final
    validation is not ready yet.
    """

    def __init__(
        self,
        *,
        decision: "AnalysisReadinessDecision",
    ) -> None:
        self.decision = (
            decision
        )

        super().__init__(
            (
                "Analysis refused because Preparation "
                "has not reached READY FOR ANALYSIS "
                "for every requested dataset."
            )
        )


class AnalysisDatasetNotAuthorizedError(
    AnalysisReadinessError,
):
    """
    Raised when an analysis attempts to use a dataset outside
    the currently authorized Preparation scope.

    Before a final analytical output has been selected, the
    immutable Preparation roots form a provisional scope.

    Once analysis_output_dataset_ids exists, only that final
    output scope is authorized for analytical execution.
    """

    def __init__(
        self,
        *,
        decision: "AnalysisReadinessDecision",
    ) -> None:
        self.decision = (
            decision
        )

        super().__init__(
            (
                "Analysis refused because one or more "
                "requested datasets are outside the "
                "authorized Preparation output scope."
            )
        )


# ============================================================
# DECISION
# ============================================================


class AnalysisReadinessDecision(
    BaseModel,
):
    workflow_id: str

    session_revision: int

    ready_for_analysis: bool

    workflow_ready_for_analysis: bool

    dataset_scope_authorized: bool

    requested_datasets_validated: bool

    # ========================================================
    # IMMUTABLE PREPARATION ROOTS
    # ========================================================

    selected_analysis_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    # ========================================================
    # FINAL ANALYTICAL OUTPUTS
    # ========================================================

    analysis_output_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    # ========================================================
    # EFFECTIVE AUTHORIZATION SCOPE
    # ========================================================

    authorized_analysis_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    # ========================================================
    # VALIDATED OUTPUTS
    # ========================================================

    validated_analysis_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    # ========================================================
    # REQUEST
    # ========================================================

    requested_analysis_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    unauthorized_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    unvalidated_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    next_stage: Optional[
        PreparationStage
    ] = None

    blocking_reasons: List[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        ANALYSIS_READINESS_GATE_RULE_VERSION
    )


# ============================================================
# NORMALIZATION
# ============================================================


def _normalize_workflow_id(
    workflow_id: str,
) -> str:
    value = (
        workflow_id.strip()
    )

    if not (
        value
    ):
        raise ValueError(
            (
                "Analysis readiness workflow_id "
                "cannot be empty."
            )
        )

    return (
        value
    )


def _normalize_dataset_ids(
    values: List[
        str
    ],
) -> List[
    str
]:
    output: List[
        str
    ] = []

    seen: Set[
        str
    ] = set()

    for raw_value in values:
        value = (
            raw_value.strip()
        )

        if not (
            value
        ):
            raise ValueError(
                (
                    "Analysis readiness dataset_id "
                    "cannot be empty."
                )
            )

        if (
            value
            in seen
        ):
            continue

        seen.add(
            value
        )

        output.append(
            value
        )

    if not (
        output
    ):
        raise ValueError(
            (
                "Analysis readiness requires at "
                "least one requested dataset."
            )
        )

    return (
        output
    )


# ============================================================
# EVALUATION
# ============================================================


def evaluate_analysis_readiness(
    *,
    workflow_id: str,
    requested_analysis_dataset_ids: Optional[
        List[
            str
        ]
    ] = None,
) -> AnalysisReadinessDecision:
    """
    Evaluate whether analytical execution is authorized.

    This function:

    - performs no statistical analysis;
    - mutates no Preparation state;
    - does not trust browser-provided readiness;
    - recomputes authorization from server-owned state.

    Two dataset scopes are intentionally different:

    selected_analysis_dataset_ids
        Immutable Preparation roots.

    analysis_output_dataset_ids
        Final materialized datasets explicitly selected for
        analytical execution.

    Authorization policy:

    1. Before a final output has been selected:

       Preparation roots are kept as a provisional
       authorization scope.

       This preserves the distinction between:

           workflow incomplete
               -> NOT READY

       and:

           completely unrelated dataset
               -> NOT AUTHORIZED

    2. Once analysis_output_dataset_ids is non-empty:

       Only those final analytical outputs are authorized.

       Preparation roots can no longer be used to bypass a
       cleaned, transformed or combined final output.

    3. Every requested analytical dataset must also appear in
       the PASSED VALIDATE stage.

    4. The complete Preparation workflow must itself report
       ready_for_analysis=True.

    If requested_analysis_dataset_ids is omitted:

    - final outputs are used when they exist;
    - otherwise the provisional Preparation-root scope is used.
    """

    normalized_workflow_id = (
        _normalize_workflow_id(
            workflow_id
        )
    )

    # ========================================================
    # SERVER-OWNED SESSION
    # ========================================================

    session = (
        get_preparation_session(
            normalized_workflow_id
        )
    )

    snapshot = (
        session.snapshot
    )

    # ========================================================
    # PREPARATION ROOTS
    # ========================================================

    selected_ids = list(
        session
        .selected_analysis_dataset_ids
    )

    # ========================================================
    # FINAL ANALYTICAL OUTPUTS
    # ========================================================

    analysis_output_ids = list(
        session
        .analysis_output_dataset_ids
    )

    # ========================================================
    # VALIDATED OUTPUTS
    # ========================================================

    validated_ids = list(
        snapshot
        .validated_analysis_dataset_ids
    )

    # ========================================================
    # EFFECTIVE AUTHORIZATION SCOPE
    #
    # Once final outputs exist they completely replace the
    # roots for analytical execution.
    #
    # The root fallback exists only while Preparation is still
    # incomplete and no final output selection exists.
    # ========================================================

    authorized_ids = (
        list(
            analysis_output_ids
        )
        if (
            analysis_output_ids
        )
        else
        list(
            selected_ids
        )
    )

    # ========================================================
    # REQUESTED SCOPE
    # ========================================================

    if (
        requested_analysis_dataset_ids
        is None
    ):
        requested_ids = list(
            authorized_ids
        )

    else:
        requested_ids = (
            _normalize_dataset_ids(
                requested_analysis_dataset_ids
            )
        )

    authorized_set = set(
        authorized_ids
    )

    validated_set = set(
        validated_ids
    )

    requested_set = set(
        requested_ids
    )

    # ========================================================
    # DATASET SCOPE AUTHORIZATION
    # ========================================================

    unauthorized_dataset_ids = sorted(
        requested_set
        -
        authorized_set
    )

    dataset_scope_authorized = (
        len(
            unauthorized_dataset_ids
        )
        ==
        0
    )

    # ========================================================
    # DATASET VALIDATION
    # ========================================================

    unvalidated_dataset_ids = sorted(
        requested_set
        -
        validated_set
    )

    requested_datasets_validated = (
        len(
            unvalidated_dataset_ids
        )
        ==
        0
    )

    # ========================================================
    # BLOCKING REASONS
    # ========================================================

    blocking_reasons = list(
        snapshot
        .blocking_reasons
    )

    if (
        unauthorized_dataset_ids
    ):
        if (
            analysis_output_ids
        ):
            blocking_reasons.append(
                (
                    "analysis: requested datasets are "
                    "outside the final analysis output "
                    "selection: "
                    f"{unauthorized_dataset_ids}"
                )
            )

        else:
            blocking_reasons.append(
                (
                    "analysis: requested datasets are "
                    "outside the Preparation root scope: "
                    f"{unauthorized_dataset_ids}"
                )
            )

    if (
        unvalidated_dataset_ids
    ):
        blocking_reasons.append(
            (
                "analysis: requested datasets have "
                "not yet reached final validated "
                "preparation outputs: "
                f"{unvalidated_dataset_ids}"
            )
        )

    # ========================================================
    # WORKFLOW READINESS
    # ========================================================

    workflow_ready = (
        snapshot
        .ready_for_analysis
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    ready_for_analysis = (
        workflow_ready
        and
        dataset_scope_authorized
        and
        requested_datasets_validated
    )

    return (
        AnalysisReadinessDecision(
            workflow_id=
                session.workflow_id,

            session_revision=
                session.revision,

            ready_for_analysis=
                ready_for_analysis,

            workflow_ready_for_analysis=
                workflow_ready,

            dataset_scope_authorized=
                dataset_scope_authorized,

            requested_datasets_validated=
                requested_datasets_validated,

            selected_analysis_dataset_ids=
                selected_ids,

            analysis_output_dataset_ids=
                analysis_output_ids,

            authorized_analysis_dataset_ids=
                authorized_ids,

            validated_analysis_dataset_ids=
                validated_ids,

            requested_analysis_dataset_ids=
                requested_ids,

            unauthorized_dataset_ids=
                unauthorized_dataset_ids,

            unvalidated_dataset_ids=
                unvalidated_dataset_ids,

            next_stage=
                snapshot.next_stage,

            blocking_reasons=
                blocking_reasons,

            rule_version=
                ANALYSIS_READINESS_GATE_RULE_VERSION,
        )
    )


# ============================================================
# REQUIRED GATE
# ============================================================


def require_analysis_readiness(
    *,
    workflow_id: str,
    requested_analysis_dataset_ids: Optional[
        List[
            str
        ]
    ] = None,
) -> AnalysisReadinessDecision:
    """
    Require a server-owned Preparation session to authorize
    analytical execution.

    Returns the verified decision only when analysis may
    proceed.

    Raises:

    - PreparationSessionNotFoundError indirectly when the
      workflow does not exist;

    - AnalysisDatasetNotAuthorizedError when a requested
      dataset is outside the currently authorized scope;

    - AnalysisNotReadyError when the dataset belongs to the
      authorized scope but Preparation or validation is
      incomplete;

    - ValueError for malformed identifiers.

    Error ordering remains intentional:

    1. reject datasets outside the effective scope;
    2. reject an incomplete workflow;
    3. reject requested datasets missing from VALIDATE;
    4. permit execution only when every condition passes.

    Before final-output selection, roots form a provisional
    scope, so a normal in-progress Preparation request still
    produces NOT READY rather than becoming falsely
    unauthorized.

    After final-output selection, only final outputs are
    authorized.
    """

    decision = (
        evaluate_analysis_readiness(
            workflow_id=
                workflow_id,

            requested_analysis_dataset_ids=
                requested_analysis_dataset_ids,
        )
    )

    # ========================================================
    # WRONG DATASET SCOPE
    # ========================================================

    if not (
        decision
        .dataset_scope_authorized
    ):
        raise (
            AnalysisDatasetNotAuthorizedError(
                decision=
                    decision
            )
        )

    # ========================================================
    # WORKFLOW NOT READY
    # ========================================================

    if not (
        decision
        .workflow_ready_for_analysis
    ):
        raise (
            AnalysisNotReadyError(
                decision=
                    decision
            )
        )

    # ========================================================
    # DATASET NOT VALIDATED
    # ========================================================

    if not (
        decision
        .requested_datasets_validated
    ):
        raise (
            AnalysisNotReadyError(
                decision=
                    decision
            )
        )

    # ========================================================
    # FINAL SAFETY CHECK
    # ========================================================

    if not (
        decision
        .ready_for_analysis
    ):
        raise (
            AnalysisNotReadyError(
                decision=
                    decision
            )
        )

    return (
        decision
    )