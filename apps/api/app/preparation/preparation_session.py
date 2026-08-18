from __future__ import annotations

from threading import (
    RLock,
)

from typing import (
    Callable,
    Dict,
    List,
)

from uuid import (
    uuid4,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.preparation.preparation_orchestrator import (
    OptionalPreparationStageSignal,
    PreparationOrchestrationInput,
    RequiredPreparationStageSignal,
    ValidationPreparationStageSignal,
    orchestrate_preparation,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
    PreparationWorkflowSnapshot,
)


# ============================================================
# VERSION
# ============================================================


PREPARATION_SESSION_RULE_VERSION = (
    "preparation_session_v0.2"
)


# ============================================================
# ERRORS
# ============================================================


class PreparationSessionNotFoundError(
    LookupError,
):
    pass


class PreparationSessionRevisionConflictError(
    RuntimeError,
):
    """
    Raised when a caller tries to commit a decision that was
    evaluated against an older Preparation session revision.
    """

    pass


# ============================================================
# STRICT INTERNAL MODEL
# ============================================================


class StrictPreparationSessionModel(
    BaseModel,
):
    model_config = ConfigDict(
        extra="forbid"
    )


# ============================================================
# SESSION STATE
# ============================================================


class PreparationSessionState(
    StrictPreparationSessionModel,
):
    """
    Server-owned preparation state.

    This object is NEVER accepted directly from the browser.

    selected_analysis_dataset_ids
        Immutable Preparation root scope.

    analysis_output_dataset_ids
        Final materialized datasets explicitly selected for
        VALIDATE / ANALYZE.

    Stage signals are modified only through backend functions
    in this module.
    """

    workflow_id: str

    revision: int = 0

    # ========================================================
    # PREPARATION ROOT SCOPE
    # ========================================================

    selected_analysis_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    # ========================================================
    # FINAL ANALYTICAL OUTPUT SCOPE
    # ========================================================

    analysis_output_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    # ========================================================
    # PREPARATION STAGES
    # ========================================================

    import_stage: RequiredPreparationStageSignal

    understand_stage: RequiredPreparationStageSignal

    quality_stage: RequiredPreparationStageSignal

    clean_stage: OptionalPreparationStageSignal

    transform_stage: OptionalPreparationStageSignal

    combine_stage: OptionalPreparationStageSignal

    validate_stage: ValidationPreparationStageSignal


# ============================================================
# PUBLIC READ MODEL
# ============================================================


class PreparationSessionView(
    StrictPreparationSessionModel,
):
    """
    Read-only representation exposed through the API.

    Internal stage signals are intentionally not returned.

    The frontend receives only:
    - immutable Preparation root scope;
    - committed analytical output scope;
    - workflow snapshot derived by the backend.
    """

    session_version: str

    workflow_id: str

    revision: int

    selected_analysis_dataset_ids: List[
        str
    ]

    analysis_output_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    snapshot: PreparationWorkflowSnapshot


# ============================================================
# NORMALIZATION — PREPARATION ROOTS
# ============================================================


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

    seen = set()


    for raw_value in values:
        value = (
            raw_value.strip()
        )


        if not (
            value
        ):
            raise ValueError(
                (
                    "Preparation session dataset_id "
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
                "Preparation session requires at least "
                "one selected analysis dataset."
            )
        )


    return (
        output
    )


# ============================================================
# NORMALIZATION — ANALYSIS OUTPUTS
# ============================================================


def _normalize_analysis_output_dataset_ids(
    values: List[
        str
    ],
) -> List[
    str
]:
    """
    Final analytical output selection must always contain at
    least one dataset.

    Empty state is valid only before any final output
    selection has been committed.
    """

    output: List[
        str
    ] = []

    seen = set()


    for raw_value in values:
        value = (
            raw_value.strip()
        )


        if not value:
            raise ValueError(
                (
                    "Analysis output dataset_id "
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


    if not output:
        raise ValueError(
            (
                "At least one analysis output dataset "
                "must be selected."
            )
        )


    return (
        output
    )


# ============================================================
# WORKFLOW ID
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
                "Preparation session workflow_id "
                "cannot be empty."
            )
        )


    return (
        value
    )


# ============================================================
# ORCHESTRATION CONVERSION
# ============================================================


def _to_orchestration_input(
    state: PreparationSessionState,
) -> PreparationOrchestrationInput:
    """
    Convert the server-owned Preparation session into the
    orchestration contract.

    Two dataset scopes are intentionally preserved:

    selected_analysis_dataset_ids
        Immutable Preparation root datasets.

    analysis_output_dataset_ids
        Explicitly selected final analytical outputs.

    An empty analysis_output_dataset_ids list is valid while
    Preparation is still in progress.

    It simply prevents the workflow from becoming
    READY FOR ANALYSIS until a final output has been selected
    and certified by VALIDATE.
    """

    return (
        PreparationOrchestrationInput(
            workflow_id=
                state.workflow_id,

            selected_analysis_dataset_ids=
                list(
                    state
                    .selected_analysis_dataset_ids
                ),

            analysis_output_dataset_ids=
                list(
                    state
                    .analysis_output_dataset_ids
                ),

            import_stage=
                state.import_stage,

            understand_stage=
                state.understand_stage,

            quality_stage=
                state.quality_stage,

            clean_stage=
                state.clean_stage,

            transform_stage=
                state.transform_stage,

            combine_stage=
                state.combine_stage,

            validate_stage=
                state.validate_stage,
        )
    )


def _build_snapshot(
    state: PreparationSessionState,
) -> PreparationWorkflowSnapshot:
    """
    Recompute the workflow snapshot from server-owned state.

    Nothing stored in the session can directly declare a
    PreparationStageStatus.
    """

    return (
        orchestrate_preparation(
            _to_orchestration_input(
                state
            )
        )
    )


def _build_view(
    state: PreparationSessionState,
) -> PreparationSessionView:
    return (
        PreparationSessionView(
            session_version=
                PREPARATION_SESSION_RULE_VERSION,

            workflow_id=
                state.workflow_id,

            revision=
                state.revision,

            selected_analysis_dataset_ids=
                list(
                    state
                    .selected_analysis_dataset_ids
                ),

            analysis_output_dataset_ids=
                list(
                    state
                    .analysis_output_dataset_ids
                ),

            snapshot=
                _build_snapshot(
                    state
                ),
        )
    )


# ============================================================
# STORE
# ============================================================


class PreparationSessionStore:
    """
    Thread-safe in-memory store.

    v0.2 distinguishes:

    - immutable Preparation root datasets;
    - explicitly selected final analytical outputs.

    Sessions disappear when the FastAPI process restarts.

    A persistent repository can replace this implementation
    later without changing the orchestration contract.
    """

    def __init__(
        self,
    ) -> None:
        self._lock = (
            RLock()
        )

        self._sessions: Dict[
            str,
            PreparationSessionState,
        ] = {}


    # ========================================================
    # CREATE
    # ========================================================


    def create(
        self,
        *,
        selected_analysis_dataset_ids: List[
            str
        ],
    ) -> PreparationSessionState:
        dataset_ids = (
            _normalize_dataset_ids(
                selected_analysis_dataset_ids
            )
        )


        with self._lock:
            while True:
                workflow_id = (
                    f"prep:{uuid4().hex}"
                )


                if (
                    workflow_id
                    not in
                    self._sessions
                ):
                    break


            state = (
                PreparationSessionState(
                    workflow_id=
                        workflow_id,

                    revision=
                        0,

                    selected_analysis_dataset_ids=
                        dataset_ids,

                    analysis_output_dataset_ids=
                        [],

                    import_stage=
                        RequiredPreparationStageSignal(
                            completed=
                                False,
                        ),

                    understand_stage=
                        RequiredPreparationStageSignal(
                            completed=
                                False,
                        ),

                    quality_stage=
                        RequiredPreparationStageSignal(
                            completed=
                                False,
                        ),

                    clean_stage=
                        OptionalPreparationStageSignal(
                            required=
                                False,
                        ),

                    transform_stage=
                        OptionalPreparationStageSignal(
                            required=
                                False,
                        ),

                    combine_stage=
                        OptionalPreparationStageSignal(
                            required=
                                False,
                        ),

                    validate_stage=
                        ValidationPreparationStageSignal(
                            completed=
                                False,

                            passed=
                                False,
                        ),
                )
            )


            # Validate complete orchestration state before
            # storing anything.
            _build_snapshot(
                state
            )


            self._sessions[
                workflow_id
            ] = (
                state
            )


            return (
                state.model_copy(
                    deep=
                        True
                )
            )


    # ========================================================
    # GET
    # ========================================================


    def get(
        self,
        workflow_id: str,
    ) -> PreparationSessionState:
        normalized_id = (
            _normalize_workflow_id(
                workflow_id
            )
        )


        with self._lock:
            state = (
                self._sessions.get(
                    normalized_id
                )
            )


            if (
                state is None
            ):
                raise PreparationSessionNotFoundError(
                    (
                        "Preparation session not found: "
                        f"{normalized_id}"
                    )
                )


            return (
                state.model_copy(
                    deep=
                        True
                )
            )


    # ========================================================
    # TRANSACTIONAL STAGE UPDATE
    # ========================================================


    def update(
        self,
        workflow_id: str,
        updater: Callable[
            [
                PreparationSessionState
            ],
            PreparationSessionState,
        ],
    ) -> PreparationSessionState:
        """
        Generic Preparation stage update.

        Neither dataset-scope contract may be changed here.

        - selected_analysis_dataset_ids:
          immutable Preparation roots.

        - analysis_output_dataset_ids:
          immutable through stage updates.

        Final analysis output selection has its own dedicated
        transactional operation below.
        """

        normalized_id = (
            _normalize_workflow_id(
                workflow_id
            )
        )


        with self._lock:
            current = (
                self._sessions.get(
                    normalized_id
                )
            )


            if (
                current is None
            ):
                raise PreparationSessionNotFoundError(
                    (
                        "Preparation session not found: "
                        f"{normalized_id}"
                    )
                )


            working_copy = (
                current.model_copy(
                    deep=
                        True
                )
            )


            candidate = (
                updater(
                    working_copy
                )
            )


            if not isinstance(
                candidate,
                PreparationSessionState,
            ):
                raise TypeError(
                    (
                        "Preparation session updater "
                        "must return "
                        "PreparationSessionState."
                    )
                )


            # =================================================
            # IMMUTABLE WORKFLOW IDENTITY
            # =================================================


            if (
                candidate.workflow_id
                !=
                current.workflow_id
            ):
                raise ValueError(
                    (
                        "Preparation session workflow_id "
                        "cannot be changed."
                    )
                )


            # =================================================
            # IMMUTABLE PREPARATION ROOT SCOPE
            # =================================================


            if (
                candidate
                .selected_analysis_dataset_ids
                !=
                current
                .selected_analysis_dataset_ids
            ):
                raise ValueError(
                    (
                        "Preparation session selected "
                        "analysis datasets cannot be "
                        "changed by stage updates."
                    )
                )


            # =================================================
            # IMMUTABLE ANALYSIS OUTPUT SCOPE
            # =================================================


            if (
                candidate
                .analysis_output_dataset_ids
                !=
                current
                .analysis_output_dataset_ids
            ):
                raise ValueError(
                    (
                        "Preparation analysis output "
                        "datasets cannot be changed by "
                        "stage updates."
                    )
                )


            candidate = (
                candidate.model_copy(
                    update={
                        "revision":
                            current.revision
                            +
                            1
                    }
                )
            )


            # =================================================
            # TRANSACTIONAL VALIDATION
            # =================================================


            # If orchestration rejects the candidate state,
            # nothing is written to the store.
            _build_snapshot(
                candidate
            )


            self._sessions[
                normalized_id
            ] = (
                candidate
            )


            return (
                candidate.model_copy(
                    deep=
                        True
                )
            )


    # ========================================================
    # TRANSACTIONAL ANALYSIS OUTPUT SELECTION
    # ========================================================


    def replace_analysis_output_dataset_ids(
        self,
        *,
        workflow_id: str,
        analysis_output_dataset_ids: List[
            str
        ],
        expected_revision: int,
    ) -> PreparationSessionState:
        """
        Dedicated transaction for final analysis-output scope.

        Safety guarantees:

        - workflow_id cannot change;
        - Preparation roots cannot change;
        - Preparation stages cannot be changed by this call;
        - caller must provide the revision against which the
          selection was evaluated;
        - an already PASSED VALIDATE stage locks the output
          scope;
        - changing outputs invalidates any previous failed /
          incomplete VALIDATE state;
        - orchestration is checked before commit.

        This method does NOT itself validate Artifact Store
        lineage.

        Production callers must first pass through
        require_analysis_output_selection().
        """

        normalized_id = (
            _normalize_workflow_id(
                workflow_id
            )
        )


        output_dataset_ids = (
            _normalize_analysis_output_dataset_ids(
                analysis_output_dataset_ids
            )
        )


        with self._lock:
            current = (
                self._sessions.get(
                    normalized_id
                )
            )


            if (
                current is None
            ):
                raise PreparationSessionNotFoundError(
                    (
                        "Preparation session not found: "
                        f"{normalized_id}"
                    )
                )


            # =================================================
            # OPTIMISTIC REVISION GUARD
            # =================================================


            if (
                current.revision
                !=
                expected_revision
            ):
                raise (
                    PreparationSessionRevisionConflictError(
                        (
                            "Preparation session changed "
                            "after analysis output selection "
                            "was evaluated. "
                            f"workflow_id={normalized_id}, "
                            "expected_revision="
                            f"{expected_revision}, "
                            "current_revision="
                            f"{current.revision}"
                        )
                    )
                )


            # =================================================
            # VALIDATION LOCK
            # =================================================


            if (
                current
                .validate_stage
                .passed
            ):
                raise ValueError(
                    (
                        "Analysis output selection cannot "
                        "change after VALIDATE has PASSED."
                    )
                )


            # =================================================
            # BUILD CANDIDATE
            #
            # Output selection is upstream of VALIDATE.
            # Therefore any previous incomplete / failed
            # VALIDATE signal becomes stale and is reset.
            # =================================================


            candidate = (
                current.model_copy(
                    deep=
                        True
                )
            )


            candidate = (
                candidate.model_copy(
                    update={
                        "analysis_output_dataset_ids":
                            output_dataset_ids,

                        "validate_stage":
                            (
                                ValidationPreparationStageSignal(
                                    completed=
                                        False,

                                    passed=
                                        False,
                                )
                            ),

                        "revision":
                            current.revision
                            +
                            1,
                    }
                )
            )


            # =================================================
            # DEFENSIVE INVARIANTS
            # =================================================


            if (
                candidate.workflow_id
                !=
                current.workflow_id
            ):
                raise ValueError(
                    (
                        "Analysis output selection cannot "
                        "change workflow_id."
                    )
                )


            if (
                candidate
                .selected_analysis_dataset_ids
                !=
                current
                .selected_analysis_dataset_ids
            ):
                raise ValueError(
                    (
                        "Analysis output selection cannot "
                        "change Preparation root datasets."
                    )
                )


            if (
                candidate.import_stage
                !=
                current.import_stage
                or
                candidate.understand_stage
                !=
                current.understand_stage
                or
                candidate.quality_stage
                !=
                current.quality_stage
                or
                candidate.clean_stage
                !=
                current.clean_stage
                or
                candidate.transform_stage
                !=
                current.transform_stage
                or
                candidate.combine_stage
                !=
                current.combine_stage
            ):
                raise ValueError(
                    (
                        "Analysis output selection cannot "
                        "modify Preparation stages."
                    )
                )


            # =================================================
            # TRANSACTIONAL VALIDATION
            # =================================================


            _build_snapshot(
                candidate
            )


            self._sessions[
                normalized_id
            ] = (
                candidate
            )


            return (
                candidate.model_copy(
                    deep=
                        True
                )
            )


    # ========================================================
    # RESET — TESTS ONLY
    # ========================================================


    def reset(
        self,
    ) -> None:
        with self._lock:
            self._sessions.clear()


# ============================================================
# GLOBAL STORE
# ============================================================


_SESSION_STORE = (
    PreparationSessionStore()
)


# ============================================================
# PUBLIC SESSION API — SERVER SIDE
# ============================================================


def create_preparation_session(
    *,
    selected_analysis_dataset_ids: List[
        str
    ],
) -> PreparationSessionView:
    state = (
        _SESSION_STORE.create(
            selected_analysis_dataset_ids=
                selected_analysis_dataset_ids
        )
    )


    return (
        _build_view(
            state
        )
    )


def get_preparation_session(
    workflow_id: str,
) -> PreparationSessionView:
    state = (
        _SESSION_STORE.get(
            workflow_id
        )
    )


    return (
        _build_view(
            state
        )
    )


# ============================================================
# ANALYSIS OUTPUT SELECTION — SERVER SIDE
# ============================================================


def record_analysis_output_selection(
    *,
    workflow_id: str,
    analysis_output_dataset_ids: List[
        str
    ],
    expected_revision: int,
) -> PreparationSessionView:
    """
    Internal backend operation.

    This function must not be exposed directly as a generic
    stage update.

    Production callers must first validate the requested
    outputs against Preparation Artifact Store lineage through
    require_analysis_output_selection().
    """

    updated = (
        _SESSION_STORE
        .replace_analysis_output_dataset_ids(
            workflow_id=
                workflow_id,

            analysis_output_dataset_ids=
                analysis_output_dataset_ids,

            expected_revision=
                expected_revision,
        )
    )


    return (
        _build_view(
            updated
        )
    )


# ============================================================
# REQUIRED STAGE UPDATE
# ============================================================


def record_required_stage_signal(
    *,
    workflow_id: str,
    stage: PreparationStage,
    completed: bool,
    dataset_ids: List[
        str
    ],
    evidence_refs: List[
        str
    ],
    blocking_reasons: List[
        str
    ],
) -> PreparationSessionView:
    """
    Internal backend operation.

    No HTTP endpoint exposes this function directly.
    """

    field_by_stage = {
        PreparationStage.IMPORT:
            "import_stage",

        PreparationStage.UNDERSTAND:
            "understand_stage",

        PreparationStage.QUALITY:
            "quality_stage",
    }


    field_name = (
        field_by_stage.get(
            stage
        )
    )


    if (
        field_name is None
    ):
        raise ValueError(
            (
                "record_required_stage_signal supports "
                "only IMPORT, UNDERSTAND and QUALITY."
            )
        )


    signal = (
        RequiredPreparationStageSignal(
            completed=
                completed,

            dataset_ids=
                dataset_ids,

            evidence_refs=
                evidence_refs,

            blocking_reasons=
                blocking_reasons,
        )
    )


    def updater(
        state: PreparationSessionState,
    ) -> PreparationSessionState:
        return (
            state.model_copy(
                update={
                    field_name:
                        signal
                }
            )
        )


    updated = (
        _SESSION_STORE.update(
            workflow_id,
            updater,
        )
    )


    return (
        _build_view(
            updated
        )
    )


# ============================================================
# OPTIONAL STAGE UPDATE
# ============================================================


def record_optional_stage_signal(
    *,
    workflow_id: str,
    stage: PreparationStage,
    required: bool,
    completed: bool,
    review_required: bool,
    blocked: bool,
    dataset_ids: List[
        str
    ],
    evidence_refs: List[
        str
    ],
    blocking_reasons: List[
        str
    ],
) -> PreparationSessionView:
    """
    Internal backend operation for Clean, Transform or Combine.

    No HTTP endpoint exposes this function directly.
    """

    field_by_stage = {
        PreparationStage.CLEAN:
            "clean_stage",

        PreparationStage.TRANSFORM:
            "transform_stage",

        PreparationStage.COMBINE:
            "combine_stage",
    }


    field_name = (
        field_by_stage.get(
            stage
        )
    )


    if (
        field_name is None
    ):
        raise ValueError(
            (
                "record_optional_stage_signal supports "
                "only CLEAN, TRANSFORM and COMBINE."
            )
        )


    signal = (
        OptionalPreparationStageSignal(
            required=
                required,

            completed=
                completed,

            review_required=
                review_required,

            blocked=
                blocked,

            dataset_ids=
                dataset_ids,

            evidence_refs=
                evidence_refs,

            blocking_reasons=
                blocking_reasons,
        )
    )


    def updater(
        state: PreparationSessionState,
    ) -> PreparationSessionState:
        return (
            state.model_copy(
                update={
                    field_name:
                        signal
                }
            )
        )


    updated = (
        _SESSION_STORE.update(
            workflow_id,
            updater,
        )
    )


    return (
        _build_view(
            updated
        )
    )


# ============================================================
# VALIDATION STAGE UPDATE
# ============================================================


def record_validation_stage_signal(
    *,
    workflow_id: str,
    completed: bool,
    passed: bool,
    dataset_ids: List[
        str
    ],
    evidence_refs: List[
        str
    ],
    blocking_reasons: List[
        str
    ],
    expected_revision: int | None = None,
) -> PreparationSessionView:
    """
    Internal backend operation.

    Only validation engines should call this function.

    When expected_revision is provided, the validation commit
    is accepted only if the Preparation session still has the
    exact revision against which validation was evaluated.

    The revision check is executed inside
    PreparationSessionStore.update(), therefore while the
    store RLock is held.

    A stale validation decision can never be committed.
    """

    signal = (
        ValidationPreparationStageSignal(
            completed=
                completed,

            passed=
                passed,

            dataset_ids=
                dataset_ids,

            evidence_refs=
                evidence_refs,

            blocking_reasons=
                blocking_reasons,
        )
    )


    def updater(
        state: PreparationSessionState,
    ) -> PreparationSessionState:
        # ====================================================
        # OPTIMISTIC REVISION GUARD
        # ====================================================

        if (
            expected_revision
            is not None
            and
            state.revision
            !=
            expected_revision
        ):
            raise (
                PreparationSessionRevisionConflictError(
                    (
                        "Preparation session changed after "
                        "final validation was evaluated. "
                        f"workflow_id={workflow_id}, "
                        "expected_revision="
                        f"{expected_revision}, "
                        "current_revision="
                        f"{state.revision}"
                    )
                )
            )


        return (
            state.model_copy(
                update={
                    "validate_stage":
                        signal
                }
            )
        )


    updated = (
        _SESSION_STORE.update(
            workflow_id,
            updater,
        )
    )


    return (
        _build_view(
            updated
        )
    )


# ============================================================
# TEST SUPPORT
# ============================================================


def reset_preparation_session_store_for_tests(
) -> None:
    """
    Test-only helper.

    Production code should never use this function.
    """

    _SESSION_STORE.reset()