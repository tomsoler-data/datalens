from __future__ import annotations


import json
import os
import sqlite3

from pathlib import (
    Path,
)

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

from app.persistence.sqlite_database import (
    ensure_ephemeral_sqlite_test_path,
    sqlite_connection,
    utc_now_iso,
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


# ============================================================
# SQLITE STORE
# PREPARATION_SESSION_SQLITE_STORE_V0_1
# ============================================================


PREPARATION_SESSION_STORE_VERSION = (
    "preparation_session_sqlite_store_v0.1"
)

LEGACY_PREPARATION_SESSION_STORE_ENV = (
    "DATALENS_PREPARATION_SESSION_STORE_PATH"
)


class PreparationSessionStoreError(
    RuntimeError,
):
    pass


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
# WORKFLOW METADATA ? DISPLAY NAME
# PREPARATION_WORKFLOW_METADATA_V0_1
# ============================================================


def _normalize_workflow_display_name(
    value: (
        str
        |
        None
    ),
) -> (
    str
    |
    None
):
    if value is None:
        return None


    normalized = (
        " ".join(
            str(
                value
            )
            .split()
        )
    )


    if not normalized:
        return None


    if (
        len(
            normalized
        )
        >
        120
    ):
        raise ValueError(
            (
                "Preparation workflow display_name "
                "cannot exceed 120 characters."
            )
        )


    return normalized


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



# ============================================================
# WORKFLOW HISTORY ? READ MODEL
# PREPARATION_SESSION_CATALOG_V0_1
# ============================================================




class PreparationSessionCatalogItem(
    StrictPreparationSessionModel,
):
    """
    Server-owned workflow-history entry.

    PreparationSession remains the canonical analytical state.

    Product metadata is stored separately from the analytical
    Preparation payload.

    PREPARATION_WORKFLOW_METADATA_V0_1
    """

    session: PreparationSessionView

    display_name: str = Field(
        min_length=1,
        max_length=120,
    )

    name_source: str = Field(
        pattern=
            "^(user|automatic)$"
    )

    created_at_utc: str = Field(
        min_length=1
    )

    updated_at_utc: str = Field(
        min_length=1
    )

    archived: bool = False

    archived_at_utc: (
        str
        |
        None
    ) = None

class PreparationSessionStore:
    """
    SQLite-backed server-owned Preparation session store.

    PREPARATION_SESSION_SQLITE_STORE_V0_1

    The orchestration contract remains unchanged:
    - workflow_id is backend-generated;
    - Preparation roots are immutable;
    - final analysis-output selection is a dedicated
      transaction;
    - stage statuses remain backend-derived;
    - every persisted state is validated by the canonical
      Preparation orchestrator before commit or return;
    - revision-based optimistic concurrency remains enforced.

    SQLite is used only as the durable control plane. No
    DataFrame is stored in this table.
    """

    def __init__(
        self,
    ) -> None:
        self._lock = (
            RLock()
        )


    # ========================================================
    # LEGACY JSON PATH
    # ========================================================


    @staticmethod
    def _legacy_store_path(
    ) -> Path:
        configured = (
            os.getenv(
                LEGACY_PREPARATION_SESSION_STORE_ENV,
                "",
            )
            .strip()
        )


        if configured:
            return (
                Path(
                    configured
                )
                .expanduser()
                .resolve()
            )


        api_root = (
            Path(
                __file__
            )
            .resolve()
            .parents[
                2
            ]
        )


        return (
            api_root
            /
            "var"
            /
            "preparation"
            /
            "preparation_sessions.json"
        )


    # ========================================================
    # SERIALIZATION
    # ========================================================


    @staticmethod
    def _serialize_state(
        state: PreparationSessionState,
    ) -> str:
        return (
            state.model_dump_json()
        )


    @staticmethod
    def _deserialize_payload(
        *,
        workflow_id: str,
        revision: int,
        payload_json: str,
    ) -> PreparationSessionState:
        try:
            state = (
                PreparationSessionState
                .model_validate_json(
                    payload_json
                )
            )

        except Exception as error:
            raise PreparationSessionStoreError(
                (
                    "Persisted Preparation session "
                    "payload is invalid. "
                    f"workflow_id={workflow_id}"
                )
            ) from error


        if (
            state.workflow_id
            !=
            workflow_id
        ):
            raise PreparationSessionStoreError(
                (
                    "Persisted Preparation workflow "
                    "identity does not match its "
                    "SQLite primary key. "
                    f"workflow_id={workflow_id}"
                )
            )


        if (
            state.revision
            !=
            revision
        ):
            raise PreparationSessionStoreError(
                (
                    "Persisted Preparation session "
                    "revision does not match its "
                    "SQLite revision column. "
                    f"workflow_id={workflow_id}, "
                    f"payload_revision={state.revision}, "
                    f"column_revision={revision}"
                )
            )


        # Re-run canonical orchestration on every read.
        _build_snapshot(
            state
        )


        return (
            state
        )


    @classmethod
    def _state_from_row(
        cls,
        row,
    ) -> PreparationSessionState:
        return (
            cls._deserialize_payload(
                workflow_id=
                    str(
                        row[
                            "workflow_id"
                        ]
                    ),

                revision=
                    int(
                        row[
                            "revision"
                        ]
                    ),

                payload_json=
                    str(
                        row[
                            "payload_json"
                        ]
                    ),
            )
        )


    # ========================================================
    # LEGACY JSON MIGRATION
    # ========================================================


    def _migrate_legacy_json_if_needed(
        self,
    ) -> None:
        legacy_path = (
            self._legacy_store_path()
        )


        if not legacy_path.exists():
            return


        try:
            raw = json.loads(
                legacy_path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception as error:
            raise PreparationSessionStoreError(
                (
                    "Legacy Preparation session JSON "
                    "could not be read: "
                    f"{legacy_path}"
                )
            ) from error


        if not isinstance(
            raw,
            dict,
        ):
            raise PreparationSessionStoreError(
                (
                    "Legacy Preparation session JSON "
                    "root must be an object."
                )
            )


        sessions = raw.get(
            "sessions"
        )


        if not isinstance(
            sessions,
            dict,
        ):
            raise PreparationSessionStoreError(
                (
                    "Legacy Preparation session JSON "
                    "must contain a `sessions` object."
                )
            )


        with sqlite_connection(
            write=True
        ) as connection:
            existing = (
                connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM preparation_sessions
                    """
                )
                .fetchone()
            )


            if (
                int(
                    existing[
                        "count"
                    ]
                )
                >
                0
            ):
                return


            now = (
                utc_now_iso()
            )


            for (
                workflow_id,
                raw_state,
            ) in sessions.items():
                try:
                    state = (
                        PreparationSessionState
                        .model_validate(
                            raw_state
                        )
                    )

                except Exception as error:
                    raise PreparationSessionStoreError(
                        (
                            "Legacy Preparation "
                            "session is invalid. "
                            f"workflow_id={workflow_id}"
                        )
                    ) from error


                if (
                    state.workflow_id
                    !=
                    workflow_id
                ):
                    raise PreparationSessionStoreError(
                        (
                            "Legacy Preparation "
                            "session identity does not "
                            "match its JSON key. "
                            f"workflow_id={workflow_id}"
                        )
                    )


                _build_snapshot(
                    state
                )


                connection.execute(
                    """
                    INSERT INTO preparation_sessions (
                        workflow_id,
                        revision,
                        payload_json,
                        created_at,
                        updated_at
                    )
                    VALUES (
                        ?,
                        ?,
                        ?,
                        ?,
                        ?
                    )
                    """,
                    (
                        state.workflow_id,
                        state.revision,
                        self._serialize_state(
                            state
                        ),
                        now,
                        now,
                    ),
                )


    # ========================================================
    # INTERNAL READ
    # ========================================================


    def _get_from_connection(
        self,
        *,
        connection,
        workflow_id: str,
    ) -> PreparationSessionState:
        row = (
            connection.execute(
                """
                SELECT
                    workflow_id,
                    revision,
                    payload_json
                FROM preparation_sessions
                WHERE workflow_id = ?
                """,
                (
                    workflow_id,
                ),
            )
            .fetchone()
        )


        if row is None:
            raise PreparationSessionNotFoundError(
                (
                    "Preparation session not found: "
                    f"{workflow_id}"
                )
            )


        return (
            self._state_from_row(
                row
            )
        )


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
            self._migrate_legacy_json_if_needed()


            for _ in range(
                10
            ):
                workflow_id = (
                    f"prep:{uuid4().hex}"
                )


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


                _build_snapshot(
                    state
                )


                now = (
                    utc_now_iso()
                )


                try:
                    with sqlite_connection(
                        write=True
                    ) as connection:
                        connection.execute(
                            """
                            INSERT INTO preparation_sessions (
                                workflow_id,
                                revision,
                                payload_json,
                                created_at,
                                updated_at
                            )
                            VALUES (
                                ?,
                                ?,
                                ?,
                                ?,
                                ?
                            )
                            """,
                            (
                                state.workflow_id,
                                state.revision,
                                self._serialize_state(
                                    state
                                ),
                                now,
                                now,
                            ),
                        )

                except sqlite3.IntegrityError:
                    continue


                return (
                    state.model_copy(
                        deep=True
                    )
                )


            raise PreparationSessionStoreError(
                (
                    "Could not allocate a unique "
                    "Preparation workflow_id."
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
            self._migrate_legacy_json_if_needed()


            with sqlite_connection(
                write=False
            ) as connection:
                state = (
                    self._get_from_connection(
                        connection=
                            connection,

                        workflow_id=
                            normalized_id,
                    )
                )


            return (
                state.model_copy(
                    deep=True
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
        normalized_id = (
            _normalize_workflow_id(
                workflow_id
            )
        )


        with self._lock:
            self._migrate_legacy_json_if_needed()


            with sqlite_connection(
                write=True
            ) as connection:
                current = (
                    self._get_from_connection(
                        connection=
                            connection,

                        workflow_id=
                            normalized_id,
                    )
                )


                working_copy = (
                    current.model_copy(
                        deep=True
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


                if (
                    candidate.workflow_id
                    !=
                    current.workflow_id
                ):
                    raise ValueError(
                        (
                            "Preparation session "
                            "workflow_id cannot be "
                            "changed."
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
                            "Preparation session selected "
                            "analysis datasets cannot be "
                            "changed by stage updates."
                        )
                    )


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


                _build_snapshot(
                    candidate
                )


                cursor = (
                    connection.execute(
                        """
                        UPDATE preparation_sessions
                        SET
                            revision = ?,
                            payload_json = ?,
                            updated_at = ?
                        WHERE
                            workflow_id = ?
                            AND
                            revision = ?
                        """,
                        (
                            candidate.revision,
                            self._serialize_state(
                                candidate
                            ),
                            utc_now_iso(),
                            normalized_id,
                            current.revision,
                        ),
                    )
                )


                if (
                    cursor.rowcount
                    !=
                    1
                ):
                    raise (
                        PreparationSessionRevisionConflictError(
                            (
                                "Preparation session "
                                "changed during stage "
                                "commit. "
                                f"workflow_id={normalized_id}, "
                                "expected_revision="
                                f"{current.revision}"
                            )
                        )
                    )


            return (
                candidate.model_copy(
                    deep=True
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
            self._migrate_legacy_json_if_needed()


            with sqlite_connection(
                write=True
            ) as connection:
                current = (
                    self._get_from_connection(
                        connection=
                            connection,

                        workflow_id=
                            normalized_id,
                    )
                )


                if (
                    current.revision
                    !=
                    expected_revision
                ):
                    raise (
                        PreparationSessionRevisionConflictError(
                            (
                                "Preparation session "
                                "changed after analysis "
                                "output selection was "
                                "evaluated. "
                                f"workflow_id={normalized_id}, "
                                "expected_revision="
                                f"{expected_revision}, "
                                "current_revision="
                                f"{current.revision}"
                            )
                        )
                    )


                if (
                    current
                    .validate_stage
                    .passed
                ):
                    raise ValueError(
                        (
                            "Analysis output selection "
                            "cannot change after "
                            "VALIDATE has PASSED."
                        )
                    )


                candidate = (
                    current.model_copy(
                        deep=True
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


                if (
                    candidate.workflow_id
                    !=
                    current.workflow_id
                ):
                    raise ValueError(
                        (
                            "Analysis output selection "
                            "cannot change workflow_id."
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
                            "Analysis output selection "
                            "cannot change Preparation "
                            "root datasets."
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
                            "Analysis output selection "
                            "cannot modify Preparation "
                            "stages."
                        )
                    )


                _build_snapshot(
                    candidate
                )


                cursor = (
                    connection.execute(
                        """
                        UPDATE preparation_sessions
                        SET
                            revision = ?,
                            payload_json = ?,
                            updated_at = ?
                        WHERE
                            workflow_id = ?
                            AND
                            revision = ?
                        """,
                        (
                            candidate.revision,
                            self._serialize_state(
                                candidate
                            ),
                            utc_now_iso(),
                            normalized_id,
                            current.revision,
                        ),
                    )
                )


                if (
                    cursor.rowcount
                    !=
                    1
                ):
                    raise (
                        PreparationSessionRevisionConflictError(
                            (
                                "Preparation session "
                                "changed during analysis "
                                "output selection commit. "
                                f"workflow_id={normalized_id}, "
                                "expected_revision="
                                f"{current.revision}"
                            )
                        )
                    )


            return (
                candidate.model_copy(
                    deep=True
                )
            )


    # ========================================================
    # RESET ? TESTS ONLY
    # ========================================================



    # ========================================================
    # WORKFLOW HISTORY ? LIST
    # PREPARATION_SESSION_CATALOG_V0_1
    # ========================================================



    # ========================================================
    # WORKFLOW CATALOG + LIFECYCLE
    # PREPARATION_WORKFLOW_METADATA_V0_1
    # ========================================================



    # ========================================================
    # WORKFLOW METADATA
    # PREPARATION_WORKFLOW_METADATA_V0_1
    # ========================================================


    @classmethod
    def _catalog_item_from_row(
        cls,
        row,
    ) -> PreparationSessionCatalogItem:
        if (
            row[
                "metadata_workflow_id"
            ]
            is None
        ):
            raise PreparationSessionStoreError(
                (
                    "Preparation workflow metadata "
                    "is missing. workflow_id="
                    f"{row['workflow_id']}"
                )
            )


        state = (
            cls._state_from_row(
                row
            )
        )


        archived_at = (
            row[
                "archived_at"
            ]
        )


        session_updated_at = str(
            row[
                "updated_at"
            ]
        )


        metadata_updated_at = str(
            row[
                "metadata_updated_at"
            ]
        )


        workflow_updated_at = max(
            session_updated_at,
            metadata_updated_at,
        )


        return (
            PreparationSessionCatalogItem(
                session=
                    _build_view(
                        state
                    ),

                display_name=
                    str(
                        row[
                            "display_name"
                        ]
                    ),

                name_source=
                    str(
                        row[
                            "name_source"
                        ]
                    ),

                created_at_utc=
                    str(
                        row[
                            "created_at"
                        ]
                    ),

                updated_at_utc=
                    workflow_updated_at,

                archived=
                    archived_at
                    is not None,

                archived_at_utc=
                    (
                        None
                        if archived_at
                        is None
                        else
                        str(
                            archived_at
                        )
                    ),
            )
        )


    def _get_catalog_item_from_connection(
        self,
        *,
        connection,
        workflow_id: str,
    ) -> PreparationSessionCatalogItem:
        row = (
            connection.execute(
                """
                SELECT
                    session.workflow_id
                        AS workflow_id,

                    session.revision
                        AS revision,

                    session.payload_json
                        AS payload_json,

                    session.created_at
                        AS created_at,

                    session.updated_at
                        AS updated_at,

                    metadata.workflow_id
                        AS metadata_workflow_id,

                    metadata.display_name
                        AS display_name,

                    metadata.name_source
                        AS name_source,

                    metadata.archived_at
                        AS archived_at,

                    metadata.updated_at
                        AS metadata_updated_at

                FROM preparation_sessions
                    AS session

                LEFT JOIN
                    preparation_workflow_metadata
                    AS metadata

                    ON metadata.workflow_id
                    =
                    session.workflow_id

                WHERE
                    session.workflow_id = ?
                """,
                (
                    workflow_id,
                ),
            )
            .fetchone()
        )


        if row is None:
            raise (
                PreparationSessionNotFoundError(
                    (
                        "Preparation session not found: "
                        f"{workflow_id}"
                    )
                )
            )


        return (
            self._catalog_item_from_row(
                row
            )
        )


    def list(
        self,
    ) -> list[
        PreparationSessionCatalogItem
    ]:
        with self._lock:
            self._migrate_legacy_json_if_needed()


            with sqlite_connection(
                write=False
            ) as connection:
                rows = (
                    connection.execute(
                        """
                        SELECT
                            session.workflow_id
                                AS workflow_id,

                            session.revision
                                AS revision,

                            session.payload_json
                                AS payload_json,

                            session.created_at
                                AS created_at,

                            session.updated_at
                                AS updated_at,

                            metadata.workflow_id
                                AS metadata_workflow_id,

                            metadata.display_name
                                AS display_name,

                            metadata.name_source
                                AS name_source,

                            metadata.archived_at
                                AS archived_at,

                            metadata.updated_at
                                AS metadata_updated_at

                        FROM preparation_sessions
                            AS session

                        LEFT JOIN
                            preparation_workflow_metadata
                            AS metadata

                            ON metadata.workflow_id
                            =
                            session.workflow_id

                        ORDER BY
                            CASE
                                WHEN
                                    metadata.archived_at
                                    IS NULL
                                THEN 0
                                ELSE 1
                            END ASC,

                            CASE
                                WHEN
                                    metadata.archived_at
                                    IS NULL
                                THEN
                                    CASE
                                        WHEN
                                            metadata.updated_at
                                            >
                                            session.updated_at
                                        THEN
                                            metadata.updated_at
                                        ELSE
                                            session.updated_at
                                    END

                                ELSE
                                    metadata.archived_at
                            END DESC,

                            session.workflow_id ASC
                        """
                    )
                    .fetchall()
                )


            return [
                self
                ._catalog_item_from_row(
                    row
                )
                .model_copy(
                    deep=True
                )

                for row
                in rows
            ]


    def rename(
        self,
        *,
        workflow_id: str,
        display_name: str,
    ) -> PreparationSessionCatalogItem:
        normalized_id = (
            _normalize_workflow_id(
                workflow_id
            )
        )


        normalized_name = (
            _normalize_workflow_display_name(
                display_name
            )
        )


        if normalized_name is None:
            raise ValueError(
                (
                    "Preparation workflow display_name "
                    "cannot be blank."
                )
            )


        with self._lock:
            self._migrate_legacy_json_if_needed()


            with sqlite_connection(
                write=True
            ) as connection:
                self._get_catalog_item_from_connection(
                    connection=
                        connection,

                    workflow_id=
                        normalized_id,
                )


                connection.execute(
                    """
                    UPDATE
                        preparation_workflow_metadata

                    SET
                        display_name = ?,
                        name_source = 'user',
                        updated_at = ?

                    WHERE
                        workflow_id = ?
                        AND
                        (
                            display_name <> ?
                            OR
                            name_source <> 'user'
                        )
                    """,
                    (
                        normalized_name,
                        utc_now_iso(),
                        normalized_id,
                        normalized_name,
                    ),
                )


                item = (
                    self
                    ._get_catalog_item_from_connection(
                        connection=
                            connection,

                        workflow_id=
                            normalized_id,
                    )
                )


            return (
                item.model_copy(
                    deep=True
                )
            )


    def archive(
        self,
        workflow_id: str,
    ) -> PreparationSessionCatalogItem:
        normalized_id = (
            _normalize_workflow_id(
                workflow_id
            )
        )


        with self._lock:
            self._migrate_legacy_json_if_needed()


            with sqlite_connection(
                write=True
            ) as connection:
                current = (
                    self
                    ._get_catalog_item_from_connection(
                        connection=
                            connection,

                        workflow_id=
                            normalized_id,
                    )
                )


                if not current.archived:
                    now = (
                        utc_now_iso()
                    )


                    connection.execute(
                        """
                        UPDATE
                            preparation_workflow_metadata

                        SET
                            archived_at = ?,
                            updated_at = ?

                        WHERE
                            workflow_id = ?
                            AND
                            archived_at IS NULL
                        """,
                        (
                            now,
                            now,
                            normalized_id,
                        ),
                    )


                item = (
                    self
                    ._get_catalog_item_from_connection(
                        connection=
                            connection,

                        workflow_id=
                            normalized_id,
                    )
                )


            return (
                item.model_copy(
                    deep=True
                )
            )


    def restore(
        self,
        workflow_id: str,
    ) -> PreparationSessionCatalogItem:
        normalized_id = (
            _normalize_workflow_id(
                workflow_id
            )
        )


        with self._lock:
            self._migrate_legacy_json_if_needed()


            with sqlite_connection(
                write=True
            ) as connection:
                current = (
                    self
                    ._get_catalog_item_from_connection(
                        connection=
                            connection,

                        workflow_id=
                            normalized_id,
                    )
                )


                if current.archived:
                    connection.execute(
                        """
                        UPDATE
                            preparation_workflow_metadata

                        SET
                            archived_at = NULL,
                            updated_at = ?

                        WHERE
                            workflow_id = ?
                            AND
                            archived_at IS NOT NULL
                        """,
                        (
                            utc_now_iso(),
                            normalized_id,
                        ),
                    )


                item = (
                    self
                    ._get_catalog_item_from_connection(
                        connection=
                            connection,

                        workflow_id=
                            normalized_id,
                    )
                )


            return (
                item.model_copy(
                    deep=True
                )
            )
    def reset(
        self,
    ) -> None:
        with self._lock:
            with sqlite_connection(
                write=True
            ) as connection:
                connection.execute(
                    """
                    DELETE FROM preparation_sessions
                    """
                )



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
    display_name: (
        str
        |
        None
    ) = None,
) -> PreparationSessionView:
    normalized_name = (
        _normalize_workflow_display_name(
            display_name
        )
    )


    state = (
        _SESSION_STORE.create(
            selected_analysis_dataset_ids=
                selected_analysis_dataset_ids
        )
    )


    if normalized_name is not None:
        _SESSION_STORE.rename(
            workflow_id=
                state.workflow_id,

            display_name=
                normalized_name,
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
# WORKFLOW HISTORY ? PUBLIC READ
# PREPARATION_SESSION_CATALOG_V0_1
# ============================================================


def list_preparation_sessions(
) -> list[
    PreparationSessionCatalogItem
]:
    return (
        _SESSION_STORE.list()
    )


# ============================================================
# WORKFLOW LIFECYCLE ? PUBLIC OPERATIONS
# PREPARATION_WORKFLOW_METADATA_V0_1
# ============================================================



def rename_preparation_session(
    *,
    workflow_id: str,
    display_name: str,
) -> PreparationSessionCatalogItem:
    return (
        _SESSION_STORE.rename(
            workflow_id=
                workflow_id,

            display_name=
                display_name,
        )
    )

def archive_preparation_session(
    workflow_id: str,
) -> PreparationSessionCatalogItem:
    return (
        _SESSION_STORE.archive(
            workflow_id
        )
    )


def restore_preparation_session(
    workflow_id: str,
) -> PreparationSessionCatalogItem:
    return (
        _SESSION_STORE.restore(
            workflow_id
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

    ensure_ephemeral_sqlite_test_path(
        namespace=
            "preparation-session-tests"
    )

    _SESSION_STORE.reset()