from __future__ import annotations

import json
import os
import shutil

from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from threading import RLock
from typing import (
    Callable,
    Dict,
    List,
    Optional,
)
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
)

from app.persistence.sqlite_database import (
    sqlite_connection,
)


# ============================================================
# VERSION
# PREPARATION_WORKFLOW_PERMANENT_DELETE_V0_1
# ============================================================


PREPARATION_WORKFLOW_DELETE_RULE_VERSION = (
    "preparation_workflow_permanent_delete_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class PreparationWorkflowDeleteError(
    RuntimeError,
):
    pass


class PreparationWorkflowDeleteNotFoundError(
    PreparationWorkflowDeleteError,
):
    pass


class PreparationWorkflowDeleteNotArchivedError(
    PreparationWorkflowDeleteError,
):
    pass


class PreparationWorkflowDeleteConfirmationError(
    PreparationWorkflowDeleteError,
):
    pass


class PreparationWorkflowDeleteRevisionConflictError(
    PreparationWorkflowDeleteError,
):
    pass


class PreparationWorkflowDeleteIntegrityError(
    PreparationWorkflowDeleteError,
):
    pass


class PreparationWorkflowDeleteRecoveryError(
    PreparationWorkflowDeleteError,
):
    pass


# ============================================================
# RESPONSE
# ============================================================


class PreparationWorkflowDeleteResult(
    BaseModel,
):
    model_config = ConfigDict(
        extra="forbid"
    )

    workflow_id: str

    display_name: str

    preparation_artifacts_deleted: int

    analysis_artifacts_deleted: int

    preparation_ui_state_deleted: int

    report_selection_deleted: int

    preparation_session_deleted: int

    workflow_metadata_deleted: int

    payload_files_removed_from_live_store: int

    quarantine_cleanup_pending: bool

    rule_version: str = (
        PREPARATION_WORKFLOW_DELETE_RULE_VERSION
    )


# ============================================================
# INTERNAL TYPES
# ============================================================


class _WorkflowAuthority(
    BaseModel,
):
    model_config = ConfigDict(
        extra="forbid"
    )

    workflow_id: str

    revision: int

    display_name: str

    archived_at: str


class _PayloadMove(
    BaseModel,
):
    model_config = ConfigDict(
        extra="forbid"
    )

    kind: str

    source: str

    quarantine: str


# ============================================================
# LOCK
# ============================================================


_DELETE_LOCK = (
    RLock()
)


# ============================================================
# KNOWN WORKFLOW TABLES
# ============================================================


_WORKFLOW_TABLE_ALLOWLIST = {
    "preparation_sessions",
    "preparation_workflow_metadata",
    "preparation_ui_state",
    "preparation_artifacts",
    "report_selection_workflows",
    "analysis_artifacts",
}


# ============================================================
# HELPERS
# ============================================================


def _required_text(
    value: object,
    *,
    field_name: str,
) -> str:
    normalized = str(
        value
        or
        ""
    ).strip()

    if not normalized:
        raise ValueError(
            (
                f"{field_name} "
                "cannot be empty."
            )
        )

    return normalized


def _utc_now_iso() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


def _api_root() -> Path:
    return (
        Path(
            __file__
        )
        .resolve()
        .parents[
            2
        ]
    )


def _quarantine_root() -> Path:
    configured = (
        os.environ.get(
            "DATALENS_WORKFLOW_DELETE_QUARANTINE_ROOT"
        )
    )

    if (
        configured is not None
        and
        configured.strip()
    ):
        return (
            Path(
                configured
            )
            .expanduser()
            .resolve()
        )

    return (
        _api_root()
        /
        "var"
        /
        "workflow_delete_quarantine"
    ).resolve()


def _analysis_data_root(
    store_root: str,
) -> Path:
    logical_store = (
        Path(
            store_root
        )
        .expanduser()
        .resolve()
    )

    return (
        logical_store.parent
        /
        logical_store.stem
    ).resolve()


def _preparation_data_root(
    store_root: str,
) -> Path:
    return (
        Path(
            store_root
        )
        .expanduser()
        .resolve()
    )


def _resolve_relative_payload(
    *,
    root: Path,
    relative_path: str,
) -> Path:
    normalized = (
        _required_text(
            relative_path,
            field_name=
                "payload_path",
        )
        .replace(
            "\\",
            "/",
        )
    )

    relative = Path(
        normalized
    )

    if (
        relative.is_absolute()
        or
        ".."
        in
        relative.parts
    ):
        raise (
            PreparationWorkflowDeleteIntegrityError(
                (
                    "Workflow deletion payload path "
                    "is outside its configured root. "
                    f"path={relative_path}"
                )
            )
        )

    resolved_root = (
        root.resolve()
    )

    resolved = (
        resolved_root
        /
        relative
    ).resolve()

    try:
        resolved.relative_to(
            resolved_root
        )

    except ValueError as error:
        raise (
            PreparationWorkflowDeleteIntegrityError(
                (
                    "Workflow deletion payload path "
                    "escapes its configured root. "
                    f"path={relative_path}"
                )
            )
        ) from error

    return (
        resolved
    )


def _quoted_identifier(
    value: str,
) -> str:
    return (
        '"'
        +
        value.replace(
            '"',
            '""',
        )
        +
        '"'
    )


# ============================================================
# AUTHORITY
# ============================================================


def _load_authority(
    *,
    connection,
    workflow_id: str,
) -> _WorkflowAuthority:
    row = (
        connection.execute(
            """
            SELECT
                session.workflow_id,
                session.revision,

                metadata.workflow_id
                    AS metadata_workflow_id,

                metadata.display_name,
                metadata.archived_at

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
            PreparationWorkflowDeleteNotFoundError(
                (
                    "Preparation workflow "
                    "does not exist. "
                    f"workflow_id={workflow_id}"
                )
            )
        )

    if (
        row[
            "metadata_workflow_id"
        ]
        is None
    ):
        raise (
            PreparationWorkflowDeleteIntegrityError(
                (
                    "Preparation workflow metadata "
                    "is missing. "
                    f"workflow_id={workflow_id}"
                )
            )
        )

    archived_at = (
        row[
            "archived_at"
        ]
    )

    if (
        archived_at is None
        or
        not str(
            archived_at
        ).strip()
    ):
        raise (
            PreparationWorkflowDeleteNotArchivedError(
                (
                    "A workflow must be archived "
                    "before permanent deletion. "
                    f"workflow_id={workflow_id}"
                )
            )
        )

    return (
        _WorkflowAuthority(
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

            display_name=
                str(
                    row[
                        "display_name"
                    ]
                ),

            archived_at=
                str(
                    archived_at
                ),
        )
    )


# ============================================================
# FUTURE-SCHEMA FAIL-CLOSED GUARD
# ============================================================


def _assert_no_unknown_workflow_ownership(
    *,
    connection,
    workflow_id: str,
) -> None:
    tables = (
        connection.execute(
            """
            SELECT name
            FROM sqlite_master

            WHERE
                type = 'table'
                AND
                name NOT LIKE 'sqlite_%'

            ORDER BY name
            """
        )
        .fetchall()
    )

    unexpected = []

    for table_row in tables:
        table = str(
            table_row[
                "name"
            ]
        )

        columns = (
            connection.execute(
                (
                    "PRAGMA table_info("
                    +
                    _quoted_identifier(
                        table
                    )
                    +
                    ")"
                )
            )
            .fetchall()
        )

        column_names = {
            str(
                row[
                    "name"
                ]
            )

            for row
            in columns
        }

        if (
            "workflow_id"
            not in
            column_names
        ):
            continue

        if (
            table
            in
            _WORKFLOW_TABLE_ALLOWLIST
        ):
            continue

        count = int(
            connection.execute(
                (
                    "SELECT COUNT(*) "
                    "FROM "
                    +
                    _quoted_identifier(
                        table
                    )
                    +
                    " WHERE workflow_id = ?"
                ),
                (
                    workflow_id,
                ),
            )
            .fetchone()[0]
        )

        if count:
            unexpected.append(
                (
                    table,
                    count,
                )
            )

    if unexpected:
        rendered = ", ".join(
            (
                f"{table}={count}"
            )

            for table, count
            in unexpected
        )

        raise (
            PreparationWorkflowDeleteIntegrityError(
                (
                    "Permanent deletion refused "
                    "because unknown workflow-owned "
                    "SQLite rows exist: "
                    +
                    rendered
                )
            )
        )


# ============================================================
# CURRENT ROW COUNTS
# ============================================================


def _workflow_row_counts(
    *,
    connection,
    workflow_id: str,
) -> Dict[
    str,
    int,
]:
    return {
        "preparation_ui_state":
            int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM preparation_ui_state
                    WHERE workflow_id = ?
                    """,
                    (
                        workflow_id,
                    ),
                ).fetchone()[0]
            ),

        "preparation_artifacts":
            int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM preparation_artifacts
                    WHERE workflow_id = ?
                    """,
                    (
                        workflow_id,
                    ),
                ).fetchone()[0]
            ),

        "report_selection_workflows":
            int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM report_selection_workflows
                    WHERE workflow_id = ?
                    """,
                    (
                        workflow_id,
                    ),
                ).fetchone()[0]
            ),

        "analysis_artifacts":
            int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM analysis_artifacts
                    WHERE workflow_id = ?
                    """,
                    (
                        workflow_id,
                    ),
                ).fetchone()[0]
            ),

        "preparation_workflow_metadata":
            int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM preparation_workflow_metadata
                    WHERE workflow_id = ?
                    """,
                    (
                        workflow_id,
                    ),
                ).fetchone()[0]
            ),
    }


# ============================================================
# PAYLOAD MANIFEST
# ============================================================


def _build_payload_manifest(
    *,
    connection,
    workflow_id: str,
    operation_root: Path,
) -> List[
    _PayloadMove
]:
    entries = []

    preparation_rows = (
        connection.execute(
            """
            SELECT
                store_root,
                dataset_id,
                data_path

            FROM preparation_artifacts

            WHERE
                workflow_id = ?

            ORDER BY
                store_root,
                dataset_id
            """,
            (
                workflow_id,
            ),
        )
        .fetchall()
    )

    analysis_rows = (
        connection.execute(
            """
            SELECT
                store_root,
                analysis_id,
                payload_path

            FROM analysis_artifacts

            WHERE
                workflow_id = ?

            ORDER BY
                store_root,
                analysis_id
            """,
            (
                workflow_id,
            ),
        )
        .fetchall()
    )

    index = 0

    for row in preparation_rows:
        source = (
            _resolve_relative_payload(
                root=
                    _preparation_data_root(
                        str(
                            row[
                                "store_root"
                            ]
                        )
                    ),

                relative_path=
                    str(
                        row[
                            "data_path"
                        ]
                    ),
            )
        )

        if not source.is_file():
            raise (
                PreparationWorkflowDeleteIntegrityError(
                    (
                        "Preparation payload file "
                        "required for deletion is "
                        "missing. "
                        f"workflow_id={workflow_id}, "
                        "dataset_id="
                        f"{row['dataset_id']}, "
                        f"path={source}"
                    )
                )
            )

        quarantine = (
            operation_root
            /
            "files"
            /
            (
                "preparation_"
                f"{index:05d}_"
                +
                source.name
            )
        )

        entries.append(
            _PayloadMove(
                kind=
                    "preparation",

                source=
                    str(
                        source
                    ),

                quarantine=
                    str(
                        quarantine
                    ),
            )
        )

        index += 1

    for row in analysis_rows:
        source = (
            _resolve_relative_payload(
                root=
                    _analysis_data_root(
                        str(
                            row[
                                "store_root"
                            ]
                        )
                    ),

                relative_path=
                    str(
                        row[
                            "payload_path"
                        ]
                    ),
            )
        )

        if not source.is_file():
            raise (
                PreparationWorkflowDeleteIntegrityError(
                    (
                        "Analysis payload file "
                        "required for deletion is "
                        "missing. "
                        f"workflow_id={workflow_id}, "
                        "analysis_id="
                        f"{row['analysis_id']}, "
                        f"path={source}"
                    )
                )
            )

        quarantine = (
            operation_root
            /
            "files"
            /
            (
                "analysis_"
                f"{index:05d}_"
                +
                source.name
            )
        )

        entries.append(
            _PayloadMove(
                kind=
                    "analysis",

                source=
                    str(
                        source
                    ),

                quarantine=
                    str(
                        quarantine
                    ),
            )
        )

        index += 1

    source_paths = [
        Path(
            entry.source
        ).resolve()

        for entry
        in entries
    ]

    if (
        len(
            source_paths
        )
        !=
        len(
            set(
                source_paths
            )
        )
    ):
        raise (
            PreparationWorkflowDeleteIntegrityError(
                (
                    "Permanent deletion manifest "
                    "contains duplicate payload paths."
                )
            )
        )

    return (
        entries
    )


# ============================================================
# DURABLE QUARANTINE MANIFEST
# ============================================================


def _write_operation_manifest(
    *,
    operation_root: Path,
    workflow_id: str,
    status: str,
    entries: List[
        _PayloadMove
    ],
) -> None:
    operation_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        operation_root
        /
        "manifest.json"
    )

    temporary = (
        operation_root
        /
        "manifest.json.tmp"
    )

    payload = {
        "manifest_version":
            "workflow_delete_quarantine_v0.1",

        "workflow_id":
            workflow_id,

        "status":
            status,

        "updated_at_utc":
            _utc_now_iso(),

        "files": [
            entry.model_dump()
            for entry
            in entries
        ],
    }

    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


# ============================================================
# MOVE / RESTORE
# ============================================================


def _move_payloads_to_quarantine(
    *,
    operation_root: Path,
    entries: List[
        _PayloadMove
    ],
) -> List[
    _PayloadMove
]:
    operation_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    operation_device = (
        operation_root.stat().st_dev
    )

    for entry in entries:
        source = Path(
            entry.source
        )

        if (
            source.stat().st_dev
            !=
            operation_device
        ):
            raise (
                PreparationWorkflowDeleteIntegrityError(
                    (
                        "Atomic quarantine move "
                        "cannot cross filesystem "
                        "devices. "
                        f"path={source}"
                    )
                )
            )

    moved = []

    for entry in entries:
        source = Path(
            entry.source
        )

        quarantine = Path(
            entry.quarantine
        )

        quarantine.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        os.replace(
            source,
            quarantine,
        )

        moved.append(
            entry
        )

    return (
        moved
    )


def _restore_moved_payloads(
    *,
    operation_root: Path,
    moved: List[
        _PayloadMove
    ],
) -> None:
    errors = []

    for entry in reversed(
        moved
    ):
        source = Path(
            entry.source
        )

        quarantine = Path(
            entry.quarantine
        )

        try:
            if not quarantine.exists():
                if source.exists():
                    continue

                raise RuntimeError(
                    (
                        "Neither live nor quarantined "
                        "payload exists."
                    )
                )

            if source.exists():
                raise RuntimeError(
                    (
                        "Cannot restore quarantined "
                        "payload because live path "
                        "already exists."
                    )
                )

            source.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            os.replace(
                quarantine,
                source,
            )

        except Exception as error:
            errors.append(
                (
                    entry.source,
                    repr(
                        error
                    ),
                )
            )

    if not errors:
        shutil.rmtree(
            operation_root,
            ignore_errors=True,
        )

        return

    raise (
        PreparationWorkflowDeleteRecoveryError(
            (
                "Permanent-delete rollback "
                "could not restore every payload: "
                +
                repr(
                    errors
                )
            )
        )
    )


# ============================================================
# CRASH RECOVERY
# ============================================================


def recover_pending_workflow_deletions(
) -> int:
    root = (
        _quarantine_root()
    )

    if not root.exists():
        return 0

    recovered = 0

    with _DELETE_LOCK:
        operation_directories = [
            path

            for path
            in root.iterdir()

            if path.is_dir()
        ]

        for operation_root in (
            operation_directories
        ):
            manifest_path = (
                operation_root
                /
                "manifest.json"
            )

            if not manifest_path.is_file():
                raise (
                    PreparationWorkflowDeleteRecoveryError(
                        (
                            "Workflow-delete quarantine "
                            "contains an operation without "
                            "a manifest. "
                            f"path={operation_root}"
                        )
                    )
                )

            try:
                raw = json.loads(
                    manifest_path.read_text(
                        encoding="utf-8"
                    )
                )

            except Exception as error:
                raise (
                    PreparationWorkflowDeleteRecoveryError(
                        (
                            "Workflow-delete quarantine "
                            "manifest is unreadable. "
                            f"path={manifest_path}"
                        )
                    )
                ) from error

            workflow_id = (
                _required_text(
                    raw.get(
                        "workflow_id"
                    ),
                    field_name=
                        "workflow_id",
                )
            )

            raw_files = raw.get(
                "files"
            )

            if not isinstance(
                raw_files,
                list,
            ):
                raise (
                    PreparationWorkflowDeleteRecoveryError(
                        (
                            "Workflow-delete quarantine "
                            "manifest files field is "
                            "invalid."
                        )
                    )
                )

            entries = [
                _PayloadMove.model_validate(
                    item
                )

                for item
                in raw_files
            ]

            with sqlite_connection(
                write=False
            ) as connection:
                session_exists = (
                    connection.execute(
                        """
                        SELECT 1
                        FROM preparation_sessions
                        WHERE workflow_id = ?
                        """,
                        (
                            workflow_id,
                        ),
                    )
                    .fetchone()
                    is not None
                )

            if not session_exists:
                shutil.rmtree(
                    operation_root
                )

                recovered += 1

                continue

            _restore_moved_payloads(
                operation_root=
                    operation_root,

                moved=
                    entries,
            )

            recovered += 1

    return (
        recovered
    )


# ============================================================
# DELETE SQLITE ROWS
# ============================================================


def _delete_exact_rows(
    *,
    connection,
    workflow_id: str,
    expected_counts: Dict[
        str,
        int,
    ],
) -> None:
    for table in [
        "report_selection_workflows",
        "analysis_artifacts",
        "preparation_artifacts",
        "preparation_ui_state",
    ]:
        cursor = (
            connection.execute(
                (
                    "DELETE FROM "
                    +
                    table
                    +
                    " WHERE workflow_id = ?"
                ),
                (
                    workflow_id,
                ),
            )
        )

        expected = int(
            expected_counts[
                table
            ]
        )

        if (
            cursor.rowcount
            !=
            expected
        ):
            raise (
                PreparationWorkflowDeleteIntegrityError(
                    (
                        "Permanent deletion row-count "
                        "mismatch. "
                        f"table={table}, "
                        f"expected={expected}, "
                        f"deleted={cursor.rowcount}"
                    )
                )
            )

    session_cursor = (
        connection.execute(
            """
            DELETE FROM preparation_sessions
            WHERE workflow_id = ?
            """,
            (
                workflow_id,
            ),
        )
    )

    if (
        session_cursor.rowcount
        !=
        1
    ):
        raise (
            PreparationWorkflowDeleteIntegrityError(
                (
                    "Permanent deletion did not "
                    "delete exactly one root "
                    "PreparationSession."
                )
            )
        )

    metadata_remaining = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM preparation_workflow_metadata
            WHERE workflow_id = ?
            """,
            (
                workflow_id,
            ),
        ).fetchone()[0]
    )

    if (
        metadata_remaining
        !=
        0
    ):
        raise (
            PreparationWorkflowDeleteIntegrityError(
                (
                    "Workflow metadata was not "
                    "removed by the expected "
                    "ON DELETE CASCADE."
                )
            )
        )


# ============================================================
# PERMANENT DELETE
# ============================================================


def delete_preparation_workflow(
    *,
    workflow_id: str,
    confirmation_workflow_id: str,
    confirmation_display_name: str,
    expected_revision: int,
    _failure_hook: Optional[
        Callable[
            [
                str
            ],
            None,
        ]
    ] = None,
) -> PreparationWorkflowDeleteResult:
    normalized_id = (
        _required_text(
            workflow_id,
            field_name=
                "workflow_id",
        )
    )

    confirmed_id = (
        _required_text(
            confirmation_workflow_id,
            field_name=
                "confirmation_workflow_id",
        )
    )

    confirmed_name = (
        _required_text(
            confirmation_display_name,
            field_name=
                "confirmation_display_name",
        )
    )

    if (
        confirmed_id
        !=
        normalized_id
    ):
        raise (
            PreparationWorkflowDeleteConfirmationError(
                (
                    "Permanent-delete workflow_id "
                    "confirmation does not match "
                    "the requested workflow."
                )
            )
        )

    if (
        not isinstance(
            expected_revision,
            int,
        )
        or
        isinstance(
            expected_revision,
            bool,
        )
        or
        expected_revision < 0
    ):
        raise ValueError(
            (
                "expected_revision must be "
                "a non-negative integer."
            )
        )

    with _DELETE_LOCK:
        recover_pending_workflow_deletions()

        operation_id = (
            "delete-"
            +
            uuid4().hex
        )

        operation_root = (
            _quarantine_root()
            /
            operation_id
        )

        moved = []

        authority = None
        counts = None
        entries = []

        try:
            with sqlite_connection(
                write=True
            ) as connection:
                authority = (
                    _load_authority(
                        connection=
                            connection,

                        workflow_id=
                            normalized_id,
                    )
                )

                if (
                    authority.display_name
                    !=
                    confirmed_name
                ):
                    raise (
                        PreparationWorkflowDeleteConfirmationError(
                            (
                                "Permanent-delete display-name "
                                "confirmation does not match "
                                "the current workflow name."
                            )
                        )
                    )

                if (
                    authority.revision
                    !=
                    expected_revision
                ):
                    raise (
                        PreparationWorkflowDeleteRevisionConflictError(
                            (
                                "Preparation workflow changed "
                                "since deletion was confirmed. "
                                f"expected_revision="
                                f"{expected_revision}, "
                                f"current_revision="
                                f"{authority.revision}"
                            )
                        )
                    )

                _assert_no_unknown_workflow_ownership(
                    connection=
                        connection,

                    workflow_id=
                        normalized_id,
                )

                counts = (
                    _workflow_row_counts(
                        connection=
                            connection,

                        workflow_id=
                            normalized_id,
                    )
                )

                if (
                    counts[
                        "preparation_workflow_metadata"
                    ]
                    !=
                    1
                ):
                    raise (
                        PreparationWorkflowDeleteIntegrityError(
                            (
                                "Permanent deletion expects "
                                "exactly one workflow metadata "
                                "row."
                            )
                        )
                    )

                entries = (
                    _build_payload_manifest(
                        connection=
                            connection,

                        workflow_id=
                            normalized_id,

                        operation_root=
                            operation_root,
                    )
                )

                _write_operation_manifest(
                    operation_root=
                        operation_root,

                    workflow_id=
                        normalized_id,

                    status=
                        "planned",

                    entries=
                        entries,
                )

                moved = (
                    _move_payloads_to_quarantine(
                        operation_root=
                            operation_root,

                        entries=
                            entries,
                    )
                )

                _write_operation_manifest(
                    operation_root=
                        operation_root,

                    workflow_id=
                        normalized_id,

                    status=
                        "quarantined",

                    entries=
                        entries,
                )

                if (
                    _failure_hook
                    is not None
                ):
                    _failure_hook(
                        "after_quarantine"
                    )

                _delete_exact_rows(
                    connection=
                        connection,

                    workflow_id=
                        normalized_id,

                    expected_counts=
                        counts,
                )

                if (
                    _failure_hook
                    is not None
                ):
                    _failure_hook(
                        "before_commit"
                    )

        except Exception as original_error:
            try:
                if moved:
                    _restore_moved_payloads(
                        operation_root=
                            operation_root,

                        moved=
                            moved,
                    )

                elif (
                    operation_root.exists()
                ):
                    shutil.rmtree(
                        operation_root,
                        ignore_errors=True,
                    )

            except Exception as recovery_error:
                raise (
                    PreparationWorkflowDeleteRecoveryError(
                        (
                            "Permanent deletion failed and "
                            "its filesystem rollback also "
                            "failed. "
                            f"original_error="
                            f"{original_error!r}; "
                            f"recovery_error="
                            f"{recovery_error!r}"
                        )
                    )
                ) from recovery_error

            raise

        cleanup_pending = False

        try:
            if operation_root.exists():
                shutil.rmtree(
                    operation_root
                )

        except OSError:
            cleanup_pending = True

        assert (
            authority
            is not None
        )

        assert (
            counts
            is not None
        )

        return (
            PreparationWorkflowDeleteResult(
                workflow_id=
                    authority.workflow_id,

                display_name=
                    authority.display_name,

                preparation_artifacts_deleted=
                    counts[
                        "preparation_artifacts"
                    ],

                analysis_artifacts_deleted=
                    counts[
                        "analysis_artifacts"
                    ],

                preparation_ui_state_deleted=
                    counts[
                        "preparation_ui_state"
                    ],

                report_selection_deleted=
                    counts[
                        "report_selection_workflows"
                    ],

                preparation_session_deleted=
                    1,

                workflow_metadata_deleted=
                    1,

                payload_files_removed_from_live_store=
                    len(
                        entries
                    ),

                quarantine_cleanup_pending=
                    cleanup_pending,
            )
        )
