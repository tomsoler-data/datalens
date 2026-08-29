from __future__ import annotations

import os
import sqlite3
import tempfile

from contextlib import (
    contextmanager,
)

from datetime import (
    datetime,
    timezone,
)

from pathlib import (
    Path,
)

from threading import (
    RLock,
)

from uuid import (
    uuid4,
)


# ============================================================
# VERSION
# ============================================================


SQLITE_DATABASE_RULE_VERSION = (
    "sqlite_database_v0.1"
)

SQLITE_SCHEMA_VERSION = (
    12
)

DATALENS_SQLITE_PATH_ENV = (
    "DATALENS_SQLITE_PATH"
)

LEGACY_PREPARATION_SESSION_PATH_ENV = (
    "DATALENS_PREPARATION_SESSION_STORE_PATH"
)


# ============================================================
# PROCESS-LOCAL SCHEMA LOCK
# ============================================================


_SCHEMA_LOCK = RLock()

_EPHEMERAL_TEST_PATH: (
    Path
    |
    None
) = None


# ============================================================
# TIME
# ============================================================


def utc_now_iso() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


# ============================================================
# DATABASE PATH
# ============================================================


def default_sqlite_database_path() -> Path:
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
        "datalens.sqlite3"
    )


def resolve_sqlite_database_path() -> Path:
    configured = (
        os.getenv(
            DATALENS_SQLITE_PATH_ENV,
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


    # --------------------------------------------------------
    # TRANSITIONAL TEST COMPATIBILITY
    #
    # Old Preparation tests already provide an isolated
    # DATALENS_PREPARATION_SESSION_STORE_PATH ending in
    # .json. During the SQLite migration we reuse that
    # isolation scope but never overwrite the JSON file.
    # --------------------------------------------------------

    legacy_configured = (
        os.getenv(
            LEGACY_PREPARATION_SESSION_PATH_ENV,
            "",
        )
        .strip()
    )


    if legacy_configured:
        legacy_path = (
            Path(
                legacy_configured
            )
            .expanduser()
            .resolve()
        )

        return (
            legacy_path
            .with_suffix(
                ".sqlite3"
            )
        )


    return (
        default_sqlite_database_path()
    )


# ============================================================
# TEST ISOLATION
# ============================================================


def ensure_ephemeral_sqlite_test_path(
    *,
    namespace: str,
) -> Path:
    """
    Configure an isolated process-local SQLite file when a
    test did not explicitly configure one.

    Production code must never call this helper.
    """

    global _EPHEMERAL_TEST_PATH


    configured = (
        os.getenv(
            DATALENS_SQLITE_PATH_ENV,
            "",
        )
        .strip()
    )

    legacy_configured = (
        os.getenv(
            LEGACY_PREPARATION_SESSION_PATH_ENV,
            "",
        )
        .strip()
    )


    if (
        configured
        or
        legacy_configured
    ):
        return (
            resolve_sqlite_database_path()
        )


    if (
        _EPHEMERAL_TEST_PATH
        is None
    ):
        root = Path(
            tempfile.gettempdir()
        )

        safe_namespace = (
            "".join(
                character
                if (
                    character.isalnum()
                    or
                    character
                    in {
                        "-",
                        "_",
                    }
                )
                else
                "-"
                for character
                in namespace
            )
        )

        _EPHEMERAL_TEST_PATH = (
            root
            /
            (
                "datalens-"
                f"{safe_namespace}-"
                f"{uuid4().hex}.sqlite3"
            )
        )


    os.environ[
        DATALENS_SQLITE_PATH_ENV
    ] = str(
        _EPHEMERAL_TEST_PATH
    )


    return (
        _EPHEMERAL_TEST_PATH
    )


# ============================================================
# CONNECTION CONFIGURATION
# ============================================================


def _connect(
) -> sqlite3.Connection:
    path = (
        resolve_sqlite_database_path()
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    connection = sqlite3.connect(
        str(
            path
        ),
        timeout=30.0,
        isolation_level=None,
    )

    connection.row_factory = (
        sqlite3.Row
    )


    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    connection.execute(
        "PRAGMA busy_timeout = 5000"
    )

    connection.execute(
        "PRAGMA journal_mode = WAL"
    )

    connection.execute(
        "PRAGMA synchronous = NORMAL"
    )


    return (
        connection
    )


# ============================================================
# SCHEMA MIGRATIONS
# ============================================================


def _apply_schema_migrations(
    connection: sqlite3.Connection,
) -> None:
    with _SCHEMA_LOCK:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )


        row = connection.execute(
            """
            SELECT COALESCE(
                MAX(version),
                0
            ) AS version
            FROM schema_migrations
            """
        ).fetchone()


        current_version = int(
            row[
                "version"
            ]
        )


        if (
            current_version
            >
            SQLITE_SCHEMA_VERSION
        ):
            raise RuntimeError(
                (
                    "DataLens SQLite schema is newer "
                    "than this application version. "
                    f"database_version={current_version}, "
                    "application_version="
                    f"{SQLITE_SCHEMA_VERSION}"
                )
            )


        if (
            current_version
            <
            1
        ):
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS
                    preparation_sessions (
                        workflow_id TEXT PRIMARY KEY,
                        revision INTEGER NOT NULL
                            CHECK (revision >= 0),
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )


                connection.execute(
                    """
                    INSERT INTO schema_migrations (
                        version,
                        name,
                        applied_at
                    )
                    VALUES (
                        ?,
                        ?,
                        ?
                    )
                    """,
                    (
                        1,
                        (
                            "initial_control_plane_"
                            "preparation_sessions"
                        ),
                        utc_now_iso(),
                    ),
                )


                connection.execute(
                    "COMMIT"
                )

            except Exception:
                if connection.in_transaction:
                    connection.execute(
                        "ROLLBACK"
                    )

                raise


        # ====================================================
        # SQLITE_SCHEMA_V2_PREPARATION_UI_STATE
        # ====================================================


        if (
            current_version
            <
            2
        ):
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS
                    preparation_ui_state (
                        workflow_id TEXT PRIMARY KEY,
                        revision INTEGER NOT NULL
                            CHECK (revision >= 0),
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )


                connection.execute(
                    """
                    INSERT INTO schema_migrations (
                        version,
                        name,
                        applied_at
                    )
                    VALUES (
                        ?,
                        ?,
                        ?
                    )
                    """,
                    (
                        2,
                        "preparation_ui_state_control_plane",
                        utc_now_iso(),
                    ),
                )


                connection.execute(
                    "COMMIT"
                )

            except Exception:
                if connection.in_transaction:
                    connection.execute(
                        "ROLLBACK"
                    )

                raise


        # ====================================================
        # SQLITE_SCHEMA_V3_PREPARATION_ARTIFACT_INDEX
        # ====================================================


        if (
            current_version
            <
            3
        ):
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS
                    preparation_artifacts (
                        store_root TEXT NOT NULL,
                        workflow_id TEXT NOT NULL,
                        dataset_id TEXT NOT NULL,
                        dataset_filename TEXT NOT NULL,
                        stage TEXT NOT NULL
                            CHECK (
                                stage IN (
                                    'source',
                                    'clean',
                                    'transform',
                                    'combine'
                                )
                            ),
                        rows INTEGER NOT NULL
                            CHECK (rows > 0),
                        columns INTEGER NOT NULL
                            CHECK (columns > 0),
                        parent_dataset_ids_json TEXT NOT NULL,
                        evidence_refs_json TEXT NOT NULL,
                        datetime_dtypes_json TEXT NOT NULL,
                        data_path TEXT NOT NULL,

                        PRIMARY KEY (
                            store_root,
                            workflow_id,
                            dataset_id
                        )
                    )
                    """
                )


                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_preparation_artifacts_scope_workflow
                    ON preparation_artifacts (
                        store_root,
                        workflow_id
                    )
                    """
                )


                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS
                    preparation_artifact_store_state (
                        store_root TEXT PRIMARY KEY,
                        legacy_manifest_imported INTEGER NOT NULL
                            CHECK (
                                legacy_manifest_imported
                                IN (0, 1)
                            ),
                        legacy_manifest_version TEXT,
                        migrated_at TEXT NOT NULL
                    )
                    """
                )


                connection.execute(
                    """
                    INSERT INTO schema_migrations (
                        version,
                        name,
                        applied_at
                    )
                    VALUES (
                        ?,
                        ?,
                        ?
                    )
                    """,
                    (
                        3,
                        "preparation_artifact_sqlite_index",
                        utc_now_iso(),
                    ),
                )


                connection.execute(
                    "COMMIT"
                )

            except Exception:
                if connection.in_transaction:
                    connection.execute(
                        "ROLLBACK"
                    )

                raise


        # ====================================================
        # SQLITE_SCHEMA_V4_REPORT_SELECTION
        # ====================================================


        if (
            current_version
            <
            4
        ):
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS
                    report_selection_workflows (
                        store_root TEXT NOT NULL,
                        workflow_id TEXT NOT NULL,
                        revision INTEGER NOT NULL
                            CHECK (revision >= 0),
                        payload_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL,

                        PRIMARY KEY (
                            store_root,
                            workflow_id
                        )
                    )
                    """
                )


                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_report_selection_workflow
                    ON report_selection_workflows (
                        workflow_id
                    )
                    """
                )


                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS
                    report_selection_store_state (
                        store_root TEXT PRIMARY KEY,

                        legacy_json_imported INTEGER NOT NULL
                            CHECK (
                                legacy_json_imported
                                IN (0, 1)
                            ),

                        legacy_rule_version TEXT,

                        migrated_at TEXT NOT NULL
                    )
                    """
                )


                connection.execute(
                    """
                    INSERT INTO schema_migrations (
                        version,
                        name,
                        applied_at
                    )
                    VALUES (
                        ?,
                        ?,
                        ?
                    )
                    """,
                    (
                        4,
                        "report_selection_control_plane",
                        utc_now_iso(),
                    ),
                )


                connection.execute(
                    "COMMIT"
                )

            except Exception:
                if connection.in_transaction:
                    connection.execute(
                        "ROLLBACK"
                    )

                raise


        # ====================================================
        # SQLITE_SCHEMA_V5_ANALYSIS_ARTIFACT_INDEX
        # ====================================================


        if (
            current_version
            <
            5
        ):
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS
                    analysis_artifacts (
                        store_root TEXT NOT NULL,

                        analysis_id TEXT NOT NULL,

                        workflow_id TEXT NOT NULL,

                        trace_id TEXT NOT NULL,

                        source_type TEXT NOT NULL
                            CHECK (
                                source_type
                                IN (
                                    'initial_request',
                                    'follow_up_prompt',
                                    'document_request',
                                    'automatic'
                                )
                            ),

                        objective TEXT NOT NULL,

                        executed INTEGER NOT NULL
                            CHECK (
                                executed
                                IN (0, 1)
                            ),

                        executed_count INTEGER NOT NULL
                            CHECK (
                                executed_count >= 0
                            ),

                        created_at_utc TEXT NOT NULL,

                        rule_version TEXT NOT NULL,

                        payload_path TEXT NOT NULL,

                        payload_json_bytes INTEGER NOT NULL
                            CHECK (
                                payload_json_bytes >= 0
                            ),

                        payload_file_bytes INTEGER NOT NULL
                            CHECK (
                                payload_file_bytes >= 0
                            ),

                        payload_sha256 TEXT NOT NULL
                            CHECK (
                                length(payload_sha256) = 64
                            ),

                        PRIMARY KEY (
                            store_root,
                            analysis_id
                        )
                    )
                    """
                )


                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_analysis_artifacts_scope_workflow
                    ON analysis_artifacts (
                        store_root,
                        workflow_id,
                        created_at_utc,
                        analysis_id
                    )
                    """
                )


                # trace_id is intentionally NOT UNIQUE.
                #
                # Requested-analysis report artifacts can share
                # one server trace while keeping distinct
                # analysis_id identities.
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_analysis_artifacts_scope_trace
                    ON analysis_artifacts (
                        store_root,
                        trace_id
                    )
                    """
                )


                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS
                    analysis_artifact_store_state (
                        store_root TEXT PRIMARY KEY,

                        legacy_json_imported INTEGER NOT NULL
                            CHECK (
                                legacy_json_imported
                                IN (0, 1)
                            ),

                        legacy_rule_version TEXT,

                        migrated_at TEXT NOT NULL
                    )
                    """
                )


                connection.execute(
                    """
                    INSERT INTO schema_migrations (
                        version,
                        name,
                        applied_at
                    )
                    VALUES (
                        ?,
                        ?,
                        ?
                    )
                    """,
                    (
                        5,
                        "analysis_artifact_metadata_index",
                        utc_now_iso(),
                    ),
                )


                connection.execute(
                    "COMMIT"
                )

            except Exception:
                if connection.in_transaction:
                    connection.execute(
                        "ROLLBACK"
                    )

                raise


        # ====================================================
        # SQLITE_SCHEMA_V6_PREPARATION_WORKFLOW_LIFECYCLE
        # PREPARATION_WORKFLOW_LIFECYCLE_V0_1
        # ====================================================


        if (
            current_version
            <
            6
        ):
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS
                    preparation_workflow_lifecycle (
                        workflow_id TEXT PRIMARY KEY,

                        archived_at TEXT,

                        updated_at TEXT NOT NULL,

                        FOREIGN KEY (
                            workflow_id
                        )
                        REFERENCES preparation_sessions (
                            workflow_id
                        )
                        ON DELETE CASCADE
                    )
                    """
                )


                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_preparation_workflow_lifecycle_archived_at
                    ON preparation_workflow_lifecycle (
                        archived_at
                    )
                    """
                )


                connection.execute(
                    """
                    INSERT INTO schema_migrations (
                        version,
                        name,
                        applied_at
                    )
                    VALUES (
                        ?,
                        ?,
                        ?
                    )
                    """,
                    (
                        6,
                        (
                            "preparation_workflow_"
                            "lifecycle_control_plane"
                        ),
                        utc_now_iso(),
                    ),
                )


                connection.execute(
                    "COMMIT"
                )


            except Exception:
                if connection.in_transaction:
                    connection.execute(
                        "ROLLBACK"
                    )

                raise


        # ====================================================
        # SQLITE_SCHEMA_V7_PREPARATION_WORKFLOW_METADATA
        # PREPARATION_WORKFLOW_METADATA_V0_1
        # ====================================================


        if (
            current_version
            <
            7
        ):
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS
                    preparation_workflow_metadata (
                        workflow_id TEXT PRIMARY KEY,

                        display_name TEXT NOT NULL
                            CHECK (
                                length(
                                    trim(
                                        display_name
                                    )
                                )
                                BETWEEN 1 AND 120
                            ),

                        name_source TEXT NOT NULL
                            CHECK (
                                name_source
                                IN (
                                    'user',
                                    'automatic'
                                )
                            ),

                        archived_at TEXT,

                        updated_at TEXT NOT NULL,

                        FOREIGN KEY (
                            workflow_id
                        )
                        REFERENCES preparation_sessions (
                            workflow_id
                        )
                        ON DELETE CASCADE
                    )
                    """
                )


                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_preparation_workflow_metadata_archived_at
                    ON preparation_workflow_metadata (
                        archived_at
                    )
                    """
                )


                # Existing sessions receive deterministic names.
                #
                # Any archive state already recorded by schema
                # v6 is preserved.
                connection.execute(
                    """
                    INSERT INTO
                        preparation_workflow_metadata (
                            workflow_id,
                            display_name,
                            name_source,
                            archived_at,
                            updated_at
                        )

                    SELECT
                        session.workflow_id,

                        (
                            'Analyse - '
                            ||
                            replace(
                                substr(
                                    session.created_at,
                                    1,
                                    16
                                ),
                                'T',
                                ' '
                            )
                            ||
                            ' UTC'
                        ),

                        'automatic',

                        lifecycle.archived_at,

                        CASE
                            WHEN
                                lifecycle.updated_at
                                IS NOT NULL
                                AND
                                lifecycle.updated_at
                                >
                                session.updated_at
                            THEN
                                lifecycle.updated_at

                            ELSE
                                session.updated_at
                        END

                    FROM preparation_sessions
                        AS session

                    LEFT JOIN
                        preparation_workflow_lifecycle
                        AS lifecycle

                        ON lifecycle.workflow_id
                        =
                        session.workflow_id
                    """
                )


                # Every future PreparationSession automatically
                # receives its product metadata, regardless of
                # which server-side creation path inserted it.
                connection.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS
                    trg_preparation_workflow_metadata_after_session_insert

                    AFTER INSERT ON preparation_sessions

                    FOR EACH ROW

                    WHEN NOT EXISTS (
                        SELECT 1
                        FROM preparation_workflow_metadata
                        WHERE workflow_id = NEW.workflow_id
                    )

                    BEGIN
                        INSERT INTO
                            preparation_workflow_metadata (
                                workflow_id,
                                display_name,
                                name_source,
                                archived_at,
                                updated_at
                            )
                        VALUES (
                            NEW.workflow_id,

                            (
                                'Analyse - '
                                ||
                                replace(
                                    substr(
                                        NEW.created_at,
                                        1,
                                        16
                                    ),
                                    'T',
                                    ' '
                                )
                                ||
                                ' UTC'
                            ),

                            'automatic',

                            NULL,

                            NEW.updated_at
                        );
                    END
                    """
                )


                # The richer metadata table now owns the
                # archive state.
                connection.execute(
                    """
                    DROP TABLE
                        preparation_workflow_lifecycle
                    """
                )


                connection.execute(
                    """
                    INSERT INTO schema_migrations (
                        version,
                        name,
                        applied_at
                    )
                    VALUES (
                        ?,
                        ?,
                        ?
                    )
                    """,
                    (
                        7,
                        (
                            "preparation_workflow_"
                            "metadata_control_plane"
                        ),
                        utc_now_iso(),
                    ),
                )


                connection.execute(
                    "COMMIT"
                )


            except Exception:
                if connection.in_transaction:
                    connection.execute(
                        "ROLLBACK"
                    )

                raise


        # ====================================================
        # SQLITE_SCHEMA_V8_ML_MODEL_ARTIFACT_INDEX
        # ML_MODEL_ARTIFACT_SQLITE_INDEX_V0_1
        # ====================================================


        if (
            current_version
            <
            8
        ):
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS
                    ml_model_artifacts (
                        store_root TEXT NOT NULL,

                        model_id TEXT NOT NULL,

                        workflow_id TEXT NOT NULL,

                        dataset_id TEXT NOT NULL,

                        problem_type TEXT NOT NULL
                            CHECK (
                                problem_type
                                IN (
                                    'regression',
                                    'classification'
                                )
                            ),

                        target_column TEXT NOT NULL,

                        estimator_key TEXT NOT NULL,

                        training_contract_json TEXT NOT NULL,

                        metrics_json TEXT NOT NULL,

                        train_rows INTEGER NOT NULL
                            CHECK (
                                train_rows > 0
                            ),

                        test_rows INTEGER NOT NULL
                            CHECK (
                                test_rows > 0
                            ),

                        created_at_utc TEXT NOT NULL,

                        serialization_format TEXT NOT NULL
                            CHECK (
                                serialization_format
                                =
                                'joblib'
                            ),

                        rule_version TEXT NOT NULL,

                        model_path TEXT NOT NULL,

                        model_file_bytes INTEGER NOT NULL
                            CHECK (
                                model_file_bytes > 0
                            ),

                        model_sha256 TEXT NOT NULL
                            CHECK (
                                length(
                                    model_sha256
                                )
                                =
                                64
                            ),

                        PRIMARY KEY (
                            store_root,
                            model_id
                        )
                    )
                    """
                )


                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_ml_model_artifacts_scope_workflow
                    ON ml_model_artifacts (
                        store_root,
                        workflow_id,
                        created_at_utc,
                        model_id
                    )
                    """
                )


                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_ml_model_artifacts_scope_dataset
                    ON ml_model_artifacts (
                        store_root,
                        workflow_id,
                        dataset_id,
                        created_at_utc,
                        model_id
                    )
                    """
                )


                connection.execute(
                    """
                    INSERT INTO schema_migrations (
                        version,
                        name,
                        applied_at
                    )
                    VALUES (
                        ?,
                        ?,
                        ?
                    )
                    """,
                    (
                        8,
                        "ml_model_artifact_metadata_index",
                        utc_now_iso(),
                    ),
                )


                connection.execute(
                    "COMMIT"
                )


            except Exception:
                if connection.in_transaction:
                    connection.execute(
                        "ROLLBACK"
                    )

                raise


        # ====================================================
        # SQLITE_SCHEMA_V9_ML_EXPERIMENT_PROVENANCE
        # ML_EXPERIMENT_PROVENANCE_V0_1
        # ====================================================


        if (
            current_version
            <
            9
        ):
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            try:
                existing_columns = {
                    str(
                        row[
                            "name"
                        ]
                    )

                    for row
                    in connection.execute(
                        """
                        PRAGMA table_info(
                            ml_model_artifacts
                        )
                        """
                    ).fetchall()
                }


                # Legacy v8 Model Artifacts cannot honestly be
                # assigned a historical Preparation revision
                # that was never persisted.
                #
                # The new columns therefore remain nullable for
                # backward compatibility. Every Model Artifact
                # created after Experiment Provenance is wired
                # into the store will populate both fields.

                if (
                    "experiment_id"
                    not in
                    existing_columns
                ):
                    connection.execute(
                        """
                        ALTER TABLE
                            ml_model_artifacts

                        ADD COLUMN
                            experiment_id TEXT
                        """
                    )


                if (
                    "experiment_provenance_json"
                    not in
                    existing_columns
                ):
                    connection.execute(
                        """
                        ALTER TABLE
                            ml_model_artifacts

                        ADD COLUMN
                            experiment_provenance_json TEXT
                        """
                    )


                connection.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS
                    idx_ml_model_artifacts_scope_experiment

                    ON ml_model_artifacts (
                        store_root,
                        experiment_id
                    )

                    WHERE
                        experiment_id
                        IS NOT NULL
                    """
                )


                connection.execute(
                    """
                    INSERT INTO schema_migrations (
                        version,
                        name,
                        applied_at
                    )
                    VALUES (
                        ?,
                        ?,
                        ?
                    )
                    """,
                    (
                        9,
                        (
                            "ml_experiment_"
                            "provenance_metadata"
                        ),
                        utc_now_iso(),
                    ),
                )


                connection.execute(
                    "COMMIT"
                )


            except Exception:
                if connection.in_transaction:
                    connection.execute(
                        "ROLLBACK"
                    )

                raise


        # ====================================================
        # SQLITE_SCHEMA_V10_ML_MONITORING_PROFILE
        # ML_MONITORING_PROFILE_V0_1
        # ====================================================


        if (
            current_version
            <
            10
        ):
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS
                    ml_monitoring_profiles (
                        store_root TEXT NOT NULL,

                        profile_id TEXT NOT NULL,

                        model_id TEXT NOT NULL,

                        workflow_id TEXT NOT NULL,

                        dataset_id TEXT NOT NULL,

                        experiment_id TEXT NOT NULL,

                        preparation_session_revision
                            INTEGER NOT NULL
                            CHECK (
                                preparation_session_revision
                                >=
                                0
                            ),

                        training_contract_sha256
                            TEXT NOT NULL
                            CHECK (
                                length(
                                    training_contract_sha256
                                )
                                =
                                64
                            ),

                        created_at_utc TEXT NOT NULL,

                        reference_scope TEXT NOT NULL
                            CHECK (
                                reference_scope
                                =
                                'training_split'
                            ),

                        reference_row_count INTEGER NOT NULL
                            CHECK (
                                reference_row_count
                                >
                                0
                            ),

                        privacy_scope TEXT NOT NULL
                            CHECK (
                                privacy_scope
                                =
                                'aggregate_only'
                            ),

                        categorical_identity TEXT NOT NULL
                            CHECK (
                                categorical_identity
                                =
                                'sha256'
                            ),

                        rule_version TEXT NOT NULL,

                        payload_json TEXT NOT NULL,

                        PRIMARY KEY (
                            store_root,
                            model_id
                        ),

                        UNIQUE (
                            store_root,
                            profile_id
                        ),

                        UNIQUE (
                            store_root,
                            experiment_id
                        ),

                        FOREIGN KEY (
                            store_root,
                            model_id
                        )
                        REFERENCES ml_model_artifacts (
                            store_root,
                            model_id
                        )
                        ON DELETE CASCADE
                    )
                    """
                )


                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_ml_monitoring_profiles_scope_workflow

                    ON ml_monitoring_profiles (
                        store_root,
                        workflow_id,
                        created_at_utc,
                        model_id
                    )
                    """
                )


                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_ml_monitoring_profiles_scope_dataset

                    ON ml_monitoring_profiles (
                        store_root,
                        workflow_id,
                        dataset_id,
                        created_at_utc,
                        model_id
                    )
                    """
                )


                connection.execute(
                    """
                    INSERT INTO schema_migrations (
                        version,
                        name,
                        applied_at
                    )
                    VALUES (
                        ?,
                        ?,
                        ?
                    )
                    """,
                    (
                        10,
                        (
                            "ml_monitoring_"
                            "profile_metadata"
                        ),
                        utc_now_iso(),
                    ),
                )


                connection.execute(
                    "COMMIT"
                )


            except Exception:
                if connection.in_transaction:
                    connection.execute(
                        "ROLLBACK"
                    )

                raise


        # ====================================================
        # SQLITE_SCHEMA_V11_ML_DRIFT_EVALUATION
        # ML_DRIFT_EVALUATION_STORE_V0_1
        # ====================================================


        if (
            current_version
            <
            11
        ):
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS
                    ml_drift_evaluations (
                        store_root TEXT NOT NULL,

                        evaluation_id TEXT NOT NULL,

                        profile_id TEXT NOT NULL,

                        model_id TEXT NOT NULL,

                        workflow_id TEXT NOT NULL,

                        reference_dataset_id TEXT NOT NULL,

                        observed_dataset_id TEXT NOT NULL,

                        experiment_id TEXT NOT NULL,

                        preparation_session_revision
                            INTEGER NOT NULL
                            CHECK (
                                preparation_session_revision
                                >=
                                0
                            ),

                        training_contract_sha256
                            TEXT NOT NULL
                            CHECK (
                                length(
                                    training_contract_sha256
                                )
                                =
                                64
                            ),

                        evaluated_at_utc TEXT NOT NULL,

                        observed_row_count INTEGER NOT NULL
                            CHECK (
                                observed_row_count
                                >
                                0
                            ),

                        warning_feature_count INTEGER NOT NULL
                            CHECK (
                                warning_feature_count
                                >=
                                0
                            ),

                        drift_feature_count INTEGER NOT NULL
                            CHECK (
                                drift_feature_count
                                >=
                                0
                            ),

                        overall_status TEXT NOT NULL
                            CHECK (
                                overall_status
                                IN (
                                    'ok',
                                    'warning',
                                    'drift'
                                )
                            ),

                        privacy_scope TEXT NOT NULL
                            CHECK (
                                privacy_scope
                                =
                                'aggregate_only'
                            ),

                        rule_version TEXT NOT NULL,

                        payload_json TEXT NOT NULL,

                        PRIMARY KEY (
                            store_root,
                            evaluation_id
                        ),

                        FOREIGN KEY (
                            store_root,
                            profile_id
                        )
                        REFERENCES ml_monitoring_profiles (
                            store_root,
                            profile_id
                        )
                        ON DELETE CASCADE
                    )
                    """
                )


                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_ml_drift_evaluations_scope_workflow

                    ON ml_drift_evaluations (
                        store_root,
                        workflow_id,
                        evaluated_at_utc,
                        evaluation_id
                    )
                    """
                )


                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_ml_drift_evaluations_scope_model

                    ON ml_drift_evaluations (
                        store_root,
                        model_id,
                        evaluated_at_utc,
                        evaluation_id
                    )
                    """
                )


                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_ml_drift_evaluations_scope_observed_dataset

                    ON ml_drift_evaluations (
                        store_root,
                        workflow_id,
                        observed_dataset_id,
                        evaluated_at_utc,
                        evaluation_id
                    )
                    """
                )


                connection.execute(
                    """
                    INSERT INTO schema_migrations (
                        version,
                        name,
                        applied_at
                    )
                    VALUES (
                        ?,
                        ?,
                        ?
                    )
                    """,
                    (
                        11,
                        "ml_drift_evaluation_metadata",
                        utc_now_iso(),
                    ),
                )


                connection.execute(
                    "COMMIT"
                )


            except Exception:
                if connection.in_transaction:
                    connection.execute(
                        "ROLLBACK"
                    )

                raise



        # ====================================================
        # SQLITE_SCHEMA_V12_ML_DRIFT_OBSERVED_SNAPSHOT
        # ML_DRIFT_OBSERVED_SNAPSHOT_BINDING_V0_1
        # ====================================================


        if (
            current_version
            <
            12
        ):
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            try:
                connection.execute(
                    """
                    ALTER TABLE ml_drift_evaluations

                    ADD COLUMN
                    observed_preparation_session_revision
                        INTEGER
                        CHECK (
                            observed_preparation_session_revision
                            IS NULL
                            OR
                            observed_preparation_session_revision
                            >=
                            0
                        )
                    """
                )


                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_ml_drift_evaluations_observed_revision

                    ON ml_drift_evaluations (
                        store_root,
                        workflow_id,
                        observed_preparation_session_revision,
                        evaluated_at_utc,
                        evaluation_id
                    )
                    """
                )


                connection.execute(
                    """
                    INSERT INTO schema_migrations (
                        version,
                        name,
                        applied_at
                    )
                    VALUES (
                        ?,
                        ?,
                        ?
                    )
                    """,
                    (
                        12,
                        "ml_drift_observed_snapshot_binding",
                        utc_now_iso(),
                    ),
                )


                connection.execute(
                    "COMMIT"
                )


            except Exception:
                if connection.in_transaction:
                    connection.execute(
                        "ROLLBACK"
                    )

                raise



# ============================================================
# PUBLIC CONNECTION CONTEXT
# ============================================================


@contextmanager
def sqlite_connection(
    *,
    write: bool = False,
):
    connection = (
        _connect()
    )

    try:
        _apply_schema_migrations(
            connection
        )


        if write:
            connection.execute(
                "BEGIN IMMEDIATE"
            )


        yield (
            connection
        )


        if (
            write
            and
            connection.in_transaction
        ):
            connection.execute(
                "COMMIT"
            )

    except Exception:
        if connection.in_transaction:
            connection.execute(
                "ROLLBACK"
            )

        raise

    finally:
        connection.close()


# ============================================================
# INSPECTION
# ============================================================


def sqlite_schema_version() -> int:
    with sqlite_connection(
        write=False
    ) as connection:
        row = connection.execute(
            """
            SELECT COALESCE(
                MAX(version),
                0
            ) AS version
            FROM schema_migrations
            """
        ).fetchone()


        return int(
            row[
                "version"
            ]
        )