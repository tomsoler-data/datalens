from __future__ import annotations


import json
import re

from pathlib import (
    Path,
    PurePosixPath,
)

from typing import (
    Any,
)


from app.persistence.sqlite_database import (
    sqlite_connection,
    utc_now_iso,
)


# ========================================================
# VERSION
# ========================================================


ANALYSIS_ARTIFACT_INDEX_VERSION = (
    "analysis_artifact_sqlite_index_v0.1"
)


ANALYSIS_SOURCE_TYPES = {
    "initial_request",
    "follow_up_prompt",
    "document_request",
    "automatic",
}


SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)


# ========================================================
# ERROR
# ========================================================


class AnalysisArtifactIndexError(
    RuntimeError
):
    pass


# ========================================================
# SCOPE
# ========================================================


def analysis_artifact_store_scope(
    store_path: Path,
) -> str:
    """
    Scope metadata to the configured legacy store path.

    Tests may configure an independent
    DATALENS_ANALYSIS_ARTIFACT_STORE_PATH, so two stores
    sharing one SQLite database remain isolated.
    """

    return str(
        store_path
        .expanduser()
        .resolve()
    )


# ========================================================
# VALIDATION
# ========================================================


def _required_text(
    entry: dict[
        str,
        Any,
    ],
    field: str,
) -> str:
    value = str(
        entry.get(
            field,
            "",
        )
    ).strip()


    if not value:
        raise AnalysisArtifactIndexError(
            (
                "Analysis artifact metadata field "
                f"{field!r} cannot be empty."
            )
        )


    return value


def _normalize_payload_path(
    raw: object,
) -> str:
    value = str(
        raw
        or
        ""
    ).strip()


    if not value:
        raise AnalysisArtifactIndexError(
            "payload_path cannot be empty."
        )


    value = value.replace(
        "\\",
        "/",
    )


    path = PurePosixPath(
        value
    )


    if (
        path.is_absolute()
        or
        ".."
        in path.parts
    ):
        raise AnalysisArtifactIndexError(
            (
                "payload_path must remain relative "
                "to the configured data-plane root."
            )
        )


    if (
        not path.parts
        or
        path.name
        in {
            "",
            ".",
        }
    ):
        raise AnalysisArtifactIndexError(
            "payload_path is invalid."
        )


    return path.as_posix()


def validate_analysis_artifact_index_entry(
    entry: object,
) -> dict[
    str,
    Any,
]:
    if not isinstance(
        entry,
        dict,
    ):
        raise AnalysisArtifactIndexError(
            (
                "Analysis artifact index entry "
                "must be an object."
            )
        )


    analysis_id = _required_text(
        entry,
        "analysis_id",
    )

    workflow_id = _required_text(
        entry,
        "workflow_id",
    )

    trace_id = _required_text(
        entry,
        "trace_id",
    )

    source_type = _required_text(
        entry,
        "source_type",
    )


    if (
        source_type
        not in
        ANALYSIS_SOURCE_TYPES
    ):
        raise AnalysisArtifactIndexError(
            (
                "Unsupported AnalysisArtifact "
                f"source_type={source_type!r}."
            )
        )


    objective = _required_text(
        entry,
        "objective",
    )

    created_at_utc = _required_text(
        entry,
        "created_at_utc",
    )

    rule_version = _required_text(
        entry,
        "rule_version",
    )

    payload_path = (
        _normalize_payload_path(
            entry.get(
                "payload_path"
            )
        )
    )


    executed_raw = entry.get(
        "executed"
    )


    if not isinstance(
        executed_raw,
        bool,
    ):
        raise AnalysisArtifactIndexError(
            (
                "executed must be a boolean."
            )
        )


    try:
        executed_count = int(
            entry.get(
                "executed_count"
            )
        )

    except Exception as error:
        raise AnalysisArtifactIndexError(
            (
                "executed_count must "
                "be an integer."
            )
        ) from error


    if executed_count < 0:
        raise AnalysisArtifactIndexError(
            (
                "executed_count cannot "
                "be negative."
            )
        )


    try:
        payload_json_bytes = int(
            entry.get(
                "payload_json_bytes"
            )
        )

        payload_file_bytes = int(
            entry.get(
                "payload_file_bytes"
            )
        )

    except Exception as error:
        raise AnalysisArtifactIndexError(
            (
                "payload byte sizes must "
                "be integers."
            )
        ) from error


    if (
        payload_json_bytes
        <
        0
        or
        payload_file_bytes
        <
        0
    ):
        raise AnalysisArtifactIndexError(
            (
                "payload byte sizes cannot "
                "be negative."
            )
        )


    payload_sha256 = str(
        entry.get(
            "payload_sha256",
            "",
        )
    ).strip().lower()


    if (
        SHA256_PATTERN.fullmatch(
            payload_sha256
        )
        is None
    ):
        raise AnalysisArtifactIndexError(
            (
                "payload_sha256 must be a "
                "64-character lowercase hex digest."
            )
        )


    return {
        "analysis_id":
            analysis_id,

        "workflow_id":
            workflow_id,

        "trace_id":
            trace_id,

        "source_type":
            source_type,

        "objective":
            objective,

        "executed":
            bool(
                executed_raw
            ),

        "executed_count":
            executed_count,

        "created_at_utc":
            created_at_utc,

        "rule_version":
            rule_version,

        "payload_path":
            payload_path,

        "payload_json_bytes":
            payload_json_bytes,

        "payload_file_bytes":
            payload_file_bytes,

        "payload_sha256":
            payload_sha256,
    }


# ========================================================
# COMPLETE SCOPE REPLACEMENT
# ========================================================


def replace_analysis_artifact_index_scope(
    *,
    store_path: Path,
    entries: list[
        dict[
            str,
            Any,
        ]
    ],
    legacy_json_imported: bool,
    legacy_rule_version: (
        str
        |
        None
    ),
) -> None:
    validated = [
        validate_analysis_artifact_index_entry(
            entry
        )

        for entry
        in entries
    ]


    analysis_ids = [
        entry[
            "analysis_id"
        ]

        for entry
        in validated
    ]


    if (
        len(
            analysis_ids
        )
        !=
        len(
            set(
                analysis_ids
            )
        )
    ):
        raise AnalysisArtifactIndexError(
            (
                "Analysis artifact index scope "
                "contains duplicate analysis_id values."
            )
        )


    scope = (
        analysis_artifact_store_scope(
            store_path
        )
    )


    with sqlite_connection(
        write=True
    ) as connection:
        connection.execute(
            """
            DELETE FROM analysis_artifacts
            WHERE store_root = ?
            """,
            (
                scope,
            ),
        )


        for entry in validated:
            connection.execute(
                """
                INSERT INTO analysis_artifacts (
                    store_root,
                    analysis_id,
                    workflow_id,
                    trace_id,
                    source_type,
                    objective,
                    executed,
                    executed_count,
                    created_at_utc,
                    rule_version,
                    payload_path,
                    payload_json_bytes,
                    payload_file_bytes,
                    payload_sha256
                )
                VALUES (
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?
                )
                """,
                (
                    scope,
                    entry[
                        "analysis_id"
                    ],
                    entry[
                        "workflow_id"
                    ],
                    entry[
                        "trace_id"
                    ],
                    entry[
                        "source_type"
                    ],
                    entry[
                        "objective"
                    ],
                    (
                        1
                        if entry[
                            "executed"
                        ]
                        else
                        0
                    ),
                    entry[
                        "executed_count"
                    ],
                    entry[
                        "created_at_utc"
                    ],
                    entry[
                        "rule_version"
                    ],
                    entry[
                        "payload_path"
                    ],
                    entry[
                        "payload_json_bytes"
                    ],
                    entry[
                        "payload_file_bytes"
                    ],
                    entry[
                        "payload_sha256"
                    ],
                ),
            )


        connection.execute(
            """
            INSERT INTO analysis_artifact_store_state (
                store_root,
                legacy_json_imported,
                legacy_rule_version,
                migrated_at
            )
            VALUES (
                ?,
                ?,
                ?,
                ?
            )
            ON CONFLICT(store_root)
            DO UPDATE SET
                legacy_json_imported =
                    excluded.legacy_json_imported,
                legacy_rule_version =
                    excluded.legacy_rule_version,
                migrated_at =
                    excluded.migrated_at
            """,
            (
                scope,
                (
                    1
                    if legacy_json_imported
                    else
                    0
                ),
                (
                    None
                    if legacy_rule_version
                    is None
                    else
                    str(
                        legacy_rule_version
                    )
                ),
                utc_now_iso(),
            ),
        )


# ========================================================
# READ
# ========================================================


def load_analysis_artifact_index_scope(
    *,
    store_path: Path,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    scope = (
        analysis_artifact_store_scope(
            store_path
        )
    )


    with sqlite_connection(
        write=False
    ) as connection:
        rows = connection.execute(
            """
            SELECT
                analysis_id,
                workflow_id,
                trace_id,
                source_type,
                objective,
                executed,
                executed_count,
                created_at_utc,
                rule_version,
                payload_path,
                payload_json_bytes,
                payload_file_bytes,
                payload_sha256
            FROM analysis_artifacts
            WHERE store_root = ?
            ORDER BY
                created_at_utc,
                analysis_id
            """,
            (
                scope,
            ),
        ).fetchall()


    return [
        {
            "analysis_id":
                str(
                    row[
                        "analysis_id"
                    ]
                ),

            "workflow_id":
                str(
                    row[
                        "workflow_id"
                    ]
                ),

            "trace_id":
                str(
                    row[
                        "trace_id"
                    ]
                ),

            "source_type":
                str(
                    row[
                        "source_type"
                    ]
                ),

            "objective":
                str(
                    row[
                        "objective"
                    ]
                ),

            "executed":
                bool(
                    int(
                        row[
                            "executed"
                        ]
                    )
                ),

            "executed_count":
                int(
                    row[
                        "executed_count"
                    ]
                ),

            "created_at_utc":
                str(
                    row[
                        "created_at_utc"
                    ]
                ),

            "rule_version":
                str(
                    row[
                        "rule_version"
                    ]
                ),

            "payload_path":
                str(
                    row[
                        "payload_path"
                    ]
                ),

            "payload_json_bytes":
                int(
                    row[
                        "payload_json_bytes"
                    ]
                ),

            "payload_file_bytes":
                int(
                    row[
                        "payload_file_bytes"
                    ]
                ),

            "payload_sha256":
                str(
                    row[
                        "payload_sha256"
                    ]
                ),
        }

        for row
        in rows
    ]


def get_analysis_artifact_index_entry(
    *,
    store_path: Path,
    analysis_id: str,
) -> (
    dict[
        str,
        Any,
    ]
    |
    None
):
    normalized_analysis_id = str(
        analysis_id
    ).strip()


    if not normalized_analysis_id:
        raise AnalysisArtifactIndexError(
            "analysis_id cannot be empty."
        )


    scope = (
        analysis_artifact_store_scope(
            store_path
        )
    )


    with sqlite_connection(
        write=False
    ) as connection:
        row = connection.execute(
            """
            SELECT
                analysis_id,
                workflow_id,
                trace_id,
                source_type,
                objective,
                executed,
                executed_count,
                created_at_utc,
                rule_version,
                payload_path,
                payload_json_bytes,
                payload_file_bytes,
                payload_sha256
            FROM analysis_artifacts
            WHERE
                store_root = ?
                AND
                analysis_id = ?
            """,
            (
                scope,
                normalized_analysis_id,
            ),
        ).fetchone()


    if row is None:
        return None


    return {
        "analysis_id":
            str(
                row[
                    "analysis_id"
                ]
            ),

        "workflow_id":
            str(
                row[
                    "workflow_id"
                ]
            ),

        "trace_id":
            str(
                row[
                    "trace_id"
                ]
            ),

        "source_type":
            str(
                row[
                    "source_type"
                ]
            ),

        "objective":
            str(
                row[
                    "objective"
                ]
            ),

        "executed":
            bool(
                int(
                    row[
                        "executed"
                    ]
                )
            ),

        "executed_count":
            int(
                row[
                    "executed_count"
                ]
            ),

        "created_at_utc":
            str(
                row[
                    "created_at_utc"
                ]
            ),

        "rule_version":
            str(
                row[
                    "rule_version"
                ]
            ),

        "payload_path":
            str(
                row[
                    "payload_path"
                ]
            ),

        "payload_json_bytes":
            int(
                row[
                    "payload_json_bytes"
                ]
            ),

        "payload_file_bytes":
            int(
                row[
                    "payload_file_bytes"
                ]
            ),

        "payload_sha256":
            str(
                row[
                    "payload_sha256"
                ]
            ),
    }


# ========================================================
# STATE
# ========================================================


def load_analysis_artifact_store_state(
    *,
    store_path: Path,
) -> (
    dict[
        str,
        Any,
    ]
    |
    None
):
    scope = (
        analysis_artifact_store_scope(
            store_path
        )
    )


    with sqlite_connection(
        write=False
    ) as connection:
        row = connection.execute(
            """
            SELECT
                legacy_json_imported,
                legacy_rule_version,
                migrated_at
            FROM analysis_artifact_store_state
            WHERE store_root = ?
            """,
            (
                scope,
            ),
        ).fetchone()


    if row is None:
        return None


    return {
        "legacy_json_imported":
            bool(
                int(
                    row[
                        "legacy_json_imported"
                    ]
                )
            ),

        "legacy_rule_version":
            (
                None
                if row[
                    "legacy_rule_version"
                ]
                is None
                else
                str(
                    row[
                        "legacy_rule_version"
                    ]
                )
            ),

        "migrated_at":
            str(
                row[
                    "migrated_at"
                ]
            ),
    }


def analysis_artifact_index_is_initialized(
    *,
    store_path: Path,
) -> bool:
    return (
        load_analysis_artifact_store_state(
            store_path=
                store_path
        )
        is not None
    )


# ========================================================
# DELETE SCOPE
# ========================================================


def delete_analysis_artifact_index_scope(
    *,
    store_path: Path,
) -> None:
    scope = (
        analysis_artifact_store_scope(
            store_path
        )
    )


    with sqlite_connection(
        write=True
    ) as connection:
        connection.execute(
            """
            DELETE FROM analysis_artifacts
            WHERE store_root = ?
            """,
            (
                scope,
            ),
        )


        connection.execute(
            """
            DELETE FROM analysis_artifact_store_state
            WHERE store_root = ?
            """,
            (
                scope,
            ),
        )


# ========================================================
# ANALYSIS_ARTIFACT_INDEX_POINT_OPERATIONS_V0_1
# ========================================================


def upsert_analysis_artifact_index_entry(
    *,
    store_path: Path,
    entry: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    """
    Insert or refresh one logical AnalysisArtifact.

    analysis_id is the identity. workflow_id ownership
    cannot change after first registration.
    """

    validated = (
        validate_analysis_artifact_index_entry(
            entry
        )
    )

    scope = (
        analysis_artifact_store_scope(
            store_path
        )
    )


    with sqlite_connection(
        write=True
    ) as connection:
        existing = connection.execute(
            """
            SELECT workflow_id
            FROM analysis_artifacts
            WHERE
                store_root = ?
                AND
                analysis_id = ?
            """,
            (
                scope,
                validated[
                    "analysis_id"
                ],
            ),
        ).fetchone()


        if (
            existing is not None
            and
            str(
                existing[
                    "workflow_id"
                ]
            )
            !=
            validated[
                "workflow_id"
            ]
        ):
            raise AnalysisArtifactIndexError(
                (
                    "AnalysisArtifact workflow ownership "
                    "cannot change for an existing "
                    "analysis_id."
                )
            )


        connection.execute(
            """
            INSERT INTO analysis_artifacts (
                store_root,
                analysis_id,
                workflow_id,
                trace_id,
                source_type,
                objective,
                executed,
                executed_count,
                created_at_utc,
                rule_version,
                payload_path,
                payload_json_bytes,
                payload_file_bytes,
                payload_sha256
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(
                store_root,
                analysis_id
            )
            DO UPDATE SET
                workflow_id =
                    excluded.workflow_id,
                trace_id =
                    excluded.trace_id,
                source_type =
                    excluded.source_type,
                objective =
                    excluded.objective,
                executed =
                    excluded.executed,
                executed_count =
                    excluded.executed_count,
                created_at_utc =
                    excluded.created_at_utc,
                rule_version =
                    excluded.rule_version,
                payload_path =
                    excluded.payload_path,
                payload_json_bytes =
                    excluded.payload_json_bytes,
                payload_file_bytes =
                    excluded.payload_file_bytes,
                payload_sha256 =
                    excluded.payload_sha256
            """,
            (
                scope,
                validated[
                    "analysis_id"
                ],
                validated[
                    "workflow_id"
                ],
                validated[
                    "trace_id"
                ],
                validated[
                    "source_type"
                ],
                validated[
                    "objective"
                ],
                (
                    1
                    if validated[
                        "executed"
                    ]
                    else
                    0
                ),
                validated[
                    "executed_count"
                ],
                validated[
                    "created_at_utc"
                ],
                validated[
                    "rule_version"
                ],
                validated[
                    "payload_path"
                ],
                validated[
                    "payload_json_bytes"
                ],
                validated[
                    "payload_file_bytes"
                ],
                validated[
                    "payload_sha256"
                ],
            ),
        )


    return dict(
        validated
    )




def upsert_analysis_artifact_index_entries_atomic(
    *,
    store_path: Path,
    entries: list[
        dict[
            str,
            Any,
        ]
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """
    Atomically insert or refresh multiple AnalysisArtifact
    metadata rows.

    All ownership checks and all upserts occur inside one
    SQLite write transaction.

    If any entry fails validation or workflow ownership,
    no metadata row from this batch is committed.
    """

    validated = [
        validate_analysis_artifact_index_entry(
            entry
        )

        for entry
        in entries
    ]


    analysis_ids = [
        entry[
            "analysis_id"
        ]

        for entry
        in validated
    ]


    if (
        len(
            analysis_ids
        )
        !=
        len(
            set(
                analysis_ids
            )
        )
    ):
        raise AnalysisArtifactIndexError(
            (
                "Atomic AnalysisArtifact batch contains "
                "duplicate analysis_id values."
            )
        )


    if not validated:
        return []


    scope = (
        analysis_artifact_store_scope(
            store_path
        )
    )


    with sqlite_connection(
        write=True
    ) as connection:

        # ----------------------------------------------------
        # Ownership preflight
        # ----------------------------------------------------

        for entry in validated:
            existing = connection.execute(
                """
                SELECT workflow_id
                FROM analysis_artifacts
                WHERE
                    store_root = ?
                    AND
                    analysis_id = ?
                """,
                (
                    scope,
                    entry[
                        "analysis_id"
                    ],
                ),
            ).fetchone()


            if (
                existing is not None
                and
                str(
                    existing[
                        "workflow_id"
                    ]
                )
                !=
                entry[
                    "workflow_id"
                ]
            ):
                raise AnalysisArtifactIndexError(
                    (
                        "AnalysisArtifact workflow ownership "
                        "cannot change for an existing "
                        "analysis_id."
                    )
                )


        # ----------------------------------------------------
        # Atomic multi-row upsert
        # ----------------------------------------------------

        for entry in validated:
            connection.execute(
                """
                INSERT INTO analysis_artifacts (
                    store_root,
                    analysis_id,
                    workflow_id,
                    trace_id,
                    source_type,
                    objective,
                    executed,
                    executed_count,
                    created_at_utc,
                    rule_version,
                    payload_path,
                    payload_json_bytes,
                    payload_file_bytes,
                    payload_sha256
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(
                    store_root,
                    analysis_id
                )
                DO UPDATE SET
                    workflow_id =
                        excluded.workflow_id,
                    trace_id =
                        excluded.trace_id,
                    source_type =
                        excluded.source_type,
                    objective =
                        excluded.objective,
                    executed =
                        excluded.executed,
                    executed_count =
                        excluded.executed_count,
                    created_at_utc =
                        excluded.created_at_utc,
                    rule_version =
                        excluded.rule_version,
                    payload_path =
                        excluded.payload_path,
                    payload_json_bytes =
                        excluded.payload_json_bytes,
                    payload_file_bytes =
                        excluded.payload_file_bytes,
                    payload_sha256 =
                        excluded.payload_sha256
                """,
                (
                    scope,
                    entry[
                        "analysis_id"
                    ],
                    entry[
                        "workflow_id"
                    ],
                    entry[
                        "trace_id"
                    ],
                    entry[
                        "source_type"
                    ],
                    entry[
                        "objective"
                    ],
                    (
                        1
                        if entry[
                            "executed"
                        ]
                        else
                        0
                    ),
                    entry[
                        "executed_count"
                    ],
                    entry[
                        "created_at_utc"
                    ],
                    entry[
                        "rule_version"
                    ],
                    entry[
                        "payload_path"
                    ],
                    entry[
                        "payload_json_bytes"
                    ],
                    entry[
                        "payload_file_bytes"
                    ],
                    entry[
                        "payload_sha256"
                    ],
                ),
            )


    return [
        dict(
            entry
        )

        for entry
        in validated
    ]


def load_analysis_artifact_index_workflow(
    *,
    store_path: Path,
    workflow_id: str,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    normalized_workflow_id = str(
        workflow_id
    ).strip()


    if not normalized_workflow_id:
        raise AnalysisArtifactIndexError(
            "workflow_id cannot be empty."
        )


    scope = (
        analysis_artifact_store_scope(
            store_path
        )
    )


    with sqlite_connection(
        write=False
    ) as connection:
        rows = connection.execute(
            """
            SELECT
                analysis_id,
                workflow_id,
                trace_id,
                source_type,
                objective,
                executed,
                executed_count,
                created_at_utc,
                rule_version,
                payload_path,
                payload_json_bytes,
                payload_file_bytes,
                payload_sha256
            FROM analysis_artifacts
            WHERE
                store_root = ?
                AND
                workflow_id = ?
            ORDER BY
                created_at_utc,
                analysis_id
            """,
            (
                scope,
                normalized_workflow_id,
            ),
        ).fetchall()


    return [
        {
            "analysis_id":
                str(
                    row[
                        "analysis_id"
                    ]
                ),

            "workflow_id":
                str(
                    row[
                        "workflow_id"
                    ]
                ),

            "trace_id":
                str(
                    row[
                        "trace_id"
                    ]
                ),

            "source_type":
                str(
                    row[
                        "source_type"
                    ]
                ),

            "objective":
                str(
                    row[
                        "objective"
                    ]
                ),

            "executed":
                bool(
                    int(
                        row[
                            "executed"
                        ]
                    )
                ),

            "executed_count":
                int(
                    row[
                        "executed_count"
                    ]
                ),

            "created_at_utc":
                str(
                    row[
                        "created_at_utc"
                    ]
                ),

            "rule_version":
                str(
                    row[
                        "rule_version"
                    ]
                ),

            "payload_path":
                str(
                    row[
                        "payload_path"
                    ]
                ),

            "payload_json_bytes":
                int(
                    row[
                        "payload_json_bytes"
                    ]
                ),

            "payload_file_bytes":
                int(
                    row[
                        "payload_file_bytes"
                    ]
                ),

            "payload_sha256":
                str(
                    row[
                        "payload_sha256"
                    ]
                ),
        }

        for row
        in rows
    ]


def delete_analysis_artifact_index_workflow(
    *,
    store_path: Path,
    workflow_id: str,
) -> None:
    normalized_workflow_id = str(
        workflow_id
    ).strip()


    if not normalized_workflow_id:
        raise AnalysisArtifactIndexError(
            "workflow_id cannot be empty."
        )


    scope = (
        analysis_artifact_store_scope(
            store_path
        )
    )


    with sqlite_connection(
        write=True
    ) as connection:
        connection.execute(
            """
            DELETE FROM analysis_artifacts
            WHERE
                store_root = ?
                AND
                workflow_id = ?
            """,
            (
                scope,
                normalized_workflow_id,
            ),
        )
