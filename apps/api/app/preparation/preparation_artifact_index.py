from __future__ import annotations

import json

from pathlib import (
    Path,
)

from threading import (
    RLock,
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


PREPARATION_ARTIFACT_INDEX_VERSION = (
    "preparation_artifact_sqlite_index_v0.1"
)


_MIGRATION_LOCK = RLock()


# ========================================================
# ERRORS
# ========================================================


class PreparationArtifactIndexError(
    RuntimeError,
):
    pass


# ========================================================
# SCOPE
# ========================================================


def preparation_artifact_store_scope(
    root: Path,
) -> str:
    return str(
        root
        .expanduser()
        .resolve()
    )


# ========================================================
# JSON
# ========================================================


def _encode_json(
    value: object,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
        sort_keys=True,
    )


def _decode_json_list(
    raw: str,
    *,
    field_name: str,
) -> list[
    Any
]:
    try:
        value = json.loads(
            raw
        )

    except Exception as error:
        raise PreparationArtifactIndexError(
            (
                "Preparation artifact SQLite "
                f"{field_name} is invalid JSON."
            )
        ) from error


    if not isinstance(
        value,
        list,
    ):
        raise PreparationArtifactIndexError(
            (
                "Preparation artifact SQLite "
                f"{field_name} must be a JSON list."
            )
        )


    return value


# ========================================================
# ENTRY VALIDATION
# ========================================================


def _validated_entry(
    *,
    workflow_id: str,
    dataset_id: str,
    raw: object,
) -> dict[
    str,
    object,
]:
    if not isinstance(
        raw,
        dict,
    ):
        raise PreparationArtifactIndexError(
            (
                "Preparation artifact metadata "
                "entry must be an object."
            )
        )


    entry = dict(
        raw
    )


    required_text_fields = [
        "workflow_id",
        "dataset_id",
        "dataset_filename",
        "stage",
        "data_path",
    ]


    for field_name in required_text_fields:
        value = entry.get(
            field_name
        )

        if (
            not isinstance(
                value,
                str,
            )
            or
            not value.strip()
        ):
            raise PreparationArtifactIndexError(
                (
                    "Preparation artifact metadata "
                    "contains an invalid "
                    f"{field_name}."
                )
            )


    if (
        entry[
            "workflow_id"
        ]
        !=
        workflow_id
    ):
        raise PreparationArtifactIndexError(
            (
                "Preparation artifact workflow "
                "identity does not match its index key."
            )
        )


    if (
        entry[
            "dataset_id"
        ]
        !=
        dataset_id
    ):
        raise PreparationArtifactIndexError(
            (
                "Preparation artifact dataset "
                "identity does not match its index key."
            )
        )


    if (
        entry[
            "stage"
        ]
        not in {
            "source",
            "clean",
            "transform",
            "combine",
        }
    ):
        raise PreparationArtifactIndexError(
            (
                "Preparation artifact contains "
                "an unsupported stage."
            )
        )


    for field_name in [
        "rows",
        "columns",
    ]:
        try:
            value = int(
                entry[
                    field_name
                ]
            )

        except Exception as error:
            raise PreparationArtifactIndexError(
                (
                    "Preparation artifact contains "
                    f"an invalid {field_name}."
                )
            ) from error


        if value <= 0:
            raise PreparationArtifactIndexError(
                (
                    "Preparation artifact "
                    f"{field_name} must be positive."
                )
            )


        entry[
            field_name
        ] = value


    for field_name in [
        "parent_dataset_ids",
        "evidence_refs",
        "datetime_dtypes",
    ]:
        value = entry.get(
            field_name,
            [],
        )

        if not isinstance(
            value,
            list,
        ):
            raise PreparationArtifactIndexError(
                (
                    "Preparation artifact "
                    f"{field_name} must be a list."
                )
            )


        entry[
            field_name
        ] = value


    return entry


# ========================================================
# REPLACE COMPLETE SCOPE
# ========================================================


def replace_preparation_artifact_index(
    *,
    root: Path,
    manifest: dict,
    legacy_manifest_imported: bool = True,
) -> None:
    workflows = manifest.get(
        "workflows"
    )


    if not isinstance(
        workflows,
        dict,
    ):
        raise PreparationArtifactIndexError(
            (
                "Preparation artifact manifest "
                "must contain a workflows object."
            )
        )


    manifest_version = manifest.get(
        "manifest_version"
    )


    if (
        manifest_version is not None
        and
        not isinstance(
            manifest_version,
            str,
        )
    ):
        raise PreparationArtifactIndexError(
            (
                "Preparation artifact manifest_version "
                "must be text when present."
            )
        )


    scope = (
        preparation_artifact_store_scope(
            root
        )
    )


    entries: list[
        dict[
            str,
            object,
        ]
    ] = []


    for (
        workflow_id,
        workflow_entries,
    ) in workflows.items():
        if (
            not isinstance(
                workflow_id,
                str,
            )
            or
            not workflow_id.strip()
        ):
            raise PreparationArtifactIndexError(
                (
                    "Preparation artifact workflow "
                    "key is invalid."
                )
            )


        if not isinstance(
            workflow_entries,
            dict,
        ):
            raise PreparationArtifactIndexError(
                (
                    "Preparation artifact workflow "
                    "entry must be an object."
                )
            )


        for (
            dataset_id,
            raw_entry,
        ) in workflow_entries.items():
            if (
                not isinstance(
                    dataset_id,
                    str,
                )
                or
                not dataset_id.strip()
            ):
                raise PreparationArtifactIndexError(
                    (
                        "Preparation artifact dataset "
                        "key is invalid."
                    )
                )


            entries.append(
                _validated_entry(
                    workflow_id=
                        workflow_id,

                    dataset_id=
                        dataset_id,

                    raw=
                        raw_entry,
                )
            )


    with sqlite_connection(
        write=True
    ) as connection:
        connection.execute(
            """
            DELETE FROM preparation_artifacts
            WHERE store_root = ?
            """,
            (
                scope,
            ),
        )


        for entry in entries:
            connection.execute(
                """
                INSERT INTO preparation_artifacts (
                    store_root,
                    workflow_id,
                    dataset_id,
                    dataset_filename,
                    stage,
                    rows,
                    columns,
                    parent_dataset_ids_json,
                    evidence_refs_json,
                    datetime_dtypes_json,
                    data_path
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
                    ?
                )
                """,
                (
                    scope,
                    entry[
                        "workflow_id"
                    ],
                    entry[
                        "dataset_id"
                    ],
                    entry[
                        "dataset_filename"
                    ],
                    entry[
                        "stage"
                    ],
                    entry[
                        "rows"
                    ],
                    entry[
                        "columns"
                    ],
                    _encode_json(
                        entry[
                            "parent_dataset_ids"
                        ]
                    ),
                    _encode_json(
                        entry[
                            "evidence_refs"
                        ]
                    ),
                    _encode_json(
                        entry[
                            "datetime_dtypes"
                        ]
                    ),
                    entry[
                        "data_path"
                    ],
                ),
            )


        connection.execute(
            """
            INSERT INTO preparation_artifact_store_state (
                store_root,
                legacy_manifest_imported,
                legacy_manifest_version,
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
                legacy_manifest_imported =
                    excluded.legacy_manifest_imported,
                legacy_manifest_version =
                    excluded.legacy_manifest_version,
                migrated_at =
                    excluded.migrated_at
            """,
            (
                scope,
                (
                    1
                    if legacy_manifest_imported
                    else
                    0
                ),
                manifest_version,
                utc_now_iso(),
            ),
        )


# ========================================================
# READ COMPLETE SCOPE
# ========================================================


def load_preparation_artifact_index(
    *,
    root: Path,
    manifest_version: str,
) -> dict:
    scope = (
        preparation_artifact_store_scope(
            root
        )
    )


    with sqlite_connection(
        write=False
    ) as connection:
        rows = connection.execute(
            """
            SELECT
                workflow_id,
                dataset_id,
                dataset_filename,
                stage,
                rows,
                columns,
                parent_dataset_ids_json,
                evidence_refs_json,
                datetime_dtypes_json,
                data_path
            FROM preparation_artifacts
            WHERE store_root = ?
            ORDER BY
                workflow_id,
                dataset_id
            """,
            (
                scope,
            ),
        ).fetchall()


    workflows: dict[
        str,
        dict[
            str,
            dict[
                str,
                object,
            ]
        ]
    ] = {}


    for row in rows:
        workflow_id = str(
            row[
                "workflow_id"
            ]
        )

        dataset_id = str(
            row[
                "dataset_id"
            ]
        )


        workflows.setdefault(
            workflow_id,
            {},
        )[
            dataset_id
        ] = {
            "workflow_id":
                workflow_id,

            "dataset_id":
                dataset_id,

            "dataset_filename":
                str(
                    row[
                        "dataset_filename"
                    ]
                ),

            "stage":
                str(
                    row[
                        "stage"
                    ]
                ),

            "rows":
                int(
                    row[
                        "rows"
                    ]
                ),

            "columns":
                int(
                    row[
                        "columns"
                    ]
                ),

            "parent_dataset_ids":
                _decode_json_list(
                    str(
                        row[
                            "parent_dataset_ids_json"
                        ]
                    ),
                    field_name=
                        "parent_dataset_ids_json",
                ),

            "evidence_refs":
                _decode_json_list(
                    str(
                        row[
                            "evidence_refs_json"
                        ]
                    ),
                    field_name=
                        "evidence_refs_json",
                ),

            "datetime_dtypes":
                _decode_json_list(
                    str(
                        row[
                            "datetime_dtypes_json"
                        ]
                    ),
                    field_name=
                        "datetime_dtypes_json",
                ),

            "data_path":
                str(
                    row[
                        "data_path"
                    ]
                ),
        }


    return {
        "manifest_version":
            manifest_version,

        "workflows":
            workflows,
    }


# ========================================================
# MIGRATION STATE
# ========================================================


def preparation_artifact_index_is_initialized(
    *,
    root: Path,
) -> bool:
    scope = (
        preparation_artifact_store_scope(
            root
        )
    )


    with sqlite_connection(
        write=False
    ) as connection:
        row = connection.execute(
            """
            SELECT
                legacy_manifest_imported
            FROM preparation_artifact_store_state
            WHERE store_root = ?
            """,
            (
                scope,
            ),
        ).fetchone()


    return (
        row is not None
    )


# ========================================================
# LEGACY MANIFEST IMPORT
# ========================================================


def import_legacy_preparation_artifact_manifest_if_needed(
    *,
    root: Path,
    manifest_path: Path,
    fallback_manifest_version: str,
) -> bool:
    """
    Import legacy manifest.json exactly once per store root.

    Returns True only when this call performs initialization.
    """

    with _MIGRATION_LOCK:
        if (
            preparation_artifact_index_is_initialized(
                root=
                    root
            )
        ):
            return False


        if manifest_path.exists():
            try:
                manifest = json.loads(
                    manifest_path.read_text(
                        encoding="utf-8"
                    )
                )

            except Exception as error:
                raise PreparationArtifactIndexError(
                    (
                        "Legacy Preparation artifact "
                        "manifest could not be read."
                    )
                ) from error

        else:
            manifest = {
                "manifest_version":
                    fallback_manifest_version,

                "workflows":
                    {},
            }


        if not isinstance(
            manifest,
            dict,
        ):
            raise PreparationArtifactIndexError(
                (
                    "Legacy Preparation artifact "
                    "manifest root must be an object."
                )
            )


        manifest.setdefault(
            "manifest_version",
            fallback_manifest_version,
        )


        replace_preparation_artifact_index(
            root=
                root,

            manifest=
                manifest,

            legacy_manifest_imported=
                True,
        )


        return True


# ========================================================
# TEST / FUTURE STORE RESET
# ========================================================


def delete_preparation_artifact_index_scope(
    *,
    root: Path,
) -> None:
    scope = (
        preparation_artifact_store_scope(
            root
        )
    )


    with sqlite_connection(
        write=True
    ) as connection:
        connection.execute(
            """
            DELETE FROM preparation_artifacts
            WHERE store_root = ?
            """,
            (
                scope,
            ),
        )


        connection.execute(
            """
            DELETE FROM preparation_artifact_store_state
            WHERE store_root = ?
            """,
            (
                scope,
            ),
        )
