from __future__ import annotations


import json
import os
import sqlite3


from pathlib import Path


from typing import Literal


from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    model_validator,
)


from app.model_lifecycle.artifact_contracts import (
    ModelLifecycleArtifactRecord,
)

from app.model_lifecycle.provenance_contracts import (
    ModelLifecycleProvenanceRecord,
)


# ============================================================
# VERSION / CONFIGURATION
# ============================================================


MODEL_LIFECYCLE_REGISTRY_RULE_VERSION = (
    "model_lifecycle_registry_v0.1"
)


MODEL_LIFECYCLE_REGISTRY_ENV = (
    "DATALENS_MODEL_LIFECYCLE_REGISTRY_PATH"
)


REGISTRY_TABLE = (
    "model_lifecycle_registry"
)


# ============================================================
# ERRORS
# ============================================================


class ModelLifecycleRegistryError(
    RuntimeError
):
    pass


class ModelLifecycleRegistryConflictError(
    ModelLifecycleRegistryError
):
    pass


class ModelLifecycleRegistryNotFoundError(
    ModelLifecycleRegistryError
):
    pass


class ModelLifecycleRegistryCorruptionError(
    ModelLifecycleRegistryError
):
    pass


# ============================================================
# ENTRY CONTRACT
# ============================================================


class ModelLifecycleRegistryEntry(
    BaseModel
):
    """
    Metadata-only registry entry.

    The registry owns this metadata record. It does not own,
    copy, mutate or delete the external model/adapter bytes.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    artifact: (
        ModelLifecycleArtifactRecord
    )


    provenance: (
        ModelLifecycleProvenanceRecord
    )


    rule_version: Literal[
        "model_lifecycle_registry_v0.1"
    ] = (
        MODEL_LIFECYCLE_REGISTRY_RULE_VERSION
    )


    @model_validator(
        mode="after",
    )
    def validate_identity_binding(
        self,
    ) -> "ModelLifecycleRegistryEntry":

        artifact = (
            self.artifact
        )

        provenance = (
            self.provenance
        )


        if (
            provenance.artifact_id
            !=
            artifact.artifact_id
        ):
            raise ValueError(
                (
                    "Registry artifact/provenance "
                    "artifact_id mismatch."
                )
            )


        if (
            provenance.artifact_family
            !=
            artifact.artifact_family
        ):
            raise ValueError(
                (
                    "Registry artifact/provenance "
                    "artifact_family mismatch."
                )
            )


        if (
            provenance.experiment_id
            !=
            artifact.experiment_id
        ):
            raise ValueError(
                (
                    "Registry artifact/provenance "
                    "experiment_id mismatch."
                )
            )


        if (
            provenance.source_contract_kind
            !=
            artifact.source_contract_kind
        ):
            raise ValueError(
                (
                    "Registry artifact/provenance "
                    "source_contract_kind mismatch."
                )
            )


        if (
            provenance.source_contract_sha256
            !=
            artifact.source_contract_sha256
        ):
            raise ValueError(
                (
                    "Registry artifact/provenance "
                    "source_contract_sha256 mismatch."
                )
            )


        return self


# ============================================================
# PATH AUTHORITY
# ============================================================


def _api_root(
) -> Path:

    return (
        Path(
            __file__
        )
        .resolve()
        .parents[
            2
        ]
    )


def resolve_model_lifecycle_registry_path(
    registry_path: Path | str | None = None,
) -> Path:

    if registry_path is not None:

        raw = str(
            registry_path
        ).strip()


        if not raw:
            raise ModelLifecycleRegistryError(
                "registry_path cannot be empty."
            )


        return Path(
            raw
        ).expanduser()


    env_value = (
        os.getenv(
            MODEL_LIFECYCLE_REGISTRY_ENV
        )
    )


    if env_value is not None:

        normalized = (
            env_value.strip()
        )


        if not normalized:
            raise ModelLifecycleRegistryError(
                (
                    "Configured lifecycle registry "
                    "path cannot be empty."
                )
            )


        return Path(
            normalized
        ).expanduser()


    return (
        _api_root()
        /
        "var"
        /
        "model_lifecycle"
        /
        "registry.sqlite3"
    )


# ============================================================
# CANONICAL SERIALIZATION
# ============================================================


def _canonical_entry_json(
    entry: ModelLifecycleRegistryEntry,
) -> str:

    try:
        return json.dumps(
            entry.model_dump(
                mode="json"
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            allow_nan=False,
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise ModelLifecycleRegistryError(
            (
                "Lifecycle registry entry could "
                "not be canonically serialized."
            )
        ) from error


# ============================================================
# DATABASE
# ============================================================


EXPECTED_COLUMNS = (
    "artifact_id",
    "provenance_id",
    "experiment_id",
    "artifact_family",
    "producer_family",
    "entry_json",
    "rule_version",
)


CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {REGISTRY_TABLE} (
    artifact_id TEXT PRIMARY KEY,
    provenance_id TEXT NOT NULL UNIQUE,
    experiment_id TEXT NOT NULL,
    artifact_family TEXT NOT NULL,
    producer_family TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    rule_version TEXT NOT NULL
)
"""


def _connect(
    path: Path,
) -> sqlite3.Connection:

    try:
        connection = sqlite3.connect(
            str(
                path
            ),
            timeout=5.0,
        )

    except sqlite3.Error as error:
        raise ModelLifecycleRegistryError(
            (
                "Could not open model lifecycle "
                f"registry: {path}"
            )
        ) from error


    connection.row_factory = (
        sqlite3.Row
    )


    return connection


def _ensure_schema(
    connection: sqlite3.Connection,
) -> None:

    try:
        connection.execute(
            CREATE_TABLE_SQL
        )

        rows = connection.execute(
            (
                "PRAGMA table_info("
                f"{REGISTRY_TABLE}"
                ")"
            )
        ).fetchall()

    except sqlite3.Error as error:
        raise ModelLifecycleRegistryError(
            "Could not initialize lifecycle registry."
        ) from error


    actual_columns = tuple(
        row[
            "name"
        ]
        for row
        in rows
    )


    if (
        actual_columns
        !=
        EXPECTED_COLUMNS
    ):
        raise ModelLifecycleRegistryCorruptionError(
            (
                "Lifecycle registry schema "
                "does not match v0.1 authority."
            )
        )


# ============================================================
# ROW DECODING
# ============================================================


def _entry_from_row(
    row: sqlite3.Row,
) -> ModelLifecycleRegistryEntry:

    try:
        payload = json.loads(
            row[
                "entry_json"
            ]
        )


        entry = (
            ModelLifecycleRegistryEntry
            .model_validate(
                payload
            )
        )

    except (
        json.JSONDecodeError,
        ValidationError,
        TypeError,
        ValueError,
    ) as error:
        raise ModelLifecycleRegistryCorruptionError(
            (
                "Lifecycle registry entry "
                "cannot be decoded."
            )
        ) from error


    expected_metadata = {
        "artifact_id":
            entry.artifact.artifact_id,

        "provenance_id":
            entry.provenance.provenance_id,

        "experiment_id":
            entry.artifact.experiment_id,

        "artifact_family":
            entry.artifact.artifact_family,

        "producer_family":
            entry.provenance.producer_family,

        "rule_version":
            entry.rule_version,
    }


    for key, expected in (
        expected_metadata.items()
    ):

        if (
            row[
                key
            ]
            !=
            expected
        ):
            raise ModelLifecycleRegistryCorruptionError(
                (
                    "Lifecycle registry indexed "
                    f"metadata mismatch: {key}"
                )
            )


    return entry


# ============================================================
# ENTRY CONSTRUCTION
# ============================================================


def _build_entry(
    *,
    artifact: ModelLifecycleArtifactRecord,
    provenance: ModelLifecycleProvenanceRecord,
) -> ModelLifecycleRegistryEntry:

    try:
        return (
            ModelLifecycleRegistryEntry(
                artifact=
                    artifact,

                provenance=
                    provenance,
            )
        )

    except ValidationError as error:
        raise ModelLifecycleRegistryError(
            (
                "Artifact and provenance cannot "
                "form one registry entry."
            )
        ) from error


# ============================================================
# REGISTER
# ============================================================


def register_model_lifecycle_entry(
    *,
    artifact: ModelLifecycleArtifactRecord,
    provenance: ModelLifecycleProvenanceRecord,
    registry_path: Path | str | None = None,
) -> ModelLifecycleRegistryEntry:
    """
    Register metadata for one externally-owned immutable artifact.

    No artifact bytes are copied, opened, modified or deleted.
    """

    entry = _build_entry(
        artifact=
            artifact,

        provenance=
            provenance,
    )


    path = (
        resolve_model_lifecycle_registry_path(
            registry_path
        )
    )


    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    except OSError as error:
        raise ModelLifecycleRegistryError(
            (
                "Could not create lifecycle "
                "registry directory."
            )
        ) from error


    connection = _connect(
        path
    )


    try:
        _ensure_schema(
            connection
        )

        connection.commit()


        connection.execute(
            "BEGIN IMMEDIATE"
        )


        existing_row = (
            connection.execute(
                f"""
                SELECT
                    artifact_id,
                    provenance_id,
                    experiment_id,
                    artifact_family,
                    producer_family,
                    entry_json,
                    rule_version
                FROM {REGISTRY_TABLE}
                WHERE artifact_id = ?
                """,
                (
                    entry.artifact.artifact_id,
                ),
            )
            .fetchone()
        )


        if existing_row is not None:

            existing_entry = (
                _entry_from_row(
                    existing_row
                )
            )


            if (
                existing_entry
                ==
                entry
            ):
                connection.commit()

                return existing_entry


            raise ModelLifecycleRegistryConflictError(
                (
                    "artifact_id already exists "
                    "with different metadata."
                )
            )


        provenance_row = (
            connection.execute(
                f"""
                SELECT artifact_id
                FROM {REGISTRY_TABLE}
                WHERE provenance_id = ?
                """,
                (
                    entry.provenance.provenance_id,
                ),
            )
            .fetchone()
        )


        if provenance_row is not None:
            raise ModelLifecycleRegistryConflictError(
                (
                    "provenance_id is already bound "
                    "to another artifact."
                )
            )


        entry_json = (
            _canonical_entry_json(
                entry
            )
        )


        try:
            connection.execute(
                f"""
                INSERT INTO {REGISTRY_TABLE} (
                    artifact_id,
                    provenance_id,
                    experiment_id,
                    artifact_family,
                    producer_family,
                    entry_json,
                    rule_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.artifact.artifact_id,
                    entry.provenance.provenance_id,
                    entry.artifact.experiment_id,
                    entry.artifact.artifact_family,
                    entry.provenance.producer_family,
                    entry_json,
                    entry.rule_version,
                ),
            )

        except sqlite3.IntegrityError as error:
            raise ModelLifecycleRegistryConflictError(
                (
                    "Lifecycle registry uniqueness "
                    "constraint rejected entry."
                )
            ) from error


        connection.commit()


        return entry


    except (
        ModelLifecycleRegistryError,
        sqlite3.Error,
    ) as error:

        connection.rollback()


        if isinstance(
            error,
            ModelLifecycleRegistryError,
        ):
            raise


        raise ModelLifecycleRegistryError(
            "Lifecycle registry write failed."
        ) from error


    finally:
        connection.close()


# ============================================================
# GET
# ============================================================


def get_model_lifecycle_entry(
    artifact_id: str,
    *,
    registry_path: Path | str | None = None,
) -> ModelLifecycleRegistryEntry:

    normalized_id = str(
        artifact_id
        if artifact_id is not None
        else ""
    ).strip()


    if not normalized_id:
        raise ModelLifecycleRegistryError(
            "artifact_id cannot be empty."
        )


    path = (
        resolve_model_lifecycle_registry_path(
            registry_path
        )
    )


    if not path.is_file():
        raise ModelLifecycleRegistryNotFoundError(
            (
                "Lifecycle artifact is not "
                f"registered: {normalized_id}"
            )
        )


    connection = _connect(
        path
    )


    try:
        _ensure_schema(
            connection
        )


        row = (
            connection.execute(
                f"""
                SELECT
                    artifact_id,
                    provenance_id,
                    experiment_id,
                    artifact_family,
                    producer_family,
                    entry_json,
                    rule_version
                FROM {REGISTRY_TABLE}
                WHERE artifact_id = ?
                """,
                (
                    normalized_id,
                ),
            )
            .fetchone()
        )


        if row is None:
            raise ModelLifecycleRegistryNotFoundError(
                (
                    "Lifecycle artifact is not "
                    f"registered: {normalized_id}"
                )
            )


        return _entry_from_row(
            row
        )


    except sqlite3.Error as error:
        raise ModelLifecycleRegistryError(
            "Lifecycle registry read failed."
        ) from error


    finally:
        connection.close()


# ============================================================
# LIST
# ============================================================


def list_model_lifecycle_entries(
    *,
    registry_path: Path | str | None = None,
) -> tuple[
    ModelLifecycleRegistryEntry,
    ...,
]:

    path = (
        resolve_model_lifecycle_registry_path(
            registry_path
        )
    )


    if not path.is_file():
        return ()


    connection = _connect(
        path
    )


    try:
        _ensure_schema(
            connection
        )


        rows = (
            connection.execute(
                f"""
                SELECT
                    artifact_id,
                    provenance_id,
                    experiment_id,
                    artifact_family,
                    producer_family,
                    entry_json,
                    rule_version
                FROM {REGISTRY_TABLE}
                ORDER BY artifact_id ASC
                """
            )
            .fetchall()
        )


        return tuple(
            _entry_from_row(
                row
            )
            for row
            in rows
        )


    except sqlite3.Error as error:
        raise ModelLifecycleRegistryError(
            "Lifecycle registry list failed."
        ) from error


    finally:
        connection.close()
