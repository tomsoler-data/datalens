from __future__ import annotations


import json
import os
import sqlite3


from pathlib import Path


from pydantic import ValidationError


from app.model_lifecycle.evaluation_decision_contracts import (
    ModelLifecycleEvaluationDecisionRecord,
)

from app.model_lifecycle.registry import (
    ModelLifecycleRegistryError,
    get_model_lifecycle_entry,
    resolve_model_lifecycle_registry_path,
)


# ============================================================
# VERSION / CONFIGURATION
# ============================================================


MODEL_LIFECYCLE_EVALUATION_DECISION_REGISTRY_RULE_VERSION = (
    "model_lifecycle_evaluation_decision_registry_v0.1"
)


MODEL_LIFECYCLE_EVALUATION_DECISION_REGISTRY_ENV = (
    "DATALENS_MODEL_LIFECYCLE_EVALUATION_DECISION_REGISTRY_PATH"
)


DECISION_REGISTRY_TABLE = (
    "model_lifecycle_evaluation_decisions"
)


# ============================================================
# ERRORS
# ============================================================


class ModelLifecycleEvaluationDecisionRegistryError(
    RuntimeError
):
    pass


class ModelLifecycleEvaluationDecisionRegistryConflictError(
    ModelLifecycleEvaluationDecisionRegistryError
):
    pass


class ModelLifecycleEvaluationDecisionRegistryNotFoundError(
    ModelLifecycleEvaluationDecisionRegistryError
):
    pass


class ModelLifecycleEvaluationDecisionRegistryCorruptionError(
    ModelLifecycleEvaluationDecisionRegistryError
):
    pass


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


def resolve_model_lifecycle_evaluation_decision_registry_path(
    registry_path: Path | str | None = None,
) -> Path:

    if registry_path is not None:

        raw = str(
            registry_path
        ).strip()


        if not raw:
            raise ModelLifecycleEvaluationDecisionRegistryError(
                "registry_path cannot be empty."
            )


        return Path(
            raw
        )


    environment_path = (
        os.getenv(
            MODEL_LIFECYCLE_EVALUATION_DECISION_REGISTRY_ENV
        )
    )


    if environment_path is not None:

        normalized = (
            environment_path.strip()
        )


        if not normalized:
            raise ModelLifecycleEvaluationDecisionRegistryError(
                (
                    "Decision registry environment "
                    "path cannot be empty."
                )
            )


        return Path(
            normalized
        )


    return (
        _api_root()
        /
        "var"
        /
        "model_lifecycle"
        /
        "evaluation_decisions.sqlite3"
    )


# ============================================================
# SQLITE
# ============================================================


def _connect(
    path: Path,
) -> sqlite3.Connection:

    try:
        connection = sqlite3.connect(
            path,
            timeout=10.0,
        )

    except sqlite3.Error as error:
        raise ModelLifecycleEvaluationDecisionRegistryError(
            "Could not open evaluation decision registry."
        ) from error


    return connection


def _validate_schema(
    connection: sqlite3.Connection,
) -> None:

    try:
        rows = (
            connection.execute(
                f"""
                PRAGMA table_info(
                    {DECISION_REGISTRY_TABLE}
                )
                """
            )
            .fetchall()
        )

    except sqlite3.Error as error:
        raise ModelLifecycleEvaluationDecisionRegistryCorruptionError(
            "Could not inspect decision registry schema."
        ) from error


    if not rows:
        raise ModelLifecycleEvaluationDecisionRegistryCorruptionError(
            "Decision registry table is missing."
        )


    actual_columns = tuple(
        row[
            1
        ]
        for row
        in rows
    )


    expected_columns = (
        "decision_id",
        "artifact_id",
        "provenance_id",
        "experiment_id",
        "evaluation_id",
        "evaluation_status",
        "promotion_eligible",
        "promotion_decision",
        "source_receipt_sha256",
        "decision_json",
        "rule_version",
    )


    if (
        actual_columns
        !=
        expected_columns
    ):
        raise ModelLifecycleEvaluationDecisionRegistryCorruptionError(
            (
                "Unexpected evaluation decision "
                "registry schema."
            )
        )


def _create_schema(
    connection: sqlite3.Connection,
) -> None:

    try:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {DECISION_REGISTRY_TABLE} (
                decision_id TEXT PRIMARY KEY,
                artifact_id TEXT NOT NULL,
                provenance_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                evaluation_id TEXT NOT NULL,
                evaluation_status TEXT NOT NULL,
                promotion_eligible INTEGER NOT NULL
                    CHECK (
                        promotion_eligible IN (0, 1)
                    ),
                promotion_decision TEXT NOT NULL,
                source_receipt_sha256 TEXT NOT NULL,
                decision_json TEXT NOT NULL,
                rule_version TEXT NOT NULL,
                UNIQUE (
                    artifact_id,
                    evaluation_id
                )
            )
            """
        )

    except sqlite3.Error as error:
        raise ModelLifecycleEvaluationDecisionRegistryError(
            "Could not create decision registry schema."
        ) from error


    _validate_schema(
        connection
    )


# ============================================================
# SERIALIZATION
# ============================================================


def _validated_decision(
    decision: ModelLifecycleEvaluationDecisionRecord,
) -> ModelLifecycleEvaluationDecisionRecord:

    try:
        return (
            ModelLifecycleEvaluationDecisionRecord.model_validate(
                decision.model_dump(
                    mode="python"
                )
            )
        )

    except (
        AttributeError,
        ValidationError,
    ) as error:
        raise ModelLifecycleEvaluationDecisionRegistryError(
            "Decision record is invalid."
        ) from error


def _canonical_decision_json(
    decision: ModelLifecycleEvaluationDecisionRecord,
) -> str:

    return json.dumps(
        decision.model_dump(
            mode="json"
        ),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )


def _decision_from_row(
    row,
) -> ModelLifecycleEvaluationDecisionRecord:

    if (
        not isinstance(
            row,
            tuple,
        )
        or
        len(
            row
        )
        !=
        11
    ):
        raise ModelLifecycleEvaluationDecisionRegistryCorruptionError(
            "Decision registry row shape is invalid."
        )


    (
        decision_id,
        artifact_id,
        provenance_id,
        experiment_id,
        evaluation_id,
        evaluation_status,
        promotion_eligible,
        promotion_decision,
        source_receipt_sha256,
        decision_json,
        rule_version,
    ) = row


    if (
        rule_version
        !=
        MODEL_LIFECYCLE_EVALUATION_DECISION_REGISTRY_RULE_VERSION
    ):
        raise ModelLifecycleEvaluationDecisionRegistryCorruptionError(
            "Decision registry rule version is invalid."
        )


    if (
        promotion_eligible
        not in
        (
            0,
            1,
        )
    ):
        raise ModelLifecycleEvaluationDecisionRegistryCorruptionError(
            "Decision registry promotion boolean is invalid."
        )


    try:
        decision = (
            ModelLifecycleEvaluationDecisionRecord.model_validate_json(
                decision_json
            )
        )

    except (
        ValidationError,
        ValueError,
        TypeError,
    ) as error:
        raise ModelLifecycleEvaluationDecisionRegistryCorruptionError(
            "Decision registry JSON is invalid."
        ) from error


    checks = (
        (
            decision.decision_id,
            decision_id,
        ),
        (
            decision.artifact_id,
            artifact_id,
        ),
        (
            decision.provenance_id,
            provenance_id,
        ),
        (
            decision.experiment_id,
            experiment_id,
        ),
        (
            decision.evaluation_id,
            evaluation_id,
        ),
        (
            decision.evaluation_status,
            evaluation_status,
        ),
        (
            int(
                decision.promotion_eligible
            ),
            promotion_eligible,
        ),
        (
            decision.promotion_decision,
            promotion_decision,
        ),
        (
            decision.source_receipt_sha256,
            source_receipt_sha256,
        ),
    )


    if any(
        actual
        !=
        expected
        for (
            actual,
            expected,
        )
        in checks
    ):
        raise ModelLifecycleEvaluationDecisionRegistryCorruptionError(
            (
                "Decision registry indexed metadata "
                "does not match decision JSON."
            )
        )


    return decision


# ============================================================
# IDENTITY BINDING
# ============================================================


def _verify_registered_identity(
    *,
    decision: ModelLifecycleEvaluationDecisionRecord,
    lifecycle_registry_path: Path | str | None,
) -> None:

    try:
        entry = get_model_lifecycle_entry(
            decision.artifact_id,
            registry_path=
                lifecycle_registry_path,
        )

    except ModelLifecycleRegistryError as error:
        raise ModelLifecycleEvaluationDecisionRegistryError(
            (
                "Decision artifact is not backed by "
                "a valid lifecycle registry entry."
            )
        ) from error


    if (
        entry.artifact.artifact_id
        !=
        decision.artifact_id
    ):
        raise ModelLifecycleEvaluationDecisionRegistryError(
            "Lifecycle artifact identity mismatch."
        )


    if (
        entry.provenance.provenance_id
        !=
        decision.provenance_id
    ):
        raise ModelLifecycleEvaluationDecisionRegistryError(
            "Lifecycle provenance identity mismatch."
        )


    if (
        entry.artifact.experiment_id
        !=
        decision.experiment_id
    ):
        raise ModelLifecycleEvaluationDecisionRegistryError(
            "Lifecycle experiment identity mismatch."
        )


# ============================================================
# REGISTER
# ============================================================


def register_model_lifecycle_evaluation_decision(
    *,
    decision: ModelLifecycleEvaluationDecisionRecord,
    lifecycle_registry_path: Path | str | None = None,
    decision_registry_path: Path | str | None = None,
) -> ModelLifecycleEvaluationDecisionRecord:
    """
    Persist one immutable evaluation decision.

    The identity/provenance registry remains a separate authority.
    The decision registry owns only decision metadata.

    Registration is idempotent for an identical record and
    fail-closed for conflicting decisions.
    """

    validated = (
        _validated_decision(
            decision
        )
    )


    lifecycle_path = (
        resolve_model_lifecycle_registry_path(
            lifecycle_registry_path
        )
    )


    decision_path = (
        resolve_model_lifecycle_evaluation_decision_registry_path(
            decision_registry_path
        )
    )


    if (
        lifecycle_path.resolve()
        ==
        decision_path.resolve()
    ):
        raise ModelLifecycleEvaluationDecisionRegistryError(
            (
                "Identity registry and decision registry "
                "must remain separate stores."
            )
        )


    _verify_registered_identity(
        decision=
            validated,

        lifecycle_registry_path=
            lifecycle_path,
    )


    try:
        decision_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    except OSError as error:
        raise ModelLifecycleEvaluationDecisionRegistryError(
            (
                "Could not create decision registry "
                "directory."
            )
        ) from error


    connection = _connect(
        decision_path
    )


    try:
        _create_schema(
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
                    decision_id,
                    artifact_id,
                    provenance_id,
                    experiment_id,
                    evaluation_id,
                    evaluation_status,
                    promotion_eligible,
                    promotion_decision,
                    source_receipt_sha256,
                    decision_json,
                    rule_version
                FROM {DECISION_REGISTRY_TABLE}
                WHERE decision_id = ?
                """,
                (
                    validated.decision_id,
                ),
            )
            .fetchone()
        )


        if existing_row is not None:

            existing = (
                _decision_from_row(
                    existing_row
                )
            )


            if (
                existing
                ==
                validated
            ):
                connection.commit()

                return existing


            raise ModelLifecycleEvaluationDecisionRegistryConflictError(
                (
                    "decision_id already exists "
                    "with different metadata."
                )
            )


        evaluation_row = (
            connection.execute(
                f"""
                SELECT
                    decision_id
                FROM {DECISION_REGISTRY_TABLE}
                WHERE artifact_id = ?
                  AND evaluation_id = ?
                """,
                (
                    validated.artifact_id,
                    validated.evaluation_id,
                ),
            )
            .fetchone()
        )


        if evaluation_row is not None:
            raise ModelLifecycleEvaluationDecisionRegistryConflictError(
                (
                    "Artifact/evaluation already has "
                    "an immutable lifecycle decision."
                )
            )


        decision_json = (
            _canonical_decision_json(
                validated
            )
        )


        try:
            connection.execute(
                f"""
                INSERT INTO {DECISION_REGISTRY_TABLE} (
                    decision_id,
                    artifact_id,
                    provenance_id,
                    experiment_id,
                    evaluation_id,
                    evaluation_status,
                    promotion_eligible,
                    promotion_decision,
                    source_receipt_sha256,
                    decision_json,
                    rule_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    validated.decision_id,
                    validated.artifact_id,
                    validated.provenance_id,
                    validated.experiment_id,
                    validated.evaluation_id,
                    validated.evaluation_status,
                    int(
                        validated.promotion_eligible
                    ),
                    validated.promotion_decision,
                    validated.source_receipt_sha256,
                    decision_json,
                    MODEL_LIFECYCLE_EVALUATION_DECISION_REGISTRY_RULE_VERSION,
                ),
            )

        except sqlite3.IntegrityError as error:
            raise ModelLifecycleEvaluationDecisionRegistryConflictError(
                (
                    "Decision registry uniqueness "
                    "constraint rejected record."
                )
            ) from error


        connection.commit()


        return validated


    except (
        ModelLifecycleEvaluationDecisionRegistryError,
        sqlite3.Error,
    ) as error:

        connection.rollback()


        if isinstance(
            error,
            ModelLifecycleEvaluationDecisionRegistryError,
        ):
            raise


        raise ModelLifecycleEvaluationDecisionRegistryError(
            "Decision registry write failed."
        ) from error


    finally:
        connection.close()


# ============================================================
# GET
# ============================================================


def get_model_lifecycle_evaluation_decision(
    decision_id: str,
    *,
    decision_registry_path: Path | str | None = None,
) -> ModelLifecycleEvaluationDecisionRecord:

    normalized_id = str(
        decision_id
        if decision_id is not None
        else ""
    ).strip()


    if not normalized_id:
        raise ModelLifecycleEvaluationDecisionRegistryError(
            "decision_id cannot be empty."
        )


    path = (
        resolve_model_lifecycle_evaluation_decision_registry_path(
            decision_registry_path
        )
    )


    if not path.is_file():
        raise ModelLifecycleEvaluationDecisionRegistryNotFoundError(
            (
                "Lifecycle evaluation decision "
                f"is not registered: {normalized_id}"
            )
        )


    connection = _connect(
        path
    )


    try:
        _validate_schema(
            connection
        )


        row = (
            connection.execute(
                f"""
                SELECT
                    decision_id,
                    artifact_id,
                    provenance_id,
                    experiment_id,
                    evaluation_id,
                    evaluation_status,
                    promotion_eligible,
                    promotion_decision,
                    source_receipt_sha256,
                    decision_json,
                    rule_version
                FROM {DECISION_REGISTRY_TABLE}
                WHERE decision_id = ?
                """,
                (
                    normalized_id,
                ),
            )
            .fetchone()
        )


        if row is None:
            raise ModelLifecycleEvaluationDecisionRegistryNotFoundError(
                (
                    "Lifecycle evaluation decision "
                    f"is not registered: {normalized_id}"
                )
            )


        return _decision_from_row(
            row
        )


    except sqlite3.Error as error:
        raise ModelLifecycleEvaluationDecisionRegistryError(
            "Decision registry read failed."
        ) from error


    finally:
        connection.close()


# ============================================================
# LIST
# ============================================================


def list_model_lifecycle_evaluation_decisions(
    *,
    decision_registry_path: Path | str | None = None,
) -> tuple[
    ModelLifecycleEvaluationDecisionRecord,
    ...,
]:

    path = (
        resolve_model_lifecycle_evaluation_decision_registry_path(
            decision_registry_path
        )
    )


    if not path.is_file():
        return ()


    connection = _connect(
        path
    )


    try:
        _validate_schema(
            connection
        )


        rows = (
            connection.execute(
                f"""
                SELECT
                    decision_id,
                    artifact_id,
                    provenance_id,
                    experiment_id,
                    evaluation_id,
                    evaluation_status,
                    promotion_eligible,
                    promotion_decision,
                    source_receipt_sha256,
                    decision_json,
                    rule_version
                FROM {DECISION_REGISTRY_TABLE}
                ORDER BY decision_id ASC
                """
            )
            .fetchall()
        )


        return tuple(
            _decision_from_row(
                row
            )
            for row
            in rows
        )


    except sqlite3.Error as error:
        raise ModelLifecycleEvaluationDecisionRegistryError(
            "Decision registry list failed."
        ) from error


    finally:
        connection.close()
