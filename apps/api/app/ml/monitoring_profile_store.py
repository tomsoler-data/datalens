from __future__ import annotations


import json


from typing import (
    Any,
)


from pydantic import (
    ValidationError,
)


from app.ml.experiment_provenance import (
    MLExperimentProvenanceRecord,
)


from app.ml.model_artifact_index import (
    ml_model_artifact_store_scope,
)


from app.ml.model_artifact_store import (
    resolve_ml_model_artifact_store_path,
)


from app.ml.monitoring_profile import (
    MLMonitoringProfileRecord,
)


from app.persistence.sqlite_database import (
    sqlite_connection,
)


# ============================================================
# VERSION
# ============================================================


ML_MONITORING_PROFILE_STORE_RULE_VERSION = (
    "ml_monitoring_profile_store_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLMonitoringProfileStoreError(
    RuntimeError
):
    pass


class MLMonitoringProfileNotFoundError(
    MLMonitoringProfileStoreError
):
    pass


class MLMonitoringProfileAlreadyExistsError(
    MLMonitoringProfileStoreError
):
    pass


class MLMonitoringProfileAuthorityError(
    MLMonitoringProfileStoreError
):
    pass


class MLMonitoringProfileWorkflowMismatchError(
    MLMonitoringProfileStoreError
):
    pass


# ============================================================
# TEXT
# ============================================================


def _required_text(
    value: object,
    *,
    field_name: str,
) -> str:

    normalized = str(
        value
        if value is not None
        else ""
    ).strip()


    if not normalized:
        raise (
            MLMonitoringProfileStoreError(
                (
                    f"{field_name} "
                    "cannot be empty."
                )
            )
        )


    return normalized


# ============================================================
# JSON
# ============================================================


def _canonical_json(
    value: object,
) -> str:

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )

    except Exception as error:
        raise (
            MLMonitoringProfileStoreError(
                (
                    "ML Monitoring Profile "
                    "cannot be serialized."
                )
            )
        ) from error


def _decode_json_object(
    raw: object,
    *,
    field_name: str,
) -> dict[
    str,
    Any,
]:

    try:
        value = json.loads(
            str(
                raw
            )
        )

    except Exception as error:
        raise (
            MLMonitoringProfileStoreError(
                (
                    f"{field_name} contains "
                    "invalid JSON."
                )
            )
        ) from error


    if not isinstance(
        value,
        dict,
    ):
        raise (
            MLMonitoringProfileStoreError(
                (
                    f"{field_name} must decode "
                    "to an object."
                )
            )
        )


    return value


# ============================================================
# STORE SCOPE
# ============================================================


def _store_root(
) -> str:

    return (
        ml_model_artifact_store_scope(
            resolve_ml_model_artifact_store_path()
        )
    )


# ============================================================
# VALIDATION
# ============================================================


def _validated_profile(
    value: object,
) -> MLMonitoringProfileRecord:

    try:
        return (
            MLMonitoringProfileRecord
            .model_validate(
                value
            )
        )

    except ValidationError as error:
        raise (
            MLMonitoringProfileStoreError(
                (
                    "Invalid ML Monitoring "
                    "Profile record."
                )
            )
        ) from error


# ============================================================
# DATABASE ROW
# ============================================================


def _row_to_record(
    row,
) -> MLMonitoringProfileRecord:

    payload = (
        _decode_json_object(
            row[
                "payload_json"
            ],
            field_name=
                "payload_json",
        )
    )


    record = (
        _validated_profile(
            payload
        )
    )


    expected = {
        "profile_id":
            str(
                row[
                    "profile_id"
                ]
            ),

        "model_id":
            str(
                row[
                    "model_id"
                ]
            ),

        "workflow_id":
            str(
                row[
                    "workflow_id"
                ]
            ),

        "dataset_id":
            str(
                row[
                    "dataset_id"
                ]
            ),

        "experiment_id":
            str(
                row[
                    "experiment_id"
                ]
            ),

        "preparation_session_revision":
            int(
                row[
                    "preparation_session_revision"
                ]
            ),

        "training_contract_sha256":
            str(
                row[
                    "training_contract_sha256"
                ]
            ),

        "created_at_utc":
            str(
                row[
                    "created_at_utc"
                ]
            ),

        "reference_scope":
            str(
                row[
                    "reference_scope"
                ]
            ),

        "reference_row_count":
            int(
                row[
                    "reference_row_count"
                ]
            ),

        "privacy_scope":
            str(
                row[
                    "privacy_scope"
                ]
            ),

        "categorical_identity":
            str(
                row[
                    "categorical_identity"
                ]
            ),

        "rule_version":
            str(
                row[
                    "rule_version"
                ]
            ),
    }


    actual = {
        "profile_id":
            record.profile_id,

        "model_id":
            record.model_id,

        "workflow_id":
            record.workflow_id,

        "dataset_id":
            record.dataset_id,

        "experiment_id":
            record.experiment_id,

        "preparation_session_revision":
            record.preparation_session_revision,

        "training_contract_sha256":
            record.training_contract_sha256,

        "created_at_utc":
            record.created_at_utc,

        "reference_scope":
            record.reference_scope,

        "reference_row_count":
            record.reference_row_count,

        "privacy_scope":
            record.privacy_scope,

        "categorical_identity":
            record.categorical_identity,

        "rule_version":
            record.rule_version,
    }


    if (
        actual
        !=
        expected
    ):
        raise (
            MLMonitoringProfileStoreError(
                (
                    "Stored ML Monitoring Profile "
                    "index fields do not match "
                    "payload_json."
                )
            )
        )


    return record


# ============================================================
# MODEL ARTIFACT AUTHORITY
# ============================================================


def _assert_model_artifact_authority(
    *,
    connection,
    store_root: str,
    profile: MLMonitoringProfileRecord,
) -> None:
    """
    Bind the monitoring reference to one already persisted
    trusted Model Artifact.

    The historical Model Artifact is authoritative.

    The current Preparation revision is deliberately NOT read
    here: a monitoring reference describes the immutable
    training snapshot of the persisted model, even if the
    Preparation workflow has subsequently advanced.
    """

    row = (
        connection.execute(
            """
            SELECT
                workflow_id,
                dataset_id,
                experiment_id,
                experiment_provenance_json,
                train_rows

            FROM ml_model_artifacts

            WHERE
                store_root = ?
                AND
                model_id = ?

            LIMIT 1
            """,
            (
                store_root,
                profile.model_id,
            ),
        )
        .fetchone()
    )


    if row is None:
        raise (
            MLMonitoringProfileAuthorityError(
                (
                    "ML Monitoring Profile "
                    "references a Model Artifact "
                    "that is not server-owned."
                )
            )
        )


    stored_workflow_id = str(
        row[
            "workflow_id"
        ]
    )


    stored_dataset_id = str(
        row[
            "dataset_id"
        ]
    )


    stored_experiment_id = (
        str(
            row[
                "experiment_id"
            ]
        )

        if row[
            "experiment_id"
        ]
        is not None

        else None
    )


    if (
        stored_workflow_id
        !=
        profile.workflow_id
    ):
        raise (
            MLMonitoringProfileAuthorityError(
                (
                    "Monitoring workflow_id does "
                    "not match Model Artifact."
                )
            )
        )


    if (
        stored_dataset_id
        !=
        profile.dataset_id
    ):
        raise (
            MLMonitoringProfileAuthorityError(
                (
                    "Monitoring dataset_id does "
                    "not match Model Artifact."
                )
            )
        )


    if (
        stored_experiment_id
        is None
        or
        stored_experiment_id
        !=
        profile.experiment_id
    ):
        raise (
            MLMonitoringProfileAuthorityError(
                (
                    "Monitoring experiment_id does "
                    "not match trusted Model "
                    "Artifact provenance."
                )
            )
        )


    raw_provenance = (
        row[
            "experiment_provenance_json"
        ]
    )


    if raw_provenance is None:
        raise (
            MLMonitoringProfileAuthorityError(
                (
                    "ML Monitoring Profile requires "
                    "a Model Artifact with trusted "
                    "Experiment Provenance."
                )
            )
        )


    provenance_payload = (
        _decode_json_object(
            raw_provenance,
            field_name=
                "experiment_provenance_json",
        )
    )


    try:
        provenance = (
            MLExperimentProvenanceRecord
            .model_validate(
                provenance_payload
            )
        )

    except ValidationError as error:
        raise (
            MLMonitoringProfileAuthorityError(
                (
                    "Persisted Model Artifact "
                    "Experiment Provenance is invalid."
                )
            )
        ) from error


    if (
        provenance.model_id
        !=
        profile.model_id
        or
        provenance.workflow_id
        !=
        profile.workflow_id
        or
        provenance.dataset_id
        !=
        profile.dataset_id
        or
        provenance.experiment_id
        !=
        profile.experiment_id
    ):
        raise (
            MLMonitoringProfileAuthorityError(
                (
                    "Monitoring identity does not "
                    "match persisted Experiment "
                    "Provenance."
                )
            )
        )


    if (
        provenance.preparation_session_revision
        !=
        profile.preparation_session_revision
    ):
        raise (
            MLMonitoringProfileAuthorityError(
                (
                    "Monitoring Preparation revision "
                    "does not match persisted "
                    "Experiment Provenance."
                )
            )
        )


    if (
        provenance.training_contract_sha256
        !=
        profile.training_contract_sha256
    ):
        raise (
            MLMonitoringProfileAuthorityError(
                (
                    "Monitoring Training Contract "
                    "fingerprint does not match "
                    "persisted Experiment Provenance."
                )
            )
        )


    stored_train_rows = int(
        row[
            "train_rows"
        ]
    )


    if (
        provenance.train_rows
        !=
        stored_train_rows
        or
        profile.reference_row_count
        !=
        stored_train_rows
    ):
        raise (
            MLMonitoringProfileAuthorityError(
                (
                    "Monitoring reference row count "
                    "does not match the persisted "
                    "training split."
                )
            )
        )


# ============================================================
# REGISTER
# ============================================================


def register_ml_monitoring_profile(
    *,
    profile: MLMonitoringProfileRecord,
) -> MLMonitoringProfileRecord:
    """
    Persist one immutable aggregate-only monitoring reference.

    Exactly one reference profile is allowed per Model Artifact.

    Existing profiles are never silently overwritten.
    """

    validated = (
        _validated_profile(
            profile
        )
    )


    store_root = (
        _store_root()
    )


    payload_json = (
        _canonical_json(
            validated.model_dump(
                mode="json"
            )
        )
    )


    with sqlite_connection(
        write=True
    ) as connection:

        _assert_model_artifact_authority(
            connection=
                connection,

            store_root=
                store_root,

            profile=
                validated,
        )


        existing = (
            connection.execute(
                """
                SELECT profile_id
                FROM ml_monitoring_profiles

                WHERE
                    store_root = ?
                    AND
                    model_id = ?

                LIMIT 1
                """,
                (
                    store_root,
                    validated.model_id,
                ),
            )
            .fetchone()
        )


        if existing is not None:
            raise (
                MLMonitoringProfileAlreadyExistsError(
                    (
                        "A Monitoring Profile already "
                        "exists for this Model Artifact."
                    )
                )
            )


        connection.execute(
            """
            INSERT INTO ml_monitoring_profiles (
                store_root,
                profile_id,
                model_id,
                workflow_id,
                dataset_id,
                experiment_id,
                preparation_session_revision,
                training_contract_sha256,
                created_at_utc,
                reference_scope,
                reference_row_count,
                privacy_scope,
                categorical_identity,
                rule_version,
                payload_json
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
                ?,
                ?
            )
            """,
            (
                store_root,

                validated.profile_id,

                validated.model_id,

                validated.workflow_id,

                validated.dataset_id,

                validated.experiment_id,

                validated
                .preparation_session_revision,

                validated
                .training_contract_sha256,

                validated.created_at_utc,

                validated.reference_scope,

                validated.reference_row_count,

                validated.privacy_scope,

                validated.categorical_identity,

                validated.rule_version,

                payload_json,
            ),
        )


    return validated


# ============================================================
# GET
# ============================================================


def get_ml_monitoring_profile(
    *,
    model_id: str,
    workflow_id: (
        str
        |
        None
    ) = None,
) -> MLMonitoringProfileRecord:

    normalized_model_id = (
        _required_text(
            model_id,
            field_name=
                "model_id",
        )
    )


    normalized_workflow_id = (
        _required_text(
            workflow_id,
            field_name=
                "workflow_id",
        )

        if workflow_id
        is not None

        else None
    )


    store_root = (
        _store_root()
    )


    with sqlite_connection(
        write=False
    ) as connection:

        row = (
            connection.execute(
                """
                SELECT *
                FROM ml_monitoring_profiles

                WHERE
                    store_root = ?
                    AND
                    model_id = ?

                LIMIT 1
                """,
                (
                    store_root,
                    normalized_model_id,
                ),
            )
            .fetchone()
        )


    if row is None:
        raise (
            MLMonitoringProfileNotFoundError(
                (
                    "ML Monitoring Profile "
                    "was not found."
                )
            )
        )


    record = (
        _row_to_record(
            row
        )
    )


    if (
        normalized_workflow_id
        is not None
        and
        record.workflow_id
        !=
        normalized_workflow_id
    ):
        raise (
            MLMonitoringProfileWorkflowMismatchError(
                (
                    "ML Monitoring Profile does "
                    "not belong to the requested "
                    "workflow."
                )
            )
        )


    return record


# ============================================================
# LIST WORKFLOW
# ============================================================


def list_ml_monitoring_profiles(
    *,
    workflow_id: str,
) -> list[
    MLMonitoringProfileRecord
]:

    normalized_workflow_id = (
        _required_text(
            workflow_id,
            field_name=
                "workflow_id",
        )
    )


    store_root = (
        _store_root()
    )


    with sqlite_connection(
        write=False
    ) as connection:

        rows = (
            connection.execute(
                """
                SELECT *
                FROM ml_monitoring_profiles

                WHERE
                    store_root = ?
                    AND
                    workflow_id = ?

                ORDER BY
                    created_at_utc ASC,
                    model_id ASC
                """,
                (
                    store_root,
                    normalized_workflow_id,
                ),
            )
            .fetchall()
        )


    return [
        _row_to_record(
            row
        )

        for row
        in rows
    ]
