from __future__ import annotations


import json


from typing import (
    Any,
)


from pydantic import (
    ValidationError,
)


from app.ml.drift_evaluation import (
    MLDriftEvaluationRecord,
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


ML_DRIFT_EVALUATION_STORE_RULE_VERSION = (
    "ml_drift_evaluation_store_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLDriftEvaluationStoreError(
    RuntimeError
):
    pass


class MLDriftEvaluationNotFoundError(
    MLDriftEvaluationStoreError
):
    pass


class MLDriftEvaluationAlreadyExistsError(
    MLDriftEvaluationStoreError
):
    pass


class MLDriftEvaluationAuthorityError(
    MLDriftEvaluationStoreError
):
    pass


class MLDriftEvaluationWorkflowMismatchError(
    MLDriftEvaluationStoreError
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
        raise MLDriftEvaluationStoreError(
            (
                f"{field_name} "
                "cannot be empty."
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
        raise MLDriftEvaluationStoreError(
            (
                "ML Drift Evaluation "
                "cannot be serialized."
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
        raise MLDriftEvaluationStoreError(
            (
                f"{field_name} contains "
                "invalid JSON."
            )
        ) from error


    if not isinstance(
        value,
        dict,
    ):
        raise MLDriftEvaluationStoreError(
            (
                f"{field_name} must decode "
                "to an object."
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


def _validated_evaluation(
    value: object,
) -> MLDriftEvaluationRecord:

    try:
        return (
            MLDriftEvaluationRecord
            .model_validate(
                value
            )
        )

    except ValidationError as error:
        raise MLDriftEvaluationStoreError(
            (
                "Invalid ML Drift "
                "Evaluation record."
            )
        ) from error


# ============================================================
# DATABASE ROW
# ============================================================


def _row_to_record(
    row,
) -> MLDriftEvaluationRecord:

    payload = (
        _decode_json_object(
            row[
                "payload_json"
            ],
            field_name="payload_json",
        )
    )


    record = (
        _validated_evaluation(
            payload
        )
    )


    expected = {
        "evaluation_id":
            str(
                row[
                    "evaluation_id"
                ]
            ),

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

        "reference_dataset_id":
            str(
                row[
                    "reference_dataset_id"
                ]
            ),

        "observed_dataset_id":
            str(
                row[
                    "observed_dataset_id"
                ]
            ),

        "observed_preparation_session_revision":
            (
                int(
                    row[
                        "observed_preparation_session_revision"
                    ]
                )

                if row[
                    "observed_preparation_session_revision"
                ]
                is not None

                else None
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

        "evaluated_at_utc":
            str(
                row[
                    "evaluated_at_utc"
                ]
            ),

        "observed_row_count":
            int(
                row[
                    "observed_row_count"
                ]
            ),

        "warning_feature_count":
            int(
                row[
                    "warning_feature_count"
                ]
            ),

        "drift_feature_count":
            int(
                row[
                    "drift_feature_count"
                ]
            ),

        "overall_status":
            str(
                row[
                    "overall_status"
                ]
            ),

        "privacy_scope":
            str(
                row[
                    "privacy_scope"
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
        "evaluation_id":
            record.evaluation_id,

        "profile_id":
            record.profile_id,

        "model_id":
            record.model_id,

        "workflow_id":
            record.workflow_id,

        "reference_dataset_id":
            record.reference_dataset_id,

        "observed_dataset_id":
            record.observed_dataset_id,

        "observed_preparation_session_revision":
            record.observed_preparation_session_revision,

        "experiment_id":
            record.experiment_id,

        "preparation_session_revision":
            record.preparation_session_revision,

        "training_contract_sha256":
            record.training_contract_sha256,

        "evaluated_at_utc":
            record.evaluated_at_utc,

        "observed_row_count":
            record.observed_row_count,

        "warning_feature_count":
            record.warning_feature_count,

        "drift_feature_count":
            record.drift_feature_count,

        "overall_status":
            record.overall_status,

        "privacy_scope":
            record.privacy_scope,

        "rule_version":
            record.rule_version,
    }


    if actual != expected:
        raise MLDriftEvaluationStoreError(
            (
                "Stored ML Drift Evaluation "
                "index fields do not match "
                "payload_json."
            )
        )


    return record


# ============================================================
# MONITORING PROFILE AUTHORITY
# ============================================================


def _assert_monitoring_profile_authority(
    *,
    connection,
    store_root: str,
    evaluation: MLDriftEvaluationRecord,
) -> None:
    """
    Bind one Drift Evaluation to one already persisted trusted
    Monitoring Profile.

    The persisted Monitoring Profile is historical authority.
    The current Preparation revision is deliberately irrelevant.
    """

    row = (
        connection.execute(
            """
            SELECT *
            FROM ml_monitoring_profiles

            WHERE
                store_root = ?
                AND
                profile_id = ?

            LIMIT 1
            """,
            (
                store_root,
                evaluation.profile_id,
            ),
        )
        .fetchone()
    )


    if row is None:
        raise MLDriftEvaluationAuthorityError(
            (
                "ML Drift Evaluation references "
                "a Monitoring Profile that is "
                "not server-owned."
            )
        )


    payload = (
        _decode_json_object(
            row[
                "payload_json"
            ],
            field_name=(
                "monitoring_profile.payload_json"
            ),
        )
    )


    try:
        profile = (
            MLMonitoringProfileRecord
            .model_validate(
                payload
            )
        )

    except ValidationError as error:
        raise MLDriftEvaluationAuthorityError(
            (
                "Persisted Monitoring Profile "
                "payload is invalid."
            )
        ) from error


    indexed_profile = {
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
    }


    payload_profile = {
        "profile_id":
            profile.profile_id,

        "model_id":
            profile.model_id,

        "workflow_id":
            profile.workflow_id,

        "dataset_id":
            profile.dataset_id,

        "experiment_id":
            profile.experiment_id,

        "preparation_session_revision":
            profile.preparation_session_revision,

        "training_contract_sha256":
            profile.training_contract_sha256,
    }


    if indexed_profile != payload_profile:
        raise MLDriftEvaluationAuthorityError(
            (
                "Persisted Monitoring Profile "
                "index fields do not match its "
                "payload."
            )
        )


    bindings = [
        (
            "profile_id",
            evaluation.profile_id,
            profile.profile_id,
        ),
        (
            "model_id",
            evaluation.model_id,
            profile.model_id,
        ),
        (
            "workflow_id",
            evaluation.workflow_id,
            profile.workflow_id,
        ),
        (
            "reference_dataset_id",
            evaluation.reference_dataset_id,
            profile.dataset_id,
        ),
        (
            "experiment_id",
            evaluation.experiment_id,
            profile.experiment_id,
        ),
        (
            "preparation_session_revision",
            evaluation.preparation_session_revision,
            profile.preparation_session_revision,
        ),
        (
            "training_contract_sha256",
            evaluation.training_contract_sha256,
            profile.training_contract_sha256,
        ),
        (
            "privacy_scope",
            evaluation.privacy_scope,
            profile.privacy_scope,
        ),
    ]


    for (
        name,
        evaluation_value,
        authority_value,
    ) in bindings:

        if evaluation_value != authority_value:
            raise MLDriftEvaluationAuthorityError(
                (
                    "ML Drift Evaluation does not "
                    "match persisted Monitoring "
                    "Profile authority. "
                    f"field={name}"
                )
            )


# ============================================================
# REGISTER
# ============================================================


def register_ml_drift_evaluation(
    *,
    evaluation: MLDriftEvaluationRecord,
) -> MLDriftEvaluationRecord:
    """
    Persist one immutable aggregate-only Drift Evaluation.

    Many evaluations may belong to one Monitoring Profile.

    An evaluation identity is never silently overwritten.
    """

    validated = (
        _validated_evaluation(
            evaluation
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

        _assert_monitoring_profile_authority(
            connection=connection,
            store_root=store_root,
            evaluation=validated,
        )


        # ====================================================
        # OBSERVED PREPARATION SNAPSHOT AUTHORITY
        #
        # Registration is the durable commit boundary.
        #
        # The revision check therefore occurs inside the same
        # SQLite write transaction as Drift persistence.
        # A Preparation change that happened after evaluation
        # cannot be silently committed as historical evidence.
        # ====================================================


        if (
            validated
            .observed_preparation_session_revision
            is None
        ):
            raise MLDriftEvaluationAuthorityError(
                (
                    "New ML Drift Evaluations require "
                    "an observed Preparation revision."
                )
            )


        observed_session = (
            connection.execute(
                """
                SELECT revision
                FROM preparation_sessions

                WHERE
                    workflow_id = ?

                LIMIT 1
                """,
                (
                    validated.workflow_id,
                ),
            )
            .fetchone()
        )


        if observed_session is None:
            raise MLDriftEvaluationAuthorityError(
                (
                    "Observed Preparation workflow "
                    "is no longer server-owned."
                )
            )


        current_observed_revision = int(
            observed_session[
                "revision"
            ]
        )


        if (
            current_observed_revision
            !=
            validated
            .observed_preparation_session_revision
        ):
            raise MLDriftEvaluationAuthorityError(
                (
                    "Observed Preparation revision "
                    "changed before Drift Evaluation "
                    "persistence. "
                    "expected_revision="
                    f"{validated.observed_preparation_session_revision}, "
                    "current_revision="
                    f"{current_observed_revision}"
                )
            )


        existing = (
            connection.execute(
                """
                SELECT evaluation_id
                FROM ml_drift_evaluations

                WHERE
                    store_root = ?
                    AND
                    evaluation_id = ?

                LIMIT 1
                """,
                (
                    store_root,
                    validated.evaluation_id,
                ),
            )
            .fetchone()
        )


        if existing is not None:
            raise MLDriftEvaluationAlreadyExistsError(
                (
                    "ML Drift Evaluation "
                    "identity already exists."
                )
            )


        connection.execute(
            """
            INSERT INTO ml_drift_evaluations (
                store_root,
                evaluation_id,
                profile_id,
                model_id,
                workflow_id,
                reference_dataset_id,
                observed_dataset_id,
                observed_preparation_session_revision,
                experiment_id,
                preparation_session_revision,
                training_contract_sha256,
                evaluated_at_utc,
                observed_row_count,
                warning_feature_count,
                drift_feature_count,
                overall_status,
                privacy_scope,
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
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                store_root,

                validated.evaluation_id,

                validated.profile_id,

                validated.model_id,

                validated.workflow_id,

                validated.reference_dataset_id,

                validated.observed_dataset_id,

                validated
                .observed_preparation_session_revision,

                validated.experiment_id,

                validated
                .preparation_session_revision,

                validated
                .training_contract_sha256,

                validated.evaluated_at_utc,

                validated.observed_row_count,

                validated.warning_feature_count,

                validated.drift_feature_count,

                validated.overall_status,

                validated.privacy_scope,

                validated.rule_version,

                payload_json,
            ),
        )


    return validated


# ============================================================
# GET
# ============================================================


def get_ml_drift_evaluation(
    *,
    evaluation_id: str,
    workflow_id: (
        str
        |
        None
    ) = None,
) -> MLDriftEvaluationRecord:

    normalized_evaluation_id = (
        _required_text(
            evaluation_id,
            field_name="evaluation_id",
        )
    )


    normalized_workflow_id = (
        _required_text(
            workflow_id,
            field_name="workflow_id",
        )

        if workflow_id is not None

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
                FROM ml_drift_evaluations

                WHERE
                    store_root = ?
                    AND
                    evaluation_id = ?

                LIMIT 1
                """,
                (
                    store_root,
                    normalized_evaluation_id,
                ),
            )
            .fetchone()
        )


    if row is None:
        raise MLDriftEvaluationNotFoundError(
            (
                "ML Drift Evaluation "
                "was not found."
            )
        )


    record = (
        _row_to_record(
            row
        )
    )


    if (
        normalized_workflow_id is not None
        and
        record.workflow_id
        !=
        normalized_workflow_id
    ):
        raise MLDriftEvaluationWorkflowMismatchError(
            (
                "ML Drift Evaluation does "
                "not belong to the requested "
                "workflow."
            )
        )


    return record


# ============================================================
# LIST MODEL
# ============================================================


def list_ml_drift_evaluations_for_model(
    *,
    model_id: str,
    workflow_id: (
        str
        |
        None
    ) = None,
) -> list[
    MLDriftEvaluationRecord
]:

    normalized_model_id = (
        _required_text(
            model_id,
            field_name="model_id",
        )
    )


    normalized_workflow_id = (
        _required_text(
            workflow_id,
            field_name="workflow_id",
        )

        if workflow_id is not None

        else None
    )


    store_root = (
        _store_root()
    )


    query = """
        SELECT *
        FROM ml_drift_evaluations

        WHERE
            store_root = ?
            AND
            model_id = ?
    """


    parameters: list[
        object
    ] = [
        store_root,
        normalized_model_id,
    ]


    if normalized_workflow_id is not None:
        query += """
            AND
            workflow_id = ?
        """

        parameters.append(
            normalized_workflow_id
        )


    query += """
        ORDER BY
            evaluated_at_utc ASC,
            evaluation_id ASC
    """


    with sqlite_connection(
        write=False
    ) as connection:

        rows = (
            connection.execute(
                query,
                tuple(
                    parameters
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


# ============================================================
# LIST WORKFLOW
# ============================================================


def list_ml_drift_evaluations_for_workflow(
    *,
    workflow_id: str,
) -> list[
    MLDriftEvaluationRecord
]:

    normalized_workflow_id = (
        _required_text(
            workflow_id,
            field_name="workflow_id",
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
                FROM ml_drift_evaluations

                WHERE
                    store_root = ?
                    AND
                    workflow_id = ?

                ORDER BY
                    evaluated_at_utc ASC,
                    evaluation_id ASC
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
