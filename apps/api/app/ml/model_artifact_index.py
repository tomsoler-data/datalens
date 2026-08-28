from __future__ import annotations


import json


from pathlib import (
    Path,
)


from typing import (
    Any,
)


from pydantic import (
    ValidationError,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.persistence.sqlite_database import (
    sqlite_connection,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_ARTIFACT_INDEX_VERSION = (
    "ml_model_artifact_sqlite_index_v0.1"
)


# ============================================================
# ERROR
# ============================================================


class MLModelArtifactIndexError(
    RuntimeError
):
    pass


# ============================================================
# SCOPE
# ============================================================


def ml_model_artifact_store_scope(
    store_path: Path,
) -> str:
    return str(
        store_path
        .expanduser()
        .resolve()
    )


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
            MLModelArtifactIndexError(
                (
                    "Model Artifact metadata "
                    "cannot be serialized as JSON."
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
            MLModelArtifactIndexError(
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
            MLModelArtifactIndexError(
                (
                    f"{field_name} must "
                    "decode to an object."
                )
            )
        )


    return value


def _decode_optional_json_object(
    raw: object,
    *,
    field_name: str,
) -> (
    dict[
        str,
        Any,
    ]
    |
    None
):

    if raw is None:
        return None


    return (
        _decode_json_object(
            raw,
            field_name=
                field_name,
        )
    )


# ============================================================
# VALIDATION
# ============================================================


def validate_ml_model_artifact_index_entry(
    entry: object,
) -> dict[
    str,
    Any,
]:
    try:
        record = (
            MLModelArtifactRecord
            .model_validate(
                entry
            )
        )

    except ValidationError as error:
        raise (
            MLModelArtifactIndexError(
                (
                    "Invalid ML Model Artifact "
                    "metadata entry."
                )
            )
        ) from error


    contract = (
        record.training_contract
        .model_dump(
            mode="json"
        )
    )


    metrics = dict(
        record.metrics
    )


    provenance = (
        record
        .experiment_provenance
    )


    experiment_provenance = (
        provenance.model_dump(
            mode="json"
        )

        if provenance
        is not None

        else None
    )


    experiment_id = (
        provenance.experiment_id

        if provenance
        is not None

        else None
    )


    return {
        "model_id":
            record.model_id,

        "workflow_id":
            record.workflow_id,

        "dataset_id":
            record.dataset_id,

        "problem_type":
            record.training_contract.problem_type,

        "target_column":
            record.training_contract.target_column,

        "estimator_key":
            record.training_contract.estimator_key,

        "experiment_id":
            experiment_id,

        "experiment_provenance":
            experiment_provenance,

        "training_contract":
            contract,

        "metrics":
            metrics,

        "train_rows":
            record.train_rows,

        "test_rows":
            record.test_rows,

        "created_at_utc":
            record.created_at_utc,

        "serialization_format":
            record.serialization_format,

        "rule_version":
            record.rule_version,

        "model_path":
            record.model_path,

        "model_file_bytes":
            record.model_file_bytes,

        "model_sha256":
            record.model_sha256,
    }


# ============================================================
# DATABASE ROW
# ============================================================


def _row_to_entry(
    row,
) -> dict[
    str,
    Any,
]:

    training_contract = (
        _decode_json_object(
            row[
                "training_contract_json"
            ],
            field_name=
                "training_contract_json",
        )
    )


    metrics = (
        _decode_json_object(
            row[
                "metrics_json"
            ],
            field_name=
                "metrics_json",
        )
    )


    experiment_provenance = (
        _decode_optional_json_object(
            row[
                "experiment_provenance_json"
            ],
            field_name=
                "experiment_provenance_json",
        )
    )


    entry = {
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

        "experiment_provenance":
            experiment_provenance,

        "training_contract":
            training_contract,

        "metrics":
            metrics,

        "train_rows":
            int(
                row[
                    "train_rows"
                ]
            ),

        "test_rows":
            int(
                row[
                    "test_rows"
                ]
            ),

        "created_at_utc":
            str(
                row[
                    "created_at_utc"
                ]
            ),

        "serialization_format":
            str(
                row[
                    "serialization_format"
                ]
            ),

        "rule_version":
            str(
                row[
                    "rule_version"
                ]
            ),

        "model_path":
            str(
                row[
                    "model_path"
                ]
            ),

        "model_file_bytes":
            int(
                row[
                    "model_file_bytes"
                ]
            ),

        "model_sha256":
            str(
                row[
                    "model_sha256"
                ]
            ),
    }


    validated = (
        validate_ml_model_artifact_index_entry(
            entry
        )
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
        validated[
            "experiment_id"
        ]
        !=
        stored_experiment_id
    ):
        raise (
            MLModelArtifactIndexError(
                (
                    "Stored experiment_id does not "
                    "match experiment provenance."
                )
            )
        )


    if (
        validated[
            "problem_type"
        ]
        !=
        str(
            row[
                "problem_type"
            ]
        )
    ):
        raise (
            MLModelArtifactIndexError(
                (
                    "Stored problem_type does not "
                    "match training_contract_json."
                )
            )
        )


    if (
        validated[
            "target_column"
        ]
        !=
        str(
            row[
                "target_column"
            ]
        )
    ):
        raise (
            MLModelArtifactIndexError(
                (
                    "Stored target_column does not "
                    "match training_contract_json."
                )
            )
        )


    if (
        validated[
            "estimator_key"
        ]
        !=
        str(
            row[
                "estimator_key"
            ]
        )
    ):
        raise (
            MLModelArtifactIndexError(
                (
                    "Stored estimator_key does not "
                    "match training_contract_json."
                )
            )
        )


    return validated


# ============================================================
# UPSERT
# ============================================================


def upsert_ml_model_artifact_index_entry(
    *,
    store_path: Path,
    entry: object,
    expected_preparation_session_revision: (
        int
        |
        None
    ) = None,
) -> dict[
    str,
    Any,
]:

    validated = (
        validate_ml_model_artifact_index_entry(
            entry
        )
    )


    normalized_expected_revision = (
        None
    )


    if (
        expected_preparation_session_revision
        is not None
    ):

        if isinstance(
            expected_preparation_session_revision,
            bool,
        ):
            raise (
                MLModelArtifactIndexError(
                    (
                        "expected Preparation session "
                        "revision must be a "
                        "non-negative integer."
                    )
                )
            )


        try:
            normalized_expected_revision = int(
                expected_preparation_session_revision
            )

        except Exception as error:
            raise (
                MLModelArtifactIndexError(
                    (
                        "expected Preparation session "
                        "revision must be a "
                        "non-negative integer."
                    )
                )
            ) from error


        if (
            normalized_expected_revision
            <
            0
        ):
            raise (
                MLModelArtifactIndexError(
                    (
                        "expected Preparation session "
                        "revision must be a "
                        "non-negative integer."
                    )
                )
            )


    store_root = (
        ml_model_artifact_store_scope(
            store_path
        )
    )


    training_contract_json = (
        _canonical_json(
            validated[
                "training_contract"
            ]
        )
    )


    metrics_json = (
        _canonical_json(
            validated[
                "metrics"
            ]
        )
    )


    experiment_provenance_json = (
        _canonical_json(
            validated[
                "experiment_provenance"
            ]
        )

        if (
            validated[
                "experiment_provenance"
            ]
            is not None
        )

        else None
    )


    with sqlite_connection(
        write=True
    ) as connection:

        # ====================================================
        # ATOMIC PREPARATION SNAPSHOT GUARD
        #
        # sqlite_connection(write=True) has already acquired
        # BEGIN IMMEDIATE before yielding this connection.
        #
        # No concurrent Preparation writer can therefore change
        # the revision between this validation and the Model
        # Artifact metadata upsert below.
        # ====================================================

        if (
            normalized_expected_revision
            is not None
        ):

            revision_row = (
                connection.execute(
                    """
                    SELECT revision
                    FROM preparation_sessions

                    WHERE
                        workflow_id = ?
                    """,
                    (
                        validated[
                            "workflow_id"
                        ],
                    ),
                )
                .fetchone()
            )


            if revision_row is None:
                raise (
                    MLModelArtifactIndexError(
                        (
                            "Preparation workflow disappeared "
                            "before atomic Model Artifact "
                            "metadata persistence."
                        )
                    )
                )


            current_revision = int(
                revision_row[
                    "revision"
                ]
            )


            if (
                current_revision
                !=
                normalized_expected_revision
            ):
                raise (
                    MLModelArtifactIndexError(
                        (
                            "Preparation session revision "
                            "changed before atomic Model "
                            "Artifact metadata persistence. "
                            "expected_revision="
                            f"{normalized_expected_revision}, "
                            "current_revision="
                            f"{current_revision}"
                        )
                    )
                )


        connection.execute(
            """
            INSERT INTO ml_model_artifacts (
                store_root,
                model_id,
                workflow_id,
                dataset_id,
                problem_type,
                target_column,
                estimator_key,
                experiment_id,
                experiment_provenance_json,
                training_contract_json,
                metrics_json,
                train_rows,
                test_rows,
                created_at_utc,
                serialization_format,
                rule_version,
                model_path,
                model_file_bytes,
                model_sha256
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

            ON CONFLICT (
                store_root,
                model_id
            )

            DO UPDATE SET
                workflow_id =
                    excluded.workflow_id,

                dataset_id =
                    excluded.dataset_id,

                problem_type =
                    excluded.problem_type,

                target_column =
                    excluded.target_column,

                estimator_key =
                    excluded.estimator_key,

                experiment_id =
                    excluded.experiment_id,

                experiment_provenance_json =
                    excluded.experiment_provenance_json,

                training_contract_json =
                    excluded.training_contract_json,

                metrics_json =
                    excluded.metrics_json,

                train_rows =
                    excluded.train_rows,

                test_rows =
                    excluded.test_rows,

                created_at_utc =
                    excluded.created_at_utc,

                serialization_format =
                    excluded.serialization_format,

                rule_version =
                    excluded.rule_version,

                model_path =
                    excluded.model_path,

                model_file_bytes =
                    excluded.model_file_bytes,

                model_sha256 =
                    excluded.model_sha256
            """,
            (
                store_root,

                validated[
                    "model_id"
                ],

                validated[
                    "workflow_id"
                ],

                validated[
                    "dataset_id"
                ],

                validated[
                    "problem_type"
                ],

                validated[
                    "target_column"
                ],

                validated[
                    "estimator_key"
                ],

                validated[
                    "experiment_id"
                ],

                experiment_provenance_json,

                training_contract_json,

                metrics_json,

                validated[
                    "train_rows"
                ],

                validated[
                    "test_rows"
                ],

                validated[
                    "created_at_utc"
                ],

                validated[
                    "serialization_format"
                ],

                validated[
                    "rule_version"
                ],

                validated[
                    "model_path"
                ],

                validated[
                    "model_file_bytes"
                ],

                validated[
                    "model_sha256"
                ],
            ),
        )


    return validated


# ============================================================
# GET ONE
# ============================================================


def get_ml_model_artifact_index_entry(
    *,
    store_path: Path,
    model_id: str,
) -> (
    dict[
        str,
        Any,
    ]
    |
    None
):

    normalized_model_id = str(
        model_id
    ).strip()


    if not normalized_model_id:
        raise (
            MLModelArtifactIndexError(
                "model_id cannot be empty."
            )
        )


    store_root = (
        ml_model_artifact_store_scope(
            store_path
        )
    )


    with sqlite_connection(
        write=False
    ) as connection:

        row = (
            connection.execute(
                """
                SELECT *
                FROM ml_model_artifacts

                WHERE
                    store_root = ?
                    AND
                    model_id = ?
                """,
                (
                    store_root,
                    normalized_model_id,
                ),
            )
            .fetchone()
        )


    if row is None:
        return None


    return (
        _row_to_entry(
            row
        )
    )


# ============================================================
# LOAD WORKFLOW
# ============================================================


def load_ml_model_artifact_index_workflow(
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
        raise (
            MLModelArtifactIndexError(
                "workflow_id cannot be empty."
            )
        )


    store_root = (
        ml_model_artifact_store_scope(
            store_path
        )
    )


    with sqlite_connection(
        write=False
    ) as connection:

        rows = (
            connection.execute(
                """
                SELECT *
                FROM ml_model_artifacts

                WHERE
                    store_root = ?
                    AND
                    workflow_id = ?

                ORDER BY
                    created_at_utc,
                    model_id
                """,
                (
                    store_root,
                    normalized_workflow_id,
                ),
            )
            .fetchall()
        )


    return [
        _row_to_entry(
            row
        )
        for row
        in rows
    ]


# ============================================================
# LOAD SCOPE
# ============================================================


def load_ml_model_artifact_index_scope(
    *,
    store_path: Path,
) -> list[
    dict[
        str,
        Any,
    ]
]:

    store_root = (
        ml_model_artifact_store_scope(
            store_path
        )
    )


    with sqlite_connection(
        write=False
    ) as connection:

        rows = (
            connection.execute(
                """
                SELECT *
                FROM ml_model_artifacts

                WHERE
                    store_root = ?

                ORDER BY
                    created_at_utc,
                    model_id
                """,
                (
                    store_root,
                ),
            )
            .fetchall()
        )


    return [
        _row_to_entry(
            row
        )
        for row
        in rows
    ]


# ============================================================
# DELETE ONE
# ============================================================


def delete_ml_model_artifact_index_entry(
    *,
    store_path: Path,
    model_id: str,
) -> bool:

    normalized_model_id = str(
        model_id
    ).strip()


    if not normalized_model_id:
        raise (
            MLModelArtifactIndexError(
                "model_id cannot be empty."
            )
        )


    store_root = (
        ml_model_artifact_store_scope(
            store_path
        )
    )


    with sqlite_connection(
        write=True
    ) as connection:

        cursor = (
            connection.execute(
                """
                DELETE FROM ml_model_artifacts

                WHERE
                    store_root = ?
                    AND
                    model_id = ?
                """,
                (
                    store_root,
                    normalized_model_id,
                ),
            )
        )


        return (
            cursor.rowcount
            >
            0
        )


# ============================================================
# DELETE WORKFLOW
# ============================================================


def delete_ml_model_artifact_index_workflow(
    *,
    store_path: Path,
    workflow_id: str,
) -> int:

    normalized_workflow_id = str(
        workflow_id
    ).strip()


    if not normalized_workflow_id:
        raise (
            MLModelArtifactIndexError(
                "workflow_id cannot be empty."
            )
        )


    store_root = (
        ml_model_artifact_store_scope(
            store_path
        )
    )


    with sqlite_connection(
        write=True
    ) as connection:

        cursor = (
            connection.execute(
                """
                DELETE FROM ml_model_artifacts

                WHERE
                    store_root = ?
                    AND
                    workflow_id = ?
                """,
                (
                    store_root,
                    normalized_workflow_id,
                ),
            )
        )


        return int(
            cursor.rowcount
        )