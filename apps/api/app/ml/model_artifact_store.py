from __future__ import annotations


import os


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


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.model_artifact_data_plane import (
    MLModelArtifactDataPlaneError,
    delete_ml_model_binary,
    read_ml_model_binary,
    write_ml_model_binary,
)


from app.ml.model_artifact_index import (
    MLModelArtifactIndexError,
    get_ml_model_artifact_index_entry,
    load_ml_model_artifact_index_workflow,
    upsert_ml_model_artifact_index_entry,
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


ML_MODEL_ARTIFACT_STORE_RULE_VERSION = (
    "ml_model_artifact_store_v0.1"
)


DEFAULT_ML_MODEL_ARTIFACT_RELATIVE_PATH = (
    "var/ml/model_artifacts.json"
)


ML_MODEL_ARTIFACT_STORE_PATH_ENV = (
    "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH"
)


# ============================================================
# ERRORS
# ============================================================


class MLModelArtifactStoreError(
    RuntimeError
):
    pass


class MLModelArtifactNotFoundError(
    MLModelArtifactStoreError
):
    pass


class MLModelArtifactAuthorityError(
    MLModelArtifactStoreError
):
    pass


class MLModelArtifactWorkflowMismatchError(
    MLModelArtifactStoreError
):
    pass


# ============================================================
# LOCK
# ============================================================


_STORE_LOCK = RLock()


# ============================================================
# PATH
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


def resolve_ml_model_artifact_store_path(
) -> Path:
    configured = (
        os.getenv(
            ML_MODEL_ARTIFACT_STORE_PATH_ENV,
            "",
        )
        .strip()
    )


    if configured:
        path = (
            Path(
                configured
            )
            .expanduser()
        )


        if path.is_absolute():
            return (
                path.resolve()
            )


        return (
            _api_root()
            /
            path
        ).resolve()


    return (
        _api_root()
        /
        DEFAULT_ML_MODEL_ARTIFACT_RELATIVE_PATH
    ).resolve()


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
        raise ValueError(
            (
                f"{field_name} "
                "cannot be empty."
            )
        )


    return normalized


# ============================================================
# TIME
# ============================================================


def _utc_now_iso(
) -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


# ============================================================
# MODEL IDENTITY
# ============================================================


def _new_server_model_id(
) -> str:
    return (
        "model:"
        +
        uuid4().hex
    )


# ============================================================
# PREPARATION AUTHORITY
# ============================================================


def _assert_preparation_authority(
    *,
    contract: MLTrainingContract,
) -> None:
    """
    A Model Artifact may only reference a workflow and dataset
    already owned by the server-side Preparation control plane.

    This check does not replace the future ML execution gate.
    The future executor must additionally require a validated
    Analysis/ML handoff before training.

    Its purpose here is narrower:
    prevent the artifact store from persisting model provenance
    for invented workflows or invented datasets.
    """

    with sqlite_connection(
        write=False
    ) as connection:

        workflow_row = (
            connection.execute(
                """
                SELECT 1
                FROM preparation_sessions

                WHERE
                    workflow_id = ?
                """,
                (
                    contract.workflow_id,
                ),
            )
            .fetchone()
        )


        if workflow_row is None:
            raise (
                MLModelArtifactAuthorityError(
                    (
                        "ML Model Artifact workflow "
                        "is not server-owned by Preparation. "
                        "workflow_id="
                        f"{contract.workflow_id}"
                    )
                )
            )


        dataset_row = (
            connection.execute(
                """
                SELECT 1
                FROM preparation_artifacts

                WHERE
                    workflow_id = ?
                    AND
                    dataset_id = ?

                LIMIT 1
                """,
                (
                    contract.workflow_id,
                    contract.dataset_id,
                ),
            )
            .fetchone()
        )


        if dataset_row is None:
            raise (
                MLModelArtifactAuthorityError(
                    (
                        "ML Model Artifact dataset "
                        "is not server-owned by the "
                        "referenced Preparation workflow. "
                        "workflow_id="
                        f"{contract.workflow_id}, "
                        "dataset_id="
                        f"{contract.dataset_id}"
                    )
                )
            )


# ============================================================
# INDEX -> RECORD
# ============================================================


def _record_from_index_entry(
    entry: object,
) -> MLModelArtifactRecord:
    """
    Reconstruct the public Model Artifact contract from one
    validated SQLite index entry.

    The SQLite index intentionally contains denormalized search
    fields:

        problem_type
        target_column
        estimator_key

    Those fields are already represented inside
    training_contract and are not part of
    MLModelArtifactRecord itself.

    They must therefore never be forwarded as arbitrary extras
    to the strict Pydantic artifact contract.
    """

    if not isinstance(
        entry,
        dict,
    ):
        raise (
            MLModelArtifactStoreError(
                (
                    "Persisted ML Model Artifact "
                    "metadata entry must be an object."
                )
            )
        )


    artifact_payload = {
        "model_id":
            entry.get(
                "model_id"
            ),

        "workflow_id":
            entry.get(
                "workflow_id"
            ),

        "dataset_id":
            entry.get(
                "dataset_id"
            ),

        "training_contract":
            entry.get(
                "training_contract"
            ),

        "metrics":
            entry.get(
                "metrics"
            ),

        "train_rows":
            entry.get(
                "train_rows"
            ),

        "test_rows":
            entry.get(
                "test_rows"
            ),

        "created_at_utc":
            entry.get(
                "created_at_utc"
            ),

        "serialization_format":
            entry.get(
                "serialization_format"
            ),

        "model_path":
            entry.get(
                "model_path"
            ),

        "model_file_bytes":
            entry.get(
                "model_file_bytes"
            ),

        "model_sha256":
            entry.get(
                "model_sha256"
            ),

        "rule_version":
            entry.get(
                "rule_version"
            ),
    }


    try:
        return (
            MLModelArtifactRecord
            .model_validate(
                artifact_payload
            )
        )

    except Exception as error:
        raise (
            MLModelArtifactStoreError(
                (
                    "Persisted ML Model Artifact "
                    "metadata is invalid."
                )
            )
        ) from error


# ============================================================
# REGISTER
# ============================================================


def register_ml_model_artifact(
    *,
    training_contract: MLTrainingContract,
    metrics: dict[
        str,
        float,
    ],
    train_rows: int,
    test_rows: int,
    model_bytes: bytes,
    created_at_utc: (
        str
        |
        None
    ) = None,
) -> MLModelArtifactRecord:
    """
    Persist one new server-owned trained model.

    Security / ownership contract:

    - model_id is generated only by DataLens;
    - model_path is generated only by the filesystem data plane;
    - callers provide opaque bytes, never a filesystem path;
    - workflow/dataset provenance must already exist in
      Preparation;
    - no deserialization occurs here;
    - metadata is committed only after the binary exists;
    - binary creation is compensated if SQLite persistence fails.
    """

    contract = (
        MLTrainingContract
        .model_validate(
            training_contract
        )
    )


    _assert_preparation_authority(
        contract=
            contract
    )


    normalized_created_at = (
        _required_text(
            created_at_utc,
            field_name=
                "created_at_utc",
        )

        if created_at_utc
        is not None

        else _utc_now_iso()
    )


    model_id = (
        _new_server_model_id()
    )


    store_path = (
        resolve_ml_model_artifact_store_path()
    )


    binary_info = None


    with _STORE_LOCK:
        try:
            binary_info = (
                write_ml_model_binary(
                    store_path=
                        store_path,

                    model_id=
                        model_id,

                    model_bytes=
                        model_bytes,
                )
            )


            record = (
                MLModelArtifactRecord(
                    model_id=
                        model_id,

                    workflow_id=
                        contract.workflow_id,

                    dataset_id=
                        contract.dataset_id,

                    training_contract=
                        contract,

                    metrics=
                        metrics,

                    train_rows=
                        train_rows,

                    test_rows=
                        test_rows,

                    created_at_utc=
                        normalized_created_at,

                    **binary_info,
                )
            )


            upsert_ml_model_artifact_index_entry(
                store_path=
                    store_path,

                entry=
                    record.model_dump(
                        mode="json"
                    ),
            )


            return record


        except (
            MLModelArtifactAuthorityError,
            MLModelArtifactStoreError,
        ):
            raise


        except Exception as error:
            if (
                binary_info
                is not None
            ):
                try:
                    delete_ml_model_binary(
                        store_path=
                            store_path,

                        model_path=
                            str(
                                binary_info[
                                    "model_path"
                                ]
                            ),
                    )

                except Exception:
                    pass


            raise (
                MLModelArtifactStoreError(
                    (
                        "ML Model Artifact "
                        "registration failed."
                    )
                )
            ) from error


# ============================================================
# GET METADATA
# ============================================================


def get_ml_model_artifact(
    *,
    model_id: str,
    workflow_id: (
        str
        |
        None
    ) = None,
) -> MLModelArtifactRecord:

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


    store_path = (
        resolve_ml_model_artifact_store_path()
    )


    try:
        entry = (
            get_ml_model_artifact_index_entry(
                store_path=
                    store_path,

                model_id=
                    normalized_model_id,
            )
        )

    except MLModelArtifactIndexError as error:
        raise (
            MLModelArtifactStoreError(
                (
                    "ML Model Artifact "
                    "metadata lookup failed."
                )
            )
        ) from error


    if entry is None:
        raise (
            MLModelArtifactNotFoundError(
                (
                    "ML Model Artifact was not found. "
                    f"model_id={normalized_model_id}"
                )
            )
        )


    record = (
        _record_from_index_entry(
            entry
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
            MLModelArtifactWorkflowMismatchError(
                (
                    "ML Model Artifact does not belong "
                    "to the requested workflow. "
                    f"model_id={normalized_model_id}, "
                    "requested_workflow_id="
                    f"{normalized_workflow_id}, "
                    "artifact_workflow_id="
                    f"{record.workflow_id}"
                )
            )
        )


    return record


# ============================================================
# LOAD VERIFIED BINARY
# ============================================================


def load_ml_model_artifact_binary(
    *,
    model_id: str,
    workflow_id: (
        str
        |
        None
    ) = None,
) -> bytes:

    record = (
        get_ml_model_artifact(
            model_id=
                model_id,

            workflow_id=
                workflow_id,
        )
    )


    store_path = (
        resolve_ml_model_artifact_store_path()
    )


    try:
        return (
            read_ml_model_binary(
                store_path=
                    store_path,

                entry=
                    record.model_dump(
                        mode="json"
                    ),
            )
        )

    except MLModelArtifactDataPlaneError as error:
        raise (
            MLModelArtifactStoreError(
                (
                    "ML Model Artifact binary "
                    "verification failed. "
                    f"model_id={record.model_id}"
                )
            )
        ) from error


# ============================================================
# LIST WORKFLOW
# ============================================================


def list_ml_model_artifacts(
    *,
    workflow_id: str,
) -> list[
    MLModelArtifactRecord
]:

    normalized_workflow_id = (
        _required_text(
            workflow_id,
            field_name=
                "workflow_id",
        )
    )


    store_path = (
        resolve_ml_model_artifact_store_path()
    )


    try:
        entries = (
            load_ml_model_artifact_index_workflow(
                store_path=
                    store_path,

                workflow_id=
                    normalized_workflow_id,
            )
        )

    except MLModelArtifactIndexError as error:
        raise (
            MLModelArtifactStoreError(
                (
                    "ML Model Artifact workflow "
                    "listing failed."
                )
            )
        ) from error


    return [
        _record_from_index_entry(
            entry
        )

        for entry
        in entries
    ]