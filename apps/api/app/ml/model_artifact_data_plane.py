from __future__ import annotations


import hashlib
import os
import uuid


from pathlib import (
    Path,
)


from typing import (
    Any,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_ARTIFACT_DATA_PLANE_VERSION = (
    "ml_model_artifact_data_plane_v0.1"
)


# ============================================================
# ERROR
# ============================================================


class MLModelArtifactDataPlaneError(
    RuntimeError
):
    pass


# ============================================================
# PATHS
# ============================================================


def ml_model_artifact_data_root(
    store_path: Path,
) -> Path:
    """
    Logical store:

        .../ml/model_artifacts.json

    Filesystem data plane:

        .../ml/model_artifacts/data/*.joblib

    The logical JSON path exists only as the scope identity.
    SQLite will own metadata in the next implementation step.
    """

    resolved_store = (
        store_path
        .expanduser()
        .resolve()
    )


    return (
        resolved_store.parent
        /
        resolved_store.stem
    ).resolve()


def _resolve_model_file(
    *,
    store_path: Path,
    model_path: str,
) -> Path:

    root = (
        ml_model_artifact_data_root(
            store_path
        )
    )


    normalized = str(
        model_path
        or
        ""
    ).strip()


    if not normalized:
        raise (
            MLModelArtifactDataPlaneError(
                "model_path cannot be empty."
            )
        )


    normalized = (
        normalized
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
            MLModelArtifactDataPlaneError(
                (
                    "Model Artifact model_path "
                    "must remain relative to the "
                    "configured data-plane root."
                )
            )
        )


    candidate = (
        root
        /
        relative
    ).resolve()


    try:
        candidate.relative_to(
            root
        )

    except ValueError as error:
        raise (
            MLModelArtifactDataPlaneError(
                (
                    "Model Artifact model_path "
                    "escapes the configured "
                    "data-plane root."
                )
            )
        ) from error


    return candidate


# ============================================================
# WRITE
# ============================================================


def write_ml_model_binary(
    *,
    store_path: Path,
    model_id: str,
    model_bytes: bytes,
) -> dict[
    str,
    Any,
]:

    normalized_model_id = str(
        model_id
    ).strip()


    if not normalized_model_id:
        raise (
            MLModelArtifactDataPlaneError(
                "model_id cannot be empty."
            )
        )


    if not isinstance(
        model_bytes,
        bytes,
    ):
        raise (
            MLModelArtifactDataPlaneError(
                (
                    "model_bytes must be "
                    "a bytes object."
                )
            )
        )


    if not model_bytes:
        raise (
            MLModelArtifactDataPlaneError(
                (
                    "model_bytes cannot "
                    "be empty."
                )
            )
        )


    model_sha256 = (
        hashlib.sha256(
            model_bytes
        )
        .hexdigest()
    )


    identity_digest = (
        hashlib.sha256(
            normalized_model_id.encode(
                "utf-8"
            )
        )
        .hexdigest()[
            :16
        ]
    )


    filename = (
        "model_"
        +
        identity_digest
        +
        "_"
        +
        uuid.uuid4().hex
        +
        ".joblib"
    )


    relative_path = (
        Path(
            "data"
        )
        /
        filename
    )


    final_path = (
        _resolve_model_file(
            store_path=
                store_path,

            model_path=
                relative_path.as_posix(),
        )
    )


    final_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    temporary = (
        final_path
        .with_name(
            final_path.name
            +
            ".tmp-"
            +
            uuid.uuid4().hex
        )
    )


    try:
        temporary.write_bytes(
            model_bytes
        )


        os.replace(
            temporary,
            final_path,
        )

    finally:
        if temporary.exists():
            temporary.unlink(
                missing_ok=True
            )


    return {
        "model_path":
            relative_path.as_posix(),

        "model_file_bytes":
            len(
                model_bytes
            ),

        "model_sha256":
            model_sha256,
    }


# ============================================================
# READ
# ============================================================


def read_ml_model_binary(
    *,
    store_path: Path,
    entry: dict[
        str,
        Any,
    ],
) -> bytes:

    if not isinstance(
        entry,
        dict,
    ):
        raise (
            MLModelArtifactDataPlaneError(
                (
                    "Model Artifact metadata "
                    "entry must be an object."
                )
            )
        )


    model_path = str(
        entry.get(
            "model_path",
            "",
        )
    ).strip()


    path = (
        _resolve_model_file(
            store_path=
                store_path,

            model_path=
                model_path,
        )
    )


    if not path.is_file():
        raise (
            MLModelArtifactDataPlaneError(
                (
                    "Model Artifact binary "
                    "file is missing: "
                    f"{model_path}"
                )
            )
        )


    try:
        expected_file_bytes = int(
            entry.get(
                "model_file_bytes",
                -1,
            )
        )

    except Exception as error:
        raise (
            MLModelArtifactDataPlaneError(
                (
                    "model_file_bytes must "
                    "be an integer."
                )
            )
        ) from error


    if expected_file_bytes <= 0:
        raise (
            MLModelArtifactDataPlaneError(
                (
                    "model_file_bytes must "
                    "be positive."
                )
            )
        )


    actual_file_bytes = (
        path.stat().st_size
    )


    if (
        actual_file_bytes
        !=
        expected_file_bytes
    ):
        raise (
            MLModelArtifactDataPlaneError(
                (
                    "Model Artifact binary "
                    "size mismatch."
                )
            )
        )


    model_bytes = (
        path.read_bytes()
    )


    actual_sha256 = (
        hashlib.sha256(
            model_bytes
        )
        .hexdigest()
    )


    expected_sha256 = str(
        entry.get(
            "model_sha256",
            "",
        )
    ).strip().lower()


    if (
        actual_sha256
        !=
        expected_sha256
    ):
        raise (
            MLModelArtifactDataPlaneError(
                (
                    "Model Artifact binary "
                    "SHA-256 mismatch."
                )
            )
        )


    return model_bytes


# ============================================================
# DELETE
# ============================================================


def delete_ml_model_binary(
    *,
    store_path: Path,
    model_path: str,
) -> None:

    path = (
        _resolve_model_file(
            store_path=
                store_path,

            model_path=
                model_path,
        )
    )


    path.unlink(
        missing_ok=True
    )