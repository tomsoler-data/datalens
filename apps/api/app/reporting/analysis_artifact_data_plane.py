from __future__ import annotations


import gzip
import hashlib
import json
import os
import uuid

from pathlib import (
    Path,
)

from threading import (
    RLock,
)

from typing import (
    Any,
)


from app.reporting.analysis_artifact_index import (
    AnalysisArtifactIndexError,
    analysis_artifact_index_is_initialized,
    replace_analysis_artifact_index_scope,
)


# ========================================================
# VERSION
# ========================================================


ANALYSIS_ARTIFACT_DATA_PLANE_VERSION = (
    "analysis_artifact_data_plane_v0.1"
)


_MIGRATION_LOCK = (
    RLock()
)


# ========================================================
# ERROR
# ========================================================


class AnalysisArtifactDataPlaneError(
    RuntimeError
):
    pass


# ========================================================
# PATHS
# ========================================================


def analysis_artifact_data_root(
    store_path: Path,
) -> Path:
    """
    Legacy:
        .../reporting/analysis_artifacts.json

    Data plane:
        .../reporting/analysis_artifacts/data/*.json.gz
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


def _resolve_payload_file(
    *,
    store_path: Path,
    payload_path: str,
) -> Path:
    root = (
        analysis_artifact_data_root(
            store_path
        )
    )


    candidate = (
        root
        /
        str(
            payload_path
        )
    ).resolve()


    if (
        candidate
        !=
        root
        and
        root
        not in candidate.parents
    ):
        raise AnalysisArtifactDataPlaneError(
            (
                "AnalysisArtifact payload_path escapes "
                "the configured data-plane root."
            )
        )


    return candidate


# ========================================================
# CANONICAL JSON
# ========================================================


def canonical_pipeline_payload_bytes(
    payload: dict[
        str,
        Any,
    ],
) -> bytes:
    if not isinstance(
        payload,
        dict,
    ):
        raise AnalysisArtifactDataPlaneError(
            (
                "pipeline_payload must be "
                "a JSON object."
            )
        )


    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ).encode(
            "utf-8"
        )

    except Exception as error:
        raise AnalysisArtifactDataPlaneError(
            (
                "pipeline_payload cannot be "
                "serialized as JSON."
            )
        ) from error


# ========================================================
# WRITE ONE PAYLOAD
# ========================================================


def write_analysis_artifact_payload(
    *,
    store_path: Path,
    analysis_id: str,
    pipeline_payload: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    normalized_analysis_id = str(
        analysis_id
    ).strip()


    if not normalized_analysis_id:
        raise AnalysisArtifactDataPlaneError(
            "analysis_id cannot be empty."
        )


    raw_bytes = (
        canonical_pipeline_payload_bytes(
            pipeline_payload
        )
    )


    payload_sha256 = (
        hashlib.sha256(
            raw_bytes
        )
        .hexdigest()
    )


    compressed = gzip.compress(
        raw_bytes,
        compresslevel=6,
        mtime=0,
    )


    identity_digest = (
        hashlib.sha256(
            normalized_analysis_id.encode(
                "utf-8"
            )
        )
        .hexdigest()[
            :16
        ]
    )


    filename = (
        "artifact_"
        +
        identity_digest
        +
        "_"
        +
        uuid.uuid4().hex
        +
        ".json.gz"
    )


    relative_path = (
        Path(
            "data"
        )
        /
        filename
    )


    final_path = (
        _resolve_payload_file(
            store_path=
                store_path,

            payload_path=
                relative_path.as_posix(),
        )
    )


    final_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    temporary = final_path.with_name(
        final_path.name
        +
        ".tmp-"
        +
        uuid.uuid4().hex
    )


    try:
        temporary.write_bytes(
            compressed
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
        "payload_path":
            relative_path.as_posix(),

        "payload_json_bytes":
            len(
                raw_bytes
            ),

        "payload_file_bytes":
            len(
                compressed
            ),

        "payload_sha256":
            payload_sha256,
    }


# ========================================================
# READ ONE PAYLOAD
# ========================================================


def read_analysis_artifact_payload(
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
    payload_path = str(
        entry.get(
            "payload_path",
            "",
        )
    ).strip()


    path = (
        _resolve_payload_file(
            store_path=
                store_path,

            payload_path=
                payload_path,
        )
    )


    if not path.exists():
        raise AnalysisArtifactDataPlaneError(
            (
                "AnalysisArtifact payload file "
                f"is missing: {payload_path}"
            )
        )


    actual_file_bytes = (
        path.stat().st_size
    )


    expected_file_bytes = int(
        entry.get(
            "payload_file_bytes",
            -1,
        )
    )


    if (
        actual_file_bytes
        !=
        expected_file_bytes
    ):
        raise AnalysisArtifactDataPlaneError(
            (
                "AnalysisArtifact compressed payload "
                "size mismatch."
            )
        )


    try:
        raw_bytes = gzip.decompress(
            path.read_bytes()
        )

    except Exception as error:
        raise AnalysisArtifactDataPlaneError(
            (
                "AnalysisArtifact payload gzip "
                "could not be decoded."
            )
        ) from error


    expected_json_bytes = int(
        entry.get(
            "payload_json_bytes",
            -1,
        )
    )


    if (
        len(
            raw_bytes
        )
        !=
        expected_json_bytes
    ):
        raise AnalysisArtifactDataPlaneError(
            (
                "AnalysisArtifact JSON payload "
                "size mismatch."
            )
        )


    digest = (
        hashlib.sha256(
            raw_bytes
        )
        .hexdigest()
    )


    if (
        digest
        !=
        str(
            entry.get(
                "payload_sha256",
                "",
            )
        )
    ):
        raise AnalysisArtifactDataPlaneError(
            (
                "AnalysisArtifact payload "
                "SHA-256 mismatch."
            )
        )


    try:
        payload = json.loads(
            raw_bytes.decode(
                "utf-8"
            )
        )

    except Exception as error:
        raise AnalysisArtifactDataPlaneError(
            (
                "AnalysisArtifact payload JSON "
                "could not be decoded."
            )
        ) from error


    if not isinstance(
        payload,
        dict,
    ):
        raise AnalysisArtifactDataPlaneError(
            (
                "AnalysisArtifact pipeline payload "
                "must decode to an object."
            )
        )


    return payload


# ========================================================
# DELETE ONE PAYLOAD
# ========================================================


def delete_analysis_artifact_payload(
    *,
    store_path: Path,
    payload_path: str,
) -> None:
    path = (
        _resolve_payload_file(
            store_path=
                store_path,

            payload_path=
                payload_path,
        )
    )


    path.unlink(
        missing_ok=True
    )


# ========================================================
# LEGACY JSON IMPORT
# ========================================================


def import_legacy_analysis_artifacts_if_needed(
    *,
    store_path: Path,
    fallback_rule_version: str,
) -> bool:
    """
    Materialize the legacy JSON store exactly once:

        metadata -> SQLite
        pipeline_payload -> one gzip file per analysis

    The legacy JSON itself is never modified.
    """

    with _MIGRATION_LOCK:
        if (
            analysis_artifact_index_is_initialized(
                store_path=
                    store_path
            )
        ):
            return False


        if not store_path.exists():
            replace_analysis_artifact_index_scope(
                store_path=
                    store_path,

                entries=[],

                legacy_json_imported=
                    True,

                legacy_rule_version=
                    fallback_rule_version,
            )

            return True


        try:
            root = json.loads(
                store_path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception as error:
            raise AnalysisArtifactDataPlaneError(
                (
                    "Legacy analysis_artifacts.json "
                    "could not be read."
                )
            ) from error


        if not isinstance(
            root,
            dict,
        ):
            raise AnalysisArtifactDataPlaneError(
                (
                    "Legacy AnalysisArtifact root "
                    "must be an object."
                )
            )


        artifacts = root.get(
            "artifacts"
        )


        if not isinstance(
            artifacts,
            dict,
        ):
            raise AnalysisArtifactDataPlaneError(
                (
                    "Legacy AnalysisArtifact store "
                    "has no artifact map."
                )
            )


        legacy_rule_version = str(
            root.get(
                "rule_version",
                fallback_rule_version,
            )
        )


        entries = []

        created_payload_paths = []


        try:
            for (
                map_analysis_id,
                raw,
            ) in artifacts.items():

                if not isinstance(
                    raw,
                    dict,
                ):
                    raise AnalysisArtifactDataPlaneError(
                        (
                            "Legacy AnalysisArtifact "
                            "record must be an object."
                        )
                    )


                analysis_id = str(
                    raw.get(
                        "analysis_id",
                        "",
                    )
                ).strip()


                if (
                    analysis_id
                    !=
                    str(
                        map_analysis_id
                    )
                ):
                    raise AnalysisArtifactDataPlaneError(
                        (
                            "Legacy map key does not "
                            "match analysis_id."
                        )
                    )


                pipeline_payload = raw.get(
                    "pipeline_payload"
                )


                if not isinstance(
                    pipeline_payload,
                    dict,
                ):
                    raise AnalysisArtifactDataPlaneError(
                        (
                            "Legacy pipeline_payload "
                            "must be an object."
                        )
                    )


                payload_info = (
                    write_analysis_artifact_payload(
                        store_path=
                            store_path,

                        analysis_id=
                            analysis_id,

                        pipeline_payload=
                            pipeline_payload,
                    )
                )


                created_payload_paths.append(
                    payload_info[
                        "payload_path"
                    ]
                )


                entry = {
                    "analysis_id":
                        analysis_id,

                    "workflow_id":
                        str(
                            raw.get(
                                "workflow_id",
                                "",
                            )
                        ),

                    "trace_id":
                        str(
                            raw.get(
                                "trace_id",
                                "",
                            )
                        ),

                    "source_type":
                        str(
                            raw.get(
                                "source_type",
                                "",
                            )
                        ),

                    "objective":
                        str(
                            raw.get(
                                "objective",
                                "",
                            )
                        ),

                    "executed":
                        bool(
                            raw.get(
                                "executed",
                                False,
                            )
                        ),

                    "executed_count":
                        int(
                            raw.get(
                                "executed_count",
                                0,
                            )
                            or
                            0
                        ),

                    "created_at_utc":
                        str(
                            raw.get(
                                "created_at_utc",
                                "",
                            )
                        ),

                    "rule_version":
                        str(
                            raw.get(
                                "rule_version",
                                legacy_rule_version,
                            )
                        ),

                    **payload_info,
                }


                entries.append(
                    entry
                )


            replace_analysis_artifact_index_scope(
                store_path=
                    store_path,

                entries=
                    entries,

                legacy_json_imported=
                    True,

                legacy_rule_version=
                    legacy_rule_version,
            )


        except (
            AnalysisArtifactIndexError,
            AnalysisArtifactDataPlaneError,
            Exception,
        ):
            for payload_path in (
                created_payload_paths
            ):
                try:
                    delete_analysis_artifact_payload(
                        store_path=
                            store_path,

                        payload_path=
                            payload_path,
                    )

                except Exception:
                    pass

            raise


        return True
