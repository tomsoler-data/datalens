from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Iterable, Literal
from uuid import uuid4

import pandas as pd


from app.persistence.sqlite_database import (
    ensure_ephemeral_sqlite_test_path,
)

from app.preparation.preparation_artifact_index import (
    PreparationArtifactIndexError,
    delete_preparation_artifact_index_scope,
    import_legacy_preparation_artifact_manifest_if_needed,
    load_preparation_artifact_index,
    replace_preparation_artifact_index,
)


# ============================================================
# VERSION
# ============================================================

PREPARATION_ARTIFACT_STORE_VERSION = (
    "preparation_artifact_store_v0.4"
)

PREPARATION_ARTIFACT_MANIFEST_VERSION = (
    "preparation_artifact_manifest_v0.2"
)

PREPARATION_ARTIFACT_STORE_ENV = (
    "DATALENS_PREPARATION_ARTIFACT_STORE_PATH"
)


# ============================================================
# TYPES
# ============================================================

PreparationArtifactStage = Literal[
    "source",
    "clean",
    "transform",
    "combine",
]


# ============================================================
# ERRORS
# ============================================================


class PreparationArtifactStoreError(
    RuntimeError
):
    pass


class PreparationArtifactWorkflowNotFoundError(
    PreparationArtifactStoreError
):
    pass


class PreparationArtifactDatasetNotFoundError(
    PreparationArtifactStoreError
):
    pass


# ============================================================
# ARTIFACT
# ============================================================


@dataclass(
    frozen=True,
)
class PreparationDatasetArtifact:
    """
    Server-owned materialized dataset produced during a
    Preparation workflow.

    DataFrames are never returned by reference from the store.
    """

    workflow_id: str

    dataset_id: str

    dataset_filename: str

    stage: PreparationArtifactStage

    dataframe: pd.DataFrame

    parent_dataset_ids: tuple[
        str,
        ...,
    ] = ()

    evidence_refs: tuple[
        str,
        ...,
    ] = ()


# ============================================================
# READ MODEL
# ============================================================


@dataclass(
    frozen=True,
)
class PreparationDatasetArtifactInfo:
    workflow_id: str

    dataset_id: str

    dataset_filename: str

    stage: PreparationArtifactStage

    rows: int

    columns: int

    parent_dataset_ids: tuple[
        str,
        ...,
    ]

    evidence_refs: tuple[
        str,
        ...,
    ]


# ============================================================
# NORMALIZATION
# ============================================================


def _required_text(
    value: str,
    *,
    field_name: str,
) -> str:
    normalized = value.strip()

    if not normalized:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    return normalized


def _normalize_ids(
    values: Iterable[
        str
    ],
) -> tuple[
    str,
    ...,
]:
    output: list[
        str
    ] = []

    seen: set[
        str
    ] = set()

    for raw_value in values:
        value = raw_value.strip()

        if not value:
            raise ValueError(
                "Artifact lineage dataset_id cannot be empty."
            )

        if value in seen:
            continue

        seen.add(
            value
        )

        output.append(
            value
        )

    return tuple(
        output
    )


def _normalize_refs(
    values: Iterable[
        str
    ],
) -> tuple[
    str,
    ...,
]:
    output: list[
        str
    ] = []

    seen: set[
        str
    ] = set()

    for raw_value in values:
        value = raw_value.strip()

        if not value:
            continue

        if value in seen:
            continue

        seen.add(
            value
        )

        output.append(
            value
        )

    return tuple(
        output
    )


# ============================================================
# DTYPE PERSISTENCE
# ============================================================


def _datetime_dtype_schema(
    dataframe: pd.DataFrame,
) -> list[
    dict[
        str,
        object,
    ]
]:
    """
    Persist exact datetime dtype resolutions.

    pandas JSON orient="table" preserves datetime semantics but
    normalizes datetime64 units to nanoseconds on some pandas /
    Python combinations. DataLens needs exact round-trip fidelity
    because downstream schema validation may distinguish ns/us/ms.

    Extension dtypes such as Int64 and boolean are already carried
    by the table schema, so this metadata is deliberately scoped to
    datetime-like columns.
    """

    output: list[
        dict[
            str,
            object,
        ]
    ] = []


    for (
        position,
        column_name,
    ) in enumerate(
        dataframe.columns
    ):
        dtype = (
            dataframe[
                column_name
            ].dtype
        )


        if (
            pd.api.types
            .is_datetime64_any_dtype(
                dtype
            )
        ):
            output.append(
                {
                    "position":
                        int(
                            position
                        ),

                    "dtype":
                        str(
                            dtype
                        ),
                }
            )


    return output


def _restore_datetime_dtypes(
    dataframe: pd.DataFrame,
    schema: object,
    *,
    workflow_id: object,
    dataset_id: object,
) -> pd.DataFrame:
    if not (
        schema
    ):
        return dataframe


    if not isinstance(
        schema,
        list,
    ):
        raise PreparationArtifactStoreError(
            (
                "Preparation artifact contains an invalid "
                "datetime dtype schema. "
                f"workflow_id={workflow_id}, "
                f"dataset_id={dataset_id}"
            )
        )


    output = dataframe.copy(
        deep=True
    )


    for item in (
        schema
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise PreparationArtifactStoreError(
                (
                    "Preparation artifact contains an invalid "
                    "datetime dtype entry. "
                    f"workflow_id={workflow_id}, "
                    f"dataset_id={dataset_id}"
                )
            )


        try:
            position = int(
                item[
                    "position"
                ]
            )

            dtype = str(
                item[
                    "dtype"
                ]
            )
        except Exception as exc:
            raise PreparationArtifactStoreError(
                (
                    "Preparation artifact datetime dtype metadata "
                    "is incomplete. "
                    f"workflow_id={workflow_id}, "
                    f"dataset_id={dataset_id}"
                )
            ) from exc


        if (
            position < 0
            or
            position >= len(
                output.columns
            )
        ):
            raise PreparationArtifactStoreError(
                (
                    "Preparation artifact datetime dtype position "
                    "is outside the restored DataFrame schema. "
                    f"workflow_id={workflow_id}, "
                    f"dataset_id={dataset_id}, "
                    f"position={position}"
                )
            )


        column_name = (
            output.columns[
                position
            ]
        )


        current_dtype = str(
            output[
                column_name
            ].dtype
        )


        if (
            current_dtype
            ==
            dtype
        ):
            continue


        try:
            output[
                column_name
            ] = (
                output[
                    column_name
                ]
                .astype(
                    dtype
                )
            )
        except Exception as exc:
            raise PreparationArtifactStoreError(
                (
                    "Preparation artifact datetime dtype could not "
                    "be restored exactly. "
                    f"workflow_id={workflow_id}, "
                    f"dataset_id={dataset_id}, "
                    f"column={column_name}, "
                    f"stored_dtype={dtype}, "
                    f"loaded_dtype={current_dtype}"
                )
            ) from exc


    return output


# ============================================================
# STORE PATH
# ============================================================


def default_preparation_artifact_store_path() -> Path:
    api_root = Path(
        __file__
    ).resolve().parents[
        2
    ]

    return (
        api_root
        /
        "var"
        /
        "preparation"
        /
        "artifacts"
    )


def preparation_artifact_store_path() -> Path:
    configured = os.getenv(
        PREPARATION_ARTIFACT_STORE_ENV,
        "",
    ).strip()

    if configured:
        return Path(
            configured
        ).expanduser().resolve()

    return default_preparation_artifact_store_path()


# ============================================================
# TEST-ONLY FILESYSTEM ISOLATION
# ============================================================


_EPHEMERAL_ARTIFACT_TEST_DIRECTORY = None


def _ensure_ephemeral_artifact_store_path_for_tests(
) -> Path:
    """
    Never let test reset() remove the production artifact root.

    Tests that already configure a non-production artifact path
    keep that path. Otherwise a process-local temporary root is
    installed through the existing environment contract.
    """

    global _EPHEMERAL_ARTIFACT_TEST_DIRECTORY


    configured = os.getenv(
        PREPARATION_ARTIFACT_STORE_ENV,
        "",
    ).strip()


    production_root = (
        default_preparation_artifact_store_path()
        .expanduser()
        .resolve()
    )


    if configured:
        candidate = (
            Path(
                configured
            )
            .expanduser()
            .resolve()
        )


        if (
            candidate
            !=
            production_root
        ):
            return candidate


    if (
        _EPHEMERAL_ARTIFACT_TEST_DIRECTORY
        is None
    ):
        _EPHEMERAL_ARTIFACT_TEST_DIRECTORY = (
            tempfile.TemporaryDirectory(
                prefix=
                    "datalens-preparation-artifact-tests-"
            )
        )


    test_root = (
        Path(
            _EPHEMERAL_ARTIFACT_TEST_DIRECTORY.name
        )
        /
        "artifacts"
    )


    os.environ[
        PREPARATION_ARTIFACT_STORE_ENV
    ] = str(
        test_root
    )


    return (
        test_root.resolve()
    )


# ============================================================
# STORE
# ============================================================


class PreparationArtifactStore:
    """
    Thread-safe local-first materialization store.

    PREPARATION_ARTIFACT_SQLITE_STORE_V0_1

    Metadata and lineage are indexed in the local DataLens
    SQLite control plane. DataFrames remain on the filesystem
    as gzip-compressed pandas JSON using orient="table".

    A legacy manifest.json is imported exactly once when a
    store root is first seen. After initialization, SQLite is
    authoritative and later changes to that legacy JSON file
    cannot overwrite the SQLite index.

    Security / integrity properties:
    - workflow_id is mandatory;
    - datasets are scoped by workflow;
    - callers never receive a shared internal DataFrame;
    - every write/read is isolated by a deep copy;
    - replacing an artifact is explicit;
    - lineage metadata remains server-owned;
    - data filenames are generated by the backend and never
      derived directly from browser-controlled paths;
    - metadata replacement is transactional in SQLite;
    - DataFrame payloads remain outside SQLite.

    The store remains single-writer. The RLock protects
    process-local filesystem operations while SQLite makes each
    metadata replacement atomic on disk.
    """

    def __init__(
        self,
    ) -> None:
        self._lock = RLock()

    @staticmethod
    def _empty_manifest() -> dict:
        return {
            "manifest_version": (
                PREPARATION_ARTIFACT_MANIFEST_VERSION
            ),
            "workflows": {},
        }

    @staticmethod
    def _manifest_path(
        root: Path,
    ) -> Path:
        return root / "manifest.json"

    @staticmethod
    def _data_directory(
        root: Path,
    ) -> Path:
        return root / "data"

    def _read_manifest(
        self,
        root: Path,
    ) -> dict:
        """
        Read the authoritative SQLite artifact index.

        On first access to a store root, import its legacy
        manifest.json exactly once.
        """

        root.mkdir(
            parents=True,
            exist_ok=True,
        )


        legacy_manifest_path = (
            self._manifest_path(
                root
            )
        )


        try:
            import_legacy_preparation_artifact_manifest_if_needed(
                root=
                    root,

                manifest_path=
                    legacy_manifest_path,

                fallback_manifest_version=
                    PREPARATION_ARTIFACT_MANIFEST_VERSION,
            )


            manifest = (
                load_preparation_artifact_index(
                    root=
                        root,

                    manifest_version=
                        PREPARATION_ARTIFACT_MANIFEST_VERSION,
                )
            )

        except PreparationArtifactIndexError as exc:
            raise PreparationArtifactStoreError(
                (
                    "Preparation artifact SQLite "
                    "index could not be read. "
                    f"root={root}"
                )
            ) from exc


        workflows = manifest.get(
            "workflows"
        )


        if not isinstance(
            workflows,
            dict,
        ):
            raise PreparationArtifactStoreError(
                (
                    "Preparation artifact SQLite "
                    "index returned an invalid "
                    "`workflows` object."
                )
            )


        return {
            "manifest_version":
                manifest.get(
                    "manifest_version",
                    PREPARATION_ARTIFACT_MANIFEST_VERSION,
                ),

            "workflows":
                workflows,
        }


    def _write_manifest(
        self,
        root: Path,
        manifest: dict,
    ) -> None:
        """
        Persist the complete metadata index transactionally
        through SQLite.

        The method name is retained deliberately so the mature
        artifact materialization state machine does not need to
        change during this persistence migration.
        """

        root.mkdir(
            parents=True,
            exist_ok=True,
        )


        try:
            replace_preparation_artifact_index(
                root=
                    root,

                manifest=
                    manifest,

                legacy_manifest_imported=
                    True,
            )

        except PreparationArtifactIndexError as exc:
            raise PreparationArtifactStoreError(
                (
                    "Preparation artifact SQLite "
                    "index could not be committed. "
                    f"root={root}"
                )
            ) from exc


    @staticmethod
    def _workflow_map(
        manifest: dict,
        workflow_id: str,
        *,
        create: bool,
    ) -> dict:
        workflows = manifest[
            "workflows"
        ]

        if create:
            return workflows.setdefault(
                workflow_id,
                {},
            )

        workflow_artifacts = workflows.get(
            workflow_id
        )

        if not isinstance(
            workflow_artifacts,
            dict,
        ):
            raise PreparationArtifactWorkflowNotFoundError(
                "Preparation artifact workflow was not found. "
                f"workflow_id={workflow_id}"
            )

        return workflow_artifacts

    @staticmethod
    def _entry_info(
        entry: dict,
    ) -> PreparationDatasetArtifactInfo:
        try:
            return PreparationDatasetArtifactInfo(
                workflow_id=str(
                    entry[
                        "workflow_id"
                    ]
                ),
                dataset_id=str(
                    entry[
                        "dataset_id"
                    ]
                ),
                dataset_filename=str(
                    entry[
                        "dataset_filename"
                    ]
                ),
                stage=entry[
                    "stage"
                ],
                rows=int(
                    entry[
                        "rows"
                    ]
                ),
                columns=int(
                    entry[
                        "columns"
                    ]
                ),
                parent_dataset_ids=tuple(
                    entry.get(
                        "parent_dataset_ids",
                        [],
                    )
                ),
                evidence_refs=tuple(
                    entry.get(
                        "evidence_refs",
                        [],
                    )
                ),
            )
        except Exception as exc:
            raise PreparationArtifactStoreError(
                "Preparation artifact manifest contains an invalid entry."
            ) from exc

    @staticmethod
    def _resolve_data_path(
        root: Path,
        entry: dict,
    ) -> Path:
        relative = entry.get(
            "data_path"
        )

        if not isinstance(
            relative,
            str,
        ) or not relative.strip():
            raise PreparationArtifactStoreError(
                "Preparation artifact entry has no valid data_path."
            )

        root_resolved = root.resolve()
        candidate = (
            root
            /
            relative
        ).resolve()

        try:
            candidate.relative_to(
                root_resolved
            )
        except ValueError as exc:
            raise PreparationArtifactStoreError(
                "Preparation artifact data_path escapes the configured "
                "artifact store root."
            ) from exc

        return candidate

    def _load_dataframe(
        self,
        *,
        root: Path,
        entry: dict,
    ) -> pd.DataFrame:
        path = self._resolve_data_path(
            root,
            entry,
        )

        if not path.exists():
            raise PreparationArtifactDatasetNotFoundError(
                "Preparation dataset artifact data file was not found. "
                f"workflow_id={entry.get('workflow_id')}, "
                f"dataset_id={entry.get('dataset_id')}"
            )

        try:
            dataframe = pd.read_json(
                path,
                orient="table",
                compression="gzip",
            )


            dataframe = (
                _restore_datetime_dtypes(
                    dataframe,
                    entry.get(
                        "datetime_dtypes"
                    ),
                    workflow_id=
                        entry.get(
                            "workflow_id"
                        ),
                    dataset_id=
                        entry.get(
                            "dataset_id"
                        ),
                )
            )
        except PreparationArtifactStoreError:
            raise

        except Exception as exc:
            raise PreparationArtifactStoreError(
                "Preparation dataset artifact could not be decoded. "
                f"workflow_id={entry.get('workflow_id')}, "
                f"dataset_id={entry.get('dataset_id')}"
            ) from exc

        expected_rows = int(
            entry[
                "rows"
            ]
        )

        expected_columns = int(
            entry[
                "columns"
            ]
        )

        actual_rows = int(
            dataframe.shape[
                0
            ]
        )

        actual_columns = int(
            dataframe.shape[
                1
            ]
        )

        if (
            actual_rows != expected_rows
            or actual_columns != expected_columns
        ):
            raise PreparationArtifactStoreError(
                "Preparation artifact shape does not match its server-owned "
                "manifest. "
                f"expected=({expected_rows}, {expected_columns}), "
                f"actual=({actual_rows}, {actual_columns})"
            )

        return dataframe

    # ========================================================
    # WRITE
    # ========================================================

    def put(
        self,
        *,
        workflow_id: str,
        dataset_id: str,
        dataset_filename: str,
        stage: PreparationArtifactStage,
        dataframe: pd.DataFrame,
        parent_dataset_ids: Iterable[
            str
        ] = (),
        evidence_refs: Iterable[
            str
        ] = (),
        replace: bool = True,
    ) -> PreparationDatasetArtifactInfo:
        normalized_workflow_id = _required_text(
            workflow_id,
            field_name="workflow_id",
        )

        normalized_dataset_id = _required_text(
            dataset_id,
            field_name="dataset_id",
        )

        normalized_filename = _required_text(
            dataset_filename,
            field_name="dataset_filename",
        )

        if stage not in {
            "source",
            "clean",
            "transform",
            "combine",
        }:
            raise ValueError(
                f"Unsupported Preparation artifact stage: {stage}"
            )

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "Preparation artifact dataframe must be a pandas DataFrame."
            )

        if dataframe.empty:
            raise ValueError(
                "Preparation artifact dataframe must contain at least one row."
            )

        parents = _normalize_ids(
            parent_dataset_ids
        )

        refs = _normalize_refs(
            evidence_refs
        )

        stored_dataframe = dataframe.copy(
            deep=True
        )

        rows = int(
            stored_dataframe.shape[
                0
            ]
        )

        columns = int(
            stored_dataframe.shape[
                1
            ]
        )

        root = preparation_artifact_store_path()

        with self._lock:
            manifest = self._read_manifest(
                root
            )

            workflow_artifacts = self._workflow_map(
                manifest,
                normalized_workflow_id,
                create=True,
            )

            previous = workflow_artifacts.get(
                normalized_dataset_id
            )

            if (
                previous is not None
                and not replace
            ):
                raise PreparationArtifactStoreError(
                    "Preparation artifact already exists. "
                    f"workflow_id={normalized_workflow_id}, "
                    f"dataset_id={normalized_dataset_id}"
                )

            data_directory = self._data_directory(
                root
            )

            data_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Dataset/workflow IDs are never used as filesystem
            # paths. A backend-generated filename prevents path
            # traversal and allows transactional replacement.
            identity_digest = hashlib.sha256(
                (
                    normalized_workflow_id
                    + "\x1f"
                    + normalized_dataset_id
                ).encode(
                    "utf-8"
                )
            ).hexdigest()[
                :12
            ]

            data_name = (
                f"artifact_{identity_digest}_{uuid4().hex}.json.gz"
            )

            data_path = data_directory / data_name

            temporary_data_path = data_directory / (
                f".{data_name}.{uuid4().hex}.tmp.gz"
            )

            try:
                stored_dataframe.to_json(
                    temporary_data_path,
                    orient="table",
                    date_format="iso",
                    force_ascii=False,
                    compression="gzip",
                    index=True,
                )

                os.replace(
                    temporary_data_path,
                    data_path,
                )

                entry = {
                    "workflow_id": normalized_workflow_id,
                    "dataset_id": normalized_dataset_id,
                    "dataset_filename": normalized_filename,
                    "stage": stage,
                    "rows": rows,
                    "columns": columns,
                    "parent_dataset_ids": list(
                        parents
                    ),
                    "evidence_refs": list(
                        refs
                    ),
                    "datetime_dtypes": (
                        _datetime_dtype_schema(
                            stored_dataframe
                        )
                    ),
                    "data_path": str(
                        data_path.relative_to(
                            root
                        )
                    ).replace(
                        "\\",
                        "/",
                    ),
                }

                workflow_artifacts[
                    normalized_dataset_id
                ] = entry

                self._write_manifest(
                    root,
                    manifest,
                )

            except Exception:
                data_path.unlink(
                    missing_ok=True
                )

                raise

            finally:
                temporary_data_path.unlink(
                    missing_ok=True
                )

            if isinstance(
                previous,
                dict,
            ):
                try:
                    old_path = self._resolve_data_path(
                        root,
                        previous,
                    )

                    if old_path != data_path:
                        old_path.unlink(
                            missing_ok=True
                        )
                except PreparationArtifactStoreError:
                    # A stale old data_path must not invalidate a
                    # successful replacement that is already committed.
                    pass

            return self._entry_info(
                entry
            )

    # ========================================================
    # READ
    # ========================================================

    def get(
        self,
        *,
        workflow_id: str,
        dataset_id: str,
    ) -> PreparationDatasetArtifact:
        normalized_workflow_id = _required_text(
            workflow_id,
            field_name="workflow_id",
        )

        normalized_dataset_id = _required_text(
            dataset_id,
            field_name="dataset_id",
        )

        root = preparation_artifact_store_path()

        with self._lock:
            manifest = self._read_manifest(
                root
            )

            workflow_artifacts = self._workflow_map(
                manifest,
                normalized_workflow_id,
                create=False,
            )

            entry = workflow_artifacts.get(
                normalized_dataset_id
            )

            if not isinstance(
                entry,
                dict,
            ):
                raise PreparationArtifactDatasetNotFoundError(
                    "Preparation dataset artifact was not found. "
                    f"workflow_id={normalized_workflow_id}, "
                    f"dataset_id={normalized_dataset_id}"
                )

            info = self._entry_info(
                entry
            )

            if (
                info.workflow_id != normalized_workflow_id
                or info.dataset_id != normalized_dataset_id
            ):
                raise PreparationArtifactStoreError(
                    "Preparation artifact manifest identity mismatch."
                )

            dataframe = self._load_dataframe(
                root=root,
                entry=entry,
            )

            return PreparationDatasetArtifact(
                workflow_id=info.workflow_id,
                dataset_id=info.dataset_id,
                dataset_filename=info.dataset_filename,
                stage=info.stage,
                dataframe=dataframe.copy(
                    deep=True
                ),
                parent_dataset_ids=info.parent_dataset_ids,
                evidence_refs=info.evidence_refs,
            )

    # ========================================================
    # GET DATAFRAME
    # ========================================================

    def get_dataframe(
        self,
        *,
        workflow_id: str,
        dataset_id: str,
    ) -> pd.DataFrame:
        return self.get(
            workflow_id=workflow_id,
            dataset_id=dataset_id,
        ).dataframe

    # ========================================================
    # LIST
    # ========================================================

    def list(
        self,
        *,
        workflow_id: str,
    ) -> list[
        PreparationDatasetArtifactInfo
    ]:
        normalized_workflow_id = _required_text(
            workflow_id,
            field_name="workflow_id",
        )

        root = preparation_artifact_store_path()

        with self._lock:
            manifest = self._read_manifest(
                root
            )

            workflows = manifest[
                "workflows"
            ]

            workflow_artifacts = workflows.get(
                normalized_workflow_id,
                {},
            )

            if not isinstance(
                workflow_artifacts,
                dict,
            ):
                raise PreparationArtifactStoreError(
                    "Preparation artifact workflow manifest is invalid."
                )

            return [
                self._entry_info(
                    entry
                )
                for entry in workflow_artifacts.values()
                if isinstance(
                    entry,
                    dict,
                )
            ]

    # ========================================================
    # MATERIALIZED DATASET MAP
    # ========================================================

    def dataframe_map(
        self,
        *,
        workflow_id: str,
        dataset_ids: Iterable[
            str
        ] | None = None,
    ) -> dict[
        str,
        pd.DataFrame
    ]:
        normalized_workflow_id = _required_text(
            workflow_id,
            field_name="workflow_id",
        )

        root = preparation_artifact_store_path()

        with self._lock:
            manifest = self._read_manifest(
                root
            )

            workflow_artifacts = self._workflow_map(
                manifest,
                normalized_workflow_id,
                create=False,
            )

            if dataset_ids is None:
                requested_ids = list(
                    workflow_artifacts.keys()
                )
            else:
                requested_ids = list(
                    _normalize_ids(
                        dataset_ids
                    )
                )

            result: dict[
                str,
                pd.DataFrame
            ] = {}

            for dataset_id in requested_ids:
                entry = workflow_artifacts.get(
                    dataset_id
                )

                if not isinstance(
                    entry,
                    dict,
                ):
                    raise PreparationArtifactDatasetNotFoundError(
                        "Preparation dataset artifact was not found. "
                        f"workflow_id={normalized_workflow_id}, "
                        f"dataset_id={dataset_id}"
                    )

                result[
                    dataset_id
                ] = self._load_dataframe(
                    root=root,
                    entry=entry,
                ).copy(
                    deep=True
                )

            return result

    # ========================================================
    # DELETE WORKFLOW
    # ========================================================

    def delete_workflow(
        self,
        *,
        workflow_id: str,
    ) -> None:
        normalized_workflow_id = _required_text(
            workflow_id,
            field_name="workflow_id",
        )

        root = preparation_artifact_store_path()

        with self._lock:
            manifest = self._read_manifest(
                root
            )

            workflow_artifacts = manifest[
                "workflows"
            ].pop(
                normalized_workflow_id,
                None,
            )

            if workflow_artifacts is None:
                return

            self._write_manifest(
                root,
                manifest,
            )

            if isinstance(
                workflow_artifacts,
                dict,
            ):
                for entry in workflow_artifacts.values():
                    if not isinstance(
                        entry,
                        dict,
                    ):
                        continue

                    try:
                        self._resolve_data_path(
                            root,
                            entry,
                        ).unlink(
                            missing_ok=True
                        )
                    except PreparationArtifactStoreError:
                        continue

    # ========================================================
    # RESET — TESTS ONLY
    # ========================================================

    def reset(
        self,
    ) -> None:
        """
        Test-only complete store reset.

        The production artifact filesystem root and production
        SQLite database are never deleted by this helper.
        """

        with self._lock:
            root = (
                _ensure_ephemeral_artifact_store_path_for_tests()
            )


            ensure_ephemeral_sqlite_test_path(
                namespace=
                    "preparation-artifact-tests"
            )


            delete_preparation_artifact_index_scope(
                root=
                    root
            )


            if root.exists():
                shutil.rmtree(
                    root
                )


# ============================================================
# GLOBAL STORE
# ============================================================

_ARTIFACT_STORE = PreparationArtifactStore()


# ============================================================
# PUBLIC HELPERS
# ============================================================


def put_preparation_artifact(
    *,
    workflow_id: str,
    dataset_id: str,
    dataset_filename: str,
    stage: PreparationArtifactStage,
    dataframe: pd.DataFrame,
    parent_dataset_ids: Iterable[
        str
    ] = (),
    evidence_refs: Iterable[
        str
    ] = (),
    replace: bool = True,
) -> PreparationDatasetArtifactInfo:
    return _ARTIFACT_STORE.put(
        workflow_id=workflow_id,
        dataset_id=dataset_id,
        dataset_filename=dataset_filename,
        stage=stage,
        dataframe=dataframe,
        parent_dataset_ids=parent_dataset_ids,
        evidence_refs=evidence_refs,
        replace=replace,
    )


def get_preparation_artifact(
    *,
    workflow_id: str,
    dataset_id: str,
) -> PreparationDatasetArtifact:
    return _ARTIFACT_STORE.get(
        workflow_id=workflow_id,
        dataset_id=dataset_id,
    )


def get_preparation_dataframe(
    *,
    workflow_id: str,
    dataset_id: str,
) -> pd.DataFrame:
    return _ARTIFACT_STORE.get_dataframe(
        workflow_id=workflow_id,
        dataset_id=dataset_id,
    )


def list_preparation_artifacts(
    *,
    workflow_id: str,
) -> list[
    PreparationDatasetArtifactInfo
]:
    return _ARTIFACT_STORE.list(
        workflow_id=workflow_id
    )


def get_preparation_dataframe_map(
    *,
    workflow_id: str,
    dataset_ids: Iterable[
        str
    ] | None = None,
) -> dict[
    str,
    pd.DataFrame
]:
    return _ARTIFACT_STORE.dataframe_map(
        workflow_id=workflow_id,
        dataset_ids=dataset_ids,
    )


def delete_preparation_artifacts(
    *,
    workflow_id: str,
) -> None:
    _ARTIFACT_STORE.delete_workflow(
        workflow_id=workflow_id
    )


def reset_preparation_artifact_store_for_tests(
) -> None:
    """
    Delete the currently configured artifact test store.

    Tests that need isolation should set
    DATALENS_PREPARATION_ARTIFACT_STORE_PATH to a temporary
    directory before calling this helper.
    """

    _ARTIFACT_STORE.reset()
