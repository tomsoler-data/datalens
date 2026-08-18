from __future__ import annotations

from dataclasses import (
    dataclass,
)

from threading import (
    RLock,
)

from typing import (
    Dict,
    Iterable,
    Literal,
)

import pandas as pd


# ============================================================
# VERSION
# ============================================================

PREPARATION_ARTIFACT_STORE_VERSION = (
    "preparation_artifact_store_v0.1"
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
    The store writes and reads deep copies.
    """

    workflow_id: str

    dataset_id: str

    dataset_filename: str

    stage: PreparationArtifactStage

    dataframe: pd.DataFrame

    parent_dataset_ids: tuple[
        str,
        ...
    ] = ()

    evidence_refs: tuple[
        str,
        ...
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
        ...
    ]

    evidence_refs: tuple[
        str,
        ...
    ]


# ============================================================
# NORMALIZATION
# ============================================================

def _required_text(
    value: str,
    *,
    field_name: str,
) -> str:

    normalized = (
        value.strip()
    )


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
    ...
]:

    output: list[
        str
    ] = []

    seen: set[
        str
    ] = set()


    for raw_value in values:

        value = (
            raw_value.strip()
        )


        if not value:

            raise ValueError(
                "Artifact lineage dataset_id "
                "cannot be empty."
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
    ...
]:

    output: list[
        str
    ] = []

    seen: set[
        str
    ] = set()


    for raw_value in values:

        value = (
            raw_value.strip()
        )


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
# STORE
# ============================================================

class PreparationArtifactStore:
    """
    Thread-safe in-memory materialization store.

    This deliberately mirrors PreparationSession v0.1's
    current in-memory lifetime.

    It is NOT durable storage.

    Security / integrity properties:
    - workflow_id is mandatory;
    - datasets are scoped by workflow;
    - callers never receive the internal DataFrame object;
    - every write stores a deep copy;
    - every read returns a deep copy;
    - replacing an artifact is explicit;
    - lineage metadata is server-owned.
    """

    def __init__(
        self,
    ) -> None:

        self._lock = (
            RLock()
        )


        self._artifacts: Dict[
            str,
            Dict[
                str,
                PreparationDatasetArtifact,
            ],
        ] = {}


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

        normalized_workflow_id = (
            _required_text(
                workflow_id,
                field_name="workflow_id",
            )
        )


        normalized_dataset_id = (
            _required_text(
                dataset_id,
                field_name="dataset_id",
            )
        )


        normalized_filename = (
            _required_text(
                dataset_filename,
                field_name="dataset_filename",
            )
        )


        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):

            raise TypeError(
                "Preparation artifact dataframe must "
                "be a pandas DataFrame."
            )


        if dataframe.empty:

            raise ValueError(
                "Preparation artifact dataframe must "
                "contain at least one row."
            )


        parents = (
            _normalize_ids(
                parent_dataset_ids
            )
        )


        refs = (
            _normalize_refs(
                evidence_refs
            )
        )


        stored_dataframe = (
            dataframe.copy(
                deep=True
            )
        )


        artifact = (
            PreparationDatasetArtifact(
                workflow_id=(
                    normalized_workflow_id
                ),

                dataset_id=(
                    normalized_dataset_id
                ),

                dataset_filename=(
                    normalized_filename
                ),

                stage=(
                    stage
                ),

                dataframe=(
                    stored_dataframe
                ),

                parent_dataset_ids=(
                    parents
                ),

                evidence_refs=(
                    refs
                ),
            )
        )


        with self._lock:

            workflow_artifacts = (
                self._artifacts.setdefault(
                    normalized_workflow_id,
                    {},
                )
            )


            if (
                not replace
                and normalized_dataset_id
                in workflow_artifacts
            ):

                raise PreparationArtifactStoreError(
                    "Preparation artifact already exists. "
                    f"workflow_id={normalized_workflow_id}, "
                    f"dataset_id={normalized_dataset_id}"
                )


            workflow_artifacts[
                normalized_dataset_id
            ] = (
                artifact
            )


        return (
            self._info(
                artifact
            )
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

        normalized_workflow_id = (
            _required_text(
                workflow_id,
                field_name="workflow_id",
            )
        )


        normalized_dataset_id = (
            _required_text(
                dataset_id,
                field_name="dataset_id",
            )
        )


        with self._lock:

            workflow_artifacts = (
                self._artifacts.get(
                    normalized_workflow_id
                )
            )


            if workflow_artifacts is None:

                raise (
                    PreparationArtifactWorkflowNotFoundError(
                        "Preparation artifact workflow "
                        "was not found. "
                        f"workflow_id="
                        f"{normalized_workflow_id}"
                    )
                )


            artifact = (
                workflow_artifacts.get(
                    normalized_dataset_id
                )
            )


            if artifact is None:

                raise (
                    PreparationArtifactDatasetNotFoundError(
                        "Preparation dataset artifact "
                        "was not found. "
                        f"workflow_id="
                        f"{normalized_workflow_id}, "
                        f"dataset_id="
                        f"{normalized_dataset_id}"
                    )
                )


            return (
                PreparationDatasetArtifact(
                    workflow_id=(
                        artifact.workflow_id
                    ),

                    dataset_id=(
                        artifact.dataset_id
                    ),

                    dataset_filename=(
                        artifact.dataset_filename
                    ),

                    stage=(
                        artifact.stage
                    ),

                    dataframe=(
                        artifact
                        .dataframe
                        .copy(
                            deep=True
                        )
                    ),

                    parent_dataset_ids=(
                        artifact
                        .parent_dataset_ids
                    ),

                    evidence_refs=(
                        artifact
                        .evidence_refs
                    ),
                )
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

        return (
            self.get(
                workflow_id=(
                    workflow_id
                ),

                dataset_id=(
                    dataset_id
                ),
            )
            .dataframe
        )


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

        normalized_workflow_id = (
            _required_text(
                workflow_id,
                field_name="workflow_id",
            )
        )


        with self._lock:

            workflow_artifacts = (
                self._artifacts.get(
                    normalized_workflow_id,
                    {},
                )
            )


            return [
                self._info(
                    artifact
                )

                for artifact
                in workflow_artifacts.values()
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
        ]
        | None = None,
    ) -> dict[
        str,
        pd.DataFrame
    ]:

        normalized_workflow_id = (
            _required_text(
                workflow_id,
                field_name="workflow_id",
            )
        )


        with self._lock:

            workflow_artifacts = (
                self._artifacts.get(
                    normalized_workflow_id
                )
            )


            if workflow_artifacts is None:

                raise (
                    PreparationArtifactWorkflowNotFoundError(
                        "Preparation artifact workflow "
                        "was not found. "
                        f"workflow_id="
                        f"{normalized_workflow_id}"
                    )
                )


            if dataset_ids is None:

                requested_ids = (
                    list(
                        workflow_artifacts.keys()
                    )
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

                artifact = (
                    workflow_artifacts.get(
                        dataset_id
                    )
                )


                if artifact is None:

                    raise (
                        PreparationArtifactDatasetNotFoundError(
                            "Preparation dataset artifact "
                            "was not found. "
                            f"workflow_id="
                            f"{normalized_workflow_id}, "
                            f"dataset_id="
                            f"{dataset_id}"
                        )
                    )


                result[
                    dataset_id
                ] = (
                    artifact
                    .dataframe
                    .copy(
                        deep=True
                    )
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

        normalized_workflow_id = (
            _required_text(
                workflow_id,
                field_name="workflow_id",
            )
        )


        with self._lock:

            self._artifacts.pop(
                normalized_workflow_id,
                None,
            )


    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
    ) -> None:

        with self._lock:

            self._artifacts.clear()


    # ========================================================
    # INFO
    # ========================================================

    @staticmethod
    def _info(
        artifact: PreparationDatasetArtifact,
    ) -> PreparationDatasetArtifactInfo:

        rows = int(
            artifact
            .dataframe
            .shape[
                0
            ]
        )


        columns = int(
            artifact
            .dataframe
            .shape[
                1
            ]
        )


        return (
            PreparationDatasetArtifactInfo(
                workflow_id=(
                    artifact.workflow_id
                ),

                dataset_id=(
                    artifact.dataset_id
                ),

                dataset_filename=(
                    artifact.dataset_filename
                ),

                stage=(
                    artifact.stage
                ),

                rows=(
                    rows
                ),

                columns=(
                    columns
                ),

                parent_dataset_ids=(
                    artifact
                    .parent_dataset_ids
                ),

                evidence_refs=(
                    artifact
                    .evidence_refs
                ),
            )
        )


# ============================================================
# GLOBAL PROCESS-LOCAL STORE
# ============================================================

_ARTIFACT_STORE = (
    PreparationArtifactStore()
)


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

    return (
        _ARTIFACT_STORE.put(
            workflow_id=(
                workflow_id
            ),

            dataset_id=(
                dataset_id
            ),

            dataset_filename=(
                dataset_filename
            ),

            stage=(
                stage
            ),

            dataframe=(
                dataframe
            ),

            parent_dataset_ids=(
                parent_dataset_ids
            ),

            evidence_refs=(
                evidence_refs
            ),

            replace=(
                replace
            ),
        )
    )


def get_preparation_artifact(
    *,
    workflow_id: str,
    dataset_id: str,
) -> PreparationDatasetArtifact:

    return (
        _ARTIFACT_STORE.get(
            workflow_id=(
                workflow_id
            ),

            dataset_id=(
                dataset_id
            ),
        )
    )


def get_preparation_dataframe(
    *,
    workflow_id: str,
    dataset_id: str,
) -> pd.DataFrame:

    return (
        _ARTIFACT_STORE.get_dataframe(
            workflow_id=(
                workflow_id
            ),

            dataset_id=(
                dataset_id
            ),
        )
    )


def list_preparation_artifacts(
    *,
    workflow_id: str,
) -> list[
    PreparationDatasetArtifactInfo
]:

    return (
        _ARTIFACT_STORE.list(
            workflow_id=(
                workflow_id
            )
        )
    )


def get_preparation_dataframe_map(
    *,
    workflow_id: str,
    dataset_ids: Iterable[
        str
    ]
    | None = None,
) -> dict[
    str,
    pd.DataFrame
]:

    return (
        _ARTIFACT_STORE.dataframe_map(
            workflow_id=(
                workflow_id
            ),

            dataset_ids=(
                dataset_ids
            ),
        )
    )


def delete_preparation_artifacts(
    *,
    workflow_id: str,
) -> None:

    _ARTIFACT_STORE.delete_workflow(
        workflow_id=(
            workflow_id
        )
    )


def reset_preparation_artifact_store_for_tests(
) -> None:

    _ARTIFACT_STORE.reset()