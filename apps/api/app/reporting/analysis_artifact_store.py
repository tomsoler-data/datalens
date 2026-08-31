from __future__ import annotations


import json
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

from typing import (
    Any,
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


import tempfile


from app.persistence.sqlite_database import (
    ensure_ephemeral_sqlite_test_path,
)

from app.reporting.analysis_artifact_index import (
    AnalysisArtifactIndexError,
    analysis_artifact_index_is_initialized,
    get_analysis_artifact_index_entry,
    load_analysis_artifact_index_scope,
    load_analysis_artifact_index_workflow,
    replace_analysis_artifact_index_scope,
    upsert_analysis_artifact_index_entry,
    upsert_analysis_artifact_index_entries_atomic,
    delete_analysis_artifact_index_workflow,
)

from app.reporting.analysis_artifact_data_plane import (
    AnalysisArtifactDataPlaneError,
    delete_analysis_artifact_payload,
    import_legacy_analysis_artifacts_if_needed,
    read_analysis_artifact_payload,
    write_analysis_artifact_payload,
)


# ANALYSIS_ARTIFACT_SQLITE_CUTOVER_V0_1


# ============================================================
# VERSION
# ============================================================

ANALYSIS_ARTIFACT_STORE_RULE_VERSION = (
    "analysis_artifact_store_v0.3"
)


DEFAULT_ANALYSIS_ARTIFACT_RELATIVE_PATH = (
    "var/reporting/analysis_artifacts.json"
)


AnalysisSourceType = Literal[
    "initial_request",
    "follow_up_prompt",
    "document_request",
    "automatic",
]


# ============================================================
# MODELS
# ============================================================

class AnalysisArtifactRecord(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    analysis_id: str = Field(
        min_length=1
    )

    workflow_id: str = Field(
        min_length=1
    )

    trace_id: str = Field(
        min_length=1
    )

    source_type: AnalysisSourceType

    objective: str = Field(
        min_length=1
    )

    executed: bool

    executed_count: int = Field(
        ge=0
    )

    pipeline_payload: dict[
        str,
        Any,
    ]

    created_at_utc: str

    rule_version: str = (
        ANALYSIS_ARTIFACT_STORE_RULE_VERSION
    )


class AnalysisArtifactSummary(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    analysis_id: str

    workflow_id: str

    trace_id: str

    source_type: AnalysisSourceType

    objective: str

    executed: bool

    executed_count: int

    created_at_utc: str


class AnalysisArtifactListResponse(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    workflow_id: str

    count: int = Field(
        ge=0
    )

    analyses: list[
        AnalysisArtifactSummary
    ]

    rule_version: str = (
        ANALYSIS_ARTIFACT_STORE_RULE_VERSION
    )


# ============================================================
# ERRORS
# ============================================================

class AnalysisArtifactStoreError(
    RuntimeError
):
    pass


class AnalysisArtifactNotFoundError(
    AnalysisArtifactStoreError
):
    def __init__(
        self,
        *,
        workflow_id: str,
        analysis_id: str,
    ) -> None:
        self.workflow_id = (
            workflow_id
        )

        self.analysis_id = (
            analysis_id
        )

        super().__init__(
            (
                "Analysis artifact was not found for "
                f"workflow_id={workflow_id}, "
                f"analysis_id={analysis_id}."
            )
        )


class AnalysisArtifactWorkflowMismatchError(
    AnalysisArtifactStoreError
):
    pass


# ============================================================
# STORAGE
# ============================================================

_STORE_LOCK = RLock()


def default_api_root() -> Path:
    return (
        Path(
            __file__
        )
        .resolve()
        .parents[
            2
        ]
    )


def resolve_analysis_artifact_store_path() -> Path:
    configured = (
        os.getenv(
            "DATALENS_ANALYSIS_ARTIFACT_STORE_PATH",
            "",
        )
        .strip()
    )


    if configured:
        path = Path(
            configured
        ).expanduser()


        if (
            path.is_absolute()
        ):
            return path


        return (
            default_api_root()
            /
            path
        ).resolve()


    return (
        default_api_root()
        /
        DEFAULT_ANALYSIS_ARTIFACT_RELATIVE_PATH
    ).resolve()


_TEST_STORE_DIRECTORY: (
    tempfile.TemporaryDirectory
    |
    None
) = None


def _production_store_path() -> Path:
    return (
        default_api_root()
        /
        DEFAULT_ANALYSIS_ARTIFACT_RELATIVE_PATH
    ).resolve()


def _ensure_ephemeral_analysis_artifact_store_path_for_tests(
) -> Path:
    """
    No-argument delete_analysis_artifacts() is a historical
    test/reset helper.

    Never let that helper target the production artifact store.
    """

    current = (
        resolve_analysis_artifact_store_path()
    )


    production = (
        _production_store_path()
    )


    if (
        current
        !=
        production
    ):
        return current


    global _TEST_STORE_DIRECTORY


    if (
        _TEST_STORE_DIRECTORY
        is None
    ):
        _TEST_STORE_DIRECTORY = (
            tempfile.TemporaryDirectory(
                prefix=
                    "datalens-analysis-artifact-tests-"
            )
        )


    temporary_path = (
        Path(
            _TEST_STORE_DIRECTORY.name
        )
        /
        "analysis_artifacts.json"
    )


    os.environ[
        "DATALENS_ANALYSIS_ARTIFACT_STORE_PATH"
    ] = str(
        temporary_path
    )


    return temporary_path


def _empty_payload() -> dict[
    str,
    Any,
]:
    return {
        "rule_version":
            ANALYSIS_ARTIFACT_STORE_RULE_VERSION,

        "artifacts":
            {},
    }


def _ensure_store_initialized() -> None:
    path = (
        resolve_analysis_artifact_store_path()
    )


    try:
        import_legacy_analysis_artifacts_if_needed(
            store_path=
                path,

            fallback_rule_version=
                ANALYSIS_ARTIFACT_STORE_RULE_VERSION,
        )

    except (
        AnalysisArtifactIndexError,
        AnalysisArtifactDataPlaneError,
    ) as error:
        raise AnalysisArtifactStoreError(
            (
                "AnalysisArtifact SQLite/data-plane "
                "initialization failed: "
                f"{error}"
            )
        ) from error


def _record_from_index_entry(
    entry: dict[
        str,
        Any,
    ],
) -> AnalysisArtifactRecord:
    path = (
        resolve_analysis_artifact_store_path()
    )


    try:
        pipeline_payload = (
            read_analysis_artifact_payload(
                store_path=
                    path,

                entry=
                    entry,
            )
        )

    except AnalysisArtifactDataPlaneError as error:
        raise AnalysisArtifactStoreError(
            (
                "AnalysisArtifact payload read failed "
                f"for analysis_id="
                f"{entry.get('analysis_id')}: "
                f"{error}"
            )
        ) from error


    analysis_id = str(
        entry[
            "analysis_id"
        ]
    )


    if (
        str(
            pipeline_payload.get(
                "analysis_id",
                "",
            )
        )
        !=
        analysis_id
    ):
        raise AnalysisArtifactStoreError(
            (
                "AnalysisArtifact payload analysis_id "
                "does not match SQLite metadata."
            )
        )


    if (
        str(
            pipeline_payload.get(
                "analysis_source_type",
                "",
            )
        )
        !=
        str(
            entry[
                "source_type"
            ]
        )
    ):
        raise AnalysisArtifactStoreError(
            (
                "AnalysisArtifact payload source type "
                "does not match SQLite metadata."
            )
        )


    return (
        AnalysisArtifactRecord(
            analysis_id=
                analysis_id,

            workflow_id=
                str(
                    entry[
                        "workflow_id"
                    ]
                ),

            trace_id=
                str(
                    entry[
                        "trace_id"
                    ]
                ),

            source_type=
                str(
                    entry[
                        "source_type"
                    ]
                ),

            objective=
                str(
                    entry[
                        "objective"
                    ]
                ),

            executed=
                bool(
                    entry[
                        "executed"
                    ]
                ),

            executed_count=
                int(
                    entry[
                        "executed_count"
                    ]
                ),

            pipeline_payload=
                pipeline_payload,

            created_at_utc=
                str(
                    entry[
                        "created_at_utc"
                    ]
                ),

            rule_version=
                str(
                    entry[
                        "rule_version"
                    ]
                ),
        )
    )


def _index_entry_for_record(
    *,
    record: AnalysisArtifactRecord,
    payload_info: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    return {
        "analysis_id":
            record.analysis_id,

        "workflow_id":
            record.workflow_id,

        "trace_id":
            record.trace_id,

        "source_type":
            record.source_type,

        "objective":
            record.objective,

        "executed":
            record.executed,

        "executed_count":
            record.executed_count,

        "created_at_utc":
            record.created_at_utc,

        "rule_version":
            record.rule_version,

        **payload_info,
    }


def _persist_record(
    record: AnalysisArtifactRecord,
) -> AnalysisArtifactRecord:
    _ensure_store_initialized()


    path = (
        resolve_analysis_artifact_store_path()
    )


    previous_entry = (
        get_analysis_artifact_index_entry(
            store_path=
                path,

            analysis_id=
                record.analysis_id,
        )
    )


    if (
        previous_entry is not None
        and
        str(
            previous_entry[
                "workflow_id"
            ]
        )
        !=
        record.workflow_id
    ):
        raise AnalysisArtifactWorkflowMismatchError(
            (
                "An existing analysis_id belongs "
                "to another workflow."
            )
        )


    try:
        payload_info = (
            write_analysis_artifact_payload(
                store_path=
                    path,

                analysis_id=
                    record.analysis_id,

                pipeline_payload=
                    record.pipeline_payload,
            )
        )


        entry = (
            _index_entry_for_record(
                record=
                    record,

                payload_info=
                    payload_info,
            )
        )


        upsert_analysis_artifact_index_entry(
            store_path=
                path,

            entry=
                entry,
        )


    except Exception:
        if (
            "payload_info"
            in locals()
        ):
            try:
                delete_analysis_artifact_payload(
                    store_path=
                        path,

                    payload_path=
                        payload_info[
                            "payload_path"
                        ],
                )

            except Exception:
                pass

        raise


    if (
        previous_entry is not None
        and
        str(
            previous_entry[
                "payload_path"
            ]
        )
        !=
        str(
            payload_info[
                "payload_path"
            ]
        )
    ):
        try:
            delete_analysis_artifact_payload(
                store_path=
                    path,

                payload_path=
                    str(
                        previous_entry[
                            "payload_path"
                        ]
                    ),
            )

        except AnalysisArtifactDataPlaneError:
            # Metadata is already committed to the new valid
            # file. A stale old file must not invalidate it.
            pass


    return record


def _read_payload() -> dict[
    str,
    Any,
]:
    """
    Compatibility full-store read.

    Public point/workflow reads below use the SQLite index
    directly and therefore do not open all 480 payload files.
    """

    with _STORE_LOCK:
        _ensure_store_initialized()


        path = (
            resolve_analysis_artifact_store_path()
        )


        entries = (
            load_analysis_artifact_index_scope(
                store_path=
                    path
            )
        )


        artifacts = {}


        for entry in entries:
            record = (
                _record_from_index_entry(
                    entry
                )
            )


            artifacts[
                record.analysis_id
            ] = (
                record.model_dump(
                    mode="json"
                )
            )


        return {
            "rule_version":
                ANALYSIS_ARTIFACT_STORE_RULE_VERSION,

            "artifacts":
                artifacts,
        }


def _write_payload(
    payload: dict[
        str,
        Any,
    ],
) -> None:
    """
    Compatibility full-store replacement.

    Normal writes use _persist_record() and therefore replace
    only one payload file plus one SQLite row.
    """

    if not isinstance(
        payload,
        dict,
    ):
        raise AnalysisArtifactStoreError(
            (
                "Analysis artifact store root "
                "must be an object."
            )
        )


    artifacts = payload.get(
        "artifacts"
    )


    if not isinstance(
        artifacts,
        dict,
    ):
        raise AnalysisArtifactStoreError(
            (
                "Analysis artifact store must "
                "contain an artifact map."
            )
        )


    with _STORE_LOCK:
        _ensure_store_initialized()


        path = (
            resolve_analysis_artifact_store_path()
        )


        previous_entries = (
            load_analysis_artifact_index_scope(
                store_path=
                    path
            )
        )


        new_entries = []

        new_paths = []


        try:
            for (
                map_analysis_id,
                raw,
            ) in artifacts.items():

                if not isinstance(
                    raw,
                    dict,
                ):
                    raise AnalysisArtifactStoreError(
                        (
                            "AnalysisArtifact record "
                            "must be an object."
                        )
                    )


                record = (
                    AnalysisArtifactRecord
                    .model_validate(
                        raw
                    )
                )


                if (
                    record.analysis_id
                    !=
                    str(
                        map_analysis_id
                    )
                ):
                    raise AnalysisArtifactStoreError(
                        (
                            "Artifact map key does not "
                            "match analysis_id."
                        )
                    )


                payload_info = (
                    write_analysis_artifact_payload(
                        store_path=
                            path,

                        analysis_id=
                            record.analysis_id,

                        pipeline_payload=
                            record.pipeline_payload,
                    )
                )


                new_paths.append(
                    payload_info[
                        "payload_path"
                    ]
                )


                new_entries.append(
                    _index_entry_for_record(
                        record=
                            record,

                        payload_info=
                            payload_info,
                    )
                )


            replace_analysis_artifact_index_scope(
                store_path=
                    path,

                entries=
                    new_entries,

                legacy_json_imported=
                    True,

                legacy_rule_version=
                    ANALYSIS_ARTIFACT_STORE_RULE_VERSION,
            )


        except Exception:
            for payload_path in new_paths:
                try:
                    delete_analysis_artifact_payload(
                        store_path=
                            path,

                        payload_path=
                            payload_path,
                    )

                except Exception:
                    pass

            raise


        referenced_paths = {
            str(
                entry[
                    "payload_path"
                ]
            )

            for entry
            in new_entries
        }


        for previous in previous_entries:
            previous_path = str(
                previous[
                    "payload_path"
                ]
            )


            if (
                previous_path
                in referenced_paths
            ):
                continue


            try:
                delete_analysis_artifact_payload(
                    store_path=
                        path,

                    payload_path=
                        previous_path,
                )

            except AnalysisArtifactDataPlaneError:
                pass


# ============================================================
# HELPERS
# ============================================================

def _workflow_id_from_datasets(
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> (
    str
    | None
):
    workflow_ids = {
        str(
            dataset.get(
                "preparation_workflow_id",
                "",
            )
        ).strip()

        for dataset
        in datasets
    }


    workflow_ids.discard(
        ""
    )


    if (
        len(
            workflow_ids
        )
        !=
        1
    ):
        return None


    return next(
        iter(
            workflow_ids
        )
    )


def _source_type_for_new_artifact(
    *,
    workflow_id: str,
    artifacts: dict[
        str,
        Any,
    ],
) -> AnalysisSourceType:
    """
    Classify only the prompt conversation.

    Automatic discoveries and document-requested analyses may be
    persisted before the first prompt-native execution. They must
    never cause the first user prompt to be misclassified as a
    follow-up.
    """

    has_prior_prompt = any(
        (
            isinstance(
                payload,
                dict,
            )
            and
            str(
                payload.get(
                    "workflow_id",
                    "",
                )
            )
            ==
            workflow_id
            and
            str(
                payload.get(
                    "source_type",
                    "",
                )
            )
            in {
                "initial_request",
                "follow_up_prompt",
            }
        )

        for payload
        in artifacts.values()
    )


    if (
        has_prior_prompt
    ):
        return (
            "follow_up_prompt"
        )


    return (
        "initial_request"
    )


def _summary(
    record: AnalysisArtifactRecord,
) -> AnalysisArtifactSummary:
    return (
        AnalysisArtifactSummary(
            analysis_id=
                record.analysis_id,

            workflow_id=
                record.workflow_id,

            trace_id=
                record.trace_id,

            source_type=
                record.source_type,

            objective=
                record.objective,

            executed=
                record.executed,

            executed_count=
                record.executed_count,

            created_at_utc=
                record.created_at_utc,
        )
    )


# ============================================================
# PUBLIC WRITE
# ============================================================

def register_native_pipeline_result(
    *,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
    pipeline_report: Any,
) -> (
    AnalysisArtifactRecord
    | None
):
    workflow_id = (
        _workflow_id_from_datasets(
            datasets
        )
    )


    if (
        workflow_id is None
    ):
        return None


    trace_id = str(
        getattr(
            pipeline_report,
            "trace_id",
            "",
        )
        or
        ""
    ).strip()


    if not trace_id:
        return None


    planner = getattr(
        pipeline_report,
        "planner",
        None,
    )


    objective = str(
        getattr(
            planner,
            "objective",
            "",
        )
        or
        ""
    ).strip()


    if not objective:
        return None


    analysis_id = (
        f"analysis:{trace_id}"
    )


    executed_count = int(
        getattr(
            pipeline_report,
            "executed_count",
            0,
        )
        or
        0
    )


    with _STORE_LOCK:
        _ensure_store_initialized()


        path = (
            resolve_analysis_artifact_store_path()
        )


        existing_entry = (
            get_analysis_artifact_index_entry(
                store_path=
                    path,

                analysis_id=
                    analysis_id,
            )
        )


        if (
            existing_entry
            is not None
        ):
            if (
                str(
                    existing_entry[
                        "workflow_id"
                    ]
                )
                !=
                workflow_id
            ):
                raise (
                    AnalysisArtifactWorkflowMismatchError(
                        (
                            "A server-generated analysis_id "
                            "already exists under another "
                            "workflow."
                        )
                    )
                )


            return (
                _record_from_index_entry(
                    existing_entry
                )
            )


        metadata_entries = (
            load_analysis_artifact_index_scope(
                store_path=
                    path
            )
        )


        artifacts_for_classification = {
            str(
                entry[
                    "analysis_id"
                ]
            ):
                {
                    "workflow_id":
                        entry[
                            "workflow_id"
                        ],

                    "source_type":
                        entry[
                            "source_type"
                        ],
                }

            for entry
            in metadata_entries
        }


        source_type = (
            _source_type_for_new_artifact(
                workflow_id=
                    workflow_id,

                artifacts=
                    artifacts_for_classification,
            )
        )


        pipeline_payload = (
            pipeline_report.model_dump(
                mode="json"
            )
            if hasattr(
                pipeline_report,
                "model_dump",
            )
            else dict(
                pipeline_report
            )
        )


        pipeline_payload[
            "analysis_id"
        ] = analysis_id

        pipeline_payload[
            "analysis_source_type"
        ] = source_type


        record = (
            AnalysisArtifactRecord(
                analysis_id=
                    analysis_id,

                workflow_id=
                    workflow_id,

                trace_id=
                    trace_id,

                source_type=
                    source_type,

                objective=
                    objective,

                executed=(
                    executed_count
                    >
                    0
                ),

                executed_count=
                    executed_count,

                pipeline_payload=
                    pipeline_payload,

                created_at_utc=(
                    datetime.now(
                        timezone.utc
                    )
                    .isoformat()
                ),
            )
        )


        _persist_record(
            record
        )


    # Report selection is an explicit user action.
    #
    # A successful prompt-native analysis is persisted and
    # remains available in the analysis history, but execution
    # alone must never mutate report composition.
    #
    # This deliberately separates:
    #
    # - analysis lifecycle / availability;
    # - report-selection state.
    #
    # The report selection API remains the only path that may
    # add an executed analysis to the report.
    return record


# ============================================================
# SERVER-OWNED GENERIC ANALYSIS WRITE
# ============================================================



def build_server_owned_analysis_record(
    *,
    workflow_id: str,
    analysis_id: str,
    trace_id: str,
    source_type: AnalysisSourceType,
    objective: str,
    executed: bool,
    executed_count: int,
    pipeline_payload: dict[
        str,
        Any,
    ],
    select_by_default: bool = False,
) -> AnalysisArtifactRecord:
    """
    Build and validate one server-owned AnalysisArtifactRecord
    without persisting it.

    Existing created_at preservation is applied by the atomic
    persistence layer at commit time.
    """

    _ = select_by_default


    normalized_workflow_id = str(
        workflow_id
    ).strip()

    normalized_analysis_id = str(
        analysis_id
    ).strip()

    normalized_trace_id = str(
        trace_id
    ).strip()

    normalized_objective = str(
        objective
    ).strip()


    if not normalized_workflow_id:
        raise ValueError(
            "workflow_id cannot be empty."
        )

    if not normalized_analysis_id:
        raise ValueError(
            "analysis_id cannot be empty."
        )

    if not normalized_trace_id:
        raise ValueError(
            "trace_id cannot be empty."
        )

    if not normalized_objective:
        raise ValueError(
            "objective cannot be empty."
        )

    if (
        executed_count
        <
        0
    ):
        raise ValueError(
            "executed_count cannot be negative."
        )


    payload_copy = dict(
        pipeline_payload
    )

    payload_copy[
        "analysis_id"
    ] = normalized_analysis_id

    payload_copy[
        "analysis_source_type"
    ] = source_type


    return (
        AnalysisArtifactRecord(
            analysis_id=
                normalized_analysis_id,

            workflow_id=
                normalized_workflow_id,

            trace_id=
                normalized_trace_id,

            source_type=
                source_type,

            objective=
                normalized_objective,

            executed=
                bool(
                    executed
                ),

            executed_count=
                int(
                    executed_count
                ),

            pipeline_payload=
                payload_copy,

            created_at_utc=(
                datetime.now(
                    timezone.utc
                )
                .isoformat()
            ),
        )
    )




def persist_server_owned_analysis_records_atomic(
    *,
    records: list[
        AnalysisArtifactRecord
    ],
) -> list[
    AnalysisArtifactRecord
]:
    """
    Persist multiple server-owned AnalysisArtifact records as
    one logical artifact-store commit.

    Atomic visibility invariant:

    - every new payload file is prepared first;
    - all metadata rows are committed in one SQLite transaction;
    - if payload preparation or metadata commit fails, every
      newly prepared payload is deleted;
    - previous metadata and previous payloads remain visible;
    - old replaced payloads are deleted only after metadata
      commit succeeds.

    Payload files use unique names, so preparing a replacement
    never mutates the previously visible payload.
    """

    if not records:
        return []


    validated_records = [
        (
            record
            if isinstance(
                record,
                AnalysisArtifactRecord,
            )
            else
            AnalysisArtifactRecord.model_validate(
                record
            )
        )

        for record
        in records
    ]


    analysis_ids = [
        record.analysis_id

        for record
        in validated_records
    ]


    if (
        len(
            analysis_ids
        )
        !=
        len(
            set(
                analysis_ids
            )
        )
    ):
        raise AnalysisArtifactStoreError(
            (
                "Atomic AnalysisArtifact batch contains "
                "duplicate analysis_id values."
            )
        )


    committed_records: list[
        AnalysisArtifactRecord
    ] = []


    with _STORE_LOCK:
        _ensure_store_initialized()


        path = (
            resolve_analysis_artifact_store_path()
        )


        previous_entries = {}


        for record in validated_records:
            previous = (
                get_analysis_artifact_index_entry(
                    store_path=
                        path,

                    analysis_id=
                        record.analysis_id,
                )
            )


            if (
                previous is not None
                and
                str(
                    previous[
                        "workflow_id"
                    ]
                )
                !=
                record.workflow_id
            ):
                raise (
                    AnalysisArtifactWorkflowMismatchError(
                        (
                            "A server-owned analysis_id already "
                            "exists under another workflow."
                        )
                    )
                )


            previous_entries[
                record.analysis_id
            ] = previous


            if previous is not None:
                record = record.model_copy(
                    update={
                        "created_at_utc":
                            str(
                                previous[
                                    "created_at_utc"
                                ]
                            ),
                    }
                )


            committed_records.append(
                record
            )


        new_payload_paths: list[
            str
        ] = []

        new_entries: list[
            dict[
                str,
                Any,
            ]
        ] = []


        try:
            # ------------------------------------------------
            # Stage all replacement/new payloads.
            # ------------------------------------------------

            for record in committed_records:
                payload_info = (
                    write_analysis_artifact_payload(
                        store_path=
                            path,

                        analysis_id=
                            record.analysis_id,

                        pipeline_payload=
                            record.pipeline_payload,
                    )
                )


                new_payload_paths.append(
                    payload_info[
                        "payload_path"
                    ]
                )


                new_entries.append(
                    _index_entry_for_record(
                        record=
                            record,

                        payload_info=
                            payload_info,
                    )
                )


            # ------------------------------------------------
            # Single metadata transaction.
            # ------------------------------------------------

            upsert_analysis_artifact_index_entries_atomic(
                store_path=
                    path,

                entries=
                    new_entries,
            )


        except Exception:
            # Metadata did not commit successfully.
            #
            # Newly staged payloads are unreachable and must
            # therefore be removed. Previous payloads remain
            # untouched and previous metadata remains visible.
            for payload_path in (
                new_payload_paths
            ):
                try:
                    delete_analysis_artifact_payload(
                        store_path=
                            path,

                        payload_path=
                            payload_path,
                    )

                except Exception:
                    pass


            raise


        # ----------------------------------------------------
        # Metadata now points only at the new payloads.
        #
        # Old replaced files are stale and may be deleted.
        # Failure here cannot invalidate the committed metadata;
        # at worst it leaves an unreachable stale file.
        # ----------------------------------------------------

        new_entry_by_id = {
            entry[
                "analysis_id"
            ]:
                entry

            for entry
            in new_entries
        }


        for record in committed_records:
            previous = (
                previous_entries[
                    record.analysis_id
                ]
            )


            if previous is None:
                continue


            previous_path = str(
                previous[
                    "payload_path"
                ]
            )

            new_path = str(
                new_entry_by_id[
                    record.analysis_id
                ][
                    "payload_path"
                ]
            )


            if (
                previous_path
                ==
                new_path
            ):
                continue


            try:
                delete_analysis_artifact_payload(
                    store_path=
                        path,

                    payload_path=
                        previous_path,
                )

            except AnalysisArtifactDataPlaneError:
                pass


    # --------------------------------------------------------
    # Preserve the existing report-selection invariant.
    #
    # Non-executable lifecycle artifacts must not remain
    # selected in a composed report.
    # --------------------------------------------------------

    non_executed = [
        record

        for record
        in committed_records

        if not record.executed
    ]


    if non_executed:
        from app.reporting.report_selection_store import (
            remove_analysis_from_report,
        )


        for record in non_executed:
            remove_analysis_from_report(
                workflow_id=
                    record.workflow_id,

                analysis_id=
                    record.analysis_id,
            )


    return (
        committed_records
    )


def register_server_owned_analysis(
    *,
    workflow_id: str,
    analysis_id: str,
    trace_id: str,
    source_type: AnalysisSourceType,
    objective: str,
    executed: bool,
    executed_count: int,
    pipeline_payload: dict[
        str,
        Any,
    ],
    select_by_default: bool = False,
) -> AnalysisArtifactRecord:
    """
    Persist or refresh one server-owned analysis artifact.

    Report composition is manual-only in v0.3.

    ``select_by_default`` is intentionally retained in the
    public signature for backward compatibility with existing
    server call sites, but it no longer causes an analysis to
    be added to the report. Executed analyses remain available
    for explicit selection through the report-selection API.

    Non-executable artifacts are still removed from an existing
    report selection. This keeps the invariant that a selected
    report item must be executable.
    """

    # Compatibility parameter: callers from older server paths
    # may still pass True. Under the manual-only policy, that
    # request must have no effect on report composition.
    _ = select_by_default


    normalized_workflow_id = str(
        workflow_id
    ).strip()

    normalized_analysis_id = str(
        analysis_id
    ).strip()

    normalized_trace_id = str(
        trace_id
    ).strip()

    normalized_objective = str(
        objective
    ).strip()


    if not normalized_workflow_id:
        raise ValueError(
            "workflow_id cannot be empty."
        )

    if not normalized_analysis_id:
        raise ValueError(
            "analysis_id cannot be empty."
        )

    if not normalized_trace_id:
        raise ValueError(
            "trace_id cannot be empty."
        )

    if not normalized_objective:
        raise ValueError(
            "objective cannot be empty."
        )

    if (
        executed_count
        <
        0
    ):
        raise ValueError(
            "executed_count cannot be negative."
        )


    payload_copy = dict(
        pipeline_payload
    )

    payload_copy[
        "analysis_id"
    ] = normalized_analysis_id

    payload_copy[
        "analysis_source_type"
    ] = source_type


    with _STORE_LOCK:
        _ensure_store_initialized()


        path = (
            resolve_analysis_artifact_store_path()
        )


        existing_entry = (
            get_analysis_artifact_index_entry(
                store_path=
                    path,

                analysis_id=
                    normalized_analysis_id,
            )
        )


        if (
            existing_entry
            is not None
        ):
            if (
                str(
                    existing_entry[
                        "workflow_id"
                    ]
                )
                !=
                normalized_workflow_id
            ):
                raise (
                    AnalysisArtifactWorkflowMismatchError(
                        (
                            "A server-owned analysis_id already "
                            "exists under another workflow."
                        )
                    )
                )


            created_at_utc = str(
                existing_entry[
                    "created_at_utc"
                ]
            )


        else:
            created_at_utc = (
                datetime.now(
                    timezone.utc
                )
                .isoformat()
            )


        record = (
            AnalysisArtifactRecord(
                analysis_id=
                    normalized_analysis_id,

                workflow_id=
                    normalized_workflow_id,

                trace_id=
                    normalized_trace_id,

                source_type=
                    source_type,

                objective=
                    normalized_objective,

                executed=
                    bool(
                        executed
                    ),

                executed_count=
                    int(
                        executed_count
                    ),

                pipeline_payload=
                    payload_copy,

                created_at_utc=
                    created_at_utc,
            )
        )


        _persist_record(
            record
        )


    if not (
        record.executed
    ):
        from app.reporting.report_selection_store import (
            remove_analysis_from_report,
        )


        remove_analysis_from_report(
            workflow_id=
                record.workflow_id,

            analysis_id=
                record.analysis_id,
        )


    return record






class AnalysisArtifactDetail(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    analysis_id: str

    workflow_id: str

    trace_id: str

    source_type: AnalysisSourceType

    objective: str

    executed: bool

    executed_count: int

    pipeline_payload: dict[
        str,
        Any,
    ]

    created_at_utc: str


class AnalysisArtifactDetailListResponse(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    workflow_id: str

    count: int = Field(
        ge=0
    )

    analyses: list[
        AnalysisArtifactDetail
    ]

    rule_version: str = (
        ANALYSIS_ARTIFACT_STORE_RULE_VERSION
    )


# ============================================================
# PUBLIC READ
# ============================================================

def get_analysis_artifact(
    *,
    workflow_id: str,
    analysis_id: str,
) -> AnalysisArtifactRecord:
    with _STORE_LOCK:
        _ensure_store_initialized()


        path = (
            resolve_analysis_artifact_store_path()
        )


        entry = (
            get_analysis_artifact_index_entry(
                store_path=
                    path,

                analysis_id=
                    analysis_id,
            )
        )


        if (
            entry is None
            or
            str(
                entry[
                    "workflow_id"
                ]
            )
            !=
            workflow_id
        ):
            raise (
                AnalysisArtifactNotFoundError(
                    workflow_id=
                        workflow_id,

                    analysis_id=
                        analysis_id,
                )
            )


        return (
            _record_from_index_entry(
                entry
            )
        )


def list_analysis_artifacts(
    *,
    workflow_id: str,
) -> list[
    AnalysisArtifactRecord
]:
    with _STORE_LOCK:
        _ensure_store_initialized()


        path = (
            resolve_analysis_artifact_store_path()
        )


        entries = (
            load_analysis_artifact_index_workflow(
                store_path=
                    path,

                workflow_id=
                    workflow_id,
            )
        )


        return [
            _record_from_index_entry(
                entry
            )

            for entry
            in entries
        ]


def list_analysis_artifact_summaries(
    *,
    workflow_id: str,
) -> AnalysisArtifactListResponse:
    records = (
        list_analysis_artifacts(
            workflow_id=
                workflow_id
        )
    )


    return (
        AnalysisArtifactListResponse(
            workflow_id=
                workflow_id,

            count=
                len(
                    records
                ),

            analyses=[
                _summary(
                    record
                )

                for record
                in records
            ],
        )
    )



def list_analysis_artifact_details(
    *,
    workflow_id: str,
) -> AnalysisArtifactDetailListResponse:
    records = (
        list_analysis_artifacts(
            workflow_id=
                workflow_id
        )
    )


    return (
        AnalysisArtifactDetailListResponse(
            workflow_id=
                workflow_id,

            count=
                len(
                    records
                ),

            analyses=[
                AnalysisArtifactDetail(
                    analysis_id=
                        record.analysis_id,

                    workflow_id=
                        record.workflow_id,

                    trace_id=
                        record.trace_id,

                    source_type=
                        record.source_type,

                    objective=
                        record.objective,

                    executed=
                        record.executed,

                    executed_count=
                        record.executed_count,

                    pipeline_payload=
                        record.pipeline_payload,

                    created_at_utc=
                        record.created_at_utc,
                )

                for record
                in records
            ],
        )
    )


# ============================================================
# TEST / WORKFLOW RESET
# ============================================================

def delete_analysis_artifacts(
    *,
    workflow_id: (
        str
        |
        None
    ) = None,
) -> None:
    with _STORE_LOCK:
        if (
            workflow_id
            is None
        ):
            _ensure_ephemeral_analysis_artifact_store_path_for_tests()

            ensure_ephemeral_sqlite_test_path(
                namespace=
                    "analysis-artifact-tests"
            )


        _ensure_store_initialized()


        path = (
            resolve_analysis_artifact_store_path()
        )


        if (
            workflow_id
            is None
        ):
            entries = (
                load_analysis_artifact_index_scope(
                    store_path=
                        path
                )
            )


            replace_analysis_artifact_index_scope(
                store_path=
                    path,

                entries=[],

                legacy_json_imported=
                    True,

                legacy_rule_version=
                    ANALYSIS_ARTIFACT_STORE_RULE_VERSION,
            )


        else:
            entries = (
                load_analysis_artifact_index_workflow(
                    store_path=
                        path,

                    workflow_id=
                        workflow_id,
                )
            )


            delete_analysis_artifact_index_workflow(
                store_path=
                    path,

                workflow_id=
                    workflow_id,
            )


        # SQLite deletion commits first. Files are data-plane
        # cleanup after the authoritative metadata commit.
        for entry in entries:
            try:
                delete_analysis_artifact_payload(
                    store_path=
                        path,

                    payload_path=
                        str(
                            entry[
                                "payload_path"
                            ]
                        ),
                )

            except AnalysisArtifactDataPlaneError:
                # A stale orphan must not roll back a committed
                # SQLite deletion.
                pass
