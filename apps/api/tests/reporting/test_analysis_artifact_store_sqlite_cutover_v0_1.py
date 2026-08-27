from __future__ import annotations


import hashlib
import json
import os
import sqlite3
import tempfile

from pathlib import (
    Path,
)


_TEMP = tempfile.TemporaryDirectory(
    prefix=
        "datalens-analysis-artifact-cutover-"
)

_ROOT = Path(
    _TEMP.name
)

_DATABASE = (
    _ROOT
    /
    "datalens.sqlite3"
)

_STORE = (
    _ROOT
    /
    "analysis_artifacts.json"
)


os.environ[
    "DATALENS_SQLITE_PATH"
] = str(
    _DATABASE
)

os.environ[
    "DATALENS_ANALYSIS_ARTIFACT_STORE_PATH"
] = str(
    _STORE
)


from app.reporting.analysis_artifact_data_plane import (
    ANALYSIS_ARTIFACT_DATA_PLANE_VERSION,
    analysis_artifact_data_root,
)

from app.reporting.analysis_artifact_index import (
    analysis_artifact_store_scope,
    get_analysis_artifact_index_entry,
    load_analysis_artifact_store_state,
)

from app.reporting.analysis_artifact_store import (
    ANALYSIS_ARTIFACT_STORE_RULE_VERSION,
    AnalysisArtifactStoreError,
    delete_analysis_artifacts,
    get_analysis_artifact,
    list_analysis_artifacts,
    register_server_owned_analysis,
)


WORKFLOW = (
    "prep:legacy-analysis-artifacts"
)


def legacy_record(
    *,
    analysis_id: str,
    source_type: str,
    requested: bool,
) -> dict:
    pipeline_payload = {
        "analysis_id":
            analysis_id,

        "analysis_source_type":
            source_type,

        "status":
            "ready",
    }


    if requested:
        pipeline_payload[
            "request_lifecycle"
        ] = {
            "status":
                "resolved",
        }

        pipeline_payload[
            "requested_plan"
        ] = {
            "status":
                "ready",
        }


    return {
        "analysis_id":
            analysis_id,

        "workflow_id":
            WORKFLOW,

        "trace_id":
            "report:requested:shared",

        "source_type":
            source_type,

        "objective":
            "Analyse de test",

        "executed":
            True,

        "executed_count":
            1,

        "pipeline_payload":
            pipeline_payload,

        "created_at_utc":
            "2026-08-25T10:00:00+00:00",

        "rule_version":
            ANALYSIS_ARTIFACT_STORE_RULE_VERSION,
    }


def seed_legacy_store() -> bytes:
    payload = {
        "rule_version":
            ANALYSIS_ARTIFACT_STORE_RULE_VERSION,

        "artifacts": {
            "analysis:report:one":
                legacy_record(
                    analysis_id=
                        "analysis:report:one",

                    source_type=
                        "document_request",

                    requested=
                        True,
                ),

            "analysis:report:two":
                legacy_record(
                    analysis_id=
                        "analysis:report:two",

                    source_type=
                        "automatic",

                    requested=
                        False,
                ),
        },
    }


    _STORE.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


    return _STORE.read_bytes()


def test_legacy_cutover() -> None:
    legacy_bytes = (
        seed_legacy_store()
    )


    records = (
        list_analysis_artifacts(
            workflow_id=
                WORKFLOW
        )
    )


    assert (
        len(
            records
        )
        ==
        2
    )


    assert (
        records[
            0
        ].trace_id
        ==
        records[
            1
        ].trace_id
    )


    data_root = (
        analysis_artifact_data_root(
            _STORE
        )
    )


    payload_files = list(
        (
            data_root
            /
            "data"
        )
        .glob(
            "*.json.gz"
        )
    )


    assert (
        len(
            payload_files
        )
        ==
        2
    )


    state = (
        load_analysis_artifact_store_state(
            store_path=
                _STORE
        )
    )


    assert state is not None

    assert (
        state[
            "legacy_json_imported"
        ]
        is True
    )


    assert (
        _STORE.read_bytes()
        ==
        legacy_bytes
    )


    requested = (
        get_analysis_artifact(
            workflow_id=
                WORKFLOW,

            analysis_id=
                "analysis:report:one",
        )
    )


    assert (
        requested.pipeline_payload[
            "request_lifecycle"
        ][
            "status"
        ]
        ==
        "resolved"
    )


    print(
        "[PASS] legacy JSON -> SQLite metadata + gzip payloads"
    )

    print(
        "[PASS] duplicate trace_id survives cutover"
    )

    print(
        "[PASS] requested lifecycle stays in pipeline_payload"
    )

    print(
        "[PASS] legacy analysis_artifacts.json unchanged"
    )


def test_sqlite_is_authoritative() -> None:
    _STORE.write_text(
        json.dumps(
            {
                "rule_version":
                    ANALYSIS_ARTIFACT_STORE_RULE_VERSION,

                "artifacts":
                    {},
            }
        ),
        encoding="utf-8",
    )


    restored = (
        list_analysis_artifacts(
            workflow_id=
                WORKFLOW
        )
    )


    assert (
        len(
            restored
        )
        ==
        2
    )


    print(
        "[PASS] legacy JSON cannot overwrite initialized SQLite"
    )


def test_refresh_replaces_only_one_payload() -> None:
    before = (
        get_analysis_artifact_index_entry(
            store_path=
                _STORE,

            analysis_id=
                "analysis:report:one",
        )
    )


    assert before is not None


    old_payload_path = (
        analysis_artifact_data_root(
            _STORE
        )
        /
        before[
            "payload_path"
        ]
    )


    old_created_at = (
        before[
            "created_at_utc"
        ]
    )


    refreshed = (
        register_server_owned_analysis(
            workflow_id=
                WORKFLOW,

            analysis_id=
                "analysis:report:one",

            trace_id=
                "report:requested:shared",

            source_type=
                "document_request",

            objective=
                "Analyse de test",

            executed=
                True,

            executed_count=
                1,

            pipeline_payload={
                "status":
                    "ready",

                "request_lifecycle":
                    {
                        "status":
                            "reconfigured",
                    },

                "requested_plan":
                    {
                        "status":
                            "ready",
                    },
            },

            select_by_default=
                False,
        )
    )


    after = (
        get_analysis_artifact_index_entry(
            store_path=
                _STORE,

            analysis_id=
                "analysis:report:one",
        )
    )


    assert after is not None

    assert (
        refreshed.created_at_utc
        ==
        old_created_at
    )

    assert (
        after[
            "payload_path"
        ]
        !=
        before[
            "payload_path"
        ]
    )


    assert not (
        old_payload_path.exists()
    )


    new_payload_path = (
        analysis_artifact_data_root(
            _STORE
        )
        /
        after[
            "payload_path"
        ]
    )


    assert (
        new_payload_path.exists()
    )


    restored = (
        get_analysis_artifact(
            workflow_id=
                WORKFLOW,

            analysis_id=
                "analysis:report:one",
        )
    )


    assert (
        restored.pipeline_payload[
            "request_lifecycle"
        ][
            "status"
        ]
        ==
        "reconfigured"
    )


    print(
        "[PASS] refresh preserves created_at_utc"
    )

    print(
        "[PASS] refresh commits new payload before old-file cleanup"
    )

    print(
        "[PASS] requested reconfiguration payload survives"
    )


def test_workflow_delete_is_durable() -> None:
    register_server_owned_analysis(
        workflow_id=
            "prep:delete-me",

        analysis_id=
            "analysis:report:delete-me",

        trace_id=
            "trace:delete",

        source_type=
            "automatic",

        objective=
            "Delete test",

        executed=
            True,

        executed_count=
            1,

        pipeline_payload={
            "status":
                "ready",
        },

        select_by_default=
            False,
    )


    entry = (
        get_analysis_artifact_index_entry(
            store_path=
                _STORE,

            analysis_id=
                "analysis:report:delete-me",
        )
    )


    assert entry is not None


    payload_file = (
        analysis_artifact_data_root(
            _STORE
        )
        /
        entry[
            "payload_path"
        ]
    )


    assert (
        payload_file.exists()
    )


    delete_analysis_artifacts(
        workflow_id=
            "prep:delete-me"
    )


    assert (
        get_analysis_artifact_index_entry(
            store_path=
                _STORE,

            analysis_id=
                "analysis:report:delete-me",
        )
        is None
    )


    assert not (
        payload_file.exists()
    )


    print(
        "[PASS] workflow delete removes SQLite metadata"
    )

    print(
        "[PASS] workflow delete removes payload file after commit"
    )


def test_payload_integrity_guard() -> None:
    entry = (
        get_analysis_artifact_index_entry(
            store_path=
                _STORE,

            analysis_id=
                "analysis:report:one",
        )
    )


    assert entry is not None


    payload_file = (
        analysis_artifact_data_root(
            _STORE
        )
        /
        entry[
            "payload_path"
        ]
    )


    payload_file.write_bytes(
        b"corrupted"
    )


    try:
        get_analysis_artifact(
            workflow_id=
                WORKFLOW,

            analysis_id=
                "analysis:report:one",
        )

    except AnalysisArtifactStoreError:
        pass

    else:
        raise AssertionError(
            (
                "Corrupted payload must fail closed."
            )
        )


    print(
        "[PASS] corrupted payload fails closed"
    )


def test_versions() -> None:
    assert (
        ANALYSIS_ARTIFACT_STORE_RULE_VERSION
        ==
        "analysis_artifact_store_v0.2"
    )

    assert (
        ANALYSIS_ARTIFACT_DATA_PLANE_VERSION
        ==
        "analysis_artifact_data_plane_v0.1"
    )


    print(
        "[PASS] public store rule version preserved"
    )

    print(
        "[PASS] data-plane version"
    )


def main() -> None:
    print()

    print(
        "=== DATALENS ANALYSIS ARTIFACT SQLITE CUTOVER v0.1 ==="
    )

    print()


    test_legacy_cutover()

    test_sqlite_is_authoritative()

    test_refresh_replaces_only_one_payload()

    test_workflow_delete_is_durable()

    test_payload_integrity_guard()

    test_versions()


    print()

    print(
        "PASS - Analysis Artifact SQLite Cutover v0.1"
    )


if __name__ == "__main__":
    main()
