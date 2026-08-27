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
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from app.reporting.analysis_artifact_store import (
    AnalysisArtifactNotFoundError,
    AnalysisArtifactRecord,
    AnalysisSourceType,
    get_analysis_artifact,
)


from app.persistence.sqlite_database import (
    ensure_ephemeral_sqlite_test_path,
)

from app.reporting.report_selection_sqlite_store import (
    ReportSelectionSQLiteStoreError,
    import_legacy_report_selection_if_needed,
    load_report_selection_sqlite_payload,
    replace_report_selection_sqlite_payload,
)


# REPORT_SELECTION_SQLITE_CUTOVER_V0_1


# ============================================================
# VERSION
# ============================================================

REPORT_SELECTION_STORE_RULE_VERSION = (
    "report_selection_store_v0.1"
)


DEFAULT_REPORT_SELECTION_RELATIVE_PATH = (
    "var/reporting/report_selection.json"
)


# ============================================================
# MODELS
# ============================================================

class ReportSelectionStoredItem(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    analysis_id: str = Field(
        min_length=1
    )

    report_order: int = Field(
        ge=1
    )

    added_at_utc: str


class ReportSelectionItemView(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    analysis_id: str

    source_type: AnalysisSourceType

    objective: str

    trace_id: str

    report_order: int

    added_at_utc: str

    executed: bool


class ReportSelectionState(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    workflow_id: str

    revision: int = Field(
        ge=0
    )

    selected_count: int = Field(
        ge=0
    )

    analyses: list[
        ReportSelectionItemView
    ]

    rule_version: str = (
        REPORT_SELECTION_STORE_RULE_VERSION
    )


class ReportSelectedAnalysisDetail(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    selection: ReportSelectionItemView

    pipeline_payload: dict[
        str,
        Any,
    ]


class ReportSelectionDetailResponse(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    workflow_id: str

    revision: int

    selected_count: int

    analyses: list[
        ReportSelectedAnalysisDetail
    ]

    rule_version: str = (
        REPORT_SELECTION_STORE_RULE_VERSION
    )


# ============================================================
# ERRORS
# ============================================================

class ReportSelectionError(
    RuntimeError
):
    pass


class ReportSelectionNotExecutableError(
    ReportSelectionError
):
    pass


class ReportSelectionIntegrityError(
    ReportSelectionError
):
    pass


class ReportSelectionReorderError(
    ReportSelectionError
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


def resolve_report_selection_store_path() -> Path:
    configured = (
        os.getenv(
            "DATALENS_REPORT_SELECTION_STORE_PATH",
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
        DEFAULT_REPORT_SELECTION_RELATIVE_PATH
    ).resolve()


def _empty_payload() -> dict[
    str,
    Any,
]:
    return {
        "rule_version":
            REPORT_SELECTION_STORE_RULE_VERSION,

        "workflows":
            {},
    }


def _read_payload() -> dict[
    str,
    Any,
]:
    """
    Read ReportSelection from the SQLite control plane.

    The legacy report_selection.json file is imported exactly
    once per configured store path. After initialization,
    SQLite is authoritative.
    """

    path = (
        resolve_report_selection_store_path()
    )


    try:
        import_legacy_report_selection_if_needed(
            store_path=
                path,

            fallback_rule_version=
                REPORT_SELECTION_STORE_RULE_VERSION,
        )


        return (
            load_report_selection_sqlite_payload(
                store_path=
                    path,

                fallback_rule_version=
                    REPORT_SELECTION_STORE_RULE_VERSION,
            )
        )

    except ReportSelectionSQLiteStoreError as error:
        raise ReportSelectionError(
            (
                "Report selection SQLite store "
                "could not be read: "
                f"{error}"
            )
        ) from error


def _write_payload(
    payload: dict[
        str,
        Any,
    ],
) -> None:
    """
    Replace the configured ReportSelection scope
    transactionally in SQLite.

    The legacy JSON file is intentionally not rewritten.
    """

    path = (
        resolve_report_selection_store_path()
    )


    try:
        replace_report_selection_sqlite_payload(
            store_path=
                path,

            payload=
                payload,

            fallback_rule_version=
                REPORT_SELECTION_STORE_RULE_VERSION,

            legacy_json_imported=
                True,
        )

    except ReportSelectionSQLiteStoreError as error:
        raise ReportSelectionError(
            (
                "Report selection SQLite store "
                "could not be committed: "
                f"{error}"
            )
        ) from error


def _workflow_payload(
    payload: dict[
        str,
        Any,
    ],
    *,
    workflow_id: str,
) -> dict[
    str,
    Any,
]:
    raw = (
        payload[
            "workflows"
        ]
        .get(
            workflow_id
        )
    )


    if not isinstance(
        raw,
        dict,
    ):
        return {
            "revision":
                0,

            "analyses":
                [],
        }


    revision = int(
        raw.get(
            "revision",
            0,
        )
        or
        0
    )


    analyses = raw.get(
        "analyses",
        [],
    )


    if not isinstance(
        analyses,
        list,
    ):
        raise ReportSelectionIntegrityError(
            (
                "Report selection analyses must be "
                "stored as a list."
            )
        )


    return {
        "revision":
            revision,

        "analyses":
            analyses,
    }


def _stored_items(
    workflow_payload: dict[
        str,
        Any,
    ],
) -> list[
    ReportSelectionStoredItem
]:
    items = [
        ReportSelectionStoredItem
        .model_validate(
            raw
        )

        for raw
        in workflow_payload[
            "analyses"
        ]
    ]


    items.sort(
        key=lambda item:
            (
                item.report_order,
                item.added_at_utc,
                item.analysis_id,
            )
    )


    return items


def _reindex(
    items: list[
        ReportSelectionStoredItem
    ],
) -> list[
    ReportSelectionStoredItem
]:
    return [
        ReportSelectionStoredItem(
            analysis_id=
                item.analysis_id,

            report_order=
                index,

            added_at_utc=
                item.added_at_utc,
        )

        for (
            index,
            item,
        )
        in enumerate(
            items,
            start=1,
        )
    ]


def _view(
    *,
    artifact: AnalysisArtifactRecord,
    stored: ReportSelectionStoredItem,
) -> ReportSelectionItemView:
    return (
        ReportSelectionItemView(
            analysis_id=
                artifact.analysis_id,

            source_type=
                artifact.source_type,

            objective=
                artifact.objective,

            trace_id=
                artifact.trace_id,

            report_order=
                stored.report_order,

            added_at_utc=
                stored.added_at_utc,

            executed=
                artifact.executed,
        )
    )


# ============================================================
# PUBLIC READ
# ============================================================

def get_report_selection(
    *,
    workflow_id: str,
) -> ReportSelectionState:
    with _STORE_LOCK:
        payload = (
            _read_payload()
        )


        workflow_payload = (
            _workflow_payload(
                payload,
                workflow_id=
                    workflow_id,
            )
        )


        stored_items = (
            _stored_items(
                workflow_payload
            )
        )


    views: list[
        ReportSelectionItemView
    ] = []


    for stored in (
        stored_items
    ):
        try:
            artifact = (
                get_analysis_artifact(
                    workflow_id=
                        workflow_id,

                    analysis_id=
                        stored.analysis_id,
                )
            )


        except AnalysisArtifactNotFoundError as error:
            raise ReportSelectionIntegrityError(
                (
                    "Report selection references a missing "
                    "server-owned analysis artifact: "
                    f"{stored.analysis_id}."
                )
            ) from error


        views.append(
            _view(
                artifact=
                    artifact,

                stored=
                    stored,
            )
        )


    return (
        ReportSelectionState(
            workflow_id=
                workflow_id,

            revision=
                int(
                    workflow_payload[
                        "revision"
                    ]
                ),

            selected_count=
                len(
                    views
                ),

            analyses=
                views,
        )
    )


def get_report_selection_details(
    *,
    workflow_id: str,
) -> ReportSelectionDetailResponse:
    state = (
        get_report_selection(
            workflow_id=
                workflow_id
        )
    )


    details: list[
        ReportSelectedAnalysisDetail
    ] = []


    for selection in (
        state.analyses
    ):
        artifact = (
            get_analysis_artifact(
                workflow_id=
                    workflow_id,

                analysis_id=
                    selection.analysis_id,
            )
        )


        details.append(
            ReportSelectedAnalysisDetail(
                selection=
                    selection,

                pipeline_payload=
                    artifact.pipeline_payload,
            )
        )


    return (
        ReportSelectionDetailResponse(
            workflow_id=
                workflow_id,

            revision=
                state.revision,

            selected_count=
                state.selected_count,

            analyses=
                details,
        )
    )


# ============================================================
# PUBLIC MUTATIONS
# ============================================================

def add_analysis_to_report(
    *,
    workflow_id: str,
    analysis_id: str,
) -> ReportSelectionState:
    artifact = (
        get_analysis_artifact(
            workflow_id=
                workflow_id,

            analysis_id=
                analysis_id,
        )
    )


    if not (
        artifact.executed
    ):
        raise ReportSelectionNotExecutableError(
            (
                "Only successfully executed analyses may "
                "be added to the report."
            )
        )


    with _STORE_LOCK:
        payload = (
            _read_payload()
        )


        workflow_payload = (
            _workflow_payload(
                payload,
                workflow_id=
                    workflow_id,
            )
        )


        items = (
            _stored_items(
                workflow_payload
            )
        )


        if any(
            item.analysis_id
            ==
            analysis_id

            for item
            in items
        ):
            return (
                get_report_selection(
                    workflow_id=
                        workflow_id
                )
            )


        items.append(
            ReportSelectionStoredItem(
                analysis_id=
                    analysis_id,

                report_order=
                    len(
                        items
                    )
                    +
                    1,

                added_at_utc=(
                    datetime.now(
                        timezone.utc
                    )
                    .isoformat()
                ),
            )
        )


        items = (
            _reindex(
                items
            )
        )


        revision = int(
            workflow_payload[
                "revision"
            ]
        ) + 1


        payload[
            "workflows"
        ][
            workflow_id
        ] = {
            "revision":
                revision,

            "analyses":
                [
                    item.model_dump(
                        mode="json"
                    )

                    for item
                    in items
                ],
        }


        payload[
            "rule_version"
        ] = (
            REPORT_SELECTION_STORE_RULE_VERSION
        )


        _write_payload(
            payload
        )


    return (
        get_report_selection(
            workflow_id=
                workflow_id
        )
    )


def remove_analysis_from_report(
    *,
    workflow_id: str,
    analysis_id: str,
) -> ReportSelectionState:
    with _STORE_LOCK:
        payload = (
            _read_payload()
        )


        workflow_payload = (
            _workflow_payload(
                payload,
                workflow_id=
                    workflow_id,
            )
        )


        items = (
            _stored_items(
                workflow_payload
            )
        )


        remaining = [
            item

            for item
            in items

            if (
                item.analysis_id
                !=
                analysis_id
            )
        ]


        if (
            len(
                remaining
            )
            ==
            len(
                items
            )
        ):
            return (
                get_report_selection(
                    workflow_id=
                        workflow_id
                )
            )


        remaining = (
            _reindex(
                remaining
            )
        )


        revision = int(
            workflow_payload[
                "revision"
            ]
        ) + 1


        payload[
            "workflows"
        ][
            workflow_id
        ] = {
            "revision":
                revision,

            "analyses":
                [
                    item.model_dump(
                        mode="json"
                    )

                    for item
                    in remaining
                ],
        }


        _write_payload(
            payload
        )


    return (
        get_report_selection(
            workflow_id=
                workflow_id
        )
    )


def reorder_report_selection(
    *,
    workflow_id: str,
    analysis_ids: list[
        str
    ],
) -> ReportSelectionState:
    with _STORE_LOCK:
        payload = (
            _read_payload()
        )


        workflow_payload = (
            _workflow_payload(
                payload,
                workflow_id=
                    workflow_id,
            )
        )


        items = (
            _stored_items(
                workflow_payload
            )
        )


        current_ids = [
            item.analysis_id
            for item
            in items
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
            raise ReportSelectionReorderError(
                (
                    "Report reorder payload contains "
                    "duplicate analysis_id values."
                )
            )


        if (
            set(
                analysis_ids
            )
            !=
            set(
                current_ids
            )
        ):
            raise ReportSelectionReorderError(
                (
                    "Report reorder payload must contain "
                    "exactly the currently selected analyses."
                )
            )


        by_id = {
            item.analysis_id:
                item

            for item
            in items
        }


        reordered = (
            _reindex(
                [
                    by_id[
                        analysis_id
                    ]

                    for analysis_id
                    in analysis_ids
                ]
            )
        )


        revision = int(
            workflow_payload[
                "revision"
            ]
        ) + 1


        payload[
            "workflows"
        ][
            workflow_id
        ] = {
            "revision":
                revision,

            "analyses":
                [
                    item.model_dump(
                        mode="json"
                    )

                    for item
                    in reordered
                ],
        }


        _write_payload(
            payload
        )


    return (
        get_report_selection(
            workflow_id=
                workflow_id
        )
    )


# ============================================================
# TEST / WORKFLOW RESET
# ============================================================

def delete_report_selection(
    *,
    workflow_id: (
        str
        | None
    ) = None,
) -> None:
    with _STORE_LOCK:
        if (
            workflow_id
            is None
        ):
            # This no-argument branch is the historical
            # test/reset operation. Never let it clear the
            # production SQLite control plane.
            ensure_ephemeral_sqlite_test_path(
                namespace=
                    "report-selection-tests"
            )


            _write_payload(
                _empty_payload()
            )

            return


        payload = (
            _read_payload()
        )


        payload[
            "workflows"
        ].pop(
            workflow_id,
            None,
        )


        _write_payload(
            payload
        )
