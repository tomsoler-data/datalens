from __future__ import annotations


import json

from pathlib import (
    Path,
)

from threading import (
    RLock,
)

from typing import (
    Any,
)


from app.persistence.sqlite_database import (
    sqlite_connection,
    utc_now_iso,
)


# ========================================================
# VERSION
# ========================================================


REPORT_SELECTION_SQLITE_STORE_VERSION = (
    "report_selection_sqlite_store_v0.1"
)


_MIGRATION_LOCK = (
    RLock()
)


# ========================================================
# ERROR
# ========================================================


class ReportSelectionSQLiteStoreError(
    RuntimeError
):
    pass


# ========================================================
# SCOPE
# ========================================================


def report_selection_store_scope(
    store_path: Path,
) -> str:
    """
    Scope SQLite rows to the configured legacy store path.

    This preserves isolation for tests that already use
    DATALENS_REPORT_SELECTION_STORE_PATH.
    """

    return str(
        store_path
        .expanduser()
        .resolve()
    )


# ========================================================
# VALIDATION
# ========================================================


def _validate_workflow_payload(
    *,
    workflow_id: str,
    raw: object,
) -> dict[
    str,
    Any,
]:
    if not isinstance(
        raw,
        dict,
    ):
        raise ReportSelectionSQLiteStoreError(
            (
                "Report selection workflow payload "
                "must be an object."
            )
        )


    try:
        revision = int(
            raw.get(
                "revision",
                0,
            )
            or
            0
        )

    except Exception as error:
        raise ReportSelectionSQLiteStoreError(
            (
                "Report selection revision "
                "must be an integer."
            )
        ) from error


    if revision < 0:
        raise ReportSelectionSQLiteStoreError(
            (
                "Report selection revision "
                "cannot be negative."
            )
        )


    analyses = raw.get(
        "analyses",
        [],
    )


    if not isinstance(
        analyses,
        list,
    ):
        raise ReportSelectionSQLiteStoreError(
            (
                "Report selection analyses "
                "must be a list."
            )
        )


    validated_items: list[
        dict[
            str,
            Any,
        ]
    ] = []


    analysis_ids = set()

    report_orders = set()


    for raw_item in analyses:
        if not isinstance(
            raw_item,
            dict,
        ):
            raise ReportSelectionSQLiteStoreError(
                (
                    "Report selection item "
                    "must be an object."
                )
            )


        analysis_id = str(
            raw_item.get(
                "analysis_id",
                "",
            )
        ).strip()


        if not analysis_id:
            raise ReportSelectionSQLiteStoreError(
                (
                    "Report selection item "
                    "has no valid analysis_id."
                )
            )


        try:
            report_order = int(
                raw_item.get(
                    "report_order"
                )
            )

        except Exception as error:
            raise ReportSelectionSQLiteStoreError(
                (
                    "Report selection report_order "
                    "must be an integer."
                )
            ) from error


        if report_order < 1:
            raise ReportSelectionSQLiteStoreError(
                (
                    "Report selection report_order "
                    "must be positive."
                )
            )


        added_at_utc = str(
            raw_item.get(
                "added_at_utc",
                "",
            )
        ).strip()


        if not added_at_utc:
            raise ReportSelectionSQLiteStoreError(
                (
                    "Report selection item "
                    "has no added_at_utc."
                )
            )


        if analysis_id in analysis_ids:
            raise ReportSelectionSQLiteStoreError(
                (
                    "Report selection contains "
                    "duplicate analysis_id values."
                )
            )


        if report_order in report_orders:
            raise ReportSelectionSQLiteStoreError(
                (
                    "Report selection contains "
                    "duplicate report_order values."
                )
            )


        analysis_ids.add(
            analysis_id
        )

        report_orders.add(
            report_order
        )


        validated_items.append(
            {
                "analysis_id":
                    analysis_id,

                "report_order":
                    report_order,

                "added_at_utc":
                    added_at_utc,
            }
        )


    validated_items.sort(
        key=lambda item:
            (
                int(
                    item[
                        "report_order"
                    ]
                ),
                str(
                    item[
                        "added_at_utc"
                    ]
                ),
                str(
                    item[
                        "analysis_id"
                    ]
                ),
            )
    )


    expected_orders = list(
        range(
            1,
            len(
                validated_items
            )
            +
            1,
        )
    )


    actual_orders = [
        int(
            item[
                "report_order"
            ]
        )

        for item
        in validated_items
    ]


    if (
        actual_orders
        !=
        expected_orders
    ):
        raise ReportSelectionSQLiteStoreError(
            (
                "Report selection report_order "
                "must be contiguous starting at 1."
            )
        )


    return {
        "revision":
            revision,

        "analyses":
            validated_items,
    }


def _validate_store_payload(
    payload: object,
    *,
    fallback_rule_version: str,
) -> dict[
    str,
    Any,
]:
    if not isinstance(
        payload,
        dict,
    ):
        raise ReportSelectionSQLiteStoreError(
            (
                "Report selection store root "
                "must be an object."
            )
        )


    workflows = payload.get(
        "workflows"
    )


    if not isinstance(
        workflows,
        dict,
    ):
        raise ReportSelectionSQLiteStoreError(
            (
                "Report selection store "
                "must contain workflows."
            )
        )


    rule_version = str(
        payload.get(
            "rule_version",
            fallback_rule_version,
        )
    ).strip()


    if not rule_version:
        rule_version = (
            fallback_rule_version
        )


    validated_workflows = {}


    for (
        workflow_id,
        raw_workflow,
    ) in workflows.items():
        normalized_workflow_id = str(
            workflow_id
        ).strip()


        if not normalized_workflow_id:
            raise ReportSelectionSQLiteStoreError(
                (
                    "Report selection contains "
                    "an empty workflow_id."
                )
            )


        validated_workflows[
            normalized_workflow_id
        ] = (
            _validate_workflow_payload(
                workflow_id=
                    normalized_workflow_id,

                raw=
                    raw_workflow,
            )
        )


    return {
        "rule_version":
            rule_version,

        "workflows":
            validated_workflows,
    }


# ========================================================
# REPLACE COMPLETE STORE SCOPE
# ========================================================


def replace_report_selection_sqlite_payload(
    *,
    store_path: Path,
    payload: dict,
    fallback_rule_version: str,
    legacy_json_imported: bool = True,
) -> None:
    validated = (
        _validate_store_payload(
            payload,

            fallback_rule_version=
                fallback_rule_version,
        )
    )


    scope = (
        report_selection_store_scope(
            store_path
        )
    )


    with sqlite_connection(
        write=True
    ) as connection:
        connection.execute(
            """
            DELETE FROM report_selection_workflows
            WHERE store_root = ?
            """,
            (
                scope,
            ),
        )


        for (
            workflow_id,
            workflow_payload,
        ) in (
            validated[
                "workflows"
            ].items()
        ):
            connection.execute(
                """
                INSERT INTO report_selection_workflows (
                    store_root,
                    workflow_id,
                    revision,
                    payload_json,
                    updated_at
                )
                VALUES (
                    ?,
                    ?,
                    ?,
                    ?,
                    ?
                )
                """,
                (
                    scope,
                    workflow_id,
                    int(
                        workflow_payload[
                            "revision"
                        ]
                    ),
                    json.dumps(
                        workflow_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(
                            ",",
                            ":",
                        ),
                    ),
                    utc_now_iso(),
                ),
            )


        connection.execute(
            """
            INSERT INTO report_selection_store_state (
                store_root,
                legacy_json_imported,
                legacy_rule_version,
                migrated_at
            )
            VALUES (
                ?,
                ?,
                ?,
                ?
            )
            ON CONFLICT(store_root)
            DO UPDATE SET
                legacy_json_imported =
                    excluded.legacy_json_imported,
                legacy_rule_version =
                    excluded.legacy_rule_version,
                migrated_at =
                    excluded.migrated_at
            """,
            (
                scope,
                (
                    1
                    if legacy_json_imported
                    else
                    0
                ),
                str(
                    validated[
                        "rule_version"
                    ]
                ),
                utc_now_iso(),
            ),
        )


# ========================================================
# READ COMPLETE STORE SCOPE
# ========================================================


def load_report_selection_sqlite_payload(
    *,
    store_path: Path,
    fallback_rule_version: str,
) -> dict[
    str,
    Any,
]:
    scope = (
        report_selection_store_scope(
            store_path
        )
    )


    with sqlite_connection(
        write=False
    ) as connection:
        rows = connection.execute(
            """
            SELECT
                workflow_id,
                revision,
                payload_json
            FROM report_selection_workflows
            WHERE store_root = ?
            ORDER BY workflow_id
            """,
            (
                scope,
            ),
        ).fetchall()


        store_state = connection.execute(
            """
            SELECT
                legacy_rule_version
            FROM report_selection_store_state
            WHERE store_root = ?
            """,
            (
                scope,
            ),
        ).fetchone()


    workflows = {}


    for row in rows:
        workflow_id = str(
            row[
                "workflow_id"
            ]
        )


        try:
            raw = json.loads(
                str(
                    row[
                        "payload_json"
                    ]
                )
            )

        except Exception as error:
            raise ReportSelectionSQLiteStoreError(
                (
                    "Report selection SQLite "
                    "payload_json is invalid."
                )
            ) from error


        validated = (
            _validate_workflow_payload(
                workflow_id=
                    workflow_id,

                raw=
                    raw,
            )
        )


        if (
            int(
                row[
                    "revision"
                ]
            )
            !=
            int(
                validated[
                    "revision"
                ]
            )
        ):
            raise ReportSelectionSQLiteStoreError(
                (
                    "Report selection SQLite revision "
                    "does not match payload_json."
                )
            )


        workflows[
            workflow_id
        ] = validated


    rule_version = (
        fallback_rule_version
    )


    if (
        store_state is not None
        and
        store_state[
            "legacy_rule_version"
        ]
    ):
        rule_version = str(
            store_state[
                "legacy_rule_version"
            ]
        )


    return {
        "rule_version":
            rule_version,

        "workflows":
            workflows,
    }


# ========================================================
# INITIALIZATION
# ========================================================


def report_selection_sqlite_is_initialized(
    *,
    store_path: Path,
) -> bool:
    scope = (
        report_selection_store_scope(
            store_path
        )
    )


    with sqlite_connection(
        write=False
    ) as connection:
        row = connection.execute(
            """
            SELECT 1
            FROM report_selection_store_state
            WHERE store_root = ?
            """,
            (
                scope,
            ),
        ).fetchone()


    return (
        row is not None
    )


# ========================================================
# LEGACY JSON IMPORT
# ========================================================


def import_legacy_report_selection_if_needed(
    *,
    store_path: Path,
    fallback_rule_version: str,
) -> bool:
    """
    Import report_selection.json exactly once for one
    configured store path.

    Returns True only when this call initializes the scope.
    """

    with _MIGRATION_LOCK:
        if (
            report_selection_sqlite_is_initialized(
                store_path=
                    store_path
            )
        ):
            return False


        if store_path.exists():
            try:
                raw = json.loads(
                    store_path.read_text(
                        encoding="utf-8"
                    )
                )

            except Exception as error:
                raise ReportSelectionSQLiteStoreError(
                    (
                        "Legacy report selection JSON "
                        "could not be read."
                    )
                ) from error

        else:
            raw = {
                "rule_version":
                    fallback_rule_version,

                "workflows":
                    {},
            }


        validated = (
            _validate_store_payload(
                raw,

                fallback_rule_version=
                    fallback_rule_version,
            )
        )


        replace_report_selection_sqlite_payload(
            store_path=
                store_path,

            payload=
                validated,

            fallback_rule_version=
                fallback_rule_version,

            legacy_json_imported=
                True,
        )


        return True


# ========================================================
# DELETE SCOPE
# ========================================================


def delete_report_selection_sqlite_scope(
    *,
    store_path: Path,
) -> None:
    scope = (
        report_selection_store_scope(
            store_path
        )
    )


    with sqlite_connection(
        write=True
    ) as connection:
        connection.execute(
            """
            DELETE FROM report_selection_workflows
            WHERE store_root = ?
            """,
            (
                scope,
            ),
        )


        connection.execute(
            """
            DELETE FROM report_selection_store_state
            WHERE store_root = ?
            """,
            (
                scope,
            ),
        )
