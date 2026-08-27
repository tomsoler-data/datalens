from __future__ import annotations

from copy import (
    deepcopy,
)

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from app.persistence.sqlite_database import (
    ensure_ephemeral_sqlite_test_path,
    sqlite_connection,
    utc_now_iso,
)


# ========================================================
# VERSION
# ========================================================


PREPARATION_UI_STATE_RULE_VERSION = (
    "preparation_ui_state_v0.1"
)

PREPARATION_UI_STATE_STORE_VERSION = (
    "preparation_ui_state_sqlite_store_v0.1"
)


# ========================================================
# READ MODEL
# ========================================================


class PreparationUiStateView(
    BaseModel,
):
    """
    Server-owned committed structured outputs used to
    rehydrate the Preparation UI after a browser refresh
    or a backend process restart.

    No DataFrame and no browser-only draft state are stored.

    PREPARATION_UI_STATE_SQLITE_STORE_V0_1
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    workflow_id: str

    revision: int = Field(
        default=0,
        ge=0,
    )

    quality_report: Optional[
        Dict[
            str,
            Any,
        ]
    ] = None

    cleaning_plan: Optional[
        Dict[
            str,
            Any,
        ]
    ] = None

    cleaning_execution: Optional[
        Dict[
            str,
            Any,
        ]
    ] = None

    semantic_review: Optional[
        Dict[
            str,
            Any,
        ]
    ] = None

    semantic_cleaning_plan: Optional[
        Dict[
            str,
            Any,
        ]
    ] = None

    semantic_cleaning_execution: Optional[
        Dict[
            str,
            Any,
        ]
    ] = None

    semantic_confirmation: Optional[
        Dict[
            str,
            Any,
        ]
    ] = None

    applied_semantic_choices: List[
        Dict[
            str,
            Any,
        ]
    ] = Field(
        default_factory=list
    )

    confirmed_semantic_issue_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    semantic_manual_resolutions: List[
        Dict[
            str,
            Any,
        ]
    ] = Field(
        default_factory=list
    )

    storage: str = (
        "sqlite"
    )

    persistent: bool = (
        True
    )

    rule_version: str = (
        PREPARATION_UI_STATE_RULE_VERSION
    )


# ========================================================
# ERRORS
# ========================================================


class PreparationUiStateStoreError(
    RuntimeError,
):
    pass


# ========================================================
# SENTINEL
# ========================================================


_UNSET = object()


# ========================================================
# NORMALIZATION
# ========================================================


def _normalize_workflow_id(
    workflow_id: str,
) -> str:
    normalized = (
        workflow_id.strip()
    )

    if not normalized:
        raise ValueError(
            (
                "Preparation UI state workflow_id "
                "cannot be empty."
            )
        )

    return normalized


# ========================================================
# EMPTY STATE
# ========================================================


def _empty_state(
    workflow_id: str,
) -> PreparationUiStateView:
    return (
        PreparationUiStateView(
            workflow_id=
                workflow_id,

            storage=
                "sqlite",

            persistent=
                True,
        )
    )


# ========================================================
# SERIALIZATION
# ========================================================


def _serialize_state(
    state: PreparationUiStateView,
) -> str:
    return (
        state.model_dump_json()
    )


def _deserialize_state(
    *,
    workflow_id: str,
    revision: int,
    payload_json: str,
) -> PreparationUiStateView:
    try:
        state = (
            PreparationUiStateView
            .model_validate_json(
                payload_json
            )
        )

    except Exception as error:
        raise PreparationUiStateStoreError(
            (
                "Persisted Preparation UI-state "
                "payload is invalid. "
                f"workflow_id={workflow_id}"
            )
        ) from error


    if (
        state.workflow_id
        !=
        workflow_id
    ):
        raise PreparationUiStateStoreError(
            (
                "Persisted Preparation UI-state "
                "workflow_id does not match its "
                "SQLite primary key. "
                f"workflow_id={workflow_id}"
            )
        )


    if (
        state.revision
        !=
        revision
    ):
        raise PreparationUiStateStoreError(
            (
                "Persisted Preparation UI-state "
                "revision does not match its "
                "SQLite revision column. "
                f"workflow_id={workflow_id}, "
                f"payload_revision={state.revision}, "
                f"column_revision={revision}"
            )
        )


    return (
        state.model_copy(
            deep=True
        )
    )


# ========================================================
# READ
# ========================================================


def get_preparation_ui_state(
    workflow_id: str,
) -> PreparationUiStateView:
    normalized = (
        _normalize_workflow_id(
            workflow_id
        )
    )


    with sqlite_connection(
        write=False
    ) as connection:
        row = (
            connection.execute(
                """
                SELECT
                    workflow_id,
                    revision,
                    payload_json
                FROM preparation_ui_state
                WHERE workflow_id = ?
                """,
                (
                    normalized,
                ),
            )
            .fetchone()
        )


    if row is None:
        return (
            _empty_state(
                normalized
            )
            .model_copy(
                deep=True
            )
        )


    return (
        _deserialize_state(
            workflow_id=
                str(
                    row[
                        "workflow_id"
                    ]
                ),

            revision=
                int(
                    row[
                        "revision"
                    ]
                ),

            payload_json=
                str(
                    row[
                        "payload_json"
                    ]
                ),
        )
    )


# ========================================================
# UPDATE
# ========================================================


def update_preparation_ui_state(
    *,
    workflow_id: str,

    quality_report: object = _UNSET,

    cleaning_plan: object = _UNSET,
    cleaning_execution: object = _UNSET,

    semantic_review: object = _UNSET,
    semantic_cleaning_plan: object = _UNSET,
    semantic_cleaning_execution: object = _UNSET,
    semantic_confirmation: object = _UNSET,

    applied_semantic_choices: object = _UNSET,

    confirmed_semantic_issue_ids: object = _UNSET,

    semantic_manual_resolutions: object = _UNSET,
) -> PreparationUiStateView:
    """
    Persist already-produced server-side structured outputs.

    _UNSET:
        preserve current value.

    None:
        explicitly clear current value.

    BEGIN IMMEDIATE is supplied by sqlite_connection()
    so local writes are serialized transactionally.
    """

    normalized = (
        _normalize_workflow_id(
            workflow_id
        )
    )


    supplied = {
        "quality_report":
            quality_report,

        "cleaning_plan":
            cleaning_plan,

        "cleaning_execution":
            cleaning_execution,

        "semantic_review":
            semantic_review,

        "semantic_cleaning_plan":
            semantic_cleaning_plan,

        "semantic_cleaning_execution":
            semantic_cleaning_execution,

        "semantic_confirmation":
            semantic_confirmation,

        "applied_semantic_choices":
            applied_semantic_choices,

        "confirmed_semantic_issue_ids":
            confirmed_semantic_issue_ids,

        "semantic_manual_resolutions":
            semantic_manual_resolutions,
    }


    with sqlite_connection(
        write=True
    ) as connection:
        row = (
            connection.execute(
                """
                SELECT
                    workflow_id,
                    revision,
                    payload_json
                FROM preparation_ui_state
                WHERE workflow_id = ?
                """,
                (
                    normalized,
                ),
            )
            .fetchone()
        )


        if row is None:
            current = (
                _empty_state(
                    normalized
                )
            )

            exists = False

        else:
            current = (
                _deserialize_state(
                    workflow_id=
                        str(
                            row[
                                "workflow_id"
                            ]
                        ),

                    revision=
                        int(
                            row[
                                "revision"
                            ]
                        ),

                    payload_json=
                        str(
                            row[
                                "payload_json"
                            ]
                        ),
                )
            )

            exists = True


        payload = (
            current.model_dump(
                mode="python"
            )
        )


        for (
            field_name,
            value,
        ) in supplied.items():
            if (
                value
                is
                _UNSET
            ):
                continue

            payload[
                field_name
            ] = deepcopy(
                value
            )


        payload[
            "workflow_id"
        ] = normalized

        payload[
            "revision"
        ] = (
            current.revision
            +
            1
        )

        payload[
            "storage"
        ] = "sqlite"

        payload[
            "persistent"
        ] = True

        payload[
            "rule_version"
        ] = (
            PREPARATION_UI_STATE_RULE_VERSION
        )


        candidate = (
            PreparationUiStateView
            .model_validate(
                payload
            )
        )


        now = (
            utc_now_iso()
        )


        if not exists:
            connection.execute(
                """
                INSERT INTO preparation_ui_state (
                    workflow_id,
                    revision,
                    payload_json,
                    created_at,
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
                    candidate.workflow_id,
                    candidate.revision,
                    _serialize_state(
                        candidate
                    ),
                    now,
                    now,
                ),
            )

        else:
            cursor = (
                connection.execute(
                    """
                    UPDATE preparation_ui_state
                    SET
                        revision = ?,
                        payload_json = ?,
                        updated_at = ?
                    WHERE
                        workflow_id = ?
                        AND
                        revision = ?
                    """,
                    (
                        candidate.revision,
                        _serialize_state(
                            candidate
                        ),
                        now,
                        normalized,
                        current.revision,
                    ),
                )
            )


            if (
                cursor.rowcount
                !=
                1
            ):
                raise PreparationUiStateStoreError(
                    (
                        "Preparation UI state changed "
                        "during commit. "
                        f"workflow_id={normalized}, "
                        "expected_revision="
                        f"{current.revision}"
                    )
                )


    return (
        candidate.model_copy(
            deep=True
        )
    )


# ========================================================
# DELETE
# ========================================================


def delete_preparation_ui_state(
    workflow_id: str,
) -> None:
    normalized = (
        _normalize_workflow_id(
            workflow_id
        )
    )


    with sqlite_connection(
        write=True
    ) as connection:
        connection.execute(
            """
            DELETE FROM preparation_ui_state
            WHERE workflow_id = ?
            """,
            (
                normalized,
            ),
        )


# ========================================================
# RESET - TESTS ONLY
# ========================================================


def reset_preparation_ui_state_store_for_tests(
) -> None:
    """
    Test-only helper.

    Never delete production UI state accidentally.
    When no isolated storage path was configured,
    move execution to an ephemeral SQLite database.
    """

    ensure_ephemeral_sqlite_test_path(
        namespace=
            "preparation-ui-state-tests"
    )


    with sqlite_connection(
        write=True
    ) as connection:
        connection.execute(
            """
            DELETE FROM preparation_ui_state
            """
        )
