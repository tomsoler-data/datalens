from __future__ import annotations


import pandas as pd


# ============================================================
# VERSION
# ============================================================


ML_OBSERVED_DATASET_AUTHORITY_RULE_VERSION = (
    "ml_observed_dataset_authority_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLObservedDatasetResolverError(
    RuntimeError
):
    pass


class MLObservedDatasetAuthorityError(
    MLObservedDatasetResolverError
):
    pass


class MLObservedDatasetNotAuthorizedError(
    MLObservedDatasetResolverError
):
    pass


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
        raise MLObservedDatasetAuthorityError(
            (
                f"{field_name} "
                "cannot be empty."
            )
        )


    return normalized


# ============================================================
# PUBLIC AUTHORITY RESOLVER
# ============================================================


def resolve_server_owned_observed_dataframe(
    *,
    handoff,
    workflow_id: str,
    observed_dataset_id: str,
) -> tuple[
    pd.DataFrame,
    int,
]:
    """
    Resolve one observed DataFrame exclusively from one
    server-owned validated Analysis Input Handoff.

    The caller may choose identities but cannot inject:

    - raw rows;
    - DataFrames;
    - Preparation revisions;
    - dataset records;
    - authorization scope.

    Returned DataFrame is always a deep copy.
    """

    normalized_workflow_id = (
        _required_text(
            workflow_id,
            field_name=
                "workflow_id",
        )
    )


    normalized_observed_dataset_id = (
        _required_text(
            observed_dataset_id,
            field_name=
                "observed_dataset_id",
        )
    )


    # ========================================================
    # WORKFLOW AUTHORITY
    # ========================================================


    handoff_workflow_id = (
        _required_text(
            getattr(
                handoff,
                "workflow_id",
                None,
            ),
            field_name=
                "handoff.workflow_id",
        )
    )


    if (
        handoff_workflow_id
        !=
        normalized_workflow_id
    ):
        raise MLObservedDatasetAuthorityError(
            (
                "Analysis Input Handoff workflow "
                "does not match the requested "
                "workflow."
            )
        )


    # ========================================================
    # REVISION AUTHORITY
    # ========================================================


    raw_revision = getattr(
        handoff,
        "session_revision",
        None,
    )


    if (
        not isinstance(
            raw_revision,
            int,
        )
        or
        isinstance(
            raw_revision,
            bool,
        )
        or
        raw_revision < 0
    ):
        raise MLObservedDatasetAuthorityError(
            (
                "Analysis Input Handoff has "
                "no valid Preparation revision."
            )
        )


    # ========================================================
    # AUTHORIZED DATASET SCOPE
    # ========================================================


    raw_dataset_ids = getattr(
        handoff,
        "dataset_ids",
        None,
    )


    if not isinstance(
        raw_dataset_ids,
        tuple,
    ):
        raise MLObservedDatasetAuthorityError(
            (
                "Analysis Input Handoff "
                "dataset scope is invalid."
            )
        )


    authorized_dataset_ids = []


    for raw_dataset_id in (
        raw_dataset_ids
    ):

        dataset_id = (
            _required_text(
                raw_dataset_id,
                field_name=
                    "handoff.dataset_id",
            )
        )


        authorized_dataset_ids.append(
            dataset_id
        )


    if (
        len(
            authorized_dataset_ids
        )
        !=
        len(
            set(
                authorized_dataset_ids
            )
        )
    ):
        raise MLObservedDatasetAuthorityError(
            (
                "Analysis Input Handoff "
                "contains duplicate dataset "
                "identities."
            )
        )


    if (
        normalized_observed_dataset_id
        not in
        authorized_dataset_ids
    ):
        raise MLObservedDatasetNotAuthorizedError(
            (
                "Requested observed dataset "
                "is not authorized by the "
                "current validated Analysis "
                "Input Handoff."
            )
        )


    # ========================================================
    # DATASET RECORDS
    # ========================================================


    raw_records = getattr(
        handoff,
        "dataset_records",
        None,
    )


    if not isinstance(
        raw_records,
        tuple,
    ):
        raise MLObservedDatasetAuthorityError(
            (
                "Analysis Input Handoff "
                "dataset records are invalid."
            )
        )


    records_by_id = {}


    for record in (
        raw_records
    ):

        if not isinstance(
            record,
            dict,
        ):
            raise MLObservedDatasetAuthorityError(
                (
                    "Analysis Input Handoff "
                    "contains an invalid "
                    "dataset record."
                )
            )


        dataset_id = (
            _required_text(
                record.get(
                    "dataset_id"
                ),
                field_name=(
                    "handoff.dataset_record."
                    "dataset_id"
                ),
            )
        )


        if (
            dataset_id
            in
            records_by_id
        ):
            raise MLObservedDatasetAuthorityError(
                (
                    "Analysis Input Handoff "
                    "contains duplicate dataset "
                    "records."
                )
            )


        records_by_id[
            dataset_id
        ] = record


    if (
        set(
            records_by_id
        )
        !=
        set(
            authorized_dataset_ids
        )
    ):
        raise MLObservedDatasetAuthorityError(
            (
                "Analysis Input Handoff "
                "dataset identities and "
                "records are inconsistent."
            )
        )


    # ========================================================
    # TRUSTED DATAFRAME
    # ========================================================


    selected_record = (
        records_by_id[
            normalized_observed_dataset_id
        ]
    )


    dataframe = (
        selected_record.get(
            "dataframe"
        )
    )


    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise MLObservedDatasetAuthorityError(
            (
                "Authorized observed dataset "
                "does not contain a trusted "
                "pandas DataFrame."
            )
        )


    return (
        dataframe.copy(
            deep=True
        ),
        raw_revision,
    )
