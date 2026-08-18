from __future__ import annotations

from dataclasses import (
    dataclass,
)

from typing import (
    Any,
)

import pandas as pd

from app.preparation.cleaning_engine import (
    CleaningExecutionResult,
    CleaningPlan,
)

from app.preparation.preparation_artifact_store import (
    PreparationDatasetArtifactInfo,
    put_preparation_artifact,
)


# ============================================================
# VERSION
# ============================================================

CLEANING_ARTIFACT_BRIDGE_VERSION = (
    "cleaning_artifact_bridge_v0.1"
)


# ============================================================
# REPORT
# ============================================================

@dataclass(
    frozen=True,
)
class CleaningArtifactMaterializationReport:
    """
    Result of materializing one Cleaning stage into the
    server-owned Preparation Artifact Store.

    No DataFrame is exposed by this report.
    """

    workflow_id: str

    dataset_ids: tuple[
        str,
        ...
    ]

    artifact_count: int

    materialization_kind: str

    bridge_version: str = (
        CLEANING_ARTIFACT_BRIDGE_VERSION
    )


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


# ============================================================
# SOURCE RECORD INDEX
# ============================================================

def _source_record_index(
    source_dataset_records: list[
        dict[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    dict[
        str,
        Any,
    ],
]:
    """
    Validate and index DataLens ingestion records.

    The function performs all validation before any artifact
    is written.
    """

    if not (
        source_dataset_records
    ):

        raise ValueError(
            "Cleaning artifact materialization requires "
            "at least one source dataset record."
        )


    output: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}


    for record in (
        source_dataset_records
    ):

        dataset_id = (
            str(
                record.get(
                    "dataset_id"
                )
                or
                ""
            )
            .strip()
        )


        if not dataset_id:

            raise ValueError(
                "Source dataset record is missing "
                "dataset_id."
            )


        if dataset_id in output:

            raise ValueError(
                "Duplicate source dataset_id during "
                "Cleaning artifact materialization: "
                f"{dataset_id}"
            )


        filename = (
            str(
                record.get(
                    "filename"
                )
                or
                dataset_id
            )
            .strip()
        )


        if not filename:

            raise ValueError(
                "Source dataset record is missing a valid "
                f"filename. dataset_id={dataset_id}"
            )


        dataframe = (
            record.get(
                "dataframe"
            )
        )


        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):

            raise TypeError(
                "Source dataset record must contain a "
                "pandas DataFrame. "
                f"dataset_id={dataset_id}"
            )


        if dataframe.empty:

            raise ValueError(
                "Source dataset cannot be materialized when "
                "it contains no rows. "
                f"dataset_id={dataset_id}"
            )


        output[
            dataset_id
        ] = (
            record
        )


    return (
        output
    )


# ============================================================
# PROVENANCE INDEX
# ============================================================

def _execution_provenance_index(
    execution: CleaningExecutionResult,
) -> dict[
    str,
    Any,
]:
    """
    Index Cleaning Executor provenance without trusting the
    caller to map datasets manually.
    """

    output: dict[
        str,
        Any,
    ] = {}


    for provenance in (
        execution.provenance
    ):

        dataset_id = (
            str(
                provenance.dataset_id
            )
            .strip()
        )


        if not dataset_id:

            raise ValueError(
                "Cleaning execution provenance contains "
                "an empty dataset_id."
            )


        if dataset_id in output:

            raise ValueError(
                "Cleaning execution contains duplicate "
                "provenance for dataset_id="
                f"{dataset_id}"
            )


        output[
            dataset_id
        ] = (
            provenance
        )


    return (
        output
    )


# ============================================================
# PASSTHROUGH MATERIALIZATION
# ============================================================

def materialize_skipped_cleaning_artifacts(
    *,
    workflow_id: str,
    source_dataset_records: list[
        dict[
            str,
            Any,
        ]
    ],
    cleaning_plan: CleaningPlan,
) -> CleaningArtifactMaterializationReport:
    """
    Materialize unchanged source datasets when deterministic
    Cleaning determined that no cleaning action is required.

    Important:
    this function refuses to act as a bypass when the plan
    actually contains cleaning actions.
    """

    normalized_workflow_id = (
        _required_text(
            workflow_id,
            field_name="workflow_id",
        )
    )


    if (
        cleaning_plan.action_count
        != 0
    ):

        raise ValueError(
            "Skipped Cleaning materialization is allowed "
            "only when cleaning_plan.action_count == 0."
        )


    record_index = (
        _source_record_index(
            source_dataset_records
        )
    )


    # ========================================================
    # PREVALIDATION COMPLETE
    #
    # No store write occurs before this point.
    # ========================================================

    artifacts: list[
        PreparationDatasetArtifactInfo
    ] = []


    for (
        dataset_id,
        record,
    ) in (
        record_index.items()
    ):

        filename = (
            str(
                record.get(
                    "filename"
                )
                or
                dataset_id
            )
        )


        dataframe = (
            record[
                "dataframe"
            ]
        )


        artifact = (
            put_preparation_artifact(
                workflow_id=(
                    normalized_workflow_id
                ),

                dataset_id=(
                    dataset_id
                ),

                dataset_filename=(
                    filename
                ),

                # CLEAN was skipped, therefore the current
                # materialization is still the original
                # source dataset.
                stage=(
                    "source"
                ),

                dataframe=(
                    dataframe
                ),

                parent_dataset_ids=[],

                evidence_refs=[
                    (
                        "cleaning:skipped"
                    ),

                    (
                        "cleaning_plan:"
                        f"{cleaning_plan.rule_version}"
                    ),
                ],

                replace=True,
            )
        )


        artifacts.append(
            artifact
        )


    dataset_ids = tuple(
        artifact.dataset_id

        for artifact
        in artifacts
    )


    return (
        CleaningArtifactMaterializationReport(
            workflow_id=(
                normalized_workflow_id
            ),

            dataset_ids=(
                dataset_ids
            ),

            artifact_count=(
                len(
                    artifacts
                )
            ),

            materialization_kind=(
                "source_passthrough"
            ),
        )
    )


# ============================================================
# EXECUTED CLEANING MATERIALIZATION
# ============================================================

def materialize_cleaning_execution_artifacts(
    *,
    workflow_id: str,
    source_dataset_records: list[
        dict[
            str,
            Any,
        ]
    ],
    cleaning_plan: CleaningPlan,
    derived_frames: dict[
        str,
        pd.DataFrame,
    ],
    execution: CleaningExecutionResult,
) -> CleaningArtifactMaterializationReport:
    """
    Persist exact DataFrames returned by Cleaning Executor.

    Safety rules:
    - the Cleaning plan must actually contain actions;
    - every source dataset must have one derived DataFrame;
    - every derived DataFrame must have executor provenance;
    - unknown derived dataset IDs are rejected;
    - missing datasets are rejected;
    - validation happens before the first store write.

    The original DataFrames remain untouched.
    """

    normalized_workflow_id = (
        _required_text(
            workflow_id,
            field_name="workflow_id",
        )
    )


    if (
        cleaning_plan.action_count
        <= 0
    ):

        raise ValueError(
            "Executed Cleaning materialization requires "
            "a CleaningPlan containing at least one action."
        )


    record_index = (
        _source_record_index(
            source_dataset_records
        )
    )


    provenance_index = (
        _execution_provenance_index(
            execution
        )
    )


    expected_dataset_ids = set(
        record_index.keys()
    )


    derived_dataset_ids = set(
        derived_frames.keys()
    )


    provenance_dataset_ids = set(
        provenance_index.keys()
    )


    # ========================================================
    # EXACT DATASET RECONCILIATION
    # ========================================================

    if (
        derived_dataset_ids
        != expected_dataset_ids
    ):

        missing = sorted(
            expected_dataset_ids
            -
            derived_dataset_ids
        )


        unexpected = sorted(
            derived_dataset_ids
            -
            expected_dataset_ids
        )


        raise ValueError(
            "Cleaning derived dataset scope does not match "
            "the source dataset scope. "
            f"missing={missing}, "
            f"unexpected={unexpected}"
        )


    if (
        provenance_dataset_ids
        != expected_dataset_ids
    ):

        missing = sorted(
            expected_dataset_ids
            -
            provenance_dataset_ids
        )


        unexpected = sorted(
            provenance_dataset_ids
            -
            expected_dataset_ids
        )


        raise ValueError(
            "Cleaning provenance dataset scope does not "
            "match the source dataset scope. "
            f"missing={missing}, "
            f"unexpected={unexpected}"
        )


    # ========================================================
    # VALIDATE EVERY DERIVED FRAME BEFORE WRITING
    # ========================================================

    for dataset_id in (
        expected_dataset_ids
    ):

        dataframe = (
            derived_frames[
                dataset_id
            ]
        )


        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):

            raise TypeError(
                "Cleaning Executor derived artifact must "
                "be a pandas DataFrame. "
                f"dataset_id={dataset_id}"
            )


        if dataframe.empty:

            raise ValueError(
                "Cleaning Executor produced an empty "
                "downstream dataset. "
                f"dataset_id={dataset_id}"
            )


        provenance = (
            provenance_index[
                dataset_id
            ]
        )


        if (
            int(
                dataframe.shape[
                    0
                ]
            )
            != int(
                provenance.rows_after
            )
        ):

            raise ValueError(
                "Cleaning Executor provenance row count "
                "does not match the derived DataFrame. "
                f"dataset_id={dataset_id}"
            )


        if (
            int(
                dataframe.shape[
                    1
                ]
            )
            != int(
                provenance.columns_after
            )
        ):

            raise ValueError(
                "Cleaning Executor provenance column count "
                "does not match the derived DataFrame. "
                f"dataset_id={dataset_id}"
            )


    # ========================================================
    # PREVALIDATION COMPLETE
    #
    # No artifact has been written before this point.
    # ========================================================

    artifacts: list[
        PreparationDatasetArtifactInfo
    ] = []


    for dataset_id in (
        record_index.keys()
    ):

        record = (
            record_index[
                dataset_id
            ]
        )


        dataframe = (
            derived_frames[
                dataset_id
            ]
        )


        provenance = (
            provenance_index[
                dataset_id
            ]
        )


        filename = (
            str(
                record.get(
                    "filename"
                )
                or
                dataset_id
            )
        )


        evidence_refs = [
            (
                "cleaning_plan:"
                f"{cleaning_plan.rule_version}"
            ),

            (
                "cleaning_execution:"
                f"{execution.rule_version}"
            ),

            (
                "cleaning_source_fingerprint:"
                f"{provenance.source_fingerprint}"
            ),

            (
                "cleaning_derived_fingerprint:"
                f"{provenance.derived_fingerprint}"
            ),
        ]


        for action_id in (
            provenance.applied_action_ids
        ):

            evidence_refs.append(
                "cleaning_action:"
                f"{action_id}"
            )


        artifact = (
            put_preparation_artifact(
                workflow_id=(
                    normalized_workflow_id
                ),

                dataset_id=(
                    dataset_id
                ),

                dataset_filename=(
                    filename
                ),

                stage=(
                    "clean"
                ),

                dataframe=(
                    dataframe
                ),

                # Same logical dataset, new materialized
                # version after Cleaning.
                parent_dataset_ids=[
                    dataset_id,
                ],

                evidence_refs=(
                    evidence_refs
                ),

                replace=True,
            )
        )


        artifacts.append(
            artifact
        )


    dataset_ids = tuple(
        artifact.dataset_id

        for artifact
        in artifacts
    )


    return (
        CleaningArtifactMaterializationReport(
            workflow_id=(
                normalized_workflow_id
            ),

            dataset_ids=(
                dataset_ids
            ),

            artifact_count=(
                len(
                    artifacts
                )
            ),

            materialization_kind=(
                "cleaned"
            ),
        )
    )