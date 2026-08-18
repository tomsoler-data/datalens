from __future__ import annotations

from dataclasses import (
    dataclass,
)

import pandas as pd

from app.preparation.join_approval import (
    ApprovedJoin,
    ApprovedJoinPlan,
)

from app.preparation.join_executor import (
    JoinExecutionResult,
)

from app.preparation.post_join_validation import (
    PostJoinValidationReport,
)

from app.preparation.preparation_artifact_store import (
    PreparationDatasetArtifact,
    PreparationDatasetArtifactInfo,
    get_preparation_artifact,
    list_preparation_artifacts,
    put_preparation_artifact,
)


# ============================================================
# VERSION
# ============================================================

JOIN_ARTIFACT_BRIDGE_VERSION = (
    "join_artifact_bridge_v0.1"
)


# ============================================================
# RESULT
# ============================================================

@dataclass(
    frozen=True,
)
class JoinArtifactMaterializationReport:
    """
    Result of persisting independently validated JOIN outputs
    into the server-owned Preparation Artifact Store.

    No DataFrame is exposed by this report.
    """

    workflow_id: str

    output_dataset_ids: tuple[
        str,
        ...
    ]

    artifact_count: int

    bridge_version: str = (
        JOIN_ARTIFACT_BRIDGE_VERSION
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


    return (
        normalized
    )


# ============================================================
# SOURCE MAP VALIDATION
# ============================================================

def _validate_source_datasets(
    source_datasets: dict[
        str,
        pd.DataFrame,
    ],
) -> dict[
    str,
    pd.DataFrame,
]:

    if not isinstance(
        source_datasets,
        dict,
    ):

        raise TypeError(
            "JOIN source_datasets must be a dataset → "
            "DataFrame mapping."
        )


    if not source_datasets:

        raise ValueError(
            "JOIN materialization requires at least one "
            "source dataset."
        )


    output: dict[
        str,
        pd.DataFrame,
    ] = {}


    for (
        raw_dataset_id,
        dataframe,
    ) in source_datasets.items():

        dataset_id = (
            _required_text(
                str(
                    raw_dataset_id
                ),

                field_name=(
                    "source dataset_id"
                ),
            )
        )


        if dataset_id in output:

            raise ValueError(
                "Duplicate JOIN source dataset_id="
                f"{dataset_id}"
            )


        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):

            raise TypeError(
                "JOIN source dataset must be a pandas "
                "DataFrame. "
                f"dataset_id={dataset_id}"
            )


        if dataframe.empty:

            raise ValueError(
                "JOIN source dataset cannot be empty. "
                f"dataset_id={dataset_id}"
            )


        output[
            dataset_id
        ] = (
            dataframe
        )


    return (
        output
    )


# ============================================================
# SERVER TRUST ANCHOR
# ============================================================

def _require_current_source_artifact(
    *,
    workflow_id: str,
    dataset_id: str,
    dataframe: pd.DataFrame,
) -> PreparationDatasetArtifact:
    """
    JOIN must consume exactly the current server-owned
    Preparation artifact.

    Allowed upstream material stages:
    - source;
    - clean;
    - transform.

    COMBINE output is intentionally not accepted as a new
    input in v0.1. Chained combine operations can be added
    later with an explicit lineage contract.
    """

    artifact = (
        get_preparation_artifact(
            workflow_id=(
                workflow_id
            ),

            dataset_id=(
                dataset_id
            ),
        )
    )


    if (
        artifact.stage
        not in {
            "source",
            "clean",
            "transform",
        }
    ):

        raise ValueError(
            "JOIN cannot consume an artifact from an "
            "unsupported Preparation stage. "
            f"dataset_id={dataset_id}, "
            f"stage={artifact.stage}"
        )


    if not (
        artifact
        .dataframe
        .equals(
            dataframe
        )
    ):

        raise ValueError(
            "JOIN source DataFrame does not match the "
            "current server-owned Preparation artifact. "
            f"dataset_id={dataset_id}"
        )


    return (
        artifact
    )


# ============================================================
# APPROVED JOIN INDEX
# ============================================================

def _executable_join_index(
    approved_plan: ApprovedJoinPlan,
) -> dict[
    str,
    ApprovedJoin,
]:

    output: dict[
        str,
        ApprovedJoin,
    ] = {}


    for join in (
        approved_plan.joins
    ):

        if not (
            join.executable
        ):

            continue


        output_dataset_id = (
            _required_text(
                join.output_dataset_id,

                field_name=(
                    "ApprovedJoin.output_dataset_id"
                ),
            )
        )


        if output_dataset_id in output:

            raise ValueError(
                "Approved JOIN plan contains duplicate "
                "executable output_dataset_id="
                f"{output_dataset_id}"
            )


        output[
            output_dataset_id
        ] = (
            join
        )


    if (
        len(
            output
        )
        !=
        approved_plan.executable_join_count
    ):

        raise ValueError(
            "Approved JOIN plan executable_join_count does "
            "not match its executable joins."
        )


    return (
        output
    )


# ============================================================
# OUTPUT COLLISION GUARD
# ============================================================

def _require_output_ids_available(
    *,
    workflow_id: str,
    output_dataset_ids: set[
        str
    ],
) -> None:
    """
    JOIN outputs create new logical datasets.

    An output is never allowed to silently replace an existing
    Preparation artifact.
    """

    existing_dataset_ids = {
        artifact.dataset_id

        for artifact in
        list_preparation_artifacts(
            workflow_id=(
                workflow_id
            )
        )
    }


    collisions = sorted(
        output_dataset_ids
        &
        existing_dataset_ids
    )


    if collisions:

        raise ValueError(
            "JOIN output dataset_id already exists in the "
            "Preparation Artifact Store. "
            f"collisions={collisions}"
        )


# ============================================================
# SOURCE SCOPE
# ============================================================

def _required_source_dataset_ids(
    executable_joins: dict[
        str,
        ApprovedJoin,
    ],
) -> set[
    str
]:

    output: set[
        str
    ] = set()


    for join in (
        executable_joins.values()
    ):

        output.add(
            _required_text(
                join.left_dataset_id,

                field_name=(
                    "ApprovedJoin.left_dataset_id"
                ),
            )
        )


        output.add(
            _required_text(
                join.right_dataset_id,

                field_name=(
                    "ApprovedJoin.right_dataset_id"
                ),
            )
        )


    return (
        output
    )


# ============================================================
# EVIDENCE
# ============================================================

def _join_evidence_refs(
    *,
    join: ApprovedJoin,
    approved_plan: ApprovedJoinPlan,
    execution: JoinExecutionResult,
    validation: PostJoinValidationReport,
) -> list[
    str
]:

    join_type = (
        join.join_type.value

        if hasattr(
            join.join_type,
            "value",
        )

        else str(
            join.join_type
        )
    )


    cardinality = (
        join.detected_cardinality.value

        if hasattr(
            join.detected_cardinality,
            "value",
        )

        else str(
            join.detected_cardinality
        )
    )


    refs = [
        (
            "join_approval:"
            f"{approved_plan.rule_version}"
        ),

        (
            "join_execution:"
            f"{execution.report.rule_version}"
        ),

        (
            "join_validation:"
            f"{validation.rule_version}"
        ),

        (
            "join_request:"
            f"{join.request_id}"
        ),

        (
            "join_type:"
            f"{join_type}"
        ),

        (
            "join_cardinality:"
            f"{cardinality}"
        ),

        (
            "join_left_dataset:"
            f"{join.left_dataset_id}"
        ),

        (
            "join_right_dataset:"
            f"{join.right_dataset_id}"
        ),
    ]


    for key_pair in (
        join.keys
    ):

        refs.append(
            (
                "join_key:"
                f"{key_pair.left_column}"
                "="
                f"{key_pair.right_column}"
            )
        )


    return (
        refs
    )


# ============================================================
# MATERIALIZATION
# ============================================================

def materialize_join_artifacts(
    *,
    workflow_id: str,
    source_datasets: dict[
        str,
        pd.DataFrame,
    ],
    approved_plan: ApprovedJoinPlan,
    execution: JoinExecutionResult,
    validation: PostJoinValidationReport,
) -> JoinArtifactMaterializationReport:
    """
    Persist only JOIN outputs explicitly authorized by:

    1. Join Planner;
    2. human approval;
    3. Join Executor;
    4. independent Post-Join Validation.

    Safety guarantees
    -----------------

    - validation must explicitly allow downstream use;

    - every executable JOIN output must exist in the executor
      result;

    - no unexpected executor output may be persisted;

    - JOIN source datasets must exactly match the current
      server-owned Preparation artifacts;

    - JOIN output dataset IDs may not overwrite any existing
      Preparation artifact;

    - all validation happens before the first store write.
    """

    normalized_workflow_id = (
        _required_text(
            workflow_id,

            field_name=(
                "workflow_id"
            ),
        )
    )


    datasets = (
        _validate_source_datasets(
            source_datasets
        )
    )


    # ========================================================
    # VALIDATION AUTHORITY
    # ========================================================

    if not (
        validation.valid_for_downstream
    ):

        raise ValueError(
            "JOIN output failed independent post-join "
            "validation and cannot be materialized."
        )


    if (
        validation.status
        !=
        "passed"
    ):

        raise ValueError(
            "JOIN validation status must be 'passed' before "
            "artifact materialization."
        )


    if not (
        approved_plan.ready_for_execution
    ):

        raise ValueError(
            "Approved JOIN plan is not ready for execution."
        )


    # ========================================================
    # EXECUTABLE OUTPUT CONTRACT
    # ========================================================

    executable_joins = (
        _executable_join_index(
            approved_plan
        )
    )


    expected_output_ids = set(
        executable_joins.keys()
    )


    actual_output_ids = set(
        execution
        .joined_datasets
        .keys()
    )


    if (
        actual_output_ids
        !=
        expected_output_ids
    ):

        missing = sorted(
            expected_output_ids
            -
            actual_output_ids
        )


        unexpected = sorted(
            actual_output_ids
            -
            expected_output_ids
        )


        raise ValueError(
            "JOIN executor output scope does not match the "
            "approved executable JOIN outputs. "
            f"missing={missing}, "
            f"unexpected={unexpected}"
        )


    # ========================================================
    # EXECUTION REPORT RECONCILIATION
    # ========================================================

    if (
        int(
            execution.report.executable_join_count
        )
        !=
        len(
            expected_output_ids
        )
    ):

        raise ValueError(
            "JOIN execution executable_join_count does not "
            "match approved executable joins."
        )


    if (
        int(
            execution.report.output_dataset_count
        )
        !=
        len(
            actual_output_ids
        )
    ):

        raise ValueError(
            "JOIN execution output_dataset_count does not "
            "match actual JOIN outputs."
        )


    reported_output_ids = set(
        execution
        .report
        .output_dataset_ids
    )


    if (
        reported_output_ids
        !=
        actual_output_ids
    ):

        raise ValueError(
            "JOIN execution output_dataset_ids do not match "
            "actual JOIN outputs."
        )


    if (
        int(
            validation.output_dataset_count
        )
        !=
        len(
            actual_output_ids
        )
    ):

        raise ValueError(
            "Post-Join Validation output_dataset_count does "
            "not match executor outputs."
        )


    # ========================================================
    # SOURCE TRUST ANCHORS
    # ========================================================

    required_source_ids = (
        _required_source_dataset_ids(
            executable_joins
        )
    )


    missing_source_ids = sorted(
        required_source_ids
        -
        set(
            datasets.keys()
        )
    )


    if missing_source_ids:

        raise ValueError(
            "JOIN materialization is missing required source "
            "datasets. "
            f"missing={missing_source_ids}"
        )


    current_artifacts: dict[
        str,
        PreparationDatasetArtifact,
    ] = {}


    for dataset_id in (
        required_source_ids
    ):

        current_artifacts[
            dataset_id
        ] = (
            _require_current_source_artifact(
                workflow_id=(
                    normalized_workflow_id
                ),

                dataset_id=(
                    dataset_id
                ),

                dataframe=(
                    datasets[
                        dataset_id
                    ]
                ),
            )
        )


    # ========================================================
    # OUTPUT DATAFRAME VALIDATION
    # ========================================================

    for output_dataset_id in (
        expected_output_ids
    ):

        dataframe = (
            execution
            .joined_datasets[
                output_dataset_id
            ]
        )


        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):

            raise TypeError(
                "JOIN executor output must be a pandas "
                "DataFrame. "
                f"dataset_id={output_dataset_id}"
            )


        if dataframe.empty:

            raise ValueError(
                "JOIN executor produced an empty output "
                "dataset. "
                f"dataset_id={output_dataset_id}"
            )


    # ========================================================
    # OUTPUT IDS MUST BE NEW
    # ========================================================

    _require_output_ids_available(
        workflow_id=(
            normalized_workflow_id
        ),

        output_dataset_ids=(
            expected_output_ids
        ),
    )


    # ========================================================
    # PREVALIDATION COMPLETE
    #
    # No Artifact Store write occurred before this point.
    # ========================================================

    persisted: list[
        PreparationDatasetArtifactInfo
    ] = []


    for output_dataset_id in (
        executable_joins.keys()
    ):

        join = (
            executable_joins[
                output_dataset_id
            ]
        )


        output_dataframe = (
            execution
            .joined_datasets[
                output_dataset_id
            ]
        )


        output_filename = (
            _required_text(
                join.output_dataset_filename,

                field_name=(
                    "ApprovedJoin.output_dataset_filename"
                ),
            )
        )


        artifact = (
            put_preparation_artifact(
                workflow_id=(
                    normalized_workflow_id
                ),

                dataset_id=(
                    output_dataset_id
                ),

                dataset_filename=(
                    output_filename
                ),

                stage=(
                    "combine"
                ),

                dataframe=(
                    output_dataframe
                ),

                parent_dataset_ids=[
                    join.left_dataset_id,
                    join.right_dataset_id,
                ],

                evidence_refs=(
                    _join_evidence_refs(
                        join=(
                            join
                        ),

                        approved_plan=(
                            approved_plan
                        ),

                        execution=(
                            execution
                        ),

                        validation=(
                            validation
                        ),
                    )
                ),

                # JOIN outputs are new datasets.
                # Existing artifacts must have been rejected
                # by _require_output_ids_available().
                replace=False,
            )
        )


        persisted.append(
            artifact
        )


    return (
        JoinArtifactMaterializationReport(
            workflow_id=(
                normalized_workflow_id
            ),

            output_dataset_ids=tuple(
                artifact.dataset_id

                for artifact
                in persisted
            ),

            artifact_count=(
                len(
                    persisted
                )
            ),
        )
    )