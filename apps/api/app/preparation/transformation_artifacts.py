from __future__ import annotations

from dataclasses import (
    dataclass,
)

from typing import (
    Literal,
)

import pandas as pd

from app.preparation.post_transformation_validation import (
    PostTransformationValidationReport,
)

from app.preparation.preparation_artifact_store import (
    PreparationDatasetArtifact,
    PreparationDatasetArtifactInfo,
    get_preparation_artifact,
    put_preparation_artifact,
)

from app.preparation.transformation_approval import (
    ApprovedTransformationPlan,
)

from app.preparation.transformation_executor import (
    TransformationExecutionResult,
)


# ============================================================
# VERSION
# ============================================================

TRANSFORMATION_ARTIFACT_BRIDGE_VERSION = (
    "transformation_artifact_bridge_v0.1"
)


# ============================================================
# TYPES
# ============================================================

TransformationArtifactMaterializationKind = Literal[
    "no_change",
    "source_transformed",
    "derived_only",
    "source_and_derived",
]


# ============================================================
# RESULT
# ============================================================

@dataclass(
    frozen=True,
)
class TransformationArtifactMaterializationReport:
    """
    Result of materializing a validated Transformation
    execution into the server-owned Preparation Artifact Store.

    DataFrames are intentionally not exposed by this report.
    """

    workflow_id: str

    source_dataset_id: str

    persisted_dataset_ids: tuple[
        str,
        ...
    ]

    derived_dataset_ids: tuple[
        str,
        ...
    ]

    artifact_count: int

    source_data_changed: bool

    materialization_kind: (
        TransformationArtifactMaterializationKind
    )

    bridge_version: str = (
        TRANSFORMATION_ARTIFACT_BRIDGE_VERSION
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
# SOURCE TRUST ANCHOR
# ============================================================

def _require_current_source_artifact(
    *,
    workflow_id: str,
    dataset_id: str,
    source_dataframe: pd.DataFrame,
) -> PreparationDatasetArtifact:
    """
    Ensure Transformation consumes exactly the DataFrame
    currently owned by the Preparation Artifact Store.

    TRANSFORM may start only from SOURCE or CLEAN.

    A later TRANSFORM / COMBINE artifact must never be
    silently transformed again.
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
        }
    ):

        raise ValueError(
            "Transformation cannot consume an artifact "
            "that already belongs to TRANSFORM or COMBINE. "
            f"dataset_id={dataset_id}, "
            f"stage={artifact.stage}"
        )


    if not (
        artifact
        .dataframe
        .equals(
            source_dataframe
        )
    ):

        raise ValueError(
            "Transformation source DataFrame does not match "
            "the current server-owned Preparation artifact. "
            f"dataset_id={dataset_id}"
        )


    return (
        artifact
    )


# ============================================================
# DERIVED OUTPUT CONTRACT
# ============================================================

def _expected_derived_output_filenames(
    approved_plan: ApprovedTransformationPlan,
) -> dict[
    str,
    str,
]:
    """
    Derive the exact authorized aggregate-output dataset
    contract from executable Transformation steps.

    Only executable steps are allowed to produce persisted
    derived datasets.
    """

    output: dict[
        str,
        str,
    ] = {}


    for step in (
        approved_plan.steps
    ):

        if not (
            step.executable
        ):

            continue


        output_dataset_id = (
            step.output_dataset_id
        )


        if (
            output_dataset_id
            is None
        ):

            continue


        dataset_id = (
            output_dataset_id.strip()
        )


        if not dataset_id:

            raise ValueError(
                "Executable Transformation step contains "
                "an empty output_dataset_id."
            )


        output_filename = (
            step.output_dataset_filename
        )


        if (
            output_filename
            is None
            or
            not output_filename.strip()
        ):

            raise ValueError(
                "Executable Transformation derived output "
                "is missing output_dataset_filename. "
                f"dataset_id={dataset_id}"
            )


        if (
            dataset_id
            in output
        ):

            raise ValueError(
                "Transformation plan contains duplicate "
                "derived output_dataset_id="
                f"{dataset_id}"
            )


        output[
            dataset_id
        ] = (
            output_filename.strip()
        )


    return output


# ============================================================
# SOURCE EVIDENCE
# ============================================================

def _source_evidence_refs(
    *,
    current_artifact: PreparationDatasetArtifact,
    approved_plan: ApprovedTransformationPlan,
    execution: TransformationExecutionResult,
    validation: PostTransformationValidationReport,
) -> list[
    str
]:

    refs = list(
        current_artifact
        .evidence_refs
    )


    additions = [
        (
            "transformation_approval:"
            f"{approved_plan.rule_version}"
        ),

        (
            "transformation_execution:"
            f"{execution.report.rule_version}"
        ),

        (
            "transformation_validation:"
            f"{validation.rule_version}"
        ),

        (
            "transformation_source_fingerprint_before:"
            f"{execution.report.source_fingerprint_before}"
        ),

        (
            "transformation_source_fingerprint_after:"
            f"{execution.report.source_fingerprint_after}"
        ),
    ]


    for step in (
        execution.report.steps
    ):

        status = (
            step.status.value
            if hasattr(
                step.status,
                "value",
            )
            else
            str(
                step.status
            )
        )


        additions.append(
            (
                "transformation_step:"
                f"{step.step_id}:"
                f"{status}"
            )
        )


    seen = set(
        refs
    )


    for ref in additions:

        if (
            ref
            in seen
        ):

            continue


        seen.add(
            ref
        )

        refs.append(
            ref
        )


    return refs


# ============================================================
# DERIVED EVIDENCE
# ============================================================

def _derived_evidence_refs(
    *,
    source_dataset_id: str,
    output_dataset_id: str,
    approved_plan: ApprovedTransformationPlan,
    execution: TransformationExecutionResult,
    validation: PostTransformationValidationReport,
) -> list[
    str
]:

    return [
        (
            "transformation_parent:"
            f"{source_dataset_id}"
        ),

        (
            "transformation_output:"
            f"{output_dataset_id}"
        ),

        (
            "transformation_approval:"
            f"{approved_plan.rule_version}"
        ),

        (
            "transformation_execution:"
            f"{execution.report.rule_version}"
        ),

        (
            "transformation_validation:"
            f"{validation.rule_version}"
        ),
    ]


# ============================================================
# MATERIALIZATION
# ============================================================

def materialize_transformation_artifacts(
    *,
    workflow_id: str,
    source_dataframe: pd.DataFrame,
    approved_plan: ApprovedTransformationPlan,
    execution: TransformationExecutionResult,
    validation: PostTransformationValidationReport,
) -> TransformationArtifactMaterializationReport:
    """
    Persist only independently validated Transformation output.

    Safety guarantees
    -----------------

    - PostTransformationValidation must explicitly allow
      downstream use;

    - the source dataset identity must match between
      approval, execution and validation;

    - source_dataframe must exactly match the current
      server-owned SOURCE/CLEAN artifact;

    - executor source-data-changed metadata must agree with
      the actual DataFrames;

    - derived output dataset IDs must exactly match the
      executable approved output contract;

    - unknown / invented derived outputs are rejected;

    - all validation occurs before the first store write.
    """

    normalized_workflow_id = (
        _required_text(
            workflow_id,
            field_name="workflow_id",
        )
    )


    if not isinstance(
        source_dataframe,
        pd.DataFrame,
    ):

        raise TypeError(
            "Transformation source_dataframe must be a "
            "pandas DataFrame."
        )


    if (
        source_dataframe.empty
    ):

        raise ValueError(
            "Transformation source_dataframe cannot be empty."
        )


    # ========================================================
    # POST-VALIDATION IS THE DOWNSTREAM AUTHORITY
    # ========================================================

    if not (
        validation.valid_for_downstream
    ):

        raise ValueError(
            "Transformation output failed independent "
            "post-transformation validation and cannot be "
            "materialized."
        )


    if (
        validation.status
        !=
        "passed"
    ):

        raise ValueError(
            "Transformation validation status must be "
            "'passed' before artifact materialization."
        )


    dataset_id = (
        _required_text(
            approved_plan.dataset_id,
            field_name="approved_plan.dataset_id",
        )
    )


    dataset_filename = (
        _required_text(
            approved_plan.dataset_filename,
            field_name="approved_plan.dataset_filename",
        )
    )


    # ========================================================
    # IDENTITY RECONCILIATION
    # ========================================================

    if (
        execution.report.dataset_id
        !=
        dataset_id
    ):

        raise ValueError(
            "Transformation execution dataset_id does not "
            "match approved plan."
        )


    if (
        execution.report.dataset_filename
        !=
        dataset_filename
    ):

        raise ValueError(
            "Transformation execution dataset_filename does "
            "not match approved plan."
        )


    if (
        validation.dataset_id
        !=
        dataset_id
    ):

        raise ValueError(
            "Transformation validation dataset_id does not "
            "match approved plan."
        )


    if (
        validation.dataset_filename
        !=
        dataset_filename
    ):

        raise ValueError(
            "Transformation validation dataset_filename does "
            "not match approved plan."
        )


    # ========================================================
    # SERVER-OWNED SOURCE TRUST ANCHOR
    # ========================================================

    current_artifact = (
        _require_current_source_artifact(
            workflow_id=(
                normalized_workflow_id
            ),

            dataset_id=(
                dataset_id
            ),

            source_dataframe=(
                source_dataframe
            ),
        )
    )


    transformed_dataframe = (
        execution.dataframe
    )


    if not isinstance(
        transformed_dataframe,
        pd.DataFrame,
    ):

        raise TypeError(
            "Transformation Executor returned an invalid "
            "main DataFrame."
        )


    if (
        transformed_dataframe.empty
    ):

        raise ValueError(
            "Transformation Executor returned an empty main "
            "DataFrame."
        )


    # ========================================================
    # ACTUAL SOURCE MUTATION RECONCILIATION
    # ========================================================

    actual_source_changed = (
        not (
            source_dataframe
            .equals(
                transformed_dataframe
            )
        )
    )


    reported_source_changed = bool(
        execution
        .report
        .source_data_changed
    )


    if (
        actual_source_changed
        !=
        reported_source_changed
    ):

        raise ValueError(
            "Transformation execution source_data_changed "
            "does not match the actual DataFrame result."
        )


    fingerprint_before = (
        execution
        .report
        .source_fingerprint_before
    )


    fingerprint_after = (
        execution
        .report
        .source_fingerprint_after
    )


    if (
        reported_source_changed
        and
        fingerprint_before
        ==
        fingerprint_after
    ):

        raise ValueError(
            "Transformation reports changed source data but "
            "before/after fingerprints are identical."
        )


    if (
        not reported_source_changed
        and
        fingerprint_before
        !=
        fingerprint_after
    ):

        raise ValueError(
            "Transformation reports unchanged source data "
            "but before/after fingerprints differ."
        )


    # ========================================================
    # DERIVED OUTPUT RECONCILIATION
    # ========================================================

    expected_derived = (
        _expected_derived_output_filenames(
            approved_plan
        )
    )


    actual_derived = (
        execution
        .derived_datasets
    )


    if not isinstance(
        actual_derived,
        dict,
    ):

        raise TypeError(
            "TransformationExecutionResult.derived_datasets "
            "must be a mapping."
        )


    expected_ids = set(
        expected_derived.keys()
    )


    actual_ids = set(
        actual_derived.keys()
    )


    if (
        actual_ids
        !=
        expected_ids
    ):

        missing = sorted(
            expected_ids
            -
            actual_ids
        )


        unexpected = sorted(
            actual_ids
            -
            expected_ids
        )


        raise ValueError(
            "Transformation derived output scope does not "
            "match executable approved outputs. "
            f"missing={missing}, "
            f"unexpected={unexpected}"
        )


    if (
        int(
            execution
            .report
            .derived_dataset_count
        )
        != len(
            actual_ids
        )
    ):

        raise ValueError(
            "Transformation execution derived_dataset_count "
            "does not match actual derived outputs."
        )


    reported_derived_ids = set(
        execution
        .report
        .derived_dataset_ids
    )


    if (
        reported_derived_ids
        !=
        actual_ids
    ):

        raise ValueError(
            "Transformation execution derived_dataset_ids "
            "do not match actual derived outputs."
        )


    for (
        output_dataset_id,
        dataframe,
    ) in (
        actual_derived.items()
    ):

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):

            raise TypeError(
                "Transformation derived output must be a "
                "pandas DataFrame. "
                f"dataset_id={output_dataset_id}"
            )


        if (
            dataframe.empty
        ):

            raise ValueError(
                "Transformation derived output cannot be "
                "empty. "
                f"dataset_id={output_dataset_id}"
            )


        if (
            output_dataset_id
            ==
            dataset_id
        ):

            raise ValueError(
                "Transformation derived output cannot "
                "overwrite its source dataset_id."
            )


    # ========================================================
    # PREVALIDATION COMPLETE
    #
    # No Artifact Store mutation occurred before this point.
    # ========================================================

    persisted: list[
        PreparationDatasetArtifactInfo
    ] = []


    # ========================================================
    # MAIN TRANSFORMED DATASET
    # ========================================================

    if (
        reported_source_changed
    ):

        artifact = (
            put_preparation_artifact(
                workflow_id=(
                    normalized_workflow_id
                ),

                dataset_id=(
                    dataset_id
                ),

                dataset_filename=(
                    dataset_filename
                ),

                stage=(
                    "transform"
                ),

                dataframe=(
                    transformed_dataframe
                ),

                parent_dataset_ids=(
                    current_artifact
                    .parent_dataset_ids
                    or
                    (
                        dataset_id,
                    )
                ),

                evidence_refs=(
                    _source_evidence_refs(
                        current_artifact=(
                            current_artifact
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

                replace=True,
            )
        )


        persisted.append(
            artifact
        )


    # ========================================================
    # DERIVED DATASETS
    # ========================================================

    for output_dataset_id in (
        expected_derived.keys()
    ):

        dataframe = (
            actual_derived[
                output_dataset_id
            ]
        )


        output_filename = (
            expected_derived[
                output_dataset_id
            ]
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
                    "transform"
                ),

                dataframe=(
                    dataframe
                ),

                parent_dataset_ids=[
                    dataset_id,
                ],

                evidence_refs=(
                    _derived_evidence_refs(
                        source_dataset_id=(
                            dataset_id
                        ),

                        output_dataset_id=(
                            output_dataset_id
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

                replace=True,
            )
        )


        persisted.append(
            artifact
        )


    persisted_ids = tuple(
        artifact.dataset_id

        for artifact
        in persisted
    )


    derived_ids = tuple(
        expected_derived.keys()
    )


    # ========================================================
    # MATERIALIZATION KIND
    # ========================================================

    if (
        reported_source_changed
        and
        derived_ids
    ):

        kind: TransformationArtifactMaterializationKind = (
            "source_and_derived"
        )


    elif (
        reported_source_changed
    ):

        kind = (
            "source_transformed"
        )


    elif (
        derived_ids
    ):

        kind = (
            "derived_only"
        )


    else:

        kind = (
            "no_change"
        )


    return (
        TransformationArtifactMaterializationReport(
            workflow_id=(
                normalized_workflow_id
            ),

            source_dataset_id=(
                dataset_id
            ),

            persisted_dataset_ids=(
                persisted_ids
            ),

            derived_dataset_ids=(
                derived_ids
            ),

            artifact_count=(
                len(
                    persisted
                )
            ),

            source_data_changed=(
                reported_source_changed
            ),

            materialization_kind=(
                kind
            ),
        )
    )