from __future__ import annotations

from dataclasses import (
    dataclass,
)

from typing import (
    Literal,
)

import pandas as pd

from app.preparation.preparation_artifact_store import (
    PreparationDatasetArtifact,
    PreparationDatasetArtifactInfo,
    get_preparation_artifact,
    put_preparation_artifact,
)

from app.preparation.semantic_cleaning import (
    SemanticCleaningExecutionResult,
    SemanticCleaningPlan,
)


# ============================================================
# VERSION
# ============================================================

SEMANTIC_CLEANING_ARTIFACT_BRIDGE_VERSION = (
    "semantic_cleaning_artifact_bridge_v0.1"
)


# ============================================================
# TYPES
# ============================================================

SemanticArtifactMaterializationKind = Literal[
    "no_change",
    "semantic_cleaned",
]


# ============================================================
# RESULT
# ============================================================

@dataclass(
    frozen=True,
)
class SemanticCleaningArtifactMaterializationReport:
    """
    Result of reconciling a Semantic Cleaning execution with
    the server-owned Preparation Artifact Store.

    No DataFrame is exposed by this report.
    """

    workflow_id: str

    dataset_ids: tuple[
        str,
        ...
    ]

    persisted_dataset_ids: tuple[
        str,
        ...
    ]

    artifact_count: int

    changed_cell_count: int

    materialization_kind: (
        SemanticArtifactMaterializationKind
    )

    bridge_version: str = (
        SEMANTIC_CLEANING_ARTIFACT_BRIDGE_VERSION
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


def _validate_frame_map(
    *,
    frames: dict[
        str,
        pd.DataFrame,
    ],
    label: str,
) -> dict[
    str,
    pd.DataFrame,
]:

    if not isinstance(
        frames,
        dict,
    ):

        raise TypeError(
            f"{label} must be a dataset → DataFrame mapping."
        )


    if not frames:

        raise ValueError(
            f"{label} must contain at least one dataset."
        )


    output: dict[
        str,
        pd.DataFrame,
    ] = {}


    for (
        raw_dataset_id,
        dataframe,
    ) in frames.items():

        dataset_id = (
            str(
                raw_dataset_id
            )
            .strip()
        )


        if not dataset_id:

            raise ValueError(
                f"{label} contains an empty dataset_id."
            )


        if dataset_id in output:

            raise ValueError(
                f"{label} contains duplicate dataset_id="
                f"{dataset_id}"
            )


        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):

            raise TypeError(
                f"{label} must contain pandas DataFrames. "
                f"dataset_id={dataset_id}"
            )


        if dataframe.empty:

            raise ValueError(
                f"{label} contains an empty DataFrame. "
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
# PROVENANCE
# ============================================================

def _provenance_index(
    execution: SemanticCleaningExecutionResult,
) -> dict[
    str,
    object,
]:

    output: dict[
        str,
        object,
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
                "Semantic Cleaning provenance contains an "
                "empty dataset_id."
            )


        if dataset_id in output:

            raise ValueError(
                "Semantic Cleaning execution contains "
                "duplicate provenance for dataset_id="
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
# SERVER TRUST ANCHOR
# ============================================================

def _require_current_artifact_matches_input(
    *,
    workflow_id: str,
    dataset_id: str,
    deterministic_frame: pd.DataFrame,
) -> PreparationDatasetArtifact:
    """
    Protect the Artifact Store from browser-side data drift.

    Semantic APIs currently rebuild deterministic cleaning
    from uploaded files.

    Before those rebuilt frames may influence the
    server-owned materialized state, they must exactly match
    the artifact already held by the server for this workflow.
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
            "Semantic Cleaning cannot replace an artifact "
            "that already belongs to a later Preparation "
            "stage. "
            f"dataset_id={dataset_id}, "
            f"stage={artifact.stage}"
        )


    if not (
        artifact
        .dataframe
        .equals(
            deterministic_frame
        )
    ):

        raise ValueError(
            "Semantic Cleaning deterministic input does not "
            "match the current server-owned Preparation "
            "artifact. "
            f"dataset_id={dataset_id}"
        )


    return (
        artifact
    )


# ============================================================
# EVIDENCE
# ============================================================

def _merge_evidence_refs(
    *,
    current_artifact: PreparationDatasetArtifact,
    semantic_plan: SemanticCleaningPlan,
    execution: SemanticCleaningExecutionResult,
    provenance,
) -> list[
    str
]:

    refs: list[
        str
    ] = list(
        current_artifact
        .evidence_refs
    )


    additions = [
        (
            "semantic_cleaning_plan:"
            f"{semantic_plan.rule_version}"
        ),

        (
            "semantic_cleaning_execution:"
            f"{execution.rule_version}"
        ),

        (
            "semantic_source_fingerprint:"
            f"{provenance.source_fingerprint}"
        ),

        (
            "semantic_derived_fingerprint:"
            f"{provenance.derived_fingerprint}"
        ),

        (
            "semantic_changed_cells:"
            f"{provenance.changed_cell_count}"
        ),
    ]


    additions.extend(
        [
            (
                "semantic_cleaning_action:"
                f"{action_id}"
            )

            for action_id
            in provenance.applied_action_ids
        ]
    )


    seen = set(
        refs
    )


    for ref in additions:

        if ref in seen:
            continue


        seen.add(
            ref
        )

        refs.append(
            ref
        )


    return refs


# ============================================================
# MATERIALIZATION
# ============================================================

def materialize_semantic_cleaning_artifacts(
    *,
    workflow_id: str,
    deterministic_frames: dict[
        str,
        pd.DataFrame,
    ],
    derived_frames: dict[
        str,
        pd.DataFrame,
    ],
    semantic_plan: SemanticCleaningPlan,
    execution: SemanticCleaningExecutionResult,
) -> SemanticCleaningArtifactMaterializationReport:
    """
    Reconcile a Semantic Cleaning execution with the current
    server-owned Preparation artifacts.

    Trust / safety rules
    --------------------

    1. deterministic_frames and derived_frames must expose
       exactly the same datasets;

    2. every dataset must have Semantic Cleaning provenance;

    3. every deterministic input must exactly match the
       current server-owned artifact;

    4. provenance row counts must match the actual frames;

    5. aggregate provenance changed-cell counts must match the
       execution report;

    6. when no cell changed, the Artifact Store is left
       untouched;

    7. only datasets with actual semantic mutations replace
       their current CLEAN artifact;

    8. all validation happens before the first store write.
    """

    normalized_workflow_id = (
        _required_text(
            workflow_id,
            field_name="workflow_id",
        )
    )


    deterministic = (
        _validate_frame_map(
            frames=(
                deterministic_frames
            ),

            label=(
                "deterministic_frames"
            ),
        )
    )


    derived = (
        _validate_frame_map(
            frames=(
                derived_frames
            ),

            label=(
                "derived_frames"
            ),
        )
    )


    provenance_by_dataset = (
        _provenance_index(
            execution
        )
    )


    deterministic_ids = set(
        deterministic.keys()
    )


    derived_ids = set(
        derived.keys()
    )


    provenance_ids = set(
        provenance_by_dataset.keys()
    )


    # ========================================================
    # EXACT SCOPE RECONCILIATION
    # ========================================================

    if (
        derived_ids
        != deterministic_ids
    ):

        missing = sorted(
            deterministic_ids
            -
            derived_ids
        )


        unexpected = sorted(
            derived_ids
            -
            deterministic_ids
        )


        raise ValueError(
            "Semantic Cleaning derived dataset scope does "
            "not match deterministic input scope. "
            f"missing={missing}, "
            f"unexpected={unexpected}"
        )


    if (
        provenance_ids
        != deterministic_ids
    ):

        missing = sorted(
            deterministic_ids
            -
            provenance_ids
        )


        unexpected = sorted(
            provenance_ids
            -
            deterministic_ids
        )


        raise ValueError(
            "Semantic Cleaning provenance scope does not "
            "match deterministic input scope. "
            f"missing={missing}, "
            f"unexpected={unexpected}"
        )


    if (
        int(
            execution.dataset_count
        )
        != len(
            deterministic_ids
        )
    ):

        raise ValueError(
            "Semantic Cleaning execution dataset_count does "
            "not match the materialized dataset scope."
        )


    # ========================================================
    # PREVALIDATE EVERYTHING BEFORE WRITING
    # ========================================================

    current_artifacts: dict[
        str,
        PreparationDatasetArtifact,
    ] = {}


    provenance_changed_total = (
        0
    )


    provenance_applied_action_count = (
        0
    )


    for dataset_id in (
        deterministic.keys()
    ):

        deterministic_frame = (
            deterministic[
                dataset_id
            ]
        )


        derived_frame = (
            derived[
                dataset_id
            ]
        )


        provenance = (
            provenance_by_dataset[
                dataset_id
            ]
        )


        # ----------------------------------------------------
        # Trust anchor:
        # browser-rebuilt input must match server state.
        # ----------------------------------------------------

        current_artifact = (
            _require_current_artifact_matches_input(
                workflow_id=(
                    normalized_workflow_id
                ),

                dataset_id=(
                    dataset_id
                ),

                deterministic_frame=(
                    deterministic_frame
                ),
            )
        )


        current_artifacts[
            dataset_id
        ] = (
            current_artifact
        )


        # ----------------------------------------------------
        # Row invariants
        # ----------------------------------------------------

        if (
            int(
                deterministic_frame.shape[
                    0
                ]
            )
            != int(
                provenance.rows_before
            )
        ):

            raise ValueError(
                "Semantic Cleaning source row count does not "
                "match provenance. "
                f"dataset_id={dataset_id}"
            )


        if (
            int(
                derived_frame.shape[
                    0
                ]
            )
            != int(
                provenance.rows_after
            )
        ):

            raise ValueError(
                "Semantic Cleaning derived row count does not "
                "match provenance. "
                f"dataset_id={dataset_id}"
            )


        # Semantic value merging must not change row count.
        if (
            int(
                provenance.rows_before
            )
            != int(
                provenance.rows_after
            )
        ):

            raise ValueError(
                "Semantic Cleaning unexpectedly changed "
                "dataset row count. "
                f"dataset_id={dataset_id}"
            )


        changed_cell_count = int(
            provenance.changed_cell_count
        )


        if (
            changed_cell_count
            <
            0
        ):

            raise ValueError(
                "Semantic Cleaning provenance contains a "
                "negative changed_cell_count. "
                f"dataset_id={dataset_id}"
            )


        provenance_changed_total += (
            changed_cell_count
        )


        provenance_applied_action_count += (
            len(
                provenance.applied_action_ids
            )
        )


        # ----------------------------------------------------
        # Fingerprint consistency
        # ----------------------------------------------------

        if (
            changed_cell_count
            ==
            0
            and
            provenance.source_fingerprint
            !=
            provenance.derived_fingerprint
        ):

            raise ValueError(
                "Semantic Cleaning reports zero changed cells "
                "but source and derived fingerprints differ. "
                f"dataset_id={dataset_id}"
            )


        if (
            changed_cell_count
            >
            0
            and
            provenance.source_fingerprint
            ==
            provenance.derived_fingerprint
        ):

            raise ValueError(
                "Semantic Cleaning reports changed cells but "
                "source and derived fingerprints are equal. "
                f"dataset_id={dataset_id}"
            )


        # ----------------------------------------------------
        # Actual frame consistency
        # ----------------------------------------------------

        actual_changed = (
            not (
                deterministic_frame
                .equals(
                    derived_frame
                )
            )
        )


        if (
            changed_cell_count
            ==
            0
            and
            actual_changed
        ):

            raise ValueError(
                "Semantic Cleaning reports no mutation but "
                "derived DataFrame differs from deterministic "
                "input. "
                f"dataset_id={dataset_id}"
            )


        if (
            changed_cell_count
            >
            0
            and
            not actual_changed
        ):

            raise ValueError(
                "Semantic Cleaning reports a mutation but "
                "derived DataFrame is unchanged. "
                f"dataset_id={dataset_id}"
            )


    # ========================================================
    # GLOBAL EXECUTION RECONCILIATION
    # ========================================================

    if (
        provenance_changed_total
        != int(
            execution.changed_cell_count
        )
    ):

        raise ValueError(
            "Semantic Cleaning execution changed_cell_count "
            "does not match per-dataset provenance."
        )


    if (
        provenance_applied_action_count
        != int(
            execution.applied_action_count
        )
    ):

        raise ValueError(
            "Semantic Cleaning execution applied_action_count "
            "does not match per-dataset provenance."
        )


    # ========================================================
    # NO MATERIAL MUTATION
    #
    # Important:
    # confirmation without data modification does NOT create
    # a fake new artifact version.
    # ========================================================

    if (
        int(
            execution.changed_cell_count
        )
        ==
        0
    ):

        return (
            SemanticCleaningArtifactMaterializationReport(
                workflow_id=(
                    normalized_workflow_id
                ),

                dataset_ids=tuple(
                    deterministic.keys()
                ),

                persisted_dataset_ids=(),

                artifact_count=(
                    0
                ),

                changed_cell_count=(
                    0
                ),

                materialization_kind=(
                    "no_change"
                ),
            )
        )


    # ========================================================
    # ALL VALIDATION COMPLETED
    #
    # No Artifact Store write occurred before this point.
    # ========================================================

    persisted: list[
        PreparationDatasetArtifactInfo
    ] = []


    for dataset_id in (
        deterministic.keys()
    ):

        provenance = (
            provenance_by_dataset[
                dataset_id
            ]
        )


        if (
            int(
                provenance.changed_cell_count
            )
            <=
            0
        ):

            continue


        current_artifact = (
            current_artifacts[
                dataset_id
            ]
        )


        parent_dataset_ids = (
            current_artifact
            .parent_dataset_ids
        )


        if not (
            parent_dataset_ids
        ):

            parent_dataset_ids = (
                dataset_id,
            )


        evidence_refs = (
            _merge_evidence_refs(
                current_artifact=(
                    current_artifact
                ),

                semantic_plan=(
                    semantic_plan
                ),

                execution=(
                    execution
                ),

                provenance=(
                    provenance
                ),
            )
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
                    current_artifact
                    .dataset_filename
                ),

                stage=(
                    "clean"
                ),

                dataframe=(
                    derived[
                        dataset_id
                    ]
                ),

                parent_dataset_ids=(
                    parent_dataset_ids
                ),

                evidence_refs=(
                    evidence_refs
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


    return (
        SemanticCleaningArtifactMaterializationReport(
            workflow_id=(
                normalized_workflow_id
            ),

            dataset_ids=tuple(
                deterministic.keys()
            ),

            persisted_dataset_ids=(
                persisted_ids
            ),

            artifact_count=(
                len(
                    persisted
                )
            ),

            changed_cell_count=(
                int(
                    execution.changed_cell_count
                )
            ),

            materialization_kind=(
                "semantic_cleaned"
            ),
        )
    )