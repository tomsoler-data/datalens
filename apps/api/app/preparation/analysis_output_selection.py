from __future__ import annotations

from typing import (
    Dict,
    List,
    Set,
    Tuple,
)

from pydantic import (
    BaseModel,
    Field,
)

from app.preparation.preparation_artifact_store import (
    PreparationDatasetArtifactInfo,
    list_preparation_artifacts,
)

from app.preparation.preparation_session import (
    PreparationSessionView,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
    PreparationStageStatus,
)


# ============================================================
# VERSION
# ============================================================


ANALYSIS_OUTPUT_SELECTION_RULE_VERSION = (
    "analysis_output_selection_v0.1"
)


# ============================================================
# ALLOWED MATERIALIZED STAGES
# ============================================================


ALLOWED_ANALYSIS_OUTPUT_ARTIFACT_STAGES = {
    "source",
    "clean",
    "transform",
    "combine",
}


# ============================================================
# CANDIDATE
# ============================================================


class AnalysisOutputCandidate(
    BaseModel,
):
    dataset_id: str

    dataset_filename: str

    artifact_stage: str

    rows: int

    columns: int

    parent_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    lineage_root_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    selectable: bool

    blocking_reasons: List[
        str
    ] = Field(
        default_factory=list
    )


# ============================================================
# REPORT
# ============================================================


class AnalysisOutputSelectionReport(
    BaseModel,
):
    workflow_id: str

    session_revision: int

    valid: bool

    preparation_root_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    requested_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    available_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    selected_analysis_output_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    invalid_requested_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    candidate_count: int

    selectable_candidate_count: int

    candidates: List[
        AnalysisOutputCandidate
    ] = Field(
        default_factory=list
    )

    blocking_reasons: List[
        str
    ] = Field(
        default_factory=list
    )

    notes: List[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        ANALYSIS_OUTPUT_SELECTION_RULE_VERSION
    )


# ============================================================
# ERROR
# ============================================================


class AnalysisOutputSelectionBlockedError(
    RuntimeError,
):
    def __init__(
        self,
        *,
        report: AnalysisOutputSelectionReport,
    ) -> None:
        self.report = (
            report
        )

        super().__init__(
            (
                "Analysis output selection is blocked. "
                f"workflow_id={report.workflow_id}, "
                "invalid_dataset_ids="
                f"{report.invalid_requested_dataset_ids}"
            )
        )


# ============================================================
# NORMALIZATION
# ============================================================


def _normalize_dataset_ids(
    values: List[
        str
    ],
) -> List[
    str
]:
    output: List[
        str
    ] = []

    seen: Set[
        str
    ] = set()


    for raw_value in values:
        value = (
            str(
                raw_value
            )
            .strip()
        )


        if not value:
            raise ValueError(
                (
                    "Analysis output dataset_id "
                    "cannot be empty."
                )
            )


        if (
            value
            in seen
        ):
            continue


        seen.add(
            value
        )

        output.append(
            value
        )


    return (
        output
    )


# ============================================================
# STAGE LOOKUP
# ============================================================


def _stage_record(
    *,
    session: PreparationSessionView,
    stage: PreparationStage,
):
    for record in (
        session.snapshot.stages
    ):
        if (
            record.stage
            ==
            stage
        ):
            return (
                record
            )


    return (
        None
    )


# ============================================================
# ARTIFACT INDEX
# ============================================================


def _artifact_index(
    artifacts: List[
        PreparationDatasetArtifactInfo
    ],
) -> Dict[
    str,
    PreparationDatasetArtifactInfo,
]:
    output: Dict[
        str,
        PreparationDatasetArtifactInfo,
    ] = {}


    for artifact in artifacts:
        dataset_id = (
            artifact.dataset_id
            .strip()
        )


        if not dataset_id:
            raise ValueError(
                (
                    "Preparation Artifact Store contains "
                    "an empty dataset_id."
                )
            )


        if (
            dataset_id
            in output
        ):
            raise ValueError(
                (
                    "Preparation Artifact Store returned "
                    "duplicate dataset_id="
                    f"{dataset_id}"
                )
            )


        output[
            dataset_id
        ] = (
            artifact
        )


    return (
        output
    )


# ============================================================
# LINEAGE
# ============================================================


LineageResolution = Tuple[
    bool,
    Set[
        str
    ],
    List[
        str
    ],
]


def _resolve_lineage_roots(
    *,
    dataset_id: str,
    artifact_index: Dict[
        str,
        PreparationDatasetArtifactInfo,
    ],
    preparation_root_dataset_ids: Set[
        str
    ],
    visiting: Set[
        str
    ],
    memo: Dict[
        str,
        LineageResolution,
    ],
) -> LineageResolution:
    """
    Resolve a materialized artifact back to the Preparation
    root dataset scope.

    Important v0.1 behaviour:

    - current source datasets are authorized roots;
    - a CLEAN/TRANSFORM artifact may preserve the same
      dataset_id and therefore self-reference in lineage;
    - derived TRANSFORM outputs must reach a root;
    - COMBINE outputs must reach their source roots;
    - missing parents are rejected;
    - lineage cycles between derived datasets are rejected.
    """

    if (
        dataset_id
        in memo
    ):
        return (
            memo[
                dataset_id
            ]
        )


    artifact = (
        artifact_index.get(
            dataset_id
        )
    )


    if (
        artifact is None
    ):
        result = (
            False,
            set(),
            [
                (
                    "Lineage references a dataset that is "
                    "not present in the Preparation Artifact "
                    "Store: "
                    f"{dataset_id}"
                ),
            ],
        )

        memo[
            dataset_id
        ] = (
            result
        )

        return (
            result
        )


    artifact_stage = (
        str(
            artifact.stage
        )
    )


    if (
        artifact_stage
        not in
        ALLOWED_ANALYSIS_OUTPUT_ARTIFACT_STAGES
    ):
        result = (
            False,
            set(),
            [
                (
                    "Artifact stage is not authorized for "
                    "analysis output selection. "
                    f"dataset_id={dataset_id}, "
                    f"stage={artifact_stage}"
                ),
            ],
        )

        memo[
            dataset_id
        ] = (
            result
        )

        return (
            result
        )


    # ========================================================
    # PREPARATION ROOT
    #
    # A root may currently be represented by SOURCE, CLEAN or
    # TRANSFORM because DataLens keeps the latest materialized
    # state under the same dataset_id.
    #
    # This also intentionally neutralizes self-lineage such as
    # parent_dataset_ids=("sales",) on the current "sales"
    # artifact.
    # ========================================================

    if (
        dataset_id
        in
        preparation_root_dataset_ids
    ):
        result = (
            True,
            {
                dataset_id,
            },
            [],
        )

        memo[
            dataset_id
        ] = (
            result
        )

        return (
            result
        )


    # ========================================================
    # CYCLE
    # ========================================================

    if (
        dataset_id
        in visiting
    ):
        return (
            False,
            set(),
            [
                (
                    "Artifact lineage contains a cycle at "
                    f"dataset_id={dataset_id}"
                ),
            ],
        )


    next_visiting = set(
        visiting
    )

    next_visiting.add(
        dataset_id
    )


    # Ignore a self-reference.
    #
    # Self-lineage can legitimately exist for a materialized
    # replacement of a root dataset, but a non-root dataset
    # cannot use a self-reference as provenance.
    parent_dataset_ids = [
        parent_id.strip()

        for parent_id
        in artifact.parent_dataset_ids

        if (
            parent_id.strip()
            and
            parent_id.strip()
            !=
            dataset_id
        )
    ]


    if not parent_dataset_ids:
        result = (
            False,
            set(),
            [
                (
                    "Derived analysis candidate has no "
                    "lineage reaching a Preparation root. "
                    f"dataset_id={dataset_id}"
                ),
            ],
        )

        memo[
            dataset_id
        ] = (
            result
        )

        return (
            result
        )


    resolved_roots: Set[
        str
    ] = set()

    reasons: List[
        str
    ] = []


    for parent_dataset_id in (
        parent_dataset_ids
    ):
        (
            parent_valid,
            parent_roots,
            parent_reasons,
        ) = (
            _resolve_lineage_roots(
                dataset_id=(
                    parent_dataset_id
                ),

                artifact_index=(
                    artifact_index
                ),

                preparation_root_dataset_ids=(
                    preparation_root_dataset_ids
                ),

                visiting=(
                    next_visiting
                ),

                memo=(
                    memo
                ),
            )
        )


        if not parent_valid:
            reasons.extend(
                parent_reasons
            )

            continue


        resolved_roots.update(
            parent_roots
        )


    if reasons:
        result = (
            False,
            resolved_roots,
            reasons,
        )

        memo[
            dataset_id
        ] = (
            result
        )

        return (
            result
        )


    if not resolved_roots:
        result = (
            False,
            set(),
            [
                (
                    "Artifact lineage does not resolve to "
                    "any authorized Preparation root. "
                    f"dataset_id={dataset_id}"
                ),
            ],
        )

        memo[
            dataset_id
        ] = (
            result
        )

        return (
            result
        )


    unauthorized_roots = (
        resolved_roots
        -
        preparation_root_dataset_ids
    )


    if unauthorized_roots:
        result = (
            False,
            resolved_roots,
            [
                (
                    "Artifact lineage reaches unauthorized "
                    "Preparation roots. "
                    "dataset_id="
                    f"{dataset_id}, "
                    "unauthorized_roots="
                    f"{sorted(unauthorized_roots)}"
                ),
            ],
        )

        memo[
            dataset_id
        ] = (
            result
        )

        return (
            result
        )


    result = (
        True,
        resolved_roots,
        [],
    )

    memo[
        dataset_id
    ] = (
        result
    )

    return (
        result
    )


# ============================================================
# EVALUATE
# ============================================================


def evaluate_analysis_output_selection(
    *,
    session: PreparationSessionView,
    requested_dataset_ids: List[
        str
    ],
) -> AnalysisOutputSelectionReport:
    """
    Evaluate which current Preparation artifacts may become
    final analytical outputs.

    This function NEVER:

    - mutates PreparationSession;
    - mutates PreparationArtifactStore;
    - marks VALIDATE as passed;
    - authorizes ANALYZE.

    It is a deterministic eligibility gate only.
    """

    requested_ids = (
        _normalize_dataset_ids(
            requested_dataset_ids
        )
    )


    preparation_root_ids = (
        _normalize_dataset_ids(
            list(
                session
                .selected_analysis_dataset_ids
            )
        )
    )


    preparation_root_set = set(
        preparation_root_ids
    )


    blocking_reasons: List[
        str
    ] = []


    # ========================================================
    # COMBINE MUST BE RESOLVED
    # ========================================================

    combine_record = (
        _stage_record(
            session=(
                session
            ),

            stage=(
                PreparationStage.COMBINE
            ),
        )
    )


    combine_resolved = (
        combine_record
        is not None
        and
        combine_record.status
        in {
            PreparationStageStatus.PASSED,
            PreparationStageStatus.SKIPPED,
        }
    )


    if not combine_resolved:
        combine_status = (
            combine_record.status.value

            if (
                combine_record
                is not None
            )

            else
            "missing"
        )


        blocking_reasons.append(
            (
                "COMBINE must be PASSED or SKIPPED before "
                "final analysis outputs can be selected. "
                f"current_status={combine_status}"
            )
        )


    # ========================================================
    # ARTIFACTS
    # ========================================================

    artifacts = (
        list_preparation_artifacts(
            workflow_id=(
                session.workflow_id
            )
        )
    )


    index = (
        _artifact_index(
            artifacts
        )
    )


    memo: Dict[
        str,
        LineageResolution,
    ] = {}


    candidates: List[
        AnalysisOutputCandidate
    ] = []


    available_dataset_ids: List[
        str
    ] = []


    for dataset_id in sorted(
        index.keys()
    ):
        artifact = (
            index[
                dataset_id
            ]
        )


        (
            lineage_valid,
            lineage_roots,
            lineage_reasons,
        ) = (
            _resolve_lineage_roots(
                dataset_id=(
                    dataset_id
                ),

                artifact_index=(
                    index
                ),

                preparation_root_dataset_ids=(
                    preparation_root_set
                ),

                visiting=set(),

                memo=(
                    memo
                ),
            )
        )


        candidate_reasons = list(
            lineage_reasons
        )


        artifact_stage = (
            str(
                artifact.stage
            )
        )


        stage_valid = (
            artifact_stage
            in
            ALLOWED_ANALYSIS_OUTPUT_ARTIFACT_STAGES
        )


        if not stage_valid:
            candidate_reasons.append(
                (
                    "Artifact stage is not allowed for "
                    "analysis output selection: "
                    f"{artifact_stage}"
                )
            )


        selectable = (
            combine_resolved
            and
            lineage_valid
            and
            stage_valid
        )


        if selectable:
            available_dataset_ids.append(
                dataset_id
            )


        candidates.append(
            AnalysisOutputCandidate(
                dataset_id=(
                    dataset_id
                ),

                dataset_filename=(
                    artifact
                    .dataset_filename
                ),

                artifact_stage=(
                    artifact_stage
                ),

                rows=(
                    artifact.rows
                ),

                columns=(
                    artifact.columns
                ),

                parent_dataset_ids=list(
                    artifact
                    .parent_dataset_ids
                ),

                lineage_root_dataset_ids=sorted(
                    lineage_roots
                ),

                selectable=(
                    selectable
                ),

                blocking_reasons=(
                    candidate_reasons
                ),
            )
        )


    # ========================================================
    # REQUESTED OUTPUTS
    # ========================================================

    available_set = set(
        available_dataset_ids
    )


    invalid_requested_ids = [
        dataset_id

        for dataset_id
        in requested_ids

        if (
            dataset_id
            not in
            available_set
        )
    ]


    if not requested_ids:
        blocking_reasons.append(
            (
                "At least one final analysis output dataset "
                "must be explicitly selected."
            )
        )


    if invalid_requested_ids:
        blocking_reasons.append(
            (
                "One or more requested analysis outputs "
                "are not authorized by current server-owned "
                "Preparation artifacts and lineage. "
                "dataset_ids="
                f"{invalid_requested_ids}"
            )
        )


    valid = (
        combine_resolved
        and
        bool(
            requested_ids
        )
        and
        not (
            invalid_requested_ids
        )
    )


    selected_ids = (
        list(
            requested_ids
        )

        if valid

        else
        []
    )


    return (
        AnalysisOutputSelectionReport(
            workflow_id=(
                session.workflow_id
            ),

            session_revision=(
                session.revision
            ),

            valid=(
                valid
            ),

            preparation_root_dataset_ids=(
                preparation_root_ids
            ),

            requested_dataset_ids=(
                requested_ids
            ),

            available_dataset_ids=(
                available_dataset_ids
            ),

            selected_analysis_output_dataset_ids=(
                selected_ids
            ),

            invalid_requested_dataset_ids=(
                invalid_requested_ids
            ),

            candidate_count=(
                len(
                    candidates
                )
            ),

            selectable_candidate_count=(
                len(
                    available_dataset_ids
                )
            ),

            candidates=(
                candidates
            ),

            blocking_reasons=(
                blocking_reasons
            ),

            notes=[
                (
                    "Preparation root dataset IDs remain "
                    "separate from final analytical output "
                    "selection."
                ),

                (
                    "Derived outputs are selectable only "
                    "when their Artifact Store lineage "
                    "resolves to authorized Preparation "
                    "roots."
                ),

                (
                    "TRANSFORM and COMBINE artifacts are "
                    "trusted only because those bridges "
                    "materialize them after their respective "
                    "post-validation gates."
                ),

                (
                    "This v0.1 selector does not mutate the "
                    "Preparation session."
                ),
            ],

            rule_version=(
                ANALYSIS_OUTPUT_SELECTION_RULE_VERSION
            ),
        )
    )


# ============================================================
# REQUIRE
# ============================================================


def require_analysis_output_selection(
    *,
    session: PreparationSessionView,
    requested_dataset_ids: List[
        str
    ],
) -> AnalysisOutputSelectionReport:
    report = (
        evaluate_analysis_output_selection(
            session=(
                session
            ),

            requested_dataset_ids=(
                requested_dataset_ids
            ),
        )
    )


    if not (
        report.valid
    ):
        raise (
            AnalysisOutputSelectionBlockedError(
                report=(
                    report
                )
            )
        )


    return (
        report
    )