from __future__ import annotations


from dataclasses import (
    dataclass,
)

from pathlib import (
    Path,
)

from typing import (
    Any,
)


from app.ingestion.loader import (
    build_dataset_manifest,
)

from app.ingestion.schemas import (
    MultiDatasetIngestion,
)

from app.preparation.analysis_readiness_gate import (
    require_analysis_readiness,
)

from app.preparation.preparation_artifact_store import (
    PreparationArtifactDatasetNotFoundError,
    PreparationArtifactWorkflowNotFoundError,
    get_preparation_artifact,
)


# ============================================================
# VERSION
# ============================================================


ANALYSIS_INPUT_HANDOFF_RULE_VERSION = (
    "analysis_input_handoff_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class AnalysisInputHandoffError(
    RuntimeError,
):
    pass


class AnalysisPreparedArtifactUnavailableError(
    AnalysisInputHandoffError,
):
    """
    Raised when Preparation reports READY FOR ANALYSIS but one
    of the certified final datasets can no longer be obtained
    from the server-owned Preparation Artifact Store.

    Analysis must fail closed in that situation.

    It must never fall back to a browser-uploaded CSV.
    """

    def __init__(
        self,
        *,
        workflow_id: str,
        dataset_id: str,
    ) -> None:
        self.workflow_id = (
            workflow_id
        )

        self.dataset_id = (
            dataset_id
        )

        super().__init__(
            (
                "Validated analysis artifact is no longer "
                "available in the Preparation Artifact Store. "
                f"workflow_id={workflow_id}, "
                f"dataset_id={dataset_id}"
            )
        )


# ============================================================
# HANDOFF RESULT
# ============================================================


@dataclass(
    frozen=True,
)
class AnalysisInputHandoff:
    """
    Immutable handoff descriptor between Preparation and
    Analysis.

    The contained DataFrames are deep-copy instances returned
    by PreparationArtifactStore.

    dataset_records intentionally use the same core structure
    as DataLens uploaded-dataset records:

        dataset_id
        filename
        extension
        dataframe

    This lets the deterministic analysis engine consume
    validated Preparation outputs without knowing whether the
    dataset originally came from:

        SOURCE
        CLEAN
        TRANSFORM
        COMBINE
    """

    workflow_id: str

    session_revision: int

    dataset_ids: tuple[
        str,
        ...
    ]

    ingestion: MultiDatasetIngestion

    dataset_records: tuple[
        dict[
            str,
            Any,
        ],
        ...
    ]

    rule_version: str = (
        ANALYSIS_INPUT_HANDOFF_RULE_VERSION
    )


# ============================================================
# FILENAME / EXTENSION
# ============================================================


def _artifact_extension(
    filename: str,
) -> str:
    """
    Infer the analytical extension from the server-owned
    artifact filename.

    Preparation artifacts currently materialize tabular
    datasets compatible with the CSV ingestion contract.

    A .csv fallback keeps internally named derived artifacts
    compatible when their filename has no explicit suffix.
    """

    normalized_filename = (
        filename.strip()
    )


    if not (
        normalized_filename
    ):
        raise ValueError(
            (
                "Preparation artifact filename "
                "cannot be empty."
            )
        )


    extension = (
        Path(
            normalized_filename
        )
        .suffix
        .lower()
    )


    if not (
        extension
    ):
        return (
            ".csv"
        )


    return (
        extension
    )


# ============================================================
# PUBLIC HANDOFF
# ============================================================


def load_validated_analysis_input(
    *,
    workflow_id: str,
) -> AnalysisInputHandoff:
    """
    Load the exact server-owned datasets authorized for
    analytical execution.

    Trust boundary:

        PreparationSession
                ↓
        Analysis Readiness Gate
                ↓
        analysis_output_dataset_ids
                ↓
        Preparation Artifact Store
                ↓
        DataFrame copies
                ↓
        analytical record contract

    Important security properties:

    - no dataset ID is accepted from the browser;
    - the effective dataset scope comes exclusively from the
      server-owned Preparation session;
    - only a READY FOR ANALYSIS workflow may cross this
      boundary;
    - only final validated analysis outputs are loaded;
    - Preparation roots are not silently substituted;
    - missing artifacts fail closed;
    - browser-uploaded CSV content is not used as a fallback;
    - Artifact Store reads return DataFrame copies.

    This function does NOT:

    - clean data;
    - transform data;
    - join data;
    - perform semantic cleaning;
    - execute statistical analysis.

    The validated Preparation output is consumed as-is.
    """

    # ========================================================
    # 1. SERVER-OWNED READINESS DECISION
    #
    # No requested dataset IDs are supplied here.
    #
    # Analysis Readiness Gate v0.2 therefore resolves the
    # effective request scope from analysis_output_dataset_ids.
    # ========================================================

    decision = (
        require_analysis_readiness(
            workflow_id=
                workflow_id
        )
    )


    dataset_ids = tuple(
        decision
        .requested_analysis_dataset_ids
    )


    if not (
        dataset_ids
    ):
        raise AnalysisInputHandoffError(
            (
                "Analysis readiness returned an empty "
                "final dataset scope."
            )
        )


    # ========================================================
    # 2. LOAD CERTIFIED ARTIFACTS
    # ========================================================

    manifests = []

    dataset_records: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for dataset_id in (
        dataset_ids
    ):
        try:
            artifact = (
                get_preparation_artifact(
                    workflow_id=
                        decision.workflow_id,

                    dataset_id=
                        dataset_id,
                )
            )


        except (
            PreparationArtifactWorkflowNotFoundError,
            PreparationArtifactDatasetNotFoundError,
        ) as error:
            raise (
                AnalysisPreparedArtifactUnavailableError(
                    workflow_id=
                        decision.workflow_id,

                    dataset_id=
                        dataset_id,
                )
            ) from error


        filename = (
            artifact
            .dataset_filename
            .strip()
        )


        extension = (
            _artifact_extension(
                filename
            )
        )


        dataframe = (
            artifact.dataframe
        )


        # ====================================================
        # 3. REBUILD ANALYTICAL MANIFEST
        #
        # The manifest is derived from the exact DataFrame that
        # will be supplied to Analysis.
        # ====================================================

        manifest = (
            build_dataset_manifest(
                dataframe,

                dataset_id=
                    artifact.dataset_id,

                filename=
                    filename,

                extension=
                    extension,
            )
        )


        manifests.append(
            manifest
        )


        # ====================================================
        # 4. ANALYSIS RECORD
        #
        # Core keys mirror load_uploaded_dataset_bundle().
        #
        # Additional Preparation metadata is server-owned and
        # useful for auditability. Existing analytical code
        # ignores unknown record keys.
        # ====================================================

        dataset_records.append(
            {
                "dataset_id":
                    artifact.dataset_id,

                "filename":
                    filename,

                "extension":
                    extension,

                "dataframe":
                    dataframe,

                "preparation_workflow_id":
                    artifact.workflow_id,

                "preparation_stage":
                    artifact.stage,

                "preparation_parent_dataset_ids":
                    list(
                        artifact
                        .parent_dataset_ids
                    ),

                "preparation_evidence_refs":
                    list(
                        artifact
                        .evidence_refs
                    ),

                "analysis_input_rule_version":
                    (
                        ANALYSIS_INPUT_HANDOFF_RULE_VERSION
                    ),
            }
        )


    # ========================================================
    # 5. INGESTION-COMPATIBLE READ MODEL
    #
    # Contextualized analysis and planner components already
    # consume MultiDatasetIngestion. Rebuilding it here lets
    # them describe the exact validated outputs instead of
    # stale browser uploads.
    # ========================================================

    total_rows = sum(
        int(
            manifest.row_count
        )

        for manifest
        in manifests
    )


    ingestion = (
        MultiDatasetIngestion(
            dataset_count=
                len(
                    manifests
                ),

            total_rows=
                total_rows,

            datasets=
                manifests,

            warnings=[
                (
                    "Analysis input was loaded from "
                    "server-owned validated Preparation "
                    "artifacts."
                )
            ],
        )
    )


    # ========================================================
    # 6. FINAL DEFENSIVE READINESS CHECK
    #
    # Preparation state may theoretically change between the
    # initial gate evaluation and the completion of Artifact
    # Store reads.
    #
    # Re-evaluating the gate guarantees that a stale workflow
    # state is not silently handed to Analysis.
    # ========================================================

    final_decision = (
        require_analysis_readiness(
            workflow_id=
                decision.workflow_id
        )
    )


    if (
        final_decision.session_revision
        !=
        decision.session_revision
    ):
        raise AnalysisInputHandoffError(
            (
                "Preparation session changed while the "
                "validated analysis input was being loaded. "
                f"workflow_id={decision.workflow_id}, "
                "initial_revision="
                f"{decision.session_revision}, "
                "current_revision="
                f"{final_decision.session_revision}"
            )
        )


    if (
        tuple(
            final_decision
            .requested_analysis_dataset_ids
        )
        !=
        dataset_ids
    ):
        raise AnalysisInputHandoffError(
            (
                "Preparation final analysis output scope "
                "changed while the analysis input was "
                "being loaded."
            )
        )


    # ========================================================
    # RESULT
    # ========================================================

    return (
        AnalysisInputHandoff(
            workflow_id=
                decision.workflow_id,

            session_revision=
                decision.session_revision,

            dataset_ids=
                dataset_ids,

            ingestion=
                ingestion,

            dataset_records=
                tuple(
                    dataset_records
                ),

            rule_version=
                ANALYSIS_INPUT_HANDOFF_RULE_VERSION,
        )
    )