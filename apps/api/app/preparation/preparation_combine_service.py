from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from app.preparation.join_approval import (
    JoinApprovalCommand,
    JoinApprovalDecision,
    apply_join_approvals,
)
from app.preparation.join_contracts import (
    JoinCardinality,
    JoinIntent,
    JoinKeyPair,
    JoinPlan,
    JoinType,
)
from app.preparation.join_executor import JoinExecutionResult, execute_join_plan
from app.preparation.join_planner import plan_joins
from app.preparation.post_join_validation import (
    PostJoinValidationReport,
    validate_join_execution,
)
from app.preparation.dataset_identity import (
    profile_dataset_identity,
)
from app.preparation.preparation_identity_resolution import (
    get_current_identity_resolution,
)
from app.preparation.preparation_artifact_store import (
    PreparationDatasetArtifactInfo,
    get_preparation_dataframe_map,
    list_preparation_artifacts,
    put_preparation_artifact,
)
from app.preparation.preparation_session import (
    PreparationSessionView,
    get_preparation_session,
    record_optional_stage_signal,
)
from app.preparation.preparation_workflow import PreparationStage


PREPARATION_COMBINE_SERVICE_VERSION = "preparation_combine_service_v0.3"


class CombineIdentityResolutionRequiredError(
    ValueError,
):
    pass


@dataclass(frozen=True)
class CombineDiscovery:
    workflow_id: str
    active_dataset_ids: tuple[str, ...]
    intent: JoinIntent | None
    plan: JoinPlan | None
    reason: str
    rule_version: str = PREPARATION_COMBINE_SERVICE_VERSION

    @property
    def has_candidate(self) -> bool:
        return self.intent is not None and self.plan is not None

    @property
    def ready_for_approval(self) -> bool:
        return bool(self.plan is not None and self.plan.ready_for_approval)


@dataclass(frozen=True)
class CombineExecution:
    workflow_id: str
    request_id: str
    output_dataset_id: str
    output_dataset_filename: str
    rows: int
    columns: int
    parent_dataset_ids: tuple[str, ...]
    validation: PostJoinValidationReport
    next_discovery: CombineDiscovery
    session: PreparationSessionView
    rule_version: str = PREPARATION_COMBINE_SERVICE_VERSION


@dataclass(frozen=True)
class _CandidateSeed:
    left_dataset_id: str
    right_dataset_id: str
    key_column: str
    expected_cardinality: JoinCardinality
    sort_key: tuple[object, ...]


def _stage_status(
    session: PreparationSessionView,
    stage: PreparationStage,
) -> str:
    record = next(
        (item for item in session.snapshot.stages if item.stage == stage),
        None,
    )
    if record is None:
        raise RuntimeError(
            f"Preparation stage is missing from snapshot: {stage.value}"
        )
    return str(record.status.value)


def _require_combine_window(session: PreparationSessionView) -> None:
    transform_status = _stage_status(session, PreparationStage.TRANSFORM)
    if transform_status not in {"passed", "skipped"}:
        raise ValueError(
            "Combine cannot start until TRANSFORM is resolved. "
            f"Current TRANSFORM status: {transform_status}."
        )

    validate_status = _stage_status(session, PreparationStage.VALIDATE)
    if validate_status == "passed":
        raise ValueError(
            "Combine cannot modify Preparation after VALIDATE has passed."
        )


def _active_artifacts(
    workflow_id: str,
) -> list[PreparationDatasetArtifactInfo]:
    """
    Return the terminal materialized Preparation frontier.

    A dataset ceases to be active as soon as another materialized
    artifact explicitly names it as a parent.

    This applies across the whole lineage:

        source -> clean -> transform -> combine

    and not only to COMBINE outputs.

    Important compatibility rule:

    some legacy CLEAN / TRANSFORM materializations may replace an
    artifact in-place and therefore contain their own dataset_id in
    parent_dataset_ids. A self-parent reference must not consume the
    artifact itself.
    """

    artifacts = list_preparation_artifacts(
        workflow_id=workflow_id
    )

    if not artifacts:
        raise ValueError(
            "Combine requires at least one materialized Preparation artifact."
        )

    known_dataset_ids = {
        artifact.dataset_id
        for artifact in artifacts
    }

    consumed_dataset_ids: set[str] = set()

    for artifact in artifacts:
        for parent_dataset_id in artifact.parent_dataset_ids:
            if (
                parent_dataset_id == artifact.dataset_id
            ):
                # In-place materialization: the current artifact
                # already represents the latest state of that ID.
                continue

            if parent_dataset_id in known_dataset_ids:
                consumed_dataset_ids.add(
                    parent_dataset_id
                )

    active = [
        artifact
        for artifact in artifacts
        if artifact.dataset_id not in consumed_dataset_ids
    ]

    return sorted(
        active,
        key=lambda artifact: (
            artifact.dataset_filename.lower(),
            artifact.dataset_id,
        ),
    )


def _is_automatic_join_key(column: object) -> bool:
    name = str(column).strip().lower()
    # Deliberately conservative: bare `id` is too ambiguous.
    return bool(name) and name.endswith("_id")


def _non_null_unique(series: pd.Series) -> bool:
    values = series.dropna()
    if values.empty:
        return False
    return not bool(values.duplicated().any())


def _shared_id_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> list[str]:
    left_names = {
        str(column)
        for column in left.columns
        if _is_automatic_join_key(column)
    }
    right_names = {
        str(column)
        for column in right.columns
        if _is_automatic_join_key(column)
    }
    return sorted(left_names & right_names)


def _candidate_seed(
    *,
    first: PreparationDatasetArtifactInfo,
    second: PreparationDatasetArtifactInfo,
    first_frame: pd.DataFrame,
    second_frame: pd.DataFrame,
    key_column: str,
) -> _CandidateSeed:
    first_unique = _non_null_unique(first_frame[key_column])
    second_unique = _non_null_unique(second_frame[key_column])

    if not first_unique and second_unique:
        left, right = first, second
        expected = JoinCardinality.MANY_TO_ONE
    elif first_unique and not second_unique:
        left, right = second, first
        expected = JoinCardinality.MANY_TO_ONE
    elif first_unique and second_unique:
        left, right = (
            (first, second)
            if first.rows >= second.rows
            else (second, first)
        )
        expected = JoinCardinality.ONE_TO_ONE
    else:
        left, right = (
            (first, second)
            if first.rows >= second.rows
            else (second, first)
        )
        expected = JoinCardinality.MANY_TO_MANY

    cardinality_rank = {
        JoinCardinality.MANY_TO_ONE: 0,
        JoinCardinality.ONE_TO_ONE: 1,
        JoinCardinality.MANY_TO_MANY: 2,
    }.get(expected, 3)

    return _CandidateSeed(
        left_dataset_id=left.dataset_id,
        right_dataset_id=right.dataset_id,
        key_column=key_column,
        expected_cardinality=expected,
        sort_key=(
            cardinality_rank,
            -left.rows,
            key_column.lower(),
            left.dataset_filename.lower(),
            right.dataset_filename.lower(),
        ),
    )


def _candidate_seeds(
    *,
    artifacts: list[PreparationDatasetArtifactInfo],
    frames: dict[str, pd.DataFrame],
) -> list[_CandidateSeed]:
    seeds: list[_CandidateSeed] = []

    for left_index in range(len(artifacts)):
        for right_index in range(left_index + 1, len(artifacts)):
            first = artifacts[left_index]
            second = artifacts[right_index]
            first_frame = frames[first.dataset_id]
            second_frame = frames[second.dataset_id]

            for key_column in _shared_id_columns(first_frame, second_frame):
                seeds.append(
                    _candidate_seed(
                        first=first,
                        second=second,
                        first_frame=first_frame,
                        second_frame=second_frame,
                        key_column=key_column,
                    )
                )

    seeds.sort(key=lambda seed: seed.sort_key)
    return seeds


def _stable_digest(parts: Iterable[str]) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _combined_filename(
    left_filename: str,
    right_filename: str,
    digest: str,
) -> str:
    stem = f"{Path(left_filename).stem}__{Path(right_filename).stem}"
    if len(stem) > 110:
        stem = f"{stem[:90]}__{digest[:12]}"
    return f"{stem}.csv"


def _intent_from_seed(
    *,
    seed: _CandidateSeed,
    artifact_by_id: dict[str, PreparationDatasetArtifactInfo],
) -> JoinIntent:
    left = artifact_by_id[seed.left_dataset_id]
    right = artifact_by_id[seed.right_dataset_id]

    digest = _stable_digest(
        [
            seed.left_dataset_id,
            seed.right_dataset_id,
            seed.key_column,
            JoinType.LEFT.value,
            seed.expected_cardinality.value,
        ]
    )

    return JoinIntent(
        request_id=f"join:{digest}",
        left_dataset_id=left.dataset_id,
        left_dataset_filename=left.dataset_filename,
        right_dataset_id=right.dataset_id,
        right_dataset_filename=right.dataset_filename,
        join_type=JoinType.LEFT,
        keys=[
            JoinKeyPair(
                left_column=seed.key_column,
                right_column=seed.key_column,
            )
        ],
        expected_cardinality=seed.expected_cardinality,
        output_dataset_id=f"combine:{digest}",
        output_dataset_filename=_combined_filename(
            left.dataset_filename,
            right.dataset_filename,
            digest,
        ),
        left_suffix="_left",
        right_suffix="_right",
    )


def _require_identity_frontier_resolved(
    *,
    workflow_id: str,
    artifacts: list[
        PreparationDatasetArtifactInfo
    ],
    frames: dict[
        str,
        pd.DataFrame,
    ],
) -> None:
    unresolved: list[
        str
    ] = []


    for artifact in artifacts:
        dataframe = frames.get(
            artifact.dataset_id
        )


        if (
            dataframe is None
        ):
            raise RuntimeError(
                (
                    "Identity gate could not load active "
                    "Preparation artifact: "
                    f"{artifact.dataset_id}"
                )
            )


        report = (
            profile_dataset_identity(
                dataframe,

                dataset_id=
                    artifact.dataset_id,

                dataset_filename=
                    artifact.dataset_filename,
            )
        )


        resolution = (
            get_current_identity_resolution(
                workflow_id=
                    workflow_id,

                dataset_id=
                    artifact.dataset_id,

                dataset_filename=
                    artifact.dataset_filename,

                artifact_stage=
                    artifact.stage,

                report=
                    report,
            )
        )


        if (
            resolution is None
        ):
            unresolved.append(
                artifact.dataset_filename
            )


    if unresolved:
        raise (
            CombineIdentityResolutionRequiredError(
                (
                    "COMBINE cannot start until row identity "
                    "has been resolved for every active dataset. "
                    "Review the following dataset(s): "
                    +
                    ", ".join(
                        unresolved
                    )
                    +
                    "."
                )
            )
        )


def _discover_without_stage_write(workflow_id: str) -> CombineDiscovery:
    session = get_preparation_session(workflow_id)
    _require_combine_window(session)

    artifacts = _active_artifacts(workflow_id)
    active_ids = tuple(
        artifact.dataset_id
        for artifact in artifacts
    )

    if len(artifacts) < 2:
        return CombineDiscovery(
            workflow_id=workflow_id,
            active_dataset_ids=active_ids,
            intent=None,
            plan=None,
            reason=(
                "No additional join is required because fewer than "
                "two active Preparation artifacts remain."
            ),
        )

    frames = get_preparation_dataframe_map(
        workflow_id=workflow_id,
        dataset_ids=active_ids,
    )

    _require_identity_frontier_resolved(
        workflow_id=workflow_id,
        artifacts=artifacts,
        frames=frames,
    )
    artifact_by_id = {
        artifact.dataset_id: artifact
        for artifact in artifacts
    }
    seeds = _candidate_seeds(
        artifacts=artifacts,
        frames=frames,
    )

    if not seeds:
        return CombineDiscovery(
            workflow_id=workflow_id,
            active_dataset_ids=active_ids,
            intent=None,
            plan=None,
            reason=(
                "No deterministic shared *_id relationship was found "
                "between the active Preparation artifacts."
            ),
        )

    first_blocked: tuple[JoinIntent, JoinPlan] | None = None

    for seed in seeds:
        intent = _intent_from_seed(
            seed=seed,
            artifact_by_id=artifact_by_id,
        )
        plan = plan_joins(
            datasets=frames,
            intents=[intent],
        )

        if plan.ready_for_approval:
            return CombineDiscovery(
                workflow_id=workflow_id,
                active_dataset_ids=active_ids,
                intent=intent,
                plan=plan,
                reason=(
                    "A deterministic relationship candidate passed "
                    "Join Planner safety checks and requires explicit "
                    "analyst approval."
                ),
            )

        if first_blocked is None:
            first_blocked = (intent, plan)

    assert first_blocked is not None
    intent, plan = first_blocked
    return CombineDiscovery(
        workflow_id=workflow_id,
        active_dataset_ids=active_ids,
        intent=intent,
        plan=plan,
        reason=(
            "Relationship candidates were detected, but none passed "
            "the deterministic Join Planner safety checks."
        ),
    )


def _evidence_refs(discovery: CombineDiscovery) -> list[str]:
    refs = [
        f"combine_service:{PREPARATION_COMBINE_SERVICE_VERSION}",
    ]
    if discovery.plan is not None:
        refs.append(f"join_plan:{discovery.plan.rule_version}")
    if discovery.intent is not None:
        refs.append(f"join_request:{discovery.intent.request_id}")
    return refs


def _sync_discovery_stage(
    discovery: CombineDiscovery,
) -> PreparationSessionView:
    if not discovery.has_candidate:
        has_existing_combine = any(
            artifact.stage == "combine"
            for artifact in list_preparation_artifacts(
                workflow_id=discovery.workflow_id
            )
        )
        return record_optional_stage_signal(
            workflow_id=discovery.workflow_id,
            stage=PreparationStage.COMBINE,
            required=has_existing_combine,
            completed=has_existing_combine,
            review_required=False,
            blocked=False,
            dataset_ids=list(discovery.active_dataset_ids),
            evidence_refs=_evidence_refs(discovery),
            blocking_reasons=[],
        )

    assert discovery.plan is not None

    if discovery.plan.ready_for_approval:
        return record_optional_stage_signal(
            workflow_id=discovery.workflow_id,
            stage=PreparationStage.COMBINE,
            required=True,
            completed=False,
            review_required=True,
            blocked=False,
            dataset_ids=list(discovery.active_dataset_ids),
            evidence_refs=_evidence_refs(discovery),
            blocking_reasons=[],
        )

    planned = discovery.plan.joins[0]
    reasons = [
        reason
        for reason in [planned.rationale, *planned.warnings]
        if str(reason).strip()
    ]
    return record_optional_stage_signal(
        workflow_id=discovery.workflow_id,
        stage=PreparationStage.COMBINE,
        required=True,
        completed=False,
        review_required=False,
        blocked=True,
        dataset_ids=list(discovery.active_dataset_ids),
        evidence_refs=_evidence_refs(discovery),
        blocking_reasons=reasons,
    )


def discover_next_combine(
    workflow_id: str,
    *,
    synchronize_stage: bool = True,
) -> CombineDiscovery:
    discovery = _discover_without_stage_write(workflow_id)
    if synchronize_stage:
        _sync_discovery_stage(discovery)
    return discovery


def approve_and_execute_next_combine(
    *,
    workflow_id: str,
    request_id: str,
    actor: str = "user",
    comment: str | None = None,
) -> CombineExecution:
    discovery = _discover_without_stage_write(workflow_id)

    if not discovery.has_candidate:
        raise ValueError(
            "No join candidate is currently available for approval."
        )

    assert discovery.intent is not None
    assert discovery.plan is not None

    if discovery.intent.request_id != request_id.strip():
        raise ValueError(
            "Join approval request_id does not match the current "
            "server-derived candidate. Refresh the Preparation plan."
        )

    if not discovery.plan.ready_for_approval:
        _sync_discovery_stage(discovery)
        raise ValueError(
            "The current join candidate is blocked and cannot be approved."
        )

    source_frames = get_preparation_dataframe_map(
        workflow_id=workflow_id,
        dataset_ids=discovery.active_dataset_ids,
    )

    approved_plan = apply_join_approvals(
        plan=discovery.plan,
        commands=[
            JoinApprovalCommand(
                request_id=discovery.intent.request_id,
                decision=JoinApprovalDecision.APPROVE,
                actor=actor,
                comment=comment,
            )
        ],
    )

    execution: JoinExecutionResult = execute_join_plan(
        datasets=source_frames,
        approved_plan=approved_plan,
    )

    validation = validate_join_execution(
        source_datasets=source_frames,
        approved_plan=approved_plan,
        execution_result=execution,
    )

    if not validation.valid_for_downstream:
        record_optional_stage_signal(
            workflow_id=workflow_id,
            stage=PreparationStage.COMBINE,
            required=True,
            completed=False,
            review_required=False,
            blocked=True,
            dataset_ids=list(discovery.active_dataset_ids),
            evidence_refs=[
                *_evidence_refs(discovery),
                f"join_approval:{approved_plan.rule_version}",
                f"join_execution:{execution.report.rule_version}",
                f"post_join_validation:{validation.rule_version}",
            ],
            blocking_reasons=[
                "Post-join validation rejected the materialized output."
            ],
        )
        raise RuntimeError(
            "Post-join validation rejected the materialized output."
        )

    output_id = discovery.intent.output_dataset_id
    output_frame = execution.joined_datasets.get(output_id)
    if output_frame is None:
        raise RuntimeError(
            "Join Executor did not return the approved output dataset: "
            f"{output_id}"
        )

    artifact_info = put_preparation_artifact(
        workflow_id=workflow_id,
        dataset_id=output_id,
        dataset_filename=discovery.intent.output_dataset_filename,
        stage="combine",
        dataframe=output_frame,
        parent_dataset_ids=[
            discovery.intent.left_dataset_id,
            discovery.intent.right_dataset_id,
        ],
        evidence_refs=[
            *_evidence_refs(discovery),
            f"join_approval:{approved_plan.rule_version}",
            f"join_execution:{execution.report.rule_version}",
            f"post_join_validation:{validation.rule_version}",
        ],
        replace=False,
    )

    next_discovery = _discover_without_stage_write(workflow_id)
    session = _sync_discovery_stage(next_discovery)

    return CombineExecution(
        workflow_id=workflow_id,
        request_id=discovery.intent.request_id,
        output_dataset_id=artifact_info.dataset_id,
        output_dataset_filename=artifact_info.dataset_filename,
        rows=artifact_info.rows,
        columns=artifact_info.columns,
        parent_dataset_ids=artifact_info.parent_dataset_ids,
        validation=validation,
        next_discovery=next_discovery,
        session=session,
    )
