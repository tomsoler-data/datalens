from __future__ import annotations

import json
from typing import Any

import pandas as pd
from pydantic import BaseModel

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from app.api.routes import (
    load_uploaded_dataset_bundle,
)

from app.preparation.cleaning_engine import (
    CleaningPlan,
    build_cleaning_plan,
    execute_cleaning_plan,
)

from app.preparation.data_quality import (
    build_data_quality_report,
)

from app.preparation.semantic_review import (
    DEFAULT_SEMANTIC_REVIEW_MODEL,
    RawSemanticDecision,
    RawSemanticReviewResponse,
    SemanticReviewReport,
    SemanticVerdict,
    ValidatedSemanticDecision,
    build_semantic_review_candidates,
    review_quality_semantics,
    validate_semantic_review_response,
)

from app.preparation.semantic_cleaning import (
    SemanticCleaningChoice,
    SemanticCleaningExecutionResult,
    SemanticCleaningPlan,
    build_semantic_cleaning_plan,
    execute_semantic_cleaning_plan,
)

from app.preparation.semantic_cleaning_artifacts import (
    materialize_semantic_cleaning_artifacts,
)

from app.preparation.semantic_confirmation import (
    SemanticConfirmationBlockedError,
    SemanticConfirmationReport,
    SemanticManualResolution,
    require_semantic_confirmation,
)

from app.preparation.preparation_session import (
    PreparationSessionNotFoundError,
    get_preparation_session,
    record_optional_stage_signal,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
)


# ============================================================
# ROUTER
# ============================================================


router = APIRouter(
    prefix="/preparation",
    tags=[
        "preparation",
    ],
)


# ============================================================
# RESPONSES
# ============================================================


class SemanticCleaningApplyResponse(
    BaseModel,
):
    plan: SemanticCleaningPlan

    execution: (
        SemanticCleaningExecutionResult
    )


class SemanticReviewConfirmationResponse(
    BaseModel,
):
    status: str

    confirmation: (
        SemanticConfirmationReport
    )

    plan: (
        SemanticCleaningPlan
    )

    execution: (
        SemanticCleaningExecutionResult
    )


# ============================================================
# DATASET HELPERS
# ============================================================


def _dataset_ids_from_records(
    records: list[
        dict[
            str,
            Any,
        ]
    ],
) -> list[
    str
]:
    dataset_ids: list[
        str
    ] = []


    for record in records:
        dataset_id = str(
            record.get(
                "dataset_id"
            )
            or
            ""
        ).strip()


        if not dataset_id:
            raise RuntimeError(
                (
                    "Semantic preparation received "
                    "an internal dataset record "
                    "without dataset_id."
                )
            )


        dataset_ids.append(
            dataset_id
        )


    if not dataset_ids:
        raise RuntimeError(
            (
                "Semantic preparation received "
                "no internal dataset records."
            )
        )


    return dataset_ids


def _validate_session_dataset_scope(
    *,
    workflow_id: str,

    records: list[
        dict[
            str,
            Any,
        ]
    ],
) -> list[
    str
]:
    """
    Ensure uploaded datasets correspond exactly to the
    server-owned Preparation Session.

    Dataset IDs are deterministic and positional:

        dataset:0001
        dataset:0002
        ...
    """

    session = (
        get_preparation_session(
            workflow_id
        )
    )


    uploaded_dataset_ids = (
        _dataset_ids_from_records(
            records
        )
    )


    expected_dataset_ids = list(
        session
        .selected_analysis_dataset_ids
    )


    if (
        uploaded_dataset_ids
        !=
        expected_dataset_ids
    ):
        raise HTTPException(
            status_code=409,

            detail={
                "error": (
                    "preparation_dataset_scope_mismatch"
                ),

                "message": (
                    "The uploaded datasets do not match "
                    "the preparation session dataset scope."
                ),

                "workflow_id": (
                    workflow_id
                ),

                "expected_dataset_ids": (
                    expected_dataset_ids
                ),

                "uploaded_dataset_ids": (
                    uploaded_dataset_ids
                ),
            },
        )


    return (
        uploaded_dataset_ids
    )


# ============================================================
# CLEAN STAGE
# ============================================================


def _clean_stage_record(
    workflow_id: str,
):
    session = (
        get_preparation_session(
            workflow_id
        )
    )


    for stage in (
        session.snapshot.stages
    ):
        if (
            stage.stage
            ==
            PreparationStage.CLEAN
        ):
            return stage


    raise RuntimeError(
        (
            "Preparation workflow is missing "
            "the CLEAN stage."
        )
    )


def _has_evidence_prefix(
    *,
    evidence_refs: list[
        str
    ],

    prefix: str,
) -> bool:
    return any(
        str(
            evidence
        ).startswith(
            prefix
        )

        for evidence
        in evidence_refs
    )


def _merge_evidence_refs(
    *,
    current: list[
        str
    ],

    additions: list[
        str
    ],
) -> list[
    str
]:
    output: list[
        str
    ] = []

    seen: set[
        str
    ] = set()


    for value in [
        *current,
        *additions,
    ]:
        normalized = str(
            value
        ).strip()


        if (
            not normalized
            or
            normalized
            in seen
        ):
            continue


        seen.add(
            normalized
        )

        output.append(
            normalized
        )


    return output


# ============================================================
# PRECONDITIONS
# ============================================================


def _require_deterministic_cleaning_precondition(
    *,
    workflow_id: str,

    deterministic_plan: CleaningPlan,
) -> None:
    """
    Semantic review must never bypass deterministic cleaning.

    If the deterministic plan contained executable actions,
    /cleaning-apply must have been executed first.
    """

    if (
        deterministic_plan.action_count
        ==
        0
    ):
        return


    clean_stage = (
        _clean_stage_record(
            workflow_id
        )
    )


    cleaning_executed = (
        _has_evidence_prefix(
            evidence_refs=
                clean_stage.evidence_refs,

            prefix=
                "cleaning_execution:",
        )
    )


    if not cleaning_executed:
        raise HTTPException(
            status_code=409,

            detail={
                "error": (
                    "deterministic_cleaning_required"
                ),

                "message": (
                    "Deterministic cleaning must be "
                    "executed before semantic review."
                ),

                "workflow_id": (
                    workflow_id
                ),

                "deterministic_action_count": (
                    deterministic_plan
                    .action_count
                ),
            },
        )


def _require_semantic_review_precondition(
    *,
    workflow_id: str,
) -> None:
    """
    Final semantic confirmation requires evidence that the
    server actually ran /semantic-review for this session.
    """

    clean_stage = (
        _clean_stage_record(
            workflow_id
        )
    )


    semantic_reviewed = (
        _has_evidence_prefix(
            evidence_refs=
                clean_stage.evidence_refs,

            prefix=
                "semantic_review:",
        )
    )


    if not semantic_reviewed:
        raise HTTPException(
            status_code=409,

            detail={
                "error": (
                    "semantic_review_required"
                ),

                "message": (
                    "Semantic review must be executed "
                    "before semantic confirmation."
                ),

                "workflow_id": (
                    workflow_id
                ),
            },
        )


# ============================================================
# SEMANTIC WORKFLOW STATE
# ============================================================


def _semantic_flagged_count(
    report: SemanticReviewReport,
) -> int:
    return sum(
        decision.verdict
        ==
        SemanticVerdict.FLAG_FOR_REVIEW

        for decision
        in report.decisions
    )


def _record_semantic_review_stage(
    *,
    workflow_id: str,

    dataset_ids: list[
        str
    ],

    deterministic_plan: CleaningPlan,

    report: SemanticReviewReport,
) -> None:
    """
    Record semantic-review evidence without granting CLEAN.

    SemanticReviewReport is proposal-only.
    """

    if (
        report.candidate_count
        ==
        0
        and
        deterministic_plan
        .protected_issue_count
        ==
        0
    ):
        return


    flagged_count = (
        _semantic_flagged_count(
            report
        )
    )


    current_clean = (
        _clean_stage_record(
            workflow_id
        )
    )


    evidence_refs = (
        _merge_evidence_refs(
            current=
                current_clean
                .evidence_refs,

            additions=[
                (
                    "semantic_review:"
                    f"{report.rule_version}"
                ),

                (
                    "semantic_review_candidates:"
                    f"{report.candidate_count}"
                ),

                (
                    "semantic_review_decisions:"
                    f"{report.decision_count}"
                ),

                (
                    "semantic_review_merge_proposals:"
                    f"{report.merge_proposal_count}"
                ),

                (
                    "semantic_review_abstentions:"
                    f"{report.abstention_count}"
                ),

                (
                    "semantic_review_flagged:"
                    f"{flagged_count}"
                ),
            ],
        )
    )


    blocking_reasons: list[
        str
    ] = []


    if (
        report.candidate_count >
        0
    ):
        blocking_reasons.append(
            (
                "La revue sémantique a produit "
                f"{report.decision_count} "
                "décision(s) validée(s) par Python. "
                "Une confirmation analyste explicite "
                "reste nécessaire."
            )
        )


    if (
        report.merge_proposal_count >
        0
    ):
        blocking_reasons.append(
            (
                f"{report.merge_proposal_count} "
                "proposition(s) de fusion sémantique "
                "nécessitent une décision humaine."
            )
        )


    if (
        report.abstention_count >
        0
    ):
        blocking_reasons.append(
            (
                f"{report.abstention_count} "
                "décision(s) ABSTAIN nécessitent "
                "une résolution analyste."
            )
        )


    if (
        flagged_count >
        0
    ):
        blocking_reasons.append(
            (
                f"{flagged_count} "
                "signal(aux) FLAG_FOR_REVIEW "
                "nécessitent une résolution analyste."
            )
        )


    if (
        report.candidate_count
        ==
        0
        and
        deterministic_plan
        .protected_issue_count
        >
        0
    ):
        blocking_reasons.append(
            (
                "Des problèmes protégés subsistent, "
                "mais aucun candidat exploitable "
                "n’a été produit par la revue "
                "sémantique automatique."
            )
        )


    record_optional_stage_signal(
        workflow_id=
            workflow_id,

        stage=
            PreparationStage.CLEAN,

        required=
            True,

        completed=
            False,

        review_required=
            True,

        blocked=
            False,

        dataset_ids=
            dataset_ids,

        evidence_refs=
            evidence_refs,

        blocking_reasons=
            blocking_reasons,
    )


# ============================================================
# SEMANTIC CONFIRMATION WORKFLOW STATE
# ============================================================


def _record_semantic_confirmation_passed(
    *,
    workflow_id: str,

    dataset_ids: list[
        str
    ],

    confirmation: (
        SemanticConfirmationReport
    ),

    execution: (
        SemanticCleaningExecutionResult
    ),
) -> None:
    current_clean = (
        _clean_stage_record(
            workflow_id
        )
    )


    evidence_refs = (
        _merge_evidence_refs(
            current=
                current_clean
                .evidence_refs,

            additions=[
                (
                    "semantic_confirmation:"
                    f"{confirmation.rule_version}"
                ),

                (
                    "semantic_confirmation_confirmed:"
                    f"{confirmation.confirmed_issue_count}"
                ),

                (
                    "semantic_confirmation_manual:"
                    f"{confirmation.manual_resolution_count}"
                ),

                (
                    "semantic_cleaning_execution:"
                    f"{execution.rule_version}"
                ),

                (
                    "semantic_cleaning_applied:"
                    f"{execution.applied_action_count}"
                ),

                (
                    "semantic_cleaning_skipped:"
                    f"{execution.skipped_action_count}"
                ),
            ],
        )
    )


    record_optional_stage_signal(
        workflow_id=
            workflow_id,

        stage=
            PreparationStage.CLEAN,

        required=
            True,

        completed=
            True,

        review_required=
            False,

        blocked=
            False,

        dataset_ids=
            dataset_ids,

        evidence_refs=
            evidence_refs,

        blocking_reasons=[],
    )


def _record_semantic_confirmation_blocked(
    *,
    workflow_id: str,

    dataset_ids: list[
        str
    ],

    confirmation: (
        SemanticConfirmationReport
    ),
) -> None:
    current_clean = (
        _clean_stage_record(
            workflow_id
        )
    )


    evidence_refs = (
        _merge_evidence_refs(
            current=
                current_clean
                .evidence_refs,

            additions=[
                (
                    "semantic_confirmation:"
                    f"{confirmation.rule_version}"
                ),

                (
                    "semantic_confirmation_unresolved:"
                    f"{len(confirmation.unresolved_issue_ids)}"
                ),
            ],
        )
    )


    reasons = (
        list(
            confirmation
            .unresolved_reasons
        )
    )


    if not reasons:
        reasons = [
            (
                "La confirmation sémantique "
                "reste incomplète."
            )
        ]


    record_optional_stage_signal(
        workflow_id=
            workflow_id,

        stage=
            PreparationStage.CLEAN,

        required=
            True,

        completed=
            False,

        review_required=
            True,

        blocked=
            False,

        dataset_ids=
            dataset_ids,

        evidence_refs=
            evidence_refs,

        blocking_reasons=
            reasons,
    )


# ============================================================
# CLEANING ACTION IDS
# ============================================================


def _parse_approved_action_ids(
    raw_value: str | None,
) -> set[
    str
]:
    if raw_value is None:
        return set()


    normalized = (
        raw_value.strip()
    )


    if not normalized:
        return set()


    try:
        parsed = json.loads(
            normalized
        )


    except json.JSONDecodeError as error:
        raise ValueError(
            "approved_action_ids_json "
            "must be a valid JSON array."
        ) from error


    if not isinstance(
        parsed,
        list,
    ):
        raise ValueError(
            "approved_action_ids_json "
            "must contain a JSON array."
        )


    action_ids: set[
        str
    ] = set()


    for value in parsed:
        if not isinstance(
            value,
            str,
        ):
            raise ValueError(
                "Every approved cleaning "
                "action id must be a string."
            )


        action_id = (
            value.strip()
        )


        if not action_id:
            raise ValueError(
                "Approved cleaning action "
                "ids cannot be empty."
            )


        action_ids.add(
            action_id
        )


    return action_ids


# ============================================================
# DATAFRAME HELPERS
# ============================================================


def _frames_from_records(
    records: list[
        dict[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    pd.DataFrame,
]:
    frames: dict[
        str,
        pd.DataFrame,
    ] = {}


    for record in records:
        dataset_id = str(
            record[
                "dataset_id"
            ]
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
                "Dataset record is missing "
                f"a DataFrame: {dataset_id}"
            )


        frames[
            dataset_id
        ] = dataframe


    return frames


def _records_with_derived_frames(
    *,
    records: list[
        dict[
            str,
            Any,
        ]
    ],

    derived_frames: dict[
        str,
        pd.DataFrame,
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    output: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for record in records:
        dataset_id = str(
            record[
                "dataset_id"
            ]
        )


        dataframe = (
            derived_frames.get(
                dataset_id
            )
        )


        if dataframe is None:
            raise KeyError(
                "Missing derived dataframe "
                f"for {dataset_id}."
            )


        derived_record = dict(
            record
        )


        derived_record[
            "dataframe"
        ] = dataframe


        output.append(
            derived_record
        )


    return output


# ============================================================
# SEMANTIC DECISIONS
# ============================================================


def _parse_semantic_decisions(
    raw_value: str,
) -> RawSemanticReviewResponse:
    normalized = (
        raw_value.strip()
    )


    if not normalized:
        raise ValueError(
            "semantic_decisions_json "
            "cannot be empty."
        )


    try:
        parsed = json.loads(
            normalized
        )


    except json.JSONDecodeError as error:
        raise ValueError(
            "semantic_decisions_json "
            "must be valid JSON."
        ) from error


    if isinstance(
        parsed,
        dict,
    ):
        parsed = (
            parsed.get(
                "decisions"
            )
        )


    if not isinstance(
        parsed,
        list,
    ):
        raise ValueError(
            "semantic_decisions_json "
            "must contain a decisions array."
        )


    decisions: list[
        RawSemanticDecision
    ] = []


    for item in parsed:
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                "Every semantic decision "
                "must be a JSON object."
            )


        decisions.append(
            RawSemanticDecision
            .model_validate(
                item
            )
        )


    return (
        RawSemanticReviewResponse(
            decisions=
                decisions
        )
    )


# ============================================================
# SEMANTIC CHOICES
# ============================================================


def _parse_semantic_choices(
    raw_value: str,
) -> list[
    SemanticCleaningChoice
]:
    normalized = (
        raw_value.strip()
    )


    if not normalized:
        return []


    try:
        parsed = json.loads(
            normalized
        )


    except json.JSONDecodeError as error:
        raise ValueError(
            "approved_semantic_choices_json "
            "must be valid JSON."
        ) from error


    if not isinstance(
        parsed,
        list,
    ):
        raise ValueError(
            "approved_semantic_choices_json "
            "must contain a JSON array."
        )


    return [
        SemanticCleaningChoice
        .model_validate(
            item
        )

        for item
        in parsed
    ]


# ============================================================
# CONFIRMED ISSUE IDS
# ============================================================


def _parse_confirmed_issue_ids(
    raw_value: str,
) -> list[
    str
]:
    normalized = (
        raw_value.strip()
    )


    if not normalized:
        raise ValueError(
            "confirmed_issue_ids_json "
            "cannot be empty."
        )


    try:
        parsed = json.loads(
            normalized
        )


    except json.JSONDecodeError as error:
        raise ValueError(
            "confirmed_issue_ids_json "
            "must be valid JSON."
        ) from error


    if not isinstance(
        parsed,
        list,
    ):
        raise ValueError(
            "confirmed_issue_ids_json "
            "must contain a JSON array."
        )


    output: list[
        str
    ] = []


    for value in parsed:
        if not isinstance(
            value,
            str,
        ):
            raise ValueError(
                "Every confirmed issue id "
                "must be a string."
            )


        issue_id = (
            value.strip()
        )


        if not issue_id:
            raise ValueError(
                "Confirmed issue ids "
                "cannot be empty."
            )


        output.append(
            issue_id
        )


    return output


# ============================================================
# MANUAL RESOLUTIONS
# ============================================================


def _parse_manual_resolutions(
    raw_value: str | None,
) -> list[
    SemanticManualResolution
]:
    if raw_value is None:
        return []


    normalized = (
        raw_value.strip()
    )


    if not normalized:
        return []


    try:
        parsed = json.loads(
            normalized
        )


    except json.JSONDecodeError as error:
        raise ValueError(
            "manual_resolutions_json "
            "must be valid JSON."
        ) from error


    if not isinstance(
        parsed,
        list,
    ):
        raise ValueError(
            "manual_resolutions_json "
            "must contain a JSON array."
        )


    return [
        SemanticManualResolution
        .model_validate(
            item
        )

        for item
        in parsed
    ]


# ============================================================
# SEMANTIC CLEANING CONTEXT
# ============================================================


def _rebuild_semantic_cleaning_context(
    *,
    dataset_files: list[
        UploadFile
    ],

    workflow_id: str,

    approved_action_ids_json: (
        str
        | None
    ),

    semantic_decisions_json: str,
) -> tuple[
    dict[
        str,
        pd.DataFrame,
    ],

    list[
        ValidatedSemanticDecision
    ],

    SemanticCleaningPlan,
]:
    (
        _,
        records,
    ) = (
        load_uploaded_dataset_bundle(
            dataset_files
        )
    )


    _validate_session_dataset_scope(
        workflow_id=
            workflow_id,

        records=
            records,
    )


    source_frames = (
        _frames_from_records(
            records
        )
    )


    source_quality = (
        build_data_quality_report(
            records
        )
    )


    deterministic_plan = (
        build_cleaning_plan(
            source_quality,
            source_frames,
        )
    )


    _require_deterministic_cleaning_precondition(
        workflow_id=
            workflow_id,

        deterministic_plan=
            deterministic_plan,
    )


    approved_ids = (
        _parse_approved_action_ids(
            approved_action_ids_json
        )
    )


    (
        deterministic_frames,
        _,
    ) = execute_cleaning_plan(
        plan=
            deterministic_plan,

        dataset_frames=
            source_frames,

        approved_action_ids=
            approved_ids,
    )


    derived_records = (
        _records_with_derived_frames(
            records=
                records,

            derived_frames=
                deterministic_frames,
        )
    )


    derived_quality = (
        build_data_quality_report(
            derived_records
        )
    )


    semantic_candidates = (
        build_semantic_review_candidates(
            quality_report=
                derived_quality,

            dataset_frames=
                deterministic_frames,
        )
    )


    raw_response = (
        _parse_semantic_decisions(
            semantic_decisions_json
        )
    )


    validated_decisions = (
        validate_semantic_review_response(
            raw_response=
                raw_response,

            candidates=
                semantic_candidates,

            dataset_frames=
                deterministic_frames,
        )
    )


    semantic_plan = (
        build_semantic_cleaning_plan(
            validated_decisions
        )
    )


    return (
        deterministic_frames,
        validated_decisions,
        semantic_plan,
    )


# ============================================================
# SEMANTIC REVIEW
# ============================================================


@router.post(
    "/semantic-review",
    response_model=
        SemanticReviewReport,
)
def review_uploaded_dataset_semantics(
    dataset_files: list[
        UploadFile
    ] = File(
        ...,
    ),

    workflow_id: str = Form(
        ...,
        min_length=1,
    ),

    approved_action_ids_json: (
        str
        | None
    ) = Form(
        default=None,
    ),

    model: str = Form(
        default=
            DEFAULT_SEMANTIC_REVIEW_MODEL,

        min_length=1,
    ),
) -> SemanticReviewReport:
    """
    Review unresolved quality signals with the local semantic
    model after deterministic cleaning.

    Proposal-only:
    - no semantic merge is executed;
    - CLEAN is never promoted to PASSED here.
    """

    try:
        (
            _,
            records,
        ) = (
            load_uploaded_dataset_bundle(
                dataset_files
            )
        )


        dataset_ids = (
            _validate_session_dataset_scope(
                workflow_id=
                    workflow_id,

                records=
                    records,
            )
        )


        source_frames = (
            _frames_from_records(
                records
            )
        )


        source_quality = (
            build_data_quality_report(
                records
            )
        )


        cleaning_plan = (
            build_cleaning_plan(
                source_quality,
                source_frames,
            )
        )


        _require_deterministic_cleaning_precondition(
            workflow_id=
                workflow_id,

            deterministic_plan=
                cleaning_plan,
        )


        approved_ids = (
            _parse_approved_action_ids(
                approved_action_ids_json
            )
        )


        (
            derived_frames,
            _,
        ) = execute_cleaning_plan(
            plan=
                cleaning_plan,

            dataset_frames=
                source_frames,

            approved_action_ids=
                approved_ids,
        )


        derived_records = (
            _records_with_derived_frames(
                records=
                    records,

                derived_frames=
                    derived_frames,
            )
        )


        derived_quality = (
            build_data_quality_report(
                derived_records
            )
        )


        report = (
            review_quality_semantics(
                quality_report=
                    derived_quality,

                dataset_frames=
                    derived_frames,

                model=
                    model,
            )
        )


        _record_semantic_review_stage(
            workflow_id=
                workflow_id,

            dataset_ids=
                dataset_ids,

            deterministic_plan=
                cleaning_plan,

            report=
                report,
        )


        return (
            report
        )


    except PreparationSessionNotFoundError as error:
        raise HTTPException(
            status_code=404,

            detail={
                "error": (
                    "preparation_session_not_found"
                ),

                "message": str(
                    error
                ),

                "workflow_id": (
                    workflow_id
                ),
            },
        ) from error


    except HTTPException:
        raise


    except (
        ValueError,
        KeyError,
    ) as error:
        raise HTTPException(
            status_code=422,
            detail=str(
                error
            ),
        ) from error


    except TypeError as error:
        raise HTTPException(
            status_code=500,

            detail=(
                "Semantic review received "
                "invalid internal dataset state: "
                f"{error}"
            ),
        ) from error


    except RuntimeError as error:
        raise HTTPException(
            status_code=503,

            detail=str(
                error
            ),
        ) from error


# ============================================================
# SEMANTIC CLEANING PLAN
# ============================================================


@router.post(
    "/semantic-cleaning-plan",
    response_model=
        SemanticCleaningPlan,
)
def build_uploaded_semantic_cleaning_plan(
    dataset_files: list[
        UploadFile
    ] = File(
        ...,
    ),

    workflow_id: str = Form(
        ...,
        min_length=1,
    ),

    semantic_decisions_json: str = Form(
        ...,
    ),

    approved_action_ids_json: (
        str
        | None
    ) = Form(
        default=None,
    ),
) -> SemanticCleaningPlan:
    """
    Revalidate a semantic review and convert exact validated
    MERGE_VALUES decisions into user-confirmable actions.

    No LLM call.
    No DataFrame mutation.
    """

    try:
        (
            _,
            _,
            semantic_plan,
        ) = (
            _rebuild_semantic_cleaning_context(
                dataset_files=
                    dataset_files,

                workflow_id=
                    workflow_id,

                approved_action_ids_json=
                    approved_action_ids_json,

                semantic_decisions_json=
                    semantic_decisions_json,
            )
        )


        return (
            semantic_plan
        )


    except PreparationSessionNotFoundError as error:
        raise HTTPException(
            status_code=404,

            detail={
                "error": (
                    "preparation_session_not_found"
                ),

                "message": str(
                    error
                ),

                "workflow_id": (
                    workflow_id
                ),
            },
        ) from error


    except HTTPException:
        raise


    except (
        ValueError,
        KeyError,
    ) as error:
        raise HTTPException(
            status_code=422,
            detail=str(
                error
            ),
        ) from error


    except TypeError as error:
        raise HTTPException(
            status_code=500,

            detail=(
                "Semantic cleaning plan received "
                "invalid internal dataset state: "
                f"{error}"
            ),
        ) from error


    except RuntimeError as error:
        raise HTTPException(
            status_code=500,

            detail=(
                "Semantic cleaning plan workflow "
                "synchronization failed: "
                f"{error}"
            ),
        ) from error


# ============================================================
# SEMANTIC CLEANING APPLY
# ============================================================


@router.post(
    "/semantic-cleaning-apply",
    response_model=
        SemanticCleaningApplyResponse,
)
def apply_uploaded_semantic_cleaning(
    dataset_files: list[
        UploadFile
    ] = File(
        ...,
    ),

    workflow_id: str = Form(
        ...,
        min_length=1,
    ),

    semantic_decisions_json: str = Form(
        ...,
    ),

    approved_semantic_choices_json: str = Form(
        ...,
    ),

    approved_action_ids_json: (
        str
        | None
    ) = Form(
        default=None,
    ),
) -> SemanticCleaningApplyResponse:
    """
    Apply exact user-confirmed semantic alias merges.

    Successful merge execution alone does not resolve the
    complete semantic review.
    """

    try:
        (
            deterministic_frames,
            _,
            semantic_plan,
        ) = (
            _rebuild_semantic_cleaning_context(
                dataset_files=
                    dataset_files,

                workflow_id=
                    workflow_id,

                approved_action_ids_json=
                    approved_action_ids_json,

                semantic_decisions_json=
                    semantic_decisions_json,
            )
        )


        choices = (
            _parse_semantic_choices(
                approved_semantic_choices_json
            )
        )


        (
            _,
            execution,
        ) = (
            execute_semantic_cleaning_plan(
                plan=
                    semantic_plan,

                dataset_frames=
                    deterministic_frames,

                approved_choices=
                    choices,
            )
        )


        return (
            SemanticCleaningApplyResponse(
                plan=
                    semantic_plan,

                execution=
                    execution,
            )
        )


    except PreparationSessionNotFoundError as error:
        raise HTTPException(
            status_code=404,

            detail={
                "error": (
                    "preparation_session_not_found"
                ),

                "message": str(
                    error
                ),

                "workflow_id": (
                    workflow_id
                ),
            },
        ) from error


    except HTTPException:
        raise


    except (
        ValueError,
        KeyError,
    ) as error:
        raise HTTPException(
            status_code=422,
            detail=str(
                error
            ),
        ) from error


    except TypeError as error:
        raise HTTPException(
            status_code=500,

            detail=(
                "Semantic cleaning apply received "
                "invalid internal dataset state: "
                f"{error}"
            ),
        ) from error


    except RuntimeError as error:
        raise HTTPException(
            status_code=500,

            detail=(
                "Semantic cleaning apply workflow "
                "synchronization failed: "
                f"{error}"
            ),
        ) from error


# ============================================================
# SEMANTIC REVIEW CONFIRMATION
# ============================================================


@router.post(
    "/semantic-review-confirm",
    response_model=
        SemanticReviewConfirmationResponse,
)
def confirm_uploaded_semantic_review(
    dataset_files: list[
        UploadFile
    ] = File(
        ...,
    ),

    workflow_id: str = Form(
        ...,
        min_length=1,
    ),

    semantic_decisions_json: str = Form(
        ...,
    ),

    confirmed_issue_ids_json: str = Form(
        ...,
    ),

    approved_semantic_choices_json: str = Form(
        default="[]",
    ),

    manual_resolutions_json: (
        str
        | None
    ) = Form(
        default=None,
    ),

    approved_action_ids_json: (
        str
        | None
    ) = Form(
        default=None,
    ),
) -> SemanticReviewConfirmationResponse:
    """
    Final human confirmation of semantic review.

    Security / integrity:

    - semantic review must already have run;
    - deterministic cleaning cannot be bypassed;
    - semantic decisions are rebuilt and Python-validated;
    - semantic cleaning plan is rebuilt server-side;
    - approved semantic choices are executed again server-side;
    - the browser cannot submit an execution result;
    - ABSTAIN and FLAG_FOR_REVIEW need explicit analyst notes;
    - CLEAN becomes PASSED only when every semantic decision
      is fully resolved.
    """

    try:
        _require_semantic_review_precondition(
            workflow_id=
                workflow_id
        )


        (
            deterministic_frames,
            validated_decisions,
            semantic_plan,
        ) = (
            _rebuild_semantic_cleaning_context(
                dataset_files=
                    dataset_files,

                workflow_id=
                    workflow_id,

                approved_action_ids_json=
                    approved_action_ids_json,

                semantic_decisions_json=
                    semantic_decisions_json,
            )
        )


        session = (
            get_preparation_session(
                workflow_id
            )
        )


        dataset_ids = list(
            session
            .selected_analysis_dataset_ids
        )


        choices = (
            _parse_semantic_choices(
                approved_semantic_choices_json
            )
        )


        (
            semantic_frames,
            execution,
        ) = (
            execute_semantic_cleaning_plan(
                plan=
                    semantic_plan,

                dataset_frames=
                    deterministic_frames,

                approved_choices=
                    choices,
            )
        )


        confirmed_issue_ids = (
            _parse_confirmed_issue_ids(
                confirmed_issue_ids_json
            )
        )


        manual_resolutions = (
            _parse_manual_resolutions(
                manual_resolutions_json
            )
        )


        try:
            confirmation = (
                require_semantic_confirmation(
                    decisions=
                        validated_decisions,

                    plan=
                        semantic_plan,

                    execution=
                        execution,

                    confirmed_issue_ids=
                        confirmed_issue_ids,

                    manual_resolutions=
                        manual_resolutions,
                )
            )


        except SemanticConfirmationBlockedError as error:
            _record_semantic_confirmation_blocked(
                workflow_id=
                    workflow_id,

                dataset_ids=
                    dataset_ids,

                confirmation=
                    error.report,
            )


            raise HTTPException(
                status_code=409,

                detail={
                    "error": (
                        "semantic_confirmation_incomplete"
                    ),

                    "message": (
                        "Semantic review still contains "
                        "unresolved analyst decisions."
                    ),

                    "workflow_id": (
                        workflow_id
                    ),

                    "confirmation": (
                        error.report.model_dump(
                            mode="json"
                        )
                    ),
                },
            ) from error


        # ====================================================
        # MATERIALIZE FINAL CONFIRMED SEMANTIC STATE
        #
        # Trust boundary:
        #
        #   semantic execution
        #       ↓
        #   Python confirmation
        #       ↓
        #   Artifact Store
        #       ↓
        #   CLEAN = PASSED
        #
        # The PreparationSession must never mark CLEAN as
        # completed before the exact confirmed DataFrames
        # exist in the server-owned Artifact Store.
        # ====================================================

        materialize_semantic_cleaning_artifacts(
            workflow_id=
                workflow_id,

            deterministic_frames=
                deterministic_frames,

            derived_frames=
                semantic_frames,

            semantic_plan=
                semantic_plan,

            execution=
                execution,
        )


        _record_semantic_confirmation_passed(
            workflow_id=
                workflow_id,

            dataset_ids=
                dataset_ids,

            confirmation=
                confirmation,

            execution=
                execution,
        )


        return (
            SemanticReviewConfirmationResponse(
                status=
                    "confirmed",

                confirmation=
                    confirmation,

                plan=
                    semantic_plan,

                execution=
                    execution,
            )
        )


    except PreparationSessionNotFoundError as error:
        raise HTTPException(
            status_code=404,

            detail={
                "error": (
                    "preparation_session_not_found"
                ),

                "message": str(
                    error
                ),

                "workflow_id": (
                    workflow_id
                ),
            },
        ) from error


    except HTTPException:
        raise


    except (
        ValueError,
        KeyError,
    ) as error:
        raise HTTPException(
            status_code=422,
            detail=str(
                error
            ),
        ) from error


    except TypeError as error:
        raise HTTPException(
            status_code=500,

            detail=(
                "Semantic review confirmation "
                "received invalid internal state: "
                f"{error}"
            ),
        ) from error


    except RuntimeError as error:
        raise HTTPException(
            status_code=500,

            detail=(
                "Semantic review confirmation "
                "failed: "
                f"{error}"
            ),
        ) from error