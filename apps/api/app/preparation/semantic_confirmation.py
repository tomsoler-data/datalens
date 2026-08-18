from __future__ import annotations

from typing import (
    Dict,
    List,
    Optional,
    Set,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.preparation.semantic_cleaning import (
    SemanticCleaningActionStatus,
    SemanticCleaningExecutionResult,
    SemanticCleaningPlan,
)

from app.preparation.semantic_review import (
    SemanticVerdict,
    ValidatedSemanticDecision,
)


# ============================================================
# VERSION
# ============================================================


SEMANTIC_CONFIRMATION_RULE_VERSION = (
    "semantic_confirmation_v0.2"
)


# ============================================================
# STRICT MODEL
# ============================================================


class StrictSemanticConfirmationModel(
    BaseModel,
):
    model_config = ConfigDict(
        extra="forbid"
    )


# ============================================================
# OPTIONAL ANALYST NOTE
# ============================================================


class SemanticManualResolution(
    StrictSemanticConfirmationModel,
):
    """
    Optional analyst note attached to a confirmed semantic
    decision.

    v0.2:

    A note is no longer required to resolve ABSTAIN or
    FLAG_FOR_REVIEW.

    Human confirmation is represented by confirmed_issue_ids.

    This model remains available so an analyst may document
    additional context when useful.
    """

    issue_id: str

    note: str = Field(
        min_length=3,
        max_length=1000,
    )


# ============================================================
# REPORT
# ============================================================


class SemanticConfirmationReport(
    StrictSemanticConfirmationModel,
):
    confirmed: bool

    decision_count: int

    confirmed_issue_count: int

    manual_resolution_count: int

    merge_action_count: int

    applied_merge_action_count: int

    skipped_merge_action_count: int

    confirmed_issue_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    manually_resolved_issue_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    unresolved_issue_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    unresolved_reasons: List[
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
        SEMANTIC_CONFIRMATION_RULE_VERSION
    )


# ============================================================
# ERROR
# ============================================================


class SemanticConfirmationBlockedError(
    RuntimeError,
):
    def __init__(
        self,
        *,
        report: SemanticConfirmationReport,
    ) -> None:
        self.report = (
            report
        )

        super().__init__(
            (
                "Semantic review confirmation "
                "is incomplete."
            )
        )


# ============================================================
# NORMALIZATION
# ============================================================


def _normalize_issue_ids(
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
            raw_value.strip()
        )


        if not value:
            raise ValueError(
                (
                    "Semantic confirmation issue_id "
                    "cannot be empty."
                )
            )


        if value in seen:
            continue


        seen.add(
            value
        )

        output.append(
            value
        )


    return output


def _normalize_manual_resolutions(
    values: Optional[
        List[
            SemanticManualResolution
        ]
    ],
) -> Dict[
    str,
    SemanticManualResolution,
]:
    """
    Normalize optional analyst notes.

    Notes are documentation only in v0.2.
    They are never required for semantic confirmation.
    """

    output: Dict[
        str,
        SemanticManualResolution,
    ] = {}


    for resolution in (
        values
        or
        []
    ):
        issue_id = (
            resolution
            .issue_id
            .strip()
        )


        if not issue_id:
            raise ValueError(
                (
                    "Semantic manual resolution "
                    "issue_id cannot be empty."
                )
            )


        if issue_id in output:
            raise ValueError(
                (
                    "Duplicate semantic manual "
                    "resolution for issue_id: "
                    f"{issue_id}"
                )
            )


        note = (
            resolution
            .note
            .strip()
        )


        if len(
            note
        ) < 3:
            raise ValueError(
                (
                    "Semantic analyst note must "
                    "contain at least 3 characters "
                    "when one is provided."
                )
            )


        output[
            issue_id
        ] = (
            resolution.model_copy(
                update={
                    "issue_id":
                        issue_id,

                    "note":
                        note,
                }
            )
        )


    return output


# ============================================================
# DECISION INDEX
# ============================================================


def _decision_index(
    decisions: List[
        ValidatedSemanticDecision
    ],
) -> Dict[
    str,
    ValidatedSemanticDecision,
]:
    output: Dict[
        str,
        ValidatedSemanticDecision,
    ] = {}


    for decision in decisions:
        issue_id = (
            decision
            .issue_id
            .strip()
        )


        if not issue_id:
            raise ValueError(
                (
                    "Validated semantic decision "
                    "contains an empty issue_id."
                )
            )


        if issue_id in output:
            raise ValueError(
                (
                    "Duplicate validated semantic "
                    "decision issue_id: "
                    f"{issue_id}"
                )
            )


        if (
            decision.python_validated
            is not True
        ):
            raise ValueError(
                (
                    "Semantic confirmation received "
                    "a decision that was not "
                    "Python-validated: "
                    f"{issue_id}"
                )
            )


        output[
            issue_id
        ] = decision


    return output


# ============================================================
# PLAN CONSISTENCY
# ============================================================


def _validate_semantic_plan(
    *,
    decisions_by_issue: Dict[
        str,
        ValidatedSemanticDecision,
    ],

    plan: SemanticCleaningPlan,
) -> Dict[
    str,
    str,
]:
    """
    Return:

        issue_id -> action_id

    Every MERGE_VALUES decision must correspond to exactly
    one semantic cleaning action.

    No cleaning action may reference a non-MERGE_VALUES
    decision.
    """

    expected_merge_issue_ids = {
        issue_id

        for (
            issue_id,
            decision,
        ) in (
            decisions_by_issue
            .items()
        )

        if (
            decision.verdict
            ==
            SemanticVerdict.MERGE_VALUES
        )
    }


    action_by_issue: Dict[
        str,
        str,
    ] = {}


    seen_action_ids: Set[
        str
    ] = set()


    for action in (
        plan.actions
    ):
        if (
            action.action_id
            in seen_action_ids
        ):
            raise ValueError(
                (
                    "Duplicate semantic cleaning "
                    "action_id in confirmation plan: "
                    f"{action.action_id}"
                )
            )


        seen_action_ids.add(
            action.action_id
        )


        if (
            action.issue_id
            in action_by_issue
        ):
            raise ValueError(
                (
                    "Multiple semantic cleaning "
                    "actions reference the same "
                    "issue_id: "
                    f"{action.issue_id}"
                )
            )


        action_by_issue[
            action.issue_id
        ] = (
            action.action_id
        )


    actual_merge_issue_ids = set(
        action_by_issue
    )


    if (
        actual_merge_issue_ids
        !=
        expected_merge_issue_ids
    ):
        missing = sorted(
            expected_merge_issue_ids
            -
            actual_merge_issue_ids
        )


        unexpected = sorted(
            actual_merge_issue_ids
            -
            expected_merge_issue_ids
        )


        raise ValueError(
            (
                "Semantic cleaning plan does not "
                "match the validated MERGE_VALUES "
                "decisions. "
                f"missing={missing}; "
                f"unexpected={unexpected}"
            )
        )


    if (
        plan.action_count
        !=
        len(
            plan.actions
        )
    ):
        raise ValueError(
            (
                "Semantic cleaning plan action_count "
                "does not match actions length."
            )
        )


    return (
        action_by_issue
    )


# ============================================================
# EXECUTION INDEX
# ============================================================


def _execution_status_by_action(
    *,
    plan: SemanticCleaningPlan,

    execution: Optional[
        SemanticCleaningExecutionResult
    ],
) -> Dict[
    str,
    SemanticCleaningActionStatus,
]:
    if (
        plan.action_count
        ==
        0
        and
        execution is None
    ):
        return {}


    if (
        execution is None
    ):
        return {}


    result_map: Dict[
        str,
        SemanticCleaningActionStatus,
    ] = {}


    plan_action_ids = {
        action.action_id

        for action
        in plan.actions
    }


    for result in (
        execution.action_results
    ):
        if (
            result.action_id
            in result_map
        ):
            raise ValueError(
                (
                    "Duplicate semantic execution "
                    "result for action_id: "
                    f"{result.action_id}"
                )
            )


        if (
            result.action_id
            not in
            plan_action_ids
        ):
            raise ValueError(
                (
                    "Semantic execution contains "
                    "an action that is absent from "
                    "the validated plan: "
                    f"{result.action_id}"
                )
            )


        result_map[
            result.action_id
        ] = (
            result.status
        )


    return (
        result_map
    )


# ============================================================
# EVALUATION
# ============================================================


def evaluate_semantic_confirmation(
    *,
    decisions: List[
        ValidatedSemanticDecision
    ],

    plan: SemanticCleaningPlan,

    execution: Optional[
        SemanticCleaningExecutionResult
    ],

    confirmed_issue_ids: List[
        str
    ],

    manual_resolutions: Optional[
        List[
            SemanticManualResolution
        ]
    ] = None,
) -> SemanticConfirmationReport:
    """
    Evaluate whether human semantic review is fully resolved.

    v0.2 rules:

    Every decision:
        requires explicit human confirmation through
        confirmed_issue_ids.

    NO_CHANGE:
        confirmation is sufficient.

    KEEP_SEPARATE:
        confirmation is sufficient.

    CONTEXTUALIZE:
        confirmation is sufficient.

    ABSTAIN:
        confirmation is sufficient.
        An analyst note is optional.

    FLAG_FOR_REVIEW:
        confirmation is sufficient.
        An analyst note is optional.

    MERGE_VALUES:
        confirmation is required AND the corresponding
        semantic cleaning action must have status APPLIED.

    The function:
        - never mutates a DataFrame;
        - never changes Preparation Session state;
        - never treats LLM output alone as human approval.
    """

    decisions_by_issue = (
        _decision_index(
            decisions
        )
    )


    normalized_confirmed = (
        _normalize_issue_ids(
            confirmed_issue_ids
        )
    )


    confirmed_set = set(
        normalized_confirmed
    )


    decision_issue_ids = set(
        decisions_by_issue
    )


    unknown_confirmations = sorted(
        confirmed_set
        -
        decision_issue_ids
    )


    if (
        unknown_confirmations
    ):
        raise ValueError(
            (
                "Semantic confirmation references "
                "unknown issue IDs: "
                f"{unknown_confirmations}"
            )
        )


    manual_by_issue = (
        _normalize_manual_resolutions(
            manual_resolutions
        )
    )


    unknown_manual_ids = sorted(
        set(
            manual_by_issue
        )
        -
        decision_issue_ids
    )


    if (
        unknown_manual_ids
    ):
        raise ValueError(
            (
                "Semantic analyst notes reference "
                "unknown issue IDs: "
                f"{unknown_manual_ids}"
            )
        )


    action_by_issue = (
        _validate_semantic_plan(
            decisions_by_issue=
                decisions_by_issue,

            plan=
                plan,
        )
    )


    execution_status = (
        _execution_status_by_action(
            plan=
                plan,

            execution=
                execution,
        )
    )


    unresolved_issue_ids: List[
        str
    ] = []


    unresolved_reasons: List[
        str
    ] = []


    for (
        issue_id,
        decision,
    ) in (
        decisions_by_issue
        .items()
    ):
        # ====================================================
        # EXPLICIT HUMAN CONFIRMATION
        # ====================================================

        if (
            issue_id
            not in
            confirmed_set
        ):
            unresolved_issue_ids.append(
                issue_id
            )


            unresolved_reasons.append(
                (
                    f"{issue_id}: analyst "
                    "confirmation is missing."
                )
            )


            continue


        # ====================================================
        # MERGE MUST ACTUALLY HAVE BEEN APPLIED
        # ====================================================

        if (
            decision.verdict
            ==
            SemanticVerdict.MERGE_VALUES
        ):
            action_id = (
                action_by_issue.get(
                    issue_id
                )
            )


            if (
                action_id is None
            ):
                unresolved_issue_ids.append(
                    issue_id
                )


                unresolved_reasons.append(
                    (
                        f"{issue_id}: validated "
                        "semantic merge has no "
                        "cleaning action."
                    )
                )


                continue


            status = (
                execution_status.get(
                    action_id
                )
            )


            if (
                status
                !=
                SemanticCleaningActionStatus.APPLIED
            ):
                unresolved_issue_ids.append(
                    issue_id
                )


                unresolved_reasons.append(
                    (
                        f"{issue_id}: semantic "
                        "merge was not applied."
                    )
                )


                continue


        # ====================================================
        # ABSTAIN / FLAG_FOR_REVIEW
        #
        # v0.2:
        # explicit analyst confirmation is sufficient.
        #
        # Optional manual notes are retained only for
        # documentation/provenance.
        # ====================================================


    # ========================================================
    # EXECUTION COUNTS
    # ========================================================

    applied_merge_action_count = sum(
        status
        ==
        SemanticCleaningActionStatus.APPLIED

        for status
        in execution_status.values()
    )


    skipped_merge_action_count = sum(
        status
        ==
        SemanticCleaningActionStatus.SKIPPED

        for status
        in execution_status.values()
    )


    confirmed = (
        len(
            unresolved_issue_ids
        )
        ==
        0
    )


    manually_resolved_issue_ids = sorted(
        manual_by_issue
    )


    return (
        SemanticConfirmationReport(
            confirmed=
                confirmed,

            decision_count=
                len(
                    decisions
                ),

            confirmed_issue_count=
                len(
                    confirmed_set
                ),

            manual_resolution_count=
                len(
                    manual_by_issue
                ),

            merge_action_count=
                plan.action_count,

            applied_merge_action_count=
                applied_merge_action_count,

            skipped_merge_action_count=
                skipped_merge_action_count,

            confirmed_issue_ids=
                sorted(
                    confirmed_set
                ),

            manually_resolved_issue_ids=
                manually_resolved_issue_ids,

            unresolved_issue_ids=
                sorted(
                    set(
                        unresolved_issue_ids
                    )
                ),

            unresolved_reasons=
                unresolved_reasons,

            notes=[
                (
                    "Semantic confirmation is "
                    "evaluated only from "
                    "Python-validated decisions."
                ),

                (
                    "Every semantic decision still "
                    "requires explicit analyst "
                    "confirmation."
                ),

                (
                    "ABSTAIN and FLAG_FOR_REVIEW "
                    "may be confirmed without "
                    "writing a manual note."
                ),

                (
                    "Analyst notes remain optional "
                    "documentation attached to "
                    "confirmed decisions."
                ),

                (
                    "MERGE_VALUES requires an "
                    "actually applied semantic "
                    "cleaning action."
                ),

                (
                    "This policy performs no "
                    "DataFrame mutation."
                ),
            ],

            rule_version=
                SEMANTIC_CONFIRMATION_RULE_VERSION,
        )
    )


# ============================================================
# REQUIRE
# ============================================================


def require_semantic_confirmation(
    *,
    decisions: List[
        ValidatedSemanticDecision
    ],

    plan: SemanticCleaningPlan,

    execution: Optional[
        SemanticCleaningExecutionResult
    ],

    confirmed_issue_ids: List[
        str
    ],

    manual_resolutions: Optional[
        List[
            SemanticManualResolution
        ]
    ] = None,
) -> SemanticConfirmationReport:
    report = (
        evaluate_semantic_confirmation(
            decisions=
                decisions,

            plan=
                plan,

            execution=
                execution,

            confirmed_issue_ids=
                confirmed_issue_ids,

            manual_resolutions=
                manual_resolutions,
        )
    )


    if not (
        report.confirmed
    ):
        raise (
            SemanticConfirmationBlockedError(
                report=
                    report
            )
        )


    return (
        report
    )