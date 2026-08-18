from __future__ import annotations


from app.preparation.data_quality import (
    QualityIssueKind,
)

from app.preparation.semantic_cleaning import (
    SemanticCleaningAction,
    SemanticCleaningActionResult,
    SemanticCleaningActionStatus,
    SemanticCleaningExecutionResult,
    SemanticCleaningPlan,
)

from app.preparation.semantic_confirmation import (
    SEMANTIC_CONFIRMATION_RULE_VERSION,
    SemanticConfirmationBlockedError,
    SemanticManualResolution,
    evaluate_semantic_confirmation,
    require_semantic_confirmation,
)

from app.preparation.semantic_review import (
    SemanticVerdict,
    ValidatedSemanticDecision,
)


# ============================================================
# HELPERS
# ============================================================


def decision(
    *,
    issue_id: str,

    verdict: SemanticVerdict,

    source_values: list[
        str
    ] | None = None,

    canonical_value: str | None = None,
) -> ValidatedSemanticDecision:
    return (
        ValidatedSemanticDecision(
            issue_id=
                issue_id,

            dataset_id=
                "dataset:0001",

            dataset_filename=
                "orders.csv",

            column=
                "category",

            kind=
                QualityIssueKind
                .POSSIBLE_SEMANTIC_ALIASES,

            verdict=
                verdict,

            confidence=
                0.95,

            rationale=
                "Synthetic confirmation test.",

            source_values=
                source_values
                or
                [],

            canonical_value=
                canonical_value,

            user_message=
                "Review this decision.",

            python_validated=
                True,

            executable=
                False,

            requires_user_confirmation=
                True,

            validation_notes=[],
        )
    )


def merge_plan(
) -> SemanticCleaningPlan:
    return (
        SemanticCleaningPlan(
            status=
                "ready",

            action_count=
                1,

            actions=[
                SemanticCleaningAction(
                    action_id=
                        "semantic:dataset:0001:test",

                    issue_id=
                        "issue:merge",

                    dataset_id=
                        "dataset:0001",

                    dataset_filename=
                        "orders.csv",

                    column=
                        "category",

                    source_values=[
                        "Premium",
                        "PREMIUM",
                    ],

                    suggested_canonical_value=
                        "Premium",

                    allowed_canonical_values=[
                        "Premium",
                        "PREMIUM",
                    ],

                    confidence=
                        0.98,

                    rationale=
                        "Equivalent aliases.",

                    requires_user_confirmation=
                        True,

                    python_validated=
                        True,
                )
            ],

            notes=[],
        )
    )


def empty_plan(
) -> SemanticCleaningPlan:
    return (
        SemanticCleaningPlan(
            status=
                "ready",

            action_count=
                0,

            actions=[],

            notes=[],
        )
    )


def merge_execution(
    *,
    status:
        SemanticCleaningActionStatus,
) -> SemanticCleaningExecutionResult:
    return (
        SemanticCleaningExecutionResult(
            status=
                "completed",

            dataset_count=
                1,

            applied_action_count=(
                1

                if (
                    status
                    ==
                    SemanticCleaningActionStatus.APPLIED
                )

                else
                0
            ),

            skipped_action_count=(
                1

                if (
                    status
                    ==
                    SemanticCleaningActionStatus.SKIPPED
                )

                else
                0
            ),

            changed_cell_count=(
                2

                if (
                    status
                    ==
                    SemanticCleaningActionStatus.APPLIED
                )

                else
                0
            ),

            action_results=[
                SemanticCleaningActionResult(
                    action_id=
                        "semantic:dataset:0001:test",

                    status=
                        status,

                    dataset_id=
                        "dataset:0001",

                    column=
                        "category",

                    source_values=[
                        "Premium",
                        "PREMIUM",
                    ],

                    canonical_value=(
                        "Premium"

                        if (
                            status
                            ==
                            SemanticCleaningActionStatus
                            .APPLIED
                        )

                        else
                        None
                    ),

                    affected_rows_actual=(
                        2

                        if (
                            status
                            ==
                            SemanticCleaningActionStatus
                            .APPLIED
                        )

                        else
                        0
                    ),

                    details={},
                )
            ],

            provenance=[],

            notes=[],
        )
    )


# ============================================================
# NO CHANGE
# ============================================================


def test_no_change_confirmation(
) -> None:
    decisions = [
        decision(
            issue_id=
                "issue:no-change",

            verdict=
                SemanticVerdict.NO_CHANGE,
        )
    ]


    report = (
        require_semantic_confirmation(
            decisions=
                decisions,

            plan=
                empty_plan(),

            execution=
                None,

            confirmed_issue_ids=[
                "issue:no-change"
            ],
        )
    )


    print(
        "\n=== NO CHANGE ==="
    )

    print(
        f"Confirmed: "
        f"{report.confirmed}"
    )


    assert (
        report.confirmed
        is True
    )


# ============================================================
# KEEP SEPARATE / CONTEXTUALIZE
# ============================================================


def test_non_mutating_decisions(
) -> None:
    decisions = [
        decision(
            issue_id=
                "issue:keep",

            verdict=
                SemanticVerdict.KEEP_SEPARATE,
        ),

        decision(
            issue_id=
                "issue:context",

            verdict=
                SemanticVerdict.CONTEXTUALIZE,
        ),
    ]


    report = (
        require_semantic_confirmation(
            decisions=
                decisions,

            plan=
                empty_plan(),

            execution=
                None,

            confirmed_issue_ids=[
                "issue:keep",
                "issue:context",
            ],
        )
    )


    print(
        "\n=== NON-MUTATING DECISIONS ==="
    )

    print(
        f"Confirmed: "
        f"{report.confirmed}"
    )


    assert (
        report.confirmed
        is True
    )


# ============================================================
# MERGE APPLIED
# ============================================================


def test_merge_applied(
) -> None:
    decisions = [
        decision(
            issue_id=
                "issue:merge",

            verdict=
                SemanticVerdict.MERGE_VALUES,

            source_values=[
                "Premium",
                "PREMIUM",
            ],

            canonical_value=
                "Premium",
        )
    ]


    report = (
        require_semantic_confirmation(
            decisions=
                decisions,

            plan=
                merge_plan(),

            execution=
                merge_execution(
                    status=
                        SemanticCleaningActionStatus
                        .APPLIED
                ),

            confirmed_issue_ids=[
                "issue:merge"
            ],
        )
    )


    print(
        "\n=== MERGE APPLIED ==="
    )

    print(
        (
            "Applied merge actions: "
            f"{report.applied_merge_action_count}"
        )
    )

    print(
        f"Confirmed: "
        f"{report.confirmed}"
    )


    assert (
        report.confirmed
        is True
    )

    assert (
        report.applied_merge_action_count
        ==
        1
    )


# ============================================================
# MERGE SKIPPED
# ============================================================


def test_merge_skipped_is_blocked(
) -> None:
    decisions = [
        decision(
            issue_id=
                "issue:merge",

            verdict=
                SemanticVerdict.MERGE_VALUES,

            source_values=[
                "Premium",
                "PREMIUM",
            ],

            canonical_value=
                "Premium",
        )
    ]


    report = (
        evaluate_semantic_confirmation(
            decisions=
                decisions,

            plan=
                merge_plan(),

            execution=
                merge_execution(
                    status=
                        SemanticCleaningActionStatus
                        .SKIPPED
                ),

            confirmed_issue_ids=[
                "issue:merge"
            ],
        )
    )


    print(
        "\n=== MERGE SKIPPED ==="
    )

    print(
        f"Confirmed: "
        f"{report.confirmed}"
    )

    print(
        (
            "Unresolved: "
            f"{report.unresolved_issue_ids}"
        )
    )


    assert (
        report.confirmed
        is False
    )

    assert (
        "issue:merge"
        in
        report.unresolved_issue_ids
    )


    try:
        require_semantic_confirmation(
            decisions=
                decisions,

            plan=
                merge_plan(),

            execution=
                merge_execution(
                    status=
                        SemanticCleaningActionStatus
                        .SKIPPED
                ),

            confirmed_issue_ids=[
                "issue:merge"
            ],
        )

    except SemanticConfirmationBlockedError:
        pass

    else:
        raise AssertionError(
            (
                "Expected "
                "SemanticConfirmationBlockedError."
            )
        )


# ============================================================
# ABSTAIN — NO TEXT REQUIRED
# ============================================================


def test_abstain_without_note_can_be_confirmed(
) -> None:
    decisions = [
        decision(
            issue_id=
                "issue:abstain",

            verdict=
                SemanticVerdict.ABSTAIN,
        )
    ]


    report = (
        require_semantic_confirmation(
            decisions=
                decisions,

            plan=
                empty_plan(),

            execution=
                None,

            confirmed_issue_ids=[
                "issue:abstain"
            ],

            manual_resolutions=[],
        )
    )


    print(
        "\n=== ABSTAIN WITHOUT NOTE ==="
    )

    print(
        f"Confirmed: "
        f"{report.confirmed}"
    )

    print(
        (
            "Optional analyst notes: "
            f"{report.manual_resolution_count}"
        )
    )


    assert (
        report.confirmed
        is True
    )

    assert (
        report.manual_resolution_count
        ==
        0
    )


# ============================================================
# FLAG FOR REVIEW — NO TEXT REQUIRED
# ============================================================


def test_flag_without_note_can_be_confirmed(
) -> None:
    decisions = [
        decision(
            issue_id=
                "issue:flag",

            verdict=
                SemanticVerdict.FLAG_FOR_REVIEW,
        )
    ]


    report = (
        require_semantic_confirmation(
            decisions=
                decisions,

            plan=
                empty_plan(),

            execution=
                None,

            confirmed_issue_ids=[
                "issue:flag"
            ],

            manual_resolutions=[],
        )
    )


    print(
        "\n=== FLAG WITHOUT NOTE ==="
    )

    print(
        f"Confirmed: "
        f"{report.confirmed}"
    )


    assert (
        report.confirmed
        is True
    )


# ============================================================
# OPTIONAL ANALYST NOTE
# ============================================================


def test_optional_note_is_preserved(
) -> None:
    decisions = [
        decision(
            issue_id=
                "issue:abstain",

            verdict=
                SemanticVerdict.ABSTAIN,
        )
    ]


    report = (
        require_semantic_confirmation(
            decisions=
                decisions,

            plan=
                empty_plan(),

            execution=
                None,

            confirmed_issue_ids=[
                "issue:abstain"
            ],

            manual_resolutions=[
                SemanticManualResolution(
                    issue_id=
                        "issue:abstain",

                    note=(
                        "Conservé après examen "
                        "analyste."
                    ),
                )
            ],
        )
    )


    print(
        "\n=== OPTIONAL ANALYST NOTE ==="
    )

    print(
        f"Confirmed: "
        f"{report.confirmed}"
    )

    print(
        (
            "Notes recorded: "
            f"{report.manual_resolution_count}"
        )
    )


    assert (
        report.confirmed
        is True
    )

    assert (
        report.manual_resolution_count
        ==
        1
    )

    assert (
        report.manually_resolved_issue_ids
        ==
        [
            "issue:abstain"
        ]
    )


# ============================================================
# MISSING CONFIRMATION
# ============================================================


def test_missing_confirmation_is_blocked(
) -> None:
    decisions = [
        decision(
            issue_id=
                "issue:no-change",

            verdict=
                SemanticVerdict.NO_CHANGE,
        )
    ]


    report = (
        evaluate_semantic_confirmation(
            decisions=
                decisions,

            plan=
                empty_plan(),

            execution=
                None,

            confirmed_issue_ids=[],
        )
    )


    print(
        "\n=== MISSING CONFIRMATION ==="
    )

    print(
        (
            "Unresolved: "
            f"{report.unresolved_issue_ids}"
        )
    )


    assert (
        report.confirmed
        is False
    )

    assert (
        report.unresolved_issue_ids
        ==
        [
            "issue:no-change"
        ]
    )


# ============================================================
# UNKNOWN ISSUE
# ============================================================


def test_unknown_confirmation_is_rejected(
) -> None:
    decisions = [
        decision(
            issue_id=
                "issue:known",

            verdict=
                SemanticVerdict.NO_CHANGE,
        )
    ]


    try:
        evaluate_semantic_confirmation(
            decisions=
                decisions,

            plan=
                empty_plan(),

            execution=
                None,

            confirmed_issue_ids=[
                "issue:unknown"
            ],
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            (
                "Expected ValueError for "
                "unknown confirmation issue."
            )
        )


# ============================================================
# PLAN MISMATCH
# ============================================================


def test_merge_plan_mismatch_is_rejected(
) -> None:
    decisions = [
        decision(
            issue_id=
                "issue:merge",

            verdict=
                SemanticVerdict.MERGE_VALUES,

            source_values=[
                "Premium",
                "PREMIUM",
            ],

            canonical_value=
                "Premium",
        )
    ]


    try:
        evaluate_semantic_confirmation(
            decisions=
                decisions,

            plan=
                empty_plan(),

            execution=
                None,

            confirmed_issue_ids=[
                "issue:merge"
            ],
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            (
                "Expected ValueError for semantic "
                "plan / decision mismatch."
            )
        )


# ============================================================
# VERSION
# ============================================================


def test_version(
) -> None:
    assert (
        SEMANTIC_CONFIRMATION_RULE_VERSION
        ==
        "semantic_confirmation_v0.2"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:
    print(
        "\n========================================"
    )

    print(
        "DataLens Semantic Confirmation v0.2"
    )

    print(
        "========================================"
    )


    test_no_change_confirmation()

    test_non_mutating_decisions()

    test_merge_applied()

    test_merge_skipped_is_blocked()

    test_abstain_without_note_can_be_confirmed()

    test_flag_without_note_can_be_confirmed()

    test_optional_note_is_preserved()

    test_missing_confirmation_is_blocked()

    test_unknown_confirmation_is_rejected()

    test_merge_plan_mismatch_is_rejected()

    test_version()


    print(
        "\n========================================"
    )

    print(
        "PASS - semantic confirmation v0.2"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()