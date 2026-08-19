from __future__ import annotations

import pandas as pd

import app.preparation.semantic_review as semantic_review

from app.preparation.data_quality import (
    CleaningOperation,
    CleaningProposal,
    DataQualityReport,
    IssueEvidence,
    QualityIssue,
    QualityIssueKind,
    QualitySeverity,
    build_data_quality_report,
)

from app.preparation.semantic_cleaning import (
    SemanticCleaningChoice,
    build_semantic_cleaning_plan,
    execute_semantic_cleaning_plan,
)


DATASET_ID = (
    "dataset:test"
)


# ============================================================
# FIXTURE
# ============================================================


def build_fixture() -> tuple[
    pd.DataFrame,
    dict,
]:
    dataframe = pd.DataFrame(
        {
            "row_id":
                list(
                    range(
                        1,
                        13,
                    )
                ),

            "category": [
                "Furniture",
                "furniture",
                "Furniture",
                "Electronics",
                "electronics",
                "Electronics",
                "Accessories",
                "Furniture",
                "electronics",
                "Accessories",
                "Furniture",
                "Electronics",
            ],

            "channel": [
                "Web",
                "web",
                "Web",
                "Store",
                "store",
                "Store",
                "Web",
                "store",
                "web",
                "Store",
                "Web",
                "Store",
            ],

            "amount": [
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
            ],
        }
    )

    record = {
        "dataset_id":
            DATASET_ID,

        "filename":
            "test.csv",

        "dataframe":
            dataframe,
    }

    return (
        dataframe,
        record,
    )


# ============================================================
# EXPLICIT QUALITY CONTRACT
# ============================================================


def alias_issue(
    *,
    issue_id: str,
    column: str,
    candidate_pairs: list[
        tuple[
            str,
            str,
        ]
    ],
) -> QualityIssue:
    """
    Build the exact deterministic contract expected by
    Semantic Review.

    This test deliberately does NOT ask the Quality detector
    to rediscover the aliases. Detection and semantic
    canonicalization are separate units and must be tested
    independently.
    """

    return QualityIssue(
        issue_id=
            issue_id,

        dataset_id=
            DATASET_ID,

        dataset_filename=
            "test.csv",

        column=
            column,

        kind=
            QualityIssueKind
            .POSSIBLE_SEMANTIC_ALIASES,

        severity=
            QualitySeverity
            .MODERATE,

        title=
            "Alias sémantiques possibles",

        explanation=(
            "Certaines modalités pourraient désigner "
            "le même concept."
        ),

        evidence=
            IssueEvidence(
                observed_count=
                    len(
                        candidate_pairs
                    ),

                affected_ratio=
                    1.0,

                examples=[
                    f"{left} / {right}"

                    for (
                        left,
                        right,
                    )
                    in candidate_pairs
                ],

                details={
                    "candidate_pairs":
                        [
                            [
                                left,
                                right,
                            ]

                            for (
                                left,
                                right,
                            )
                            in candidate_pairs
                        ],
                },
            ),

        proposal=
            CleaningProposal(
                operation=
                    CleaningOperation
                    .REVIEW_VALUES,

                automatic_safe=
                    False,

                description=(
                    "Soumettre les rapprochements "
                    "à la revue sémantique."
                ),

                requires_user_confirmation=
                    True,
            ),

        semantic_review_recommended=
            True,
    )


def alias_only_quality_report(
    record: dict,
) -> DataQualityReport:
    """
    Reuse the real report envelope, but replace its issues by
    explicit deterministic alias evidence.

    Why:
    - Data Quality owns alias detection heuristics;
    - Semantic Review owns interpretation/canonicalization;
    - this unit test targets only the second responsibility.
    """

    base_report = (
        build_data_quality_report(
            [
                record
            ]
        )
    )


    issues = [
        alias_issue(
            issue_id=
                "quality:category:aliases",

            column=
                "category",

            candidate_pairs=[
                (
                    "Furniture",
                    "furniture",
                ),
                (
                    "Electronics",
                    "electronics",
                ),
            ],
        ),

        alias_issue(
            issue_id=
                "quality:channel:aliases",

            column=
                "channel",

            candidate_pairs=[
                (
                    "Web",
                    "web",
                ),
                (
                    "Store",
                    "store",
                ),
            ],
        ),
    ]


    return base_report.model_copy(
        deep=True,

        update={
            "issues":
                issues,

            "issue_count":
                len(
                    issues
                ),

            "moderate_count":
                len(
                    issues
                ),

            "semantic_review_count":
                len(
                    issues
                ),
        },
    )


# ============================================================
# 1. ONE QUALITY ISSUE -> MULTIPLE SEMANTIC CANDIDATES
# ============================================================


def test_multiple_alias_groups_become_independent_candidates(
) -> None:
    (
        dataframe,
        record,
    ) = build_fixture()


    quality_report = (
        alias_only_quality_report(
            record
        )
    )


    candidates = (
        semantic_review
        .build_semantic_review_candidates(
            quality_report=
                quality_report,

            dataset_frames={
                DATASET_ID:
                    dataframe,
            },
        )
    )


    category_candidates = [
        candidate

        for candidate
        in candidates

        if (
            candidate.column
            ==
            "category"
        )
    ]


    channel_candidates = [
        candidate

        for candidate
        in candidates

        if (
            candidate.column
            ==
            "channel"
        )
    ]


    assert (
        len(
            category_candidates
        )
        ==
        2
    )


    assert (
        len(
            channel_candidates
        )
        ==
        2
    )


    assert all(
        len(
            candidate
            .candidate_groups
        )
        ==
        1

        for candidate
        in candidates
    )


    assert (
        len(
            {
                candidate.issue_id

                for candidate
                in candidates
            }
        )
        ==
        len(
            candidates
        )
    )


    assert (
        category_candidates[
            0
        ].issue_id
        ==
        "quality:category:aliases"
    )


    assert (
        channel_candidates[
            0
        ].issue_id
        ==
        "quality:channel:aliases"
    )


    assert (
        category_candidates[
            1
        ].issue_id
        .startswith(
            "quality:category:aliases:alias:"
        )
    )


    assert (
        channel_candidates[
            1
        ].issue_id
        .startswith(
            "quality:channel:aliases:alias:"
        )
    )


    print(
        "Multiple alias groups become independent semantic candidates: PASS"
    )


# ============================================================
# 2. STRICT ALIASES DO NOT REQUIRE THE LLM
# ============================================================


def test_strict_aliases_do_not_depend_on_llm(
) -> None:
    (
        dataframe,
        record,
    ) = build_fixture()


    quality_report = (
        alias_only_quality_report(
            record
        )
    )


    original = (
        semantic_review
        ._ollama_chat_one
    )


    try:
        def forbidden_llm_call(
            **kwargs,
        ):
            raise AssertionError(
                "Strict alias canonicalization "
                "should not call the LLM."
            )


        semantic_review._ollama_chat_one = (
            forbidden_llm_call
        )


        report = (
            semantic_review
            .review_quality_semantics(
                quality_report=
                    quality_report,

                dataset_frames={
                    DATASET_ID:
                        dataframe,
                },
            )
        )


    finally:
        semantic_review._ollama_chat_one = (
            original
        )


    assert (
        report.decision_count
        ==
        4
    )


    assert (
        report.merge_proposal_count
        ==
        4
    )


    assert (
        report.abstention_count
        ==
        0
    )


    assert all(
        decision.verdict.value
        ==
        "merge_values"

        for decision
        in report.decisions
    )


    assert all(
        decision.python_validated
        is True

        for decision
        in report.decisions
    )


    assert all(
        decision.executable
        is False

        for decision
        in report.decisions
    )


    assert all(
        decision.requires_user_confirmation
        is True

        for decision
        in report.decisions
    )


    print(
        "Strict case/whitespace aliases are proposed by Python without LLM dependency: PASS"
    )


# ============================================================
# 3. CANONICAL VALUES ARE EXISTING VALUES
# ============================================================


def test_canonical_value_is_existing_and_deterministic(
) -> None:
    (
        dataframe,
        record,
    ) = build_fixture()


    quality_report = (
        alias_only_quality_report(
            record
        )
    )


    report = (
        semantic_review
        .review_quality_semantics(
            quality_report=
                quality_report,

            dataset_frames={
                DATASET_ID:
                    dataframe,
            },
        )
    )


    canonical_by_group = {
        frozenset(
            decision.source_values
        ):
            decision.canonical_value

        for decision
        in report.decisions
    }


    assert (
        canonical_by_group[
            frozenset(
                {
                    "Furniture",
                    "furniture",
                }
            )
        ]
        ==
        "Furniture"
    )


    assert (
        canonical_by_group[
            frozenset(
                {
                    "Electronics",
                    "electronics",
                }
            )
        ]
        ==
        "Electronics"
    )


    assert (
        canonical_by_group[
            frozenset(
                {
                    "Web",
                    "web",
                }
            )
        ]
        ==
        "Web"
    )


    assert (
        canonical_by_group[
            frozenset(
                {
                    "Store",
                    "store",
                }
            )
        ]
        ==
        "Store"
    )


    all_observed = {
        str(
            value
        )

        for column in [
            "category",
            "channel",
        ]

        for value in (
            dataframe[
                column
            ].dropna()
        )
    }


    assert all(
        decision.canonical_value
        in
        all_observed

        for decision
        in report.decisions
    )


    print(
        "Canonical values are existing observed values selected deterministically: PASS"
    )


# ============================================================
# 4. NO APPROVAL -> NO MUTATION
# ============================================================


def test_no_user_approval_means_no_mutation(
) -> None:
    (
        dataframe,
        record,
    ) = build_fixture()


    quality_report = (
        alias_only_quality_report(
            record
        )
    )


    review = (
        semantic_review
        .review_quality_semantics(
            quality_report=
                quality_report,

            dataset_frames={
                DATASET_ID:
                    dataframe,
            },
        )
    )


    plan = (
        build_semantic_cleaning_plan(
            review.decisions
        )
    )


    (
        derived,
        execution,
    ) = (
        execute_semantic_cleaning_plan(
            plan=
                plan,

            dataset_frames={
                DATASET_ID:
                    dataframe,
            },

            approved_choices=
                [],
        )
    )


    assert (
        plan.action_count
        ==
        4
    )


    assert (
        execution.applied_action_count
        ==
        0
    )


    assert (
        execution.skipped_action_count
        ==
        4
    )


    assert (
        derived[
            DATASET_ID
        ].equals(
            dataframe
        )
    )


    print(
        "Strict canonicalization remains non-executable without user approval: PASS"
    )


# ============================================================
# 5. APPROVAL -> ALL GROUPS MATERIALIZED
# ============================================================


def test_approved_alias_groups_are_all_materialized(
) -> None:
    (
        dataframe,
        record,
    ) = build_fixture()


    source_copy = (
        dataframe.copy(
            deep=True
        )
    )


    quality_report = (
        alias_only_quality_report(
            record
        )
    )


    review = (
        semantic_review
        .review_quality_semantics(
            quality_report=
                quality_report,

            dataset_frames={
                DATASET_ID:
                    dataframe,
            },
        )
    )


    plan = (
        build_semantic_cleaning_plan(
            review.decisions
        )
    )


    assert (
        plan.action_count
        ==
        4
    )


    choices = [
        SemanticCleaningChoice(
            action_id=
                action.action_id,

            canonical_value=
                action
                .suggested_canonical_value,
        )

        for action
        in plan.actions
    ]


    (
        derived,
        execution,
    ) = (
        execute_semantic_cleaning_plan(
            plan=
                plan,

            dataset_frames={
                DATASET_ID:
                    dataframe,
            },

            approved_choices=
                choices,
        )
    )


    result = (
        derived[
            DATASET_ID
        ]
    )


    assert (
        set(
            result[
                "category"
            ].unique()
        )
        ==
        {
            "Furniture",
            "Electronics",
            "Accessories",
        }
    )


    assert (
        set(
            result[
                "channel"
            ].unique()
        )
        ==
        {
            "Web",
            "Store",
        }
    )


    assert (
        execution.applied_action_count
        ==
        4
    )


    assert (
        execution.changed_cell_count
        >
        0
    )


    assert (
        dataframe.equals(
            source_copy
        )
    )


    print(
        "Approved canonicalization materializes every alias group without mutating source: PASS"
    )


# ============================================================
# 6. COMPONENT VERSIONS
# ============================================================


def test_component_versions(
) -> None:
    assert (
        semantic_review
        .SEMANTIC_REVIEW_RULE_VERSION
        ==
        "semantic_review_v0.3"
    )


    assert (
        semantic_review
        .SEMANTIC_CANONICALIZATION_RULE_VERSION
        ==
        "semantic_canonicalization_v0.1"
    )


    print(
        "Semantic Review compatibility and canonicalization component versions: PASS"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        "=== DATALENS SEMANTIC CANONICALIZATION v0.1 ==="
    )

    print()


    test_multiple_alias_groups_become_independent_candidates()

    test_strict_aliases_do_not_depend_on_llm()

    test_canonical_value_is_existing_and_deterministic()

    test_no_user_approval_means_no_mutation()

    test_approved_alias_groups_are_all_materialized()

    test_component_versions()


    print()

    print(
        "Semantic Canonicalization v0.1: PASS"
    )


if __name__ == "__main__":
    main()
