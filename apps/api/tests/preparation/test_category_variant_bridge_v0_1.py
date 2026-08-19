from __future__ import annotations

import pandas as pd

import app.preparation.semantic_review as semantic_review

from app.preparation.data_quality import (
    CATEGORY_VARIANT_BRIDGE_RULE_VERSION,
    QUALITY_ENGINE_RULE_VERSION,
    QualityIssueKind,
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


def build_fixture() -> tuple[
    pd.DataFrame,
    dict,
]:
    dataframe = pd.DataFrame(
        {
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

            "quantity": [
                1,
                2,
                1,
                3,
                2,
                4,
                1,
                2,
                3,
                1,
                4,
                2,
            ],
        }
    )

    record = {
        "dataset_id":
            DATASET_ID,

        "filename":
            "orders_test.csv",

        "dataframe":
            dataframe,
    }

    return (
        dataframe,
        record,
    )


def category_variant_issues(
    quality_report,
):
    return [
        issue

        for issue
        in quality_report.issues

        if (
            issue.kind
            ==
            QualityIssueKind
            .CATEGORY_FORMAT_VARIANTS
        )
    ]


def test_data_quality_detects_real_format_variants(
) -> None:
    (
        dataframe,
        record,
    ) = build_fixture()


    report = (
        build_data_quality_report(
            [
                record
            ]
        )
    )


    issues = (
        category_variant_issues(
            report
        )
    )


    assert (
        len(
            issues
        )
        ==
        2
    )


    by_column = {
        issue.column:
            issue

        for issue
        in issues
    }


    assert (
        by_column[
            "category"
        ].evidence.details[
            "variant_groups"
        ]
        ==
        [
            [
                "Furniture",
                "furniture",
            ],
            [
                "Electronics",
                "electronics",
            ],
        ]
    )


    assert (
        by_column[
            "channel"
        ].evidence.details[
            "variant_groups"
        ]
        ==
        [
            [
                "Web",
                "web",
            ],
            [
                "Store",
                "store",
            ],
        ]
    )


    assert all(
        issue.semantic_review_recommended
        is True

        for issue
        in issues
    )


    assert all(
        issue.proposal.automatic_safe
        is False

        for issue
        in issues
    )


    assert all(
        issue.proposal.requires_user_confirmation
        is True

        for issue
        in issues
    )


    print(
        "Data Quality detects category/channel formatting variants and forwards them safely: PASS"
    )


def test_semantic_review_receives_four_independent_groups(
) -> None:
    (
        dataframe,
        record,
    ) = build_fixture()


    quality_report = (
        build_data_quality_report(
            [
                record
            ]
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


    format_candidates = [
        candidate

        for candidate
        in candidates

        if (
            candidate.kind
            ==
            QualityIssueKind
            .CATEGORY_FORMAT_VARIANTS
        )
    ]


    assert (
        len(
            format_candidates
        )
        ==
        4
    )


    exact_groups = {
        frozenset(
            candidate.candidate_values
        )

        for candidate
        in format_candidates
    }


    assert exact_groups == {
        frozenset(
            {
                "Furniture",
                "furniture",
            }
        ),
        frozenset(
            {
                "Electronics",
                "electronics",
            }
        ),
        frozenset(
            {
                "Web",
                "web",
            }
        ),
        frozenset(
            {
                "Store",
                "store",
            }
        ),
    }


    print(
        "Semantic Review receives four independent deterministic format groups: PASS"
    )


def test_strict_variants_are_proposed_without_gemma(
) -> None:
    (
        dataframe,
        record,
    ) = build_fixture()


    quality_report = (
        build_data_quality_report(
            [
                record
            ]
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
                "Strict category formatting variants "
                "must not require Gemma."
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


    format_decisions = [
        decision

        for decision
        in report.decisions

        if (
            decision.kind
            ==
            QualityIssueKind
            .CATEGORY_FORMAT_VARIANTS
        )
    ]


    assert (
        len(
            format_decisions
        )
        ==
        4
    )


    assert all(
        decision.verdict.value
        ==
        "merge_values"

        for decision
        in format_decisions
    )


    assert all(
        decision.executable
        is False

        for decision
        in format_decisions
    )


    assert all(
        decision.requires_user_confirmation
        is True

        for decision
        in format_decisions
    )


    print(
        "Strict formatting variants are proposed by Python without Gemma: PASS"
    )


def test_user_approval_materializes_clean_categories(
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
        build_data_quality_report(
            [
                record
            ]
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


    format_decisions = [
        decision

        for decision
        in review.decisions

        if (
            decision.kind
            ==
            QualityIssueKind
            .CATEGORY_FORMAT_VARIANTS
        )
    ]


    plan = (
        build_semantic_cleaning_plan(
            format_decisions
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
        derived_frames,
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
        derived_frames[
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
        dataframe.equals(
            source_copy
        )
    )


    print(
        "Approved format canonicalization materializes clean categories without mutating source: PASS"
    )


def test_distinct_categories_are_not_false_positives(
) -> None:
    dataframe = pd.DataFrame(
        {
            "category": [
                "Furniture",
                "Electronics",
                "Accessories",
                "Furniture",
                "Electronics",
                "Accessories",
            ],

            "channel": [
                "Web",
                "Store",
                "Web",
                "Store",
                "Web",
                "Store",
            ],
        }
    )


    report = (
        build_data_quality_report(
            [
                {
                    "dataset_id":
                        DATASET_ID,

                    "filename":
                        "clean.csv",

                    "dataframe":
                        dataframe,
                }
            ]
        )
    )


    issues = (
        category_variant_issues(
            report
        )
    )


    assert issues == []


    print(
        "Distinct clean categories are not flagged as formatting variants: PASS"
    )


def test_versions(
) -> None:
    assert (
        QUALITY_ENGINE_RULE_VERSION
        ==
        "data_quality_engine_v0.2"
    )


    assert (
        CATEGORY_VARIANT_BRIDGE_RULE_VERSION
        ==
        "category_variant_bridge_v0.1"
    )


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
        "Quality/Semantic bridge component versions: PASS"
    )


def main() -> None:
    print(
        "=== DATALENS CATEGORY VARIANT BRIDGE v0.1 ==="
    )

    print()


    test_data_quality_detects_real_format_variants()

    test_semantic_review_receives_four_independent_groups()

    test_strict_variants_are_proposed_without_gemma()

    test_user_approval_materializes_clean_categories()

    test_distinct_categories_are_not_false_positives()

    test_versions()


    print()

    print(
        "Category Variant Bridge v0.1: PASS"
    )


if __name__ == "__main__":
    main()
