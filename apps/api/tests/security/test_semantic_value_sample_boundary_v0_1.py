from __future__ import annotations


from pathlib import (
    Path,
)

from unittest.mock import (
    patch,
)


import pandas as pd


import app.preparation.semantic_review as semantic_review_module

import app.semantics.profiler as profiler_module

import app.semantics.provider as semantic_provider_module


from app.preparation.data_quality import (
    QualityIssueKind,
)

from app.preparation.semantic_review import (
    SemanticReviewCandidate,
)

from app.security.semantic_value_sample import (
    MAX_SEMANTIC_VALUE_SAMPLE_VALUES,
    SEMANTIC_VALUE_SAMPLE_BOUNDARY_RULE_VERSION,
    SemanticValueSampleBoundaryError,
)


TEST_RULE_VERSION = (
    "semantic_value_sample_boundary_test_v0.1"
)


def _candidate(
    *,
    values: list[
        str
    ],
    deterministic_details=None,
) -> SemanticReviewCandidate:

    return (
        SemanticReviewCandidate(
            issue_id=
                "quality:sample-boundary",

            dataset_id=
                "dataset:0001",

            dataset_filename=
                "orders.csv",

            column=
                "segment",

            kind=
                QualityIssueKind
                .POSSIBLE_SEMANTIC_ALIASES,

            severity=
                "warning",

            title=
                "Possible semantic aliases",

            explanation=
                "Values may represent aliases.",

            observed_count=
                100,

            affected_ratio=
                0.2,

            examples=
                list(
                    values
                ),

            candidate_values=
                list(
                    values
                ),

            candidate_groups=[
                list(
                    values
                )
            ],

            context={
                "column_dtype":
                    "object",

                "sample_unique_values":
                    list(
                        values
                    ),

                "deterministic_details":
                    (
                        deterministic_details
                        or {}
                    ),

                "source_quality_issue_id":
                    "quality:sample-boundary",

                "alias_group_index":
                    0,

                "semantic_canonicalization_rule_version":
                    "semantic_canonicalization_v0.1",
            },
        )
    )


def test_rule_and_budget(
) -> None:

    assert (
        SEMANTIC_VALUE_SAMPLE_BOUNDARY_RULE_VERSION
        ==
        "semantic_value_sample_boundary_v0.1"
    )


    assert (
        MAX_SEMANTIC_VALUE_SAMPLE_VALUES
        ==
        5
    )


    assert (
        MAX_SEMANTIC_VALUE_SAMPLE_VALUES
        <=
        5
    )


def test_profiler_sample_is_bounded_and_unique(
) -> None:

    series = pd.Series(
        [
            "A",
            "A",
            None,
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
        ]
    )


    values = (
        profiler_module
        .sample_values(
            series
        )
    )


    assert (
        values
        ==
        [
            "A",
            "B",
            "C",
            "D",
            "E",
        ]
    )


    assert (
        len(
            values
        )
        <=
        MAX_SEMANTIC_VALUE_SAMPLE_VALUES
    )


def test_profiler_context_is_bounded(
) -> None:

    series = pd.Series(
        [
            f"value-{index}"

            for index
            in range(
                20
            )
        ]
    )


    context = (
        profiler_module
        .build_column_context(
            dataset_id=
                "dataset:0001",

            filename=
                "orders.csv",

            column=
                "segment",

            series=
                series,

            peer_columns=[
                "segment",
                "amount",
            ],
        )
    )


    assert (
        len(
            context[
                "sample_values"
            ]
        )
        <=
        MAX_SEMANTIC_VALUE_SAMPLE_VALUES
    )


def test_preparation_unique_sample_is_bounded(
) -> None:

    dataframe = pd.DataFrame(
        {
            "segment": [
                f"value-{index}"

                for index
                in range(
                    20
                )
            ]
        }
    )


    values = (
        semantic_review_module
        ._safe_unique_values(
            dataframe,
            "segment",
        )
    )


    assert (
        len(
            values
        )
        ==
        MAX_SEMANTIC_VALUE_SAMPLE_VALUES
    )


def test_over_budget_candidate_is_blocked_before_transport(
) -> None:

    candidate = (
        _candidate(
            values=[
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
            ]
        )
    )


    with patch.object(
        semantic_review_module,
        "open_local_llm_request",
    ) as transport:

        try:
            (
                semantic_review_module
                ._ollama_chat_one(
                    model=
                        "gemma3:4b",

                    candidate=
                        candidate,

                    ollama_chat_url=
                        (
                            "http://127.0.0.1:"
                            "11434/api/chat"
                        ),

                    timeout_seconds=
                        1.0,
                )
            )

        except (
            SemanticValueSampleBoundaryError
        ):
            pass

        else:
            raise AssertionError(
                (
                    "Over-budget semantic values "
                    "were not rejected."
                )
            )


    transport.assert_not_called()


    # Each individual collection is within the limit here,
    # but the union contains six distinct values. The
    # aggregate privacy budget must still fail closed.
    aggregate_candidate = (
        _candidate(
            values=[
                "A",
                "B",
                "C",
            ]
        )
    )


    aggregate_candidate.candidate_groups = [
        [
            "D",
            "E",
            "F",
        ]
    ]


    with patch.object(
        semantic_review_module,
        "open_local_llm_request",
    ) as aggregate_transport:

        try:
            (
                semantic_review_module
                ._ollama_chat_one(
                    model=
                        "gemma3:4b",

                    candidate=
                        aggregate_candidate,

                    ollama_chat_url=
                        (
                            "http://127.0.0.1:"
                            "11434/api/chat"
                        ),

                    timeout_seconds=
                        1.0,
                )
            )

        except (
            SemanticValueSampleBoundaryError
        ):
            pass

        else:
            raise AssertionError(
                (
                    "Aggregate semantic value "
                    "budget was bypassed."
                )
            )


    aggregate_transport.assert_not_called()


def test_internal_deterministic_details_do_not_reach_prompt(
) -> None:

    poison = (
        "POISON_DETERMINISTIC_DETAILS_91f03a"
    )


    candidate = (
        _candidate(
            values=[
                "A",
                "B",
            ],

            deterministic_details={
                "candidate_pairs": [
                    [
                        poison,
                        poison,
                    ]
                ],

                "arbitrary_nested_payload": {
                    "secret":
                        poison
                },
            },
        )
    )


    prompt = (
        semantic_review_module
        ._user_prompt(
            candidate
        )
    )


    assert (
        poison
        not in
        prompt
    )


    assert (
        "deterministic_details"
        not in
        prompt
    )


    assert (
        '"sample_unique_values"'
        not in
        prompt
    )


    assert (
        '"examples"'
        not in
        prompt
    )


    assert (
        '"candidate_values"'
        in
        prompt
    )


def test_semantic_sample_classifications_are_locked(
) -> None:

    provider_source = (
        Path(
            semantic_provider_module
            .__file__
        )
        .read_text(
            encoding="utf-8-sig"
        )
    )


    preparation_source = (
        Path(
            semantic_review_module
            .__file__
        )
        .read_text(
            encoding="utf-8-sig"
        )
    )


    assert (
        "classified_llm_chat"
        in
        provider_source
    )


    assert (
        ".SEMANTIC_VALUE_SAMPLE"
        in
        provider_source
    )


    assert (
        "open_local_llm_request"
        in
        preparation_source
    )


    assert (
        ".SEMANTIC_VALUE_SAMPLE"
        in
        preparation_source
    )


def main(
) -> None:

    print(
        "=== DATALENS SEMANTIC VALUE "
        "SAMPLE BOUNDARY v0.1 ==="
    )

    print()


    tests = [
        (
            "Rule version and privacy budget",
            test_rule_and_budget,
        ),
        (
            "Profiler sample bounded and unique",
            test_profiler_sample_is_bounded_and_unique,
        ),
        (
            "Profiler context bounded",
            test_profiler_context_is_bounded,
        ),
        (
            "Preparation unique sample bounded",
            test_preparation_unique_sample_is_bounded,
        ),
        (
            "Over-budget blocked before transport",
            test_over_budget_candidate_is_blocked_before_transport,
        ),
        (
            "Internal deterministic details hidden",
            test_internal_deterministic_details_do_not_reach_prompt,
        ),
        (
            "Semantic sample classifications locked",
            test_semantic_sample_classifications_are_locked,
        ),
    ]


    for (
        label,
        test,
    ) in tests:

        test()

        print(
            f"[PASS] {label}"
        )


    print()

    print(
        "PASS - 7/7 semantic value "
        "sample boundary checks"
    )

    print(
        f"Rule: {TEST_RULE_VERSION}"
    )


if __name__ == "__main__":
    main()
