from __future__ import annotations


import json


from collections import Counter
from pathlib import Path


CASES_PATH = Path(
    "artifacts/adaptation/holdouts/"
    "datalens_semantic_qlora_v0.4_"
    "hospital_emergency_department_operations_"
    "holdout_v0.1_cases.json"
)


EXPECTED_DOMAIN = (
    "hospital_emergency_department_operations"
)


EXPECTED_RELATIONS = [
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
    "unrelated",
    "uncertain",
]


EXPECTED_PROTOCOL_COMMIT = (
    "98e73521d633406c8de728aa667dc2525f6787a3"
)


EXPECTED_PROTOCOL_FREEZE = (
    "e00f9f2d6ef4a8eea47da58b2b200536dc8337c48282f301153a63a3dc215ce6"
)


def load_holdout() -> dict:

    return json.loads(
        CASES_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_holdout_identity() -> None:

    holdout = load_holdout()

    assert (
        holdout["domain"]
        ==
        EXPECTED_DOMAIN
    )

    assert (
        holdout["protocol_commit"]
        ==
        EXPECTED_PROTOCOL_COMMIT
    )

    assert (
        holdout[
            "protocol_aggregate_freeze_sha256"
        ]
        ==
        EXPECTED_PROTOCOL_FREEZE
    )


def test_holdout_has_exactly_30_cases() -> None:

    holdout = load_holdout()

    cases = holdout[
        "cases"
    ]

    assert len(cases) == 30

    assert (
        holdout["case_count"]
        ==
        30
    )


def test_relation_balance_is_exact() -> None:

    cases = load_holdout()[
        "cases"
    ]

    counts = Counter(
        case["gold_relation"]

        for case
        in cases
    )

    assert set(
        counts
    ) == set(
        EXPECTED_RELATIONS
    )

    for relation in EXPECTED_RELATIONS:

        assert (
            counts[relation]
            ==
            6
        )


def test_case_ids_are_unique_and_sequential() -> None:

    cases = load_holdout()[
        "cases"
    ]

    ids = [
        case[
            "case_id"
        ]

        for case
        in cases
    ]

    assert len(
        ids
    ) == len(
        set(ids)
    )

    assert ids == [
        f"hospital-{index:03d}"

        for index
        in range(
            1,
            31,
        )
    ]


def test_all_cases_use_selected_domain() -> None:

    for case in load_holdout()[
        "cases"
    ]:

        assert (
            case["domain"]
            ==
            EXPECTED_DOMAIN
        )


def test_case_schema_is_exact() -> None:

    expected_keys = {
        "case_id",
        "domain",
        "left_metric",
        "left_description",
        "right_metric",
        "right_description",
        "gold_relation",
        "gold_reason",
    }


    for case in load_holdout()[
        "cases"
    ]:

        assert (
            set(
                case
            )
            ==
            expected_keys
        )


def test_metric_pairs_are_unique() -> None:

    pairs = []


    for case in load_holdout()[
        "cases"
    ]:

        pair = (
            case[
                "left_metric"
            ],
            case[
                "right_metric"
            ],
        )

        pairs.append(
            pair
        )


    assert len(
        pairs
    ) == len(
        set(pairs)
    )


def test_gold_reason_is_non_empty() -> None:

    for case in load_holdout()[
        "cases"
    ]:

        reason = case[
            "gold_reason"
        ]

        assert isinstance(
            reason,
            str,
        )

        assert reason.strip()


def test_authoring_was_model_blind() -> None:

    holdout = load_holdout()

    assert (
        holdout[
            "model_outputs_observed_during_authoring"
        ]
        is False
    )

    assert (
        holdout[
            "base_or_adapter_execution_during_authoring"
        ]
        is False
    )


def test_external_medical_knowledge_not_required() -> None:

    assert (
        load_holdout()[
            "external_medical_knowledge_required"
        ]
        is False
    )


def main() -> None:

    tests = [
        test_holdout_identity,
        test_holdout_has_exactly_30_cases,
        test_relation_balance_is_exact,
        test_case_ids_are_unique_and_sequential,
        test_all_cases_use_selected_domain,
        test_case_schema_is_exact,
        test_metric_pairs_are_unique,
        test_gold_reason_is_non_empty,
        test_authoring_was_model_blind,
        test_external_medical_knowledge_not_required,
    ]


    print(
        "=== R8 HOSPITAL INDEPENDENT HOLDOUT v0.1 ==="
    )

    print()


    for test in tests:

        test()

        print(
            f"[PASS] {test.__name__}"
        )


    print()

    print(
        "PASS - hospital independent holdout v0.1"
    )


if __name__ == "__main__":

    main()
