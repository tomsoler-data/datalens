from __future__ import annotations


import hashlib
import json


from collections import Counter
from pathlib import Path


CASES_PATH = Path(
    "artifacts/adaptation/holdouts/"
    "datalens_semantic_qlora_v0.4_"
    "hospital_emergency_department_operations_"
    "holdout_v0.1_cases.json"
)


FREEZE_PATH = Path(
    "artifacts/adaptation/holdouts/"
    "datalens_semantic_qlora_v0.4_"
    "hospital_emergency_department_operations_"
    "holdout_v0.1_freeze.json"
)


STRUCTURAL_TEST_PATH = Path(
    "tests/adaptation/"
    "test_hospital_independent_holdout_v0_1.py"
)


EXPECTED_CASES_SHA = (
    "92e4f21e7323cd053fd5f53b51e14932"
    "97fdd28cfb717be56f7be963f298d9fc"
)


EXPECTED_PROTOCOL_COMMIT = (
    "98e73521d633406c8de728aa667dc2525f6787a3"
)


EXPECTED_PROTOCOL_FREEZE = (
    "e00f9f2d6ef4a8eea47da58b2b200536"
    "dc8337c48282f301153a63a3dc215ce6"
)


EXPECTED_TRAINING_SHA = (
    "4fd00586f2d53d6de57f5cbc5f1d7bfb"
    "2e512960e60b30c28596aaefbac322b7"
)


EXPECTED_RELATIONS = [
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
    "unrelated",
    "uncertain",
]


def sha256(
    path: Path,
) -> str:

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_freeze() -> dict:

    return json.loads(
        FREEZE_PATH.read_text(
            encoding="utf-8"
        )
    )


def load_cases() -> dict:

    return json.loads(
        CASES_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_cases_sha_is_frozen() -> None:

    freeze = load_freeze()

    assert (
        sha256(
            CASES_PATH
        )
        ==
        EXPECTED_CASES_SHA
    )

    assert (
        freeze[
            "protected_holdout"
        ][
            "cases_sha256"
        ]
        ==
        EXPECTED_CASES_SHA
    )


def test_protocol_authority_is_frozen() -> None:

    authority = load_freeze()[
        "protocol_authority"
    ]

    assert (
        authority[
            "commit"
        ]
        ==
        EXPECTED_PROTOCOL_COMMIT
    )

    assert (
        authority[
            "aggregate_freeze_sha256"
        ]
        ==
        EXPECTED_PROTOCOL_FREEZE
    )

    assert (
        authority[
            "remote_preregistered_before_holdout_construction"
        ]
        is True
    )


def test_training_authority_is_frozen() -> None:

    authority = load_freeze()[
        "training_authority"
    ]

    assert (
        authority[
            "dataset_sha256"
        ]
        ==
        EXPECTED_TRAINING_SHA
    )

    assert (
        authority[
            "dataset_example_count"
        ]
        ==
        230
    )

    assert (
        authority[
            "selected_domain_present_in_training"
        ]
        is False
    )


def test_structural_test_identity_is_frozen() -> None:

    contract = load_freeze()[
        "structural_contract"
    ]

    assert (
        sha256(
            STRUCTURAL_TEST_PATH
        )
        ==
        contract[
            "test_sha256"
        ]
    )

    assert (
        contract[
            "passed_before_freeze"
        ]
        is True
    )


def test_holdout_balance_is_frozen() -> None:

    freeze = load_freeze()

    cases = load_cases()[
        "cases"
    ]

    counts = Counter(
        case[
            "gold_relation"
        ]

        for case
        in cases
    )

    assert len(
        cases
    ) == 30

    assert (
        freeze[
            "protected_holdout"
        ][
            "case_count"
        ]
        ==
        30
    )

    assert (
        freeze[
            "protected_holdout"
        ][
            "balanced_cases_per_relation"
        ]
        ==
        6
    )

    for relation in EXPECTED_RELATIONS:

        assert (
            counts[
                relation
            ]
            ==
            6
        )

        assert (
            freeze[
                "protected_holdout"
            ][
                "relation_counts"
            ][
                relation
            ]
            ==
            6
        )


def test_independence_evidence_is_frozen() -> None:

    evidence = load_freeze()[
        "independence_evidence"
    ]

    assert (
        evidence[
            "hospital_domain_absent_from_training"
        ]
        is True
    )

    zero_fields = (
        "exact_training_metric_name_pair_reuse_count",
        "exact_complete_metric_pair_reuse_count",
        "exact_training_example_reuse_count",
        "individual_metric_name_overlap_count",
        "individual_definition_overlap_count",
        "reverse_pair_duplicate_count",
        "prompt_relation_label_leakage_count",
    )

    for field in zero_fields:

        assert (
            evidence[
                field
            ]
            ==
            0
        )

    assert (
        evidence[
            "uncertain_cases"
        ]
        ==
        6
    )

    assert (
        evidence[
            "uncertain_cases_with_structural_ambiguity_cues"
        ]
        ==
        6
    )


def test_model_execution_boundary_is_frozen() -> None:

    boundary = load_freeze()[
        "authoring_boundary"
    ]

    assert (
        boundary[
            "model_outputs_observed_during_authoring"
        ]
        is False
    )

    assert (
        boundary[
            "base_model_executed_during_authoring"
        ]
        is False
    )

    assert (
        boundary[
            "adapter_executed_during_authoring"
        ]
        is False
    )

    assert (
        boundary[
            "inference_executed_before_freeze"
        ]
        is False
    )

    assert (
        boundary[
            "training_executed_during_r8_holdout_authoring"
        ]
        is False
    )


def test_gold_is_immutable_after_observation() -> None:

    policy = load_freeze()[
        "gold_policy"
    ]

    assert (
        policy[
            "gold_relations_frozen"
        ]
        is True
    )

    assert (
        policy[
            "gold_reasons_frozen"
        ]
        is True
    )

    assert (
        policy[
            "gold_changes_after_model_output_observation_permitted"
        ]
        is False
    )

    assert (
        policy[
            "case_changes_after_model_output_observation_permitted"
        ]
        is False
    )


def test_single_use_policy_is_fail_closed() -> None:

    policy = load_freeze()[
        "single_use_policy"
    ]

    assert (
        policy[
            "official_consumption_count_maximum"
        ]
        ==
        1
    )

    assert (
        policy[
            "same_holdout_for_base_and_adapted"
        ]
        is True
    )

    assert (
        policy[
            "reexecution_after_results_observed"
        ]
        is False
    )

    assert (
        policy[
            "model_selection_from_results"
        ]
        is False
    )

    assert (
        policy[
            "hyperparameter_tuning_from_results"
        ]
        is False
    )


def main() -> None:

    tests = [
        test_cases_sha_is_frozen,
        test_protocol_authority_is_frozen,
        test_training_authority_is_frozen,
        test_structural_test_identity_is_frozen,
        test_holdout_balance_is_frozen,
        test_independence_evidence_is_frozen,
        test_model_execution_boundary_is_frozen,
        test_gold_is_immutable_after_observation,
        test_single_use_policy_is_fail_closed,
    ]


    print(
        "=== R8 HOSPITAL HOLDOUT FREEZE v0.1 ==="
    )

    print()


    for test in tests:

        test()

        print(
            f"[PASS] {test.__name__}"
        )


    print()

    print(
        "PASS - hospital independent "
        "holdout freeze v0.1"
    )


if __name__ == "__main__":

    main()
