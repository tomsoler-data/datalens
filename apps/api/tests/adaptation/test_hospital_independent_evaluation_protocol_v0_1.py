from __future__ import annotations


import json


from pathlib import Path


PROTOCOL_PATH = Path(
    "artifacts/adaptation/evaluation/"
    "datalens_semantic_qlora_v0.4_"
    "hospital_independent_evaluation_protocol_v0.1.json"
)


EXPECTED_RELATIONS = [
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
    "unrelated",
    "uncertain",
]


EXPECTED_DOMAIN = (
    "hospital_emergency_department_operations"
)


EXPECTED_PARENT = (
    "c3794df356c6bb4373e01e10c6b37eaad4d2ad18"
)


def load_protocol() -> dict:

    return json.loads(
        PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_protocol_identity() -> None:

    protocol = load_protocol()

    assert (
        protocol["record_rule_version"]
        ==
        "qlora_v0.4_hospital_independent_evaluation_protocol_v0.1"
    )

    assert (
        protocol["experiment_id"]
        ==
        "datalens-semantic-qlora-v0.4"
    )

    assert (
        protocol["protocol_parent_git_commit"]
        ==
        EXPECTED_PARENT
    )


def test_post_training_methodology_is_explicit() -> None:

    methodology = load_protocol()[
        "methodology"
    ]

    assert (
        methodology[
            "candidate_training_completed_before_protocol"
        ]
        is True
    )

    assert (
        methodology[
            "benchmark_cases_existed_during_training"
        ]
        is False
    )

    assert (
        methodology[
            "benchmark_gold_existed_during_training"
        ]
        is False
    )

    assert (
        methodology[
            "candidate_tuned_on_this_benchmark"
        ]
        is False
    )

    assert (
        methodology[
            "claim_pretraining_frozen_benchmark"
        ]
        is False
    )

    assert (
        methodology[
            "protocol_must_be_committed_before_case_construction"
        ]
        is True
    )

    assert (
        methodology[
            "cases_must_be_frozen_before_first_model_execution"
        ]
        is True
    )


def test_task_contract() -> None:

    task = load_protocol()[
        "task"
    ]

    assert (
        task["domain"]
        ==
        EXPECTED_DOMAIN
    )

    assert (
        task["relations"]
        ==
        EXPECTED_RELATIONS
    )

    output_contract = task[
        "output_contract"
    ]

    assert (
        output_contract["required_keys"]
        ==
        [
            "relation",
            "reason",
        ]
    )

    assert (
        output_contract["additional_properties"]
        is False
    )

    assert (
        output_contract["relation_allowed_values"]
        ==
        EXPECTED_RELATIONS
    )


def test_benchmark_is_balanced_30_case_design() -> None:

    design = load_protocol()[
        "benchmark_design"
    ]

    assert (
        design["planned_case_count"]
        ==
        30
    )

    assert (
        design["balanced_cases_per_relation"]
        ==
        6
    )

    assert (
        design["relation_count"]
        ==
        5
    )

    assert (
        design["balanced"]
        is True
    )

    assert (
        6
        *
        5
        ==
        30
    )


def test_case_authoring_is_model_blind() -> None:

    constraints = load_protocol()[
        "benchmark_design"
    ][
        "case_authoring_constraints"
    ]

    assert (
        constraints[
            "model_outputs_visible_during_case_authoring"
        ]
        is False
    )

    assert (
        constraints[
            "base_or_adapter_execution_before_freeze_permitted"
        ]
        is False
    )

    assert (
        constraints[
            "exact_training_example_reuse_permitted"
        ]
        is False
    )

    assert (
        constraints[
            "training_metric_pair_reuse_permitted"
        ]
        is False
    )

    assert (
        constraints[
            "historical_holdout_case_reuse_permitted"
        ]
        is False
    )


def test_generation_is_same_and_deterministic() -> None:

    generation = load_protocol()[
        "generation"
    ]

    assert (
        generation["mode"]
        ==
        "deterministic_greedy"
    )

    assert (
        generation["do_sample"]
        is False
    )

    assert (
        generation["num_beams"]
        ==
        1
    )

    assert (
        generation["max_new_tokens"]
        ==
        64
    )

    assert (
        generation["evaluation_order"]
        ==
        [
            "pinned_base",
            "same_base_plus_frozen_candidate_adapter",
        ]
    )

    assert (
        generation[
            "same_prompt_for_base_and_adapted"
        ]
        is True
    )


def test_absolute_acceptance_gates_are_frozen() -> None:

    gates = load_protocol()[
        "acceptance_gates"
    ][
        "adapted_absolute_gates"
    ]

    assert (
        gates["accuracy_minimum"]
        ==
        0.7
    )

    assert (
        gates["macro_accuracy_minimum"]
        ==
        0.7
    )

    assert (
        gates["per_relation_accuracy_minimum"]
        ==
        0.5
    )

    assert (
        gates["uncertain_accuracy_minimum"]
        ==
        0.666667
    )

    assert (
        gates["strict_json_validity_rate"]
        ==
        1.0
    )


def test_non_regression_and_promotion_signal_are_frozen() -> None:

    acceptance = load_protocol()[
        "acceptance_gates"
    ]

    non_regression = acceptance[
        "non_regression_gates"
    ]

    assert (
        non_regression["accuracy_delta_minimum"]
        ==
        0.0
    )

    assert (
        non_regression["macro_accuracy_delta_minimum"]
        ==
        0.0
    )

    promotion = acceptance[
        "promotion_signal_gate"
    ]

    assert (
        promotion[
            "adapted_correct_case_count_delta_minimum"
        ]
        ==
        1
    )

    assert (
        promotion[
            "equivalent_accuracy_delta_minimum"
        ]
        ==
        0.033333
    )


def test_single_use_policy_is_fail_closed() -> None:

    policy = load_protocol()[
        "single_use_policy"
    ]

    assert (
        policy[
            "one_official_consumption_for_v0_4"
        ]
        is True
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

    assert (
        policy[
            "retraining_then_retesting_same_holdout"
        ]
        is False
    )

    assert (
        policy[
            "reexecution_after_results_observed"
        ]
        is False
    )


def test_no_r8_protected_data_exists_in_protocol_state() -> None:

    boundaries = load_protocol()[
        "protected_boundaries"
    ]

    assert (
        boundaries["r8_cases_created"]
        is False
    )

    assert (
        boundaries["r8_gold_created"]
        is False
    )

    assert (
        boundaries["r8_cases_read"]
        is False
    )

    assert (
        boundaries["r8_results_observed"]
        is False
    )

    assert (
        boundaries[
            "greenhouse_reexecution_permitted"
        ]
        is False
    )


def main() -> None:

    tests = [
        test_protocol_identity,
        test_post_training_methodology_is_explicit,
        test_task_contract,
        test_benchmark_is_balanced_30_case_design,
        test_case_authoring_is_model_blind,
        test_generation_is_same_and_deterministic,
        test_absolute_acceptance_gates_are_frozen,
        test_non_regression_and_promotion_signal_are_frozen,
        test_single_use_policy_is_fail_closed,
        test_no_r8_protected_data_exists_in_protocol_state,
    ]


    print(
        "=== R8 HOSPITAL INDEPENDENT EVALUATION "
        "PROTOCOL v0.1 ==="
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
        "evaluation protocol v0.1"
    )


if __name__ == "__main__":

    main()
