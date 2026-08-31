from __future__ import annotations

import ast

from collections import Counter
from pathlib import Path


from app.adaptation.airport_ground_operations_holdout import (
    AIRPORT_HOLDOUT_FREEZE_RULE_VERSION,
    AIRPORT_HOLDOUT_ID,
    AIRPORT_HOLDOUT_RULE_VERSION,
    CASES_PATH,
    FREEZE_PATH,
    RELATIONS,
    load_json_object,
    sha256_file,
    validate_cases,
    validate_freeze,
)


print(
    "=== DATALENS QLORA v0.4 AIRPORT HOLDOUT v0.1 ==="
)

print()


assert (
    AIRPORT_HOLDOUT_RULE_VERSION
    ==
    "qlora_v0.4_airport_holdout_v0.1"
)


assert (
    AIRPORT_HOLDOUT_FREEZE_RULE_VERSION
    ==
    "qlora_v0.4_airport_holdout_freeze_v0.1"
)


cases_payload = load_json_object(
    CASES_PATH
)


freeze = load_json_object(
    FREEZE_PATH
)


validate_cases(
    cases_payload
)


validate_freeze(
    cases_sha256=
        sha256_file(
            CASES_PATH
        ),

    freeze=
        freeze,
)


print(
    "Holdout identity: PASS"
)

print(
    "Deterministic validation: PASS"
)


assert (
    cases_payload[
        "holdout_id"
    ]
    ==
    AIRPORT_HOLDOUT_ID
)


assert (
    cases_payload[
        "case_count"
    ]
    ==
    30
)


counts = Counter(
    case[
        "expected_relation"
    ]

    for case
    in cases_payload[
        "cases"
    ]
)


assert counts == Counter(
    {
        relation:
            6

        for relation
        in RELATIONS
    }
)


print(
    "30 cases: PASS"
)

print(
    "Balanced 6 per relation: PASS"
)


assert (
    cases_payload[
        "status"
    ]
    ==
    "frozen_before_v0_4_training_data_authoring"
)


policy = cases_payload[
    "independence_policy"
]


assert (
    policy[
        "airport_domain_forbidden_in_v0_4_training_data"
    ]
    is True
)


assert (
    policy[
        "must_not_be_used_for_training_data_authoring"
    ]
    is True
)


assert (
    policy[
        "evaluation_results_may_not_drive_hyperparameter_tuning"
    ]
    is True
)


print(
    "Independent-holdout governance: PASS"
)


protocol = cases_payload[
    "protocol"
]


assert (
    protocol[
        "generation"
    ]
    ==
    "deterministic_greedy"
)


assert (
    protocol[
        "decoding"
    ][
        "do_sample"
    ]
    is False
)


assert (
    protocol[
        "assistant_output"
    ]
    ==
    "strict_json_relation_plus_reason"
)


assert (
    protocol[
        "evaluation_order"
    ]
    ==
    [
        "pinned_base",
        "same_base_plus_frozen_candidate_adapter",
    ]
)


print(
    "Evaluation protocol frozen: PASS"
)


gate = cases_payload[
    "preregistered_acceptance_gate"
]


assert (
    gate[
        "minimum_adapted_accuracy"
    ]
    ==
    0.70
)


assert (
    gate[
        "minimum_adapted_macro_accuracy"
    ]
    ==
    0.70
)


assert (
    gate[
        "minimum_per_relation_accuracy"
    ]
    ==
    0.50
)


assert (
    gate[
        "minimum_uncertain_accuracy"
    ]
    ==
    0.666667
)


assert (
    gate[
        "strict_json_validity_rate"
    ]
    ==
    1.0
)


assert (
    gate[
        "adapted_accuracy_must_not_be_lower_than_base"
    ]
    is True
)


assert (
    gate[
        "adapted_macro_accuracy_must_not_be_lower_than_base"
    ]
    is True
)


print(
    "Acceptance gates preregistered: PASS"
)


prerequisites = cases_payload[
    "prerequisite_gates"
]


assert (
    prerequisites[
        "hotel_diagnostic_must_pass_before_holdout_is_consumed"
    ]
    is True
)


assert (
    prerequisites[
        "known_regression_and_safety_gates_must_pass"
    ]
    is True
)


print(
    "Holdout-consumption prerequisites: PASS"
)


assert (
    freeze[
        "frozen_before_v0_4_training_data_authoring"
    ]
    is True
)


assert (
    freeze[
        "frozen_before_v0_4_training"
    ]
    is True
)


assert (
    freeze[
        "v0_4_training_dataset_authored"
    ]
    is False
)


assert (
    freeze[
        "v0_4_training_executed"
    ]
    is False
)


assert (
    freeze[
        "evaluation_executed"
    ]
    is False
)


assert (
    freeze[
        "results_observed"
    ]
    is False
)


assert (
    freeze[
        "final_acceptance_opened"
    ]
    is False
)


print(
    "Pre-training freeze state: PASS"
)


source_path = (
    Path(__file__)
    .resolve()
    .parent
    / "app"
    / "adaptation"
    / "airport_ground_operations_holdout.py"
)


source = source_path.read_text(
    encoding="utf-8-sig"
)


tree = ast.parse(
    source
)


heavy_modules = {
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "safetensors",
}


for node in ast.walk(
    tree
):
    if isinstance(
        node,
        ast.Import,
    ):
        for alias in node.names:
            assert (
                alias.name.split(
                    "."
                )[0]
                not in
                heavy_modules
            )

    elif isinstance(
        node,
        ast.ImportFrom,
    ):
        module = (
            node.module
            or
            ""
        )

        assert (
            module.split(
                "."
            )[0]
            not in
            heavy_modules
        )


print(
    "Offline-only implementation: PASS"
)


print()

print(
    "SAFETY"
)

print(
    "  v0.4 training dataset authored: False"
)

print(
    "  v0.4 training executed: False"
)

print(
    "  Holdout evaluated: False"
)

print(
    "  Holdout results observed: False"
)

print(
    "  Model loaded: False"
)

print(
    "  Adapter loaded: False"
)

print(
    "  CUDA requested: False"
)

print(
    "  Final Acceptance loaded: False"
)

print(
    "  Final Acceptance evaluated: False"
)


print()

print(
    (
        "Cases SHA256: "
        f"{sha256_file(CASES_PATH)}"
    )
)


print()

print(
    "DATALENS QLORA v0.4 AIRPORT HOLDOUT v0.1: PASS"
)
