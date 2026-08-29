from __future__ import annotations

import ast

from pathlib import Path


from app.adaptation.experiment_design_v0_4 import (
    DESIGN_PATH,
    EXPECTED_TRAINING_EXAMPLE_COUNT,
    QLORA_V04_EXPERIMENT_DESIGN_RULE_VERSION,
    QLORA_V04_EXPERIMENT_ID,
    RELATIONS,
    TRAINING_TARGET_COUNTS,
    V04_HOLDOUT_CASES_PER_RELATION,
    V04_HOLDOUT_DOMAIN,
    V04_HOLDOUT_EXPECTED_CASE_COUNT,
    load_json_object,
    validate_experiment_design,
)


print(
    "=== DATALENS QLORA v0.4 EXPERIMENT DESIGN v0.1 ==="
)

print()


assert (
    QLORA_V04_EXPERIMENT_DESIGN_RULE_VERSION
    ==
    "qlora_v0.4_experiment_design_v0.1"
)


assert (
    QLORA_V04_EXPERIMENT_ID
    ==
    "datalens-semantic-qlora-v0.4"
)


assert (
    EXPECTED_TRAINING_EXAMPLE_COUNT
    ==
    230
)


assert (
    TRAINING_TARGET_COUNTS
    ==
    {
        "same_metric_different_state":
            50,

        "same_process_different_stage":
            50,

        "related_distinct_metric":
            50,

        "unrelated":
            40,

        "uncertain":
            40,
    }
)


print(
    "Experiment identity: PASS"
)

print(
    "Training target distribution: PASS"
)


design = load_json_object(
    DESIGN_PATH
)


validate_experiment_design(
    design
)


print(
    "Deterministic design validation: PASS"
)


assert (
    design[
        "task"
    ][
        "assistant_target_format"
    ]
    ==
    "strict_json_object"
)


assert (
    design[
        "task"
    ][
        "assistant_target_schema"
    ][
        "required"
    ]
    ==
    [
        "relation",
        "reason",
    ]
)


assert (
    set(
        design[
            "task"
        ][
            "assistant_target_schema"
        ][
            "fields"
        ][
            "relation"
        ][
            "allowed_values"
        ]
    )
    ==
    set(
        RELATIONS
    )
)


print(
    "Structured relation + reason target: PASS"
)


assert (
    design[
        "dataset_design"
    ][
        "contrastive_grouping_required"
    ]
    is True
)


assert (
    design[
        "dataset_design"
    ][
        "hard_negative_required"
    ]
    is True
)


assert (
    design[
        "dataset_design"
    ][
        "minimum_examples_per_contrastive_group"
    ]
    ==
    3
)


assert (
    design[
        "dataset_design"
    ][
        "minimum_distinct_domains"
    ]
    ==
    12
)


print(
    "Contrastive dataset requirements: PASS"
)


assert (
    V04_HOLDOUT_DOMAIN
    ==
    "airport_ground_operations"
)


assert (
    V04_HOLDOUT_EXPECTED_CASE_COUNT
    ==
    30
)


assert (
    V04_HOLDOUT_CASES_PER_RELATION
    ==
    6
)


holdout = (
    design[
        "evaluation_design"
    ][
        "new_independent_holdout"
    ]
)


assert (
    holdout[
        "status"
    ]
    ==
    "required_not_yet_authored"
)


assert (
    holdout[
        "must_be_frozen_before_training"
    ]
    is True
)


assert (
    holdout[
        "must_be_frozen_before_training_dataset_freeze"
    ]
    is True
)


assert (
    holdout[
        "must_not_be_used_for_training"
    ]
    is True
)


assert (
    holdout[
        "must_not_be_used_for_hyperparameter_tuning"
    ]
    is True
)


print(
    "Independent holdout pre-training gate: PASS"
)


hotel = (
    design[
        "evaluation_design"
    ][
        "hotel"
    ]
)


assert (
    hotel[
        "role"
    ]
    ==
    "diagnostic_regression_only"
)


assert (
    hotel[
        "may_count_as_new_independent_holdout"
    ]
    is False
)


assert (
    hotel[
        "may_be_used_for_training"
    ]
    is False
)


assert (
    hotel[
        "may_be_used_for_v0_4_hyperparameter_tuning"
    ]
    is False
)


print(
    "Hotel governance: PASS"
)


final_acceptance = (
    design[
        "evaluation_design"
    ][
        "final_acceptance"
    ]
)


assert (
    final_acceptance[
        "remains_closed"
    ]
    is True
)


assert (
    final_acceptance[
        "may_be_loaded_before_all_prior_gates_pass"
    ]
    is False
)


print(
    "Final Acceptance remains closed: PASS"
)


assert (
    design[
        "learning_objective"
    ][
        "training_loss_is_acceptance_evidence"
    ]
    is False
)


assert (
    design[
        "prohibited_actions_before_design_gates"
    ][
        "gpu_training"
    ]
    is True
)


assert (
    design[
        "prohibited_actions_before_design_gates"
    ][
        "v0_4_training_before_holdout_freeze"
    ]
    is True
)


print(
    "Pre-training safety gates: PASS"
)


source_path = (
    Path(__file__)
    .resolve()
    .parent
    / "app"
    / "adaptation"
    / "experiment_design_v0_4.py"
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
    "Offline-only design implementation: PASS"
)


print()

print(
    "SAFETY"
)

print(
    "  Training dataset authored: False"
)

print(
    "  v0.4 holdout cases authored: False"
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
    "  Training executed: False"
)

print(
    "  Evaluation executed: False"
)

print(
    "  Final Acceptance loaded: False"
)

print(
    "  Final Acceptance evaluated: False"
)


print()

print(
    "DATALENS QLORA v0.4 EXPERIMENT DESIGN v0.1: PASS"
)
