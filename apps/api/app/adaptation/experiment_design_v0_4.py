from __future__ import annotations

import argparse
import hashlib
import json

from pathlib import Path
from typing import (
    Any,
)


QLORA_V04_EXPERIMENT_DESIGN_RULE_VERSION = (
    "qlora_v0.4_experiment_design_v0.1"
)


QLORA_V04_EXPERIMENT_ID = (
    "datalens-semantic-qlora-v0.4"
)


QLORA_V04_TASK_SCHEMA_VERSION = (
    "semantic_relation_reasoning_target_v0.1"
)


QLORA_V04_HOLDOUT_POLICY_VERSION = (
    "qlora_v0.4_holdout_policy_v0.1"
)


RELATIONS = (
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
    "unrelated",
    "uncertain",
)


TRAINING_TARGET_COUNTS = {
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


EXPECTED_TRAINING_EXAMPLE_COUNT = (
    sum(
        TRAINING_TARGET_COUNTS.values()
    )
)


V04_HOLDOUT_DOMAIN = (
    "airport_ground_operations"
)


V04_HOLDOUT_CASES_PER_RELATION = 6


V04_HOLDOUT_EXPECTED_CASE_COUNT = (
    len(
        RELATIONS
    )
    *
    V04_HOLDOUT_CASES_PER_RELATION
)


ROOT = Path(__file__).resolve().parents[2]


DESIGN_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "design"
    / (
        "datalens_semantic_qlora_v0.4_"
        "experiment_design_v0.1.json"
    )
)


FREEZE_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "design"
    / (
        "datalens_semantic_qlora_v0.4_"
        "experiment_design_v0.1_freeze.json"
    )
)


FAILURE_ANALYSIS_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "evaluation"
    / (
        "datalens_semantic_qlora_v0.3_"
        "reasoning_failure_analysis_v0.1.json"
    )
)


V03_EVALUATION_REPORT_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "evaluation"
    / (
        "datalens_semantic_qlora_v0.3_"
        "reasoning_evaluation_v0.2_report.json"
    )
)


SOURCE_V03_CLOSURE_COMMIT = (
    "448b231"
)


SOURCE_V03_FAILURE_ANALYSIS_SHA256 = (
    "0b5465125de253dca043a37525516b5c"
    "ca94c7bc5786d8d38e06db14040f4a05"
)


SOURCE_V03_EVALUATION_REPORT_SHA256 = (
    "1194baabef17891f84f2879a135586c7"
    "da95363cbac49c4258a8133d81c6206b"
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                8 * 1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_json_object(
    path: Path,
) -> dict[
    str,
    Any,
]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return payload


def canonical_training_target_example(
) -> dict[
    str,
    str,
]:
    return {
        "relation":
            "same_metric_different_state",

        "reason":
            (
                "Both metrics represent the same "
                "underlying quantity in different states."
            ),
    }


def build_experiment_design(
) -> dict[
    str,
    Any,
]:
    return {
        "base_model": {
            "model_family":
                "gemma3",

            "repository":
                "google/gemma-3-4b-it",

            "revision":
                (
                    "093f9f388b31de276ce2de164bdc208"
                    "1324b9767"
                ),

            "reuse_v0_3_base_checkpoint":
                True,
        },

        "dataset_design": {
            "contrastive_grouping_required":
                True,

            "example_count_target":
                EXPECTED_TRAINING_EXAMPLE_COUNT,

            "hard_negative_required":
                True,

            "minimum_distinct_domains":
                12,

            "minimum_examples_per_contrastive_group":
                3,

            "relation_target_counts":
                TRAINING_TARGET_COUNTS,

            "raw_evaluation_evidence_allowed_as_training_source":
                False,

            "synthetic_examples_allowed":
                True,

            "training_source_must_be_independently_authored":
                True,
        },

        "evaluation_design": {
            "final_acceptance": {
                "domain":
                    "commercial_greenhouse_operations",

                "may_be_loaded_before_all_prior_gates_pass":
                    False,

                "remains_closed":
                    True,
            },

            "hotel": {
                "may_be_used_for_training":
                    False,

                "may_be_used_for_v0_4_hyperparameter_tuning":
                    False,

                "may_count_as_new_independent_holdout":
                    False,

                "role":
                    "diagnostic_regression_only",
            },

            "new_independent_holdout": {
                "case_count":
                    V04_HOLDOUT_EXPECTED_CASE_COUNT,

                "cases_per_relation":
                    V04_HOLDOUT_CASES_PER_RELATION,

                "domain":
                    V04_HOLDOUT_DOMAIN,

                "must_be_frozen_before_training_dataset_freeze":
                    True,

                "must_be_frozen_before_training":
                    True,

                "must_not_be_used_for_training":
                    True,

                "must_not_be_used_for_hyperparameter_tuning":
                    True,

                "relations":
                    list(
                        RELATIONS
                    ),

                "status":
                    "required_not_yet_authored",
            },
        },

        "experiment_id":
            QLORA_V04_EXPERIMENT_ID,

        "learning_objective": {
            "primary":
                (
                    "Improve canonical semantic-relation "
                    "decision quality while preserving "
                    "short auditable reasoning."
                ),

            "secondary":
                (
                    "Reduce decision-boundary ambiguity "
                    "through contrastive supervision."
                ),

            "training_loss_is_acceptance_evidence":
                False,
        },

        "model_adaptation": {
            "adaptation_method":
                "qlora",

            "lora_alpha":
                32,

            "lora_dropout":
                0.05,

            "lora_rank":
                16,

            "quantization":
                "nf4_double_quant_bf16",

            "target_strategy":
                "language_model_all_linear",
        },

        "prohibited_actions_before_design_gates": {
            "final_acceptance_load":
                True,

            "final_acceptance_evaluation":
                True,

            "gpu_training":
                True,

            "training_dataset_freeze_before_holdout_freeze":
                True,

            "v0_4_training_before_holdout_freeze":
                True,
        },

        "reason_for_revision": {
            "source_candidate":
                "datalens-semantic-qlora-v0.3",

            "source_diagnostic_signal":
                "negative_signal",

            "source_findings": [
                (
                    "The adapted candidate scored 3/19 "
                    "versus 6/19 for the pinned base."
                ),

                (
                    "All three changed argmax decisions "
                    "were regressions."
                ),

                (
                    "Expected-class margins improved on "
                    "11/19 cases but no incorrect case "
                    "crossed to a correct positive margin."
                ),

                (
                    "same_metric_different_state margins "
                    "improved on 4/5 cases while accuracy "
                    "remained 0/5."
                ),

                (
                    "v0.3 training used natural-language "
                    "reasoning targets rather than direct "
                    "canonical relation supervision."
                ),
            ],
        },

        "relations":
            list(
                RELATIONS
            ),

        "rule_version":
            QLORA_V04_EXPERIMENT_DESIGN_RULE_VERSION,

        "status":
            "design_frozen_before_v0_4_data_authoring",

        "task": {
            "assistant_target_format":
                "strict_json_object",

            "assistant_target_schema": {
                "additional_properties":
                    False,

                "fields": {
                    "reason": {
                        "min_words":
                            6,

                        "max_words":
                            45,

                        "type":
                            "string",
                    },

                    "relation": {
                        "allowed_values":
                            list(
                                RELATIONS
                            ),

                        "type":
                            "string",
                    },
                },

                "required": [
                    "relation",
                    "reason",
                ],
            },

            "canonical_example":
                canonical_training_target_example(),

            "reason_must_not_introduce_unknown_facts":
                True,

            "rule_version":
                QLORA_V04_TASK_SCHEMA_VERSION,

            "supervision_type":
                (
                    "canonical_relation_plus_"
                    "short_reason"
                ),
        },

        "training_strategy": {
            "assistant_only_loss":
                True,

            "early_stopping_from_independent_holdout":
                False,

            "gradient_checkpointing":
                True,

            "initial_epoch_budget":
                2,

            "max_sequence_length":
                256,

            "packing":
                False,

            "sequence_truncation":
                "fail_closed",

            "training_seed":
                42,
        },
    }


def validate_experiment_design(
    design: dict[
        str,
        Any,
    ],
) -> None:
    expected = build_experiment_design()

    if design != expected:
        raise RuntimeError(
            (
                "QLoRA v0.4 experiment design "
                "does not match frozen contract."
            )
        )

    if (
        design[
            "dataset_design"
        ][
            "example_count_target"
        ]
        !=
        230
    ):
        raise RuntimeError(
            "Expected v0.4 target dataset size is 230."
        )

    if (
        set(
            design[
                "relations"
            ]
        )
        !=
        set(
            RELATIONS
        )
    ):
        raise RuntimeError(
            "Canonical relation set mismatch."
        )

    if (
        design[
            "evaluation_design"
        ][
            "hotel"
        ][
            "role"
        ]
        !=
        "diagnostic_regression_only"
    ):
        raise RuntimeError(
            "Hotel role must remain diagnostic only."
        )

    if (
        design[
            "evaluation_design"
        ][
            "hotel"
        ][
            "may_count_as_new_independent_holdout"
        ]
        is not False
    ):
        raise RuntimeError(
            "Hotel cannot count as the v0.4 independent holdout."
        )

    holdout = (
        design[
            "evaluation_design"
        ][
            "new_independent_holdout"
        ]
    )

    if (
        holdout[
            "domain"
        ]
        !=
        "airport_ground_operations"
    ):
        raise RuntimeError(
            "Unexpected v0.4 holdout domain."
        )

    if (
        holdout[
            "case_count"
        ]
        !=
        30
    ):
        raise RuntimeError(
            "v0.4 holdout must contain 30 cases."
        )

    if (
        holdout[
            "cases_per_relation"
        ]
        !=
        6
    ):
        raise RuntimeError(
            "v0.4 holdout must be balanced 6/class."
        )

    if (
        holdout[
            "must_be_frozen_before_training"
        ]
        is not True
    ):
        raise RuntimeError(
            "v0.4 holdout must be frozen before training."
        )

    if (
        holdout[
            "must_be_frozen_before_training_dataset_freeze"
        ]
        is not True
    ):
        raise RuntimeError(
            (
                "v0.4 holdout must be frozen "
                "before training dataset freeze."
            )
        )

    final_acceptance = (
        design[
            "evaluation_design"
        ][
            "final_acceptance"
        ]
    )

    if (
        final_acceptance[
            "remains_closed"
        ]
        is not True
    ):
        raise RuntimeError(
            "Final Acceptance must remain closed."
        )

    if (
        final_acceptance[
            "may_be_loaded_before_all_prior_gates_pass"
        ]
        is not False
    ):
        raise RuntimeError(
            "Final Acceptance loading policy mismatch."
        )

    dataset = design[
        "dataset_design"
    ]

    if (
        dataset[
            "contrastive_grouping_required"
        ]
        is not True
    ):
        raise RuntimeError(
            "Contrastive grouping is required."
        )

    if (
        dataset[
            "hard_negative_required"
        ]
        is not True
    ):
        raise RuntimeError(
            "Hard negatives are required."
        )

    if (
        dataset[
            "raw_evaluation_evidence_allowed_as_training_source"
        ]
        is not False
    ):
        raise RuntimeError(
            "Evaluation evidence cannot become training data."
        )

    task = design[
        "task"
    ]

    if (
        task[
            "assistant_target_format"
        ]
        !=
        "strict_json_object"
    ):
        raise RuntimeError(
            "v0.4 target must use strict JSON."
        )

    required = set(
        task[
            "assistant_target_schema"
        ][
            "required"
        ]
    )

    if required != {
        "relation",
        "reason",
    }:
        raise RuntimeError(
            "Target schema fields mismatch."
        )


def main(
) -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=(
            "validate",
        ),
    )

    args = parser.parse_args()

    if args.mode != "validate":
        raise RuntimeError(
            "Unsupported mode."
        )

    if not DESIGN_PATH.is_file():
        raise FileNotFoundError(
            DESIGN_PATH
        )

    design = load_json_object(
        DESIGN_PATH
    )

    validate_experiment_design(
        design
    )

    print(
        "DATALENS QLORA v0.4 EXPERIMENT DESIGN VALIDATION: PASS"
    )


if __name__ == "__main__":
    main()
