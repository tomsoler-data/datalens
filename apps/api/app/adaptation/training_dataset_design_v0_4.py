from __future__ import annotations

import argparse
import hashlib
import json

from pathlib import Path
from typing import Any


TRAINING_DATASET_DESIGN_RULE_VERSION = (
    "qlora_v0.4_training_dataset_design_v0.1"
)


TRAINING_RECORD_SCHEMA_VERSION = (
    "qlora_v0.4_training_record_schema_v0.1"
)


CONTRASTIVE_GROUP_RULE_VERSION = (
    "qlora_v0.4_contrastive_group_v0.1"
)


PROVENANCE_POLICY_VERSION = (
    "qlora_v0.4_training_provenance_policy_v0.1"
)


RELATIONS = (
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
    "unrelated",
    "uncertain",
)


RELATION_TARGET_COUNTS = {
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


FULL_GROUP_RELATIONS = (
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
    "unrelated",
    "uncertain",
)


FOCUSED_GROUP_RELATIONS = (
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
)


FULL_GROUP_COUNT = 40


FOCUSED_GROUP_COUNT = 10


TOTAL_GROUP_COUNT = (
    FULL_GROUP_COUNT
    +
    FOCUSED_GROUP_COUNT
)


EXPECTED_EXAMPLE_COUNT = (
    FULL_GROUP_COUNT
    *
    len(
        FULL_GROUP_RELATIONS
    )
    +
    FOCUSED_GROUP_COUNT
    *
    len(
        FOCUSED_GROUP_RELATIONS
    )
)


FORBIDDEN_DOMAINS = (
    "airport_ground_operations",
    "hotel_operations",
    "commercial_greenhouse_operations",
)


DOMAIN_PLAN = {
    "retail_store_operations": {
        "focused_groups":
            1,

        "full_groups":
            3,
    },

    "subscription_billing_operations": {
        "focused_groups":
            1,

        "full_groups":
            3,
    },

    "telecom_field_service": {
        "focused_groups":
            1,

        "full_groups":
            3,
    },

    "renewable_power_operations": {
        "focused_groups":
            1,

        "full_groups":
            3,
    },

    "food_processing_operations": {
        "focused_groups":
            1,

        "full_groups":
            3,
    },

    "maritime_terminal_operations": {
        "focused_groups":
            1,

        "full_groups":
            2,
    },

    "construction_site_operations": {
        "focused_groups":
            1,

        "full_groups":
            2,
    },

    "insurance_claim_operations": {
        "focused_groups":
            1,

        "full_groups":
            2,
    },

    "media_streaming_operations": {
        "focused_groups":
            1,

        "full_groups":
            2,
    },

    "municipal_waste_operations": {
        "focused_groups":
            1,

        "full_groups":
            2,
    },

    "agricultural_irrigation_operations": {
        "focused_groups":
            0,

        "full_groups":
            3,
    },

    "rail_freight_operations": {
        "focused_groups":
            0,

        "full_groups":
            3,
    },

    "data_center_facility_operations": {
        "focused_groups":
            0,

        "full_groups":
            3,
    },

    "laboratory_sample_operations": {
        "focused_groups":
            0,

        "full_groups":
            3,
    },

    "procurement_operations": {
        "focused_groups":
            0,

        "full_groups":
            3,
    },
}


ROOT = Path(__file__).resolve().parents[2]


DESIGN_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "design"
    / (
        "datalens_semantic_qlora_v0.4_"
        "training_dataset_design_v0.1.json"
    )
)


FREEZE_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "design"
    / (
        "datalens_semantic_qlora_v0.4_"
        "training_dataset_design_v0.1_freeze.json"
    )
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
) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return value


def _validate_domain_plan(
) -> None:
    if len(
        DOMAIN_PLAN
    ) < 12:
        raise RuntimeError(
            "At least 12 training domains are required."
        )

    if (
        set(
            DOMAIN_PLAN
        )
        &
        set(
            FORBIDDEN_DOMAINS
        )
    ):
        raise RuntimeError(
            "Protected evaluation domain appears in training plan."
        )

    full_count = sum(
        config[
            "full_groups"
        ]

        for config
        in DOMAIN_PLAN.values()
    )

    focused_count = sum(
        config[
            "focused_groups"
        ]

        for config
        in DOMAIN_PLAN.values()
    )

    if (
        full_count
        !=
        FULL_GROUP_COUNT
    ):
        raise RuntimeError(
            (
                "Domain plan does not allocate "
                "exactly 40 full groups."
            )
        )

    if (
        focused_count
        !=
        FOCUSED_GROUP_COUNT
    ):
        raise RuntimeError(
            (
                "Domain plan does not allocate "
                "exactly 10 focused groups."
            )
        )

    for domain, config in DOMAIN_PLAN.items():
        group_count = (
            config[
                "full_groups"
            ]
            +
            config[
                "focused_groups"
            ]
        )

        if group_count < 3:
            raise RuntimeError(
                (
                    "Each domain must receive at "
                    f"least 3 contrastive groups: {domain}"
                )
            )


def build_dataset_design(
) -> dict[str, Any]:
    _validate_domain_plan()

    return {
        "authoring_policy": {
            "airport_cases_may_be_read_during_authoring":
                False,

            "airport_domain_may_appear_in_training_rows":
                False,

            "evaluation_artifacts_may_be_used_as_training_sources":
                False,

            "final_acceptance_material_may_be_read_during_authoring":
                False,

            "hotel_case_material_may_be_read_during_authoring":
                False,

            "independent_manual_authoring_required":
                True,

            "training_builder_may_bind_holdouts_by_hash_only":
                True,
        },

        "contamination_policy": {
            "contamination_report_required_before_dataset_freeze":
                True,

            "exact_match_forbidden":
                True,

            "final_acceptance_material_forbidden":
                True,

            "near_duplicate_similarity_threshold":
                0.92,

            "normalized_match_forbidden":
                True,

            "protected_evidence_corpus_required":
                True,

            "structured_record_match_forbidden":
                True,
        },

        "contrastive_design": {
            "focused_group": {
                "count":
                    FOCUSED_GROUP_COUNT,

                "examples_per_group":
                    3,

                "relations":
                    list(
                        FOCUSED_GROUP_RELATIONS
                    ),

                "same_anchor_metric_required":
                    True,
            },

            "full_group": {
                "count":
                    FULL_GROUP_COUNT,

                "examples_per_group":
                    5,

                "relations":
                    list(
                        FULL_GROUP_RELATIONS
                    ),

                "same_anchor_metric_required":
                    True,
            },

            "hard_negative_definition":
                (
                    "A semantically plausible near-neighbor "
                    "pair whose lexical or process similarity "
                    "could support an incorrect neighboring "
                    "relation without careful distinction."
                ),

            "hard_negative_required":
                True,

            "minimum_hard_negatives_per_focused_group":
                1,

            "minimum_hard_negatives_per_full_group":
                2,

            "rule_version":
                CONTRASTIVE_GROUP_RULE_VERSION,

            "total_group_count":
                TOTAL_GROUP_COUNT,
        },

        "dataset_id":
            "adaptation:datalens-semantic:training:v0.4",

        "dataset_version":
            "datalens_semantic_adaptation_training_v0.4",

        "domain_plan":
            DOMAIN_PLAN,

        "expected_example_count":
            EXPECTED_EXAMPLE_COUNT,

        "experiment_id":
            "datalens-semantic-qlora-v0.4",

        "forbidden_domains":
            list(
                FORBIDDEN_DOMAINS
            ),

        "provenance_policy": {
            "authoring_method":
                "independent_manual_semantic_design",

            "evaluation_case_ids_forbidden":
                True,

            "evaluation_dataset_ids_forbidden":
                True,

            "evaluation_source_paths_forbidden":
                True,

            "per_example_provenance_required":
                True,

            "protected_source_material_may_not_be_paraphrased":
                True,

            "rule_version":
                PROVENANCE_POLICY_VERSION,

            "source_artifact_paths_must_be_empty":
                True,

            "source_dataset_ids_must_be_empty":
                True,

            "source_ids_required":
                True,
        },

        "record_schema": {
            "additional_properties":
                False,

            "fields": {
                "contrastive_group_id":
                    "string",

                "domain":
                    "string",

                "example_id":
                    "string",

                "hard_negative":
                    "boolean",

                "left_description":
                    "string",

                "left_metric":
                    "string",

                "provenance": {
                    "authoring_method":
                        "string",

                    "source_artifact_paths":
                        "list[string]",

                    "source_dataset_ids":
                        "list[string]",

                    "source_ids":
                        "list[string]",
                },

                "right_description":
                    "string",

                "right_metric":
                    "string",

                "target": {
                    "reason":
                        "string",

                    "relation":
                        "enum",
                },
            },

            "reason_max_words":
                45,

            "reason_min_words":
                6,

            "required_fields": [
                "contrastive_group_id",
                "domain",
                "example_id",
                "hard_negative",
                "left_description",
                "left_metric",
                "provenance",
                "right_description",
                "right_metric",
                "target",
            ],

            "rule_version":
                TRAINING_RECORD_SCHEMA_VERSION,

            "target_relations":
                list(
                    RELATIONS
                ),
        },

        "relation_target_counts":
            RELATION_TARGET_COUNTS,

        "relations":
            list(
                RELATIONS
            ),

        "rule_version":
            TRAINING_DATASET_DESIGN_RULE_VERSION,

        "safety": {
            "airport_holdout_results_may_be_observed":
                False,

            "final_acceptance_may_be_loaded":
                False,

            "gpu_required":
                False,

            "training_may_execute":
                False,
        },

        "status":
            "frozen_before_training_example_authoring",
    }


def validate_dataset_design(
    design: dict[str, Any],
) -> None:
    expected = build_dataset_design()

    if design != expected:
        raise RuntimeError(
            "Training dataset design differs from frozen contract."
        )

    if (
        EXPECTED_EXAMPLE_COUNT
        !=
        230
    ):
        raise RuntimeError(
            "Training example total must equal 230."
        )

    if (
        TOTAL_GROUP_COUNT
        !=
        50
    ):
        raise RuntimeError(
            "Contrastive group total must equal 50."
        )

    if (
        RELATION_TARGET_COUNTS
        !=
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
    ):
        raise RuntimeError(
            "Relation distribution mismatch."
        )

    if (
        sum(
            RELATION_TARGET_COUNTS.values()
        )
        !=
        EXPECTED_EXAMPLE_COUNT
    ):
        raise RuntimeError(
            "Relation counts do not sum to dataset size."
        )

    _validate_domain_plan()

    authoring = design[
        "authoring_policy"
    ]

    for key in (
        "airport_cases_may_be_read_during_authoring",
        "airport_domain_may_appear_in_training_rows",
        "evaluation_artifacts_may_be_used_as_training_sources",
        "final_acceptance_material_may_be_read_during_authoring",
        "hotel_case_material_may_be_read_during_authoring",
    ):
        if (
            authoring[
                key
            ]
            is not False
        ):
            raise RuntimeError(
                (
                    "Training authoring isolation "
                    f"violation: {key}"
                )
            )

    if (
        authoring[
            "training_builder_may_bind_holdouts_by_hash_only"
        ]
        is not True
    ):
        raise RuntimeError(
            "Holdout binding must remain hash-only."
        )

    provenance = design[
        "provenance_policy"
    ]

    for key in (
        "evaluation_case_ids_forbidden",
        "evaluation_dataset_ids_forbidden",
        "evaluation_source_paths_forbidden",
        "per_example_provenance_required",
        "protected_source_material_may_not_be_paraphrased",
        "source_artifact_paths_must_be_empty",
        "source_dataset_ids_must_be_empty",
        "source_ids_required",
    ):
        if (
            provenance[
                key
            ]
            is not True
        ):
            raise RuntimeError(
                (
                    "Training provenance policy "
                    f"violation: {key}"
                )
            )

    contamination = design[
        "contamination_policy"
    ]

    if (
        contamination[
            "near_duplicate_similarity_threshold"
        ]
        !=
        0.92
    ):
        raise RuntimeError(
            "Contamination threshold changed."
        )

    for key in (
        "contamination_report_required_before_dataset_freeze",
        "exact_match_forbidden",
        "final_acceptance_material_forbidden",
        "normalized_match_forbidden",
        "protected_evidence_corpus_required",
        "structured_record_match_forbidden",
    ):
        if (
            contamination[
                key
            ]
            is not True
        ):
            raise RuntimeError(
                (
                    "Contamination policy "
                    f"violation: {key}"
                )
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

    validate_dataset_design(
        design
    )

    print(
        "DATALENS QLORA v0.4 TRAINING DATASET DESIGN VALIDATION: PASS"
    )


if __name__ == "__main__":
    main()
