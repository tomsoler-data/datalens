from __future__ import annotations

import ast

from pathlib import Path


from app.adaptation.training_dataset_design_v0_4 import (
    DESIGN_PATH,
    DOMAIN_PLAN,
    EXPECTED_EXAMPLE_COUNT,
    FOCUSED_GROUP_COUNT,
    FORBIDDEN_DOMAINS,
    FULL_GROUP_COUNT,
    RELATION_TARGET_COUNTS,
    RELATIONS,
    TOTAL_GROUP_COUNT,
    TRAINING_DATASET_DESIGN_RULE_VERSION,
    build_dataset_design,
    load_json_object,
    validate_dataset_design,
)


print(
    "=== DATALENS QLORA v0.4 TRAINING DATASET DESIGN v0.1 ==="
)

print()


assert (
    TRAINING_DATASET_DESIGN_RULE_VERSION
    ==
    "qlora_v0.4_training_dataset_design_v0.1"
)


assert (
    EXPECTED_EXAMPLE_COUNT
    ==
    230
)


assert (
    FULL_GROUP_COUNT
    ==
    40
)


assert (
    FOCUSED_GROUP_COUNT
    ==
    10
)


assert (
    TOTAL_GROUP_COUNT
    ==
    50
)


print(
    "Dataset identity / size: PASS"
)

print(
    "40 full + 10 focused groups: PASS"
)


assert (
    RELATION_TARGET_COUNTS
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


assert (
    sum(
        RELATION_TARGET_COUNTS.values()
    )
    ==
    230
)


print(
    "Relation distribution 50/50/50/40/40: PASS"
)


assert len(
    DOMAIN_PLAN
) == 15


assert not (
    set(
        DOMAIN_PLAN
    )
    &
    set(
        FORBIDDEN_DOMAINS
    )
)


assert (
    sum(
        value[
            "full_groups"
        ]

        for value
        in DOMAIN_PLAN.values()
    )
    ==
    40
)


assert (
    sum(
        value[
            "focused_groups"
        ]

        for value
        in DOMAIN_PLAN.values()
    )
    ==
    10
)


assert all(
    (
        value[
            "full_groups"
        ]
        +
        value[
            "focused_groups"
        ]
    )
    >=
    3

    for value
    in DOMAIN_PLAN.values()
)


print(
    "15-domain allocation: PASS"
)

print(
    "Airport/Hotel/Greenhouse excluded: PASS"
)


design = load_json_object(
    DESIGN_PATH
)


validate_dataset_design(
    design
)


assert (
    design
    ==
    build_dataset_design()
)


print(
    "Deterministic design recomputation: PASS"
)


contrastive = design[
    "contrastive_design"
]


assert (
    contrastive[
        "full_group"
    ][
        "examples_per_group"
    ]
    ==
    5
)


assert (
    contrastive[
        "focused_group"
    ][
        "examples_per_group"
    ]
    ==
    3
)


assert (
    contrastive[
        "minimum_hard_negatives_per_full_group"
    ]
    ==
    2
)


assert (
    contrastive[
        "minimum_hard_negatives_per_focused_group"
    ]
    ==
    1
)


assert (
    contrastive[
        "hard_negative_required"
    ]
    is True
)


print(
    "Contrastive / hard-negative contract: PASS"
)


schema = design[
    "record_schema"
]


assert (
    schema[
        "additional_properties"
    ]
    is False
)


assert (
    schema[
        "reason_min_words"
    ]
    ==
    6
)


assert (
    schema[
        "reason_max_words"
    ]
    ==
    45
)


assert (
    schema[
        "target_relations"
    ]
    ==
    list(
        RELATIONS
    )
)


print(
    "Training-record schema: PASS"
)


authoring = design[
    "authoring_policy"
]


assert (
    authoring[
        "airport_cases_may_be_read_during_authoring"
    ]
    is False
)


assert (
    authoring[
        "airport_domain_may_appear_in_training_rows"
    ]
    is False
)


assert (
    authoring[
        "hotel_case_material_may_be_read_during_authoring"
    ]
    is False
)


assert (
    authoring[
        "final_acceptance_material_may_be_read_during_authoring"
    ]
    is False
)


assert (
    authoring[
        "evaluation_artifacts_may_be_used_as_training_sources"
    ]
    is False
)


assert (
    authoring[
        "training_builder_may_bind_holdouts_by_hash_only"
    ]
    is True
)


print(
    "Holdout blindness / isolation policy: PASS"
)


provenance = design[
    "provenance_policy"
]


assert (
    provenance[
        "per_example_provenance_required"
    ]
    is True
)


assert (
    provenance[
        "source_artifact_paths_must_be_empty"
    ]
    is True
)


assert (
    provenance[
        "source_dataset_ids_must_be_empty"
    ]
    is True
)


assert (
    provenance[
        "source_ids_required"
    ]
    is True
)


print(
    "Per-example provenance contract: PASS"
)


contamination = design[
    "contamination_policy"
]


assert (
    contamination[
        "contamination_report_required_before_dataset_freeze"
    ]
    is True
)


assert (
    contamination[
        "near_duplicate_similarity_threshold"
    ]
    ==
    0.92
)


assert (
    contamination[
        "final_acceptance_material_forbidden"
    ]
    is True
)


print(
    "Contamination gate contract: PASS"
)


safety = design[
    "safety"
]


assert (
    safety[
        "airport_holdout_results_may_be_observed"
    ]
    is False
)


assert (
    safety[
        "final_acceptance_may_be_loaded"
    ]
    is False
)


assert (
    safety[
        "training_may_execute"
    ]
    is False
)


print(
    "Pre-authoring safety gates: PASS"
)


source_path = (
    Path(__file__)
    .resolve()
    .parent
    / "app"
    / "adaptation"
    / "training_dataset_design_v0_4.py"
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
    "  Training examples authored: False"
)

print(
    "  Airport cases read by module: False"
)

print(
    "  Airport evaluation executed: False"
)

print(
    "  Airport results observed: False"
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
    "  Final Acceptance loaded: False"
)

print(
    "  Final Acceptance evaluated: False"
)


print()

print(
    "DATALENS QLORA v0.4 TRAINING DATASET DESIGN v0.1: PASS"
)
