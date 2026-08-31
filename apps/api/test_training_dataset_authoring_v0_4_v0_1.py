from __future__ import annotations

import ast

from collections import Counter, defaultdict
from pathlib import Path


from app.adaptation.training_dataset_authoring_v0_4 import (
    AUTHORING_RULE_VERSION,
    DATASET_PATH,
    EXPECTED_DOMAIN_GROUPS,
    EXPECTED_RELATION_COUNTS,
    FOCUSED_RELATIONS,
    FULL_RELATIONS,
    GROUP_SPECS,
    build_records,
    load_jsonl,
    sha256_file,
    validate_existing_dataset,
    validate_group_specs,
    validate_records,
)


print(
    "=== DATALENS QLORA v0.4 TRAINING DATASET AUTHORING v0.1 ==="
)

print()


assert (
    AUTHORING_RULE_VERSION
    ==
    "qlora_v0.4_training_dataset_authoring_v0.1"
)


validate_group_specs()


assert len(
    GROUP_SPECS
) == 50


full_groups = [
    item
    for item
    in GROUP_SPECS
    if item[
        "group_type"
    ]
    ==
    "full"
]


focused_groups = [
    item
    for item
    in GROUP_SPECS
    if item[
        "group_type"
    ]
    ==
    "focused"
]


assert len(
    full_groups
) == 40


assert len(
    focused_groups
) == 10


print(
    "50 manually authored groups: PASS"
)

print(
    "40 full + 10 focused groups: PASS"
)


records = load_jsonl(
    DATASET_PATH
)


validate_records(
    records
)


assert len(
    records
) == 230


assert (
    records
    ==
    build_records()
)


print(
    "230 deterministic records: PASS"
)


relation_counts = Counter(
    record[
        "target"
    ][
        "relation"
    ]

    for record
    in records
)


assert (
    relation_counts
    ==
    Counter(
        EXPECTED_RELATION_COUNTS
    )
)


print(
    "Relation distribution 50/50/50/40/40: PASS"
)


group_rows = defaultdict(
    list
)


for record in records:
    group_rows[
        record[
            "contrastive_group_id"
        ]
    ].append(
        record
    )


assert len(
    group_rows
) == 50


spec_by_id = {
    spec[
        "group_id"
    ]:
        spec

    for spec
    in GROUP_SPECS
}


for group_id, rows in group_rows.items():
    spec = spec_by_id[
        group_id
    ]

    anchors = {
        row[
            "left_metric"
        ]

        for row
        in rows
    }

    assert len(
        anchors
    ) == 1

    relations = {
        row[
            "target"
        ][
            "relation"
        ]

        for row
        in rows
    }

    if (
        spec[
            "group_type"
        ]
        ==
        "full"
    ):
        assert relations == set(
            FULL_RELATIONS
        )

        assert len(
            rows
        ) == 5

        assert sum(
            row[
                "hard_negative"
            ]

            for row
            in rows
        ) >= 2

    else:
        assert relations == set(
            FOCUSED_RELATIONS
        )

        assert len(
            rows
        ) == 3

        assert sum(
            row[
                "hard_negative"
            ]

            for row
            in rows
        ) >= 1


print(
    "Same-anchor contrastive structure: PASS"
)

print(
    "Hard-negative minimums: PASS"
)


domain_groups = defaultdict(
    set
)


for record in records:
    domain_groups[
        record[
            "domain"
        ]
    ].add(
        record[
            "contrastive_group_id"
        ]
    )


assert len(
    domain_groups
) == 15


for domain, (
    expected_full,
    expected_focused,
) in EXPECTED_DOMAIN_GROUPS.items():
    specs = [
        item

        for item
        in GROUP_SPECS

        if item[
            "domain"
        ]
        ==
        domain
    ]

    assert (
        sum(
            item[
                "group_type"
            ]
            ==
            "full"

            for item
            in specs
        )
        ==
        expected_full
    )

    assert (
        sum(
            item[
                "group_type"
            ]
            ==
            "focused"

            for item
            in specs
        )
        ==
        expected_focused
    )

    assert len(
        domain_groups[
            domain
        ]
    ) >= 3


print(
    "15-domain allocation: PASS"
)


for record in records:
    assert (
        record[
            "provenance"
        ][
            "source_artifact_paths"
        ]
        ==
        []
    )

    assert (
        record[
            "provenance"
        ][
            "source_dataset_ids"
        ]
        ==
        []
    )

    assert len(
        record[
            "provenance"
        ][
            "source_ids"
        ]
    ) == 1

    assert (
        record[
            "provenance"
        ][
            "authoring_method"
        ]
        ==
        "independent_manual_semantic_design"
    )


print(
    "Per-example provenance: PASS"
)


serialized = "\n".join(
    str(
        record
    ).casefold()

    for record
    in records
)


assert "airport" not in serialized
assert "hotel" not in serialized
assert "greenhouse" not in serialized


print(
    "Protected evaluation-domain tokens absent: PASS"
)


example_ids = [
    record[
        "example_id"
    ]

    for record
    in records
]


assert len(
    set(
        example_ids
    )
) == 230


source_ids = [
    record[
        "provenance"
    ][
        "source_ids"
    ][
        0
    ]

    for record
    in records
]


assert len(
    set(
        source_ids
    )
) == 230


print(
    "Unique example/provenance IDs: PASS"
)


validate_existing_dataset()


print(
    "Deterministic dataset recomputation: PASS"
)


source_path = (
    Path(__file__)
    .resolve()
    .parent
    / "app"
    / "adaptation"
    / "training_dataset_authoring_v0_4.py"
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


assert (
    "airport_ground_operations_holdout"
    not in
    source
)


assert (
    "reasoning_benchmark_v0.1_cases"
    not in
    source
)


assert (
    "greenhouse_operations_final_acceptance"
    not in
    source
)


print(
    "No holdout case-file dependency: PASS"
)


print()

print(
    "SAFETY"
)

print(
    "  Airport case content read: False"
)

print(
    "  Hotel case content read: False"
)

print(
    "  Final Acceptance content read: False"
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
    "  Evaluation executed: False"
)

print(
    "  Training executed: False"
)


print()

print(
    f"Dataset SHA256: {sha256_file(DATASET_PATH)}"
)


print()

print(
    "DATALENS QLORA v0.4 TRAINING DATASET AUTHORING v0.1: PASS"
)
