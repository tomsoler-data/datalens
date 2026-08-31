from __future__ import annotations

import ast
import hashlib
import inspect
import json

from collections import Counter
from pathlib import Path


print(
    "=== DATALENS QLORA v0.4 TRAINING DATASET CANONICALIZER TEST v0.1 ==="
)

print()


ROOT = Path.cwd().resolve()


from app.adaptation import adaptation_dataset

from app.adaptation.training_dataset_canonicalizer_v0_4 import (
    AUTHORING_RELATIVE_PATH,
    EXPECTED_AUTHORING_SHA256,
    EXPECTED_CANONICAL_SHA256,
    EXPECTED_EXAMPLE_COUNT,
    EXPECTED_RELATION_COUNTS,
    PROTECTED_DOMAIN_TOKENS,
    QLORA_V0_4_CANONICALIZER_RULE_VERSION,
    RELATIONS,
    build_canonical_examples,
    canonical_dataset_bytes,
    canonical_dataset_sha256,
    load_authoring_records,
)


EXPECTED_RULE_VERSION = (
    "qlora_v0.4_training_dataset_canonicalizer_v0.1"
)


if (
    QLORA_V0_4_CANONICALIZER_RULE_VERSION
    !=
    EXPECTED_RULE_VERSION
):
    raise RuntimeError(
        "Canonicalizer rule version mismatch."
    )


print(
    "Canonicalizer rule version: PASS"
)


authoring_path = (
    ROOT
    /
    AUTHORING_RELATIVE_PATH
)


if not authoring_path.is_file():
    raise FileNotFoundError(
        authoring_path
    )


actual_authoring_sha256 = hashlib.sha256(
    authoring_path.read_bytes()
).hexdigest()


if (
    actual_authoring_sha256
    !=
    EXPECTED_AUTHORING_SHA256
):
    raise RuntimeError(
        "Frozen authoring SHA mismatch."
    )


print(
    "Frozen authoring binding: PASS"
)


records = load_authoring_records(
    repository_root=
        ROOT,
)


if len(
    records
) != EXPECTED_EXAMPLE_COUNT:
    raise RuntimeError(
        "Authoring count mismatch."
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


if (
    relation_counts
    !=
    Counter(
        EXPECTED_RELATION_COUNTS
    )
):
    raise RuntimeError(
        "Authoring relation distribution mismatch."
    )


print(
    "230 authoring records: PASS"
)

print(
    "Relation distribution 50/50/50/40/40: PASS"
)


serialized_authoring = (
    authoring_path
    .read_text(
        encoding="utf-8"
    )
    .casefold()
)


for token in PROTECTED_DOMAIN_TOKENS:
    if token in serialized_authoring:
        raise RuntimeError(
            (
                "Protected domain token found: "
                f"{token}"
            )
        )


print(
    "Protected domain isolation: PASS"
)


examples_a = (
    build_canonical_examples(
        repository_root=
            ROOT,
    )
)


examples_b = (
    build_canonical_examples(
        repository_root=
            ROOT,
    )
)


if len(
    examples_a
) != EXPECTED_EXAMPLE_COUNT:
    raise RuntimeError(
        "Canonical example count mismatch."
    )


if examples_a != examples_b:
    raise RuntimeError(
        "Canonical object recomputation mismatch."
    )


print(
    "230 canonical examples: PASS"
)

print(
    "Deterministic object recomputation: PASS"
)


if len(
    {
        example.example_id

        for example
        in examples_a
    }
) != EXPECTED_EXAMPLE_COUNT:
    raise RuntimeError(
        "Canonical example IDs are not unique."
    )


source_ids = []


for example in examples_a:
    if (
        len(
            example.messages
        )
        !=
        2
    ):
        raise RuntimeError(
            "Canonical chat must contain two messages."
        )

    if (
        example.messages[
            0
        ].role
        !=
        "user"
        or
        example.messages[
            1
        ].role
        !=
        "assistant"
    ):
        raise RuntimeError(
            "Canonical chat shape mismatch."
        )

    assistant_payload = json.loads(
        example.messages[
            1
        ].content
    )

    if (
        tuple(
            sorted(
                assistant_payload
            )
        )
        !=
        (
            "reason",
            "relation",
        )
    ):
        raise RuntimeError(
            "Assistant JSON schema mismatch."
        )

    relation = assistant_payload[
        "relation"
    ]

    if relation not in RELATIONS:
        raise RuntimeError(
            "Assistant JSON relation invalid."
        )

    reason_words = len(
        assistant_payload[
            "reason"
        ].split()
    )

    if not (
        6
        <=
        reason_words
        <=
        45
    ):
        raise RuntimeError(
            "Assistant reason length contract failed."
        )

    if (
        example.provenance.origin
        !=
        "independently_authored"
    ):
        raise RuntimeError(
            "Canonical provenance origin mismatch."
        )

    if (
        example.provenance.source_dataset_ids
        !=
        ()
    ):
        raise RuntimeError(
            "Canonical source_dataset_ids not empty."
        )

    if (
        example.provenance.source_artifact_paths
        !=
        ()
    ):
        raise RuntimeError(
            "Canonical source_artifact_paths not empty."
        )

    if len(
        example.provenance.source_ids
    ) != 1:
        raise RuntimeError(
            "Canonical source_ids cardinality mismatch."
        )

    source_ids.extend(
        example.provenance.source_ids
    )


if (
    len(
        set(
            source_ids
        )
    )
    !=
    EXPECTED_EXAMPLE_COUNT
):
    raise RuntimeError(
        "Canonical source IDs are not unique."
    )


print(
    "User -> assistant shape: PASS"
)

print(
    "Strict relation/reason JSON: PASS"
)

print(
    "Reason length contract: PASS"
)

print(
    "Independent provenance: PASS"
)

print(
    "Unique example/provenance IDs: PASS"
)


bytes_a = canonical_dataset_bytes(
    repository_root=
        ROOT,
)


bytes_b = canonical_dataset_bytes(
    repository_root=
        ROOT,
)


if bytes_a != bytes_b:
    raise RuntimeError(
        "Canonical byte recomputation mismatch."
    )


actual_canonical_sha256 = hashlib.sha256(
    bytes_a
).hexdigest()


if (
    actual_canonical_sha256
    !=
    EXPECTED_CANONICAL_SHA256
):
    raise RuntimeError(
        (
            "Canonical SHA mismatch.\n"
            f"Expected: {EXPECTED_CANONICAL_SHA256}\n"
            f"Actual:   {actual_canonical_sha256}"
        )
    )


if (
    canonical_dataset_sha256(
        repository_root=
            ROOT,
    )
    !=
    EXPECTED_CANONICAL_SHA256
):
    raise RuntimeError(
        "Canonical SHA helper mismatch."
    )


# Verify that the public result is exactly the same
# serialization primitive used by the adaptation framework.
framework_bytes = (
    adaptation_dataset._dataset_jsonl_bytes(
        examples_a
    )
)


if framework_bytes != bytes_a:
    raise RuntimeError(
        "Framework serialization mismatch."
    )


print(
    "Canonical framework serialization: PASS"
)

print(
    "Deterministic byte recomputation: PASS"
)

print(
    (
        "Canonical SHA256: "
        f"{actual_canonical_sha256}"
    )
)


# ------------------------------------------------------------
# STATIC SAFETY
# ------------------------------------------------------------


canonicalizer_module = __import__(
    (
        "app.adaptation."
        "training_dataset_canonicalizer_v0_4"
    ),
    fromlist=[
        "*",
    ],
)


module_source = inspect.getsource(
    canonicalizer_module
)


tree = ast.parse(
    module_source
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
            if (
                alias.name.split(
                    "."
                )[0]
                in
                heavy_modules
            ):
                raise RuntimeError(
                    (
                        "Heavy ML import detected: "
                        f"{alias.name}"
                    )
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

        if (
            module.split(
                "."
            )[0]
            in
            heavy_modules
        ):
            raise RuntimeError(
                (
                    "Heavy ML import detected: "
                    f"{module}"
                )
            )


for forbidden_dependency in (
    "airport_ground_operations_holdout",
    "adapted_reasoning_benchmark",
    "greenhouse_operations_final_acceptance",
):
    if forbidden_dependency in module_source.casefold():
        raise RuntimeError(
            (
                "Canonicalizer must not depend on "
                "protected evaluation definitions: "
                f"{forbidden_dependency}"
            )
        )


print()

print(
    "SAFETY"
)

print(
    "  Heavy ML imports: False"
)

print(
    "  Holdout definition dependencies: False"
)

print(
    "  Canonical dataset written: False"
)

print(
    "  Airport evaluated: False"
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
    "  Evaluation executed: False"
)

print(
    "  Training executed: False"
)

print(
    "  Final Acceptance evaluated: False"
)


print()

print(
    "DATALENS QLORA v0.4 TRAINING DATASET CANONICALIZER TEST v0.1: PASS"
)
