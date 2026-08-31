from __future__ import annotations

import argparse
import hashlib
import json

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from app.adaptation import adaptation_dataset


QLORA_V0_4_CANONICALIZER_RULE_VERSION = (
    "qlora_v0.4_training_dataset_canonicalizer_v0.1"
)


DATASET_ID = (
    "adaptation:datalens-semantic:training:v0.4"
)


DATASET_VERSION = (
    "datalens_semantic_adaptation_training_v0.4"
)


AUTHORING_RELATIVE_PATH = (
    "artifacts/adaptation/datasets/"
    "datalens_semantic_training_v0.4_"
    "authoring_v0.1.jsonl"
)


CANONICAL_RELATIVE_PATH = (
    "artifacts/adaptation/datasets/"
    "datalens_semantic_training_v0.4.jsonl"
)


EXPECTED_AUTHORING_SHA256 = (
    "d7bc77768160363160d96097e3b7df92"
    "4b6fc6302039dc5f5bb289b9da2ba90a"
)


EXPECTED_CANONICAL_SHA256 = (
    "4fd00586f2d53d6de57f5cbc5f1d7bfb"
    "2e512960e60b30c28596aaefbac322b7"
)


EXPECTED_EXAMPLE_COUNT = 230


RELATIONS = (
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
    "unrelated",
    "uncertain",
)


EXPECTED_RELATION_COUNTS = {
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


PROTECTED_DOMAIN_TOKENS = (
    "airport",
    "hotel",
    "greenhouse",
)


def _sha256_bytes(
    value: bytes,
) -> str:
    return hashlib.sha256(
        value
    ).hexdigest()


def _sha256_file(
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


def _repository_root(
    repository_root: Path | None,
) -> Path:
    root = (
        Path.cwd()
        if repository_root is None
        else repository_root
    )

    root = (
        root
        .expanduser()
        .resolve()
    )

    if not root.is_dir():
        raise NotADirectoryError(
            root
        )

    return root


def authoring_dataset_path(
    *,
    repository_root: Path,
) -> Path:
    return (
        repository_root
        /
        AUTHORING_RELATIVE_PATH
    ).resolve()


def canonical_dataset_path(
    *,
    repository_root: Path,
) -> Path:
    return (
        repository_root
        /
        CANONICAL_RELATIVE_PATH
    ).resolve()


def load_authoring_records(
    *,
    repository_root: Path,
) -> Tuple[
    Mapping[
        str,
        Any,
    ],
    ...,
]:
    path = authoring_dataset_path(
        repository_root=
            repository_root,
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    actual_sha256 = _sha256_file(
        path
    )

    if (
        actual_sha256
        !=
        EXPECTED_AUTHORING_SHA256
    ):
        raise RuntimeError(
            (
                "QLoRA v0.4 authoring dataset SHA "
                "does not match its frozen identity.\n"
                f"Expected: {EXPECTED_AUTHORING_SHA256}\n"
                f"Actual:   {actual_sha256}"
            )
        )

    serialized = path.read_text(
        encoding="utf-8"
    )

    lowered = serialized.casefold()

    for token in PROTECTED_DOMAIN_TOKENS:
        if token in lowered:
            raise RuntimeError(
                (
                    "Protected evaluation-domain token "
                    "found in the authoring dataset: "
                    f"{token}"
                )
            )

    records = []

    for line_number, line in enumerate(
        serialized.splitlines(),
        start=1,
    ):
        if not line.strip():
            raise RuntimeError(
                (
                    "Blank line in QLoRA v0.4 "
                    "authoring JSONL: "
                    f"{line_number}"
                )
            )

        value = json.loads(
            line
        )

        if not isinstance(
            value,
            dict,
        ):
            raise RuntimeError(
                (
                    "Authoring JSONL record must be "
                    "an object. Line: "
                    f"{line_number}"
                )
            )

        records.append(
            value
        )

    validate_authoring_records(
        records
    )

    return tuple(
        records
    )


def validate_authoring_records(
    records: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> None:
    if (
        len(
            records
        )
        !=
        EXPECTED_EXAMPLE_COUNT
    ):
        raise RuntimeError(
            (
                "QLoRA v0.4 authoring example count "
                "changed.\n"
                f"Expected: {EXPECTED_EXAMPLE_COUNT}\n"
                f"Actual:   {len(records)}"
            )
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
            (
                "QLoRA v0.4 relation distribution "
                "changed.\n"
                f"Actual: {relation_counts}"
            )
        )

    example_ids = []

    provenance_ids = []

    groups = set()

    domains = set()

    for index, record in enumerate(
        records,
        start=1,
    ):
        required = {
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
        }

        if (
            set(
                record
            )
            !=
            required
        ):
            raise RuntimeError(
                (
                    "Unexpected authoring record schema "
                    f"at index {index}."
                )
            )

        example_id = record[
            "example_id"
        ]

        if not isinstance(
            example_id,
            str,
        ):
            raise RuntimeError(
                "example_id must be a string."
            )

        example_ids.append(
            example_id
        )

        groups.add(
            record[
                "contrastive_group_id"
            ]
        )

        domains.add(
            record[
                "domain"
            ]
        )

        relation = record[
            "target"
        ][
            "relation"
        ]

        if relation not in RELATIONS:
            raise RuntimeError(
                (
                    "Unexpected semantic relation: "
                    f"{relation}"
                )
            )

        reason = record[
            "target"
        ][
            "reason"
        ]

        if not isinstance(
            reason,
            str,
        ):
            raise RuntimeError(
                "Target reason must be a string."
            )

        word_count = len(
            reason.split()
        )

        if not (
            6
            <=
            word_count
            <=
            45
        ):
            raise RuntimeError(
                (
                    "Target reason violates the "
                    "6-45 word contract for "
                    f"{example_id}: {word_count}"
                )
            )

        provenance = record[
            "provenance"
        ]

        if (
            provenance[
                "authoring_method"
            ]
            !=
            "independent_manual_semantic_design"
        ):
            raise RuntimeError(
                (
                    "Unexpected authoring method for "
                    f"{example_id}."
                )
            )

        if (
            provenance[
                "source_artifact_paths"
            ]
            !=
            []
        ):
            raise RuntimeError(
                (
                    "source_artifact_paths must be "
                    "empty for "
                    f"{example_id}."
                )
            )

        if (
            provenance[
                "source_dataset_ids"
            ]
            !=
            []
        ):
            raise RuntimeError(
                (
                    "source_dataset_ids must be "
                    "empty for "
                    f"{example_id}."
                )
            )

        source_ids = provenance[
            "source_ids"
        ]

        if (
            not isinstance(
                source_ids,
                list,
            )
            or
            len(
                source_ids
            )
            !=
            1
            or
            not isinstance(
                source_ids[
                    0
                ],
                str,
            )
        ):
            raise RuntimeError(
                (
                    "Exactly one source_id is required "
                    "for "
                    f"{example_id}."
                )
            )

        provenance_ids.append(
            source_ids[
                0
            ]
        )

    if (
        len(
            set(
                example_ids
            )
        )
        !=
        EXPECTED_EXAMPLE_COUNT
    ):
        raise RuntimeError(
            "QLoRA v0.4 example IDs are not unique."
        )

    if (
        len(
            set(
                provenance_ids
            )
        )
        !=
        EXPECTED_EXAMPLE_COUNT
    ):
        raise RuntimeError(
            "QLoRA v0.4 provenance IDs are not unique."
        )

    if len(
        groups
    ) != 50:
        raise RuntimeError(
            (
                "Expected 50 contrastive groups; "
                f"got {len(groups)}."
            )
        )

    if len(
        domains
    ) != 15:
        raise RuntimeError(
            (
                "Expected 15 training domains; "
                f"got {len(domains)}."
            )
        )


def build_user_message(
    record: Mapping[
        str,
        Any,
    ],
) -> str:
    return "\n".join(
        (
            (
                "Classify the semantic relationship "
                "between two tabular metrics."
            ),

            (
                "Allowed relations: "
                +
                ", ".join(
                    RELATIONS
                )
                +
                "."
            ),

            (
                "Use uncertain when the definitions "
                "do not support a safe classification."
            ),

            (
                "Return exactly one JSON object with "
                "the keys relation and reason."
            ),

            (
                "The reason must contain 6 to 45 words "
                "and use only the supplied definitions."
            ),

            (
                "Domain: "
                f"{record['domain']}"
            ),

            (
                "Metric A name: "
                f"{record['left_metric']}"
            ),

            (
                "Metric A definition: "
                f"{record['left_description']}"
            ),

            (
                "Metric B name: "
                f"{record['right_metric']}"
            ),

            (
                "Metric B definition: "
                f"{record['right_description']}"
            ),
        )
    )


def build_assistant_message(
    record: Mapping[
        str,
        Any,
    ],
) -> str:
    target = record[
        "target"
    ]

    return json.dumps(
        {
            "relation":
                target[
                    "relation"
                ],

            "reason":
                target[
                    "reason"
                ],
        },
        ensure_ascii=True,
        separators=(
            ",",
            ":",
        ),
    )


def canonicalize_record(
    record: Mapping[
        str,
        Any,
    ],
) -> adaptation_dataset.AdaptationTrainingExample:
    provenance = record[
        "provenance"
    ]

    return (
        adaptation_dataset.AdaptationTrainingExample(
            example_id=
                record[
                    "example_id"
                ],

            messages=(
                adaptation_dataset.AdaptationMessage(
                    role=
                        "user",

                    content=
                        build_user_message(
                            record
                        ),
                ),

                adaptation_dataset.AdaptationMessage(
                    role=
                        "assistant",

                    content=
                        build_assistant_message(
                            record
                        ),
                ),
            ),

            provenance=(
                adaptation_dataset.AdaptationExampleProvenance(
                    origin=
                        "independently_authored",

                    source_ids=
                        tuple(
                            provenance[
                                "source_ids"
                            ]
                        ),

                    source_dataset_ids=
                        (),

                    source_artifact_paths=
                        (),
                )
            ),

            tags=(
                "semantic_relation_reasoning",
                "qlora_v0.4",
                record[
                    "domain"
                ],
                record[
                    "contrastive_group_id"
                ],
                record[
                    "target"
                ][
                    "relation"
                ],
                (
                    "hard_negative"
                    if
                    record[
                        "hard_negative"
                    ]
                    else
                    "non_hard_negative"
                ),
            ),
        )
    )


def build_canonical_examples(
    *,
    repository_root: Path,
) -> Tuple[
    adaptation_dataset.AdaptationTrainingExample,
    ...,
]:
    records = load_authoring_records(
        repository_root=
            repository_root,
    )

    examples = tuple(
        canonicalize_record(
            record
        )

        for record
        in records
    )

    if (
        len(
            examples
        )
        !=
        EXPECTED_EXAMPLE_COUNT
    ):
        raise RuntimeError(
            "Canonical example count mismatch."
        )

    if (
        len(
            {
                example.example_id

                for example
                in examples
            }
        )
        !=
        EXPECTED_EXAMPLE_COUNT
    ):
        raise RuntimeError(
            "Canonical example IDs are not unique."
        )

    for example in examples:
        if (
            len(
                example.messages
            )
            !=
            2
        ):
            raise RuntimeError(
                (
                    "Canonical example must contain "
                    "exactly two messages: "
                    f"{example.example_id}"
                )
            )

        if (
            example.messages[
                0
            ].role
            !=
            "user"
        ):
            raise RuntimeError(
                (
                    "First canonical message must be "
                    "user for "
                    f"{example.example_id}."
                )
            )

        if (
            example.messages[
                1
            ].role
            !=
            "assistant"
        ):
            raise RuntimeError(
                (
                    "Second canonical message must be "
                    "assistant for "
                    f"{example.example_id}."
                )
            )

        assistant_payload = json.loads(
            example.messages[
                1
            ].content
        )

        if (
            set(
                assistant_payload
            )
            !=
            {
                "relation",
                "reason",
            }
        ):
            raise RuntimeError(
                (
                    "Assistant target must contain "
                    "exactly relation and reason."
                )
            )

        if (
            assistant_payload[
                "relation"
            ]
            not in
            RELATIONS
        ):
            raise RuntimeError(
                (
                    "Invalid assistant relation for "
                    f"{example.example_id}."
                )
            )

        if (
            example.provenance.origin
            !=
            "independently_authored"
        ):
            raise RuntimeError(
                (
                    "Canonical provenance origin "
                    "mismatch for "
                    f"{example.example_id}."
                )
            )

        if (
            example.provenance.source_dataset_ids
            !=
            ()
        ):
            raise RuntimeError(
                (
                    "Canonical source_dataset_ids must "
                    "be empty."
                )
            )

        if (
            example.provenance.source_artifact_paths
            !=
            ()
        ):
            raise RuntimeError(
                (
                    "Canonical source_artifact_paths "
                    "must be empty."
                )
            )

    return examples


def canonical_dataset_bytes(
    *,
    repository_root: Path,
) -> bytes:
    examples = build_canonical_examples(
        repository_root=
            repository_root,
    )

    payload = (
        adaptation_dataset._dataset_jsonl_bytes(
            examples
        )
    )

    actual_sha256 = _sha256_bytes(
        payload
    )

    if (
        actual_sha256
        !=
        EXPECTED_CANONICAL_SHA256
    ):
        raise RuntimeError(
            (
                "Canonical QLoRA v0.4 dataset SHA "
                "does not match the preregistered "
                "governance-gate identity.\n"
                f"Expected: {EXPECTED_CANONICAL_SHA256}\n"
                f"Actual:   {actual_sha256}"
            )
        )

    return payload


def canonical_dataset_sha256(
    *,
    repository_root: Path,
) -> str:
    return _sha256_bytes(
        canonical_dataset_bytes(
            repository_root=
                repository_root,
        )
    )


def validate_canonicalizer(
    *,
    repository_root: Path,
) -> None:
    repository_root = (
        repository_root
        .expanduser()
        .resolve()
    )

    records = load_authoring_records(
        repository_root=
            repository_root,
    )

    first_examples = (
        build_canonical_examples(
            repository_root=
                repository_root,
        )
    )

    second_examples = (
        build_canonical_examples(
            repository_root=
                repository_root,
        )
    )

    first_bytes = (
        adaptation_dataset._dataset_jsonl_bytes(
            first_examples
        )
    )

    second_bytes = (
        adaptation_dataset._dataset_jsonl_bytes(
            second_examples
        )
    )

    if first_bytes != second_bytes:
        raise RuntimeError(
            "Canonicalization is not deterministic."
        )

    actual_sha256 = _sha256_bytes(
        first_bytes
    )

    if (
        actual_sha256
        !=
        EXPECTED_CANONICAL_SHA256
    ):
        raise RuntimeError(
            (
                "Canonical dataset SHA mismatch.\n"
                f"Expected: {EXPECTED_CANONICAL_SHA256}\n"
                f"Actual:   {actual_sha256}"
            )
        )

    print(
        "=== DATALENS QLORA v0.4 TRAINING DATASET CANONICALIZER v0.1 ==="
    )

    print()

    print(
        f"Authoring records: {len(records)}"
    )

    print(
        f"Canonical examples: {len(first_examples)}"
    )

    print(
        "User -> assistant shape: PASS"
    )

    print(
        "Structured JSON targets: PASS"
    )

    print(
        "Independent provenance conversion: PASS"
    )

    print(
        "Protected evaluation-domain tokens absent: PASS"
    )

    print(
        "Deterministic recomputation: PASS"
    )

    print(
        (
            "Authoring SHA256: "
            f"{EXPECTED_AUTHORING_SHA256}"
        )
    )

    print(
        (
            "Canonical SHA256: "
            f"{actual_sha256}"
        )
    )

    print()

    print(
        "SAFETY"
    )

    print(
        "  Canonical dataset written: False"
    )

    print(
        "  Holdout case content read: False"
    )

    print(
        "  Airport evaluated: False"
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
        "DATALENS QLORA v0.4 TRAINING DATASET CANONICALIZER v0.1: PASS"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=(
            "validate",
        ),
    )

    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
    )

    arguments = parser.parse_args()

    if arguments.command == "validate":
        validate_canonicalizer(
            repository_root=
                _repository_root(
                    arguments.repository_root
                ),
        )

        return

    raise RuntimeError(
        "Unsupported command."
    )


if __name__ == "__main__":
    main()
