from __future__ import annotations

import hashlib
import json

from collections import Counter
from pathlib import Path
from typing import Any


AIRPORT_HOLDOUT_RULE_VERSION = (
    "qlora_v0.4_airport_holdout_v0.1"
)


AIRPORT_HOLDOUT_FREEZE_RULE_VERSION = (
    "qlora_v0.4_airport_holdout_freeze_v0.1"
)


AIRPORT_HOLDOUT_ID = (
    "adaptation:datalens-semantic-qlora-v0.4:"
    "airport-ground-operations:holdout:v0.1"
)


RELATIONS = (
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
    "unrelated",
    "uncertain",
)


ROOT = Path(__file__).resolve().parents[2]


CASES_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "holdouts"
    / (
        "datalens_semantic_qlora_v0.4_"
        "airport_ground_operations_holdout_v0.1_cases.json"
    )
)


FREEZE_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "holdouts"
    / (
        "datalens_semantic_qlora_v0.4_"
        "airport_ground_operations_holdout_v0.1_freeze.json"
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


def validate_cases(
    payload: dict[str, Any],
) -> None:
    if (
        payload[
            "rule_version"
        ]
        !=
        AIRPORT_HOLDOUT_RULE_VERSION
    ):
        raise RuntimeError(
            "Airport holdout rule mismatch."
        )

    if (
        payload[
            "holdout_id"
        ]
        !=
        AIRPORT_HOLDOUT_ID
    ):
        raise RuntimeError(
            "Airport holdout identity mismatch."
        )

    if (
        payload[
            "domain"
        ]
        !=
        "airport_ground_operations"
    ):
        raise RuntimeError(
            "Airport holdout domain mismatch."
        )

    if (
        payload[
            "case_count"
        ]
        !=
        30
    ):
        raise RuntimeError(
            "Airport holdout case count mismatch."
        )

    cases = payload[
        "cases"
    ]

    if len(
        cases
    ) != 30:
        raise RuntimeError(
            "Airport holdout must have 30 cases."
        )

    case_ids = [
        case[
            "case_id"
        ]

        for case
        in cases
    ]

    if len(
        set(
            case_ids
        )
    ) != 30:
        raise RuntimeError(
            "Airport holdout case IDs are not unique."
        )

    counts = Counter(
        case[
            "expected_relation"
        ]

        for case
        in cases
    )

    expected_counts = Counter(
        {
            relation:
                6

            for relation
            in RELATIONS
        }
    )

    if counts != expected_counts:
        raise RuntimeError(
            "Airport holdout is not balanced 6/class."
        )

    if (
        payload[
            "relation_counts"
        ]
        !=
        {
            relation:
                6

            for relation
            in RELATIONS
        }
    ):
        raise RuntimeError(
            "Stored relation counts mismatch."
        )

    if (
        payload[
            "relations"
        ]
        !=
        list(
            RELATIONS
        )
    ):
        raise RuntimeError(
            "Relation order/set mismatch."
        )

    required_case_fields = {
        "case_id",
        "left_metric",
        "left_description",
        "right_metric",
        "right_description",
        "expected_relation",
    }

    pair_keys = set()

    for case in cases:
        if (
            set(
                case
            )
            !=
            required_case_fields
        ):
            raise RuntimeError(
                (
                    "Unexpected case schema: "
                    f"{case.get('case_id')}"
                )
            )

        if (
            case[
                "expected_relation"
            ]
            not in
            RELATIONS
        ):
            raise RuntimeError(
                "Unknown relation."
            )

        if len(
            case[
                "left_description"
            ]
        ) < 40:
            raise RuntimeError(
                "Left description is too short."
            )

        if len(
            case[
                "right_description"
            ]
        ) < 40:
            raise RuntimeError(
                "Right description is too short."
            )

        pair_key = tuple(
            sorted(
                (
                    case[
                        "left_metric"
                    ].casefold(),

                    case[
                        "right_metric"
                    ].casefold(),
                )
            )
        )

        if pair_key in pair_keys:
            raise RuntimeError(
                "Duplicate metric pair."
            )

        pair_keys.add(
            pair_key
        )

    policy = payload[
        "independence_policy"
    ]

    for key in (
        "airport_domain_forbidden_in_v0_4_training_data",
        "case_material_forbidden_in_training",
        "evaluation_results_may_not_drive_hyperparameter_tuning",
        "if_results_are_observed_then_future_experiments_require_new_holdout",
        "must_not_be_used_for_training",
        "must_not_be_used_for_training_data_authoring",
    ):
        if (
            policy[
                key
            ]
            is not True
        ):
            raise RuntimeError(
                (
                    "Independent holdout policy "
                    f"violation: {key}"
                )
            )

    protocol = payload[
        "protocol"
    ]

    if (
        protocol[
            "generation"
        ]
        !=
        "deterministic_greedy"
    ):
        raise RuntimeError(
            "Holdout generation protocol mismatch."
        )

    if (
        protocol[
            "decoding"
        ][
            "do_sample"
        ]
        is not False
    ):
        raise RuntimeError(
            "Holdout decoding must be deterministic."
        )

    if (
        protocol[
            "evaluation_order"
        ]
        !=
        [
            "pinned_base",
            "same_base_plus_frozen_candidate_adapter",
        ]
    ):
        raise RuntimeError(
            "Holdout evaluation order mismatch."
        )

    gate = payload[
        "preregistered_acceptance_gate"
    ]

    if (
        gate[
            "minimum_adapted_accuracy"
        ]
        !=
        0.70
    ):
        raise RuntimeError(
            "Accuracy gate changed."
        )

    if (
        gate[
            "minimum_adapted_macro_accuracy"
        ]
        !=
        0.70
    ):
        raise RuntimeError(
            "Macro accuracy gate changed."
        )

    if (
        gate[
            "minimum_per_relation_accuracy"
        ]
        !=
        0.50
    ):
        raise RuntimeError(
            "Per-relation gate changed."
        )

    if (
        gate[
            "minimum_uncertain_accuracy"
        ]
        !=
        0.666667
    ):
        raise RuntimeError(
            "Uncertain gate changed."
        )

    if (
        gate[
            "strict_json_validity_rate"
        ]
        !=
        1.0
    ):
        raise RuntimeError(
            "Strict JSON gate changed."
        )

    if (
        gate[
            "training_loss_is_acceptance_evidence"
        ]
        is not False
    ):
        raise RuntimeError(
            "Training loss cannot be acceptance evidence."
        )

    prerequisites = payload[
        "prerequisite_gates"
    ]

    for key in (
        "candidate_adapter_must_be_frozen",
        "hotel_diagnostic_must_pass_before_holdout_is_consumed",
        "known_regression_and_safety_gates_must_pass",
        "training_must_be_complete",
    ):
        if (
            prerequisites[
                key
            ]
            is not True
        ):
            raise RuntimeError(
                (
                    "Holdout prerequisite gate "
                    f"violation: {key}"
                )
            )


def validate_freeze(
    *,
    cases_sha256: str,
    freeze: dict[str, Any],
) -> None:
    if (
        freeze[
            "freeze_rule_version"
        ]
        !=
        AIRPORT_HOLDOUT_FREEZE_RULE_VERSION
    ):
        raise RuntimeError(
            "Airport holdout freeze rule mismatch."
        )

    if (
        freeze[
            "cases_sha256"
        ]
        !=
        cases_sha256
    ):
        raise RuntimeError(
            "Freeze -> cases SHA mismatch."
        )

    if (
        freeze[
            "status"
        ]
        !=
        "frozen"
    ):
        raise RuntimeError(
            "Airport holdout is not frozen."
        )

    for key in (
        "frozen_before_v0_4_training_data_authoring",
        "frozen_before_v0_4_training",
        "independent_holdout",
    ):
        if (
            freeze[
                key
            ]
            is not True
        ):
            raise RuntimeError(
                (
                    "Holdout freeze timing/policy "
                    f"violation: {key}"
                )
            )

    for key in (
        "evaluation_executed",
        "results_observed",
        "used_for_hyperparameter_tuning",
        "used_for_training",
        "v0_4_training_dataset_authored",
        "v0_4_training_executed",
        "final_acceptance_opened",
    ):
        if (
            freeze[
                key
            ]
            is not False
        ):
            raise RuntimeError(
                (
                    "Holdout pre-training state "
                    f"violation: {key}"
                )
            )


def validate_existing(
) -> None:
    if not CASES_PATH.is_file():
        raise FileNotFoundError(
            CASES_PATH
        )

    if not FREEZE_PATH.is_file():
        raise FileNotFoundError(
            FREEZE_PATH
        )

    cases = load_json_object(
        CASES_PATH
    )

    freeze = load_json_object(
        FREEZE_PATH
    )

    validate_cases(
        cases
    )

    validate_freeze(
        cases_sha256=
            sha256_file(
                CASES_PATH
            ),

        freeze=
            freeze,
    )


if __name__ == "__main__":
    validate_existing()

    print(
        "DATALENS QLORA v0.4 AIRPORT HOLDOUT VALIDATION: PASS"
    )
