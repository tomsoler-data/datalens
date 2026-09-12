from __future__ import annotations


import hashlib
import json


from pathlib import Path
from typing import Any, Mapping, Sequence


from app.adaptation.training_dataset_canonicalizer_v0_4 import (
    build_user_message,
)


HOSPITAL_INDEPENDENT_EVALUATION_RUNNER_RULE_VERSION = (
    "qlora_v0.4_hospital_independent_evaluation_runner_v0.1"
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


PROTOCOL_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "evaluation"
    / (
        "datalens_semantic_qlora_v0.4_"
        "hospital_independent_evaluation_protocol_v0.1.json"
    )
)


HOLDOUT_CASES_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "holdouts"
    / (
        "datalens_semantic_qlora_v0.4_"
        "hospital_emergency_department_operations_"
        "holdout_v0.1_cases.json"
    )
)


HOLDOUT_FREEZE_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "holdouts"
    / (
        "datalens_semantic_qlora_v0.4_"
        "hospital_emergency_department_operations_"
        "holdout_v0.1_freeze.json"
    )
)


EXPECTED_PROTOCOL_COMMIT = (
    "98e73521d633406c8de728aa667dc2525f6787a3"
)


EXPECTED_FROZEN_HOLDOUT_COMMIT = (
    "510f098a03a0e3263f607bcb68c31818b6e1c13b"
)


EXPECTED_PROTOCOL_SHA256 = (
    "0e958d67a6294f8666485a6274b1eee2"
    "1b9f06fe2e300ba93241a7ffaba3127d"
)


EXPECTED_PROTOCOL_AGGREGATE_FREEZE_SHA256 = (
    "e00f9f2d6ef4a8eea47da58b2b200536"
    "dc8337c48282f301153a63a3dc215ce6"
)


EXPECTED_HOLDOUT_CASES_SHA256 = (
    "92e4f21e7323cd053fd5f53b51e14932"
    "97fdd28cfb717be56f7be963f298d9fc"
)


EXPECTED_HOLDOUT_FREEZE_SHA256 = (
    "67baedb38d7807348d7ee6436f7f1a37"
    "30c5eeb3f1bd68bc1e2556a030411bed"
)


EXPECTED_DOMAIN = (
    "hospital_emergency_department_operations"
)


EXPECTED_RELATIONS = (
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
    "unrelated",
    "uncertain",
)


MODEL_VISIBLE_FIELDS = (
    "domain",
    "left_metric",
    "left_description",
    "right_metric",
    "right_description",
)


PROTECTED_GOLD_FIELDS = (
    "gold_relation",
    "gold_reason",
)


EXPECTED_CASE_COUNT = 30


# ============================================================
# BASIC AUTHORITIES
# ============================================================


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()


    with path.open(
        "rb"
    ) as handle:

        for chunk in iter(
            lambda:
                handle.read(
                    8
                    *
                    1024
                    *
                    1024
                ),
            b"",
        ):

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
            encoding="utf-8"
        )
    )


    if not isinstance(
        payload,
        dict,
    ):

        raise TypeError(
            (
                "Expected JSON object: "
                f"{path}"
            )
        )


    return payload


def _require_exact_sha(
    *,
    path: Path,
    expected_sha256: str,
    label: str,
) -> None:

    if not path.is_file():

        raise FileNotFoundError(
            path
        )


    actual = sha256_file(
        path
    )


    if actual != expected_sha256:

        raise RuntimeError(
            (
                f"{label} SHA mismatch.\n"
                f"Expected: {expected_sha256}\n"
                f"Actual:   {actual}"
            )
        )


# ============================================================
# PRE-EXECUTION AUTHORITY
# ============================================================


def validate_pre_execution_authorities(
) -> None:

    _require_exact_sha(
        path=PROTOCOL_PATH,
        expected_sha256=
            EXPECTED_PROTOCOL_SHA256,
        label="R8 public protocol",
    )


    _require_exact_sha(
        path=HOLDOUT_FREEZE_PATH,
        expected_sha256=
            EXPECTED_HOLDOUT_FREEZE_SHA256,
        label="R8 hospital holdout freeze",
    )


    protocol = load_json_object(
        PROTOCOL_PATH
    )


    freeze = load_json_object(
        HOLDOUT_FREEZE_PATH
    )


    if (
        protocol[
            "protocol_parent_git_commit"
        ]
        !=
        (
            "c3794df356c6bb4373e01e10c6b37eaad4d2ad18"
        )
    ):

        raise RuntimeError(
            "Unexpected protocol parent authority."
        )


    methodology = protocol[
        "methodology"
    ]


    if (
        methodology[
            "cases_must_be_frozen_before_first_model_execution"
        ]
        is not True
    ):

        raise RuntimeError(
            "Protocol does not require pre-execution freeze."
        )


    single_use = protocol[
        "single_use_policy"
    ]


    if (
        single_use[
            "one_official_consumption_for_v0_4"
        ]
        is not True
    ):

        raise RuntimeError(
            "Single-use authority missing."
        )


    if (
        single_use[
            "reexecution_after_results_observed"
        ]
        is not False
    ):

        raise RuntimeError(
            "Reexecution policy is not fail-closed."
        )


    if (
        freeze[
            "status"
        ]
        !=
        "frozen_before_first_model_execution"
    ):

        raise RuntimeError(
            "Holdout is not frozen before execution."
        )


    protocol_authority = freeze[
        "protocol_authority"
    ]


    if (
        protocol_authority[
            "commit"
        ]
        !=
        EXPECTED_PROTOCOL_COMMIT
    ):

        raise RuntimeError(
            "Freeze protocol commit mismatch."
        )


    if (
        protocol_authority[
            "aggregate_freeze_sha256"
        ]
        !=
        EXPECTED_PROTOCOL_AGGREGATE_FREEZE_SHA256
    ):

        raise RuntimeError(
            "Freeze protocol identity mismatch."
        )


    protected = freeze[
        "protected_holdout"
    ]


    if (
        protected[
            "cases_sha256"
        ]
        !=
        EXPECTED_HOLDOUT_CASES_SHA256
    ):

        raise RuntimeError(
            "Holdout cases identity mismatch."
        )


    if (
        protected[
            "case_count"
        ]
        !=
        EXPECTED_CASE_COUNT
    ):

        raise RuntimeError(
            "Unexpected holdout case count."
        )


    if (
        tuple(
            protected[
                "relations"
            ]
        )
        !=
        EXPECTED_RELATIONS
    ):

        raise RuntimeError(
            "Unexpected holdout relation taxonomy."
        )


# ============================================================
# LABEL-BLIND BOUNDARY
# ============================================================


def build_label_blind_record(
    case: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    str,
]:

    visible: dict[
        str,
        str,
    ] = {}


    for field in MODEL_VISIBLE_FIELDS:

        if field not in case:

            raise KeyError(
                (
                    "Missing model-visible field: "
                    f"{field}"
                )
            )


        value = case[
            field
        ]


        if (
            not isinstance(
                value,
                str,
            )
            or
            not value.strip()
        ):

            raise TypeError(
                (
                    "Model-visible field must be "
                    "a non-empty string: "
                    f"{field}"
                )
            )


        visible[
            field
        ] = value


    if set(
        visible
    ) != set(
        MODEL_VISIBLE_FIELDS
    ):

        raise RuntimeError(
            "Label-blind record field set changed."
        )


    if any(
        field in visible

        for field
        in PROTECTED_GOLD_FIELDS
    ):

        raise RuntimeError(
            "Protected gold field crossed model boundary."
        )


    if (
        visible[
            "domain"
        ]
        !=
        EXPECTED_DOMAIN
    ):

        raise RuntimeError(
            "Unexpected evaluation domain."
        )


    return visible


def build_label_blind_prompt(
    case: Mapping[
        str,
        Any,
    ],
) -> str:

    visible = build_label_blind_record(
        case
    )


    prompt = build_user_message(
        visible
    )


    if not isinstance(
        prompt,
        str,
    ):

        raise TypeError(
            "Prompt builder did not return a string."
        )


    if not prompt.strip():

        raise RuntimeError(
            "Prompt builder returned an empty prompt."
        )


    return prompt


def build_execution_input(
    case: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    str,
]:

    case_id = case.get(
        "case_id"
    )


    if (
        not isinstance(
            case_id,
            str,
        )
        or
        not case_id.strip()
    ):

        raise TypeError(
            "case_id must be a non-empty string."
        )


    return {
        "case_id":
            case_id,

        "prompt":
            build_label_blind_prompt(
                case
            ),
    }


def build_execution_inputs(
    cases: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> tuple[
    dict[
        str,
        str,
    ],
    ...,
]:

    inputs = tuple(
        build_execution_input(
            case
        )

        for case
        in cases
    )


    if len(
        inputs
    ) != EXPECTED_CASE_COUNT:

        raise RuntimeError(
            (
                "Expected exactly "
                f"{EXPECTED_CASE_COUNT} execution inputs; "
                f"got {len(inputs)}."
            )
        )


    case_ids = [
        item[
            "case_id"
        ]

        for item
        in inputs
    ]


    if len(
        case_ids
    ) != len(
        set(
            case_ids
        )
    ):

        raise RuntimeError(
            "Execution case IDs are not unique."
        )


    return inputs


# ============================================================
# PROTECTED HOLDOUT LOAD
#
# This function is intentionally not called during D4-B.
# It exists for a later execution/preflight gate.
# ============================================================


def load_frozen_holdout_for_execution(
) -> tuple[
    dict[
        str,
        Any,
    ],
    ...,
]:

    validate_pre_execution_authorities()


    _require_exact_sha(
        path=HOLDOUT_CASES_PATH,
        expected_sha256=
            EXPECTED_HOLDOUT_CASES_SHA256,
        label="R8 hospital holdout cases",
    )


    payload = load_json_object(
        HOLDOUT_CASES_PATH
    )


    if (
        payload[
            "domain"
        ]
        !=
        EXPECTED_DOMAIN
    ):

        raise RuntimeError(
            "Hospital holdout domain mismatch."
        )


    cases = payload[
        "cases"
    ]


    if not isinstance(
        cases,
        list,
    ):

        raise TypeError(
            "Holdout cases must be a list."
        )


    if len(
        cases
    ) != EXPECTED_CASE_COUNT:

        raise RuntimeError(
            "Hospital holdout case count changed."
        )


    return tuple(
        dict(
            case
        )

        for case
        in cases
    )
