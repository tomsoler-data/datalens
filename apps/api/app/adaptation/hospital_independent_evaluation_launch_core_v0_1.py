from __future__ import annotations


import hashlib
import re


from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)

from typing import Any


HOSPITAL_INDEPENDENT_EVALUATION_LAUNCH_CORE_RULE_VERSION = (
    "qlora_v0.4_hospital_independent_evaluation_launch_core_v0.1"
)


RUNTIME_REUSE_FREEZE_PARENT_COMMIT = (
    "3472ffe6bcb4c2d435c7eabe70bc7a47e7b693ce"
)


GREENHOUSE_RUNTIME_COMMITTED_SHA256 = (
    "8e922f46d65048ab6bcebeab8eca7e52"
    "b8bf83100524bea84658530039b40cea"
)

QLORA_RUNTIME_COMMITTED_SHA256 = (
    "20e41ab00606296893276a84e53746c0"
    "6618b8cabca74fef77cb743c5e80ab7c"
)

HOSPITAL_RUNNER_COMMITTED_SHA256 = (
    "4f1f69ff01b9a9c096dd42fc694a6f9"
    "f11f0ba30a23f6fc2a8dddc92deff900d"
)

HOSPITAL_EXECUTION_COMMITTED_SHA256 = (
    "28566b79ad34e2587ced709f68ca19b1"
    "dfaa5d4074c07ba769db3a11306ee1c1"
)

HOSPITAL_PREFLIGHT_COMMITTED_SHA256 = (
    "bd9b55863b001e921c0cc398e2c9f7d"
    "7882eb111aa0fec07b231dfdcc5aca2a5"
)


EXPECTED_CASE_COUNT = 30


REQUIRED_DEPENDENCY_KEYS = frozenset(
    {
        "build_preflight_snapshot",
        "prepare_runtime_authority",
        "load_base_model",
        "claim_single_use_consumption",
        "load_frozen_holdout_for_execution",
        "build_execution_inputs",
        "run_model_pass",
        "attach_adapter",
    }
)


def _require_commit_sha(
    value: str,
) -> str:

    if (
        not isinstance(
            value,
            str,
        )
        or
        re.fullmatch(
            r"[0-9a-f]{40}",
            value,
        )
        is None
    ):

        raise ValueError(
            "execution_authority_commit must be a "
            "40-character lowercase Git SHA."
        )

    return value


def _require_utc_timestamp(
    value: str,
) -> str:

    if (
        not isinstance(
            value,
            str,
        )
        or
        not value.endswith(
            "Z"
        )
        or
        len(
            value
        )
        <
        20
    ):

        raise ValueError(
            "claimed_at_utc must be an explicit UTC timestamp."
        )

    return value


def _validate_dependencies(
    dependencies: Mapping[
        str,
        Callable[
            ...,
            Any,
        ],
    ],
) -> dict[
    str,
    Callable[
        ...,
        Any,
    ],
]:

    if not isinstance(
        dependencies,
        Mapping,
    ):

        raise TypeError(
            "dependencies must be a mapping."
        )

    actual_keys = set(
        dependencies
    )

    expected_keys = set(
        REQUIRED_DEPENDENCY_KEYS
    )

    if actual_keys != expected_keys:

        missing = sorted(
            expected_keys
            -
            actual_keys
        )

        extra = sorted(
            actual_keys
            -
            expected_keys
        )

        raise RuntimeError(
            (
                "Launch dependency set changed. "
                f"missing={missing}, extra={extra}"
            )
        )

    validated = {}

    for (
        key,
        function,
    ) in dependencies.items():

        if not callable(
            function
        ):

            raise TypeError(
                (
                    "Launch dependency must be callable: "
                    f"{key}"
                )
            )

        validated[
            key
        ] = function

    return validated


def _validate_preflight_snapshot(
    snapshot: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:

    if not isinstance(
        snapshot,
        Mapping,
    ):

        raise TypeError(
            "Preflight snapshot must be a mapping."
        )

    required = {
        "case_count":
            EXPECTED_CASE_COUNT,

        "single_use_available":
            True,

        "protected_cases_opened":
            False,

        "gold_opened":
            False,

        "model_loaded":
            False,

        "adapter_attached":
            False,

        "generation_started":
            False,
    }

    for (
        key,
        expected,
    ) in required.items():

        if (
            snapshot.get(
                key
            )
            !=
            expected
        ):

            raise RuntimeError(
                (
                    "Unsafe or unexpected preflight snapshot: "
                    f"{key}"
                )
            )

    return dict(
        snapshot
    )


def _freeze_execution_inputs(
    execution_inputs: Sequence[
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

    if (
        not isinstance(
            execution_inputs,
            Sequence,
        )
        or
        isinstance(
            execution_inputs,
            (
                str,
                bytes,
                bytearray,
            ),
        )
    ):

        raise TypeError(
            "execution_inputs must be a sequence."
        )

    if (
        len(
            execution_inputs
        )
        !=
        EXPECTED_CASE_COUNT
    ):

        raise RuntimeError(
            (
                "Expected exactly "
                f"{EXPECTED_CASE_COUNT} execution inputs."
            )
        )

    frozen = []

    for item in execution_inputs:

        if not isinstance(
            item,
            Mapping,
        ):

            raise TypeError(
                "Execution input must be a mapping."
            )

        if (
            set(
                item
            )
            !=
            {
                "case_id",
                "prompt",
            }
        ):

            raise RuntimeError(
                (
                    "Model execution boundary changed. "
                    "Expected exactly case_id + prompt."
                )
            )

        case_id = item[
            "case_id"
        ]

        prompt = item[
            "prompt"
        ]

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

        if (
            not isinstance(
                prompt,
                str,
            )
            or
            not prompt.strip()
        ):

            raise TypeError(
                "prompt must be a non-empty string."
            )

        frozen.append(
            {
                "case_id":
                    case_id,

                "prompt":
                    prompt,
            }
        )

    case_ids = [
        item[
            "case_id"
        ]

        for item
        in frozen
    ]

    if (
        len(
            case_ids
        )
        !=
        len(
            set(
                case_ids
            )
        )
    ):

        raise RuntimeError(
            "Execution case IDs are not unique."
        )

    return tuple(
        frozen
    )


def _prompt_sha256(
    prompt: str,
) -> str:

    return hashlib.sha256(
        prompt.encode(
            "utf-8"
        )
    ).hexdigest()


def _validate_pass_results(
    *,
    model_label: str,
    results: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    execution_inputs: Sequence[
        Mapping[
            str,
            str,
        ]
    ],
) -> tuple[
    dict[
        str,
        Any,
    ],
    ...,
]:

    if (
        not isinstance(
            results,
            Sequence,
        )
        or
        isinstance(
            results,
            (
                str,
                bytes,
                bytearray,
            ),
        )
    ):

        raise TypeError(
            "Model pass results must be a sequence."
        )

    if (
        len(
            results
        )
        !=
        EXPECTED_CASE_COUNT
    ):

        raise RuntimeError(
            (
                "Unexpected model-pass result count: "
                f"{model_label}"
            )
        )

    validated = []

    for (
        expected_input,
        result,
    ) in zip(
        execution_inputs,
        results,
        strict=True,
    ):

        if not isinstance(
            result,
            Mapping,
        ):

            raise TypeError(
                "Model result must be a mapping."
            )

        if (
            result.get(
                "case_id"
            )
            !=
            expected_input[
                "case_id"
            ]
        ):

            raise RuntimeError(
                (
                    "Model result case identity changed: "
                    f"{model_label}"
                )
            )

        if (
            result.get(
                "model_label"
            )
            !=
            model_label
        ):

            raise RuntimeError(
                (
                    "Model result label changed: "
                    f"{model_label}"
                )
            )

        expected_prompt_sha = (
            _prompt_sha256(
                expected_input[
                    "prompt"
                ]
            )
        )

        if (
            result.get(
                "prompt_sha256"
            )
            !=
            expected_prompt_sha
        ):

            raise RuntimeError(
                (
                    "Model result prompt authority changed: "
                    f"{model_label}"
                )
            )

        validated.append(
            dict(
                result
            )
        )

    return tuple(
        validated
    )


def run_official_launch_core(
    *,
    execution_authority_commit: str,
    claimed_at_utc: str,
    dependencies: Mapping[
        str,
        Callable[
            ...,
            Any,
        ],
    ],
    torch_module: Any,
) -> dict[
    str,
    Any,
]:

    execution_authority_commit = (
        _require_commit_sha(
            execution_authority_commit
        )
    )

    claimed_at_utc = (
        _require_utc_timestamp(
            claimed_at_utc
        )
    )

    safe = _validate_dependencies(
        dependencies
    )

    preflight_snapshot = (
        _validate_preflight_snapshot(
            safe[
                "build_preflight_snapshot"
            ]()
        )
    )

    runtime_pair = (
        safe[
            "prepare_runtime_authority"
        ]()
    )

    if (
        not isinstance(
            runtime_pair,
            tuple,
        )
        or
        len(
            runtime_pair
        )
        !=
        2
    ):

        raise RuntimeError(
            (
                "prepare_runtime_authority must return "
                "(authority, tokenizer)."
            )
        )

    (
        runtime_authority,
        tokenizer,
    ) = runtime_pair

    # Load only non-protected runtime material before claiming
    # the one-shot benchmark.
    base_model = (
        safe[
            "load_base_model"
        ](
            torch_module=
                torch_module,

            authority=
                runtime_authority,
        )
    )

    # Final action before the first protected read.
    consumption_claim = (
        safe[
            "claim_single_use_consumption"
        ](
            execution_authority_commit=
                execution_authority_commit,

            claimed_at_utc=
                claimed_at_utc,
        )
    )

    if not isinstance(
        consumption_claim,
        Mapping,
    ):

        raise TypeError(
            "Single-use claim must return a mapping."
        )

    if (
        consumption_claim.get(
            "status"
        )
        !=
        "claimed_before_holdout_open"
    ):

        raise RuntimeError(
            "Single-use claim contract changed."
        )

    if (
        consumption_claim.get(
            "execution_authority_commit"
        )
        !=
        execution_authority_commit
    ):

        raise RuntimeError(
            "Single-use claim commit authority changed."
        )

    if (
        consumption_claim.get(
            "claimed_at_utc"
        )
        !=
        claimed_at_utc
    ):

        raise RuntimeError(
            "Single-use claim timestamp changed."
        )

    protected_cases = (
        safe[
            "load_frozen_holdout_for_execution"
        ]()
    )

    execution_inputs = (
        _freeze_execution_inputs(
            safe[
                "build_execution_inputs"
            ](
                protected_cases
            )
        )
    )

    # Protected case structures are no longer needed once the
    # exact label-blind execution boundary has been produced.
    del protected_cases

    base_results = (
        _validate_pass_results(
            model_label=
                "base",

            results=
                safe[
                    "run_model_pass"
                ](
                    model_label=
                        "base",

                    model=
                        base_model,

                    tokenizer=
                        tokenizer,

                    execution_inputs=
                        execution_inputs,

                    torch_module=
                        torch_module,
                ),

            execution_inputs=
                execution_inputs,
        )
    )

    # Adapter attachment cannot happen before all Base results
    # have returned and passed identity/prompt validation.
    adapted_model = (
        safe[
            "attach_adapter"
        ](
            model=
                base_model,
        )
    )

    adapted_results = (
        _validate_pass_results(
            model_label=
                "adapted",

            results=
                safe[
                    "run_model_pass"
                ](
                    model_label=
                        "adapted",

                    model=
                        adapted_model,

                    tokenizer=
                        tokenizer,

                    execution_inputs=
                        execution_inputs,

                    torch_module=
                        torch_module,
                ),

            execution_inputs=
                execution_inputs,
        )
    )

    return {
        "rule_version":
            HOSPITAL_INDEPENDENT_EVALUATION_LAUNCH_CORE_RULE_VERSION,

        "execution_authority_commit":
            execution_authority_commit,

        "claimed_at_utc":
            claimed_at_utc,

        "case_count":
            EXPECTED_CASE_COUNT,

        "preflight_rule_version":
            preflight_snapshot.get(
                "rule_version"
            ),

        "consumption_claim":
            dict(
                consumption_claim
            ),

        "base_results":
            base_results,

        "adapted_results":
            adapted_results,

        "scoring_started":
            False,

        "report_written":
            False,

        "receipt_written":
            False,
    }
