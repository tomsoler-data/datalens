from __future__ import annotations


import hashlib
import json


from typing import Any, Mapping, Sequence


HOSPITAL_INDEPENDENT_EVALUATION_EXECUTION_RULE_VERSION = (
    "qlora_v0.4_hospital_independent_evaluation_execution_v0.1"
)


EXPECTED_RELATIONS = (
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
    "unrelated",
    "uncertain",
)


MODEL_EXECUTION_FIELDS = (
    "case_id",
    "prompt",
)


MAX_NEW_TOKENS = 64


EOS_TOKEN_IDS = (
    1,
    106,
)


PAD_TOKEN_ID = 0


# ============================================================
# STRICT OUTPUT CONTRACT
# ============================================================


def _invalid_output(
    *,
    decoded_output: str,
    invalid_reason: str,
    terminal_stop_token_id: (
        int
        | None
    ),
    generation_budget_exhausted: bool,
) -> dict[
    str,
    Any,
]:

    return {
        "strict_json_valid":
            False,

        "predicted_relation":
            None,

        "reason":
            None,

        "reason_word_count":
            None,

        "invalid_reason":
            invalid_reason,

        "decoded_output":
            decoded_output,

        "decoded_output_sha256":
            hashlib.sha256(
                decoded_output.encode(
                    "utf-8"
                )
            ).hexdigest(),

        "terminal_stop_token_id":
            terminal_stop_token_id,

        "generation_budget_exhausted":
            generation_budget_exhausted,
    }


def parse_generated_output(
    *,
    decoded_output: str,
    terminal_stop_token_id: (
        int
        | None
    ) = None,
    generation_budget_exhausted: bool = False,
) -> dict[
    str,
    Any,
]:

    normalized = (
        decoded_output.strip()
    )


    if generation_budget_exhausted:

        return _invalid_output(
            decoded_output=
                normalized,

            invalid_reason=
                "generation_budget_exhausted",

            terminal_stop_token_id=
                terminal_stop_token_id,

            generation_budget_exhausted=
                True,
        )


    try:

        payload = json.loads(
            normalized
        )

    except json.JSONDecodeError:

        return _invalid_output(
            decoded_output=
                normalized,

            invalid_reason=
                "json_parse_failed",

            terminal_stop_token_id=
                terminal_stop_token_id,

            generation_budget_exhausted=
                False,
        )


    if not isinstance(
        payload,
        dict,
    ):

        return _invalid_output(
            decoded_output=
                normalized,

            invalid_reason=
                "json_value_not_object",

            terminal_stop_token_id=
                terminal_stop_token_id,

            generation_budget_exhausted=
                False,
        )


    if set(
        payload
    ) != {
        "relation",
        "reason",
    }:

        return _invalid_output(
            decoded_output=
                normalized,

            invalid_reason=
                "json_key_set_mismatch",

            terminal_stop_token_id=
                terminal_stop_token_id,

            generation_budget_exhausted=
                False,
        )


    relation = payload[
        "relation"
    ]

    reason = payload[
        "reason"
    ]


    if (
        not isinstance(
            relation,
            str,
        )
        or
        relation
        not in
        EXPECTED_RELATIONS
    ):

        return _invalid_output(
            decoded_output=
                normalized,

            invalid_reason=
                "relation_not_allowed",

            terminal_stop_token_id=
                terminal_stop_token_id,

            generation_budget_exhausted=
                False,
        )


    if not isinstance(
        reason,
        str,
    ):

        return _invalid_output(
            decoded_output=
                normalized,

            invalid_reason=
                "reason_not_string",

            terminal_stop_token_id=
                terminal_stop_token_id,

            generation_budget_exhausted=
                False,
        )


    word_count = len(
        reason.split()
    )


    if (
        word_count < 6
        or
        word_count > 45
    ):

        return _invalid_output(
            decoded_output=
                normalized,

            invalid_reason=
                "reason_word_count_out_of_range",

            terminal_stop_token_id=
                terminal_stop_token_id,

            generation_budget_exhausted=
                False,
        )


    return {
        "strict_json_valid":
            True,

        "predicted_relation":
            relation,

        "reason":
            reason,

        "reason_word_count":
            word_count,

        "invalid_reason":
            None,

        "decoded_output":
            normalized,

        "decoded_output_sha256":
            hashlib.sha256(
                normalized.encode(
                    "utf-8"
                )
            ).hexdigest(),

        "terminal_stop_token_id":
            terminal_stop_token_id,

        "generation_budget_exhausted":
            False,
    }


# ============================================================
# GENERATED TOKEN POST-PROCESSING
# ============================================================


def process_generated_token_ids(
    *,
    tokenizer: Any,
    generated_token_ids: Sequence[
        int
    ],
) -> dict[
    str,
    Any,
]:

    token_ids = [
        int(
            value
        )

        for value
        in generated_token_ids
    ]


    if (
        len(
            token_ids
        )
        >
        MAX_NEW_TOKENS
    ):

        raise RuntimeError(
            (
                "Model generated more tokens "
                "than frozen budget."
            )
        )


    if not token_ids:

        return _invalid_output(
            decoded_output="",
            invalid_reason=
                "empty_generation",
            terminal_stop_token_id=None,
            generation_budget_exhausted=False,
        )


    terminal_stop_token_id: (
        int
        | None
    ) = None


    body_ids = token_ids


    if (
        token_ids[
            -1
        ]
        in
        EOS_TOKEN_IDS
    ):

        terminal_stop_token_id = (
            token_ids[
                -1
            ]
        )

        body_ids = token_ids[
            :-1
        ]

        budget_exhausted = False


    elif (
        len(
            token_ids
        )
        ==
        MAX_NEW_TOKENS
    ):

        budget_exhausted = True


    else:

        return _invalid_output(
            decoded_output="",
            invalid_reason=
                "missing_terminal_stop_token",
            terminal_stop_token_id=None,
            generation_budget_exhausted=False,
        )


    decoded = tokenizer.decode(
        body_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


    return parse_generated_output(
        decoded_output=
            decoded,

        terminal_stop_token_id=
            terminal_stop_token_id,

        generation_budget_exhausted=
            budget_exhausted,
    )


# ============================================================
# MODEL-FACING INPUT CONTRACT
# ============================================================


def validate_execution_input(
    execution_input: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    str,
]:

    if set(
        execution_input
    ) != set(
        MODEL_EXECUTION_FIELDS
    ):

        raise RuntimeError(
            (
                "Model execution input must contain "
                "exactly case_id and prompt."
            )
        )


    case_id = execution_input[
        "case_id"
    ]

    prompt = execution_input[
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


    return {
        "case_id":
            case_id,

        "prompt":
            prompt,
    }


# ============================================================
# SINGLE MODEL EXECUTION
# ============================================================


def generate_execution_input(
    *,
    model: Any,
    tokenizer: Any,
    execution_input: Mapping[
        str,
        Any,
    ],
    torch_module: Any,
) -> dict[
    str,
    Any,
]:

    model_input = validate_execution_input(
        execution_input
    )


    prompt = model_input[
        "prompt"
    ]


    input_ids = (
        tokenizer.apply_chat_template(
            [
                {
                    "role":
                        "user",

                    "content":
                        prompt,
                }
            ],

            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
            return_tensors="pt",
            return_dict=False,
        )
    )


    if (
        getattr(
            input_ids,
            "ndim",
            None,
        )
        !=
        2
        or
        int(
            input_ids.shape[
                0
            ]
        )
        !=
        1
    ):

        raise RuntimeError(
            "Unexpected chat-template tensor shape."
        )


    input_ids = input_ids.to(
        "cuda"
    )


    attention_mask = (
        torch_module.ones_like(
            input_ids,
            dtype=
                torch_module.long,
            device=
                "cuda",
        )
    )


    prompt_token_count = int(
        input_ids.shape[
            1
        ]
    )


    model_limit = int(
        getattr(
            model.config,
            "max_position_embeddings",
            0,
        )
    )


    if model_limit <= 0:

        raise RuntimeError(
            "Model context limit unavailable."
        )


    if (
        prompt_token_count
        +
        MAX_NEW_TOKENS
        >
        model_limit
    ):

        raise RuntimeError(
            (
                "Hospital prompt exceeds model "
                "context with frozen generation budget."
            )
        )


    with torch_module.inference_mode():

        generated = model.generate(
            input_ids=
                input_ids,

            attention_mask=
                attention_mask,

            do_sample=False,

            num_beams=1,

            max_new_tokens=
                MAX_NEW_TOKENS,

            eos_token_id=
                list(
                    EOS_TOKEN_IDS
                ),

            pad_token_id=
                PAD_TOKEN_ID,
        )


    if (
        getattr(
            generated,
            "ndim",
            None,
        )
        !=
        2
        or
        int(
            generated.shape[
                0
            ]
        )
        !=
        1
    ):

        raise RuntimeError(
            "Unexpected generate() output shape."
        )


    new_ids = (
        generated[
            0,
            prompt_token_count:
        ]
        .detach()
        .cpu()
        .tolist()
    )


    processed = process_generated_token_ids(
        tokenizer=
            tokenizer,

        generated_token_ids=
            new_ids,
    )


    processed[
        "case_id"
    ] = model_input[
        "case_id"
    ]


    processed[
        "prompt_sha256"
    ] = hashlib.sha256(
        prompt.encode(
            "utf-8"
        )
    ).hexdigest()


    processed[
        "prompt_token_count"
    ] = prompt_token_count


    processed[
        "generated_token_count"
    ] = len(
        new_ids
    )


    return processed


# ============================================================
# MODEL PASS
# ============================================================


def run_model_pass(
    *,
    model_label: str,
    model: Any,
    tokenizer: Any,
    execution_inputs: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    torch_module: Any,
) -> tuple[
    dict[
        str,
        Any,
    ],
    ...,
]:

    if model_label not in {
        "base",
        "adapted",
    }:

        raise ValueError(
            "model_label must be base or adapted."
        )


    validated_inputs = tuple(
        validate_execution_input(
            execution_input
        )

        for execution_input
        in execution_inputs
    )


    case_ids = [
        execution_input[
            "case_id"
        ]

        for execution_input
        in validated_inputs
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


    outputs = []


    for execution_input in validated_inputs:

        result = generate_execution_input(
            model=
                model,

            tokenizer=
                tokenizer,

            execution_input=
                execution_input,

            torch_module=
                torch_module,
        )


        result[
            "model_label"
        ] = model_label


        outputs.append(
            result
        )


    return tuple(
        outputs
    )


# ============================================================
# BASE -> ADAPTED ORCHESTRATION
# ============================================================


def run_base_then_adapted(
    *,
    base_model: Any,
    adapted_model: Any,
    tokenizer: Any,
    execution_inputs: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    torch_module: Any,
) -> dict[
    str,
    Any,
]:

    frozen_inputs = tuple(
        validate_execution_input(
            execution_input
        )

        for execution_input
        in execution_inputs
    )


    base_results = run_model_pass(
        model_label=
            "base",

        model=
            base_model,

        tokenizer=
            tokenizer,

        execution_inputs=
            frozen_inputs,

        torch_module=
            torch_module,
    )


    adapted_results = run_model_pass(
        model_label=
            "adapted",

        model=
            adapted_model,

        tokenizer=
            tokenizer,

        execution_inputs=
            frozen_inputs,

        torch_module=
            torch_module,
    )


    base_case_ids = tuple(
        item[
            "case_id"
        ]

        for item
        in base_results
    )


    adapted_case_ids = tuple(
        item[
            "case_id"
        ]

        for item
        in adapted_results
    )


    if base_case_ids != adapted_case_ids:

        raise RuntimeError(
            "Base/adapted case ordering changed."
        )


    base_prompt_sha = tuple(
        item[
            "prompt_sha256"
        ]

        for item
        in base_results
    )


    adapted_prompt_sha = tuple(
        item[
            "prompt_sha256"
        ]

        for item
        in adapted_results
    )


    if base_prompt_sha != adapted_prompt_sha:

        raise RuntimeError(
            "Base/adapted prompts differ."
        )


    return {
        "execution_order":
            (
                "base",
                "adapted",
            ),

        "base":
            base_results,

        "adapted":
            adapted_results,
    }
